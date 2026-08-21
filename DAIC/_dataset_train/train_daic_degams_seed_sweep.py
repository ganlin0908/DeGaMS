"""DeGaMS ABDE dynamic seed sweep (DEV-driven selection; single TEST; optional ensemble).

Protocol (anti-leakage):
  1. seeds = range(start_seed, start_seed + num_seeds)  — no hardcoded seed lists
  2. Per seed: train; each epoch DEV thr search in anchored PHQ band; save best-DEV ckpt
  3. Global best_seed = argmax DEV Macro-F1  (TEST never used for selection)
  4. Official: load best seed ckpt + its DEV best_thr → evaluate TEST once
  5. Optional: per-seed TEST on locked best-DEV ckpts (reporting only)
  6. Optional --ensemble: average TEST soft PHQ scores; thr = mean(per-seed best_thr)

Reuses MentalBERT + Mel + dual-visual + hybrid filter from train_daic_degams / dataset / model.
Does not invent a new architecture.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from cuda_exit import exit_clean  # noqa: E402
from daic_degams_dataset import DAICDeGaMSDataset, collate_degams  # noqa: E402
from degams_model import DeGaMSConfig, DeGaMSModel, degams_total_loss  # noqa: E402
from train_daic_degams import (  # noqa: E402
    collect_gate_stats,
    eval_dev_with_thr_search,
    metrics_at_threshold,
    predict_scores,
    set_seed,
)

OUT = Path(
    os.environ.get("DEGAMS_RUN_ROOT")
    or str(Path(__file__).resolve().parents[2] / "runs")
) / "daic_degams_seed_sweep"


def _build_loaders(args: argparse.Namespace, device: torch.device):
    filter_mode = args.filter_mode
    kw = filter_mode != "none"
    ds_kw = dict(
        max_len=args.max_len,
        keyword_filter=kw,
        filter_mode=filter_mode,
        hybrid_target_keep=args.hybrid_target_keep,
        hybrid_min_frac=args.hybrid_min_frac,
    )
    train_ds = DAICDeGaMSDataset("train", **ds_kw)
    dev_ds = DAICDeGaMSDataset("dev", **ds_kw)
    test_ds = DAICDeGaMSDataset("test", **ds_kw)
    if len(train_ds) == 0:
        raise SystemExit("Empty train set — features not ready?")

    y_train = np.array([int(x["y"].item()) for x in train_ds._cache])
    n0 = float((y_train == 0).sum())
    n1 = float((y_train == 1).sum())
    class_counts = (n0, n1)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_degams,
        pin_memory=pin,
    )
    # Fresh shuffle each seed via set_seed before DataLoader iter; rebuild train
    # loader per seed so shuffle order differs.
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_degams,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_degams,
        pin_memory=pin,
    )
    vdim = int(train_ds[0]["visual"].size(1))
    mel_dim = int(train_ds[0]["mel"].size(1))
    return {
        "train_ds": train_ds,
        "dev_ds": dev_ds,
        "test_ds": test_ds,
        "train_loader": train_loader,
        "dev_loader": dev_loader,
        "test_loader": test_loader,
        "class_counts": class_counts,
        "n0": n0,
        "n1": n1,
        "vdim": vdim,
        "mel_dim": mel_dim,
        "ds_kw": ds_kw,
        "pin": pin,
    }


def _make_cfg(args: argparse.Namespace, vdim: int) -> DeGaMSConfig:
    lam_sscl = 0.0  # SSCL retired in degams_total_loss
    lam_lbcl = float(getattr(args, "lambda_lbcl", 0.004))
    if getattr(args, "ablate_lbcl", False):
        lam_lbcl = 0.0
    lam_cmcl = float(getattr(args, "lambda_cmcl", 0.05))
    if getattr(args, "ablate_cmcl", False):
        lam_cmcl = 0.0
    # Auto: OF29∥DenseNet768 → DenseVisualEncoder
    hi_dim = 768 if int(vdim) >= 29 + 768 else 0
    use_dual = (not args.no_dual_visual) and hi_dim == 0
    return DeGaMSConfig(
        mel_dim=int(getattr(args, "_mel_dim", 88)),
        visual_dim=vdim,
        hi_visual_dim=hi_dim,
        dropout=args.dropout,
        lambda_sscl=lam_sscl,
        lambda_lbcl=lam_lbcl,
        lambda_mse=float(getattr(args, "lambda_mse", 0.01)),
        lambda_kl=float(getattr(args, "lambda_kl", 0.8)),
        lambda_cmcl=lam_cmcl,
        cmcl_temp=float(getattr(args, "cmcl_temp", 0.07)),
        cmcl_include_av=not bool(getattr(args, "no_cmcl_av", False)),
        soft_temp=float(getattr(args, "soft_temp", 0.35)),
        max_len=args.max_len,
        class_weight_power=args.class_weight_power,
        neg_emphasis=args.neg_emphasis,
        use_dual_visual=use_dual,
        drop_text=bool(args.drop_text),
        drop_audio=bool(args.drop_audio),
        drop_visual=bool(args.drop_visual),
        ablate_soft_phq=bool(getattr(args, "ablate_soft_phq", False)),
        ablate_msaf=bool(getattr(args, "ablate_msaf", False)),
        ablate_decomp=bool(getattr(args, "ablate_decomp", False)),
        ablate_turn_attn=bool(getattr(args, "ablate_turn_attn", False)),
        ablate_gate=bool(getattr(args, "ablate_gate", False)),
        ablate_cmcl=bool(getattr(args, "ablate_cmcl", False)),
        gate_temp=float(getattr(args, "gate_temp", 1.5)),
        fusion_mode=str(getattr(args, "fusion_mode", "gate")),
        gate_quality_prior=bool(getattr(args, "gate_quality_prior", False)),
        gate_quality_scale=float(getattr(args, "gate_quality_scale", 2.0)),
    )


def train_one_seed(
    args: argparse.Namespace,
    seed: int,
    run_dir: Path,
    device: torch.device,
    shared: dict,
) -> dict:
    """Train one seed; select ckpt/thr by DEV Macro-F1 only (no TEST)."""
    set_seed(seed)

    # Rebuild train loader so shuffle order is seed-dependent after set_seed.
    train_loader = DataLoader(
        shared["train_ds"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_degams,
        pin_memory=shared["pin"],
    )
    dev_loader = shared["dev_loader"]
    class_counts = shared["class_counts"]
    args._mel_dim = shared.get("mel_dim", 88)
    cfg = _make_cfg(args, shared["vdim"])
    model = DeGaMSModel(cfg).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_dev_f1 = -1.0
    best_thr = 10.0
    best_epoch = 0
    best_metrics: dict = {}
    ckpt_path = run_dir / f"checkpoint_best_dev_seed_{seed}.pt"
    history: list[dict] = []

    print(
        f"[seed {seed}] start train epochs={args.epochs} "
        f"thr_band=[{args.thr_min},{args.thr_max}] step={args.thr_step}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for batch in train_loader:
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.bfloat16):
                out = model(batch)
                loss, _parts = degams_total_loss(
                    out,
                    phq_items=batch["phq_items"],
                    phq_total=batch["phq_total"],
                    y_bin=batch["y"],
                    cfg=cfg,
                    class_counts=class_counts,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.detach())
            n += 1
        sched.step()

        y_dev, s_dev = predict_scores(model, dev_loader, device)
        min_p = args.min_precision if args.min_precision and args.min_precision > 0 else None
        max_r = args.max_recall if getattr(args, "max_recall", 0) and args.max_recall > 0 else None
        req_p_ge_r = bool(getattr(args, "require_precision_ge_recall", False))
        prefer_thr = getattr(args, "prefer_thr", None)
        prefer_w = float(getattr(args, "prefer_thr_weight", 0.0) or 0.0)
        if prefer_thr is not None and float(prefer_thr) <= 0:
            prefer_thr = None
        dev_sel = eval_dev_with_thr_search(
            y_dev,
            s_dev,
            thr_min=args.thr_min,
            thr_max=args.thr_max,
            thr_step=args.thr_step,
            criterion=args.thr_criterion,
            min_precision=min_p,
            max_recall=max_r,
            require_precision_ge_recall=req_p_ge_r,
            prefer_thr=float(prefer_thr) if prefer_thr is not None else None,
            prefer_thr_weight=prefer_w,
        )
        m = dev_sel["metrics"]
        row = {
            "epoch": epoch,
            "loss": total / max(n, 1),
            "best_thr": dev_sel["best_thr"],
            "dev_macro_f1": float(m["macro_f1"]),
            "dev_dep_f1": float(m["f1"]),
            "dev_p": float(m["precision"]),
            "dev_r": float(m["recall"]),
            "dev_acc": float(m["accuracy"]),
            "min_precision": min_p,
        }
        history.append(row)
        print(
            f"[seed {seed}] epoch {epoch:03d} loss={row['loss']:.4f} "
            f"DEV Macro-F1={row['dev_macro_f1']:.4f} depF1={row['dev_dep_f1']:.4f} "
            f"P={row['dev_p']:.3f} R={row['dev_r']:.3f} Acc={row['dev_acc']:.3f} "
            f"thr*={row['best_thr']:.2f}"
            + (f" minP>={min_p:.2f}" if min_p is not None else "")
        )

        score = float(dev_sel["best_macro_f1"])
        if score > best_dev_f1 + 1e-6:
            best_dev_f1 = score
            best_thr = float(dev_sel["best_thr"])
            best_epoch = epoch
            best_metrics = dict(m)
            sel_note = "DEV Macro-F1 @ anchored PHQ thr sweep (no TEST)"
            if min_p is not None:
                sel_note += f"; min_precision>={min_p:.3f}"
            if max_r is not None:
                sel_note += f"; max_recall<={max_r:.3f}"
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "seed": seed,
                    "epoch": epoch,
                    "best_dev_f1": best_dev_f1,
                    "best_dev_macro_f1": best_dev_f1,
                    "best_thr": best_thr,
                    "dev_metrics": best_metrics,
                    "class_counts": class_counts,
                    "min_precision": min_p,
                    "max_recall": max_r,
                    "selection": sel_note,
                },
                ckpt_path,
            )

    if not ckpt_path.is_file():
        raise RuntimeError(f"No DEV-best checkpoint saved for seed={seed}")

    seed_summary = {
        "seed": seed,
        "best_dev_f1": best_dev_f1,
        "best_thr": best_thr,
        "best_epoch": best_epoch,
        "ckpt_path": str(ckpt_path),
        "dev_metrics": best_metrics,
        "history": history,
        # TEST filled later in reporting pass only (after selection locked)
        "test_f1": None,
        "test_acc": None,
        "test_metrics": None,
    }
    (run_dir / f"seed_{seed}_summary.json").write_text(
        json.dumps(seed_summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    return seed_summary


def select_best_seed(
    seed_results: list[dict],
    *,
    thr_center: float | None = None,
    thr_radius: float | None = None,
) -> dict:
    """Pick global best by DEV Macro-F1; DEV-only tie-breaks (no TEST).

    When ``thr_center`` and ``thr_radius`` are set, first prefer seeds whose
    DEV best_thr lies in ``[center-radius, center+radius]`` (stability window).
    If none qualify, fall back to all seeds.

    When Macro/Acc/F1 tie (common on small DEV), prefer **thr closer to
    thr_center** (default 10), then lower thr, then higher seed id.
    """
    center = 10.0 if thr_center is None else float(thr_center)
    pool = list(seed_results)
    if thr_radius is not None and float(thr_radius) > 0:
        lo, hi = center - float(thr_radius), center + float(thr_radius)
        in_window = [
            r
            for r in pool
            if lo - 1e-9 <= float(r.get("best_thr", center)) <= hi + 1e-9
        ]
        if in_window:
            pool = in_window

    def _key(r: dict) -> tuple:
        dm = r.get("dev_metrics") or {}
        thr = float(r.get("best_thr", center))
        return (
            float(r["best_dev_f1"]),
            float(dm.get("accuracy", 0.0) or 0.0),
            float(dm.get("f1", 0.0) or 0.0),
            -abs(thr - center),  # closer to clinical center wins ties
            -thr,  # lower thr among equal distance (recall-leaning)
            int(r["seed"]),
        )

    return max(pool, key=_key)


def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> tuple[DeGaMSModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = DeGaMSConfig(**{k: v for k, v in ckpt["cfg"].items() if k in DeGaMSConfig.__dataclass_fields__})
    model = DeGaMSModel(cfg).to(device)
    # strict=False: older ckpts lack residual_fuse / newer keys added after training
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, ckpt


def eval_test(
    model: DeGaMSModel,
    test_loader: DataLoader,
    device: torch.device,
    thr: float,
) -> dict:
    y_te, s_te = predict_scores(model, test_loader, device)
    return metrics_at_threshold(y_te, s_te, thr)


def ensemble_test(
    seed_results: list[dict],
    test_loader: DataLoader,
    device: torch.device,
    *,
    dev_loader: DataLoader | None = None,
    top_k: int | None = None,
    thr_min: float = 8.0,
    thr_max: float = 12.0,
    thr_step: float = 0.1,
    thr_criterion: str = "macro_f1",
    min_precision: float | None = None,
    max_recall: float | None = None,
    require_precision_ge_recall: bool = False,
    prefer_thr: float | None = None,
    prefer_thr_weight: float = 0.0,
) -> dict:
    """Average soft PHQ scores across (top-k) seed ckpts; thr from DEV re-search.

    Prefer DEV thr search on averaged DEV scores (anti-leakage). Fallback to
    arithmetic mean of per-seed best_thr only when ``dev_loader`` is None.
    """
    members = list(seed_results)
    if top_k is not None and int(top_k) > 0 and int(top_k) < len(members):
        members = sorted(members, key=lambda r: float(r["best_dev_f1"]), reverse=True)[
            : int(top_k)
        ]
    members = sorted(members, key=lambda r: int(r["seed"]))

    te_stack = []
    dev_stack = []
    y_te_ref = None
    y_dev_ref = None
    per_seed_thrs = []
    member_seeds = []
    for r in members:
        model, _ckpt = load_model_from_ckpt(Path(r["ckpt_path"]), device)
        y_te, s_te = predict_scores(model, test_loader, device)
        if y_te_ref is None:
            y_te_ref = y_te
        elif not np.array_equal(y_te_ref, y_te):
            raise RuntimeError("TEST label order mismatch across seeds")
        te_stack.append(s_te)
        per_seed_thrs.append(float(r["best_thr"]))
        member_seeds.append(int(r["seed"]))
        if dev_loader is not None:
            y_dev, s_dev = predict_scores(model, dev_loader, device)
            if y_dev_ref is None:
                y_dev_ref = y_dev
            elif not np.array_equal(y_dev_ref, y_dev):
                raise RuntimeError("DEV label order mismatch across seeds")
            dev_stack.append(s_dev)

    mean_te = np.mean(np.stack(te_stack, axis=0), axis=0)
    mean_of_thrs = float(np.mean(per_seed_thrs))
    thr_method = "mean_of_per_seed_thr"
    ensemble_thr = mean_of_thrs
    dev_sel = None
    if dev_loader is not None and dev_stack:
        mean_dev = np.mean(np.stack(dev_stack, axis=0), axis=0)
        min_p = min_precision if min_precision and min_precision > 0 else None
        max_r = max_recall if max_recall and max_recall > 0 else None
        dev_sel = eval_dev_with_thr_search(
            y_dev_ref,
            mean_dev,
            thr_min=thr_min,
            thr_max=thr_max,
            thr_step=thr_step,
            criterion=thr_criterion,
            min_precision=min_p,
            max_recall=max_r,
            require_precision_ge_recall=require_precision_ge_recall,
            prefer_thr=prefer_thr,
            prefer_thr_weight=prefer_thr_weight,
        )
        ensemble_thr = float(dev_sel["best_thr"])
        thr_method = "dev_researched_on_mean_scores"

    metrics = metrics_at_threshold(y_te_ref, mean_te, ensemble_thr)
    return {
        "mean_thr": ensemble_thr,  # backward-compat key (= ensemble thr used on TEST)
        "ensemble_thr": ensemble_thr,
        "mean_of_per_seed_thrs": mean_of_thrs,
        "thr_method": thr_method,
        "n_seeds": len(members),
        "member_seeds": member_seeds,
        "top_k": int(top_k) if top_k else None,
        "per_seed_thrs": per_seed_thrs,
        "dev_macro_f1_at_ensemble_thr": (
            float(dev_sel["best_macro_f1"]) if dev_sel is not None else None
        ),
        "metrics": metrics,
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "macro_f1": metrics["macro_f1"],
        "accuracy": metrics["accuracy"],
    }


def _format_metrics_row(m: dict) -> str:
    return (
        f"P={m['precision']:.4f} R={m['recall']:.4f} "
        f"dep-F1={m['f1']:.4f} Macro-F1={m['macro_f1']:.4f} Acc={m['accuracy']:.4f}"
    )


def write_summary_table(
    run_dir: Path,
    seed_results: list[dict],
    best: dict,
    official_test: dict,
    ensemble: dict | None,
    args: argparse.Namespace,
    seeds: list[int],
    gate_stats: dict | None = None,
) -> dict:
    rows = []
    lines = [
        "# DeGaMS ABDE seed sweep summary",
        "",
        f"- Seeds: `{seeds}` (= range({args.start_seed}, {args.start_seed}+{args.num_seeds}))",
        f"- Selection: **DEV Macro-F1 only** (TEST never used to pick seed / thr / epoch)",
        f"- Thr band: PHQ soft-score [{args.thr_min}, {args.thr_max}] step {args.thr_step}",
        f"- Official thr-window: center={getattr(args, 'sel_thr_center', 10.0)} "
        f"radius={getattr(args, 'sel_thr_radius', None)}",
        f"- min_precision: {args.min_precision if args.min_precision and args.min_precision > 0 else 'off'}",
        f"- max_recall: {args.max_recall if getattr(args, 'max_recall', 0) and args.max_recall > 0 else 'off'}",
        f"- neg_emphasis: {args.neg_emphasis}",
        f"- Global best seed: **{best['seed']}** (DEV Macro-F1={best['best_dev_f1']:.4f}, thr*={best['best_thr']:.2f})",
        f"- Official TEST (global best once): {_format_metrics_row(official_test)}",
        "",
        "| Seed | best_dev_f1 | best_thr | test_f1 | test_acc | note |",
        "|------|-------------|----------|---------|----------|------|",
    ]
    for r in seed_results:
        mark = " **Global Best**" if int(r["seed"]) == int(best["seed"]) else ""
        te = r.get("test_metrics") or {}
        test_f1 = te.get("f1", r.get("test_f1"))
        test_acc = te.get("accuracy", r.get("test_acc"))
        row = {
            "seed": int(r["seed"]),
            "best_dev_f1": float(r["best_dev_f1"]),
            "best_thr": float(r["best_thr"]),
            "best_epoch": int(r["best_epoch"]),
            "ckpt_path": r["ckpt_path"],
            "test_f1": float(test_f1) if test_f1 is not None else None,
            "test_acc": float(test_acc) if test_acc is not None else None,
            "test_precision": float(te["precision"]) if te else None,
            "test_recall": float(te["recall"]) if te else None,
            "test_macro_f1": float(te["macro_f1"]) if te else None,
            "is_global_best": int(r["seed"]) == int(best["seed"]),
        }
        rows.append(row)
        tf = f"{row['test_f1']:.4f}" if row["test_f1"] is not None else "n/a"
        ta = f"{row['test_acc']:.4f}" if row["test_acc"] is not None else "n/a"
        lines.append(
            f"| {row['seed']} | {row['best_dev_f1']:.4f} | {row['best_thr']:.2f} | "
            f"{tf} | {ta} |{mark} |"
        )

    if ensemble is not None:
        thr_method = ensemble.get("thr_method", "mean_of_per_seed_thr")
        members = ensemble.get("member_seeds")
        mem_s = f" seeds={members}" if members is not None else ""
        top_k = ensemble.get("top_k")
        top_s = f" top_k={top_k}" if top_k else ""
        lines.extend(
            [
                "",
                "## Ensemble TEST",
                f"- thr={ensemble.get('ensemble_thr', ensemble['mean_thr']):.4f} "
                f"({thr_method}) over {ensemble['n_seeds']} ckpts{top_s}{mem_s}",
                f"- mean_of_per_seed_thrs={ensemble.get('mean_of_per_seed_thrs', ensemble['mean_thr']):.4f}",
                f"- {_format_metrics_row(ensemble['metrics'])}",
            ]
        )

    if gate_stats:
        gd = gate_stats.get("dev") or {}
        gt = gate_stats.get("test") or {}
        lines.extend(["", "## Gate diagnostics (global-best ckpt)"])
        if gd:
            lines.append(
                f"- DEV mean gates T/A/V="
                f"{gd['mean_t']:.3f}/{gd['mean_a']:.3f}/{gd['mean_v']:.3f} "
                f"(entropy={gd['entropy_of_mean']:.3f}, uniform={gd['uniform_entropy']:.3f})"
            )
        if gt:
            lines.append(
                f"- TEST mean gates T/A/V="
                f"{gt['mean_t']:.3f}/{gt['mean_a']:.3f}/{gt['mean_v']:.3f} "
                f"(entropy={gt['entropy_of_mean']:.3f})"
            )
        if not gd and not gt:
            lines.append("- Gate ablated or unavailable")

    lines.extend(
        [
            "",
            "## Anti-leakage notes",
            "- Checkpoint / thr / seed chosen exclusively on DEV.",
            "- Official TEST metrics come from a single eval of the locked global-best ckpt+thr.",
            "- Per-seed TEST columns are a post-selection reporting pass only.",
        ]
    )

    md = "\n".join(lines) + "\n"
    (run_dir / "SUMMARY.md").write_text(md, encoding="utf-8")

    payload = {
        "protocol": (
            "DeGaMS ABDE seed sweep · MentalBERT + Mel + dual-visual + hybrid · "
            "dynamic seeds · DEV Macro-F1 select · single TEST · optional ensemble"
        ),
        "modality_ablation": {
            "drop_text": bool(args.drop_text),
            "drop_audio": bool(args.drop_audio),
            "drop_visual": bool(args.drop_visual),
            "kept": "".join(
                m
                for m, d in (
                    ("T", args.drop_text),
                    ("A", args.drop_audio),
                    ("V", args.drop_visual),
                )
                if not d
            ),
        },
        "anti_leakage": {
            "selection_uses_test": False,
            "seed_selection": "argmax DEV Macro-F1",
            "thr_selection": "DEV grid search in anchored PHQ band",
            "ckpt_selection": "per-seed best DEV Macro-F1 across epochs",
            "official_test": "single eval of global best seed ckpt + its DEV best_thr",
            "per_seed_test": "reporting only after selection locked",
        },
        "seeds": seeds,
        "num_seeds": args.num_seeds,
        "start_seed": args.start_seed,
        "thr_band": {"min": args.thr_min, "max": args.thr_max, "step": args.thr_step},
        "min_precision": float(args.min_precision) if args.min_precision else 0.0,
        "max_recall": float(args.max_recall) if getattr(args, "max_recall", 0) else 0.0,
        "neg_emphasis": float(args.neg_emphasis),
        "global_best_seed": int(best["seed"]),
        "global_best_dev_f1": float(best["best_dev_f1"]),
        "global_best_thr": float(best["best_thr"]),
        "global_best_ckpt": best["ckpt_path"],
        "official_test": official_test,
        "ensemble_enabled": ensemble is not None,
        "ensemble_test": ensemble,
        "gate_stats": gate_stats,
        "per_seed": rows,
        "args": vars(args),
        "run_dir": str(run_dir),
    }
    (run_dir / "SUMMARY.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print(md)
    print("saved", run_dir / "SUMMARY.json")
    print("saved", run_dir / "SUMMARY.md")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DeGaMS ABDE dynamic seed sweep (DEV-driven)")
    p.add_argument("--num_seeds", type=int, default=5, help="Number of consecutive seeds")
    p.add_argument("--start_seed", type=int, default=0, help="First seed (inclusive)")
    p.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Explicit comma-separated seeds (overrides --start_seed/--num_seeds), e.g. 0,1,2,69369",
    )
    p.add_argument(
        "--ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average TEST soft scores across per-seed best-DEV ckpts (default: True)",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument(
        "--filter-mode",
        type=str,
        default="hybrid",
        choices=["hybrid", "keyword", "none"],
    )
    p.add_argument("--hybrid-target-keep", type=int, default=48)
    p.add_argument("--hybrid-min-frac", type=float, default=0.25)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--neg-emphasis", type=float, default=1.0)
    p.add_argument("--no-dual-visual", action="store_true")
    # Anchored clinical PHQ band; scores are soft PHQ totals (not probs)
    p.add_argument("--thr-min", type=float, default=8.0)
    p.add_argument("--thr-max", type=float, default=12.0)
    p.add_argument("--thr-step", type=float, default=0.1)
    p.add_argument(
        "--thr-criterion",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "balanced_score", "f1"],
    )
    p.add_argument(
        "--min-precision",
        type=float,
        default=0.0,
        help="DEV thr must meet depression Precision≥this (0=disabled). "
        "Maximizes Macro-F1 among feasible points; fallback if none.",
    )
    p.add_argument(
        "--max-recall",
        type=float,
        default=0.0,
        help="DEV thr must meet depression Recall≤this (0=disabled). "
        "Applied with min-precision; falls back to min-P-only if none feasible.",
    )
    p.add_argument(
        "--require-precision-ge-recall",
        action="store_true",
        help="DEV thr must also satisfy Precision≥Recall (prefer balanced points).",
    )
    p.add_argument(
        "--prefer-thr",
        type=float,
        default=0.0,
        help="Soft-prefer DEV thr near this PHQ value (0=disabled). Used as tie-break / mild penalty.",
    )
    p.add_argument(
        "--prefer-thr-weight",
        type=float,
        default=0.0,
        help="Penalty weight * |thr-prefer_thr| subtracted from criterion (e.g. 0.002).",
    )
    p.add_argument(
        "--ensemble-top-k",
        type=int,
        default=0,
        help="Ensemble only top-k seeds by DEV Macro-F1 (0=all seeds).",
    )
    p.add_argument(
        "--sel-thr-center",
        type=float,
        default=10.0,
        help="Official seed selection: prefer DEV thr near this center",
    )
    p.add_argument(
        "--sel-thr-radius",
        type=float,
        default=0.0,
        help="If >0, prefer seeds with |best_thr-center|<=radius (0=disabled)",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument(
        "--skip-per-seed-test",
        action="store_true",
        help="Only run official single TEST (+ensemble); skip per-seed TEST reporting",
    )
    # Modality ablations (group A)
    p.add_argument("--drop-text", action="store_true", help="Ablate text modality")
    p.add_argument("--drop-audio", action="store_true", help="Ablate audio modality")
    p.add_argument("--drop-visual", action="store_true", help="Ablate visual modality")
    # Model/loss ablations (group C)
    p.add_argument("--lambda-sscl", type=float, default=0.0, help="Ignored; SSCL retired")
    p.add_argument("--lambda-lbcl", type=float, default=0.004)
    p.add_argument("--lambda-mse", type=float, default=0.01)
    p.add_argument("--lambda-kl", type=float, default=0.8)
    p.add_argument("--soft-temp", type=float, default=0.35)
    p.add_argument("--ablate-sscl", action="store_true", help="Disable SSCL (lambda_sscl=0)")
    p.add_argument("--ablate-lbcl", action="store_true", help="Disable LBCL (lambda_lbcl=0)")
    p.add_argument(
        "--ablate-soft-phq",
        action="store_true",
        help="Hard CE on PHQ items instead of soft KL targets",
    )
    p.add_argument(
        "--ablate-msaf",
        action="store_true",
        help="Replace MSAF with masked mean pooling",
    )
    p.add_argument(
        "--ablate-decomp",
        action="store_true",
        help="Skip shared/specific decomposition; fuse encoder H directly",
    )
    p.add_argument("--lambda-cmcl", type=float, default=0.05, help="CMCL weight (0 disables)")
    p.add_argument("--cmcl-temp", type=float, default=0.07)
    p.add_argument("--no-cmcl-av", action="store_true", help="Skip A–V pair in CMCL")
    p.add_argument("--ablate-cmcl", action="store_true", help="Disable CMCL (lambda_cmcl=0)")
    p.add_argument(
        "--ablate-gate",
        action="store_true",
        help="Equal-sum fusion without learned Modality Gate",
    )
    p.add_argument(
        "--gate-temp",
        type=float,
        default=1.5,
        help="ModalityGate softmax temperature (>1 softer; only if Gate on)",
    )
    p.add_argument(
        "--fusion-mode",
        type=str,
        default="gate",
        choices=["gate", "equal", "residual"],
        help="gate=ModalityGate; equal=ft+fa+fv; residual=ft+αfa+βfv",
    )
    p.add_argument(
        "--gate-quality-prior",
        action="store_true",
        help="Bias audio gate logit by per-turn audio_quality",
    )
    p.add_argument("--gate-quality-scale", type=float, default=2.0)
    p.add_argument(
        "--ablate-turn-attn",
        action="store_true",
        help="Replace TurnAttn with masked mean pooling over turns",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if str(getattr(args, "seeds", "") or "").strip():
        seeds = sorted({int(x.strip()) for x in args.seeds.split(",") if x.strip()})
    else:
        seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    if not seeds:
        raise SystemExit("--num_seeds must be >= 1 (or pass --seeds)")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_root = Path(args.out_dir) if args.out_dir else OUT
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    if not args.drop_text:
        kept.append("T")
    if not args.drop_audio:
        kept.append("A")
    if not args.drop_visual:
        kept.append("V")
    if not kept:
        raise SystemExit("Cannot drop all modalities")
    min_p = args.min_precision if args.min_precision and args.min_precision > 0 else None
    max_r = args.max_recall if getattr(args, "max_recall", 0) and args.max_recall > 0 else None
    print(
        f"[DeGaMS seed-sweep] seeds={seeds} device={device} ensemble={args.ensemble} "
        f"filter={args.filter_mode} dual_visual={not args.no_dual_visual} "
        f"neg_emphasis={args.neg_emphasis} min_precision={min_p} max_recall={max_r} "
        f"modalities={'+'.join(kept)} "
        f"drop=(text={args.drop_text},audio={args.drop_audio},visual={args.drop_visual})"
    )
    print(
        f"[anti-leakage] selection=DEV Macro-F1"
        + (f" under P>={min_p}" if min_p is not None else "")
        + (f" and R<={max_r}" if max_r is not None else "")
        + "; official TEST=global best once; per-seed TEST=reporting"
    )

    shared = _build_loaders(args, device)
    print(
        f"[data] n_train={len(shared['train_ds'])} n_dev={len(shared['dev_ds'])} "
        f"n_test={len(shared['test_ds'])} class n0={shared['n0']:.0f} n1={shared['n1']:.0f}"
    )
    print("[filter train]", shared["train_ds"].filter_stats)
    print("[filter dev]", shared["dev_ds"].filter_stats)
    print("[filter test]", shared["test_ds"].filter_stats)

    seed_results: list[dict] = []
    for seed in seeds:
        result = train_one_seed(args, seed, run_dir, device, shared)
        seed_results.append(result)
        print(
            f"[seed {seed}] DONE best_dev_f1={result['best_dev_f1']:.4f} "
            f"best_thr={result['best_thr']:.2f} epoch={result['best_epoch']}"
        )

    # --- Selection locked on DEV only ---
    best = select_best_seed(
        seed_results,
        thr_center=float(getattr(args, "sel_thr_center", 10.0) or 10.0),
        thr_radius=(
            float(args.sel_thr_radius)
            if getattr(args, "sel_thr_radius", None) is not None
            and float(args.sel_thr_radius) > 0
            else None
        ),
    )
    print(
        f"[select] Global Best Seed={best['seed']} "
        f"DEV Macro-F1={best['best_dev_f1']:.4f} thr*={best['best_thr']:.2f} "
        f"ckpt={best['ckpt_path']}"
    )

    # --- Official: single TEST eval of global best ---
    model, _ckpt = load_model_from_ckpt(Path(best["ckpt_path"]), device)
    official_test = eval_test(model, shared["test_loader"], device, float(best["best_thr"]))
    print("[official TEST]", _format_metrics_row(official_test))

    gate_stats = None
    if not bool(getattr(args, "ablate_gate", False)):
        g_dev = collect_gate_stats(model, shared["dev_loader"], device)
        g_te = collect_gate_stats(model, shared["test_loader"], device)
        gate_stats = {"dev": g_dev, "test": g_te}
        if g_dev:
            print(
                f"[gate DEV] mean T/A/V="
                f"{g_dev['mean_t']:.3f}/{g_dev['mean_a']:.3f}/{g_dev['mean_v']:.3f} "
                f"H={g_dev['entropy_of_mean']:.3f}"
            )
        if g_te:
            print(
                f"[gate TEST] mean T/A/V="
                f"{g_te['mean_t']:.3f}/{g_te['mean_a']:.3f}/{g_te['mean_v']:.3f} "
                f"H={g_te['entropy_of_mean']:.3f}"
            )

    # --- Reporting pass: per-seed TEST (does not feed selection) ---
    if not args.skip_per_seed_test:
        print("[reporting] per-seed TEST on locked best-DEV ckpts (not used for selection)")
        for r in seed_results:
            m, _ = load_model_from_ckpt(Path(r["ckpt_path"]), device)
            te = eval_test(m, shared["test_loader"], device, float(r["best_thr"]))
            r["test_metrics"] = te
            r["test_f1"] = float(te["f1"])
            r["test_acc"] = float(te["accuracy"])
            (run_dir / f"seed_{r['seed']}_summary.json").write_text(
                json.dumps(r, ensure_ascii=False, indent=2, default=float),
                encoding="utf-8",
            )
            print(f"  seed {r['seed']} TEST {_format_metrics_row(te)}")

    ensemble = None
    if args.ensemble:
        top_k = int(getattr(args, "ensemble_top_k", 0) or 0) or None
        prefer_thr = getattr(args, "prefer_thr", None)
        prefer_w = float(getattr(args, "prefer_thr_weight", 0.0) or 0.0)
        if prefer_thr is not None and float(prefer_thr) <= 0:
            prefer_thr = None
        min_p = args.min_precision if args.min_precision and args.min_precision > 0 else None
        max_r = args.max_recall if getattr(args, "max_recall", 0) and args.max_recall > 0 else None
        print(
            "[ensemble] avg soft PHQ; thr=DEV re-search on mean DEV scores"
            + (f"; top_k={top_k}" if top_k else "")
        )
        ensemble = ensemble_test(
            seed_results,
            shared["test_loader"],
            device,
            dev_loader=shared["dev_loader"],
            top_k=top_k,
            thr_min=args.thr_min,
            thr_max=args.thr_max,
            thr_step=args.thr_step,
            thr_criterion=args.thr_criterion,
            min_precision=min_p,
            max_recall=max_r,
            require_precision_ge_recall=bool(getattr(args, "require_precision_ge_recall", False)),
            prefer_thr=float(prefer_thr) if prefer_thr is not None else None,
            prefer_thr_weight=prefer_w,
        )
        print("[ensemble TEST]", _format_metrics_row(ensemble["metrics"]))

    write_summary_table(
        run_dir, seed_results, best, official_test, ensemble, args, seeds, gate_stats=gate_stats
    )
    return 0


if __name__ == "__main__":
    # Hard-exit: do not touch CUDA before leave (sync/empty_cache/gc crash on WDDM).
    exit_clean(main())
