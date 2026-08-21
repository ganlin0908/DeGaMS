"""CMDC DeGaMS ABDE seed sweep (video_fold · soft PHQ · Group C ready).

Mirrors DAIC/E-DAIC protocol; one fold per run via --fold / DEGAMS_CMDC_FOLD.

CMDC-specific anti-saturation selection (critical)
--------------------------------------------------
CMDC video_fold DEV has only ~7 subjects and HC/MDD PHQ totals are linearly
separable inside the clinical band [8,12]. Re-searching thr every epoch on
soft PHQ therefore drives DEV Macro-F1 to 1.0 within a few epochs while train
loss (PHQ regression) can still fall — this is a *metric ceiling*, not leakage.

Default ``--sel-mode fixed_thr``:
  - Checkpoint / seed ranking uses Macro-F1 @ locked clinical thr (default 10)
    with DEV PHQ-MAE as tie-break (lower MAE wins).
  - Thr-sweep Macro-F1 is logged as diagnostic only and never selects ckpts.
  - Ensemble / official TEST also lock that clinical thr (no DEV re-search).
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

from cmdc_degams_dataset import (  # noqa: E402
    CMDCDeGaMSDataset,
    assert_fold_disjoint,
    collate_degams,
)
from cuda_exit import exit_clean  # noqa: E402
from degams_model import DeGaMSConfig, DeGaMSModel, degams_total_loss  # noqa: E402
from train_daic_degams import (  # noqa: E402
    collect_gate_stats,
    eval_dev_with_thr_search,
    metrics_at_threshold,
    predict_scores,
    set_seed,
)
from train_daic_degams_seed_sweep import (  # noqa: E402
    ensemble_test,
    eval_test,
    load_model_from_ckpt,
    write_summary_table,
)

OUT = Path(
    os.environ.get("DEGAMS_RUN_ROOT")
    or str(Path(__file__).resolve().parents[2] / "runs")
) / "cmdc_degams_seed_sweep"
FEAT = Path(os.environ.get("DEGAMS_ROOT", r"D:\Depression")) / "egemaps_experiment" / "artifacts" / "degams_cmdc_tav"


@torch.no_grad()
def predict_scores_and_phq(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (y_bin, phq_pred, phq_total_gt) for continuous DEV monitoring."""
    model.eval()
    ys, scores, totals = [], [], []
    for batch in loader:
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        out = model(batch)
        ys.append(batch["y"].cpu().numpy())
        scores.append(out["phq_pred"].float().cpu().numpy())
        totals.append(batch["phq_total"].float().cpu().numpy())
    return np.concatenate(ys), np.concatenate(scores), np.concatenate(totals)


def _phq_mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred.astype(np.float64) - target.astype(np.float64))))


def _binary_auroc(y: np.ndarray, scores: np.ndarray) -> float:
    """Threshold-free ranking metric (Mann–Whitney). Returns nan if one-class."""
    y = y.astype(int)
    scores = scores.astype(np.float64)
    pos = scores[y == 1]
    neg = scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # P(score_pos > score_neg) + 0.5 P(equal)
    correct = 0.0
    for p in pos:
        correct += float((neg < p).sum()) + 0.5 * float((neg == p).sum())
    return float(correct / (pos.size * neg.size))


def _build_loaders(args: argparse.Namespace, device: torch.device):
    fold = int(args.fold)
    os.environ["DEGAMS_CMDC_FOLD"] = str(fold)
    os.environ.setdefault("DEGAMS_FEATURE_ROOT", str(FEAT))
    split_sizes = assert_fold_disjoint(fold)
    print(f"[anti-leak] fold{fold} sizes={split_sizes} disjoint=OK", flush=True)
    ds_kw = dict(
        max_len=args.max_len,
        keyword_filter=False,
        filter_mode="none",
        require_visual=True,
        fold=fold,
    )
    train_ds = CMDCDeGaMSDataset("train", **ds_kw)
    dev_ds = CMDCDeGaMSDataset("dev", **ds_kw)
    test_ds = CMDCDeGaMSDataset("test", **ds_kw)
    if len(train_ds) == 0:
        raise SystemExit(f"Empty train set fold={fold} — features not ready?")
    # Loaded-ID disjointness (features may drop subjects)
    id_sets = {
        "train": set(train_ds.ids),
        "dev": set(dev_ds.ids),
        "test": set(test_ds.ids),
    }
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        inter = id_sets[a] & id_sets[b]
        if inter:
            raise SystemExit(f"Loaded feature ID leakage {a}∩{b}: {sorted(inter)}")

    y_train = np.array([int(x["y"].item()) for x in train_ds._cache])
    n0 = float((y_train == 0).sum())
    n1 = float((y_train == 1).sum())
    class_counts = (n0, n1)

    # Cap BS to train size (CMDC folds are small)
    if args.batch_size > len(train_ds):
        print(f"[bs] cap {args.batch_size} → {len(train_ds)}", flush=True)
        args.batch_size = len(train_ds)

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_degams, pin_memory=pin
    )
    dev_loader = DataLoader(
        dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_degams, pin_memory=pin
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_degams, pin_memory=pin
    )
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
        "vdim": int(train_ds[0]["visual"].size(1)),
        "mel_dim": int(train_ds[0]["mel"].size(1)),
        "ds_kw": ds_kw,
        "pin": pin,
        "fold": fold,
    }


def _make_cfg(args: argparse.Namespace, vdim: int, mel_dim: int) -> DeGaMSConfig:
    lam_lbcl = float(getattr(args, "lambda_lbcl", 0.004))
    if getattr(args, "ablate_lbcl", False):
        lam_lbcl = 0.0
    lam_cmcl = float(getattr(args, "lambda_cmcl", 0.05))
    if getattr(args, "ablate_cmcl", False):
        lam_cmcl = 0.0
    hi_dim = 768 if int(vdim) >= 29 + 768 else 0
    use_dual = (not bool(getattr(args, "no_dual_visual", False))) and hi_dim == 0
    return DeGaMSConfig(
        mel_dim=int(mel_dim),
        visual_dim=vdim,
        hi_visual_dim=hi_dim,
        dropout=args.dropout,
        lambda_sscl=0.0,
        lambda_lbcl=lam_lbcl,
        lambda_mse=float(getattr(args, "lambda_mse", 0.01)),
        lambda_kl=float(getattr(args, "lambda_kl", 2.0)),
        lambda_cmcl=lam_cmcl,
        lambda_gate_entropy=float(getattr(args, "lambda_gate_entropy", 0.02)),
        cmcl_temp=float(getattr(args, "cmcl_temp", 0.07)),
        cmcl_include_av=not bool(getattr(args, "no_cmcl_av", False)),
        soft_temp=float(getattr(args, "soft_temp", 0.25)),
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
        gate_temp=float(getattr(args, "gate_temp", 0.75)),
        gate_min_mass=float(getattr(args, "gate_min_mass", 0.08)),
        train_mod_dropout=float(getattr(args, "train_mod_dropout", 0.12)),
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
    set_seed(seed)
    train_loader = DataLoader(
        shared["train_ds"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_degams,
        pin_memory=shared["pin"],
    )
    cfg = _make_cfg(args, shared["vdim"], shared["mel_dim"])
    model = DeGaMSModel(cfg).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Selection uses continuous / fixed-thr metrics — NEVER thr-sweep Macro alone.
    sel_mode = str(getattr(args, "sel_mode", "fixed_thr") or "fixed_thr")
    lock_thr = float(getattr(args, "lock_thr", 10.0) or 10.0)
    best_sel = -1e18  # higher is better after transforming MAE
    best_dev_f1 = -1.0  # stored = Macro@lock_thr (for seed ranking / ensemble top-k)
    best_thr = lock_thr
    best_epoch = 0
    best_metrics: dict = {}
    best_mae = float("inf")
    ckpt_path = run_dir / f"checkpoint_best_dev_seed_{seed}.pt"
    history: list[dict] = []
    class_counts = shared["class_counts"]
    dev_loader = shared["dev_loader"]
    patience = int(getattr(args, "early_stop_patience", 0) or 0)
    stale = 0

    print(
        f"[seed {seed}] fold={shared['fold']} epochs={args.epochs} "
        f"sel_mode={sel_mode} lock_thr={lock_thr} "
        f"vdim={shared['vdim']} mel={shared['mel_dim']} hi={cfg.hi_visual_dim}",
        flush=True,
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

        # Eval AFTER optimizer step (epoch end) — model is in eval mode inside predict_*.
        y_dev, s_dev, t_dev = predict_scores_and_phq(model, dev_loader, device)
        m_fixed = metrics_at_threshold(y_dev, s_dev, lock_thr)
        mae = _phq_mae(s_dev, t_dev)
        auroc = _binary_auroc(y_dev, s_dev)

        # Diagnostic only: thr-sweep Macro (saturates to 1.0 on CMDC DEV — do NOT select).
        min_p = args.min_precision if args.min_precision and args.min_precision > 0 else None
        max_r = args.max_recall if getattr(args, "max_recall", 0) and args.max_recall > 0 else None
        prefer_thr = getattr(args, "prefer_thr", None)
        if prefer_thr is not None and float(prefer_thr) <= 0:
            prefer_thr = None
        prefer_w = float(getattr(args, "prefer_thr_weight", 0.0) or 0.0)
        if sel_mode == "thr_sweep":
            dev_sel = eval_dev_with_thr_search(
                y_dev,
                s_dev,
                thr_min=args.thr_min,
                thr_max=args.thr_max,
                thr_step=args.thr_step,
                criterion=args.thr_criterion,
                min_precision=min_p,
                max_recall=max_r,
                require_precision_ge_recall=bool(
                    getattr(args, "require_precision_ge_recall", False)
                ),
                prefer_thr=float(prefer_thr) if prefer_thr is not None else None,
                prefer_thr_weight=prefer_w,
            )
            m_sweep = dict(dev_sel["metrics"])
            sweep_thr = float(dev_sel["best_thr"])
            sweep_macro = float(dev_sel["best_macro_f1"])
        else:
            # Cheap diagnostic: evaluate endpoints of band only for logging.
            m_sweep = metrics_at_threshold(y_dev, s_dev, lock_thr)
            sweep_thr = lock_thr
            # Full thr search still logged sparsely for debugging saturation.
            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                diag = eval_dev_with_thr_search(
                    y_dev,
                    s_dev,
                    thr_min=args.thr_min,
                    thr_max=args.thr_max,
                    thr_step=args.thr_step,
                    criterion=args.thr_criterion,
                )
                m_sweep = dict(diag["metrics"])
                sweep_thr = float(diag["best_thr"])
                sweep_macro = float(diag["best_macro_f1"])
            else:
                sweep_macro = float("nan")

        # Selection score (higher better)
        if sel_mode == "phq_mae":
            sel_score = -mae
            sel_metrics = dict(m_fixed)
            sel_thr = lock_thr
        elif sel_mode == "thr_sweep":
            sel_score = sweep_macro
            sel_metrics = m_sweep
            sel_thr = sweep_thr
        else:  # fixed_thr (default)
            # Macro@lock_thr primary; continuous MAE keeps improving after Macro saturates.
            mae_w = float(getattr(args, "mae_sel_weight", 0.01) or 0.01)
            sel_score = float(m_fixed["macro_f1"]) - mae_w * mae
            sel_metrics = dict(m_fixed)
            sel_thr = lock_thr

        row = {
            "epoch": epoch,
            "loss": total / max(n, 1),
            "sel_mode": sel_mode,
            "lock_thr": lock_thr,
            "best_thr": sel_thr,
            "dev_macro_f1": float(m_fixed["macro_f1"]),  # @ lock_thr — classification
            "dev_dep_f1": float(m_fixed["f1"]),
            "dev_p": float(m_fixed["precision"]),
            "dev_r": float(m_fixed["recall"]),
            "dev_acc": float(m_fixed["accuracy"]),
            "dev_phq_mae": mae,
            "dev_auroc": auroc if auroc == auroc else None,
            "dev_macro_f1_sweep": (
                float(sweep_macro) if sweep_macro == sweep_macro else None
            ),
            "dev_thr_sweep": sweep_thr,
            "sel_score": float(sel_score),
        }
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            sm = row["dev_macro_f1_sweep"]
            sm_s = f"{sm:.4f}" if sm is not None else "n/a"
            au_s = f"{auroc:.4f}" if auroc == auroc else "n/a"
            print(
                f"[seed {seed}] ep{epoch:03d} loss={row['loss']:.4f} "
                f"Macro@{lock_thr:g}={row['dev_macro_f1']:.4f} "
                f"MAE={mae:.3f} AUROC={au_s} sweepMacro={sm_s}",
                flush=True,
            )

        improved = sel_score > best_sel + 1e-6
        if improved:
            best_sel = float(sel_score)
            best_dev_f1 = float(m_fixed["macro_f1"])
            best_thr = float(sel_thr)
            best_epoch = epoch
            best_metrics = dict(sel_metrics)
            best_mae = mae
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "seed": seed,
                    "epoch": epoch,
                    "best_dev_f1": best_dev_f1,
                    "best_thr": best_thr,
                    "best_phq_mae": best_mae,
                    "sel_mode": sel_mode,
                    "lock_thr": lock_thr,
                    "sel_score": best_sel,
                    "dev_metrics": best_metrics,
                    "class_counts": class_counts,
                    "corpus": "cmdc",
                    "fold": shared["fold"],
                    "selection": (
                        f"CMDC {sel_mode}: Macro@{lock_thr} + PHQ-MAE tie-break; "
                        "thr-sweep Macro is diagnostic only"
                    ),
                },
                ckpt_path,
            )
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                print(
                    f"[seed {seed}] early-stop at ep{epoch} "
                    f"(patience={patience}, best_ep={best_epoch})",
                    flush=True,
                )
                break

    if not ckpt_path.is_file():
        raise RuntimeError(f"No DEV-best checkpoint for seed={seed}")
    seed_summary = {
        "seed": seed,
        "best_dev_f1": best_dev_f1,
        "best_thr": best_thr,
        "best_epoch": best_epoch,
        "best_phq_mae": best_mae,
        "sel_mode": sel_mode,
        "lock_thr": lock_thr,
        "sel_score": best_sel,
        "ckpt_path": str(ckpt_path),
        "dev_metrics": best_metrics,
        "history": history,
        "test_f1": None,
        "test_acc": None,
        "test_metrics": None,
    }
    (run_dir / f"seed_{seed}_summary.json").write_text(
        json.dumps(seed_summary, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    return seed_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CMDC DeGaMS seed sweep (video_fold)")
    p.add_argument("--fold", type=int, default=int(os.environ.get("DEGAMS_CMDC_FOLD") or "1"))
    p.add_argument("--num_seeds", type=int, default=5)
    p.add_argument("--start_seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default="")
    p.add_argument("--ensemble", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=0.0015)
    p.add_argument("--weight-decay", type=float, default=5e-5)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-len", type=int, default=128)
    p.add_argument("--filter-mode", type=str, default="none", choices=["hybrid", "keyword", "none"])
    p.add_argument("--hybrid-target-keep", type=int, default=48)
    p.add_argument("--hybrid-min-frac", type=float, default=0.25)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--neg-emphasis", type=float, default=1.55)
    p.add_argument("--no-dual-visual", action="store_true")
    p.add_argument("--thr-min", type=float, default=8.0)
    p.add_argument("--thr-max", type=float, default=12.0)
    p.add_argument("--thr-step", type=float, default=0.1)
    p.add_argument("--thr-criterion", type=str, default="macro_f1", choices=["macro_f1", "balanced_score", "f1"])
    p.add_argument("--min-precision", type=float, default=0.0)
    p.add_argument("--max-recall", type=float, default=0.0)
    p.add_argument("--require-precision-ge-recall", action="store_true")
    p.add_argument("--prefer-thr", type=float, default=0.0)
    p.add_argument("--prefer-thr-weight", type=float, default=0.0)
    p.add_argument("--ensemble-top-k", type=int, default=5)
    p.add_argument("--sel-thr-center", type=float, default=10.0)
    p.add_argument("--sel-thr-radius", type=float, default=1.5)
    p.add_argument(
        "--sel-mode",
        type=str,
        default="fixed_thr",
        choices=["fixed_thr", "phq_mae", "thr_sweep"],
        help="CMDC ckpt selection. fixed_thr=Macro@lock_thr (+MAE tie-break); "
        "phq_mae=min DEV PHQ MAE; thr_sweep=legacy saturating DEV thr search "
        "(NOT recommended on CMDC).",
    )
    p.add_argument(
        "--lock-thr",
        type=float,
        default=10.0,
        help="Clinical PHQ threshold locked for fixed_thr selection / TEST / ensemble",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="Stop if selection score does not improve for N epochs (0=disabled)",
    )
    p.add_argument(
        "--mae-sel-weight",
        type=float,
        default=0.01,
        help="Weight of DEV PHQ-MAE in fixed_thr selection: Macro - w*MAE",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--skip-per-seed-test", action="store_true")
    p.add_argument("--drop-text", action="store_true")
    p.add_argument("--drop-audio", action="store_true")
    p.add_argument("--drop-visual", action="store_true")
    p.add_argument("--lambda-sscl", type=float, default=0.0)
    p.add_argument("--lambda-lbcl", type=float, default=0.004)
    p.add_argument("--lambda-mse", type=float, default=0.01)
    p.add_argument("--lambda-kl", type=float, default=2.0)
    p.add_argument("--soft-temp", type=float, default=0.25)
    p.add_argument(
        "--lambda-gate-entropy",
        type=float,
        default=0.02,
        help="Encourage multi-modal gate mass (anti A-collapse)",
    )
    p.add_argument(
        "--gate-min-mass",
        type=float,
        default=0.08,
        help="Floor mass per present modality in ModalityGate",
    )
    p.add_argument(
        "--train-mod-dropout",
        type=float,
        default=0.12,
        help="Stochastic T/V dropout prob during Full TAV training",
    )
    p.add_argument("--ablate-sscl", action="store_true")
    p.add_argument("--ablate-lbcl", action="store_true")
    p.add_argument("--ablate-soft-phq", action="store_true")
    p.add_argument("--ablate-msaf", action="store_true")
    p.add_argument("--ablate-decomp", action="store_true")
    p.add_argument("--lambda-cmcl", type=float, default=0.05)
    p.add_argument("--cmcl-temp", type=float, default=0.07)
    p.add_argument("--no-cmcl-av", action="store_true")
    p.add_argument("--ablate-cmcl", action="store_true")
    p.add_argument("--ablate-gate", action="store_true")
    p.add_argument("--gate-temp", type=float, default=0.75)
    p.add_argument("--fusion-mode", type=str, default="gate", choices=["gate", "equal", "residual"])
    p.add_argument("--gate-quality-prior", action="store_true")
    p.add_argument("--gate-quality-scale", type=float, default=2.0)
    p.add_argument("--ablate-turn-attn", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if str(getattr(args, "seeds", "") or "").strip():
        seeds = sorted({int(x.strip()) for x in args.seeds.split(",") if x.strip()})
    else:
        seeds = list(range(args.start_seed, args.start_seed + args.num_seeds))
    if not seeds:
        raise SystemExit("no seeds")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_root = Path(args.out_dir) if args.out_dir else OUT
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    kept = [m for m, drop in (("T", args.drop_text), ("A", args.drop_audio), ("V", args.drop_visual)) if not drop]
    if not kept:
        raise SystemExit("Cannot drop all modalities")

    print(
        f"[CMDC DeGaMS] fold={args.fold} seeds={seeds} device={device} "
        f"mods={'+'.join(kept)} sel_mode={args.sel_mode} lock_thr={args.lock_thr} "
        f"ablate_lbcl={args.ablate_lbcl} ablate_soft_phq={args.ablate_soft_phq} "
        f"ablate_decomp={args.ablate_decomp}",
        flush=True,
    )
    if args.sel_mode != "thr_sweep":
        print(
            "[anti-saturation] ckpt/seed use Macro@lock_thr (+PHQ-MAE); "
            "thr-sweep Macro is diagnostic only (CMDC DEV n≈7 is PHQ-separable)",
            flush=True,
        )
    shared = _build_loaders(args, device)
    print(
        f"[data] n_train={len(shared['train_ds'])} n_dev={len(shared['dev_ds'])} "
        f"n_test={len(shared['test_ds'])} n0={shared['n0']:.0f} n1={shared['n1']:.0f} "
        f"vdim={shared['vdim']} mel={shared['mel_dim']} bs={args.batch_size}",
        flush=True,
    )

    seed_results = []
    for seed in seeds:
        result = train_one_seed(args, seed, run_dir, device, shared)
        seed_results.append(result)
        print(
            f"[seed {seed}] DONE Macro@{args.lock_thr:g}={result['best_dev_f1']:.4f} "
            f"MAE={result.get('best_phq_mae')} thr={result['best_thr']:.2f} "
            f"ep={result['best_epoch']}",
            flush=True,
        )

    # Prefer continuous MAE when Macro@lock_thr saturates across seeds.
    def _cmdc_seed_key(r: dict) -> tuple:
        return (
            float(r.get("best_dev_f1") or 0.0),
            -float(r.get("best_phq_mae") if r.get("best_phq_mae") is not None else 1e9),
            -int(r.get("best_epoch") or 0),  # earlier epoch if still tied
            int(r["seed"]),
        )

    best = max(seed_results, key=_cmdc_seed_key)
    print(
        f"[select] seed={best['seed']} Macro@{args.lock_thr:g}={best['best_dev_f1']:.4f} "
        f"MAE={best.get('best_phq_mae')} ep={best['best_epoch']}",
        flush=True,
    )
    lock_thr = float(getattr(args, "lock_thr", 10.0) or 10.0)
    use_lock = str(getattr(args, "sel_mode", "fixed_thr")) != "thr_sweep"
    eval_thr = lock_thr if use_lock else float(best["best_thr"])

    model, _ = load_model_from_ckpt(Path(best["ckpt_path"]), device)
    official_test = eval_test(model, shared["test_loader"], device, eval_thr)
    print(
        f"[official TEST] thr={eval_thr:g} P={official_test['precision']:.4f} "
        f"R={official_test['recall']:.4f} F1={official_test['f1']:.4f} "
        f"Macro={official_test['macro_f1']:.4f}",
        flush=True,
    )

    gate_stats = None
    if not bool(getattr(args, "ablate_gate", False)):
        gate_stats = {
            "dev": collect_gate_stats(model, shared["dev_loader"], device),
            "test": collect_gate_stats(model, shared["test_loader"], device),
        }

    if not args.skip_per_seed_test:
        for r in seed_results:
            m, _ = load_model_from_ckpt(Path(r["ckpt_path"]), device)
            te_thr = lock_thr if use_lock else float(r["best_thr"])
            te = eval_test(m, shared["test_loader"], device, te_thr)
            r["test_metrics"] = te
            r["test_f1"] = float(te["f1"])
            r["test_acc"] = float(te["accuracy"])
            (run_dir / f"seed_{r['seed']}_summary.json").write_text(
                json.dumps(r, ensure_ascii=False, indent=2, default=float),
                encoding="utf-8",
            )

    ensemble = None
    if args.ensemble:
        top_k = int(getattr(args, "ensemble_top_k", 0) or 0) or None
        members = sorted(seed_results, key=_cmdc_seed_key, reverse=True)
        if top_k is not None and top_k < len(members):
            members = members[: int(top_k)]
        if use_lock:
            te_stack = []
            y_ref = None
            t_ref = None
            for r in members:
                m, ck = load_model_from_ckpt(Path(r["ckpt_path"]), device)
                # Guard: ckpt drop flags must match this run's CLI (anti cache-mix)
                cfg_d = ck.get("cfg") or {}
                for key, want in (
                    ("drop_text", bool(args.drop_text)),
                    ("drop_audio", bool(args.drop_audio)),
                    ("drop_visual", bool(args.drop_visual)),
                ):
                    got = bool(cfg_d.get(key, False))
                    if got != want:
                        raise RuntimeError(
                            f"ckpt drop mismatch {r['ckpt_path']}: {key} ckpt={got} args={want}"
                        )
                y_te, s_te, t_te = predict_scores_and_phq(m, shared["test_loader"], device)
                if y_ref is None:
                    y_ref = y_te
                    t_ref = t_te
                elif not np.array_equal(y_ref, y_te):
                    raise RuntimeError("TEST label order mismatch across seeds")
                te_stack.append(s_te)
            mean_te = np.mean(np.stack(te_stack, axis=0), axis=0)
            m_ens = metrics_at_threshold(y_ref, mean_te, lock_thr)
            # Continuous metrics — expose real ablation gaps when F1@thr saturates
            mae = _phq_mae(mean_te, t_ref)
            auroc = _binary_auroc(y_ref, mean_te)
            pred_bin = (mean_te >= lock_thr).astype(int)
            ensemble = {
                "mean_thr": lock_thr,
                "ensemble_thr": lock_thr,
                "mean_of_per_seed_thrs": lock_thr,
                "thr_method": "locked_clinical_thr",
                "n_seeds": len(members),
                "n_members": len(members),
                "member_seeds": [int(r["seed"]) for r in members],
                "top_k": int(top_k) if top_k else None,
                "per_seed_thrs": [lock_thr] * len(members),
                "dev_macro_f1_at_ensemble_thr": None,
                "metrics": m_ens,
                "precision": m_ens["precision"],
                "recall": m_ens["recall"],
                "f1": m_ens["f1"],
                "macro_f1": m_ens["macro_f1"],
                "accuracy": m_ens["accuracy"],
                "continuous": {
                    "phq_mae": mae,
                    "auroc": auroc,
                    "phq_pred_mean": mean_te.tolist(),
                    "phq_true": t_ref.astype(float).tolist(),
                    "y_true": y_ref.astype(int).tolist(),
                    "y_pred_at_lock_thr": pred_bin.tolist(),
                    "note": (
                        "When HC/MDD PHQ bands are separable at lock_thr, F1 may "
                        "saturate identically across modalities even if phq_pred differs."
                    ),
                },
            }
        else:
            prefer_thr = getattr(args, "prefer_thr", None)
            if prefer_thr is not None and float(prefer_thr) <= 0:
                prefer_thr = None
            min_p = args.min_precision if args.min_precision and args.min_precision > 0 else None
            max_r = (
                args.max_recall if getattr(args, "max_recall", 0) and args.max_recall > 0 else None
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
                require_precision_ge_recall=bool(
                    getattr(args, "require_precision_ge_recall", False)
                ),
                prefer_thr=float(prefer_thr) if prefer_thr is not None else None,
                prefer_thr_weight=float(getattr(args, "prefer_thr_weight", 0.0) or 0.0),
            )
        print(
            f"[ensemble TEST] P={ensemble['precision']:.4f} R={ensemble['recall']:.4f} "
            f"F1={ensemble['f1']:.4f} Macro={ensemble['macro_f1']:.4f} "
            f"thr={ensemble['ensemble_thr']:.2f} method={ensemble.get('thr_method')}",
            flush=True,
        )
        cont = ensemble.get("continuous") or {}
        if cont:
            print(
                f"[ensemble continuous] PHQ-MAE={cont.get('phq_mae'):.4f} "
                f"AUROC={cont.get('auroc')} "
                f"(F1@thr may saturate while MAE/AUROC still differ)",
                flush=True,
            )

    payload = write_summary_table(
        run_dir, seed_results, best, official_test, ensemble, args, seeds, gate_stats=gate_stats
    )
    payload["corpus"] = "cmdc"
    payload["fold"] = int(args.fold)
    payload["sel_mode"] = str(args.sel_mode)
    payload["lock_thr"] = float(args.lock_thr)
    payload["protocol"] = (
        "CMDC DeGaMS · chinese-RoBERTa + wav2vec L6 + Q.npy∥OF29zeros · "
        f"video_fold · soft-PHQ · Gate ON · sel_mode={args.sel_mode} lock_thr={args.lock_thr} · "
        "thr-sweep Macro diagnostic only"
    )
    (run_dir / "SUMMARY.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    print("saved", run_dir / "SUMMARY.json", flush=True)
    return 0


if __name__ == "__main__":
    exit_clean(main())
