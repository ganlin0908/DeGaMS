"""DeGaMS ABDE dynamic seed sweep on E-DAIC (DEV-driven; single TEST; optional ensemble).

Paper protocol (same as DAIC seed sweep; reported runs):
  1. Train once per seed for --epochs (default 100).
  2. Each epoch: DEV grid-search τ in [8, 12] by Macro-F1 only (no retraining).
  3. Rank seeds by DEV Macro-F1; evaluate official TEST once.
  4. Optional --ensemble: average soft PHQ; τ re-fit on averaged DEV scores only.

Features: MentalBERT [768-D] + Enhanced wav2vec [768-D] + Visual Dense (OF29 + DenseNet768 = 797-D) + hybrid turns.
Reported runs set DEGAMS_FEATURE_ROOT to the avboost tree (visual→visual_dense, wav2vec→wav2vec_enh).
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
from degams_model import DeGaMSConfig, DeGaMSModel, degams_total_loss  # noqa: E402
from edaic_degams_dataset import EDAICDeGaMSDataset, collate_degams  # noqa: E402
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
    select_best_seed,
    write_summary_table,
)

OUT = Path(
    os.environ.get("DEGAMS_RUN_ROOT")
    or str(Path(__file__).resolve().parents[2] / "runs")
) / "edaic_degams_seed_sweep_tuned"


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
    train_ds = EDAICDeGaMSDataset("train", **ds_kw)
    dev_ds = EDAICDeGaMSDataset("dev", **ds_kw)
    test_ds = EDAICDeGaMSDataset("test", **ds_kw)
    if len(train_ds) == 0:
        raise SystemExit("Empty train set — E-DAIC features/mel not ready?")

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


def _make_cfg(args: argparse.Namespace, vdim: int, mel_dim: int = 88) -> DeGaMSConfig:
    lam_sscl = 0.0  # SSCL retired in degams_total_loss
    lam_lbcl = float(getattr(args, "lambda_lbcl", 0.004))
    if getattr(args, "ablate_lbcl", False):
        lam_lbcl = 0.0
    lam_cmcl = float(getattr(args, "lambda_cmcl", 0.05))
    if getattr(args, "ablate_cmcl", False):
        lam_cmcl = 0.0
    hi_dim = 768 if int(vdim) >= 29 + 768 else 0
    # clean_v2 OpenFace-210 → dual-visual ON by default (OF head); dense → DenseVisual
    use_dual = bool(getattr(args, "dual_visual", True)) and hi_dim == 0
    if bool(getattr(args, "no_dual_visual", False)):
        use_dual = False
    return DeGaMSConfig(
        mel_dim=int(mel_dim),
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
    set_seed(seed)
    train_loader = DataLoader(
        shared["train_ds"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_degams,
        pin_memory=shared["pin"],
    )
    dev_loader = shared["dev_loader"]
    class_counts = shared["class_counts"]
    cfg = _make_cfg(args, shared["vdim"], shared.get("mel_dim", 88))
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
        f"thr_band=[{args.thr_min},{args.thr_max}] step={args.thr_step} "
        f"vdim={shared['vdim']} dual_visual={cfg.use_dual_visual}"
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
        # Paper protocol: unconstrained Macro-F1 on the DEV PHQ grid (no extra thr knobs).
        dev_sel = eval_dev_with_thr_search(
            y_dev,
            s_dev,
            thr_min=args.thr_min,
            thr_max=args.thr_max,
            thr_step=args.thr_step,
            criterion=args.thr_criterion,
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
        }
        history.append(row)
        print(
            f"[seed {seed}] epoch {epoch:03d} loss={row['loss']:.4f} "
            f"DEV Macro-F1={row['dev_macro_f1']:.4f} depF1={row['dev_dep_f1']:.4f} "
            f"P={row['dev_p']:.3f} R={row['dev_r']:.3f} Acc={row['dev_acc']:.3f} "
            f"thr*={row['best_thr']:.2f}"
        )

        score = float(dev_sel["best_macro_f1"])
        if score > best_dev_f1 + 1e-6:
            best_dev_f1 = score
            best_thr = float(dev_sel["best_thr"])
            best_epoch = epoch
            best_metrics = dict(m)
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
                    "corpus": "edaic",
                    "selection": "DEV Macro-F1 @ anchored PHQ thr sweep (no TEST)",
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
    p = argparse.ArgumentParser(description="E-DAIC DeGaMS ABDE dynamic seed sweep")
    p.add_argument("--num_seeds", type=int, default=5)
    p.add_argument("--start_seed", type=int, default=0)
    p.add_argument("--seeds", type=str, default="")
    p.add_argument(
        "--ensemble",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--epochs", type=int, default=100)
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
    p.add_argument("--neg-emphasis", type=float, default=1.55)
    p.add_argument(
        "--dual-visual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dual visual heads when vdim=210; Visual Dense 797-d uses DenseVisual (this flag is ignored)",
    )
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
        "--ensemble-top-k",
        type=int,
        default=0,
        help="Optional reporting only; 0=all seeds.",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--skip-per-seed-test", action="store_true")
    p.add_argument("--drop-text", action="store_true")
    p.add_argument("--drop-audio", action="store_true")
    p.add_argument("--drop-visual", action="store_true")
    p.add_argument("--lambda-sscl", type=float, default=0.0, help="Ignored; SSCL retired")
    p.add_argument("--lambda-lbcl", type=float, default=0.004)
    p.add_argument("--lambda-mse", type=float, default=0.01)
    p.add_argument("--lambda-kl", type=float, default=0.8)
    p.add_argument("--soft-temp", type=float, default=0.35)
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
    p.add_argument("--gate-temp", type=float, default=1.5)
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
    p.add_argument("--ablate-turn-attn", action="store_true")
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

    print(
        f"[E-DAIC DeGaMS seed-sweep] seeds={seeds} device={device} ensemble={args.ensemble} "
        f"filter={args.filter_mode} dual_visual={bool(getattr(args, 'dual_visual', False))} "
        f"neg_emphasis={args.neg_emphasis} modalities={'+'.join(kept)} "
        f"bs={args.batch_size} lr={args.lr} "
        f"ablate_cmcl={args.ablate_cmcl} ablate_gate={args.ablate_gate} "
        f"ablate_turn_attn={args.ablate_turn_attn}"
    )
    print("[anti-leakage] selection=DEV Macro-F1; official TEST=global best once")

    shared = _build_loaders(args, device)
    print(
        f"[data] n_train={len(shared['train_ds'])} n_dev={len(shared['dev_ds'])} "
        f"n_test={len(shared['test_ds'])} class n0={shared['n0']:.0f} n1={shared['n1']:.0f} "
        f"vdim={shared['vdim']}"
    )
    print("[filter train]", shared["train_ds"].filter_stats)
    print("[filter dev]", shared["dev_ds"].filter_stats)
    print("[filter test]", shared["test_ds"].filter_stats)

    print("[protocol] DEV Macro-F1 thr grid only; official TEST once")
    seed_results: list[dict] = []
    for seed in seeds:
        result = train_one_seed(args, seed, run_dir, device, shared)
        seed_results.append(result)
        print(
            f"[seed {seed}] DONE best_dev_f1={result['best_dev_f1']:.4f} "
            f"best_thr={result['best_thr']:.2f} epoch={result['best_epoch']}"
        )

    best = select_best_seed(
        seed_results,
        thr_center=10.0,
        thr_radius=None,
    )
    print(
        f"[select] Global Best Seed={best['seed']} "
        f"DEV Macro-F1={best['best_dev_f1']:.4f} thr*={best['best_thr']:.2f}"
    )

    model, _ckpt = load_model_from_ckpt(Path(best["ckpt_path"]), device)
    official_test = eval_test(model, shared["test_loader"], device, float(best["best_thr"]))
    print(
        "[official TEST] "
        f"P={official_test['precision']:.4f} R={official_test['recall']:.4f} "
        f"dep-F1={official_test['f1']:.4f} Macro-F1={official_test['macro_f1']:.4f} "
        f"Acc={official_test['accuracy']:.4f}"
    )

    gate_stats = None
    if not bool(getattr(args, "ablate_gate", False)):
        gate_stats = {
            "dev": collect_gate_stats(model, shared["dev_loader"], device),
            "test": collect_gate_stats(model, shared["test_loader"], device),
        }

    if not args.skip_per_seed_test:
        print("[reporting] per-seed TEST on locked best-DEV ckpts")
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

    ensemble = None
    if args.ensemble:
        top_k = int(getattr(args, "ensemble_top_k", 0) or 0) or None
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
        )
        print(
            "[ensemble TEST] "
            f"P={ensemble['precision']:.4f} R={ensemble['recall']:.4f} "
            f"dep-F1={ensemble['f1']:.4f} Macro-F1={ensemble['macro_f1']:.4f} "
            f"Acc={ensemble['accuracy']:.4f} thr={ensemble['ensemble_thr']:.2f} "
            f"method={ensemble['thr_method']} members={ensemble.get('member_seeds')}"
        )

    payload = write_summary_table(
        run_dir, seed_results, best, official_test, ensemble, args, seeds, gate_stats=gate_stats
    )
    # Annotate corpus in SUMMARY.json
    payload["corpus"] = "edaic"
    payload["protocol"] = (
        "E-DAIC DeGaMS ABDE seed sweep · MentalBERT + Enhanced wav2vec + Visual Dense (avboost) · "
        "hybrid · soft-PHQ · DEV Macro-F1 select · single TEST · optional ensemble"
    )
    payload["features"] = {
        "text": "MentalBERT [T,768] via DEGAMS_FEATURE_ROOT/text",
        "audio": "Enhanced wav2vec 2.0 [T,768] (wav2vec_enh) via DEGAMS_FEATURE_ROOT/wav2vec",
        "visual": "Combined Visual [T,797] (OpenFace 29-d + DenseNet 768-d visual_dense) via DEGAMS_FEATURE_ROOT/visual",
        "labels": "EDAIC/label/{train,dev,test}_split.csv + detailed_lables PHQ items (train/dev only)",
    }
    (run_dir / "SUMMARY.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    exit_clean(main())
