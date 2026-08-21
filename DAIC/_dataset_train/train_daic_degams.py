"""Train DeGaMS ABDE on DAIC (MentalBERT + Mel + dual-visual + hybrid turns).

Protocol improvements vs keyword-only baseline:
  A — DEV PHQ threshold sweep; select thr by Macro-F1 (also report paper thr=10)
  B — Hybrid utterance retention (keyword must-keep + longest non-kw fill → ~30–60 turns)
  C — SKIP (keep MentalBERT)
  D — Dual visual projection heads (CLNF 210 vs OpenFace 29)
  E — Checkpoint by DEV Macro-F1; mild class-weight rebalance on KL/MSE
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))

from daic_degams_dataset import DAICDeGaMSDataset, collate_degams  # noqa: E402
from degams_model import DeGaMSConfig, DeGaMSModel, degams_total_loss  # noqa: E402

OUT = Path(
    os.environ.get("DEGAMS_RUN_ROOT")
    or str(Path(__file__).resolve().parents[2] / "runs")
) / "daic_degams_abde"


def set_seed(seed: int) -> None:
    """Fully reset RNG state for a seed (CPU + CUDA + cudnn)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def binary_prf1(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(len(y_true), 1)
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "accuracy": acc,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def metrics_at_threshold(y_true: np.ndarray, scores: np.ndarray, thr: float) -> dict:
    pred = (scores >= thr).astype(int)
    dep = binary_prf1(y_true, pred)
    # Macro-F1 = mean of F1(class=1) and F1(class=0)
    neg = binary_prf1(1 - y_true, 1 - pred)
    macro_f1 = 0.5 * (dep["f1"] + neg["f1"])
    # Balanced selection score (prefer overall, not recall-only)
    balanced = 0.5 * (dep["f1"] + macro_f1)
    return {
        "threshold": float(thr),
        "depression": dep,
        "non_depression_f1": float(neg["f1"]),
        "macro_f1": float(macro_f1),
        "balanced_score": float(balanced),
        "accuracy": float(dep["accuracy"]),
        "precision": float(dep["precision"]),
        "recall": float(dep["recall"]),
        "f1": float(dep["f1"]),
    }


def sweep_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    thr_min: float = 4.0,
    thr_max: float = 16.0,
    thr_step: float = 0.25,
    criterion: str = "macro_f1",
    min_precision: float | None = None,
    max_recall: float | None = None,
    require_precision_ge_recall: bool = False,
    prefer_thr: float | None = None,
    prefer_thr_weight: float = 0.0,
) -> dict:
    """Pick threshold on DEV by overall criterion (default Macro-F1).

    If min_precision is set, only consider operating points with depression
    Precision >= min_precision (and ≥1 predicted positive). If max_recall is
    set, also require Recall <= max_recall (blocks extreme high-R / low-P
    points). If require_precision_ge_recall, also require P>=R on DEV.
    Soft prefer_thr (e.g. 10) with prefer_thr_weight mildly penalizes extreme thr.
    Fallback order: (full constraints) → drop P>=R → P only → max precision.
    """
    best = None
    grid = []
    thr = thr_min
    w_pref = float(prefer_thr_weight) if prefer_thr is not None else 0.0

    def _rank_key(m: dict) -> float:
        key = float(m.get(criterion, m["macro_f1"]))
        if prefer_thr is not None and w_pref > 0:
            key = key - w_pref * abs(float(m["threshold"]) - float(prefer_thr))
        return key

    def _tie_better(m: dict, cur: dict) -> bool:
        # Prefer higher precision; then closer to prefer_thr (or lower thr); then balanced
        if m["precision"] > cur["precision"] + 1e-12:
            return True
        if abs(m["precision"] - cur["precision"]) > 1e-12:
            return False
        if prefer_thr is not None:
            d_new = abs(float(m["threshold"]) - float(prefer_thr))
            d_old = abs(float(cur["threshold"]) - float(prefer_thr))
            if d_new < d_old - 1e-12:
                return True
            if d_new > d_old + 1e-12:
                return False
        else:
            if m["threshold"] < cur["threshold"] - 1e-12:
                return True
            if m["threshold"] > cur["threshold"] + 1e-12:
                return False
        return m["balanced_score"] > cur["balanced_score"]

    while thr <= thr_max + 1e-9:
        m = metrics_at_threshold(y_true, scores, thr)
        grid.append(m)
        n_pred = m["depression"]["tp"] + m["depression"]["fp"]
        ok = n_pred > 0 and (
            min_precision is None or m["precision"] + 1e-12 >= float(min_precision)
        )
        if ok and max_recall is not None:
            ok = m["recall"] <= float(max_recall) + 1e-12
        if ok and require_precision_ge_recall:
            ok = m["precision"] + 1e-12 >= m["recall"]
        if ok:
            key = _rank_key(m)
            if best is None or key > best["score"] + 1e-12:
                best = {"score": key, "metrics": m}
            elif abs(key - best["score"]) <= 1e-12 and _tie_better(m, best["metrics"]):
                best = {"score": key, "metrics": m}
        thr = round(thr + thr_step, 10)
    used_max_recall = max_recall is not None and best is not None
    used_p_ge_r = require_precision_ge_recall and best is not None
    if best is None and require_precision_ge_recall:
        # Fallback: drop P>=R, keep min_precision (+ max_recall if set)
        thr = thr_min
        while thr <= thr_max + 1e-9:
            m = metrics_at_threshold(y_true, scores, thr)
            n_pred = m["depression"]["tp"] + m["depression"]["fp"]
            ok = n_pred > 0 and (
                min_precision is None or m["precision"] + 1e-12 >= float(min_precision)
            )
            if ok and max_recall is not None:
                ok = m["recall"] <= float(max_recall) + 1e-12
            if ok:
                key = float(m.get(criterion, m["macro_f1"]))
                if best is None or key > best["score"] + 1e-12:
                    best = {"score": key, "metrics": m}
            thr = round(thr + thr_step, 10)
        used_p_ge_r = False
        used_max_recall = max_recall is not None and best is not None
    if best is None and max_recall is not None and min_precision is not None:
        # Fallback: drop max_recall, keep min_precision
        thr = thr_min
        while thr <= thr_max + 1e-9:
            m = metrics_at_threshold(y_true, scores, thr)
            n_pred = m["depression"]["tp"] + m["depression"]["fp"]
            ok = n_pred > 0 and m["precision"] + 1e-12 >= float(min_precision)
            if ok:
                key = float(m.get(criterion, m["macro_f1"]))
                if best is None or key > best["score"] + 1e-12:
                    best = {"score": key, "metrics": m}
            thr = round(thr + thr_step, 10)
        used_max_recall = False
    if best is None and min_precision is not None:
        # Fallback: maximize precision, then criterion
        cand = [m for m in grid if (m["depression"]["tp"] + m["depression"]["fp"]) > 0]
        if cand:
            m = max(
                cand,
                key=lambda x: (
                    x["precision"],
                    float(x.get(criterion, x["macro_f1"])),
                    x["accuracy"],
                ),
            )
            best = {"score": float(m.get(criterion, m["macro_f1"])), "metrics": m}
    if best is None:
        # No thr in band predicted ≥1 positive (e.g. all soft PHQ < thr_min).
        # Pick by criterion among the grid without the n_pred>0 constraint —
        # do NOT recurse (that previously infinite-looped when already unconstrained).
        if not grid:
            m = metrics_at_threshold(y_true, scores, 10.0)
        else:
            m = max(
                grid,
                key=lambda x: (
                    float(x.get(criterion, x["macro_f1"])),
                    x["accuracy"],
                    -abs(x["threshold"] - 10.0),
                ),
            )
        best = {"score": float(m.get(criterion, m["macro_f1"])), "metrics": m}
    if not grid:
        grid = [best["metrics"]]
    crit_key = criterion if criterion in grid[0] else "macro_f1"
    return {
        "criterion": criterion,
        "min_precision": min_precision,
        "max_recall": max_recall,
        "max_recall_applied": used_max_recall,
        "require_precision_ge_recall": require_precision_ge_recall,
        "precision_ge_recall_applied": used_p_ge_r,
        "selected_threshold": best["metrics"]["threshold"],
        "selected": best["metrics"],
        "at_thr10": metrics_at_threshold(y_true, scores, 10.0),
        "grid_size": len(grid),
        "grid_top5": sorted(grid, key=lambda x: -x[crit_key])[:5],
    }


@torch.no_grad()
def predict_scores(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, scores = [], []
    for batch in loader:
        batch = {
            k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
            for k, v in batch.items()
        }
        out = model(batch)
        ys.append(batch["y"].cpu().numpy())
        scores.append(out["phq_pred"].float().cpu().numpy())
    return np.concatenate(ys), np.concatenate(scores)


def collect_gate_stats(model, loader, device) -> dict | None:
    """Mean/std ModalityGate weights on a split (None if Gate ablated)."""
    model.eval()
    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = {
                k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
            out = model(batch)
            g = out.get("modality_gates")
            if g is None:
                return None
            chunks.append(g.float().cpu())
    if not chunks:
        return None
    g_all = torch.cat(chunks, dim=0).numpy()  # [N,3]
    mean = g_all.mean(axis=0)
    std = g_all.std(axis=0)
    # Entropy of mean gate (nats); uniform 3-way ≈ 1.099
    p = mean.clip(1e-8, 1.0)
    p = p / p.sum()
    ent = float(-(p * np.log(p)).sum())
    return {
        "n": int(g_all.shape[0]),
        "mean_t": float(mean[0]),
        "mean_a": float(mean[1]),
        "mean_v": float(mean[2]),
        "std_t": float(std[0]),
        "std_a": float(std[1]),
        "std_v": float(std[2]),
        "entropy_of_mean": ent,
        "uniform_entropy": float(np.log(3.0)),
    }


def eval_dev_with_thr_search(
    y_dev: np.ndarray,
    s_dev: np.ndarray,
    *,
    thr_min: float = 8.0,
    thr_max: float = 12.0,
    thr_step: float = 0.1,
    criterion: str = "macro_f1",
    min_precision: float | None = None,
    max_recall: float | None = None,
    require_precision_ge_recall: bool = False,
    prefer_thr: float | None = None,
    prefer_thr_weight: float = 0.0,
) -> dict:
    """DEV-only PHQ soft-score threshold search (anchored clinical band by default).

    Model outputs soft PHQ totals (not probabilities), so thr lives in PHQ space.
    """
    sweep = sweep_threshold(
        y_dev,
        s_dev,
        thr_min=thr_min,
        thr_max=thr_max,
        thr_step=thr_step,
        criterion=criterion,
        min_precision=min_precision,
        max_recall=max_recall,
        require_precision_ge_recall=require_precision_ge_recall,
        prefer_thr=prefer_thr,
        prefer_thr_weight=prefer_thr_weight,
    )
    selected = sweep["selected"]
    return {
        "best_thr": float(sweep["selected_threshold"]),
        "best_dev_f1": float(selected.get(criterion, selected["macro_f1"])),
        "best_macro_f1": float(selected["macro_f1"]),
        "metrics": selected,
        "at_thr10": sweep["at_thr10"],
        "sweep": sweep,
    }


def train_one(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    out_root = Path(args.out_dir) if args.out_dir else OUT
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = out_root / f"seed{args.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

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
    print(
        f"[DeGaMS-ABDE] seed={args.seed} device={device} filter={filter_mode} "
        f"dual_visual={not args.no_dual_visual} "
        f"n_train={len(train_ds)} n_dev={len(dev_ds)} n_test={len(test_ds)}"
    )
    print("[filter train]", train_ds.filter_stats)
    print("[filter dev]", dev_ds.filter_stats)
    print("[filter test]", test_ds.filter_stats)
    if len(train_ds) == 0:
        raise SystemExit("Empty train set — features not ready?")

    y_train = np.array([int(x["y"].item()) for x in train_ds._cache])
    n0 = float((y_train == 0).sum())
    n1 = float((y_train == 1).sum())
    class_counts = (n0, n1)
    print(f"[class balance] n0={n0:.0f} n1={n1:.0f} power={args.class_weight_power}")

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

    vdim = int(train_ds[0]["visual"].size(1))
    lam_sscl = float(getattr(args, "lambda_sscl", 0.02))
    lam_lbcl = float(getattr(args, "lambda_lbcl", 0.004))
    if getattr(args, "ablate_sscl", False):
        lam_sscl = 0.0
    if getattr(args, "ablate_lbcl", False):
        lam_lbcl = 0.0
    cfg = DeGaMSConfig(
        visual_dim=vdim,
        dropout=args.dropout,
        lambda_sscl=lam_sscl,
        lambda_lbcl=lam_lbcl,
        lambda_mse=0.01,
        lambda_kl=0.8,
        lambda_cmcl=0.0 if getattr(args, "ablate_cmcl", False) else float(getattr(args, "lambda_cmcl", 0.05)),
        cmcl_temp=float(getattr(args, "cmcl_temp", 0.07)),
        cmcl_include_av=not bool(getattr(args, "no_cmcl_av", False)),
        max_len=args.max_len,
        class_weight_power=args.class_weight_power,
        neg_emphasis=args.neg_emphasis,
        use_dual_visual=not args.no_dual_visual,
        drop_text=bool(getattr(args, "drop_text", False)),
        drop_audio=bool(getattr(args, "drop_audio", False)),
        drop_visual=bool(getattr(args, "drop_visual", False)),
        ablate_soft_phq=bool(getattr(args, "ablate_soft_phq", False)),
        ablate_msaf=bool(getattr(args, "ablate_msaf", False)),
        ablate_decomp=bool(getattr(args, "ablate_decomp", False)),
        ablate_turn_attn=bool(getattr(args, "ablate_turn_attn", False)),
        ablate_gate=bool(getattr(args, "ablate_gate", False)),
        ablate_cmcl=bool(getattr(args, "ablate_cmcl", False)),
        gate_temp=float(getattr(args, "gate_temp", 1.5)),
    )
    model = DeGaMSModel(cfg).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.9)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_score = -1.0
    best_path = run_dir / "best.pt"
    history = []

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
                loss, parts = degams_total_loss(
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
        m_dev10 = metrics_at_threshold(y_dev, s_dev, 10.0)
        # Precision-oriented ckpt score: DEV thr sweep under min-precision constraint
        if args.ckpt_min_precision is not None and args.ckpt_min_precision > 0:
            sw = sweep_threshold(
                y_dev,
                s_dev,
                thr_min=args.thr_min,
                thr_max=args.thr_max,
                thr_step=args.thr_step,
                criterion=args.thr_criterion,
                min_precision=args.ckpt_min_precision,
            )
            m_sel = sw["selected"]
            sel = float(m_sel.get(args.thr_criterion, m_sel["macro_f1"]))
            # soft bonus for meeting precision floor
            if m_sel["precision"] + 1e-12 >= float(args.ckpt_min_precision):
                sel = sel + 0.05 * float(m_sel["precision"])
            else:
                sel = float(m_sel["precision"]) * 0.5 + 0.25 * float(m_sel["f1"])
        else:
            m_sel = m_dev10
            sel = float(m_dev10["macro_f1"])
        row = {
            "epoch": epoch,
            "loss": total / max(n, 1),
            "dev_macro_f1": m_dev10["macro_f1"],
            "dev_dep_f1": m_dev10["f1"],
            "dev_balanced": m_dev10["balanced_score"],
            "dev_p": m_dev10["precision"],
            "dev_r": m_dev10["recall"],
            "dev_acc": m_dev10["accuracy"],
            "sel_score": sel,
            "sel_thr": float(m_sel.get("threshold", 10.0)),
            "sel_p": float(m_sel["precision"]),
            "sel_r": float(m_sel["recall"]),
            "sel_f1": float(m_sel["f1"]),
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['loss']:.4f} "
            f"devMacroF1@10={m_dev10['macro_f1']:.4f} depF1={m_dev10['f1']:.4f} "
            f"P@10={m_dev10['precision']:.3f} R@10={m_dev10['recall']:.3f} Acc@10={m_dev10['accuracy']:.3f} "
            f"sel={sel:.4f} thr*={row['sel_thr']:.2f} P*={row['sel_p']:.3f} R*={row['sel_r']:.3f}"
        )
        if sel > best_score + 1e-6:
            best_score = sel
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "seed": args.seed,
                    "epoch": epoch,
                    "best_dev_macro_f1": float(m_dev10["macro_f1"]),
                    "best_dev_dep_f1": float(m_dev10["f1"]),
                    "best_sel_score": best_score,
                    "dev_metrics_thr10": m_dev10,
                    "dev_metrics_selected": m_sel,
                    "class_counts": class_counts,
                },
                best_path,
            )

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    y_dev, s_dev = predict_scores(model, dev_loader, device)
    y_te, s_te = predict_scores(model, test_loader, device)

    # A: DEV threshold sweep (optionally precision-constrained)
    final_min_p = args.final_min_precision
    if final_min_p is None and args.ckpt_min_precision:
        final_min_p = 0.68  # default precision target for final thr when precision-oriented
    sweep = sweep_threshold(
        y_dev,
        s_dev,
        thr_min=args.thr_min,
        thr_max=args.thr_max,
        thr_step=args.thr_step,
        criterion=args.thr_criterion,
        min_precision=final_min_p,
    )
    thr_star = float(sweep["selected_threshold"])
    # Precision-band override ONLY when running precision-oriented protocol
    if args.ckpt_min_precision is not None or final_min_p is not None:
        band = []
        t = args.thr_min
        while t <= args.thr_max + 1e-9:
            m = metrics_at_threshold(y_dev, s_dev, t)
            if 0.68 <= m["precision"] <= 0.72 and (m["depression"]["tp"] + m["depression"]["fp"]) > 0:
                band.append(m)
            t = round(t + args.thr_step, 10)
        if band:
            pick = max(band, key=lambda x: (x["f1"], x["accuracy"], -abs(x["threshold"] - 10.0)))
            thr_star = float(pick["threshold"])
            sweep["selected_threshold"] = thr_star
            sweep["selected"] = pick
            sweep["band_pick"] = pick

    # Ablation: unrestricted DEV thr sweep (do NOT use as primary selection)
    sweep_unrestricted = sweep_threshold(
        y_dev,
        s_dev,
        thr_min=4.0,
        thr_max=16.0,
        thr_step=args.thr_step,
        criterion=args.thr_criterion,
        min_precision=None,
    )
    thr_unrestricted = float(sweep_unrestricted["selected_threshold"])

    dev_at_star = metrics_at_threshold(y_dev, s_dev, thr_star)
    dev_at_10 = metrics_at_threshold(y_dev, s_dev, 10.0)
    test_at_star = metrics_at_threshold(y_te, s_te, thr_star)
    test_at_10 = metrics_at_threshold(y_te, s_te, 10.0)
    test_at_unrestricted = metrics_at_threshold(y_te, s_te, thr_unrestricted)
    dev_at_unrestricted = metrics_at_threshold(y_dev, s_dev, thr_unrestricted)

    print("DEV thr*:", thr_star, dev_at_star)
    print("DEV thr=10:", dev_at_10)
    print("TEST thr*:", test_at_star)
    print("TEST thr=10:", test_at_10)
    print(
        "ABLATION unrestricted DEV thr*:",
        thr_unrestricted,
        "TEST:",
        test_at_unrestricted,
    )

    n_vis = sum(1 for x in train_ds._cache if bool(x["present_v"]))
    summary = {
        "protocol": "DeGaMS ABDE · MentalBERT + Mel + dual-visual + hybrid turns + Macro-F1 select + DEV thr sweep",
        "playbook": ["A_thr_sweep", "B_hybrid_turns", "D_dual_visual", "E_macro_f1_mild_weights"],
        "skipped": ["C_sentence_bert"],
        "seed": args.seed,
        "run_dir": str(run_dir),
        "best_ckpt": str(best_path),
        "best_epoch": int(ckpt["epoch"]),
        "best_dev_macro_f1_thr10": float(ckpt["best_dev_macro_f1"]),
        "selection_rule": (
            f"ckpt by constrained {args.thr_criterion} (min_P={args.ckpt_min_precision}); "
            f"final thr min_P={final_min_p} / band [0.68,0.72] if available"
            if args.ckpt_min_precision
            else (
                f"max DEV Macro-F1 at thr=10 during training; then DEV thr sweep by Macro-F1 "
                f"in anchored band [{args.thr_min},{args.thr_max}] step {args.thr_step}"
            )
        ),
        "selected_threshold": thr_star,
        "thr_band": {"min": args.thr_min, "max": args.thr_max, "step": args.thr_step},
        "thr_sweep": {
            "criterion": sweep["criterion"],
            "min_precision": sweep.get("min_precision"),
            "selected_threshold": thr_star,
            "grid_top5": sweep["grid_top5"],
            "anchored_band": [args.thr_min, args.thr_max],
        },
        "thr_sweep_unrestricted_ablation": {
            "note": "ablation only — not used for official selection",
            "selected_threshold": thr_unrestricted,
            "dev": dev_at_unrestricted,
            "test": test_at_unrestricted,
            "grid_top5": sweep_unrestricted["grid_top5"],
        },
        "neg_emphasis": args.neg_emphasis,
        "ckpt_min_precision": args.ckpt_min_precision,
        "final_min_precision": final_min_p,
        "dev_at_selected_thr": dev_at_star,
        "dev_at_thr10": dev_at_10,
        "test_at_selected_thr": test_at_star,
        "test_at_thr10": test_at_10,
        # convenience aliases
        "dev_metrics": dev_at_star,
        "test_metrics": test_at_star,
        "n_train": len(train_ds),
        "n_dev": len(dev_ds),
        "n_test": len(test_ds),
        "n_train_with_visual": n_vis,
        "visual_dim": vdim,
        "visual_source": "dual-head: CLNF Linear(210) vs OpenFace Linear(29); shared temporal backbone",
        "text_source": "existing MentalBERT packs (not Sentence-BERT)",
        "filter_mode": filter_mode,
        "hybrid_target_keep": args.hybrid_target_keep,
        "hybrid_min_frac": args.hybrid_min_frac,
        "class_weight_power": args.class_weight_power,
        "class_counts_train": {"n0": n0, "n1": n1},
        "filter_stats": {
            "train": train_ds.filter_stats,
            "dev": dev_ds.filter_stats,
            "test": test_ds.filter_stats,
        },
        "history": history,
        "args": vars(args),
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print("saved", run_dir / "summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
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
    p.add_argument(
        "--neg-emphasis",
        type=float,
        default=1.0,
        help="Multiply negative-class loss weight (>1 → precision-oriented)",
    )
    p.add_argument(
        "--ckpt-min-precision",
        type=float,
        default=None,
        help="If set, select ckpt by DEV thr-sweep F1/macro-F1 under Precision≥this",
    )
    p.add_argument(
        "--final-min-precision",
        type=float,
        default=None,
        help="If set, final DEV thr must meet Precision≥this (default 0.68 when ckpt-min-precision set)",
    )
    p.add_argument("--no-dual-visual", action="store_true")
    p.add_argument("--drop-text", action="store_true")
    p.add_argument("--drop-audio", action="store_true")
    p.add_argument("--drop-visual", action="store_true")
    p.add_argument("--lambda-sscl", type=float, default=0.02)
    p.add_argument("--lambda-lbcl", type=float, default=0.004)
    p.add_argument("--ablate-sscl", action="store_true")
    p.add_argument("--ablate-lbcl", action="store_true")
    p.add_argument("--ablate-soft-phq", action="store_true")
    p.add_argument("--ablate-msaf", action="store_true")
    p.add_argument("--ablate-decomp", action="store_true")
    p.add_argument("--lambda-cmcl", type=float, default=0.05, help="CMCL weight (0 disables)")
    p.add_argument("--cmcl-temp", type=float, default=0.07)
    p.add_argument("--no-cmcl-av", action="store_true", help="Skip A–V pair in CMCL")
    p.add_argument("--ablate-cmcl", action="store_true", help="Disable CMCL loss")
    p.add_argument("--ablate-gate", action="store_true", help="Equal-sum fusion (no Modality Gate)")
    p.add_argument("--ablate-turn-attn", action="store_true", help="Mean pool over turns")
    p.add_argument("--thr-min", type=float, default=4.0)
    p.add_argument("--thr-max", type=float, default=16.0)
    p.add_argument("--thr-step", type=float, default=0.25)
    p.add_argument(
        "--thr-criterion",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "balanced_score", "f1"],
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out-dir", type=str, default="")
    return p.parse_args()


if __name__ == "__main__":
    from cuda_exit import exit_clean

    train_one(parse_args())
    exit_clean(0)
