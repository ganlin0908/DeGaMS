"""Build dense visual packs: OpenFace-29 ∥ DenseNet-PCA-768 → [T, 797].

Also writes per-turn ``visual_turn_mask`` (False when OF+DN both empty) and
optional short-gap linear fill for isolated missing turns (default max gap=2).

No raw video required — uses official E-DAIC DenseNet-201 CSVs (covers all
DAIC clean_v2 sessions via shared IDs) + existing OpenFace-210 packs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

CODE_FEAT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DEGAMS_ROOT", r"D:\Depression"))
sys.path.insert(0, str(CODE_FEAT))

from clean_v2_paths import (  # noqa: E402
    DENSENET_PCA_PATH,
    DENSENET_SOURCE_DIM,
    HI_VISUAL_DIM,
    OPENFACE_CORE_DIM,
    VISUAL_DENSE_DIM,
    daic_dirs,
    edaic_dirs,
    ensure_clean_tree,
)
from rebuild_edaic_visual_768 import (  # noqa: E402
    load_densenet_frames,
    load_projection,
    project_turns,
)

FRAME_PAD_SECONDS = 0.05


def pool_densenet_turns(
    timestamps: np.ndarray,
    frames: np.ndarray,
    meta: Sequence[dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean-pool DenseNet frames inside each turn window.

    Returns (pooled [T,1920], n_frames [T]). Empty window → zeros + n=0
    (strict; no nearest-frame hallucination).
    """
    t = len(meta)
    pooled = np.zeros((t, DENSENET_SOURCE_DIM), dtype=np.float32)
    n_frames = np.zeros(t, dtype=np.int32)
    for i, turn in enumerate(meta):
        start = float(turn.get("start_time") or 0.0)
        end = float(turn["end_time"]) if turn.get("end_time") is not None else start + 0.1
        if end < start:
            start, end = end, start
        left = int(np.searchsorted(timestamps, start - FRAME_PAD_SECONDS, side="left"))
        right = int(np.searchsorted(timestamps, end + FRAME_PAD_SECONDS, side="right"))
        if right > left:
            pooled[i] = frames[left:right].mean(axis=0, dtype=np.float32)
            n_frames[i] = int(right - left)
    return pooled, n_frames


def short_gap_linear_fill(
    feat: np.ndarray,
    valid: np.ndarray,
    *,
    max_gap: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear-interpolate isolated invalid turns when gap ≤ max_gap.

    Does not invent long missing stretches. Returns filled feat + updated valid.
    """
    if max_gap <= 0 or feat.size == 0:
        return feat, valid
    out = feat.copy()
    v = valid.astype(bool).copy()
    n = len(v)
    i = 0
    while i < n:
        if v[i]:
            i += 1
            continue
        j = i
        while j < n and not v[j]:
            j += 1
        gap = j - i
        left = i - 1
        right = j if j < n else -1
        if gap <= max_gap and left >= 0 and right >= 0 and v[left] and v[right]:
            for k in range(i, j):
                alpha = (k - left) / float(right - left)
                out[k] = (1.0 - alpha) * out[left] + alpha * out[right]
                v[k] = True
        i = j
    return out, v


def load_openface_pack(path: Path) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    vp = np.load(path, allow_pickle=True)
    visual = np.asarray(vp["visual"], dtype=np.float32)
    of = visual[:, :OPENFACE_CORE_DIM].copy()
    if "n_frames_per_turn" in vp.files:
        nf = np.asarray(vp["n_frames_per_turn"], dtype=np.int32)
    else:
        nf = (np.abs(of).sum(axis=1) > 0).astype(np.int32)
    present = bool(vp["present"]) if "present" in vp.files else True
    src = "openface"
    if "source" in vp.files:
        src = str(
            np.asarray(vp["source"]).item()
            if np.asarray(vp["source"]).shape == ()
            else vp["source"]
        )
        if src.startswith("error") or src == "missing":
            present = False
    return of, nf, present, src


def process_sid(
    corpus: str,
    sid: str,
    text_path: Path,
    of_path: Path,
    artifact: dict,
    *,
    max_gap: int,
) -> dict:
    pack = torch.load(text_path, map_location="cpu", weights_only=False)
    meta = list(pack.get("meta_data") or [])
    of, nf_of, present_of, of_src = load_openface_pack(of_path)
    t = len(meta)
    if of.shape[0] != t:
        t = min(of.shape[0], t)
        of, nf_of, meta = of[:t], nf_of[:t], meta[:t]

    hi = np.zeros((t, HI_VISUAL_DIM), dtype=np.float32)
    nf_dn = np.zeros(t, dtype=np.int32)
    dn_ok = False
    dn_err = ""
    try:
        ts, frames = load_densenet_frames(sid)
        pooled, nf_dn = pool_densenet_turns(ts, frames, meta)
        # project only turns with frames; keep zeros elsewhere
        has_dn = nf_dn > 0
        if has_dn.any():
            proj = project_turns(pooled[has_dn], artifact).numpy()
            hi[has_dn] = proj
            dn_ok = True
    except Exception as e:
        dn_err = str(e)

    of_valid = (nf_of > 0) & (np.abs(of).sum(axis=1) > 0)
    dn_valid = nf_dn > 0
    if max_gap > 0:
        of, of_valid = short_gap_linear_fill(of, of_valid, max_gap=max_gap)
        hi, dn_valid = short_gap_linear_fill(hi, dn_valid, max_gap=max_gap)

    turn_mask = of_valid | dn_valid
    # zero invalid branches so concat cannot leak sparse zeros into the other stream
    of = of * of_valid[:, None].astype(np.float32)
    hi = hi * dn_valid[:, None].astype(np.float32)
    # L2-normalize OF core per valid turn (DenseNet already L2'd by PCA path)
    norms = np.linalg.norm(of, axis=1, keepdims=True)
    of = np.where(norms > 1e-6, of / np.clip(norms, 1e-6, None), of).astype(np.float32)

    visual = np.concatenate([of, hi], axis=1).astype(np.float32)
    assert visual.shape[1] == VISUAL_DENSE_DIM
    present = bool(present_of or dn_ok) and bool(turn_mask.any())

    return {
        "npz": {
            "visual": visual,
            "of_core": of,
            "hi_visual": hi,
            "visual_turn_mask": turn_mask.astype(np.bool_),
            "of_valid": of_valid.astype(np.bool_),
            "dn_valid": dn_valid.astype(np.bool_),
            "n_frames_of": nf_of.astype(np.int32),
            "n_frames_dn": nf_dn.astype(np.int32),
            "visual_dim": np.int32(VISUAL_DENSE_DIM),
            "openface_core_dim": np.int32(OPENFACE_CORE_DIM),
            "hi_visual_dim": np.int32(HI_VISUAL_DIM),
            "present": np.bool_(present),
            "source": np.asarray(f"of29+densenet_pca768|{of_src}"),
            "session_id": np.asarray(sid),
            "n_turns": np.int32(t),
            "layout": np.asarray(
                "visual=[OF29 | DenseNet-PCA768]; mask=of_valid|dn_valid; short-gap fill"
            ),
        },
        "meta": {
            "session_id": sid,
            "corpus": corpus,
            "present": present,
            "densenet_ok": dn_ok,
            "densenet_error": dn_err,
            "openface_source": of_src,
            "visual_dim": VISUAL_DENSE_DIM,
            "n_turns": t,
            "frac_turn_mask": float(turn_mask.mean()) if t else 0.0,
            "frac_of_valid": float(of_valid.mean()) if t else 0.0,
            "frac_dn_valid": float(dn_valid.mean()) if t else 0.0,
            "max_gap_fill": max_gap,
            "pca_path": str(DENSENET_PCA_PATH),
            "extracted_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


def run_corpus(corpus: str, *, overwrite: bool, max_gap: int, limit: int) -> None:
    ensure_clean_tree()
    dirs = daic_dirs() if corpus == "daic" else edaic_dirs()
    of_dir: Path = dirs["visual"]
    text_dir: Path = dirs["text"]
    out_dir: Path = dirs["visual_dense"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if not DENSENET_PCA_PATH.is_file():
        raise FileNotFoundError(
            f"Missing DenseNet PCA artifact: {DENSENET_PCA_PATH}. "
            "Run EDAIC/rebuild_edaic_visual_768.py first."
        )
    artifact = load_projection()
    packs = sorted(text_dir.glob("*_features.pt"))
    if limit > 0:
        packs = packs[:limit]

    stats = {
        "ok": 0,
        "present": 0,
        "missing": 0,
        "densenet_ok": 0,
        "mean_frac_mask": [],
    }
    for tp in tqdm(packs, desc=f"{corpus}-visual-dense"):
        sid = tp.name[: -len("_features.pt")]
        of_path = of_dir / f"{sid}.npz"
        dest = out_dir / f"{sid}.npz"
        meta_p = out_dir / f"{sid}.meta.json"
        if dest.is_file() and meta_p.is_file() and not overwrite:
            stats["ok"] += 1
            continue
        if not of_path.is_file():
            stats["missing"] += 1
            continue
        result = process_sid(
            corpus, sid, tp, of_path, artifact, max_gap=max_gap
        )
        np.savez_compressed(dest, **result["npz"])
        meta_p.write_text(json.dumps(result["meta"], indent=2), encoding="utf-8")
        stats["ok"] += 1
        if result["meta"]["present"]:
            stats["present"] += 1
            stats["mean_frac_mask"].append(result["meta"]["frac_turn_mask"])
        else:
            stats["missing"] += 1
        if result["meta"]["densenet_ok"]:
            stats["densenet_ok"] += 1

    summary = {
        "corpus": corpus,
        "ok": stats["ok"],
        "present": stats["present"],
        "missing_or_absent": stats["missing"],
        "densenet_ok": stats["densenet_ok"],
        "mean_frac_turn_mask": float(np.mean(stats["mean_frac_mask"]))
        if stats["mean_frac_mask"]
        else 0.0,
        "out_dir": str(out_dir),
        "visual_dim": VISUAL_DENSE_DIM,
        "max_gap_fill": max_gap,
    }
    (dirs["base"] / "_visual_dense_extract_stats.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["daic", "edaic", "both"], default="both")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--max-gap-fill",
        type=int,
        default=2,
        help="Linear-fill isolated missing turns up to this gap (0=strict mask only)",
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    corpora = ["daic", "edaic"] if args.corpus == "both" else [args.corpus]
    for c in corpora:
        run_corpus(
            c,
            overwrite=args.overwrite,
            max_gap=int(args.max_gap_fill),
            limit=int(args.limit),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
