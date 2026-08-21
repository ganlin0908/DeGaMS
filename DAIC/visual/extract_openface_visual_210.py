"""Unified OpenFace Pose+Gaze+AU_r → [T,210] (29 core + zero-pad) for DAIC & E-DAIC."""
from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from clean_v2_paths import (
    AU_R_KEYS,
    DAIC_DATA,
    EDAIC_RAW,
    GAZE_KEYS,
    OPENFACE_CORE_DIM,
    POSE_KEYS,
    VISUAL_DIM,
    daic_dirs,
    edaic_dirs,
    ensure_clean_tree,
)


def _f(x: str) -> float:
    x = (x or "").strip()
    if not x:
        return 0.0
    xl = x.lower()
    if "ind" in xl or "nan" in xl or "inf" in xl or "#" in xl:
        return 0.0
    try:
        return float(x)
    except ValueError:
        return 0.0


def parse_openface_csv(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            rows.append({(k or "").strip(): v for k, v in r.items()})
    if not rows:
        raise ValueError(f"empty OpenFace: {path}")
    n = len(rows)
    ts = np.zeros(n, dtype=np.float64)
    success = np.zeros(n, dtype=np.int32)
    core = np.zeros((n, OPENFACE_CORE_DIM), dtype=np.float32)
    keys = POSE_KEYS + GAZE_KEYS + AU_R_KEYS
    for i, r in enumerate(rows):
        ts[i] = _f(r.get("timestamp") or r.get("Timestamp") or "0")
        success[i] = int(_f(r.get("success") or "0"))
        for j, k in enumerate(keys):
            core[i, j] = _f(r.get(k) or "0")
    feat = np.zeros((n, VISUAL_DIM), dtype=np.float32)
    feat[:, :OPENFACE_CORE_DIM] = core
    return ts, success, feat


def parse_openface_from_tar(tpath: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with tarfile.open(tpath, "r:gz") as tf:
        cands = [
            n
            for n in tf.getnames()
            if "OpenFace" in n
            and "Pose" in n
            and "BoVW" not in n
            and n.lower().endswith(".csv")
        ]
        if not cands:
            raise FileNotFoundError(f"no OpenFace CSV in {tpath.name}")
        name = sorted(cands)[0]
        raw = tf.extractfile(name).read()
    # write temp parse via StringIO
    text = raw.decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(text))
    rows = [{(k or "").strip(): v for k, v in r.items()} for r in reader]
    if not rows:
        raise ValueError(f"empty OpenFace in {tpath.name}")
    n = len(rows)
    ts = np.zeros(n, dtype=np.float64)
    success = np.zeros(n, dtype=np.int32)
    core = np.zeros((n, OPENFACE_CORE_DIM), dtype=np.float32)
    keys = POSE_KEYS + GAZE_KEYS + AU_R_KEYS
    for i, r in enumerate(rows):
        ts[i] = _f(r.get("timestamp") or "0")
        success[i] = int(_f(r.get("success") or "0"))
        for j, k in enumerate(keys):
            core[i, j] = _f(r.get(k) or "0")
    feat = np.zeros((n, VISUAL_DIM), dtype=np.float32)
    feat[:, :OPENFACE_CORE_DIM] = core
    return ts, success, feat


def resolve_openface(corpus: str, sid: str) -> Tuple[Optional[object], str]:
    """Return (loader_result_or_path, source_tag). Prefer E-DAIC CSV for both."""
    edaic_csv = EDAIC_RAW / f"{sid}_P" / "features" / f"{sid}_OpenFace2.1.0_Pose_gaze_AUs.csv"
    if edaic_csv.is_file():
        return edaic_csv, "edaic_openface_csv"
    if (EDAIC_RAW / f"{sid}_P").is_dir():
        hits = list((EDAIC_RAW / f"{sid}_P").rglob("*OpenFace*Pose*AUs.csv"))
        hits = [h for h in hits if "BoVW" not in h.name]
        if hits:
            return hits[0], "edaic_openface_csv_glob"
    if corpus == "daic":
        tar = DAIC_DATA / f"{sid}_P.tar.gz"
        if tar.is_file():
            return tar, "daic_tar_openface"
    return None, "missing"


def turn_mean_pool(
    ts: np.ndarray,
    success: np.ndarray,
    feat: np.ndarray,
    start: float,
    end: float,
) -> Tuple[np.ndarray, int]:
    if end < start:
        start, end = end, start
    if end - start < 1e-3:
        end = start + 0.05
    m = (ts >= start) & (ts <= end) & (success == 1)
    idx = np.where(m)[0]
    if idx.size == 0:
        m2 = (ts >= start) & (ts <= end)
        idx = np.where(m2)[0]
    if idx.size == 0:
        return np.zeros(VISUAL_DIM, dtype=np.float32), 0
    return feat[idx].mean(axis=0).astype(np.float32), int(idx.size)


def load_of(src, tag: str):
    if tag.startswith("edaic_openface"):
        return parse_openface_csv(Path(src))
    if tag == "daic_tar_openface":
        return parse_openface_from_tar(Path(src))
    raise ValueError(tag)


def process_sid(corpus: str, sid: str, text_path: Path) -> dict:
    pack = torch.load(text_path, map_location="cpu", weights_only=False)
    meta = list(pack.get("meta_data") or [])
    src, tag = resolve_openface(corpus, sid)
    if src is None:
        visual = np.zeros((len(meta), VISUAL_DIM), dtype=np.float32)
        n_frames = [0] * len(meta)
        present = False
    else:
        try:
            ts, success, feat = load_of(src, tag)
            present = True
        except Exception as e:
            visual = np.zeros((len(meta), VISUAL_DIM), dtype=np.float32)
            n_frames = [0] * len(meta)
            present = False
            tag = f"error:{e}"
            ts = success = feat = None
        if present and ts is not None:
            vecs = []
            n_frames = []
            for turn in meta:
                t0 = float(turn.get("start_time") or 0.0)
                t1 = float(turn["end_time"]) if turn.get("end_time") is not None else t0 + 0.1
                v, nf = turn_mean_pool(ts, success, feat, t0, t1)
                vecs.append(v)
                n_frames.append(nf)
            visual = np.stack(vecs, axis=0)

    return {
        "npz": {
            "visual": visual.astype(np.float32),
            "turn_mask": np.ones(visual.shape[0], dtype=np.bool_),
            "n_frames_per_turn": np.asarray(n_frames, dtype=np.int32),
            "visual_dim": np.int32(VISUAL_DIM),
            "openface_core_dim": np.int32(OPENFACE_CORE_DIM),
            "present": np.bool_(present),
            "source": np.asarray(tag),
            "session_id": np.asarray(sid),
            "n_turns": np.int32(visual.shape[0]),
            "layout": np.asarray(
                "OpenFace: [6 pose | 6 gaze | 17 AU_r | zeros→210]; DualVisual uses first 29"
            ),
        },
        "meta": {
            "session_id": sid,
            "corpus": corpus,
            "source": tag,
            "present": bool(present),
            "visual_dim": VISUAL_DIM,
            "openface_core_dim": OPENFACE_CORE_DIM,
            "n_turns": int(visual.shape[0]),
            "mean_frames_per_turn": float(np.mean(n_frames)) if n_frames else 0.0,
            "frac_turns_with_frames": float(np.mean([1 if x > 0 else 0 for x in n_frames]))
            if n_frames
            else 0.0,
            "extracted_at": datetime.now().isoformat(timespec="seconds"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["daic", "edaic", "both"], default="both")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="")
    args = ap.parse_args()
    ensure_clean_tree()

    jobs = []
    if args.corpus in ("daic", "both"):
        jobs.append(("daic", daic_dirs()))
    if args.corpus in ("edaic", "both"):
        jobs.append(("edaic", edaic_dirs()))

    for corpus, dirs in jobs:
        packs = sorted(dirs["text"].glob("*_features.pt"))
        if args.shard:
            i, n = map(int, args.shard.split("/"))
            packs = [p for k, p in enumerate(packs) if k % n == i]
        if args.limit > 0:
            packs = packs[: args.limit]
        out = dirs["visual"]
        stats = {"ok": 0, "present": 0, "missing": 0}
        for p in tqdm(packs, desc=f"{corpus}-openface-210"):
            sid = p.name[: -len("_features.pt")]
            dest = out / f"{sid}.npz"
            meta_p = out / f"{sid}.meta.json"
            if dest.is_file() and meta_p.is_file() and not args.overwrite:
                stats["ok"] += 1
                continue
            result = process_sid(corpus, sid, p)
            np.savez_compressed(dest, **result["npz"])
            meta_p.write_text(json.dumps(result["meta"], indent=2), encoding="utf-8")
            stats["ok"] += 1
            if result["meta"]["present"]:
                stats["present"] += 1
            else:
                stats["missing"] += 1
        (dirs["base"] / f"_openface_extract_stats.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        print(f"{corpus} OpenFace210 {stats} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
