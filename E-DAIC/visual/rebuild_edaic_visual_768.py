"""
Replace E-DAIC OpenFace [T, 31] features with real DenseNet-201 [T, 768] features.

The E-DAIC release contains one 1920-D DenseNet-201 embedding per second in
raw/data/{id}_P/features/{id}_densenet201.csv. This script:

1. mean-pools those frame embeddings inside the exact turn windows already
   stored in extracted_features/{id}_features.pt;
2. fits a 768-D PCA projection on TRAIN turns only (no DEV/TEST leakage);
3. L2-normalizes each projected turn and atomically overwrites only the visual
   field in every existing feature pack.

Text, audio, labels, turn ordering, and metadata are preserved.

Run:
  python E-DAIC/visual/rebuild_edaic_visual_768.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(os.environ.get("DEGAMS_ROOT", r"/path/to/degams_data"))
EDAIC = ROOT / "EDAIC"
RAW = EDAIC / "raw" / "data"
FEATURES = EDAIC / "extracted_features"
LABELS = EDAIC / "label"
ARTIFACT_DIR = EDAIC / "artifacts"
PROJECTION_PATH = ARTIFACT_DIR / "densenet201_turn_pca_768.pt"
REPORT_PATH = EDAIC / "VISUAL_768_REBUILD_REPORT.json"

SOURCE_DIM = 1920
TARGET_DIM = 768
FRAME_PAD_SECONDS = 0.05
RANDOM_SEED = 42


def read_split_ids(split: str) -> List[str]:
    path = LABELS / f"{split}_split.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [str(int(row["Participant_ID"])) for row in csv.DictReader(handle)]


def all_split_ids() -> Tuple[List[str], Dict[str, str]]:
    ordered: List[str] = []
    split_by_id: Dict[str, str] = {}
    for split in ("train", "dev", "test"):
        for sid in read_split_ids(split):
            if sid in split_by_id:
                raise ValueError(f"Participant {sid} occurs in multiple E-DAIC splits")
            split_by_id[sid] = split
            ordered.append(sid)
    return ordered, split_by_id


def densenet_path(sid: str) -> Path:
    direct = RAW / f"{sid}_P" / "features" / f"{sid}_densenet201.csv"
    if direct.is_file():
        return direct
    matches = list((RAW / f"{sid}_P").glob("**/*densenet201.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one DenseNet-201 CSV for {sid}, found {len(matches)}"
        )
    return matches[0]


def load_densenet_frames(sid: str) -> Tuple[np.ndarray, np.ndarray]:
    path = densenet_path(sid)
    frame = pd.read_csv(
        path,
        usecols=lambda name: name == "timeStamp" or name.startswith("neuron_"),
        dtype=np.float32,
    )
    if "timeStamp" not in frame:
        raise KeyError(f"{path} has no timeStamp column")
    feature_columns = [f"neuron_{i}" for i in range(SOURCE_DIM)]
    missing = [name for name in feature_columns if name not in frame]
    if missing:
        raise ValueError(f"{path} is missing DenseNet columns: {missing[:5]}")
    timestamps = frame["timeStamp"].to_numpy(dtype=np.float64, copy=True)
    values = frame[feature_columns].to_numpy(dtype=np.float32, copy=True)
    if values.ndim != 2 or values.shape[1] != SOURCE_DIM or values.shape[0] == 0:
        raise ValueError(f"Invalid DenseNet matrix for {sid}: {values.shape}")
    if not np.isfinite(timestamps).all():
        raise ValueError(f"Non-finite timestamps in {path}")
    nonfinite = ~np.isfinite(values)
    if nonfinite.any():
        values[nonfinite] = 0.0
    order = np.argsort(timestamps, kind="stable")
    return timestamps[order], values[order]


def load_pack(sid: str) -> Tuple[Path, dict]:
    path = FEATURES / f"{sid}_features.pt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing E-DAIC feature pack: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return path, payload


def pool_frames_to_turns(
    timestamps: np.ndarray,
    frame_features: np.ndarray,
    metadata: Sequence[dict],
) -> Tuple[np.ndarray, int]:
    pooled = np.empty((len(metadata), SOURCE_DIM), dtype=np.float32)
    fallback_count = 0
    for index, turn in enumerate(metadata):
        start = float(turn["start_time"])
        end = float(turn["end_time"])
        if end < start:
            start, end = end, start
        left = int(np.searchsorted(timestamps, start - FRAME_PAD_SECONDS, side="left"))
        right = int(np.searchsorted(timestamps, end + FRAME_PAD_SECONDS, side="right"))
        if right > left:
            pooled[index] = frame_features[left:right].mean(axis=0, dtype=np.float32)
        else:
            midpoint = 0.5 * (start + end)
            nearest = int(np.argmin(np.abs(timestamps - midpoint)))
            pooled[index] = frame_features[nearest]
            fallback_count += 1
    return pooled, fallback_count


def pool_subject(sid: str) -> Tuple[np.ndarray, int]:
    _, payload = load_pack(sid)
    metadata = list(payload.get("meta_data") or [])
    text = payload["text_features"]
    audio = payload["audio_features"]
    if not metadata:
        raise ValueError(f"{sid}: feature pack has no turn metadata")
    if len(metadata) != int(text.shape[0]) or len(metadata) != int(audio.shape[0]):
        raise ValueError(
            f"{sid}: alignment mismatch meta={len(metadata)}, "
            f"text={tuple(text.shape)}, audio={tuple(audio.shape)}"
        )
    timestamps, frames = load_densenet_frames(sid)
    return pool_frames_to_turns(timestamps, frames, metadata)


def fit_train_projection(
    train_ids: Sequence[str],
    device: torch.device,
    niter: int,
) -> Tuple[dict, Dict[str, np.ndarray], Dict[str, int]]:
    pooled_by_id: Dict[str, np.ndarray] = {}
    fallback_by_id: Dict[str, int] = {}
    matrices: List[np.ndarray] = []
    for sid in tqdm(train_ids, desc="Pool TRAIN DenseNet turns", unit="subject"):
        pooled, fallbacks = pool_subject(sid)
        pooled_by_id[sid] = pooled
        fallback_by_id[sid] = fallbacks
        matrices.append(pooled)

    train = np.concatenate(matrices, axis=0)
    if train.shape[0] < TARGET_DIM:
        raise ValueError(
            f"Need at least {TARGET_DIM} TRAIN turns for PCA, found {train.shape[0]}"
        )
    train_tensor = torch.from_numpy(train).to(device)
    mean = train_tensor.mean(dim=0)
    centered = train_tensor - mean

    torch.manual_seed(RANDOM_SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(RANDOM_SEED)
    print(
        f"Fitting PCA on {tuple(centered.shape)} TRAIN turns using {device} "
        f"(q={TARGET_DIM}, niter={niter})",
        flush=True,
    )
    _, singular_values, components = torch.pca_lowrank(
        centered,
        q=TARGET_DIM,
        center=False,
        niter=niter,
    )
    explained_variance = singular_values.square() / max(train.shape[0] - 1, 1)
    total_variance = centered.var(dim=0, unbiased=True).sum()
    explained_ratio = explained_variance / total_variance.clamp_min(1e-12)

    artifact = {
        "source": "official E-DAIC per-second DenseNet-201 embeddings",
        "source_dim": SOURCE_DIM,
        "target_dim": TARGET_DIM,
        "fit_split": "train",
        "fit_subject_ids": list(train_ids),
        "fit_turn_count": int(train.shape[0]),
        "random_seed": RANDOM_SEED,
        "pca_niter": int(niter),
        "mean": mean.detach().cpu().float(),
        "components": components.detach().cpu().float(),
        "explained_variance": explained_variance.detach().cpu().float(),
        "explained_variance_ratio": explained_ratio.detach().cpu().float(),
        "explained_variance_ratio_sum": float(explained_ratio.sum().item()),
        "output_normalization": "L2 per turn",
        "created_unix": time.time(),
    }
    return artifact, pooled_by_id, fallback_by_id


def load_projection() -> dict:
    if not PROJECTION_PATH.is_file():
        raise FileNotFoundError(
            f"Projection artifact not found: {PROJECTION_PATH}. Run without "
            "--transform-only first."
        )
    artifact = torch.load(PROJECTION_PATH, map_location="cpu", weights_only=False)
    if tuple(artifact["components"].shape) != (SOURCE_DIM, TARGET_DIM):
        raise ValueError(
            f"Unexpected PCA components shape: {tuple(artifact['components'].shape)}"
        )
    return artifact


def project_turns(pooled: np.ndarray, artifact: dict) -> torch.Tensor:
    values = torch.from_numpy(pooled).float()
    projected = (values - artifact["mean"]) @ artifact["components"]
    projected = F.normalize(projected, p=2.0, dim=-1, eps=1e-12)
    if tuple(projected.shape) != (pooled.shape[0], TARGET_DIM):
        raise AssertionError(f"Unexpected projected shape: {tuple(projected.shape)}")
    if not torch.isfinite(projected).all():
        raise ValueError("Projected visual features contain non-finite values")
    return projected.contiguous()


def atomic_save(payload: dict, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".visual768.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def overwrite_subject(
    sid: str,
    split: str,
    artifact: dict,
    pooled: np.ndarray | None = None,
    fallback_count: int | None = None,
) -> dict:
    path, payload = load_pack(sid)
    if pooled is None or fallback_count is None:
        pooled, fallback_count = pool_subject(sid)
    visual = project_turns(pooled, artifact)
    if visual.shape[0] != payload["text_features"].shape[0]:
        raise ValueError(
            f"{sid}: projected T={visual.shape[0]} != text T="
            f"{payload['text_features'].shape[0]}"
        )

    payload["visual_features"] = visual
    payload["visual_dim"] = TARGET_DIM
    payload["visual_source"] = (
        "official E-DAIC DenseNet-201 1920-D per-second embeddings; "
        "turn mean-pool; TRAIN-only PCA-768; per-turn L2 normalization"
    )
    payload["visual_source_dim"] = SOURCE_DIM
    payload["visual_projection"] = str(PROJECTION_PATH)
    payload["visual_projection_fit_split"] = "train"
    payload["visual_frame_fallback_turns"] = int(fallback_count)
    payload.pop("visual_columns", None)
    atomic_save(payload, path)
    return {
        "participant_id": sid,
        "split": split,
        "turns": int(visual.shape[0]),
        "nearest_frame_fallback_turns": int(fallback_count),
        "mean_l2_norm": float(visual.norm(dim=1).mean().item()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild E-DAIC visual features as DenseNet-PCA 768-D turns"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fit-only",
        action="store_true",
        help="fit/save TRAIN-only PCA but do not overwrite feature packs",
    )
    mode.add_argument(
        "--transform-only",
        action="store_true",
        help="reuse the saved PCA and overwrite feature packs",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="PCA device; auto uses CUDA when available",
    )
    parser.add_argument(
        "--pca-niter",
        type=int,
        default=4,
        help="randomized PCA subspace iterations",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="transform only the first N subjects (testing only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pca_niter < 1:
        raise ValueError("--pca-niter must be >= 1")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")

    ids, split_by_id = all_split_ids()
    train_ids = [sid for sid in ids if split_by_id[sid] == "train"]
    missing_raw = [sid for sid in ids if not densenet_path(sid).is_file()]
    missing_packs = [sid for sid in ids if not (FEATURES / f"{sid}_features.pt").is_file()]
    if missing_raw or missing_packs:
        raise FileNotFoundError(
            f"Cannot rebuild all E-DAIC subjects: missing_raw={missing_raw[:10]}, "
            f"missing_packs={missing_packs[:10]}"
        )

    pooled_train: Dict[str, np.ndarray] = {}
    fallback_train: Dict[str, int] = {}
    if args.transform_only:
        artifact = load_projection()
    else:
        artifact, pooled_train, fallback_train = fit_train_projection(
            train_ids, device=device, niter=args.pca_niter
        )
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, PROJECTION_PATH)
        print(
            f"Saved {PROJECTION_PATH} | explained variance="
            f"{artifact['explained_variance_ratio_sum']:.6f}",
            flush=True,
        )

    if args.fit_only:
        return 0

    selected = ids[: args.limit] if args.limit > 0 else ids
    rows = []
    for sid in tqdm(selected, desc="Overwrite E-DAIC visual features", unit="subject"):
        rows.append(
            overwrite_subject(
                sid,
                split_by_id[sid],
                artifact,
                pooled=pooled_train.get(sid),
                fallback_count=fallback_train.get(sid),
            )
        )

    report = {
        "status": "complete" if len(selected) == len(ids) else "partial",
        "source_dim": SOURCE_DIM,
        "target_dim": TARGET_DIM,
        "projection_path": str(PROJECTION_PATH),
        "fit_split": "train",
        "fit_turn_count": int(artifact["fit_turn_count"]),
        "explained_variance_ratio_sum": float(
            artifact["explained_variance_ratio_sum"]
        ),
        "subjects_transformed": len(rows),
        "turns_transformed": sum(row["turns"] for row in rows),
        "nearest_frame_fallback_turns": sum(
            row["nearest_frame_fallback_turns"] for row in rows
        ),
        "subjects": rows,
        "completed_unix": time.time(),
    }
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Done: subjects={report['subjects_transformed']} "
        f"turns={report['turns_transformed']} target_dim={TARGET_DIM} "
        f"report={REPORT_PATH}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
