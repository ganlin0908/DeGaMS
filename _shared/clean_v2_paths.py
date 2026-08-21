"""Shared constants / paths for DeGaMS cleaned TAV features (v2)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("DEGAMS_ROOT", r"D:\Depression"))
CLEAN_ROOT = ROOT / "egemaps_experiment" / "artifacts" / "degams_clean_v2"

DAIC_SRC_TEXT = ROOT / "egemaps_experiment" / "1_feature_build" / "features" / "daic"
EDAIC_SRC_TEXT = ROOT / "EDAIC" / "extracted_features"
EDAIC_SRC_TEXT_ALT = ROOT / "egemaps_experiment" / "1_feature_build" / "features" / "edaic"

DAIC_WAV = ROOT / "DAICWOZ" / "all_raw_wavs"
EDAIC_RAW = ROOT / "EDAIC" / "raw" / "data"
DAIC_DATA = ROOT / "DAICWOZ" / "data"

# Mel / VAD
TARGET_SR = 16000
N_MELS = 80
MAX_TURN_SEC = 30.0
FRAME_MS = 25
HOP_MS = 10
VAD_REL = 1.5
VAD_ABS_DB = -50.0  # soft absolute floor after session-ish peak; keep mild
PAUSE_MIN_SEC = 0.08
PAUSE_FEAT_DIM = 8
# audio_quality soft weights (no hard delete)
Q_VOICED_FLOOR_SEC = 0.15
Q_SNR_CENTER = 8.0
Q_SNR_SCALE = 3.0

# OpenFace → 210 (29 core + zero pad; DualVisual OF head uses first 29)
OPENFACE_CORE_DIM = 29
VISUAL_DIM = 210
# DenseNet-201 (official E-DAIC) → TRAIN-only PCA-768; concat with OF29 → 797
DENSENET_SOURCE_DIM = 1920
HI_VISUAL_DIM = 768
VISUAL_DENSE_DIM = OPENFACE_CORE_DIM + HI_VISUAL_DIM  # 797
DENSENET_PCA_PATH = (
    ROOT / "EDAIC" / "artifacts" / "densenet201_turn_pca_768.pt"
)
POSE_KEYS = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
GAZE_KEYS = ["gaze_0_x", "gaze_0_y", "gaze_0_z", "gaze_1_x", "gaze_1_y", "gaze_1_z"]
AU_R_KEYS = [
    "AU01_r",
    "AU02_r",
    "AU04_r",
    "AU05_r",
    "AU06_r",
    "AU07_r",
    "AU09_r",
    "AU10_r",
    "AU12_r",
    "AU14_r",
    "AU15_r",
    "AU17_r",
    "AU20_r",
    "AU23_r",
    "AU25_r",
    "AU26_r",
    "AU45_r",
]
assert len(POSE_KEYS) + len(GAZE_KEYS) + len(AU_R_KEYS) == OPENFACE_CORE_DIM
assert VISUAL_DENSE_DIM == 797

PAUSE_FEAT_NAMES = [
    "pause_ratio",
    "speech_ratio",
    "speech_dur",
    "max_pause_sec",
    "mean_pause_sec",
    "n_pauses",
    "snr_db_norm",  # snr/20 clipped to [0,1] scale helper
    "rms_speech_db_norm",
]


def daic_dirs() -> dict[str, Path]:
    base = CLEAN_ROOT / "daic"
    return {
        "text": base / "text",
        "mel": base / "mel",
        "visual": base / "visual",
        "visual_dense": base / "visual_dense",  # OF29 ∥ DenseNet-PCA768
        "wav2vec": base / "wav2vec",
        "wav2vec_enh": base / "wav2vec_enh",  # DeepFilterNet + Silero soft-gate
        "base": base,
    }


def edaic_dirs() -> dict[str, Path]:
    base = CLEAN_ROOT / "edaic"
    return {
        "text": base / "text",
        "mel": base / "mel",
        "visual": base / "visual",
        "visual_dense": base / "visual_dense",
        "wav2vec": base / "wav2vec",
        "wav2vec_enh": base / "wav2vec_enh",
        "base": base,
    }


def ensure_clean_tree() -> None:
    for d in (*daic_dirs().values(), *edaic_dirs().values()):
        d.mkdir(parents=True, exist_ok=True)
