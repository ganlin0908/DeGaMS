"""
E-DAIC feature extraction -> extracted_features/{id}_features.pt

Output schema (aligned with DAIC-WOZ + visual extension):
  text_features:   FloatTensor [T, 768]   MentalBERT CLS (same as DAIC)
  audio_features:  FloatTensor [T, 88]    openSMILE eGeMAPSv02 functionals
  visual_features: FloatTensor [T, 768]   DenseNet-201 + TRAIN-only PCA
  meta_data:       list[dict]             segment timing / text

Strategy:
  - If DAIC-WOZ {id}_features.pt exists (188 overlap): reuse text/audio, only compute visual
  - Else: full MentalBERT + eGeMAPS + DenseNet/PCA aligned to Transcript rows

Run:
  python E-DAIC/Text/extract_edaic_features.py
  python E-DAIC/Text/extract_edaic_features.py --limit 3
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from rebuild_edaic_visual_768 import (
    PROJECTION_PATH,
    TARGET_DIM,
    load_densenet_frames,
    load_projection,
    pool_frames_to_turns,
    project_turns,
)

ROOT = Path(os.environ.get("DEGAMS_ROOT", r"D:\Depression"))
EDAIC_RAW = ROOT / "EDAIC" / "raw" / "data"
EDAIC_OUT = ROOT / "EDAIC" / "extracted_features"
DAIC_FEAT = ROOT / "DAICWOZ" / "extracted_features"
LOG = ROOT / "EDAIC" / "logs" / "extract_edaic.log"

TEXT_MODEL = "mental/mental-bert-base-uncased"
TEXT_FALLBACK = "bert-base-uncased"
AUDIO_DIM = 88
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
# Prefer local HF cache snapshot (MentalBERT is gated).
MENTALBERT_LOCAL = Path(
    r"C:\Users\Administrator\.cache\huggingface\hub"
    r"\models--mental--mental-bert-base-uncased\snapshots"
    r"\24809aa822c76639760d0d934742d1b42f89942f"
)
MAX_UTT_DUR_SEC = 60.0  # drop broken E-DAIC ASR rows spanning almost whole file

POSE_COLS = ["pose_Tx", "pose_Ty", "pose_Tz", "pose_Rx", "pose_Ry", "pose_Rz"]
GAZE_COLS = [
    "gaze_0_x",
    "gaze_0_y",
    "gaze_0_z",
    "gaze_1_x",
    "gaze_1_y",
    "gaze_1_z",
    "gaze_angle_x",
    "gaze_angle_y",
]
AU_R_COLS = [
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
VISUAL_COLS = POSE_COLS + GAZE_COLS + AU_R_COLS  # 31-D
VISUAL_DIM = TARGET_DIM



raise RuntimeError(
    "openSMILE/eGeMAPS 88-d extraction is DISABLED. "
    "Audio must come from Wav2Vec2 (egemaps_experiment/4_main_train/extract_wav2vec_session_embeds.py). Do not reintroduce eGeMAPS into feature packs."
)


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def list_subjects() -> List[str]:
    return sorted(
        [p.name.replace("_P", "") for p in EDAIC_RAW.glob("*_P") if p.is_dir()],
        key=lambda x: int(x) if x.isdigit() else x,
    )


def read_transcript(sid: str) -> List[dict]:
    path = EDAIC_RAW / f"{sid}_P" / f"{sid}_Transcript.csv"
    if not path.exists():
        # case variants
        hits = list((EDAIC_RAW / f"{sid}_P").glob("*Transcript*.csv"))
        if not hits:
            raise FileNotFoundError(f"No transcript for {sid}")
        path = hits[0]
    rows = []
    dropped_long = 0
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = (row.get("Text") or row.get("value") or "").strip()
            if not text:
                continue
            try:
                start = float(row.get("Start_Time") or row.get("start_time") or 0.0)
                end = float(row.get("End_Time") or row.get("end_time") or start)
            except ValueError:
                continue
            if end <= start:
                end = start + 0.05
            if (end - start) > MAX_UTT_DUR_SEC:
                dropped_long += 1
                continue
            rows.append(
                {
                    "segment_id": i,
                    "start_time": start,
                    "end_time": end,
                    "text": text,
                    "confidence": row.get("Confidence"),
                }
            )
    if dropped_long:
        log(f"{sid}: dropped {dropped_long} ultra-long transcript rows (>{MAX_UTT_DUR_SEC}s)")
    if not rows:
        raise RuntimeError(f"Empty transcript after filtering: {sid}")
    return rows


def daic_participant_segments(daic: dict) -> Tuple[torch.Tensor, torch.Tensor, List[dict]]:
    """
    Keep Participant turns from DAIC .pt (drop Ellie) so T/A align to subject speech.

    Also drop ultra-long / non-monotonic ASR tails the same way ``read_transcript``
    does for ``edaic_full`` (corrupt rows often restart near the first utterance
    and span almost the whole session).
    """
    text = daic["text_features"].float()
    audio = daic["audio_features"].float()
    meta = list(daic.get("meta_data") or [])
    if len(meta) != text.size(0):
        # no usable meta: keep all (assumes already participant-only)
        segs = [
            {
                "segment_id": i,
                "start_time": float(i),
                "end_time": float(i) + 1.0,
                "speaker": "Participant",
                "text": "",
            }
            for i in range(text.size(0))
        ]
        return text, audio, segs

    keep_idx: List[int] = []
    segs: List[dict] = []
    dropped_long = 0
    dropped_nonmono = 0
    prev_start: Optional[float] = None
    for i, m in enumerate(meta):
        spk = str(m.get("speaker") or "Participant").strip().lower()
        if "ellie" in spk or "interviewer" in spk or spk in {"e", "therapist", "agent"}:
            continue
        start = float(m.get("start_time", 0.0))
        end = float(m.get("end_time", start + 0.5))
        if end <= start:
            end = start + 0.05
        if (end - start) > MAX_UTT_DUR_SEC:
            dropped_long += 1
            continue
        if prev_start is not None and start + 1e-6 < prev_start:
            dropped_nonmono += 1
            continue
        keep_idx.append(i)
        segs.append(
            {
                "segment_id": len(segs),
                "start_time": start,
                "end_time": end,
                "speaker": "Participant",
                "text": m.get("text") or "",
            }
        )
        prev_start = start
    if dropped_long or dropped_nonmono:
        log(
            f"daic_participant_segments: dropped_long={dropped_long} "
            f"dropped_nonmono={dropped_nonmono} (>{MAX_UTT_DUR_SEC}s or start rewind)"
        )
    if not keep_idx:
        raise RuntimeError("No Participant segments found in DAIC meta_data")
    idx = torch.tensor(keep_idx, dtype=torch.long)
    return text.index_select(0, idx), audio.index_select(0, idx), segs


def find_openface_csv(sid: str) -> Path:
    folder = EDAIC_RAW / f"{sid}_P"
    candidates = [
        folder / "features" / f"{sid}_OpenFace2.1.0_Pose_gaze_AUs.csv",
        folder / f"{sid}_OpenFace2.1.0_Pose_gaze_AUs.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    hits = list(folder.rglob("*OpenFace*Pose_gaze_AUs.csv"))
    if not hits:
        raise FileNotFoundError(f"No OpenFace Pose_gaze_AUs for {sid}")
    return hits[0]


def load_openface_matrix(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return timestamps [N], success [N], feats [N, 31]."""
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # strip spaces in header names
        header = [h.strip() for h in header]
        idx = {name: i for i, name in enumerate(header)}
        missing = [c for c in (["timestamp", "success"] + VISUAL_COLS) if c not in idx]
        if missing:
            raise KeyError(f"{path.name} missing columns: {missing[:8]}...")

        ts_i, suc_i = idx["timestamp"], idx["success"]
        feat_idx = [idx[c] for c in VISUAL_COLS]
        ts_list, suc_list, feat_list = [], [], []
        for row in reader:
            if not row or len(row) <= max(feat_idx):
                continue
            try:
                ts = float(row[ts_i])
                suc = float(row[suc_i])
                feats = [float(row[j]) for j in feat_idx]
            except ValueError:
                continue
            ts_list.append(ts)
            suc_list.append(suc)
            feat_list.append(feats)
    if not ts_list:
        raise RuntimeError(f"No OpenFace rows in {path}")
    return (
        np.asarray(ts_list, dtype=np.float64),
        np.asarray(suc_list, dtype=np.float64),
        np.asarray(feat_list, dtype=np.float32),
    )


def align_visual_to_segments(
    timestamps: np.ndarray,
    success: np.ndarray,
    feats: np.ndarray,
    segments: List[dict],
) -> torch.Tensor:
    """Mean-pool successful OpenFace frames inside each utterance window -> [T,31]."""
    out = np.zeros((len(segments), VISUAL_DIM), dtype=np.float32)
    for i, seg in enumerate(segments):
        start, end = seg["start_time"], seg["end_time"]
        # small pad for short utterances
        pad = 0.05
        mask = (timestamps >= (start - pad)) & (timestamps <= (end + pad)) & (success > 0.5)
        if not np.any(mask):
            # fallback: nearest frame
            j = int(np.argmin(np.abs(timestamps - 0.5 * (start + end))))
            out[i] = feats[j]
        else:
            out[i] = feats[mask].mean(axis=0)
    # z-score per session for stability (safe if std~0)
    mu = out.mean(axis=0, keepdims=True)
    sd = out.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)
    out = (out - mu) / sd
    return torch.from_numpy(out)


def build_visual_768(sid: str, segments: List[dict]) -> torch.Tensor:
    """Build turn-locked official DenseNet-201/PCA visual embeddings."""
    if not PROJECTION_PATH.is_file():
        raise FileNotFoundError(
            f"Missing TRAIN-only visual projection: {PROJECTION_PATH}. Run "
            "rebuild_edaic_visual_768.py first."
        )
    timestamps, frame_features = load_densenet_frames(sid)
    pooled, _ = pool_frames_to_turns(timestamps, frame_features, segments)
    return project_turns(pooled, load_projection())


@torch.no_grad()
def encode_text_segments(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
) -> torch.Tensor:
    embs = []
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=128
        ).to(device)
        out = model(**inputs)
        embs.append(out.last_hidden_state[0, 0, :].detach().cpu())
    return torch.stack(embs, dim=0)


def init_egemaps():
    import opensmile

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    if len(smile.feature_names) != AUDIO_DIM:
        raise RuntimeError(
            f"Expected {AUDIO_DIM} eGeMAPS features, found {len(smile.feature_names)}"
        )
    return smile


def encode_audio_segments(
    wav: np.ndarray,
    sr: int,
    segments: List[dict],
    smile,
) -> torch.Tensor:
    embs = []
    n = len(wav)
    for seg in segments:
        s = max(0, int(seg["start_time"] * sr))
        e = min(n, int(seg["end_time"] * sr))
        chunk = wav[s:e].astype(np.float32)
        if chunk.size < max(1, int(0.05 * sr)):
            vec = np.zeros(AUDIO_DIM, dtype=np.float32)
        else:
            values = smile.process_signal(chunk, sr).to_numpy(
                dtype=np.float64
            ).reshape(-1)
            if values.size != AUDIO_DIM:
                raise RuntimeError(
                    f"Expected {AUDIO_DIM} eGeMAPS values, found {values.size}"
                )
            vec = np.nan_to_num(
                values, nan=0.0, posinf=0.0, neginf=0.0
            ).astype(np.float32)
        embs.append(torch.from_numpy(vec))
    return torch.stack(embs, dim=0).float()


def resolve_mentalbert_path() -> str:
    if MENTALBERT_LOCAL.exists() and (MENTALBERT_LOCAL / "config.json").exists():
        return str(MENTALBERT_LOCAL)
    return TEXT_MODEL


def init_models(device: torch.device):
    text_src = resolve_mentalbert_path()
    log(f"Loading text model from: {text_src}")
    try:
        local_only = Path(text_src).exists()
        tok = AutoTokenizer.from_pretrained(
            text_src, token=HF_TOKEN, local_files_only=local_only
        )
        text_model = AutoModel.from_pretrained(
            text_src, token=HF_TOKEN, local_files_only=local_only
        )
        log("MentalBERT loaded")
    except Exception as e:
        # Do NOT silently fall back for full extract — keep modality consistent with DAIC.
        raise RuntimeError(
            "Failed to load MentalBERT (needed for E-DAIC-only subjects). "
            f"Set HF_TOKEN or ensure local cache at {MENTALBERT_LOCAL}. Underlying: {e}"
        ) from e
    log("Loading openSMILE eGeMAPSv02 functionals")
    smile = init_egemaps()
    text_model = text_model.to(device).eval()
    return tok, text_model, smile


def process_one(
    sid: str,
    tokenizer,
    text_model,
    smile,
    device: torch.device,
    force: bool = False,
) -> str:
    out_path = EDAIC_OUT / f"{sid}_features.pt"
    if out_path.exists() and not force:
        return "SKIP"

    daic_path = DAIC_FEAT / f"{sid}_features.pt"
    if daic_path.exists():
        # Overlap subjects: reuse DAIC MentalBERT Participant turns;
        # Align DenseNet/PCA and eGeMAPS to the same DAIC Participant timestamps.
        daic = torch.load(daic_path, map_location="cpu", weights_only=False)
        text, _, segments = daic_participant_segments(daic)
        visual = build_visual_768(sid, segments)
        wav_path = EDAIC_RAW / f"{sid}_P" / f"{sid}_AUDIO.wav"
        wav, sr = librosa.load(str(wav_path), sr=16000, mono=True)
        audio = encode_audio_segments(wav, sr, segments, smile)
        t = min(text.size(0), audio.size(0), visual.size(0))
        text, audio, visual = text[:t], audio[:t], visual[:t]
        segments = segments[:t]
        meta = [
            {
                "segment_id": i,
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "speaker": "Participant",
                "text": seg["text"],
                "source": "daic_participant+edaic_visual",
            }
            for i, seg in enumerate(segments)
        ]
        mode = "REUSE_DAIC_TA"
    else:
        segments = read_transcript(sid)
        visual = build_visual_768(sid, segments)
        if text_model is None or tokenizer is None:
            raise RuntimeError(f"{sid} needs full extract but models are not loaded")
        wav_path = EDAIC_RAW / f"{sid}_P" / f"{sid}_AUDIO.wav"
        if not wav_path.exists():
            hits = list((EDAIC_RAW / f"{sid}_P").glob("*.wav"))
            if not hits:
                raise FileNotFoundError(f"No wav for {sid}")
            wav_path = hits[0]
        wav, sr = librosa.load(str(wav_path), sr=16000, mono=True)
        texts = [s["text"] for s in segments]
        text = encode_text_segments(texts, tokenizer, text_model, device)
        audio = encode_audio_segments(wav, sr, segments, smile)
        t = min(text.size(0), audio.size(0), visual.size(0))
        text, audio, visual = text[:t], audio[:t], visual[:t]
        segments = segments[:t]
        meta = [
            {
                "segment_id": i,
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "speaker": "Participant",
                "text": seg["text"],
                "source": "edaic_full",
            }
            for i, seg in enumerate(segments)
        ]
        mode = "FULL_EXTRACT"

    payload = {
        "text_features": text.contiguous(),
        "audio_features": audio.contiguous(),
        "visual_features": visual.contiguous(),
        "meta_data": meta,
        "visual_dim": VISUAL_DIM,
        "visual_source": (
            "official E-DAIC DenseNet-201 1920-D per-second embeddings; "
            "turn mean-pool; TRAIN-only PCA-768; per-turn L2 normalization"
        ),
        "visual_source_dim": 1920,
        "visual_projection": str(PROJECTION_PATH),
        "visual_projection_fit_split": "train",
        "audio_model": "opensmile.eGeMAPSv02.Functionals",
        "audio_dim": AUDIO_DIM,
    }
    EDAIC_OUT.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return mode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="process only first N subjects")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ids", type=str, default="", help="comma-separated ids, e.g. 300,304")
    parser.add_argument(
        "--mode",
        choices=["all", "reuse", "full"],
        default="all",
        help="reuse=DAIC-overlap only; full=E-DAIC-only; all=both",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} out={EDAIC_OUT} mode={args.mode}")
    subjects = list_subjects()
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        subjects = [s for s in subjects if s in want]
    if args.mode == "reuse":
        subjects = [s for s in subjects if (DAIC_FEAT / f"{s}_features.pt").exists()]
    elif args.mode == "full":
        subjects = [s for s in subjects if not (DAIC_FEAT / f"{s}_features.pt").exists()]
    if args.limit and args.limit > 0:
        subjects = subjects[: args.limit]

    need_full = [s for s in subjects if not (DAIC_FEAT / f"{s}_features.pt").exists()]
    tokenizer = text_model = smile = None
    if need_full:
        tokenizer, text_model, smile = init_models(device)
    else:
        smile = init_egemaps()
        log("No E-DAIC-only subjects in this run; MentalBERT not loaded.")

    counts = {"REUSE_DAIC_TA": 0, "FULL_EXTRACT": 0, "SKIP": 0, "FAIL": 0}
    for i, sid in enumerate(subjects, 1):
        try:
            # lazy-load models if we encounter first full extract under mixed set
            if (
                not (DAIC_FEAT / f"{sid}_features.pt").exists()
                and text_model is None
            ):
                tokenizer, text_model, smile = init_models(device)
            status = process_one(
                sid,
                tokenizer,
                text_model,
                smile,
                device,
                force=args.force,
            )
            counts[status] = counts.get(status, 0) + 1
            log(f"[{i}/{len(subjects)}] {sid} -> {status}")
        except Exception as e:
            counts["FAIL"] += 1
            log(f"[{i}/{len(subjects)}] {sid} -> FAIL {e}")

    log(f"DONE counts={counts}")
    summary = ROOT / "EDAIC" / "EXTRACT_STATUS.txt"
    summary.write_text(
        f"counts={counts}\nvisual_dim={VISUAL_DIM}\nout={EDAIC_OUT}\nlog={LOG}\n",
        encoding="utf-8",
    )
    return 0 if counts["FAIL"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
