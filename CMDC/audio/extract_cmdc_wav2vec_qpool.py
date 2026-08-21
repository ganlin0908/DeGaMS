"""CMDC Wav2Vec2 Q-pool turn embeds (protocol from FEATURE_AUDIT_20260729).

Per-question encode with TencentGameMate/chinese-wav2vec2-base hidden layer 6,
mean-pool frames → [768] per Q; stack to audio_turns [T,768].

Writes: egemaps_experiment/artifacts/wav2vec_embeds/cmdc/{sid}.pt
  audio: [768]  (mean over turns)
  audio_turns: [T,768]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm.auto import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

ROOT = Path(os.environ.get("DEGAMS_ROOT", r"/path/to/degams_data"))
FEAT = ROOT / "egemaps_experiment" / "1_feature_build" / "features" / "cmdc"
OUT = ROOT / "egemaps_experiment" / "artifacts" / "wav2vec_embeds" / "cmdc"
EXTRACTED = ROOT / "CMDC" / "raw" / "extracted"

CKPT = "TencentGameMate/chinese-wav2vec2-base"
HIDDEN_LAYER = 6  # 0-based into hidden_states (after embed); audit: layer 6
TARGET_SR = 16000
MAX_TURN_SEC = 120.0


def resolve_q_wav(sid: str, turn: dict) -> Optional[Path]:
    ap = turn.get("audio_path")
    if ap:
        p = Path(str(ap))
        if p.is_file():
            return p
    q = str(turn.get("question") or "").strip()
    if not q:
        return None
    # Prefer subject folder under extracted/
    for base in (EXTRACTED / sid, EXTRACTED):
        cand = base / sid / f"{q}.wav" if base == EXTRACTED else base / f"{q}.wav"
        if cand.is_file():
            return cand
        # also try nested
        hits = list((EXTRACTED / sid).glob(f"{q}.*")) if (EXTRACTED / sid).is_dir() else []
        for h in hits:
            if h.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a"}:
                return h
    return None


def load_mono(path: Path, max_sec: float = MAX_TURN_SEC) -> Tuple[np.ndarray, int]:
    info = sf.info(str(path))
    sr0 = int(info.samplerate)
    n_max = int(max_sec * sr0)
    stop = min(info.frames, n_max) if n_max > 0 else info.frames
    wav, _ = sf.read(str(path), start=0, stop=stop, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr0 != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr0, target_sr=TARGET_SR)
    if wav.size < TARGET_SR // 20:
        wav = np.zeros(TARGET_SR // 10, dtype=np.float32)
    return wav, TARGET_SR


def encode_turn(
    model: Wav2Vec2Model,
    fe: Wav2Vec2FeatureExtractor,
    wav: np.ndarray,
    device: torch.device,
    *,
    layer: int = HIDDEN_LAYER,
) -> torch.Tensor:
    inputs = fe(wav, sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        out = model(input_values, output_hidden_states=True)
    # hidden_states[0]=embed, [1]=layer1, ... → layer k at index k
    hs = out.hidden_states[layer].squeeze(0)  # [Tf, 768]
    if hs.numel() == 0:
        return torch.zeros(768)
    return hs.mean(dim=0).float().cpu()


def process_pack(
    pack_path: Path,
    model: Wav2Vec2Model,
    fe: Wav2Vec2FeatureExtractor,
    device: torch.device,
) -> dict:
    pack = torch.load(pack_path, map_location="cpu", weights_only=False)
    sid = str(pack.get("subject_id") or pack_path.name.replace("_features.pt", ""))
    meta = list(pack.get("meta_data") or [])
    text = pack["text_features"]
    t = int(text.size(0))
    if len(meta) != t:
        meta = meta[:t] + [{}] * max(0, t - len(meta))

    turns: List[torch.Tensor] = []
    ok = 0
    missing = 0
    for turn in meta[:t]:
        path = resolve_q_wav(sid, turn)
        if path is None or not path.is_file():
            turns.append(torch.zeros(768))
            missing += 1
            continue
        try:
            wav, _ = load_mono(path)
            emb = encode_turn(model, fe, wav, device)
            turns.append(emb)
            ok += 1
        except Exception:
            turns.append(torch.zeros(768))
            missing += 1

    at = torch.stack(turns, dim=0) if turns else torch.zeros(t, 768)
    return {
        "audio": at.mean(dim=0) if at.size(0) else torch.zeros(768),
        "audio_turns": at,
        "checkpoint": CKPT,
        "hidden_layer": HIDDEN_LAYER,
        "sr": TARGET_SR,
        "pool": "q_mean_layer6",
        "align": "meta_question_wav",
        "session_id": sid,
        "n_turns": int(at.size(0)),
        "n_ok_turns": ok,
        "n_missing_turns": missing,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not FEAT.is_dir():
        raise SystemExit(f"Missing feature dir: {FEAT}")
    if not EXTRACTED.is_dir():
        raise SystemExit(
            f"Missing CMDC raw extracted/: {EXTRACTED}\n"
            "Run: python CMDC/extract_zips.py first"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    print(f"loading {CKPT} layer={HIDDEN_LAYER} device={device}", flush=True)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(CKPT)
    model = Wav2Vec2Model.from_pretrained(CKPT).to(device).eval()

    packs = sorted(FEAT.glob("*_features.pt"))
    if args.limit > 0:
        packs = packs[: args.limit]
    print(f"sessions={len(packs)} → {OUT}", flush=True)

    for tp in tqdm(packs, desc="cmdc-qpool"):
        sid = tp.name.replace("_features.pt", "")
        dest = OUT / f"{sid}.pt"
        if dest.is_file() and not args.overwrite:
            continue
        out = process_pack(tp, model, fe, device)
        torch.save(out, dest)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
