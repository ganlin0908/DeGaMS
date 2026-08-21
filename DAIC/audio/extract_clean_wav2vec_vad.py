"""Re-extract wav2vec2 turn embeds with VAD speech-frame pooling (clean_v2).

Original ``wav2vec_embeds`` used full-turn frame mean (silence mixed in).
This writes ``degams_clean_v2/{corpus}/wav2vec/{sid}.pt`` with:
  - audio_turns [T,768] mean-pooled on speech frames only
  - audio_quality / pause_feats aligned (same as Mel clean_v2)
  - Participant-only turns via clean_v2 text meta
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

CODE_FEAT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE_FEAT))

from audio_front_vad import (  # noqa: E402
    extract_turn_audio,
    load_mono_slice,
    speech_mask,
)
from clean_v2_paths import (  # noqa: E402
    CLEAN_ROOT,
    DAIC_WAV,
    EDAIC_RAW,
    TARGET_SR,
    daic_dirs,
    edaic_dirs,
    ensure_clean_tree,
)
from extract_clean_mel_vad import resolve_wav  # noqa: E402

CKPT = "facebook/wav2vec2-base"


def _speech_frame_pool(
    model: Wav2Vec2Model,
    processor: Wav2Vec2Processor,
    wav: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, float]:
    """Return [768] pooled on VAD speech frames + speech_frac."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size < TARGET_SR // 20:
        return torch.zeros(768), 0.0
    mask, _ = speech_mask(wav, TARGET_SR)
    speech_frac = float(mask.mean()) if mask.size else 0.0

    inputs = processor(wav, sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        hs = model(input_values).last_hidden_state.squeeze(0)  # [Tf, 768]
    tf = int(hs.size(0))
    if tf <= 0:
        return torch.zeros(768), speech_frac
    # resample VAD mask (10ms hop) → wav2vec frames
    if mask.size == 0:
        emb = hs.mean(dim=0)
    else:
        idx = (np.linspace(0, max(mask.size - 1, 0), tf)).astype(np.int64)
        m = torch.from_numpy(mask[idx]).to(device=device, dtype=torch.bool)
        if bool(m.any()):
            emb = hs[m].mean(dim=0)
        else:
            emb = hs.mean(dim=0)
    return emb.float().cpu(), speech_frac


def process_session(
    corpus: str,
    sid: str,
    pack: dict,
    model: Wav2Vec2Model,
    processor: Wav2Vec2Processor,
    device: torch.device,
) -> dict:
    meta = list(pack.get("meta_data") or [])
    turns: List[torch.Tensor] = []
    quals: List[float] = []
    pauses: List[np.ndarray] = []
    speech_fracs: List[float] = []
    n_ok = 0
    for turn in meta:
        try:
            path, t0, t1 = resolve_wav(corpus, sid, turn)
            if not path.is_file():
                raise FileNotFoundError(path)
            wav, sr = load_mono_slice(path, t0, t1)
            feat = extract_turn_audio(wav, sr)
            emb, sfrac = _speech_frame_pool(model, processor, wav, device)
            # soft scale by same audio_quality as Mel clean_v2
            q = float(feat["audio_quality"])
            emb = emb * max(q, 0.05)
            turns.append(emb)
            quals.append(q)
            pauses.append(feat["pause_feats"])
            speech_fracs.append(sfrac)
            n_ok += 1
        except Exception:
            turns.append(torch.zeros(768))
            quals.append(0.05)
            pauses.append(np.zeros(8, dtype=np.float32))
            speech_fracs.append(0.0)

    at = torch.stack(turns, dim=0)
    return {
        "audio": at.mean(dim=0),
        "audio_turns": at,
        "audio_quality": torch.tensor(quals, dtype=torch.float32),
        "pause_feats": torch.from_numpy(np.stack(pauses, axis=0)),
        "speech_frac": torch.tensor(speech_fracs, dtype=torch.float32),
        "checkpoint": CKPT,
        "sr": TARGET_SR,
        "pool": "vad_speech_frame_mean_per_turn",
        "align": "clean_v2_text_turn",
        "soft_quality_scaled": True,
        "session_id": sid,
        "n_turns": int(at.size(0)),
        "n_ok_turns": n_ok,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }


def run_corpus(corpus: str, *, overwrite: bool, device: torch.device) -> None:
    ensure_clean_tree()
    dirs = daic_dirs() if corpus == "daic" else edaic_dirs()
    text_dir: Path = dirs["text"]
    out_dir = CLEAN_ROOT / corpus / "wav2vec"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{corpus}] loading {CKPT} …", flush=True)
    processor = Wav2Vec2Processor.from_pretrained(CKPT)
    model = Wav2Vec2Model.from_pretrained(CKPT).to(device).eval()

    paths = sorted(text_dir.glob("*_features.pt"))
    print(f"[{corpus}] sessions={len(paths)} → {out_dir}", flush=True)
    for tp in tqdm(paths, desc=f"w2v-vad:{corpus}"):
        sid = tp.name.replace("_features.pt", "")
        dest = out_dir / f"{sid}.pt"
        if dest.is_file() and not overwrite:
            continue
        pack = torch.load(tp, map_location="cpu", weights_only=False)
        out = process_session(corpus, sid, pack, model, processor, device)
        torch.save(out, dest)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[{corpus}] done", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=["daic", "edaic", "both"], default="both")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpora = ["edaic", "daic"] if args.corpus == "both" else [args.corpus]
    for c in corpora:
        run_corpus(c, overwrite=args.overwrite, device=device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
