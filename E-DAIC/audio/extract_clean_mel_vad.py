"""Extract VAD Mel + pause_feats + soft audio_quality for DAIC / E-DAIC clean_v2."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm

from audio_front_vad import extract_turn_audio, load_mono_slice
from clean_v2_paths import (
    DAIC_WAV,
    EDAIC_RAW,
    N_MELS,
    PAUSE_FEAT_DIM,
    PAUSE_FEAT_NAMES,
    TARGET_SR,
    daic_dirs,
    edaic_dirs,
    ensure_clean_tree,
)


def resolve_wav(corpus: str, sid: str, turn: dict) -> Tuple[Path, float, Optional[float]]:
    ap = turn.get("audio_path")
    if ap and Path(ap).is_file():
        return Path(ap), 0.0, None
    if corpus == "daic":
        full = DAIC_WAV / f"{sid}_AUDIO.wav"
        return full, float(turn.get("start_time") or 0.0), (
            float(turn["end_time"]) if turn.get("end_time") is not None else None
        )
    full = EDAIC_RAW / f"{sid}_P" / f"{sid}_AUDIO.wav"
    if not full.is_file():
        hits = list((EDAIC_RAW / f"{sid}_P").glob("*.wav"))
        full = hits[0] if hits else full
    return full, float(turn.get("start_time") or 0.0), (
        float(turn["end_time"]) if turn.get("end_time") is not None else None
    )


def process(corpus: str, sid: str, pack: dict) -> dict:
    meta = list(pack.get("meta_data") or [])
    mels: List[np.ndarray] = []
    pauses: List[np.ndarray] = []
    quals: List[float] = []
    snrs: List[float] = []
    n_ok = 0
    for turn in meta:
        try:
            path, t0, t1 = resolve_wav(corpus, sid, turn)
            if not path.is_file():
                raise FileNotFoundError(path)
            wav, sr = load_mono_slice(path, t0, t1)
            feat = extract_turn_audio(wav, sr)
            mels.append(feat["mel"])
            pauses.append(feat["pause_feats"])
            quals.append(float(feat["audio_quality"]))
            snrs.append(float(feat["snr_db"]))
            n_ok += 1
        except Exception:
            mels.append(np.zeros(N_MELS, dtype=np.float32))
            pauses.append(np.zeros(PAUSE_FEAT_DIM, dtype=np.float32))
            quals.append(0.05)
            snrs.append(0.0)

    mel = torch.from_numpy(np.stack(mels, axis=0))
    pause = torch.from_numpy(np.stack(pauses, axis=0))
    quality = torch.tensor(quals, dtype=torch.float32)
    return {
        "mel": mel.mean(dim=0),
        "mel_turns": mel,
        "pause_feats": pause,
        "audio_quality": quality,
        "pause_feat_names": PAUSE_FEAT_NAMES,
        "n_mels": N_MELS,
        "pause_dim": PAUSE_FEAT_DIM,
        "sr": TARGET_SR,
        "align": "text_turn_vad_speech_frames",
        "no_hard_silence_snr_drop": True,
        "session_id": sid,
        "n_turns": int(mel.size(0)),
        "n_ok_turns": int(n_ok),
        "snr_db_mean": float(np.mean(snrs)) if snrs else 0.0,
        "audio_quality_mean": float(quality.mean()) if quality.numel() else 0.0,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["daic", "edaic", "both"], default="both")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="", help="i/N")
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
        out = dirs["mel"]
        ok = 0
        for p in tqdm(packs, desc=f"{corpus}-mel-vad"):
            sid = p.name[: -len("_features.pt")]
            dest = out / f"{sid}.pt"
            if dest.is_file() and not args.overwrite:
                ok += 1
                continue
            pack = torch.load(p, map_location="cpu", weights_only=False)
            torch.save(process(corpus, sid, pack), dest)
            ok += 1
        print(f"{corpus} mel wrote/kept {ok}/{len(packs)} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
