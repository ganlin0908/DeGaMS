"""DeepFilterNet residual enhance + Silero VAD soft-gate → wav2vec2 turns.

Writes ``degams_clean_v2/{corpus}/wav2vec_enh/{sid}.pt`` with the same schema as
``extract_clean_wav2vec_vad.py``, plus:
  - silero_speech_prob mean / soft gate
  - deepfilternet mix ratio (residual with raw)

Backend priority:
  1) official ``deep-filter.exe`` CLI (Windows binary; works without pyDF)
  2) Python ``df.enhance`` if installed
  3) noisereduce spectral gate
  4) raw + Silero only
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor

CODE_FEAT = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("DEGAMS_ROOT", r"/path/to/degams_data"))
sys.path.insert(0, str(CODE_FEAT))

from audio_front_vad import (  # noqa: E402
    extract_turn_audio,
    load_mono_slice,
    speech_mask,
)
from clean_v2_paths import (  # noqa: E402
    TARGET_SR,
    daic_dirs,
    edaic_dirs,
    ensure_clean_tree,
)
from extract_clean_mel_vad import resolve_wav  # noqa: E402

CKPT = "facebook/wav2vec2-base"
DF_MIX = 0.7  # enhanced * mix + raw * (1-mix); keeps pause/weak-speech cues
DF_CLI = ROOT / "tools" / "deepfilter" / "deep-filter.exe"
DF_CACHE = ROOT / "egemaps_experiment" / "artifacts" / "deepfilter_session_cache"


class _Enhancers:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.df_model = None
        self.df_state = None
        self.df_cli: Optional[Path] = None
        self._nr = None
        self.silero = None
        self.silero_utils = None
        self._session_enh: Dict[str, Path] = {}
        self.backend = "none"
        self._init_df()
        self._init_silero()

    def _init_df(self) -> None:
        if DF_CLI.is_file():
            self.df_cli = DF_CLI
            self.backend = "deep-filter-cli"
            DF_CACHE.mkdir(parents=True, exist_ok=True)
            print(f"[audio-enh] DeepFilterNet CLI ready: {DF_CLI}", flush=True)
            return
        try:
            from df.enhance import enhance, init_df  # type: ignore

            self.df_model, self.df_state, _ = init_df()
            self._df_enhance = enhance
            self.backend = "df.enhance"
            print("[audio-enh] DeepFilterNet Python ready", flush=True)
            return
        except Exception as e:
            print(f"[audio-enh] DeepFilterNet Python unavailable ({e})", flush=True)
        try:
            import noisereduce as nr  # type: ignore

            self._nr = nr
            self.backend = "noisereduce"
            print("[audio-enh] noisereduce spectral-gate fallback ready", flush=True)
        except Exception as e:
            print(f"[audio-enh] no enhance backend ({e}); raw+Silero only", flush=True)

    def _init_silero(self) -> None:
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            self.silero = model.to(self.device).eval()
            self.silero_utils = utils
            print("[audio-enh] Silero VAD ready", flush=True)
        except Exception as e:
            print(f"[audio-enh] Silero unavailable ({e}); energy VAD only", flush=True)

    def _cli_enhance_file(self, wav_path: Path) -> Path:
        """Run deep-filter.exe once per session WAV; cache under DF_CACHE."""
        key = hashlib.md5(str(wav_path.resolve()).encode("utf-8")).hexdigest()[:16]
        out_dir = DF_CACHE / f"{wav_path.stem}_{key}"
        out_dir.mkdir(parents=True, exist_ok=True)
        existing = list(out_dir.glob("*.wav"))
        if existing:
            return existing[0]
        cmd = [
            str(self.df_cli),
            str(wav_path),
            "-o",
            str(out_dir),
            "-D",
            "-a",
            "100",
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        outs = sorted(out_dir.glob("*.wav"))
        if not outs:
            raise FileNotFoundError(f"deep-filter produced no wav in {out_dir}")
        return outs[0]

    def ensure_session_enhanced(self, wav_path: Path) -> Optional[Path]:
        if self.df_cli is None or not wav_path.is_file():
            return None
        key = str(wav_path.resolve())
        if key in self._session_enh and self._session_enh[key].is_file():
            return self._session_enh[key]
        try:
            enh_path = self._cli_enhance_file(wav_path)
            self._session_enh[key] = enh_path
            return enh_path
        except Exception as e:
            print(f"[audio-enh] CLI failed for {wav_path.name}: {e}", flush=True)
            return None

    def enhance_slice(
        self,
        raw: np.ndarray,
        sr: int,
        *,
        enh_session: Optional[Path] = None,
        t0: float = 0.0,
        t1: Optional[float] = None,
    ) -> np.ndarray:
        """Residual mix of enhanced + raw for one turn."""
        raw = np.asarray(raw, dtype=np.float32).reshape(-1)
        if raw.size < sr // 20:
            return raw
        enh = None
        if enh_session is not None and enh_session.is_file():
            try:
                enh, _ = load_mono_slice(enh_session, t0, t1, sr_out=sr)
            except Exception:
                enh = None
        if enh is None and self.df_model is not None:
            try:
                audio = torch.from_numpy(raw).float().unsqueeze(0)
                if sr != int(self.df_state.sr()):
                    import torchaudio

                    audio = torchaudio.functional.resample(
                        audio, sr, int(self.df_state.sr())
                    )
                e = self._df_enhance(self.df_model, self.df_state, audio)
                e = e.squeeze(0).detach().cpu().numpy().astype(np.float32)
                if e.shape[0] != raw.shape[0]:
                    import librosa

                    e = librosa.resample(
                        e, orig_sr=int(self.df_state.sr()), target_sr=sr
                    )
                enh = e
            except Exception:
                enh = None
        if enh is None and self._nr is not None:
            try:
                enh = np.asarray(
                    self._nr.reduce_noise(
                        y=raw, sr=sr, stationary=False, prop_decrease=0.75
                    ),
                    dtype=np.float32,
                ).reshape(-1)
            except Exception:
                enh = None
        if enh is None:
            return raw
        n = min(enh.shape[0], raw.shape[0])
        out = raw.copy()
        out[:n] = DF_MIX * enh[:n] + (1.0 - DF_MIX) * raw[:n]
        return out

    def silero_probs(self, wav: np.ndarray, sr: int) -> np.ndarray:
        """Return per-chunk speech probabilities ~aligned to 10ms energy frames."""
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        n_frames = max(1, 1 + (wav.size - int(sr * 0.025)) // max(1, int(sr * 0.01)))
        if self.silero is None:
            mask, _ = speech_mask(wav, sr)
            if mask.size == 0:
                return np.zeros(n_frames, dtype=np.float32)
            out = mask.astype(np.float32)
            if out.size < n_frames:
                out = np.pad(out, (0, n_frames - out.size))
            return out[:n_frames]
        try:
            get_speech_timestamps, _, _, _, get_speech_probs = self.silero_utils
            audio = torch.from_numpy(wav).float()
            if sr != 16000:
                import torchaudio

                audio = torchaudio.functional.resample(
                    audio.unsqueeze(0), sr, 16000
                ).squeeze(0)
                sr_use = 16000
            else:
                sr_use = sr
            if callable(get_speech_probs):
                with torch.no_grad():
                    probs = get_speech_probs(audio, self.silero, sampling_rate=sr_use)
                probs = np.asarray(probs, dtype=np.float32).reshape(-1)
            else:
                with torch.no_grad():
                    ts = get_speech_timestamps(
                        audio, self.silero, sampling_rate=sr_use, threshold=0.35
                    )
                hop = max(1, int(sr_use * 0.01))
                n = max(1, audio.numel() // hop)
                probs = np.zeros(n, dtype=np.float32)
                for seg in ts:
                    a = int(seg["start"]) // hop
                    b = max(a + 1, int(seg["end"]) // hop)
                    probs[a:b] = 1.0
            if probs.size != n_frames:
                idx = np.linspace(0, max(probs.size - 1, 0), n_frames).astype(np.int64)
                probs = probs[idx]
            return probs.astype(np.float32)
        except Exception:
            mask, _ = speech_mask(wav, sr)
            out = (
                mask.astype(np.float32)
                if mask.size
                else np.zeros(n_frames, dtype=np.float32)
            )
            if out.size < n_frames:
                out = np.pad(out, (0, n_frames - out.size))
            return out[:n_frames]


def _soft_pool(
    model: Wav2Vec2Model,
    processor: Wav2Vec2Processor,
    wav: np.ndarray,
    probs: np.ndarray,
    device: torch.device,
) -> Tuple[torch.Tensor, float, float]:
    """Soft-gate wav2vec frames by Silero/energy speech probability."""
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    if wav.size < TARGET_SR // 20:
        return torch.zeros(768), 0.0, 0.0
    mean_p = float(probs.mean()) if probs.size else 0.0

    inputs = processor(wav, sampling_rate=TARGET_SR, return_tensors="pt", padding=False)
    input_values = inputs.input_values.to(device)
    with torch.no_grad():
        hs = model(input_values).last_hidden_state.squeeze(0)
    tf = int(hs.size(0))
    if tf <= 0:
        return torch.zeros(768), mean_p, 0.0
    if probs.size == 0:
        emb = hs.mean(dim=0)
        return emb.float().cpu(), mean_p, 0.0
    idx = np.linspace(0, max(probs.size - 1, 0), tf).astype(np.int64)
    w = torch.from_numpy(probs[idx]).to(device=device, dtype=hs.dtype).clamp(0.0, 1.0)
    w = w + 0.05
    emb = (hs * w.unsqueeze(-1)).sum(dim=0) / w.sum().clamp_min(1e-3)
    speech_frac = float((probs >= 0.5).mean()) if probs.size else 0.0
    return emb.float().cpu(), mean_p, speech_frac


def process_session(
    corpus: str,
    sid: str,
    pack: dict,
    model: Wav2Vec2Model,
    processor: Wav2Vec2Processor,
    enh: _Enhancers,
    device: torch.device,
) -> dict:
    meta = list(pack.get("meta_data") or [])
    enh_session = None
    if meta and enh.df_cli is not None:
        try:
            path0, _, _ = resolve_wav(corpus, sid, meta[0])
            if path0.is_file():
                enh_session = enh.ensure_session_enhanced(path0)
        except Exception:
            enh_session = None

    turns: List[torch.Tensor] = []
    quals: List[float] = []
    pauses: List[np.ndarray] = []
    speech_fracs: List[float] = []
    silero_means: List[float] = []
    n_ok = 0
    for turn in meta:
        try:
            path, t0, t1 = resolve_wav(corpus, sid, turn)
            if not path.is_file():
                raise FileNotFoundError(path)
            wav, sr = load_mono_slice(path, t0, t1)
            wav_enh = enh.enhance_slice(
                wav, sr, enh_session=enh_session, t0=t0, t1=t1
            )
            feat = extract_turn_audio(wav_enh, sr)
            probs = enh.silero_probs(wav_enh, sr)
            emb, mean_p, sfrac = _soft_pool(model, processor, wav_enh, probs, device)
            q = float(feat["audio_quality"])
            q = float(np.clip(0.5 * q + 0.5 * max(mean_p, 0.05), 0.05, 1.0))
            emb = emb * q
            turns.append(emb)
            quals.append(q)
            pauses.append(feat["pause_feats"])
            speech_fracs.append(sfrac)
            silero_means.append(mean_p)
            n_ok += 1
        except Exception:
            turns.append(torch.zeros(768))
            quals.append(0.05)
            pauses.append(np.zeros(8, dtype=np.float32))
            speech_fracs.append(0.0)
            silero_means.append(0.0)

    at = torch.stack(turns, dim=0)
    return {
        "audio": at.mean(dim=0),
        "audio_turns": at,
        "audio_quality": torch.tensor(quals, dtype=torch.float32),
        "pause_feats": torch.from_numpy(np.stack(pauses, axis=0)),
        "speech_frac": torch.tensor(speech_fracs, dtype=torch.float32),
        "silero_speech_prob": torch.tensor(silero_means, dtype=torch.float32),
        "checkpoint": CKPT,
        "sr": TARGET_SR,
        "pool": "silero_softgate_wav2vec_mean",
        "enhance": f"{enh.backend}_residual_mix={DF_MIX}",
        "align": "clean_v2_text_turn",
        "soft_quality_scaled": True,
        "session_id": sid,
        "n_turns": int(at.size(0)),
        "n_ok_turns": n_ok,
        "extracted_at": datetime.now().isoformat(timespec="seconds"),
    }


def run_corpus(corpus: str, *, overwrite: bool, device: torch.device, limit: int) -> None:
    ensure_clean_tree()
    dirs = daic_dirs() if corpus == "daic" else edaic_dirs()
    text_dir: Path = dirs["text"]
    out_dir = dirs["wav2vec_enh"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{corpus}] loading {CKPT} …", flush=True)
    processor = Wav2Vec2Processor.from_pretrained(CKPT)
    model = Wav2Vec2Model.from_pretrained(CKPT).to(device).eval()
    enh = _Enhancers(device)

    paths = sorted(text_dir.glob("*_features.pt"))
    if limit > 0:
        paths = paths[:limit]
    print(
        f"[{corpus}] sessions={len(paths)} → {out_dir} backend={enh.backend}",
        flush=True,
    )
    for tp in tqdm(paths, desc=f"w2v-enh:{corpus}"):
        sid = tp.name.replace("_features.pt", "")
        dest = out_dir / f"{sid}.pt"
        if dest.is_file() and not overwrite:
            continue
        pack = torch.load(tp, map_location="cpu", weights_only=False)
        out = process_session(corpus, sid, pack, model, processor, enh, device)
        torch.save(out, dest)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"[{corpus}] done", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", choices=["daic", "edaic", "both"], default="both")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpora = ["edaic", "daic"] if args.corpus == "both" else [args.corpus]
    for c in corpora:
        run_corpus(c, overwrite=args.overwrite, device=device, limit=int(args.limit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
