"""Speech-frame Mel + pause features + soft audio_quality (no hard delete)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import librosa
import numpy as np

from clean_v2_paths import (
    FRAME_MS,
    HOP_MS,
    MAX_TURN_SEC,
    N_MELS,
    PAUSE_FEAT_DIM,
    PAUSE_MIN_SEC,
    Q_SNR_CENTER,
    Q_SNR_SCALE,
    Q_VOICED_FLOOR_SEC,
    TARGET_SR,
    VAD_ABS_DB,
    VAD_REL,
)

EPS = 1e-12


def load_mono_slice(
    path,
    start: float = 0.0,
    end: Optional[float] = None,
    *,
    sr_out: int = TARGET_SR,
) -> Tuple[np.ndarray, int]:
    import soundfile as sf

    info = sf.info(str(path))
    sr0 = int(info.samplerate)
    n = int(info.frames)
    i0 = int(max(0.0, start) * sr0)
    i1 = int(end * sr0) if end is not None else n
    i1 = min(max(i1, i0 + 1), n)
    wav, _ = sf.read(str(path), start=i0, stop=i1, dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1)
    if sr0 != sr_out:
        wav = librosa.resample(wav, orig_sr=sr0, target_sr=sr_out)
    max_n = int(MAX_TURN_SEC * sr_out)
    if wav.size > max_n:
        wav = wav[:max_n]
    if wav.size < sr_out // 20:
        wav = np.zeros(max(sr_out // 10, 1), dtype=np.float32)
    return wav, sr_out


def frame_rms(x: np.ndarray, sr: int) -> np.ndarray:
    n = max(1, int(sr * FRAME_MS / 1000))
    h = max(1, int(sr * HOP_MS / 1000))
    x = np.asarray(x, dtype=np.float64)
    if x.size < n:
        return np.array([float(np.sqrt(np.mean(x**2) + EPS))], dtype=np.float64)
    n_frames = 1 + (x.size - n) // h
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n_frames, n), strides=(x.strides[0] * h, x.strides[0])
    )
    return np.sqrt(np.mean(frames**2, axis=1) + EPS)


def speech_mask(x: np.ndarray, sr: int) -> np.ndarray:
    """Relative VAD after peak-normalizing the clip (stabilizes soft speech)."""
    x = np.asarray(x, dtype=np.float64)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak > 1e-6:
        x = x / peak
    fr = frame_rms(x, sr)
    noise = float(np.percentile(fr, 20)) if fr.size else EPS
    abs_floor = 10 ** (VAD_ABS_DB / 20.0)
    thr = max(abs_floor, noise * VAD_REL)
    return fr >= thr, fr


def pause_stats(mask: np.ndarray, sr: int) -> Dict[str, float]:
    hop = max(1, int(sr * HOP_MS / 1000))
    frame_dur = hop / float(sr)
    if mask.size == 0:
        return {
            "pause_ratio": 1.0,
            "speech_ratio": 0.0,
            "speech_dur": 0.0,
            "max_pause_sec": 0.0,
            "mean_pause_sec": 0.0,
            "n_pauses": 0.0,
        }
    speech_ratio = float(mask.mean())
    pause_ratio = 1.0 - speech_ratio
    speech_dur = float(mask.sum() * frame_dur)
    # pause runs
    pauses = []
    i = 0
    n = mask.size
    while i < n:
        if mask[i]:
            i += 1
            continue
        j = i
        while j < n and not mask[j]:
            j += 1
        dur = (j - i) * frame_dur
        if dur >= PAUSE_MIN_SEC:
            pauses.append(dur)
        i = j
    return {
        "pause_ratio": pause_ratio,
        "speech_ratio": speech_ratio,
        "speech_dur": speech_dur,
        "max_pause_sec": float(max(pauses) if pauses else 0.0),
        "mean_pause_sec": float(np.mean(pauses) if pauses else 0.0),
        "n_pauses": float(len(pauses)),
    }


def estimate_snr_db(fr: np.ndarray, mask: np.ndarray) -> float:
    if mask.any() and (~mask).any():
        sig = float(np.mean(fr[mask] ** 2))
        noi = float(np.mean(fr[~mask] ** 2))
    elif mask.any():
        sig = float(np.mean(fr[mask] ** 2))
        noi = float(np.percentile(fr, 20) ** 2)
    else:
        sig = float(np.mean(fr**2)) if fr.size else EPS
        noi = float(np.percentile(fr, 20) ** 2) if fr.size else EPS
    return float(10.0 * np.log10((sig + EPS) / (noi + EPS)))


def soft_audio_quality(speech_dur: float, snr_db: float) -> float:
    """Soft weight in (0,1]; never hard-zero a turn."""
    # voiced duration factor
    d = speech_dur / max(Q_VOICED_FLOOR_SEC, 1e-6)
    w_dur = float(np.clip(d, 0.05, 1.0))
    # SNR logistic around center (soft speech OK if voiced)
    w_snr = 1.0 / (1.0 + np.exp(-(snr_db - Q_SNR_CENTER) / max(Q_SNR_SCALE, 1e-6)))
    w_snr = float(np.clip(w_snr, 0.15, 1.0))
    return float(np.clip(0.5 * w_dur + 0.5 * w_snr, 0.05, 1.0))


def speech_rms_normalize(x: np.ndarray, mask: np.ndarray, sr: int, target_rms: float = 0.1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    fr = frame_rms(x.astype(np.float64), sr)
    hop = max(1, int(sr * HOP_MS / 1000))
    n = max(1, int(sr * FRAME_MS / 1000))
    if mask.any():
        # approximate sample mask from frame mask
        sample_mask = np.zeros(x.size, dtype=bool)
        for i, m in enumerate(mask):
            if not m:
                continue
            a = i * hop
            b = min(x.size, a + n)
            sample_mask[a:b] = True
        speech = x[sample_mask]
        if speech.size < 8:
            speech = x
    else:
        speech = x
    rms = float(np.sqrt(np.mean(speech.astype(np.float64) ** 2) + EPS))
    if rms < 1e-8:
        return x
    return (x * (target_rms / rms)).astype(np.float32)


def mel_speech_mean(wav: np.ndarray, mask: np.ndarray, sr: int) -> np.ndarray:
    """log-Mel mean over speech frames only; fallback to all frames if none."""
    S = librosa.feature.melspectrogram(
        y=wav.astype(np.float32),
        sr=sr,
        n_mels=N_MELS,
        n_fft=1024,
        hop_length=max(1, int(sr * HOP_MS / 1000)),
        power=2.0,
    )
    logS = librosa.power_to_db(S, ref=np.max)  # [n_mels, T_f]
    t = logS.shape[1]
    if mask.size != t:
        # resample mask to mel frames
        idx = (np.linspace(0, max(mask.size - 1, 0), t)).astype(np.int64) if mask.size else np.zeros(t, np.int64)
        m = mask[idx] if mask.size else np.ones(t, dtype=bool)
    else:
        m = mask
    if m.any():
        return logS[:, m].mean(axis=1).astype(np.float32)
    return logS.mean(axis=1).astype(np.float32)


def extract_turn_audio(wav: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
    mask, fr = speech_mask(wav, sr)
    stats = pause_stats(mask, sr)
    snr = estimate_snr_db(fr, mask)
    q = soft_audio_quality(stats["speech_dur"], snr)
    wav_n = speech_rms_normalize(wav, mask, sr)
    # recompute mask on normalized? keep original mask for pause semantics
    mel = mel_speech_mean(wav_n, mask, sr)

    speech = wav_n
    if mask.any():
        hop = max(1, int(sr * HOP_MS / 1000))
        n = max(1, int(sr * FRAME_MS / 1000))
        sample_mask = np.zeros(wav_n.size, dtype=bool)
        for i, m in enumerate(mask):
            if m:
                a = i * hop
                b = min(wav_n.size, a + n)
                sample_mask[a:b] = True
        if sample_mask.any():
            speech = wav_n[sample_mask]
    rms_db = 20.0 * np.log10(float(np.sqrt(np.mean(speech.astype(np.float64) ** 2) + EPS)))

    pause_vec = np.array(
        [
            stats["pause_ratio"],
            stats["speech_ratio"],
            min(stats["speech_dur"] / 10.0, 1.0),  # normalize ~[0,1]
            min(stats["max_pause_sec"] / 5.0, 1.0),
            min(stats["mean_pause_sec"] / 2.0, 1.0),
            min(stats["n_pauses"] / 10.0, 1.0),
            float(np.clip(snr / 20.0, 0.0, 1.0)),
            float(np.clip((rms_db + 60.0) / 60.0, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )
    assert pause_vec.size == PAUSE_FEAT_DIM
    return {
        "mel": mel,
        "pause_feats": pause_vec,
        "audio_quality": np.float32(q),
        "snr_db": np.float32(snr),
        "speech_dur": np.float32(stats["speech_dur"]),
        "pause_ratio": np.float32(stats["pause_ratio"]),
    }
