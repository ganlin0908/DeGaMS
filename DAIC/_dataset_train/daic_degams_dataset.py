"""DAIC dataset for DeGaMS-aligned training (MentalBERT + Mel + CLNF visual).

Supports Audibert-style depression-keyword utterance filtering and hybrid retention
(keyword must-keep + additional non-keyword Participant turns).
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(os.environ.get("DEGAMS_ROOT", r"/path/to/degams_data"))
# Default: cleaned v2 (Ellie-free text, OpenFace-210) + VAD-pooled wav2vec2
# Override with DEGAMS_FEATURE_ROOT (e.g. .../degams_avboost_v1/daic) for isolated AV-boost.
_DEFAULT_CLEAN = ROOT / "egemaps_experiment" / "artifacts" / "degams_clean_v2" / "daic"
_feat_root = os.environ.get("DEGAMS_FEATURE_ROOT")
CLEAN_V2 = Path(_feat_root) if _feat_root else _DEFAULT_CLEAN
TEXT_DIR = CLEAN_V2 / "text"
MEL_DIR = CLEAN_V2 / "mel"  # pause / audio_quality reference
VIS_DIR = CLEAN_V2 / "visual"
# Dense / enhanced packs live under clean_v2 but are ONLY used when DEGAMS_AVBOOST=1
# (or when FEATURE_ROOT already points at avboost tree where visual/wav2vec ARE boosted).
_AVBOOST = (
    os.environ.get("DEGAMS_AVBOOST") or ""
).strip() in ("1", "true", "True", "yes")
_USING_FEATURE_ROOT = bool(
    os.environ.get("DEGAMS_FEATURE_ROOT")
)
VIS_DENSE_DIR = (
    (CLEAN_V2 / "visual_dense")
    if (_AVBOOST and not _USING_FEATURE_ROOT)
    else (CLEAN_V2 / "_no_visual_dense")
)
W2V_CLEAN_DIR = CLEAN_V2 / "wav2vec"  # VAD speech-frame pooled (or enh when FEATURE_ROOT=avboost)
W2V_ENH_DIR = (
    (CLEAN_V2 / "wav2vec_enh")
    if (_AVBOOST and not _USING_FEATURE_ROOT)
    else (CLEAN_V2 / "_no_wav2vec_enh")
)
W2V_RAW_DIR = ROOT / "egemaps_experiment" / "artifacts" / "wav2vec_embeds" / "daic"
OLD_FEAT_DIR = ROOT / "egemaps_experiment" / "1_feature_build" / "features" / "daic"
LABEL = ROOT / "DAICWOZ" / "label"
AUDIO_DIM = 768
MEL_SPEC_DIM = 80
PAUSE_DIM = 8
MEL_DIM = AUDIO_DIM  # batch key still "mel"; dim is wav2vec
OPENFACE_CORE_DIM = 29
HI_VISUAL_DIM = 768
VISUAL_DENSE_DIM = OPENFACE_CORE_DIM + HI_VISUAL_DIM


PHQ_ITEMS = [
    "PHQ8_NoInterest",
    "PHQ8_Depressed",
    "PHQ8_Sleep",
    "PHQ8_Tired",
    "PHQ8_Appetite",
    "PHQ8_Failure",
    "PHQ8_Concentrating",
    "PHQ8_Moving",
]

# Approximate Audibert / Toto-style depression keywords (substring match, lowercased).
DEPRESSION_KEYWORDS: Sequence[str] = (
    "sleep",
    "slept",
    "insomni",
    "sad",
    "depress",
    "anxiet",
    "anxious",
    "worry",
    "worried",
    "hopeless",
    "helpless",
    "worthless",
    "guilt",
    "guilty",
    "tired",
    "fatigue",
    "exhaust",
    "energy",
    "interest",
    "anhedon",
    "suicid",
    "kill myself",
    "death",
    "die",
    "dying",
    "appetite",
    "hungry",
    "weight",
    "concentrat",
    "focus",
    "lonely",
    "alone",
    "cry",
    "crying",
    "tears",
    "empty",
    "numb",
    "irritab",
    "angry",
    "anger",
    "motivat",
    "useless",
    "failure",
    "fail",
    "burden",
    "self-harm",
    "self harm",
    "hurt myself",
    "nervous",
    "panic",
    "stress",
    "overwhelm",
    "restless",
    "slow",
    "moving",
    "pleasure",
    "enjoy",
)

# Visual source kind codes (session-level).
VIS_KIND_MISSING = 0
VIS_KIND_CLNF = 1
VIS_KIND_OPENFACE = 2


def _load_visual_pack(sid: str, n_raw: int) -> tuple[torch.Tensor, torch.Tensor, bool, int]:
    """Load visual turns from dense dir (optional) or VIS_DIR (OF-210 or 797)."""
    dense_path = VIS_DENSE_DIR / f"{sid}.npz"
    vpath = VIS_DIR / f"{sid}.npz"
    force_of210 = os.environ.get("DEGAMS_FORCE_OF210", "").strip() in (
        "1",
        "true",
        "True",
        "yes",
    )

    def _from_npz(path: Path) -> tuple[torch.Tensor, torch.Tensor, bool, int]:
        vp = np.load(path, allow_pickle=True)
        v = torch.from_numpy(np.asarray(vp["visual"], dtype=np.float32))
        present_v = bool(vp["present"]) if "present" in vp.files else True
        if "visual_turn_mask" in vp.files:
            valid = torch.from_numpy(np.asarray(vp["visual_turn_mask"], dtype=np.bool_))
        elif "n_frames_per_turn" in vp.files:
            nf = torch.from_numpy(np.asarray(vp["n_frames_per_turn"], dtype=np.int32))
            valid = nf > 0
        else:
            valid = (
                torch.from_numpy(np.abs(np.asarray(vp["visual"])).sum(axis=1) > 0)
                if v.ndim == 2
                else torch.zeros(0, dtype=torch.bool)
            )
        kind = VIS_KIND_OPENFACE if present_v else VIS_KIND_MISSING
        if "source" in vp.files:
            src = str(
                np.asarray(vp["source"]).item()
                if np.asarray(vp["source"]).shape == ()
                else vp["source"]
            )
            if src.startswith("error") or src == "missing":
                kind = VIS_KIND_MISSING
                present_v = False
        dim = int(v.size(1)) if v.ndim == 2 else 210
        t_v = min(n_raw, v.size(0), int(valid.size(0)) if valid.numel() else v.size(0))
        visual = torch.zeros(n_raw, dim)
        visual_valid = torch.zeros(n_raw, dtype=torch.bool)
        visual[:t_v] = v[:t_v]
        if valid.numel():
            visual_valid[:t_v] = valid[:t_v]
        else:
            visual_valid[:t_v] = visual[:t_v].abs().sum(dim=-1) > 0
        if dim >= OPENFACE_CORE_DIM:
            visual_valid = visual_valid & (
                visual[:, :OPENFACE_CORE_DIM].abs().sum(dim=-1) > 0
            )
        if not present_v:
            kind = VIS_KIND_MISSING
            visual_valid.zero_()
        return visual, visual_valid, present_v, kind

    if (not force_of210) and dense_path.is_file():
        return _from_npz(dense_path)
    if vpath.is_file():
        return _from_npz(vpath)
    return (
        torch.zeros(n_raw, 210),
        torch.zeros(n_raw, dtype=torch.bool),
        False,
        VIS_KIND_MISSING,
    )

def _compose_mel_with_pause(mel_pack: dict) -> torch.Tensor:
    """Legacy Mel path [T,88]; kept for tools. Training uses ``_load_wav2vec_audio``."""
    mel = mel_pack["mel_turns"].float()
    if "pause_feats" in mel_pack and "audio_quality" in mel_pack:
        pause = mel_pack["pause_feats"].float()
        q = mel_pack["audio_quality"].float().view(-1, 1).clamp(0.05, 1.0)
        t = min(mel.size(0), pause.size(0), q.size(0))
        mel, pause, q = mel[:t], pause[:t], q[:t]
        return torch.cat([mel * q, pause], dim=-1)
    pad = torch.zeros(mel.size(0), PAUSE_DIM, dtype=mel.dtype)
    return torch.cat([mel, pad], dim=-1)


def _load_wav2vec_audio(sid: str, mel_pack: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """wav2vec2 turns [T,768] + per-turn quality [T] in [0,1]. Prefer enhanced → VAD-clean."""
    for cand in (W2V_ENH_DIR / f"{sid}.pt", W2V_CLEAN_DIR / f"{sid}.pt"):
        if cand.is_file():
            wpack = torch.load(cand, map_location="cpu", weights_only=False)
            at = wpack["audio_turns"].float()
            if "audio_quality" in wpack:
                q = wpack["audio_quality"].float().view(-1).clamp(0.05, 1.0)
            elif "speech_frac" in wpack:
                q = wpack["speech_frac"].float().view(-1).clamp(0.05, 1.0)
            else:
                q = torch.ones(at.size(0), dtype=torch.float32)
            t = min(at.size(0), q.size(0))
            return at[:t], q[:t]

    # Fallback: raw embeds + Ellie filter + soft quality scale
    wpack = torch.load(W2V_RAW_DIR / f"{sid}.pt", map_location="cpu", weights_only=False)
    at = wpack["audio_turns"].float()
    old_path = OLD_FEAT_DIR / f"{sid}_features.pt"
    if old_path.is_file():
        old = torch.load(old_path, map_location="cpu", weights_only=False)
        meta = list(old.get("meta_data") or [])
        if meta and len(meta) == at.size(0):
            idx = [
                i
                for i, m in enumerate(meta)
                if str(m.get("speaker") or "").lower() == "participant"
            ]
            if idx:
                at = at.index_select(0, torch.tensor(idx, dtype=torch.long))
    if mel_pack is not None and "audio_quality" in mel_pack:
        q = mel_pack["audio_quality"].float().view(-1).clamp(0.05, 1.0)
        t = min(at.size(0), q.size(0))
        at = at[:t] * q[:t].unsqueeze(-1)
        return at, q[:t]
    return at, torch.ones(at.size(0), dtype=torch.float32)


def _keyword_pattern(keywords: Sequence[str] = DEPRESSION_KEYWORDS) -> re.Pattern:
    """Build regex: short tokens use word boundaries; longer stems allow substring."""
    parts = []
    for k in keywords:
        esc = re.escape(k)
        if " " in k or len(k) >= 5:
            parts.append(esc)
        else:
            parts.append(rf"\b{esc}\b")
    return re.compile("|".join(parts), re.IGNORECASE)


_KW_RE = _keyword_pattern()


def turn_has_depression_keyword(text: str) -> bool:
    if not text:
        return False
    return _KW_RE.search(text) is not None


def select_keyword_turn_indices(
    meta: List[dict],
    *,
    min_keep: int = 1,
    fallback_top_n: int = 8,
) -> List[int]:
    """Keep turns whose transcript matches a depression keyword.

    If none match, fall back to the longest ``fallback_top_n`` turns (by char length)
    so the sample remains usable.
    """
    hits = [i for i, t in enumerate(meta) if turn_has_depression_keyword(str(t.get("text") or ""))]
    if len(hits) >= min_keep:
        return hits
    scored = sorted(
        range(len(meta)),
        key=lambda i: len(str(meta[i].get("text") or "")),
        reverse=True,
    )
    n = min(max(fallback_top_n, min_keep), len(meta))
    return sorted(scored[:n]) if meta else []


def select_hybrid_turn_indices(
    meta: List[dict],
    *,
    target_keep: int = 48,
    min_keep: int = 1,
    fallback_top_n: int = 16,
    min_non_kw_chars: int = 20,
    min_frac_of_raw: float = 0.25,
) -> List[int]:
    """Hybrid retention: keyword turns MUST keep, then fill with non-keyword turns.

    Policy (documented):
      - Always keep every depression-keyword hit.
      - Fill remaining quota up to ``max(target_keep, ceil(min_frac_of_raw * n_raw))``
        with the longest non-keyword turns (char length >= ``min_non_kw_chars``),
        preferring mid-interview positions on length ties (absolute distance to center).
      - If zero keyword hits: fall back to longest ``fallback_top_n`` turns.
      - Empty meta → empty list (caller handles).

    Target avg kept turns is roughly 30–60 (default target_keep=48) rather than ~10.
    """
    n = len(meta)
    if n == 0:
        return []

    kw_hits = [i for i, t in enumerate(meta) if turn_has_depression_keyword(str(t.get("text") or ""))]
    if not kw_hits:
        scored = sorted(
            range(n),
            key=lambda i: len(str(meta[i].get("text") or "")),
            reverse=True,
        )
        k = min(max(fallback_top_n, min_keep), n)
        return sorted(scored[:k])

    quota = max(int(target_keep), int(np.ceil(min_frac_of_raw * n)), len(kw_hits), min_keep)
    quota = min(quota, n)
    remaining = [i for i in range(n) if i not in set(kw_hits)]
    center = 0.5 * (n - 1)

    def _extra_key(i: int):
        text = str(meta[i].get("text") or "")
        L = len(text)
        # Longer first; on ties prefer closer to interview center (more contentful middle).
        return (L, -abs(i - center))

    candidates = [
        i
        for i in remaining
        if len(str(meta[i].get("text") or "")) >= min_non_kw_chars
    ]
    candidates.sort(key=_extra_key, reverse=True)
    n_extra = max(0, quota - len(kw_hits))
    extras = candidates[:n_extra]
    # If still short (very short transcripts), relax char floor.
    if len(kw_hits) + len(extras) < min(quota, n):
        used = set(kw_hits) | set(extras)
        more = sorted(
            [i for i in remaining if i not in used],
            key=_extra_key,
            reverse=True,
        )
        extras = extras + more[: max(0, quota - len(kw_hits) - len(extras))]
    return sorted(set(kw_hits) | set(extras))


def _read_split(split: str) -> Dict[str, dict]:
    """Return sid -> label fields."""
    out: Dict[str, dict] = {}
    if split == "train":
        path = LABEL / "train_split_Depression_AVEC2017.csv"
    elif split == "dev":
        path = LABEL / "dev_split_Depression_AVEC2017.csv"
    elif split == "test":
        path = LABEL / "full_test_split.csv"
    else:
        raise ValueError(split)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        sid = str(r.get("Participant_ID") or r.get("participant_ID")).strip()
        if split == "test":
            y = int(float(r["PHQ_Binary"]))
            score = float(r["PHQ_Score"])
            items = None
        else:
            y = int(float(r["PHQ8_Binary"]))
            score = float(r["PHQ8_Score"])
            items = []
            for k in PHQ_ITEMS:
                v = (r.get(k) or "").strip()
                items.append(int(float(v)) if v != "" else 0)
        out[sid] = {"y": y, "phq_total": score, "phq_items": items}
    return out


def _parse_visual_kind(source) -> int:
    s = str(source).lower() if source is not None else ""
    if "clnf" in s or "zip" in s:
        return VIS_KIND_CLNF
    if "openface" in s or "tar" in s:
        return VIS_KIND_OPENFACE
    return VIS_KIND_MISSING


class DAICDeGaMSDataset(Dataset):
    def __init__(
        self,
        split: str,
        max_len: int = 128,
        keyword_filter: bool = True,
        filter_mode: str = "hybrid",
        hybrid_target_keep: int = 48,
        hybrid_min_frac: float = 0.25,
        require_visual: bool = False,
    ) -> None:
        """
        filter_mode:
          - "none": keep all turns (up to max_len)
          - "keyword": keyword-only (legacy ~10 turns)
          - "hybrid": keyword must-keep + longest non-keyword fill (default ABDE)
        """
        self.split = split
        self.max_len = max_len
        self.keyword_filter = keyword_filter and filter_mode != "none"
        self.filter_mode = "none" if not keyword_filter else filter_mode
        self.hybrid_target_keep = hybrid_target_keep
        self.hybrid_min_frac = hybrid_min_frac
        self.labels = _read_split(split)
        self.ids = sorted(
            sid
            for sid in self.labels
            if (TEXT_DIR / f"{sid}_features.pt").is_file()
            and (
                (W2V_CLEAN_DIR / f"{sid}.pt").is_file()
                or (W2V_RAW_DIR / f"{sid}.pt").is_file()
            )
            and (not require_visual or (VIS_DIR / f"{sid}.npz").is_file())
        )
        self._cache: List[dict] = [self._load(sid) for sid in self.ids]
        self.filter_stats = self._summarize_filter_stats()

    def _summarize_filter_stats(self) -> dict:
        if not self._cache:
            return {"n": 0}
        n_raw = [int(x["n_turns_raw"]) for x in self._cache]
        n_kept = [int(x["n_turns_kept"]) for x in self._cache]
        n_kw = [int(x["n_keyword_hits"]) for x in self._cache]
        n_fallback = sum(1 for x in self._cache if x["keyword_fallback"])
        kinds = [int(x["visual_kind"].item()) for x in self._cache]
        return {
            "n": len(self._cache),
            "keyword_filter": self.keyword_filter,
            "filter_mode": self.filter_mode,
            "hybrid_target_keep": self.hybrid_target_keep,
            "hybrid_min_frac": self.hybrid_min_frac,
            "avg_turns_raw": float(np.mean(n_raw)),
            "avg_turns_kept": float(np.mean(n_kept)),
            "median_turns_kept": float(np.median(n_kept)),
            "avg_keyword_hits": float(np.mean(n_kw)),
            "n_keyword_fallback": n_fallback,
            "frac_fallback": n_fallback / max(len(self._cache), 1),
            "n_vis_clnf": sum(1 for k in kinds if k == VIS_KIND_CLNF),
            "n_vis_openface": sum(1 for k in kinds if k == VIS_KIND_OPENFACE),
            "n_vis_missing": sum(1 for k in kinds if k == VIS_KIND_MISSING),
        }

    def __len__(self) -> int:
        return len(self._cache)

    def _load(self, sid: str) -> dict:
        text_pack = torch.load(TEXT_DIR / f"{sid}_features.pt", map_location="cpu", weights_only=False)
        mel_pack = None
        mel_path = MEL_DIR / f"{sid}.pt"
        if mel_path.is_file():
            mel_pack = torch.load(mel_path, map_location="cpu", weights_only=False)
        text = text_pack["text_features"].float()
        mel, audio_quality = _load_wav2vec_audio(sid, mel_pack)
        meta = list(text_pack.get("meta_data") or [])
        n_raw = min(
            text.size(0),
            mel.size(0),
            audio_quality.size(0),
            len(meta) if meta else text.size(0),
        )
        text, mel, audio_quality = text[:n_raw], mel[:n_raw], audio_quality[:n_raw]
        meta = meta[:n_raw]

        visual, visual_valid, present_v, visual_kind = _load_visual_pack(sid, n_raw)

        n_keyword_hits = sum(1 for t in meta if turn_has_depression_keyword(str(t.get("text") or "")))
        keyword_fallback = False
        if self.filter_mode == "hybrid" and meta:
            idx = select_hybrid_turn_indices(
                meta,
                target_keep=self.hybrid_target_keep,
                min_frac_of_raw=self.hybrid_min_frac,
            )
            keyword_fallback = n_keyword_hits == 0
            idx_t = torch.tensor(idx, dtype=torch.long)
            text = text.index_select(0, idx_t)
            mel = mel.index_select(0, idx_t)
            audio_quality = audio_quality.index_select(0, idx_t)
            visual = visual.index_select(0, idx_t)
            visual_valid = visual_valid.index_select(0, idx_t)
        elif self.filter_mode == "keyword" and meta:
            idx = select_keyword_turn_indices(meta)
            keyword_fallback = n_keyword_hits == 0
            idx_t = torch.tensor(idx, dtype=torch.long)
            text = text.index_select(0, idx_t)
            mel = mel.index_select(0, idx_t)
            audio_quality = audio_quality.index_select(0, idx_t)
            visual = visual.index_select(0, idx_t)
            visual_valid = visual_valid.index_select(0, idx_t)

        t = min(
            text.size(0),
            mel.size(0),
            visual.size(0),
            audio_quality.size(0),
            visual_valid.size(0),
            self.max_len,
        )
        text, mel, visual, audio_quality = text[:t], mel[:t], visual[:t], audio_quality[:t]
        visual_valid = visual_valid[:t]
        lab = self.labels[sid]
        items = lab["phq_items"]
        if items is None:
            items_t = torch.zeros(8, dtype=torch.long)
        else:
            items_t = torch.tensor(items, dtype=torch.long)
        return {
            "sid": sid,
            "text": text,
            "mel": mel,
            "visual": visual,
            "audio_quality": audio_quality,
            "mask": torch.ones(t, dtype=torch.bool),
            "visual_valid": visual_valid,
            "present_v": torch.tensor(present_v, dtype=torch.bool),
            "visual_kind": torch.tensor(visual_kind, dtype=torch.long),
            "y": torch.tensor(lab["y"], dtype=torch.long),
            "phq_total": torch.tensor(lab["phq_total"], dtype=torch.float32),
            "phq_items": items_t,
            "has_items": torch.tensor(items is not None, dtype=torch.bool),
            "n_turns_raw": n_raw,
            "n_turns_kept": t,
            "n_keyword_hits": n_keyword_hits,
            "keyword_fallback": keyword_fallback,
        }

    def __getitem__(self, i: int) -> dict:
        return self._cache[i]


def collate_degams(batch: List[dict]) -> dict:
    bsz = len(batch)
    t_len = max(x["text"].size(0) for x in batch)
    td, md, vd = batch[0]["text"].size(1), batch[0]["mel"].size(1), batch[0]["visual"].size(1)
    text = torch.zeros(bsz, t_len, td)
    mel = torch.zeros(bsz, t_len, md)
    visual = torch.zeros(bsz, t_len, vd)
    audio_quality = torch.ones(bsz, t_len)
    mask = torch.zeros(bsz, t_len, dtype=torch.bool)
    visual_valid = torch.zeros(bsz, t_len, dtype=torch.bool)
    present_v = torch.zeros(bsz, dtype=torch.bool)
    visual_kind = torch.zeros(bsz, dtype=torch.long)
    y = torch.zeros(bsz, dtype=torch.long)
    phq_total = torch.zeros(bsz)
    phq_items = torch.zeros(bsz, 8, dtype=torch.long)
    has_items = torch.zeros(bsz, dtype=torch.bool)
    sids = []
    for i, x in enumerate(batch):
        t = x["text"].size(0)
        text[i, :t] = x["text"]
        mel[i, :t] = x["mel"]
        visual[i, :t] = x["visual"]
        if "audio_quality" in x:
            q = x["audio_quality"].float().view(-1)
            audio_quality[i, : min(t, q.size(0))] = q[:t]
        mask[i, :t] = True
        if "visual_valid" in x:
            vv = x["visual_valid"].bool().view(-1)
            visual_valid[i, : min(t, vv.size(0))] = vv[:t]
        else:
            visual_valid[i, :t] = True
        present_v[i] = x["present_v"]
        visual_kind[i] = x["visual_kind"]
        y[i] = x["y"]
        phq_total[i] = x["phq_total"]
        phq_items[i] = x["phq_items"]
        has_items[i] = x["has_items"]
        sids.append(x["sid"])
    return {
        "text": text,
        "mel": mel,
        "visual": visual,
        "audio_quality": audio_quality,
        "mask": mask,
        "visual_valid": visual_valid,
        "present_v": present_v,
        "visual_kind": visual_kind,
        "y": y,
        "phq_total": phq_total,
        "phq_items": phq_items,
        "has_items": has_items,
        "sids": sids,
    }
