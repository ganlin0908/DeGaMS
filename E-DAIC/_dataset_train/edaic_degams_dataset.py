"""E-DAIC dataset for DeGaMS-aligned training (clean_v2).

Mirrors ``daic_degams_dataset.py`` protocol:
  - hybrid / keyword turn filtering (English transcripts)
  - soft PHQ-8 items for train/dev (from detailed_lables.csv); test has_items=False
  - visual: OpenFace Pose/Gaze/AU_r → 210-d (same as DAIC clean_v2)
  - audio: VAD-pooled wav2vec2 [T,768] (fallback: raw wav2vec + soft quality)

Anti-leakage: TEST item labels are never loaded even if present in CSV.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from daic_degams_dataset import (
    VIS_KIND_MISSING,
    VIS_KIND_OPENFACE,
    collate_degams,
    select_hybrid_turn_indices,
    select_keyword_turn_indices,
    turn_has_depression_keyword,
)

ROOT = Path(os.environ.get("DEGAMS_ROOT", r"/path/to/degams_data"))
_DEFAULT_CLEAN = ROOT / "egemaps_experiment" / "artifacts" / "degams_clean_v2" / "edaic"
_feat_root = os.environ.get("DEGAMS_FEATURE_ROOT")
CLEAN_V2 = Path(_feat_root) if _feat_root else _DEFAULT_CLEAN
TEXT_DIR = CLEAN_V2 / "text"
MEL_DIR = CLEAN_V2 / "mel"
VIS_DIR = CLEAN_V2 / "visual"
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
W2V_CLEAN_DIR = CLEAN_V2 / "wav2vec"
W2V_ENH_DIR = (
    (CLEAN_V2 / "wav2vec_enh")
    if (_AVBOOST and not _USING_FEATURE_ROOT)
    else (CLEAN_V2 / "_no_wav2vec_enh")
)
W2V_RAW_DIR = ROOT / "egemaps_experiment" / "artifacts" / "wav2vec_embeds" / "edaic"
LABEL = ROOT / "EDAIC" / "label"
OPENFACE_CORE_DIM = 29


# detailed_lables.csv column names → ordered PHQ-8 items
PHQ_ITEM_COLS = [
    "PHQ8_1_NoInterest",
    "PHQ8_2_Depressed",
    "PHQ8_3_Sleep",
    "PHQ8_4_Tired",
    "PHQ8_5_Appetite",
    "PHQ8_6_Failure",
    "PHQ8_7_Concentration",
    "PHQ8_8_Psychomotor",
]


def _text_dir() -> Path:
    return TEXT_DIR


def _read_detailed_items() -> Dict[str, List[int]]:
    """Participant → 8 PHQ item scores (all splits present in CSV; caller gates usage)."""
    path = LABEL / "detailed_lables.csv"
    out: Dict[str, List[int]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            sid = str(r.get("Participant") or "").strip()
            if not sid:
                continue
            items = []
            for k in PHQ_ITEM_COLS:
                v = (r.get(k) or "").strip()
                items.append(int(float(v)) if v != "" else 0)
            out[sid] = items
    return out


def _read_split(split: str) -> Dict[str, dict]:
    """Return sid -> label fields. Soft PHQ items only for train/dev."""
    if split == "train":
        path = LABEL / "train_split.csv"
    elif split == "dev":
        path = LABEL / "dev_split.csv"
    elif split == "test":
        path = LABEL / "test_split.csv"
    else:
        raise ValueError(split)

    items_all = _read_detailed_items() if split in ("train", "dev") else {}
    out: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        sid = str(r.get("Participant_ID") or r.get("participant_ID")).strip()
        y = int(float(r["PHQ_Binary"]))
        score = float(r["PHQ_Score"])
        items = items_all.get(sid) if split in ("train", "dev") else None
        out[sid] = {"y": y, "phq_total": score, "phq_items": items}
    return out


class EDAICDeGaMSDataset(Dataset):
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
        self.split = split
        self.max_len = max_len
        self.keyword_filter = keyword_filter and filter_mode != "none"
        self.filter_mode = "none" if not keyword_filter else filter_mode
        self.hybrid_target_keep = hybrid_target_keep
        self.hybrid_min_frac = hybrid_min_frac
        self.labels = _read_split(split)
        self.text_dir = _text_dir()
        self.ids = sorted(
            sid
            for sid in self.labels
            if (self.text_dir / f"{sid}_features.pt").is_file()
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
            "n_vis_openface": sum(1 for k in kinds if k == VIS_KIND_OPENFACE),
            "n_vis_missing": sum(1 for k in kinds if k == VIS_KIND_MISSING),
            "text_dir": str(self.text_dir),
            "mel_dir": str(MEL_DIR),
            "vis_dir": str(VIS_DIR),
        }

    def __len__(self) -> int:
        return len(self._cache)

    def _load_wav2vec(self, sid: str, mel_pack: dict | None) -> tuple[torch.Tensor, torch.Tensor]:
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
        wpack = torch.load(W2V_RAW_DIR / f"{sid}.pt", map_location="cpu", weights_only=False)
        at = wpack["audio_turns"].float()
        if mel_pack is not None and "audio_quality" in mel_pack:
            q = mel_pack["audio_quality"].float().view(-1).clamp(0.05, 1.0)
            t = min(at.size(0), q.size(0))
            at = at[:t] * q[:t].unsqueeze(-1)
            return at, q[:t]
        return at, torch.ones(at.size(0), dtype=torch.float32)

    def _load_visual(self, sid: str, n_raw: int) -> tuple[torch.Tensor, torch.Tensor, bool, int]:
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
            dim = int(v.size(1)) if v.ndim == 2 else 210
            t_v = min(
                n_raw, v.size(0), int(valid.size(0)) if valid.numel() else v.size(0)
            )
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
    def _load(self, sid: str) -> dict:
        text_pack = torch.load(
            self.text_dir / f"{sid}_features.pt", map_location="cpu", weights_only=False
        )
        mel_pack = None
        mel_path = MEL_DIR / f"{sid}.pt"
        if mel_path.is_file():
            mel_pack = torch.load(mel_path, map_location="cpu", weights_only=False)
        text = text_pack["text_features"].float()
        mel, audio_quality = self._load_wav2vec(sid, mel_pack)
        meta = list(text_pack.get("meta_data") or [])
        n_raw = min(
            text.size(0),
            mel.size(0),
            audio_quality.size(0),
            len(meta) if meta else text.size(0),
        )
        text, mel, audio_quality = text[:n_raw], mel[:n_raw], audio_quality[:n_raw]
        meta = meta[:n_raw]

        visual, visual_valid, present_v, visual_kind = self._load_visual(sid, n_raw)

        n_keyword_hits = sum(
            1 for t in meta if turn_has_depression_keyword(str(t.get("text") or ""))
        )
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


__all__ = ["EDAICDeGaMSDataset", "collate_degams", "VIS_KIND_DENSENET"]
