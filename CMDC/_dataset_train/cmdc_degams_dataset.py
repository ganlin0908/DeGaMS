"""CMDC TAV dataset for DeGaMS Group C (video_fold + PHQ-1..8).

Anti-leakage / anti-saturation notes
------------------------------------
- Fold CSV labels are the protocol ground truth for y / PHQ_Score.
  SubjectInfo supplies PHQ-1..8 items only (train/dev); test has_items=False.
- Construction asserts train∩dev∩test = ∅ for the active fold.
- CSV PHQ_Binary / PHQ_Score are cross-checked against SubjectInfo when present.
"""
from __future__ import annotations

import csv
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from daic_degams_dataset import (  # noqa: F401
    PAUSE_DIM,
    VIS_KIND_MISSING,
    VIS_KIND_OPENFACE,
    collate_degams,
)

ROOT = Path(os.environ.get("DEGAMS_ROOT", r"/path/to/degams_data"))
_DEFAULT = ROOT / "egemaps_experiment" / "artifacts" / "degams_cmdc_tav"
_feat_root = os.environ.get("DEGAMS_FEATURE_ROOT")
CLEAN = Path(_feat_root) if _feat_root else _DEFAULT
TEXT_DIR = CLEAN / "text"
VIS_DIR = CLEAN / "visual"
W2V_DIR = CLEAN / "wav2vec"
LABEL = ROOT / "CMDC" / "label"
SUBJECT_INFO = ROOT / "CMDC" / "raw" / "extracted" / "SubjectInfo.xlsx"
PHQ_ITEM_COLS = [f"PHQ-{i}" for i in range(1, 9)]
VIS_DIM = 797


def _fold() -> int:
    return int(os.environ.get("DEGAMS_CMDC_FOLD") or "1")


def _read_phq_items() -> Dict[str, dict]:
    info = pd.read_excel(SUBJECT_INFO)
    ids = info["ID"].astype(str).tolist()
    if ids.count("MDD20") > 1:
        dup = [i for i, x in enumerate(ids) if x == "MDD20"]
        info.at[dup[1], "ID"] = "MDD21"
    out: Dict[str, dict] = {}
    for _, row in info.iterrows():
        sid = str(row["ID"]).strip()
        items = []
        for c in PHQ_ITEM_COLS:
            v = row.get(c)
            try:
                items.append(int(float(v)) if pd.notna(v) else 0)
            except Exception:
                items.append(0)
        try:
            total = int(float(row.get("PHQtotal") or sum(items)))
        except Exception:
            total = int(sum(items))
        try:
            y = int(float(row.get("MDD") or 0))
        except Exception:
            y = 1 if total >= 10 else 0
        out[sid] = {"y": y, "phq_total": total, "phq_items": items}
    return out


_PHQ_CACHE: Optional[Dict[str, dict]] = None


def _phq_all() -> Dict[str, dict]:
    global _PHQ_CACHE
    if _PHQ_CACHE is None:
        _PHQ_CACHE = _read_phq_items()
    return _PHQ_CACHE


def _read_split(split: str, fold: int) -> Dict[str, dict]:
    """Load fold CSV as protocol labels; SubjectInfo only for soft PHQ items."""
    path = LABEL / f"video_fold{fold}" / f"{split}_split.csv"
    phq = _phq_all()
    out: Dict[str, dict] = {}
    mismatches: List[str] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            sid = str(row.get("Participant_ID") or "").strip()
            if not sid:
                continue
            y = int(float(row.get("PHQ_Binary") or 0))
            score = int(float(row.get("PHQ_Score") or 0))
            base = phq.get(sid)
            if base is not None:
                if int(base["y"]) != y:
                    mismatches.append(f"{sid}: CSV_y={y} SubjectInfo_MDD={base['y']}")
                if int(base["phq_total"]) != score:
                    mismatches.append(
                        f"{sid}: CSV_PHQ={score} SubjectInfo_PHQ={base['phq_total']}"
                    )
                items = list(base["phq_items"])
            else:
                items = [0] * 8
                warnings.warn(f"[CMDC] {sid} missing in SubjectInfo; items zeroed", stacklevel=2)
            use_items = split in ("train", "dev")
            out[sid] = {
                "y": y,
                "phq_total": score,
                "phq_items": items if use_items else None,
            }
    if mismatches:
        raise RuntimeError(
            f"CMDC fold{fold}/{split} label mismatch vs SubjectInfo: " + "; ".join(mismatches[:8])
        )
    return out


def assert_fold_disjoint(fold: int) -> dict:
    """Fail fast if train/dev/test share any Participant_ID within a fold."""
    sets: Dict[str, Set[str]] = {}
    for sp in ("train", "dev", "test"):
        path = LABEL / f"video_fold{fold}" / f"{sp}_split.csv"
        with path.open("r", encoding="utf-8", newline="") as f:
            sets[sp] = {
                str(r.get("Participant_ID") or "").strip()
                for r in csv.DictReader(f)
                if str(r.get("Participant_ID") or "").strip()
            }
    overlaps = {
        "train∩dev": sorted(sets["train"] & sets["dev"]),
        "train∩test": sorted(sets["train"] & sets["test"]),
        "dev∩test": sorted(sets["dev"] & sets["test"]),
    }
    bad = {k: v for k, v in overlaps.items() if v}
    if bad:
        raise RuntimeError(f"CMDC fold{fold} split leakage: {bad}")
    return {k: len(v) for k, v in sets.items()}


def _load_visual(sid: str, n: int) -> tuple[torch.Tensor, torch.Tensor, bool, int]:
    path = VIS_DIR / f"{sid}.npz"
    if not path.is_file():
        return (
            torch.zeros(n, VIS_DIM),
            torch.zeros(n, dtype=torch.bool),
            False,
            VIS_KIND_MISSING,
        )
    vp = np.load(path, allow_pickle=True)
    v = torch.from_numpy(np.asarray(vp["visual"], dtype=np.float32))
    if v.dim() == 1:
        v = v.unsqueeze(0)
    if v.size(-1) != VIS_DIM:
        if v.size(-1) < VIS_DIM:
            v = torch.cat([v, torch.zeros(v.size(0), VIS_DIM - v.size(-1))], dim=-1)
        else:
            v = v[..., :VIS_DIM]
    t = min(n, v.size(0))
    v = v[:t]
    if v.size(0) < n:
        v = torch.cat([v, torch.zeros(n - v.size(0), VIS_DIM)], dim=0)
    valid = torch.ones(n, dtype=torch.bool)
    return v, valid, True, VIS_KIND_OPENFACE


class CMDCDeGaMSDataset(Dataset):
    """CMDC video_fold TAV for DeGaMS; filter_mode ignored (keep all Q turns)."""

    def __init__(
        self,
        split: str,
        max_len: int = 128,
        keyword_filter: bool = False,
        filter_mode: str = "none",
        hybrid_target_keep: int = 48,
        hybrid_min_frac: float = 0.25,
        require_visual: bool = True,
        fold: Optional[int] = None,
    ) -> None:
        del keyword_filter, filter_mode, hybrid_target_keep, hybrid_min_frac
        self.split = split
        self.max_len = int(max_len)
        self.filter_mode = "none"
        self.fold = int(fold if fold is not None else _fold())
        self.labels = _read_split(split, self.fold)
        self.ids = sorted(
            sid
            for sid in self.labels
            if (TEXT_DIR / f"{sid}_features.pt").is_file()
            and (W2V_DIR / f"{sid}.pt").is_file()
            and (not require_visual or (VIS_DIR / f"{sid}.npz").is_file())
        )
        self._cache = [self._load(sid) for sid in self.ids]
        self.filter_stats = {
            "n": len(self._cache),
            "filter_mode": "none",
            "fold": self.fold,
            "avg_turns_kept": float(
                np.mean([int(x["n_turns_kept"]) for x in self._cache]) if self._cache else 0.0
            ),
        }

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, i: int) -> dict:
        return self._cache[i]

    def _load(self, sid: str) -> dict:
        text_pack = torch.load(
            TEXT_DIR / f"{sid}_features.pt", map_location="cpu", weights_only=False
        )
        text = text_pack["text_features"].float()
        wpack = torch.load(W2V_DIR / f"{sid}.pt", map_location="cpu", weights_only=False)
        mel = wpack["audio_turns"].float()
        q = wpack.get("audio_quality")
        q = (
            torch.ones(mel.size(0), dtype=torch.float32)
            if q is None
            else q.float().view(-1).clamp(0.05, 1.0)
        )
        n_raw = min(text.size(0), mel.size(0), q.size(0))
        text, mel, q = text[:n_raw], mel[:n_raw], q[:n_raw]
        visual, visual_valid, present_v, visual_kind = _load_visual(sid, n_raw)
        t = min(
            text.size(0),
            mel.size(0),
            visual.size(0),
            q.size(0),
            visual_valid.size(0),
            self.max_len,
        )
        text, mel, visual, q = text[:t], mel[:t], visual[:t], q[:t]
        visual_valid = visual_valid[:t]
        lab = self.labels[sid]
        items = lab["phq_items"]
        items_t = (
            torch.zeros(8, dtype=torch.long)
            if items is None
            else torch.tensor(items, dtype=torch.long)
        )
        return {
            "sid": sid,
            "text": text,
            "mel": mel,
            "visual": visual,
            "audio_quality": q,
            "mask": torch.ones(t, dtype=torch.bool),
            "visual_valid": visual_valid,
            "present_v": torch.tensor(present_v, dtype=torch.bool),
            "visual_kind": torch.tensor(visual_kind, dtype=torch.long),
            "y": torch.tensor(lab["y"], dtype=torch.long),
            "phq_total": torch.tensor(float(lab["phq_total"]), dtype=torch.float32),
            "phq_items": items_t,
            "has_items": torch.tensor(items is not None, dtype=torch.bool),
            "n_turns_raw": n_raw,
            "n_turns_kept": t,
            "n_keyword_hits": 0,
            "keyword_fallback": False,
        }
