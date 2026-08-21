"""
Enrich CMDC *_features.pt with official per-question visual embeddings.

Visual source (corpus release, prefer in order):
  1) Q{n}.npy          shape (768,)
  2) vector_Q{n}.csv   single row / headerless 768 floats

Aligned to existing meta_data[*].question (Q1..Q12).
Only subjects with VideoAvailabe=1 in SubjectInfo.xlsx are enriched
(official 45-subject video subset).

Does not re-extract text/audio. Does not change locked training hyperparameters.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch

ROOT = Path(os.environ.get("DEGAMS_ROOT", r"D:\Depression")) / "CMDC"
EXTRACTED = ROOT / "raw" / "extracted"
FEAT = ROOT / "extracted_features"
SUBJECT_INFO = EXTRACTED / "SubjectInfo.xlsx"
VISUAL_DIM = 768


def video_subject_ids() -> List[str]:
    info = pd.read_excel(SUBJECT_INFO)
    ids = info["ID"].astype(str).tolist()
    if ids.count("MDD20") > 1:
        dup = [i for i, x in enumerate(ids) if x == "MDD20"]
        info.at[dup[1], "ID"] = "MDD21"
    vids = info.loc[info["VideoAvailabe"] == 1, "ID"].astype(str).tolist()
    return sorted(vids, key=lambda x: (x[:3], int(x[3:])))


def load_vector_csv(path: Path) -> Optional[np.ndarray]:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return None
    vals = [float(x) for x in raw.replace("\n", ",").split(",") if x.strip()]
    if len(vals) != VISUAL_DIM:
        return None
    return np.asarray(vals, dtype=np.float32)


def load_question_visual(subject_dir: Path, q: str) -> Optional[np.ndarray]:
    """q like 'Q1'."""
    npy = subject_dir / f"{q}.npy"
    if npy.is_file():
        arr = np.load(npy)
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        if arr.size == VISUAL_DIM:
            return arr
    m = re.fullmatch(r"Q(\d+)", q, flags=re.I)
    if m:
        alt = subject_dir / f"vector_Q{int(m.group(1))}.csv"
        if alt.is_file():
            return load_vector_csv(alt)
    return None


def enrich_one(sid: str) -> str:
    pt_path = FEAT / f"{sid}_features.pt"
    if not pt_path.is_file():
        return f"FAIL {sid}: missing pt"
    subject_dir = EXTRACTED / sid
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    meta = list(payload.get("meta_data") or [])
    text = payload["text_features"]
    t = int(text.size(0))
    if len(meta) != t:
        return f"FAIL {sid}: meta T={len(meta)} != text T={t}"

    rows = []
    missing = 0
    for m in meta:
        q = str(m.get("question") or "")
        vec = load_question_visual(subject_dir, q)
        if vec is None:
            rows.append(np.zeros(VISUAL_DIM, dtype=np.float32))
            missing += 1
        else:
            rows.append(vec)
    visual = torch.from_numpy(np.stack(rows, axis=0))
    payload["visual_features"] = visual.float()
    payload["visual_dim"] = VISUAL_DIM
    payload["visual_source"] = "Q*.npy|vector_Q*.csv"
    payload["visual_missing_turns"] = missing
    torch.save(payload, pt_path)
    return f"OK {sid} visual={tuple(visual.shape)} missing_turns={missing}"


def main() -> int:
    vids = video_subject_ids()
    print(f"Video subjects (VideoAvailabe=1): {len(vids)}")
    ok = fail = 0
    for sid in vids:
        msg = enrich_one(sid)
        print(msg)
        if msg.startswith("OK"):
            ok += 1
        else:
            fail += 1
    print(f"Done ok={ok} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
