"""Drop Ellie / interviewer turns from DAIC text packs → clean_v2/daic/text."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm.auto import tqdm

from clean_v2_paths import DAIC_SRC_TEXT, daic_dirs, ensure_clean_tree


def is_ellie(speaker: str) -> bool:
    sp = (speaker or "").strip().lower()
    return (
        "ellie" in sp
        or "interviewer" in sp
        or sp in {"e", "therapist", "agent"}
    )


def participant_indices(meta: List[dict]) -> List[int]:
    keep = []
    for i, m in enumerate(meta):
        if not is_ellie(str(m.get("speaker") or "")):
            keep.append(i)
    return keep


def clean_pack(pack: dict) -> Tuple[dict, dict]:
    meta = list(pack.get("meta_data") or [])
    text = pack["text_features"]
    n = int(text.size(0))
    if len(meta) != n:
        # assume already participant-only / unusable speaker labels
        meta = [
            {
                **(meta[i] if i < len(meta) else {}),
                "segment_id": i,
                "speaker": "Participant",
                "text": (meta[i].get("text") if i < len(meta) else "") or "",
            }
            for i in range(n)
        ]
        keep = list(range(n))
    else:
        keep = participant_indices(meta)
        if not keep:
            raise RuntimeError("No Participant turns left after Ellie filter")

    idx = torch.tensor(keep, dtype=torch.long)
    new_meta = []
    for j, i in enumerate(keep):
        m = dict(meta[i])
        m["segment_id"] = j
        m["speaker"] = "Participant"
        new_meta.append(m)

    out = {
        "text_features": text.index_select(0, idx).contiguous(),
        "meta_data": new_meta,
        "clean_v2": {
            "ellie_removed": True,
            "n_raw": n,
            "n_kept": len(keep),
            "n_dropped": n - len(keep),
        },
    }
    # preserve non-sequence fields if useful
    for k in ("audio_model", "text_model", "pooling"):
        if k in pack:
            out[k] = pack[k]

    # drop old visual/audio tensors (will be rebuilt)
    stats = {
        "n_raw": n,
        "n_kept": len(keep),
        "n_dropped": n - len(keep),
        "had_ellie": (n - len(keep)) > 0,
    }
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    ensure_clean_tree()
    out_dir = daic_dirs()["text"]
    packs = sorted(DAIC_SRC_TEXT.glob("*_features.pt"))
    if args.limit > 0:
        packs = packs[: args.limit]

    report = {
        "n_total": 0,
        "n_with_ellie": 0,
        "n_dropped_turns": 0,
        "per_sid": {},
        "finished_at": None,
    }
    for p in tqdm(packs, desc="daic-ellie-clean"):
        sid = p.name[: -len("_features.pt")]
        dest = out_dir / f"{sid}_features.pt"
        if dest.is_file() and not args.overwrite:
            report["n_total"] += 1
            continue
        pack = torch.load(p, map_location="cpu", weights_only=False)
        cleaned, stats = clean_pack(pack)
        torch.save(cleaned, dest)
        report["n_total"] += 1
        if stats["had_ellie"]:
            report["n_with_ellie"] += 1
            report["n_dropped_turns"] += stats["n_dropped"]
        report["per_sid"][sid] = stats

    report["finished_at"] = datetime.now().isoformat(timespec="seconds")
    (daic_dirs()["base"] / "_ellie_clean_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(
        f"DAIC Ellie clean: packs={report['n_total']} "
        f"with_ellie={report['n_with_ellie']} "
        f"dropped_turns={report['n_dropped_turns']} → {out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
