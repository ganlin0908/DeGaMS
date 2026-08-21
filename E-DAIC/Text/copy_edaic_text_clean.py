"""Copy E-DAIC text+meta into clean_v2 (Participant-only already); strip DenseNet visual."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm

from clean_v2_paths import EDAIC_SRC_TEXT, EDAIC_SRC_TEXT_ALT, edaic_dirs, ensure_clean_tree


def resolve_src() -> Path:
    if EDAIC_SRC_TEXT.is_dir() and any(EDAIC_SRC_TEXT.glob("*_features.pt")):
        return EDAIC_SRC_TEXT
    return EDAIC_SRC_TEXT_ALT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    ensure_clean_tree()
    src = resolve_src()
    out_dir = edaic_dirs()["text"]
    packs = sorted(src.glob("*_features.pt"))
    if args.limit > 0:
        packs = packs[: args.limit]

    n = 0
    for p in tqdm(packs, desc="edaic-text-copy"):
        sid = p.name[: -len("_features.pt")]
        dest = out_dir / f"{sid}_features.pt"
        if dest.is_file() and not args.overwrite:
            n += 1
            continue
        pack = torch.load(p, map_location="cpu", weights_only=False)
        meta = list(pack.get("meta_data") or [])
        for i, m in enumerate(meta):
            m = dict(m)
            m["speaker"] = "Participant"
            m["segment_id"] = i
            meta[i] = m
        out = {
            "text_features": pack["text_features"].float().contiguous(),
            "meta_data": meta,
            "clean_v2": {
                "source": str(p),
                "stripped_old_visual": "visual_features" in pack,
                "note": "OpenFace visual rebuilt separately",
            },
        }
        torch.save(out, dest)
        n += 1

    report = {
        "src": str(src),
        "n": n,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }
    (edaic_dirs()["base"] / "_text_copy_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"E-DAIC text copy: {n} → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
