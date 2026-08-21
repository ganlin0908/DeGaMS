"""Build degams_clean_v2 feature tree: Ellie clean → text → mel VAD → OpenFace 210."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from clean_v2_paths import CLEAN_ROOT, ensure_clean_tree

HERE = Path(__file__).resolve().parent
PY = sys.executable


def run(script: str, extra: list[str]) -> None:
    cmd = [PY, "-u", str(HERE / script), *extra]
    print("RUN", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(HERE))
    if rc != 0:
        raise SystemExit(rc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-mel", action="store_true")
    ap.add_argument("--skip-visual", action="store_true")
    ap.add_argument("--only-daic", action="store_true")
    ap.add_argument("--only-edaic", action="store_true")
    args = ap.parse_args()
    ensure_clean_tree()

    common = []
    if args.overwrite:
        common.append("--overwrite")
    if args.limit > 0:
        common += ["--limit", str(args.limit)]

    # 1) text packs
    if not args.only_edaic:
        run("clean_daic_ellie.py", common)
    if not args.only_daic:
        run("copy_edaic_text_clean.py", common)

    # 2) mel
    if not args.skip_mel:
        corp = "both"
        if args.only_daic:
            corp = "daic"
        elif args.only_edaic:
            corp = "edaic"
        run("extract_clean_mel_vad.py", ["--corpus", corp, *common])

    # 3) visual
    if not args.skip_visual:
        corp = "both"
        if args.only_daic:
            corp = "daic"
        elif args.only_edaic:
            corp = "edaic"
        run("extract_openface_visual_210.py", ["--corpus", corp, *common])

    summary = {
        "clean_root": str(CLEAN_ROOT),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "steps": {
            "daic_ellie_clean": not args.only_edaic,
            "edaic_text_copy": not args.only_daic,
            "mel_vad": not args.skip_mel,
            "openface_210": not args.skip_visual,
        },
        "schema": {
            "text": "{corpus}/text/{sid}_features.pt — text_features[T,768] + meta Participant-only",
            "mel": "{corpus}/mel/{sid}.pt — mel_turns[T,80] + pause_feats[T,8] + audio_quality[T]",
            "visual": "{corpus}/visual/{sid}.npz — visual[T,210] OpenFace core29+pad",
        },
    }
    (CLEAN_ROOT / "BUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (CLEAN_ROOT / "README.md").write_text(
        "\n".join(
            [
                "# DeGaMS clean TAV features v2",
                "",
                f"Root: `{CLEAN_ROOT}`",
                "",
                "## Contents",
                "- `daic/text` — Ellie removed (Participant turns only)",
                "- `edaic/text` — text+meta (old DenseNet visual stripped)",
                "- `{daic,edaic}/mel` — speech-frame Mel + pause_feats + soft audio_quality (no hard delete)",
                "- `{daic,edaic}/visual` — OpenFace Pose/Gaze/AU_r → 210-d (29+pad)",
                "",
                "## Build",
                "```bat",
                f"{PY} -u {HERE / 'build_degams_clean_v2.py'} --overwrite",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
