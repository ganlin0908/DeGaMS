"""Build isolated AV-boost feature tree (does NOT modify clean_v2 contents).

Layout:
  egemaps_experiment/artifacts/degams_avboost_v1/{daic,edaic}/
    text     -> junction to clean_v2/.../text
    mel      -> junction to clean_v2/.../mel
    visual   -> junction to clean_v2/.../visual_dense   (797-d OF∥DenseNet)
    wav2vec  -> junction to clean_v2/.../wav2vec_enh    (DeepFilter+Silero)

Original ``degams_clean_v2`` stays the legacy TAV source (OF-210 + VAD wav2vec).
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(os.environ.get("DEGAMS_ROOT", r"/path/to/degams_data"))
CLEAN = ROOT / "egemaps_experiment" / "artifacts" / "degams_clean_v2"
OUT = ROOT / "egemaps_experiment" / "artifacts" / "degams_avboost_v1"


def _junction(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        # remove junction / symlink / dir link
        subprocess.run(["cmd", "/c", "rmdir", str(link)], check=False)
        if link.exists():
            link.unlink(missing_ok=True)
    if not target.is_dir():
        raise FileNotFoundError(f"missing target: {target}")
    # Windows directory junction (no admin needed)
    r = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"mklink failed: {link} -> {target}\n{r.stdout}\n{r.stderr}")


def build_corpus(corpus: str) -> dict:
    src = CLEAN / corpus
    dst = OUT / corpus
    dst.mkdir(parents=True, exist_ok=True)
    mapping = {
        "text": src / "text",
        "mel": src / "mel",
        "visual": src / "visual_dense",
        "wav2vec": src / "wav2vec_enh",
    }
    for name, target in mapping.items():
        link = dst / name
        _junction(link, target)
        n = len(list(link.glob("*")))
        print(f"  {corpus}/{name} -> {target} ({n} entries)", flush=True)
    meta = {
        "corpus": corpus,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "mapping": {k: str(v) for k, v in mapping.items()},
        "note": "Isolated AV-boost view; clean_v2 untouched as legacy TAV",
    }
    (dst / "AVBOOST_README.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "root": str(OUT),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "corpora": {},
    }
    for corp in ("daic", "edaic"):
        print(f"[build] {corp}", flush=True)
        report["corpora"][corp] = build_corpus(corp)
    (OUT / "MANIFEST.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT / "README.md").write_text(
        "\n".join(
            [
                "# degams_avboost_v1",
                "",
                "Isolated feature root for DenseNet∥OpenFace visual + DeepFilter/Silero audio.",
                "",
                "- Does **not** overwrite `degams_clean_v2` (legacy OF-210 + VAD wav2vec).",
                "- Train with `DEGAMS_FEATURE_ROOT=.../degams_avboost_v1/<corpus>`.",
                "",
                "See `MANIFEST.json`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
