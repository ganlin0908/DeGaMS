"""Build DeGaMS feature tree for CMDC TAV (video subjects).

Writes $DEGAMS_ROOT/egemaps_experiment/artifacts/degams_cmdc_tav/
  text/{sid}_features.pt
  wav2vec/{sid}.pt
  visual/{sid}.npz   # visual [T,797] = OF29 zeros ∥ Q*.npy 768
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

ROOT = Path(os.environ.get("DEGAMS_ROOT", r"/path/to/degams_data"))
SRC_TEXT = ROOT / "egemaps_experiment" / "1_feature_build" / "features" / "cmdc"
SRC_W2V = ROOT / "egemaps_experiment" / "artifacts" / "wav2vec_embeds" / "cmdc"
OUT = ROOT / "egemaps_experiment" / "artifacts" / "degams_cmdc_tav"
SUBJECT_INFO = ROOT / "CMDC" / "raw" / "extracted" / "SubjectInfo.xlsx"
OF_DIM = 29
HI_DIM = 768
VIS_DIM = OF_DIM + HI_DIM  # 797


def video_ids() -> list[str]:
    info = pd.read_excel(SUBJECT_INFO)
    ids = info["ID"].astype(str).tolist()
    if ids.count("MDD20") > 1:
        dup = [i for i, x in enumerate(ids) if x == "MDD20"]
        info.at[dup[1], "ID"] = "MDD21"
    vids = info.loc[info["VideoAvailabe"] == 1, "ID"].astype(str).tolist()
    return sorted(vids, key=lambda x: (x[:3], int(x[3:])))


def main() -> int:
    for sub in ("text", "wav2vec", "visual", "mel"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    vids = video_ids()
    report = {"n": 0, "ok": 0, "fail": [], "built_at": datetime.now().isoformat(timespec="seconds")}
    for sid in tqdm(vids, desc="cmdc-degams-feat"):
        report["n"] += 1
        src = SRC_TEXT / f"{sid}_features.pt"
        w2v = SRC_W2V / f"{sid}.pt"
        if not src.is_file() or not w2v.is_file():
            report["fail"].append(f"{sid}: missing text/w2v")
            continue
        pack = torch.load(src, map_location="cpu", weights_only=False)
        if "visual_features" not in pack:
            report["fail"].append(f"{sid}: no visual_features")
            continue
        text = pack["text_features"].float()
        vis768 = pack["visual_features"].float()
        t = int(text.size(0))
        vis768 = vis768[:t]
        if vis768.size(0) < t:
            pad = torch.zeros(t - vis768.size(0), 768)
            vis768 = torch.cat([vis768, pad], dim=0)
        of = torch.zeros(t, OF_DIM)
        visual = torch.cat([of, vis768], dim=-1)  # [T,797]
        assert visual.size(-1) == VIS_DIM

        # text pack
        out_text = {
            "text_features": text.contiguous(),
            "meta_data": list(pack.get("meta_data") or [])[:t],
            "subject_id": sid,
            "dataset": "CMDC",
            "clean_v2": {"source": "cmdc_tav_features", "n_turns": t},
        }
        torch.save(out_text, OUT / "text" / f"{sid}_features.pt")

        # wav2vec
        wpack = torch.load(w2v, map_location="cpu", weights_only=False)
        at = wpack["audio_turns"].float()[:t]
        if at.size(0) < t:
            at = torch.cat([at, torch.zeros(t - at.size(0), 768)], dim=0)
        q = torch.ones(t, dtype=torch.float32)
        pause = torch.zeros(t, 8, dtype=torch.float32)
        out_w = {
            "audio": at.mean(dim=0),
            "audio_turns": at.contiguous(),
            "audio_quality": q,
            "pause_feats": pause,
            "speech_frac": q.clone(),
            "checkpoint": wpack.get("checkpoint", "TencentGameMate/chinese-wav2vec2-base"),
            "sr": 16000,
            "pool": wpack.get("pool", "q_mean_layer6"),
            "align": "cmdc_question",
            "session_id": sid,
            "n_turns": t,
            "n_ok_turns": int(wpack.get("n_ok_turns") or t),
        }
        torch.save(out_w, OUT / "wav2vec" / f"{sid}.pt")
        # mel stub (dataset may look for pause reference)
        torch.save(
            {"audio_quality": q, "pause_feats": pause, "session_id": sid},
            OUT / "mel" / f"{sid}.pt",
        )

        # visual npz
        np.savez_compressed(
            OUT / "visual" / f"{sid}.npz",
            visual=visual.numpy().astype(np.float32),
            of_core=of.numpy().astype(np.float32),
            hi_visual=vis768.numpy().astype(np.float32),
            visual_turn_mask=np.ones(t, dtype=np.bool_),
            of_valid=np.zeros(t, dtype=np.bool_),
            dn_valid=np.ones(t, dtype=np.bool_),
            visual_dim=np.int32(VIS_DIM),
            openface_core_dim=np.int32(OF_DIM),
            hi_visual_dim=np.int32(HI_DIM),
            present=np.bool_(True),
            source="cmdc_Qnpy_padOF29",
            session_id=sid,
            n_turns=np.int32(t),
            layout="of29_zeros||q768",
        )
        report["ok"] += 1

    (OUT / "BUILD.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] == report["n"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
