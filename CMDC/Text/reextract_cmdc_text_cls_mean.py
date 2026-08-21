"""Re-encode CMDC text turns as 0.5*(CLS + masked-mean) into feature packs.

Keeps Wav2Vec2 / visual untouched. Backs up previous ``text_features`` to
``text_features_cls_backup`` once per pack.

Protocol:
  * max_length = 512
  * If tokenized length ≤ 512: single-pass 0.5*(CLS+mean)
  * If longer: sliding-window chunks (stride = max_length // 2), mean-pool
    per-window 0.5*(CLS+mean) vectors → one turn embedding

Usage:
  python CMDC/Text/reextract_cmdc_text_cls_mean.py --force
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModel, AutoTokenizer

ROOT = Path(os.environ.get("DEGAMS_ROOT", r"D:\Depression"))
FEAT_DIR = ROOT / "egemaps_experiment" / "1_feature_build" / "features" / "cmdc"
TEXT_MODELS = ["hfl/chinese-roberta-wwm-ext", "bert-base-chinese"]
MAX_LENGTH = 512
STRIDE = MAX_LENGTH // 2  # 256
LOG = ROOT / "CMDC" / "logs" / "reextract_text_cls_mean.log"
POOLING_TAG = f"0.5*(CLS+mean) max_len={MAX_LENGTH} slide_stride={STRIDE}"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_encoder(device: torch.device):
    last_err: Optional[Exception] = None
    for name in TEXT_MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(name)
            model = AutoModel.from_pretrained(name).to(device).eval()
            log(f"Loaded text encoder: {name}")
            return tok, model, name
        except Exception as e:  # noqa: BLE001
            last_err = e
            log(f"Failed {name}: {e}")
    raise RuntimeError(f"No text encoder available: {last_err}")


@torch.no_grad()
def _encode_token_ids(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """Encode one [1, L] window → 0.5*(CLS+mean) [768]."""
    out = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device))
    hs = out.last_hidden_state
    cls = hs[:, 0, :]
    mask = attention_mask.to(device).unsqueeze(-1).float()
    mean = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return (0.5 * (cls + mean)).squeeze(0).float().cpu()


@torch.no_grad()
def encode_cls_mean(tok, model, text: str, device: torch.device) -> torch.Tensor:
    """Single-pass or sliding-window mean over 0.5*(CLS+mean) chunks."""
    raw = text or ""
    # Full tokenization without truncation to decide windowing.
    enc = tok(
        raw,
        add_special_tokens=True,
        truncation=False,
        padding=False,
        return_attention_mask=True,
        return_tensors=None,
    )
    ids: List[int] = list(enc["input_ids"])
    if len(ids) <= MAX_LENGTH:
        inputs = tok(
            raw,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        return _encode_token_ids(
            model, inputs["input_ids"], inputs["attention_mask"], device
        )

    # Sliding windows over token ids (already include [CLS]/[SEP] from tokenizer).
    # Re-window content tokens and re-add specials per chunk for valid RoBERTa inputs.
    # Safer: strip specials, slide content, wrap each chunk with specials.
    special_ids = set()
    for attr in ("cls_token_id", "sep_token_id", "pad_token_id", "bos_token_id", "eos_token_id"):
        tid = getattr(tok, attr, None)
        if tid is not None:
            special_ids.add(int(tid))
    content = [t for t in ids if t not in special_ids]
    if not content:
        content = ids[1:-1] if len(ids) > 2 else ids

    windows: List[torch.Tensor] = []
    start = 0
    while start < len(content):
        chunk = content[start : start + (MAX_LENGTH - 2)]
        if not chunk:
            break
        piece = tok.decode(chunk, skip_special_tokens=True)
        inputs = tok(
            piece,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )
        windows.append(
            _encode_token_ids(model, inputs["input_ids"], inputs["attention_mask"], device)
        )
        if start + (MAX_LENGTH - 2) >= len(content):
            break
        start += STRIDE
    if not windows:
        return torch.zeros(model.config.hidden_size, dtype=torch.float32)
    return torch.stack(windows, dim=0).mean(dim=0)


def process_pack(path: Path, tok, model, device: torch.device, model_name: str) -> Tuple[int, int]:
    pack = torch.load(path, weights_only=False)
    meta = pack.get("meta_data") or []
    if not meta:
        raise RuntimeError(f"{path.name}: empty meta_data")
    old = pack["text_features"].float()
    if "text_features_cls_backup" not in pack:
        pack["text_features_cls_backup"] = old.detach().cpu().contiguous()
    feats = []
    for m in meta:
        text = str(m.get("text") or "")
        feats.append(encode_cls_mean(tok, model, text, device))
    new = torch.stack(feats, dim=0)
    if new.shape != old.shape:
        raise RuntimeError(f"{path.name}: shape {tuple(new.shape)} vs {tuple(old.shape)}")
    pack["text_features"] = new.contiguous()
    pack["text_model"] = model_name
    pack["text_pooling"] = POOLING_TAG
    torch.save(pack, path)
    return int(new.size(0)), int(new.size(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if pooling tag already matches slide512",
    )
    args = ap.parse_args()

    paths = sorted(FEAT_DIR.glob("*_features.pt"))
    if args.limit > 0:
        paths = paths[: args.limit]
    if not paths:
        log(f"No packs in {FEAT_DIR}")
        return 2

    device = torch.device(args.device)
    tok, model, name = load_encoder(device)
    ok = skip = fail = 0
    for path in paths:
        try:
            pack = torch.load(path, weights_only=False)
            pooling = str(pack.get("text_pooling") or "")
            if (not args.force) and ("slide_stride" in pooling and f"max_len={MAX_LENGTH}" in pooling):
                log(f"SKIP {path.name} (already {pooling})")
                skip += 1
                continue
            n, d = process_pack(path, tok, model, device, name)
            log(f"OK {path.name} T={n}x{d}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            log(f"FAIL {path.name}: {e}")
            fail += 1
    log(f"Done ok={ok} skip={skip} fail={fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
