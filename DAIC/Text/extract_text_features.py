"""DAIC-WOZ text branch: MentalBERT turn embeddings -> {sample_id}.pt [768].

Reads the pre-segmented transcript JSON and writes one pooled 768-D tensor per
segment, plus a provenance manifest recording seed, checkpoint, and pooling mode.
"""

import datetime
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ==========================================
# 1. Reproducibility configuration (seed & paths)
# ==========================================
GLOBAL_SEED = 42
MODEL_NAME = "mental/mental-bert-base-uncased"  # official MentalBERT checkpoint
FALLBACK_MODEL_NAME = "bert-base-uncased"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
POOLING_STRATEGY = "cls"  # 'cls' ([CLS] token) or 'mean' (mask-aware mean pooling)
BATCH_SIZE = 128  # 128-256 saturates an RTX 5090
MAX_LENGTH = 512  # BERT truncation limit

_ROOT = Path(os.environ.get("DEGAMS_ROOT", r"/path/to/degams_data"))
JSON_INPUT_PATH = str(_ROOT / "DAICWOZ" / "processed_output" / "processed_segments.json")
OUTPUT_FEATURE_DIR = str(_ROOT / "DAICWOZ" / "features" / "text")


def set_deterministic_seed(seed: int):
    """Pin every RNG and disable cuDNN autotuning so repeated runs agree exactly."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 2. Data loading
# ==========================================
class TextSegmentDataset(Dataset):
    def __init__(self, json_path: str):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'sample_id': item['sample_id'],
            'text_raw': item['text_raw']
        }


# ==========================================
# 3. Extractor engine
# ==========================================
class SCIEmbeddingExtractor:
    def __init__(self, model_name: str, device: torch.device):
        self.device = device
        print(f"[*] Loading Pre-trained Model and Tokenizer: {model_name}...")
        token = HF_TOKEN if HF_TOKEN else None
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
            self.model = AutoModel.from_pretrained(model_name, token=token)
        except Exception as e:
            print(f"[!] Failed to load {model_name}: {e}")
            print(f"[*] Falling back to {FALLBACK_MODEL_NAME} so extraction can continue...")
            self.tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL_NAME)
            self.model = AutoModel.from_pretrained(FALLBACK_MODEL_NAME)
        self.model = self.model.to(self.device)
        self.model.eval()  # freeze dropout / normalisation layers

    def _execute_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor,
                         strategy: str) -> torch.Tensor:
        """
        Collapse the [seq_len, hidden] token matrix into one sentence-level vector.
        """
        if strategy == "cls":
            # Option A: the [CLS] token, which carries the sentence-level summary.
            return last_hidden_state[:, 0, :]

        elif strategy == "mean":
            # Option B: mask-aware mean pooling, so padding cannot bias the mean.
            # Expand the mask: [batch_size, seq_len] -> [batch_size, seq_len, hidden_dim]
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            # Numerator: sum over real tokens only.
            sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
            # Denominator: per-sample real-token count, floored at 1e-9 against divide-by-zero.
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            return sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unknown pooling strategy: {strategy}")

    def extract_and_save(self, dataloader: DataLoader, output_dir: Path, strategy: str):
        output_dir.mkdir(parents=True, exist_ok=True)

        # no_grad drops the autograd graph, which saves memory and time.
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Extracting text embeddings"):
                sample_ids = batch['sample_id']
                texts = batch['text_raw']

                # Pad / truncate to a uniform batch and move it to the device.
                inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt"
                ).to(self.device)

                # Forward pass -> last hidden state.
                outputs = self.model(**inputs)

                # Pool to a [batch_size, 768] feature matrix.
                pooled_features = self._execute_pooling(
                    outputs.last_hidden_state,
                    inputs['attention_mask'],
                    strategy
                )

                # --- device-independent save ---
                # Move back to host memory and detach: tensors still bound to a GPU
                # cannot be safely deserialised on machines with different hardware.
                features_cpu = pooled_features.cpu().detach()

                for i, sample_id in enumerate(sample_ids):
                    single_feature = features_cpu[i]  # shape [768]
                    save_path = output_dir / f"{sample_id}.pt"
                    torch.save(single_feature, save_path)


# ==========================================
# 4. Pipeline entry point
# ==========================================
def main():
    set_deterministic_seed(GLOBAL_SEED)

    # Use the accelerator when one is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up run logging.
    log_path = Path(OUTPUT_FEATURE_DIR) / "extraction_run.log"
    os.makedirs(OUTPUT_FEATURE_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_path, encoding='utf-8'), logging.StreamHandler()]
    )

    logging.info("=" * 60)
    logging.info("STARTING TEXT FEATURE EXTRACTION PIPELINE FOR SCI REPRODUCTION")
    logging.info("=" * 60)
    logging.info(f"Target Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    logging.info(f"Base Backbone Model: {MODEL_NAME}")
    logging.info(f"Mathematical Pooling Layer: {POOLING_STRATEGY.upper()}")

    # Load the dataset.
    dataset = TextSegmentDataset(JSON_INPUT_PATH)
    # pin_memory locks the host pages so host-to-device transfer runs at full bandwidth.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Run the extractor.
    extractor = SCIEmbeddingExtractor(model_name=MODEL_NAME, device=device)
    extractor.extract_and_save(dataloader, Path(OUTPUT_FEATURE_DIR), strategy=POOLING_STRATEGY)

    # --- provenance manifest ---
    meta_data = {
        "pipeline_purpose": "SCI Paper Multimodal Depression Classification",
        "timestamp": datetime.datetime.now().isoformat(),
        "random_seed": GLOBAL_SEED,
        "backbone_checkpoint": MODEL_NAME,
        "embedding_dimension": 768,
        "pooling_strategy": POOLING_STRATEGY,
        "max_sequence_length": MAX_LENGTH,
        "total_extracted_samples": len(dataset),
        "hardware_infrastructure": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }

    meta_path = Path(OUTPUT_FEATURE_DIR) / "extraction_meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=4, ensure_ascii=False)

    logging.info("[OK] All text segments converted into dense tensors successfully.")
    logging.info(f"[OK] Meta validation manifest written to: {meta_path}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
