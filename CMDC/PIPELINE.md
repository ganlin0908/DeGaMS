# CMDC · DeGaMS feature pipeline (TAV / video_fold)

## Text
1. `Text/reextract_cmdc_text_cls_mean.py`
   - Encoder: `hfl/chinese-roberta-wwm-ext` (fallback `bert-base-chinese`)
   - Pooling: `0.5*(CLS + masked-mean)`, `max_length=512`
   - Long turns: sliding window stride=256, mean over window vectors
   - Writes back: `egemaps_experiment/1_feature_build/features/cmdc/{sid}_features.pt`

## Audio
1. `audio/extract_cmdc_wav2vec_qpool.py`
   - `TencentGameMate/chinese-wav2vec2-base`, hidden **layer 6**
   - Per-question wav → mean-pool frames → `[T,768]`
   - Out: `egemaps_experiment/artifacts/wav2vec_embeds/cmdc/{sid}.pt`

## Visual
1. `visual/enrich_cmdc_visual.py`
   - Official per-question `Q{n}.npy` (768) or `vector_Q{n}.csv`
   - VideoAvailable=1 subjects only (45)

## Assemble
1. `_dataset_train/build_cmdc_degams_features.py` — run last; consumes all three
   modality outputs above and writes the tree the Dataset reads:
   `egemaps_experiment/artifacts/degams_cmdc_tav/{text,wav2vec,visual,mel}/`
   - `visual [T,797]` = OF29 **zeros** ∥ Q768

### Note on the 797-D visual layout

CMDC ships no OpenFace features, so the leading 29 dimensions are structurally zero and the
768-D official per-question embeddings occupy `visual[..., 29:797]`. The 797-D layout is kept
so that a **single** `DeGaMSModel` definition serves all three corpora unchanged.

Because `visual_dim >= 29 + 768`, the model selects `DenseVisualEncoder` with
`hi_visual_dim=768` (never the `DualVisual` path). Its OpenFace branch therefore receives an
all-zero input, so `proj_of` collapses to a learned bias that is constant across all turns and
subjects and is absorbed by the subsequent `fuse` layer. It contributes no CMDC-specific
information and cannot leak any: the effective visual representation is the 768-D embedding
alone. The `mel/` entries are likewise stubs (`audio_quality=1`, `pause_feats=0`), since CMDC
audio enters through the wav2vec branch only.

## Train
- `_dataset_train/cmdc_degams_dataset.py` → `CMDCDeGaMSDataset`
- `_dataset_train/train_cmdc_degams_seed_sweep.py`
- Model: `../_shared/degams_model.py` (`DeGaMSModel`)
- Splits: `CMDC/label/video_fold{1..5}/`, selected with `--fold` or `DEGAMS_CMDC_FOLD`
