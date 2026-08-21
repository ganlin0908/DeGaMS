# DAIC · DeGaMS feature pipeline (avboost)

## Text
1. `Text/extract_text_features.py` — MentalBERT CLS (`mental/mental-bert-base-uncased`)
2. `Text/clean_daic_ellie.py` — drop Ellie turns → clean_v2 text
   → `egemaps_experiment/artifacts/degams_clean_v2/daic/text/`

## Audio
1. `audio/audio_front_vad.py` — shared VAD helpers
2. `audio/extract_clean_wav2vec_enhance.py` — DeepFilter + Silero → wav2vec2-base
   → `.../degams_clean_v2/daic/wav2vec_enh/` (exposed via `degams_avboost_v1` junction)

## Visual
1. `visual/extract_openface_visual_210.py` — OpenFace → [T,210] (29+pad)
2. `visual/rebuild_edaic_visual_768.py` — DenseNet frames → turn + TRAIN PCA-768
3. `visual/extract_visual_dense_of_dn.py` — OF29 ∥ DN768 → visual_dense [T,797]

## Assemble
1. `_dataset_train/build_degams_clean_v2.py` — driver for the baseline tree; runs the text,
   mel-VAD, and OpenFace-210 steps above → `degams_clean_v2/{daic,edaic}/`
2. `_dataset_train/build_degams_avboost_v1_tree.py` — run last; builds the tree the reported
   runs actually train on, as Windows directory junctions into `degams_clean_v2`:

   | `degams_avboost_v1/<corpus>/` | → | `degams_clean_v2/<corpus>/` |
   |---|---|---|
   | `text` | → | `text` |
   | `mel` | → | `mel` |
   | `visual` | → | `visual_dense` (797-D OF29∥DenseNet) |
   | `wav2vec` | → | `wav2vec_enh` (DeepFilterNet + Silero) |

   It creates no new features and never writes into `degams_clean_v2`, so the baseline tree
   stays intact for comparison.

## Train
- `_dataset_train/daic_degams_dataset.py`
- `_dataset_train/train_daic_degams.py` / `train_daic_degams_seed_sweep.py`
- Model: `../_shared/degams_model.py` (`DeGaMSModel`)
- Splits: official AVEC 2017 partitions (`train_split_Depression_AVEC2017.csv`,
  `dev_split_Depression_AVEC2017.csv`, `full_test_split.csv`)

### Selecting the feature tree

Point `DEGAMS_FEATURE_ROOT` at `degams_avboost_v1/<corpus>` to reproduce the reported results,
or leave it unset to train on the `degams_clean_v2` baseline (OF-210 + VAD wav2vec).

Do **not** combine it with `DEGAMS_AVBOOST=1`. That variable is a separate legacy route which
reads `degams_clean_v2` and substitutes `visual_dense` / `wav2vec_enh` in place; setting
`DEGAMS_FEATURE_ROOT` deliberately disables it. The reported runs use the junction tree only.
