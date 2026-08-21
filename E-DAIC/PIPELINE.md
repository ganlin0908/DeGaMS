# E-DAIC · DeGaMS feature pipeline (avboost)

## Text
1. `Text/extract_edaic_features.py` — MentalBERT (overlap reuse DAIC where applicable)
2. `Text/copy_edaic_text_clean.py` — copy into clean_v2, drop legacy visual packs
   → `egemaps_experiment/artifacts/degams_clean_v2/edaic/text/`

## Audio / Visual
Same shared scripts as DAIC under `audio/` and `visual/` (corpus switch via args / paths).

## Assemble
Same two steps as DAIC, which build both corpora in one pass — see `../DAIC/PIPELINE.md`:
1. `_dataset_train/build_degams_clean_v2.py` → `degams_clean_v2/edaic/`
2. `_dataset_train/build_degams_avboost_v1_tree.py` → `degams_avboost_v1/edaic/` (junctions)

## Train
- `_dataset_train/edaic_degams_dataset.py` (extends `DAIC/_dataset_train/daic_degams_dataset.py`)
- `_dataset_train/train_edaic_degams_seed_sweep.py`
- Model: `../_shared/degams_model.py` (`DeGaMSModel`)
- Splits: official AVEC 2019 DDS partitions
- E-DAIC visual is single-source DenseNet-PCA 768-D, so train with `--no-dual-visual`

### Selecting the feature tree

As in DAIC: set `DEGAMS_FEATURE_ROOT` to `degams_avboost_v1/edaic` for the reported results, and
do not also set `DEGAMS_AVBOOST=1` (see `../DAIC/PIPELINE.md` for why the two are exclusive).
