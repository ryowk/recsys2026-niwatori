#!/bin/bash
# Build preprocessing artifacts.
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
. scripts/_run_lib.sh

split_artifact_complete artifacts/preprocessed/splits/cv5 || \
  uv run python scripts/build_public_splits.py \
    --out-dir artifacts/preprocessed/splits/cv5 --name public_labeled_v2_5fold --n-splits 5 --seed 20260515
([ -f artifacts/preprocessed/catalog_id_map.parquet ] && \
 [ -f artifacts/preprocessed/catalog_id_map.manifest.json ]) || \
  uv run python scripts/build_catalog_id_map.py
# This command validates the existing NPZ and only rebuilds an incomplete file.
uv run python preprocessing/dense_track_encoder.py  # GPU
