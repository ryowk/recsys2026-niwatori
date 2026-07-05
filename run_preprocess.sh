#!/bin/bash
# Preprocessing: 5-fold CV split + TPD1->catalog map + dense track embeddings.
# Depends on: download_datasets.sh (challenge + TPD2 + Qwen in the HF cache).
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

uv run python scripts/build_public_splits.py \
  --out-dir artifacts/cache/splits/cv5 --name public_labeled_v2_5fold --n-splits 5 --seed 20260515
uv run python scripts/build_spotify_uuid_map.py                      # TPD2 (online first run)
uv run python preprocessing/dense_track_encoder.py --target blind_b  # -> artifacts/cache/dense_track_emb.npz (GPU)
