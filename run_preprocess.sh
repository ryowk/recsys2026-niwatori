#!/bin/bash
# Preprocessing: CV splits + TPD1->catalog track-id map.
# The 3-fold split (cv3) is required by the tfidf source; the 5-fold split (cv5)
# is the primary CV split used by the union / reranker.
# build_spotify_uuid_map.py reads TalkPlayData-2 (run online once; needs HF_TOKEN).
set -eu
cd "$(dirname "$0")"

uv run python scripts/build_public_splits.py \
  --out-dir artifacts/cache/splits/cv3 --name public_labeled_v1 --n-splits 3
uv run python scripts/build_public_splits.py \
  --out-dir artifacts/cache/splits/cv5 --name public_labeled_v2_5fold --n-splits 5 --seed 20260515
uv run python scripts/build_spotify_uuid_map.py

# Optional (GPU, only to regenerate shipped caches from scratch):
#   uv run python preprocessing/dense_track_encoder.py     # artifacts/cache/dense_track_emb.npz
# dense query features (artifacts/cache/dense_qfeat/*.npz) are (re)encoded
# automatically by the reranker when cache rows are missing (GPU).
