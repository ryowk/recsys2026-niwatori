#!/bin/bash
# Download the trained weights + the derived caches / union artifact needed to
# reproduce the Blind-B ranking, from our Hugging Face dataset repo, then expand
# the union seed artifact into the runtime tree.
#   HF_REPO=ryowk/recsys2026-niwatori bash download_weights.sh
#
# Pulls only the load-only inference set (~0.3GB): the LightGBM model, the dense
# caches, and the Blind-B / public union artifacts. Enough for run_inference.sh
# to reproduce the submitted top-20 bit-for-bit. (The two-tower weights, used
# only by the auxiliary run_blind_b.sh, are fetched separately — see below.)
set -eu
cd "$(dirname "$0")"

HF_REPO="${HF_REPO:-ryowk/recsys2026-niwatori}"

uv run hf download "$HF_REPO" --repo-type dataset --local-dir artifacts \
  --include "weights/reranker_lgbm.txt" \
            "cache/dense_track_emb.npz" \
            "cache/dense_qfeat/blind_b.npz" \
            "cache/runs_seed/retriever/union/**"

# The auxiliary run_blind_b.sh (regenerate from weights) additionally needs the
# two-tower weights — fetch them separately when using that path:
#   uv run hf download "$HF_REPO" --repo-type dataset --local-dir artifacts --include "weights/two_tower/**"

# cache/runs_seed/ mirrors the union artifact the reranker reads at inference;
# expand it into artifacts/runs/ (the runtime root).
mkdir -p artifacts/runs
cp -r artifacts/cache/runs_seed/. artifacts/runs/

echo "ready: artifacts/weights, artifacts/cache, artifacts/runs (union seeded)"
