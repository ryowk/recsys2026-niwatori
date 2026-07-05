#!/bin/bash
# Maintainer helper: push the trained weights + the derived caches / union
# artifact needed to reproduce the Blind-B ranking to the Hugging Face dataset
# repo. Requires write access (HF_TOKEN).
#   bash upload_weights.sh          # HF_REPO defaults to ryowk/recsys2026-niwatori
set -eu
cd "$(dirname "$0")"

HF_REPO="${HF_REPO:-ryowk/recsys2026-niwatori}"
UNION_CFG=blind_b_safe_combined_tpd1_parts_cooc500_cv5
RUNS=artifacts/runs/retriever

# trained weights: reranker LightGBM + two-tower LoRA (full_public + fold0..4)
uv run hf upload "$HF_REPO" artifacts/weights weights --repo-type dataset

# derived caches consumed by the deterministic load-only ranking
uv run hf upload "$HF_REPO" artifacts/cache/dense_track_emb.npz \
  cache/dense_track_emb.npz --repo-type dataset
uv run hf upload "$HF_REPO" artifacts/cache/dense_qfeat/blind_b.npz \
  cache/dense_qfeat/blind_b.npz --repo-type dataset

# union artifact -> cache/runs_seed (expanded to artifacts/runs by download_weights.sh):
#   blind_b full dir (candidates + source_features) + public candidates.npz
uv run hf upload "$HF_REPO" "$RUNS/union/$UNION_CFG/blind_b" \
  "cache/runs_seed/retriever/union/$UNION_CFG/blind_b" --repo-type dataset
uv run hf upload "$HF_REPO" "$RUNS/union/$UNION_CFG/public_labeled/candidates.npz" \
  "cache/runs_seed/retriever/union/$UNION_CFG/public_labeled/candidates.npz" --repo-type dataset

echo "uploaded to https://huggingface.co/datasets/$HF_REPO"
echo "note: if the repo still holds the earlier full upload, delete the unused"
echo "      files (cache/two_tower, cache/dense_qfeat/{train,devset}, cache/splits,"
echo "      cache/spotify_uuid_map.parquet, per-source runs_seed) to slim it."
