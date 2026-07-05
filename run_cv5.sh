#!/bin/bash
# 5-fold CV validation on public_labeled (129,592 rows).
#
# PREREQUISITE — NOT a load-only flow. The public-labeled per-source retriever
# artifacts are NOT part of download_weights.sh (only the Blind-B ones are).
# Build them first (the public/CV portion of run_train.sh):
#   bash run_preprocess.sh
#   bash scripts/build_stage1_sources.sh
#   bash scripts/build_stage2_sources.sh   # GPU (two-tower)
# This script only rebuilds the public union (incl. the 36GB source_features)
# and *fits* the reranker per fold (LightGBM, n_jobs=-1) — expect hours of CPU
# and a high-RAM peak (~96GB with the full candidate pool).
# Expected: ndcg@20 ≈ 0.2743 (folds ≈ 0.2767 / 0.2755 / 0.2741 / 0.2721 / 0.2732),
# candidate recall@20 = 0.4102. Small last-digit wobble is normal (LightGBM
# histogram nondeterminism with n_jobs=-1).
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

UNION_CFG=blind_b_safe_combined_tpd1_parts_cooc500_cv5
RERANK_CFG=blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5

probe=artifacts/runs/retriever/bm25_5field_thought/top500_bsafe/fit_free_all_rows/public_labeled/candidates.npz
if [ ! -f "$probe" ]; then
  echo "error: public-labeled retriever sources are missing (e.g. $probe)." >&2
  echo "run_cv5.sh does not build retriever sources. Build them first:" >&2
  echo "  bash run_preprocess.sh && bash scripts/build_stage1_sources.sh && bash scripts/build_stage2_sources.sh" >&2
  exit 1
fi

uv run python retriever/union/main.py --config "$UNION_CFG" --target public_labeled
uv run python reranker/protocol_098_union_rich_lgbm/main.py --config "$RERANK_CFG" --target public_labeled

echo "scores: artifacts/results/reranker/protocol_098_union_rich_lgbm/$RERANK_CFG/cv5_oof/public_labeled/scores.json"
