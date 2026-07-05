#!/bin/bash
# Reranker 5-fold CV on public_labeled -> CV scores (NOT on the submission path).
# Fits the LightGBM per fold over the public union. CPU, hours, high RAM (~96GB).
# Depends on: run_retriever_cv5.sh (the public union + its source_features).
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
RERANK_CFG=blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5
uv run python reranker/protocol_098_union_rich_lgbm/main.py --config "$RERANK_CFG" --target public_labeled
echo "CV scores: artifacts/results/reranker/protocol_098_union_rich_lgbm/$RERANK_CFG/cv5_oof/public_labeled/scores.json"
