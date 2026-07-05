#!/bin/bash
# Reranker: fit the final model on all public rows, rank blind_b (from scratch,
# no --load-model). Writes artifacts/runs/reranker/.../full_public/blind_b/.
# Depends on: run_retriever_cv5.sh (public union — the final model is fit on it)
#         AND run_retriever_blind_b.sh (blind union — used for the prediction).
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
RERANK_CFG=blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5
uv run python reranker/protocol_098_union_rich_lgbm/main.py --config "$RERANK_CFG" --target blind_b
echo "ranked: artifacts/runs/reranker/protocol_098_union_rich_lgbm/$RERANK_CFG/full_public/blind_b"
