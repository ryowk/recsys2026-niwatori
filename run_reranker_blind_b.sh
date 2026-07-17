#!/bin/bash
# Fit on the public union and rank the Blind-B union.
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
RERANK_CFG=combined_tpd1_parts_cooc500_t200_cv5
OUT=artifacts/runs/reranker/union_lambdarank/$RERANK_CFG/full_public/blind_b
. scripts/_run_lib.sh
if ranked_artifact_complete "$OUT"; then
  echo "skip reranker (complete): $OUT"
else
  uv run python -m reranker.union_lambdarank.main --config "$RERANK_CFG" --target blind_b
fi
echo "ranked: $OUT"
