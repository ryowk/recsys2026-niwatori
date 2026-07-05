#!/bin/bash
# Responder: Qwen3.6-27B, 10 seeded runs -> lexical-diversity selection -> submission zip.
# Depends on: run_reranker_blind_b.sh (the blind_b ranked artifact). Needs an 80GB GPU.
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
RERANK_CFG=blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5
uv run python scripts/build_responder.py \
  --base-config rich_context_hierpop_tagchain \
  --ranked-artifact "artifacts/runs/reranker/protocol_098_union_rich_lgbm/$RERANK_CFG/full_public/blind_b" \
  --target blind_b \
  --out-dir "artifacts/runs/responder/qwen36_10run_diverse/$RERANK_CFG" \
  --n-runs 10 --seed 0 --selection-objective lexdiv --n-random-orders 30
echo "submission: artifacts/runs/responder/qwen36_10run_diverse/$RERANK_CFG/blind_b.submission.zip"
