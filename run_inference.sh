#!/bin/bash
# Blind-B inference (load-only, deterministic, bit-exact). Prerequisites:
#   1. uv sync
#   2. bash download_datasets.sh          (challenge datasets + Qwen; needs HF_TOKEN)
#   3. HF_REPO=... bash download_weights.sh (our weights + caches + union artifact)
#
# The reranker loads the shipped LightGBM model and ranks blind_b from the
# shipped union artifact + dense caches. No source rebuilding, no GPU for the
# ranking — deterministic, ~10 min CPU, reproduces the submitted top-20
# bit-for-bit (confirm with: verify_blind_b_ranking.py --strict).
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

RERANK_CFG=blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5

# A. reranker: load the shipped LightGBM model, rank blind_b from the shipped
#    union artifact (candidates + source features) — no fit, no GPU.
uv run python reranker/protocol_098_union_rich_lgbm/main.py \
  --config "$RERANK_CFG" --target blind_b \
  --load-model artifacts/weights/reranker_lgbm.txt

# B. responder: Qwen3.6-27B, 10 seeded runs -> lexical-diversity selection (80GB GPU)
uv run python scripts/build_responder.py \
  --base-config rich_context_hierpop_tagchain \
  --ranked-artifact "artifacts/runs/reranker/protocol_098_union_rich_lgbm/$RERANK_CFG/full_public/blind_b" \
  --target blind_b \
  --out-dir "artifacts/runs/responder/qwen36_10run_diverse/$RERANK_CFG" \
  --n-runs 10 --seed 0 --selection-objective lexdiv --n-random-orders 30

echo "submission: artifacts/runs/responder/qwen36_10run_diverse/$RERANK_CFG/blind_b.submission.zip"
