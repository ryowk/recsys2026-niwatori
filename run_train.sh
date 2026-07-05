#!/bin/bash
# Train-from-scratch: rebuild every fitted artifact (retriever sources, two-tower
# LoRA models, union candidates, reranker LightGBM) and then run blind-B inference
# in fit mode (which also writes artifacts/runs/reranker/.../model.txt).
# Roughly half a day on 1 GPU (16-24GB for two-tower) + 64-core CPU with >=128GB RAM.
set -eu
cd "$(dirname "$0")"

uv sync
bash run_preprocess.sh                       # splits (cv3+cv5) + spotify_uuid_map (needs TPD2, online once)

export HF_HUB_OFFLINE=1
bash scripts/build_stage1_sources.sh         # fit-free/metadata-fit sources, public_labeled + blind_b (CPU)
bash scripts/build_stage2_sources.sh         # two-tower LoRA + cooc/transition (+TPD1) sources (GPU + CPU)

UNION_CFG=blind_b_safe_combined_tpd1_parts_cooc500_cv5
RERANK_CFG=blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5

# union + 5-fold CV reranker on public_labeled (fits 5 fold models, reports CV)
uv run python retriever/union/main.py --config "$UNION_CFG" --target public_labeled
uv run python reranker/protocol_098_union_rich_lgbm/main.py --config "$RERANK_CFG" --target public_labeled

# blind side: union + reranker in FIT mode (fits the full-public model, saves
# artifacts/runs/reranker/.../full_public/blind_b/model.txt, ranks blind_b)
uv run python retriever/union/main.py --config "$UNION_CFG" --target blind_b
uv run python reranker/protocol_098_union_rich_lgbm/main.py --config "$RERANK_CFG" --target blind_b

# copy the freshly trained model into the weights slot used by run_inference.sh
mkdir -p artifacts/weights
cp "artifacts/runs/reranker/protocol_098_union_rich_lgbm/$RERANK_CFG/full_public/blind_b/model.txt" \
   artifacts/weights/reranker_lgbm.txt

# responder (same as run_inference.sh step C)
uv run python scripts/build_responder.py \
  --base-config rich_context_hierpop_tagchain \
  --ranked-artifact "artifacts/runs/reranker/protocol_098_union_rich_lgbm/$RERANK_CFG/full_public/blind_b" \
  --target blind_b \
  --out-dir "artifacts/runs/responder/qwen36_10run_diverse/$RERANK_CFG" \
  --n-runs 10 --seed 0 --selection-objective lexdiv --n-random-orders 30

echo "submission: artifacts/runs/responder/qwen36_10run_diverse/$RERANK_CFG/blind_b.submission.zip"
