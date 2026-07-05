#!/bin/bash
# Blind-B from the trained weights (AUXILIARY reproduce-from-weights path).
#
# The official, fast, bit-exact reproduction is run_inference.sh (load the
# shipped LightGBM model + the shipped union artifact). This script instead
# REGENERATES the retriever artifacts locally from the trained two-tower weights
# (--load-models-dir: load + encode, no training) and ranks blind_b — a check
# that the shipped seed artifacts are reproducible from weights.
#
# It is GPU-heavy (~2h) and NOT bit-reproducible: the dense / two-tower encodes
# run in bf16 on the GPU, so the ranking is close (high top-20 overlap) but not
# identical. Verify with the soft report (verify_blind_b_ranking.py), not --strict.
#
# Needs: challenge datasets + TalkPlayData-1/2 + Qwen in the HF cache, a GPU, and
# the two-tower weights under artifacts/weights/two_tower/.
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TWO_TOWER_WEIGHTS="${TWO_TOWER_WEIGHTS:-$PWD/artifacts/weights/two_tower}"
RUN_RESPONDER="${RUN_RESPONDER:-1}"

UNION_CFG=blind_b_safe_combined_tpd1_parts_cooc500_cv5
RERANK_CFG=blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5

# 1. preprocessing: CV splits + TPD1->catalog map + dense track embedding cache
bash run_preprocess.sh
uv run python preprocessing/dense_track_encoder.py --target blind_b   # -> artifacts/cache/dense_track_emb.npz (GPU)

# 2. retriever sources for public_labeled AND blind_b
bash scripts/build_stage1_sources.sh                                  # fit-free / metadata-fit (CPU)
bash scripts/build_stage2_sources.sh                                  # two-tower (load weights, encode) + cooc/transition

# 3. unions. The public union only needs candidates.npz for the reranker's
#    feature-stack fit (--load-model never reads its 36GB source_features), so
#    build it candidates-only. The blind union needs its source_features.
uv run python retriever/union/main.py --config "$UNION_CFG" --target public_labeled --no-source-features
uv run python retriever/union/main.py --config "$UNION_CFG" --target blind_b

# 4. reranker: load the shipped LightGBM model, rank blind_b (no fit)
uv run python reranker/protocol_098_union_rich_lgbm/main.py \
  --config "$RERANK_CFG" --target blind_b --load-model artifacts/weights/reranker_lgbm.txt

# 5. check top-20 overlap vs the submitted ranking (soft — the GPU encodes above
#    are not bit-reproducible, so expect high overlap, not an exact match)
uv run python scripts/verify_blind_b_ranking.py

# 6. responder (GPU; set RUN_RESPONDER=0 to skip)
if [ "$RUN_RESPONDER" = "1" ]; then
  uv run python scripts/build_responder.py \
    --base-config rich_context_hierpop_tagchain \
    --ranked-artifact "artifacts/runs/reranker/protocol_098_union_rich_lgbm/$RERANK_CFG/full_public/blind_b" \
    --target blind_b \
    --out-dir "artifacts/runs/responder/qwen36_10run_diverse/$RERANK_CFG" \
    --n-runs 10 --seed 0 --selection-objective lexdiv --n-random-orders 30
  echo "submission: artifacts/runs/responder/qwen36_10run_diverse/$RERANK_CFG/blind_b.submission.zip"
fi
