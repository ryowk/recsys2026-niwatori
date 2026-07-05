#!/bin/bash
# Retriever sources for blind_b (submission prediction side) + the blind_b union.
# Two-tower full-public model is TRAINED here — no weights are loaded.
# Depends on: run_preprocess.sh (cv5 split, spotify_map, dense_track_emb).
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
LOG="${LOG:-artifacts/runs/run_retriever_blind_b.log}"; mkdir -p "$(dirname "$LOG")"
. scripts/_run_lib.sh

RUNS=artifacts/runs/retriever
SPLIT=artifacts/cache/splits/cv5
UCFG=retriever/union/configs/union_v1.yaml
BCFG=retriever/union/configs/blind_b_safe_cv5.yaml
UNION_CFG=blind_b_safe_combined_tpd1_parts_cooc500_cv5

for src in bm25_5field_thought tag_intent_bm25; do
  OUT=$RUNS/$src/top500_bsafe/fit_free_all_rows/blind_b/candidates.npz
  have "$OUT" && ts "skip $src (exists)" || run "$src" uv run python scripts/build_basic_retrievers.py \
    --config-file "$UCFG" --config top500_bsafe --target blind_b --top-k 500 --only "$src"
done
for src in history_artist history_album last_music_artist last_music_album exact_album_artist_source exact_title_artist_source; do
  OUT=$RUNS/$src/top500/fit_free_all_rows/blind_b/candidates.npz
  have "$OUT" && ts "skip $src (exists)" || run "$src" uv run python scripts/build_basic_retrievers.py \
    --config-file "$UCFG" --config top500 --target blind_b --top-k 500 --only "$src"
done
TFB=$RUNS/protocol_tfidf_lgbm_k300/protocol_v1_bsafe/fit_free_all_rows/blind_b/candidates.npz
have "$TFB" && ts "skip tfidf (exists)" || run "tfidf" uv run python scripts/run_tfidf_lgbm.py \
  --name protocol_tfidf_lgbm_k300 --config protocol_v1_bsafe --candidate-k 300 --blind-target blind_b
TTB=$RUNS/two_tower_lora_thought/oof5_top500_bsafe/full_public/blind_b/candidates.npz
have "$TTB" && ts "skip two_tower (exists)" || run "two_tower full_public (train)" uv run python scripts/build_two_tower_lora_oof.py \
  --mode full_public --config oof5_top500_bsafe --split-dir "$SPLIT" --blind-target blind_b --epochs 2
for src_cfg in "cooc_album oof5_top500" "cooc_artist_name oof5_top500"; do
  set -- $src_cfg; SRC=$1; CFG=$2
  OUT=$RUNS/$SRC/$CFG/full_public/blind_b/candidates.npz
  have "$OUT" && ts "skip $SRC (exists)" || run "$SRC" uv run python scripts/build_train_fit_retriever_artifacts.py \
    --config-file "$BCFG" --config "$CFG" --source "$SRC" --mode blind --blind-target blind_b --top-k 500 --split-dir "$SPLIT"
done
for src_cfg in "cooc_track_combined_tpd1 oof5_top500_parts" "transition_track_combined_tpd1 oof5_top500_prob_parts"; do
  set -- $src_cfg; SRC=$1; CFG=$2; CFILE=retriever/$SRC/configs/$CFG.yaml
  OUT=$RUNS/$SRC/$CFG/full_public/blind_b/candidates.npz
  have "$OUT" && ts "skip $SRC (exists)" || run "$SRC" uv run python scripts/build_combined_tpd1_retrievers.py \
    --source "$SRC" --config "$CFG" --config-file "$CFILE" --target blind_b --split-dir "$SPLIT" --offline
done
UNB=$RUNS/union/$UNION_CFG/blind_b/candidates.npz
have "$UNB" && ts "skip union blind_b (exists)" || run "union blind_b" uv run python retriever/union/main.py --config "$UNION_CFG" --target blind_b
ts "retriever_blind_b done"
