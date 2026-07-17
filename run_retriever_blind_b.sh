#!/bin/bash
# Build full-fit Blind-B retriever artifacts and their union.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
LOG="${LOG:-artifacts/runs/run_retriever_blind_b.log}"; mkdir -p "$(dirname "$LOG")"
. scripts/_run_lib.sh

RUNS=artifacts/runs/retriever
SPLIT=artifacts/preprocessed/splits/cv5
UCFG=retriever/fit_free_sources.yaml
BCFG=retriever/union/configs/combined_tpd1_parts_cooc500_cv5.yaml
UNION_CFG=combined_tpd1_parts_cooc500_cv5

for src in bm25_5field tag_intent_bm25; do
  OUT=$RUNS/$src/top500/fit_free_all_rows/blind_b
  candidate_artifact_complete "$OUT" && ts "skip $src (complete)" || run "$src" uv run python -m "retriever.$src.main" \
    --config-file "$UCFG" --config top500 --target blind_b --top-k 500
done
for src in history_artist history_album last_music_artist last_music_album exact_album_artist_source exact_title_artist_source; do
  OUT=$RUNS/$src/top500/fit_free_all_rows/blind_b
  candidate_artifact_complete "$OUT" && ts "skip $src (complete)" || run "$src" uv run python -m "retriever.$src.main" \
    --config-file "$UCFG" --config top500 --target blind_b --top-k 500
done
TFB=$RUNS/tfidf_catalog/top300/fit_free_all_rows/blind_b
candidate_artifact_complete "$TFB" && ts "skip tfidf (complete)" || run "tfidf" uv run python -m retriever.tfidf_catalog.main --target blind_b
TTB=$RUNS/two_tower_lora/oof5_top500/full_public/blind_b
candidate_artifact_complete "$TTB" && ts "skip two_tower (complete)" || run "two_tower full_public (train)" uv run python -m retriever.two_tower_lora.main \
  --mode full_public --config oof5_top500 --split-dir "$SPLIT" --inference-target blind_b --epochs 2
for src_cfg in "cooc_album oof5_top500" "cooc_artist_name oof5_top500"; do
  set -- $src_cfg; SRC=$1; CFG=$2
  OUT=$RUNS/$SRC/$CFG/full_public/blind_b
  candidate_artifact_complete "$OUT" && ts "skip $SRC (complete)" || run "$SRC" uv run python -m "retriever.$SRC.main" \
    --config-file "$BCFG" --config "$CFG" --mode inference --inference-target blind_b --top-k 500 --split-dir "$SPLIT"
done
for src_cfg in "cooc_track_combined_tpd1 oof5_top500_parts" "transition_track_combined_tpd1 oof5_top500_prob_parts"; do
  set -- $src_cfg; SRC=$1; CFG=$2; CFILE=retriever/$SRC/configs/$CFG.yaml
  OUT=$RUNS/$SRC/$CFG/full_public/blind_b
  candidate_artifact_complete "$OUT" && ts "skip $SRC (complete)" || run "$SRC" uv run python -m "retriever.$SRC.main" \
    --config "$CFG" --config-file "$CFILE" --target blind_b --split-dir "$SPLIT"
done
UNB=$RUNS/union/$UNION_CFG/blind_b
union_artifact_complete "$UNB" && ts "skip union blind_b (complete)" || run "union blind_b" uv run python -m retriever.union.main --config "$UNION_CFG" --target blind_b
ts "retriever_blind_b done"
