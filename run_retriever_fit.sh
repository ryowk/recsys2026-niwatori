#!/bin/bash
# Build OOF retriever artifacts and the public union.
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
LOG="${LOG:-artifacts/runs/run_retriever_fit.log}"; mkdir -p "$(dirname "$LOG")"
. scripts/_run_lib.sh

RUNS=artifacts/runs/retriever
SPLIT=artifacts/preprocessed/splits/cv5
UCFG=retriever/fit_free_sources.yaml
BCFG=retriever/union/configs/combined_tpd1_parts_cooc500_cv5.yaml
UNION_CFG=combined_tpd1_parts_cooc500_cv5

for src in bm25_5field tag_intent_bm25; do
  OUT=$RUNS/$src/top500/fit_free_all_rows/public_labeled
  candidate_artifact_complete "$OUT" && ts "skip $src (complete)" || run "$src" uv run python -m "retriever.$src.main" \
    --config-file "$UCFG" --config top500 --target public_labeled --top-k 500
done
for src in history_artist history_album last_music_artist last_music_album exact_album_artist_source exact_title_artist_source; do
  OUT=$RUNS/$src/top500/fit_free_all_rows/public_labeled
  candidate_artifact_complete "$OUT" && ts "skip $src (complete)" || run "$src" uv run python -m "retriever.$src.main" \
    --config-file "$UCFG" --config top500 --target public_labeled --top-k 500
done
TF=$RUNS/tfidf_catalog/top300/fit_free_all_rows/public_labeled
candidate_artifact_complete "$TF" && ts "skip tfidf (complete)" || run "tfidf" uv run python -m retriever.tfidf_catalog.main --target public_labeled
TT=$RUNS/two_tower_lora/oof5_top500/cv5_oof/public_labeled
candidate_artifact_complete "$TT" && ts "skip two_tower (complete)" || run "two_tower cv5 (train 5 folds)" uv run python -m retriever.two_tower_lora.main \
  --mode cv5_oof --config oof5_top500 --split-dir "$SPLIT" --epochs 2
for src_cfg in "cooc_album oof5_top500" "cooc_artist_name oof5_top500"; do
  set -- $src_cfg; SRC=$1; CFG=$2
  OUT=$RUNS/$SRC/$CFG/cv5_oof/public_labeled
  candidate_artifact_complete "$OUT" && ts "skip $SRC (complete)" || run "$SRC" uv run python -m "retriever.$SRC.main" \
    --config-file "$BCFG" --config "$CFG" --mode public --top-k 500 --split-dir "$SPLIT" --artifact-mode cv5_oof
done
for src_cfg in "cooc_track_combined_tpd1 oof5_top500_parts" "transition_track_combined_tpd1 oof5_top500_prob_parts"; do
  set -- $src_cfg; SRC=$1; CFG=$2; CFILE=retriever/$SRC/configs/$CFG.yaml
  OUT=$RUNS/$SRC/$CFG/cv5_oof/public_labeled
  candidate_artifact_complete "$OUT" && ts "skip $SRC (complete)" || run "$SRC" uv run python -m "retriever.$SRC.main" \
    --config "$CFG" --config-file "$CFILE" --target public_labeled --split-dir "$SPLIT" --artifact-mode cv5_oof
done
UN=$RUNS/union/$UNION_CFG/public_labeled
union_artifact_complete "$UN" && ts "skip union public (complete)" || run "union public" uv run python -m retriever.union.main --config "$UNION_CFG" --target public_labeled
ts "retriever_fit done"
