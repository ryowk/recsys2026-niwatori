#!/bin/bash
# Retriever sources for public_labeled (reranker-train / CV side) + the public union.
# Two-tower is TRAINED here (5-fold OOF models) — no weights are loaded.
# Depends on: run_preprocess.sh (cv5 split, spotify_map, dense_track_emb).
set -u
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
LOG="${LOG:-artifacts/runs/run_retriever_cv5.log}"; mkdir -p "$(dirname "$LOG")"
. scripts/_run_lib.sh

RUNS=artifacts/runs/retriever
SPLIT=artifacts/cache/splits/cv5
UCFG=retriever/union/configs/union_v1.yaml
BCFG=retriever/union/configs/blind_b_safe_cv5.yaml
UNION_CFG=blind_b_safe_combined_tpd1_parts_cooc500_cv5

for src in bm25_5field_thought tag_intent_bm25; do
  OUT=$RUNS/$src/top500_bsafe/fit_free_all_rows/public_labeled/candidates.npz
  have "$OUT" && ts "skip $src (exists)" || run "$src" uv run python scripts/build_basic_retrievers.py \
    --config-file "$UCFG" --config top500_bsafe --target public_labeled --top-k 500 --only "$src"
done
for src in history_artist history_album last_music_artist last_music_album exact_album_artist_source exact_title_artist_source; do
  OUT=$RUNS/$src/top500/fit_free_all_rows/public_labeled/candidates.npz
  have "$OUT" && ts "skip $src (exists)" || run "$src" uv run python scripts/build_basic_retrievers.py \
    --config-file "$UCFG" --config top500 --target public_labeled --top-k 500 --only "$src"
done
TF=$RUNS/protocol_tfidf_lgbm_k300/protocol_v1_bsafe/fit_free_all_rows/public_labeled/candidates.npz
have "$TF" && ts "skip tfidf (exists)" || run "tfidf" uv run python scripts/run_tfidf_lgbm.py \
  --name protocol_tfidf_lgbm_k300 --config protocol_v1_bsafe --candidate-k 300
TT=$RUNS/two_tower_lora_thought/oof5_top500_bsafe/cv5_oof/public_labeled/candidates.npz
have "$TT" && ts "skip two_tower (exists)" || run "two_tower cv5 (train 5 folds)" uv run python scripts/build_two_tower_lora_oof.py \
  --mode cv5_oof --config oof5_top500_bsafe --split-dir "$SPLIT" --epochs 2
for src_cfg in "cooc_album oof5_top500" "cooc_artist_name oof5_top500"; do
  set -- $src_cfg; SRC=$1; CFG=$2
  OUT=$RUNS/$SRC/$CFG/cv5_oof/public_labeled/candidates.npz
  have "$OUT" && ts "skip $SRC (exists)" || run "$SRC" uv run python scripts/build_train_fit_retriever_artifacts.py \
    --config-file "$BCFG" --config "$CFG" --source "$SRC" --mode public --top-k 500 --split-dir "$SPLIT" --artifact-mode cv5_oof
done
for src_cfg in "cooc_track_combined_tpd1 oof5_top500_parts" "transition_track_combined_tpd1 oof5_top500_prob_parts"; do
  set -- $src_cfg; SRC=$1; CFG=$2; CFILE=retriever/$SRC/configs/$CFG.yaml
  OUT=$RUNS/$SRC/$CFG/cv5_oof/public_labeled/candidates.npz
  have "$OUT" && ts "skip $SRC (exists)" || run "$SRC" uv run python scripts/build_combined_tpd1_retrievers.py \
    --source "$SRC" --config "$CFG" --config-file "$CFILE" --target public_labeled --split-dir "$SPLIT" --artifact-mode cv5_oof --offline
done
UN=$RUNS/union/$UNION_CFG/public_labeled/candidates.npz
have "$UN" && ts "skip union public (exists)" || run "union public" uv run python retriever/union/main.py --config "$UNION_CFG" --target public_labeled
ts "retriever_cv5 done"
