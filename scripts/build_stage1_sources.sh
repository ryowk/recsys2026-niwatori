#!/bin/bash
# Stage-1 retriever sources (CPU, fit-free / metadata-fit): bm25, tag_intent,
# tfidf, history/last/exact — for BOTH public_labeled and blind_b.
# Resumable: each step skips when its candidates.npz already exists.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

RUNS=artifacts/runs/retriever
UCFG=retriever/union/configs/union_v1.yaml
LOG="${LOG:-artifacts/runs/build_stage1_sources.log}"
mkdir -p "$(dirname "$LOG")"

have () { [ -f "$1" ]; }
ts () { echo "=== $1 $(date) ===" | tee -a "$LOG"; }
run () {
  local label="$1"; shift
  ts "run $label"
  if ! "$@" >> "$LOG" 2>&1; then
    ts "ABORT $label exit=$?"
    exit 1
  fi
  ts "done $label"
}

# --- bm25 / tag_intent (message-only query text) ---
for target in public_labeled blind_b; do
  for src in bm25_5field_thought tag_intent_bm25; do
    OUT=$RUNS/$src/top500_bsafe/fit_free_all_rows/$target/candidates.npz
    have "$OUT" && ts "skip $src $target (exists)" || \
      run "$src $target" uv run python scripts/build_basic_retrievers.py \
        --config-file "$UCFG" --config top500_bsafe --target "$target" --top-k 500 --only "$src"
  done
done

# --- history / last / exact (fit-free) ---
for target in public_labeled blind_b; do
  for src in history_artist history_album last_music_artist last_music_album \
             exact_album_artist_source exact_title_artist_source; do
    OUT=$RUNS/$src/top500/fit_free_all_rows/$target/candidates.npz
    have "$OUT" && ts "skip $src $target (exists)" || \
      run "$src $target" uv run python scripts/build_basic_retrievers.py \
        --config-file "$UCFG" --config top500 --target "$target" --top-k 500 --only "$src"
  done
done

# --- tfidf (fit on track metadata only; cv3 splits hardcoded by the runner) ---
TF=$RUNS/protocol_tfidf_lgbm_k300/protocol_v1_bsafe/fit_free_all_rows/public_labeled/candidates.npz
have "$TF" && ts "skip tfidf public (exists)" || \
  run "tfidf public" uv run python scripts/run_tfidf_lgbm.py \
    --name protocol_tfidf_lgbm_k300 --config protocol_v1_bsafe --candidate-k 300
TFB=$RUNS/protocol_tfidf_lgbm_k300/protocol_v1_bsafe/fit_free_all_rows/blind_b/candidates.npz
have "$TFB" && ts "skip tfidf blind_b (exists)" || \
  run "tfidf blind_b" uv run python scripts/run_tfidf_lgbm.py \
    --name protocol_tfidf_lgbm_k300 --config protocol_v1_bsafe --candidate-k 300 --blind-target blind_b

ts "stage1 sources done"
