#!/bin/bash
# Stage-2 retriever sources (labeled-fit): two-tower LoRA (GPU), plain cooc
# (cooc_album / cooc_artist_name) and combined-TPD1 cooc/transition — 5-fold OOF
# artifacts for public_labeled and full-public artifacts for blind_b.
# Resumable: each step skips when its candidates.npz already exists.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

RUNS=artifacts/runs/retriever
SPLIT=artifacts/cache/splits/cv5
BCFG=retriever/union/configs/blind_b_safe_cv5.yaml
# When TWO_TOWER_WEIGHTS points at a weights dir (full_public.pt + fold0..4.pt),
# the two-tower steps load those and encode instead of training (minutes vs hours).
TT_WEIGHTS="${TWO_TOWER_WEIGHTS:-}"
TT_LOAD=""
[ -n "$TT_WEIGHTS" ] && TT_LOAD="--load-models-dir $TT_WEIGHTS"
LOG="${LOG:-artifacts/runs/build_stage2_sources.log}"
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

# --- two-tower LoRA: 5 fold models (OOF, train rows) + full-public model (blind) — GPU ---
TT=$RUNS/two_tower_lora_thought/oof5_top500_bsafe/cv5_oof/public_labeled/candidates.npz
have "$TT" && ts "skip two_tower cv5 public (exists)" || \
  run "two_tower cv5 public" uv run python scripts/build_two_tower_lora_oof.py \
    --mode cv5_oof --config oof5_top500_bsafe --split-dir "$SPLIT" --epochs 2 $TT_LOAD
TTB=$RUNS/two_tower_lora_thought/oof5_top500_bsafe/full_public/blind_b/candidates.npz
have "$TTB" && ts "skip two_tower full_public blind_b (exists)" || \
  run "two_tower full_public blind_b" uv run python scripts/build_two_tower_lora_oof.py \
    --mode full_public --config oof5_top500_bsafe --split-dir "$SPLIT" --blind-target blind_b --epochs 2 $TT_LOAD

# --- plain cooc sources used by the final union (challenge statistics only) ---
for src_cfg in "cooc_album oof5_top500" "cooc_artist_name oof5_top500"; do
  set -- $src_cfg; SRC=$1; CFG=$2
  OUT=$RUNS/$SRC/$CFG/cv5_oof/public_labeled/candidates.npz
  have "$OUT" && ts "skip $SRC cv5 public (exists)" || \
    run "$SRC cv5 public" uv run python scripts/build_train_fit_retriever_artifacts.py \
      --config-file "$BCFG" --config "$CFG" --source "$SRC" --mode public --top-k 500 \
      --split-dir "$SPLIT" --artifact-mode cv5_oof
  OUTB=$RUNS/$SRC/$CFG/full_public/blind_b/candidates.npz
  have "$OUTB" && ts "skip $SRC blind_b (exists)" || \
    run "$SRC blind_b" uv run python scripts/build_train_fit_retriever_artifacts.py \
      --config-file "$BCFG" --config "$CFG" --source "$SRC" --mode blind --blind-target blind_b --top-k 500 \
      --split-dir "$SPLIT"
done

# --- combined-TPD1 cooc/transition (challenge counts + TalkPlayData-1 counts) ---
for src_cfg in \
  "cooc_track_combined_tpd1 oof5_top500_parts" \
  "transition_track_combined_tpd1 oof5_top500_prob_parts"; do
  set -- $src_cfg; SRC=$1; CFG=$2
  CFILE=retriever/$SRC/configs/$CFG.yaml
  OUT=$RUNS/$SRC/$CFG/cv5_oof/public_labeled/candidates.npz
  have "$OUT" && ts "skip $SRC cv5 public (exists)" || \
    run "$SRC cv5 public" uv run python scripts/build_combined_tpd1_retrievers.py \
      --source "$SRC" --config "$CFG" --config-file "$CFILE" \
      --target public_labeled --split-dir "$SPLIT" --artifact-mode cv5_oof --offline
  OUTB=$RUNS/$SRC/$CFG/full_public/blind_b/candidates.npz
  have "$OUTB" && ts "skip $SRC blind_b (exists)" || \
    run "$SRC blind_b" uv run python scripts/build_combined_tpd1_retrievers.py \
      --source "$SRC" --config "$CFG" --config-file "$CFILE" \
      --target blind_b --split-dir "$SPLIT" --offline
done

ts "stage2 sources done"
