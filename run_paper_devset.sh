#!/bin/bash
# Train-to-Devset paper evaluation; see docs/paper_evaluation.md.
set -euo pipefail
cd "$(dirname "$0")"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
RUNS=artifacts/runs
CV5=artifacts/preprocessed/splits/cv5
SPLIT=artifacts/preprocessed/splits/paper_train_cv5
UCFG=retriever/union/configs/paper_train5_devset.yaml
BASE_UCFG=retriever/fit_free_sources.yaml
LOG="${LOG:-artifacts/runs/paper_train5_devset.log}"
mkdir -p "$(dirname "$LOG")"
. scripts/_run_lib.sh

run() {
  local label="$1"
  shift
  printf '=== %s %s ===\n' "$label" "$(date -u +%FT%TZ)" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

split_artifact_complete "$CV5" || run "public 5-fold split" \
  uv run python scripts/build_public_splits.py \
    --out-dir "$CV5" --name public_labeled_v2_5fold --seed 20260515 \
    --n-splits 5

split_artifact_complete "$SPLIT" || run "train-only 5-fold split" \
  uv run python scripts/build_public_splits.py \
    --out-dir "$SPLIT" --name paper_train_cv5 --seed 20260515 \
    --n-splits 5 --source-splits train

if ! have artifacts/preprocessed/catalog_id_map.parquet || \
   ! have artifacts/preprocessed/catalog_id_map.manifest.json; then
  run "catalog ID map" \
  uv run python scripts/build_catalog_id_map.py
fi
# The encoder validates the NPZ schema before deciding that it can be reused.
run "dense track features" uv run python preprocessing/dense_track_encoder.py

# Fit-free sources are functions of the target row and track catalog.  Build
# the public-labeled base artifacts before slicing them into Train and Devset.
for source in bm25_5field tag_intent_bm25; do
  artifact="$RUNS/retriever/$source/top500/fit_free_all_rows/public_labeled"
  candidate_artifact_complete "$artifact" || run "$source public-labeled" \
    uv run python -m "retriever.$source.main" \
      --config-file "$BASE_UCFG" --config top500 \
      --target public_labeled --top-k 500
done
for source in history_artist history_album last_music_artist last_music_album \
              exact_album_artist_source exact_title_artist_source; do
  artifact="$RUNS/retriever/$source/top500/fit_free_all_rows/public_labeled"
  candidate_artifact_complete "$artifact" || run "$source public-labeled" \
    uv run python -m "retriever.$source.main" \
      --config-file "$BASE_UCFG" --config top500 \
      --target public_labeled --top-k 500
done
TFIDF="$RUNS/retriever/tfidf_catalog/top300/fit_free_all_rows/public_labeled"
candidate_artifact_complete "$TFIDF" || run "tfidf public-labeled" \
  uv run python -m retriever.tfidf_catalog.main --target public_labeled

while read -r component config; do
  input="$RUNS/retriever/$component/$config/fit_free_all_rows/public_labeled"
  output="$RUNS/retriever/$component/$config/fit_free_train5_dev"
  for source_split in train devset; do
    target=devset
    [ "$source_split" = train ] && target=public_labeled
    candidate_artifact_complete "$output/$target" || run "slice $component $source_split" \
      uv run python scripts/slice_fit_free_retriever_artifact.py \
        --input "$input" --output "$output/$target" \
        --source-split "$source_split" --split-dir "$SPLIT"
  done
done <<'EOF'
bm25_5field top500
tfidf_catalog top300
history_artist top500
history_album top500
last_music_artist top500
last_music_album top500
exact_album_artist_source top500
tag_intent_bm25 top500
exact_title_artist_source top500
EOF

TT="$RUNS/retriever/two_tower_lora/paper_train5_top500"
candidate_artifact_complete "$TT/train5_oof/public_labeled" || run "two-tower train5 OOF" \
  uv run python -m retriever.two_tower_lora.main \
    --mode train5_oof --config paper_train5_top500 \
    --split-dir "$SPLIT" --epochs 2 --batch-size 64 --top-k 500
candidate_artifact_complete "$TT/full_train/devset" || run "two-tower full-train devset" \
  uv run python -m retriever.two_tower_lora.main \
    --mode full_train --config paper_train5_top500 \
    --split-dir "$SPLIT" --inference-target devset \
    --epochs 2 --batch-size 64 --top-k 500

for source in cooc_album cooc_artist_name; do
  base="$RUNS/retriever/$source/paper_train5_top500"
  if ! candidate_artifact_complete "$base/train5_oof/public_labeled" || \
     ! candidate_artifact_complete "$base/full_train/devset"; then
    run "$source train5/full-train" \
      uv run python -m "retriever.$source.main" \
        --config-file "$UCFG" \
        --config paper_train5_top500 --mode both \
        --inference-target devset --top-k 500 --split-dir "$SPLIT" \
        --artifact-mode train5_oof
  fi
done

while read -r source config config_file; do
  base="$RUNS/retriever/$source/$config"
  candidate_artifact_complete "$base/train5_oof/public_labeled" || \
    run "$source train5 OOF" uv run python -m "retriever.$source.main" \
      --config "$config" --config-file "$config_file" \
      --target public_labeled --split-dir "$SPLIT" \
      --artifact-mode train5_oof
  candidate_artifact_complete "$base/full_train/devset" || \
    run "$source full-train devset" uv run python -m "retriever.$source.main" \
      --config "$config" --config-file "$config_file" \
      --target devset --split-dir "$SPLIT"
done <<'EOF'
cooc_track_combined_tpd1 paper_train5_top500_parts retriever/cooc_track_combined_tpd1/configs/oof5_top500_parts.yaml
transition_track_combined_tpd1 paper_train5_top500_prob_parts retriever/transition_track_combined_tpd1/configs/oof5_top500_prob_parts.yaml
EOF

UNION="$RUNS/retriever/union/paper_train5_devset"
union_artifact_complete "$UNION/public_labeled" || run "train OOF union" \
  uv run python -m retriever.union.main --config paper_train5_devset \
    --target public_labeled
union_artifact_complete "$UNION/devset" || run "devset union" \
  uv run python -m retriever.union.main --config paper_train5_devset \
    --target devset

for variant in full no_provenance provenance_only; do
  config="paper_train5_devset_$variant"
  ranked="$RUNS/reranker/union_lambdarank/$config/full_train/devset"
  ranked_artifact_complete "$ranked" || run "reranker $variant" \
    uv run python -m reranker.union_lambdarank.main \
      --config "$config" --target devset
done

# External-data ablation: reuse every non-TPD1 artifact, rebuild only the two
# combined statistical sources, their unions, and one full-feature reranker.
while read -r source config config_file; do
  base="$RUNS/retriever/$source/$config"
  candidate_artifact_complete "$base/train5_oof/public_labeled" || \
    run "$source without TPD1 train5 OOF" uv run python -m "retriever.$source.main" \
      --config "$config" --config-file "$config_file" \
      --target public_labeled --split-dir "$SPLIT" --artifact-mode train5_oof \
      --disable-tpd1
  candidate_artifact_complete "$base/full_train/devset" || \
    run "$source without TPD1 devset" uv run python -m "retriever.$source.main" \
      --config "$config" --config-file "$config_file" \
      --target devset --split-dir "$SPLIT" --disable-tpd1
done <<'EOF'
cooc_track_combined_tpd1 paper_train5_top500_parts_no_tpd1 retriever/cooc_track_combined_tpd1/configs/oof5_top500_parts.yaml
transition_track_combined_tpd1 paper_train5_top500_prob_parts_no_tpd1 retriever/transition_track_combined_tpd1/configs/oof5_top500_prob_parts.yaml
EOF

NO_TPD_UNION="$RUNS/retriever/union/paper_train5_devset_without_tpd1"
for target in public_labeled devset; do
  union_artifact_complete "$NO_TPD_UNION/$target" || run "without TPD1 union $target" \
    uv run python -m retriever.union.builder \
      --config paper_train5_devset_without_tpd1 \
      --config-file "$UCFG" --target "$target" \
      --source-override "cooc_track=$RUNS/retriever/cooc_track_combined_tpd1/paper_train5_top500_parts_no_tpd1/$( [ "$target" = public_labeled ] && echo train5_oof/public_labeled || echo full_train/devset )" \
      --source-override "transition_track=$RUNS/retriever/transition_track_combined_tpd1/paper_train5_top500_prob_parts_no_tpd1/$( [ "$target" = public_labeled ] && echo train5_oof/public_labeled || echo full_train/devset )"
done

NO_TPD_RANKED="$RUNS/reranker/union_lambdarank/paper_train5_devset_without_tpd1/full_train/devset"
ranked_artifact_complete "$NO_TPD_RANKED" || run "reranker without TPD1" \
  uv run python -m reranker.union_lambdarank.main \
    --config paper_train5_devset_without_tpd1 --target devset

run "paper analysis" uv run python scripts/analyze_paper_results.py
