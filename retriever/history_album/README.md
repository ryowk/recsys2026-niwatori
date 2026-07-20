# retriever/history_album

## Logic and purpose

Collects albums played before the target turn and returns other catalog tracks from those albums. It targets continuation within an observed album.

## Configuration and artifacts

Parameters live in `main.py`. Inputs are pre-target music history and catalog metadata. Artifacts are written to `artifacts/runs/retriever/history_album/<config>/fit_free_all_rows/<target>/`; `score__primary` is source-local.

## Fit and leakage

Fit-free. Album IDs come only from music rows with `turn_number < target_turn`.
