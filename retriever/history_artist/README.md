# retriever/history_artist

## Logic and purpose

Collects artists played before the target turn and returns other catalog tracks by those artists. It represents conversation-wide artist preference.

## Configuration and artifacts

Parameters live in `main.py`. Inputs are pre-target music history and catalog metadata. Artifacts are written to `artifacts/runs/retriever/history_artist/<config>/fit_free_all_rows/<target>/`; `score__primary` is source-local.

## Fit and leakage

Fit-free. Artist IDs come only from music rows with `turn_number < target_turn`.
