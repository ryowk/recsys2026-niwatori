# retriever/last_music_album

## Logic and purpose

Returns catalog tracks from the album of the last music track before the target turn. It provides an album-level recency signal.

## Configuration and artifacts

Parameters live in `main.py`. Inputs are pre-target music history and catalog metadata. Artifacts are written to `artifacts/runs/retriever/last_music_album/<config>/fit_free_all_rows/<target>/`.

## Fit and leakage

Fit-free. The seed album comes from the latest pre-target music turn.
