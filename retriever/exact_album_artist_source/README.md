# retriever/exact_album_artist_source

## Logic and purpose

Finds album and artist strings in the current user message and, when provided, current thought, then returns exact catalog matches. It targets explicit album and artist requests.

## Configuration and artifacts

Parameters live in `main.py`. Inputs are the current turn and catalog metadata. Artifacts are written to `artifacts/runs/retriever/exact_album_artist_source/<config>/fit_free_all_rows/<target>/`.

## Fit and leakage

Fit-free. Album/artist matching reads the current message and supplied current thought, and returns catalog tracks.
