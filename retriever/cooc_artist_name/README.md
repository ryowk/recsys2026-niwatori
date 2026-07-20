# retriever/cooc_artist_name

## Logic and purpose

Counts co-occurrence between normalized artist names in labeled sessions and expands neighboring artists to catalog tracks. It provides a broader related-artist signal than exact artist IDs.

## Configuration and artifacts

Component configs define the candidate width; union configs define downstream caps. Artifacts are written to `artifacts/runs/retriever/cooc_artist_name/<config>/<fit_mode>/<target>/`; `score__primary` is the artist-name co-occurrence count.

## Fit and leakage

Artist-name counts are OOF for reranker-fit rows and full-fit for inference. Expansion starts from artist names in pre-target history.
