# retriever/cooc_album

## Logic and purpose

Counts album co-occurrence in labeled sessions and expands albums near the observed history to catalog tracks. It provides a broader album-level signal than exact history matching.

## Configuration and artifacts

Component configs define the candidate width; union configs define downstream caps and thresholds. Artifacts are written to `artifacts/runs/retriever/cooc_album/<config>/<fit_mode>/<target>/`; `score__primary` is the album co-occurrence count.

## Fit and leakage

Album counts are OOF for reranker-fit rows and full-fit for inference. Expansion starts from albums in pre-target history.
