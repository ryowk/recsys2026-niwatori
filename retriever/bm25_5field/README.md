# retriever/bm25_5field

## Logic and purpose

Builds a BM25 query from the current user message and observed conversation, then searches catalog track, artist, album, tag, and release-date fields. It supplies the union's broad lexical pool.

## Configuration and artifacts

`main.py` and `retriever/fit_free_sources.yaml` are the parameter sources of truth. Artifacts are written to `artifacts/runs/retriever/bm25_5field/<config>/fit_free_all_rows/<target>/`.

## Fit and leakage

Fit-free. The index is built from catalog metadata; each query uses the current message and observed conversation.
