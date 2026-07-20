# retriever/tag_intent_bm25

## Logic and purpose

Extracts genre, mood, and usage terms from the current message and searches catalog tags with BM25. It covers intent requests that do not name a track or artist.

## Configuration and artifacts

`main.py` and `retriever/fit_free_sources.yaml` are the parameter sources of truth. Artifacts are written to `artifacts/runs/retriever/tag_intent_bm25/<config>/fit_free_all_rows/<target>/`.

## Fit and leakage

Fit-free. The index uses catalog tags and the query uses the current message.
