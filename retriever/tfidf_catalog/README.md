# retriever/tfidf_catalog

## Logic and purpose

Fits a TF-IDF vectorizer on track metadata and retrieves by similarity to the visible conversation query. It provides a lexical signal distinct from BM25.

The component implementation is in `main.py`.

## Configuration and artifacts

`main.py` is the parameter source of truth. Inputs are catalog metadata, target conversation, and public splits. Artifacts are written to `artifacts/runs/retriever/tfidf_catalog/<config>/<artifact_mode>/<target>/` and retain TF-IDF score, source rank, and row keys.

## Fit and leakage

Fit-free. The vectorizer is fit on catalog metadata and applied to the visible conversation query.
