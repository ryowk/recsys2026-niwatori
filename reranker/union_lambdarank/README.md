# reranker/union_lambdarank

## Logic and purpose

Joins context-track, metadata, user and dense-query, source-presence, rank, and source-local score features on union candidates, then fits LightGBM LambdaRank. Raw scores from different sources are not treated as one calibrated scale.

The entry point is `main.py`; runner, feature, and protocol code lives in the same directory.

## Configuration and artifacts

`configs/*.yaml` is the source of truth for features, row filters, and model parameters. Inputs are union artifacts and preprocessing caches. Ranked artifacts, model, and manifest are written below `artifacts/runs/reranker/union_lambdarank/<config>/<fit_mode>/<target>/`.

## Fit and leakage

Fitting uses OOF retriever features; inference uses full-fit source features. Blind-B-unavailable feature columns remain neutral in both paths.
