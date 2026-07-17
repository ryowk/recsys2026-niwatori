# retriever/union

## Logic and purpose

Aligns configured retriever artifacts by row key, concatenates candidates in source order, and removes duplicates. Source presence, rank, and source-local scores are realigned to the merged positions for reranking.

The entry point is `main.py`; merge logic is in adjacent `builder.py`.

## Configuration and artifacts

`configs/*.yaml` is the source of truth for source lists, caps, thresholds, and artifact modes. Inputs follow `artifacts/runs/retriever/<component>/<config>/<fit_mode>/<target>/`; outputs are candidates, turns, source features, and a manifest under `artifacts/runs/retriever/union/<config>/<target>/`.

## Fit and leakage

The union does not fit. It resolves supervised and count sources to OOF artifacts for reranker-fit rows and full-fit artifacts for inference.
