# retriever/cooc_track_combined_tpd1

## Logic and purpose

Combines track co-occurrence from challenge sessions and catalog-mapped TPD1 sessions. Combined, challenge-only, and TPD1 scores remain separate in the artifact.

## Configuration and artifacts

Component and union configs are the parameter and downstream-cap sources of truth. Artifacts are written to `artifacts/runs/retriever/cooc_track_combined_tpd1/<config>/<fit_mode>/<target>/`.

## Fit and leakage

Challenge co-occurrence statistics are OOF for fit rows and full-fit for inference. The fold-independent TPD1 table is mapped to the challenge catalog. Queries use tracks in pre-target history.
