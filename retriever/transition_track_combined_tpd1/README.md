# retriever/transition_track_combined_tpd1

## Logic and purpose

Combines next-track transitions from challenge sessions and catalog-mapped TPD1 sessions, then retrieves successors of the last observed track. Combined, challenge-only, and TPD1 scores remain separate.

## Configuration and artifacts

Component and union configs are the parameter and downstream-cap sources of truth. Artifacts are written to `artifacts/runs/retriever/transition_track_combined_tpd1/<config>/<fit_mode>/<target>/`.

## Fit and leakage

Challenge transition statistics are OOF for fit rows and full-fit for inference. The fold-independent TPD1 table is mapped to the challenge catalog. The transition starts from the latest pre-target track.
