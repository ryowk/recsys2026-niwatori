# responder/qwen36_27b

## Logic and purpose

Prompts Qwen3.6-27B with top-ranked tracks and available conversation context. It generates response text without changing track IDs or order, then selects one seeded candidate by lexical diversity.

The entry point is `main.py`; prompt and generation code is in `component.py`, and selection is in `ensemble.py`.

## Configuration and artifacts

`configs/default.yaml` is the parameter source of truth. The input is a ranked artifact. Resumable runs, selection metadata, prediction JSON, ZIP, and manifest are written under `artifacts/runs/responder/qwen36_27b/default/<target>/`.

## Fit and leakage

The component does not fit on challenge labels or fold assignments, does not impute missing Blind-B context, and validates the submission schema without modifying the ranked tracks.
