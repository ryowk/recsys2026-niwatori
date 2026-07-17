# retriever/two_tower_lora

## Logic and purpose

Trains a Qwen-embedding LoRA query tower against a track tower built from metadata, audio, image, collaborative, and related features. It adds semantic matching beyond lexical and count sources.

Artifact execution is in `main.py`; model code is in adjacent `model.py`.

## Configuration and artifacts

CLI defaults and component configs are the parameter sources of truth. Artifacts are written to `artifacts/runs/retriever/two_tower_lora/<config>/<fit_mode>/<target>/`.

## Fit and leakage

Reranker-fit rows use fold-excluded models; inference uses a full-fit model whose training population excludes the inference target.
