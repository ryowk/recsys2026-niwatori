# Reproducing the Paper Evaluation

## Evaluation protocol

The paper uses the `Train -> Devset` execution defined in [`folds.md`](folds.md), with the submitted source list and model configuration.

## Requirements

Complete the setup in the repository [`README.md`](../README.md). Compute requirements are in [`runtime.md`](runtime.md).

## Run the complete evaluation

```bash
bash run_paper_devset.sh
```

The command builds the required retriever, union, reranker, ablation, and analysis artifacts in dependency order. Completed stages are reused.

The driver finishes by running:

```bash
uv run python scripts/analyze_paper_results.py
```

## Implementation map

| Path | Purpose |
|---|---|
| `run_paper_devset.sh` | Single entry point from split construction through final analysis |
| `retriever/union/configs/paper_train5_devset.yaml` | Resolves the submitted 14-source union, caps, and thresholds for Train and Devset artifacts |
| `reranker/union_lambdarank/configs/paper_train5_devset_*.yaml` | Full, context-track-only, per-retriever-only, and no-TPD1 reranker variants |
| `scripts/slice_fit_free_retriever_artifact.py` | Splits fit-free public artifacts into Train and Devset artifacts with row-key validation |
| `scripts/analyze_paper_results.py` | Computes retrieval, union, RRF, reranker-ablation, and error-decomposition results and generates paper tables and figures |

## Canonical outputs

The canonical machine-readable result is:

```text
artifacts/results/paper/train5_devset/report.json
```

The same directory contains per-retriever, complementarity, ranking, ablation, and error-breakdown CSV files and the generated Figure 3.

`scripts/analyze_paper_results.py` validates source coverage, row-key alignment, and configuration parity before updating `paper/generated_*.tex`, `paper/results_mode.tex`, and `paper/figures/retriever_recall_vs_size.pdf`.

Paper build instructions are in [`paper/README.md`](../paper/README.md).
