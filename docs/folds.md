# Fit Scope and Fold Handling

This repository supports two executions of the same submitted pipeline:

| execution | reranker-fit population | inference target |
|---|---|---|
| Blind-B submission | Train + Devset | Blind B |
| paper evaluation | Train | Devset |

Five folds are used only to create leakage-safe retriever candidates and features for reranker training rows. The repository does not expose a separate cross-validation evaluation workflow.

## Blind-B submission

`run_full.sh` uses `artifacts/preprocessed/splits/cv5`.

1. Fit-free retrievers use the visible row context and track catalog directly.
2. Learned and count-based retrievers produce five-fold out-of-fold (OOF) candidates for every Train + Devset row. Each row is scored by a model or count table fitted without that row's fold.
3. The same retrievers are fitted on all Train + Devset rows and applied to Blind B.
4. One LambdaRank model is fitted on gold-in-pool public rows using the OOF retriever features, then applied to the full-public Blind-B union.
5. The responder reads the resulting top 20 and generates the final text. It has no folds and does not fit on ranking labels.

## Paper evaluation

`run_paper_devset.sh` uses `artifacts/preprocessed/splits/paper_train_cv5`.

1. Learned and count-based retrievers produce five-fold OOF candidates for Train rows only.
2. Those retrievers are fitted on all Train rows and applied once to Devset.
3. One LambdaRank model is fitted on gold-in-pool Train rows using the OOF retriever features, then applied to the full-Train Devset union.
4. The responder is not run.

This is the direct fit-scope translation from `Train + Devset -> Blind B` to `Train -> Devset`.

## Artifact modes

| mode | meaning | consumer |
|---|---|---|
| `fit_free_all_rows` | no labeled fit | Blind-B fit rows and inference rows |
| `fit_free_train5_dev` | Train/Devset slices of a fit-free artifact | paper fit rows and inference rows |
| `cv5_oof` | five-fold OOF over Train + Devset | Blind-B reranker fit |
| `train5_oof` | five-fold OOF over Train | paper reranker fit |
| `full_public` | fitted on Train + Devset | Blind-B inference |
| `full_train` | fitted on Train | paper Devset inference |

OOF artifacts are used for reranker-fit rows; full-fit artifacts are used for Blind B or Devset inference.
