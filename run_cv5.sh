#!/bin/bash
# 5-fold CV validation on public_labeled (129,592 rows): build the public
# retriever sources + union, then fit the reranker per fold and report nDCG.
# Expected ndcg@20 ≈ 0.2743 (folds ≈ 0.2767 / 0.2755 / 0.2741 / 0.2721 / 0.2732),
# candidate recall@20 = 0.4102 (small last-digit wobble from LightGBM n_jobs=-1).
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
bash run_preprocess.sh
bash run_retriever_cv5.sh
bash run_reranker_cv5.sh
