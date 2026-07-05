#!/bin/bash
# Full from-scratch Blind-B pipeline (NO Hugging Face weights — two-tower is
# trained, the reranker is fit). Submission path only. The reranker 5-fold CV
# (run_reranker_cv5.sh) is a separate side branch and is NOT run here.
#
# Dependency order:
#   run_preprocess ─┬─▶ run_retriever_cv5 ─┐
#                   └─▶ run_retriever_blind_b ─┴─▶ run_reranker_blind_b ─▶ run_responder_blind_b
#
# ~9-14h on 1 GPU (two-tower training) + a >=128GB-RAM CPU host (reranker fit).
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

bash run_preprocess.sh
bash run_retriever_cv5.sh
bash run_retriever_blind_b.sh
bash run_reranker_blind_b.sh
bash run_responder_blind_b.sh

echo "done. submission: artifacts/runs/responder/qwen36_10run_diverse/blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5/blind_b.submission.zip"
