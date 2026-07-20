#!/bin/bash
# Full from-scratch Blind-B pipeline.
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

bash run_preprocess.sh
bash run_retriever_fit.sh
bash run_retriever_blind_b.sh
bash run_reranker_blind_b.sh
bash run_responder_blind_b.sh

echo "done. submission: artifacts/runs/responder/qwen36_27b/default/blind_b/submission.zip"
