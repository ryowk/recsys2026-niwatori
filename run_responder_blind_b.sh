#!/bin/bash
# Generate Blind-B responses and package the submission.
set -eu
cd "$(dirname "$0")"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
OUT=artifacts/runs/responder/qwen36_27b/default/blind_b
uv run python -m responder.qwen36_27b.main
echo "submission: $OUT/submission.zip"
