#!/bin/bash
# Download all external inputs into the local Hugging Face cache (HF_HOME).
# Requires HF_TOKEN with access to the gated talkpl-ai challenge repos.
# Run ONCE online; afterwards everything works with HF_HUB_OFFLINE=1.
set -eu
cd "$(dirname "$0")"
unset HF_HUB_OFFLINE || true

# Challenge datasets (gated) + external TalkPlayData — materialized through the
# `datasets` library so the processed cache is ready for offline runs.
uv run python - <<'PY'
from datasets import load_dataset

REPOS = [
    ("talkpl-ai/TalkPlayData-Challenge-Dataset", None),
    ("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", None),
    ("talkpl-ai/TalkPlayData-Challenge-User-Metadata", None),
    ("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", None),
    ("talkpl-ai/TalkPlayData-Challenge-User-Embeddings", None),
    ("talkpl-ai/TalkPlayData-Challenge-Blind-B", None),
    ("talkpl-ai/TalkPlayData-1", "train"),   # external: cooc/transition stats + two-tower training mix
    ("talkpl-ai/TalkPlayData-2", None),      # external: spotify_uuid_map construction only
]
for repo, split in REPOS:
    print(f"--- {repo} ---", flush=True)
    load_dataset(repo, split=split)
PY

# Pretrained models (referenced by ID, never re-uploaded):
#   Qwen/Qwen3.6-27B          responder LLM (bf16, ~52GB)
#   Qwen/Qwen3-Embedding-0.6B two-tower / dense-query base encoder
uv run hf download Qwen/Qwen3.6-27B
uv run hf download Qwen/Qwen3-Embedding-0.6B

echo "datasets + models cached. export HF_HUB_OFFLINE=1 for all further runs."
