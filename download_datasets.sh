#!/bin/bash
# Cache pipeline inputs in HF_HOME.
set -eu
cd "$(dirname "$0")"
unset HF_HUB_OFFLINE || true

uv run python - <<'PY'
from datasets import load_dataset

REPOS = [
    ("talkpl-ai/TalkPlayData-Challenge-Dataset", None),
    ("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", None),
    ("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings", None),
    ("talkpl-ai/TalkPlayData-Challenge-User-Embeddings", None),
    ("talkpl-ai/TalkPlayData-Challenge-Blind-B", None),
    ("talkpl-ai/TalkPlayData-1", "train"),   # external: cooc/transition statistics only
    ("talkpl-ai/TalkPlayData-2", None),      # external: catalog ID mapping only
]
for repo, split in REPOS:
    print(f"--- {repo} ---", flush=True)
    load_dataset(repo, split=split)
PY

uv run hf download Qwen/Qwen3.6-27B
uv run hf download Qwen/Qwen3-Embedding-0.6B

echo "datasets + models cached. export HF_HUB_OFFLINE=1 for all further runs."
