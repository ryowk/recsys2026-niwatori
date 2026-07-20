"""Load the TalkPlayData Challenge datasets from Hugging Face."""

from typing import Literal

from datasets import Dataset, DatasetDict, load_dataset

REPOS: dict[str, str] = {
    "dataset": "talkpl-ai/TalkPlayData-Challenge-Dataset",
    "track": "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    "track_emb": "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
    "user_emb": "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
    "blind_b": "talkpl-ai/TalkPlayData-Challenge-Blind-B",
}

Name = Literal["dataset", "track", "track_emb", "user_emb", "blind_b"]


def load(name: Name, split: str | None = None) -> Dataset | DatasetDict:
    """Load one dataset by short name, or all splits when `split` is omitted."""
    return load_dataset(REPOS[name], split=split)
