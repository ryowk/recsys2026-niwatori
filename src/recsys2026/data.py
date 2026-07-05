"""HuggingFace 上の TalkPlayData Challenge データセットをロードするヘルパー。"""

from typing import Literal

from datasets import Dataset, DatasetDict, load_dataset

REPOS: dict[str, str] = {
    "dataset":   "talkpl-ai/TalkPlayData-Challenge-Dataset",
    "track":     "talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
    "user":      "talkpl-ai/TalkPlayData-Challenge-User-Metadata",
    "track_emb": "talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
    "user_emb":  "talkpl-ai/TalkPlayData-Challenge-User-Embeddings",
    "blind_a":   "talkpl-ai/TalkPlayData-Challenge-Blind-A",
    "blind_b":   "talkpl-ai/TalkPlayData-Challenge-Blind-B",
}

Name = Literal[
    "dataset", "track", "user", "track_emb", "user_emb", "blind_a", "blind_b"
]


def load(name: Name, split: str | None = None) -> Dataset | DatasetDict:
    """short name から HF データセットをロードする。

    split を省略すると全 split を含む DatasetDict を返す。
    """
    return load_dataset(REPOS[name], split=split)
