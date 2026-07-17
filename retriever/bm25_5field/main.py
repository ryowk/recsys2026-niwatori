"""Build the five-field BM25 retriever artifact."""

from pathlib import Path

from recsys2026.fit_free_runner import FitFreeSpec, main as run_component
from recsys2026.retriever_common import track_metadata_text


def query_text(example, track_index) -> str:
    parts: list[str] = []
    for turn in example.chat_history:
        role = turn.get("role", "user")
        content = turn.get("content", "")
        if role == "music":
            metadata = track_index.meta_by_id.get(content)
            if metadata is None:
                continue
            role = "assistant"
            content = track_metadata_text(content, metadata)
        parts.append(f"{role}: {content}")
    parts.append(f"user: {example.user_query}")
    return "\n".join(parts).lower()


SPEC = FitFreeSpec(
    name="bm25_5field",
    source_path=Path(__file__),
    bm25_variants=(
        (
            "5field",
            ("track_name", "artist_name", "album_name", "release_date", "tag_list"),
        ),
    ),
    bm25_name="5field",
    query_fn=query_text,
)


if __name__ == "__main__":
    run_component(SPEC)
