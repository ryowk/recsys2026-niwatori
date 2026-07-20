"""Build the tag-intent BM25 retriever artifact."""

import re
from pathlib import Path

from recsys2026.fit_free_runner import FitFreeSpec, main as run_component
from recsys2026.retriever_common import norm_name

INTENT_TERMS = (
    "acoustic",
    "afrobeat",
    "ambient",
    "blues",
    "christmas",
    "classical",
    "country",
    "dance",
    "disco",
    "drill",
    "drum and bass",
    "edm",
    "electronic",
    "folk",
    "funk",
    "gospel",
    "grunge",
    "hip hop",
    "house",
    "indie",
    "jazz",
    "k pop",
    "latin",
    "metal",
    "opera",
    "piano",
    "pop",
    "punk",
    "r b",
    "rap",
    "reggae",
    "rock",
    "salsa",
    "soul",
    "techno",
    "trap",
    "workout",
    "party",
    "romantic",
    "sad",
    "happy",
    "chill",
    "relaxing",
    "energetic",
    "upbeat",
    "focus",
    "sleep",
    "summer",
)


def query_text(example, _track_index) -> str:
    text = norm_name(example.user_query)
    terms = [
        term for term in INTENT_TERMS if re.search(rf"\b{re.escape(term)}\b", text)
    ]
    terms.extend(
        match.group(0)
        for match in re.finditer(
            r"\b(19[5-9]0|20[0-2]0)s?\b|\b([5-9]0)s\b|\b(00s|2000s|2010s|2020s)\b",
            text,
        )
    )
    if "r b" in terms:
        terms.append("r&b")
    return " ".join(dict.fromkeys(terms))


SPEC = FitFreeSpec(
    name="tag_intent_bm25",
    source_path=Path(__file__),
    bm25_variants=(("tag_list", ("tag_list",)),),
    bm25_name="tag_list",
    query_fn=query_text,
)


if __name__ == "__main__":
    run_component(SPEC)
