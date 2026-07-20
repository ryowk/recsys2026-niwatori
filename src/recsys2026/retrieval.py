"""Conversation text rendering shared by the final learned components."""

from __future__ import annotations

from .data import load
from .submission import InferenceInput

_EXPAND_FIELDS = ("track_name", "artist_name", "album_name", "release_date", "tag_list")
_TRACK_META_CACHE: dict[str, dict] | None = None


def _track_meta_lookup() -> dict[str, dict]:
    """Load the track-ID-to-metadata lookup on first use."""
    global _TRACK_META_CACHE
    if _TRACK_META_CACHE is None:
        meta = load("track", split="all_tracks")
        _TRACK_META_CACHE = {row["track_id"]: row for row in meta}
    return _TRACK_META_CACHE


def _expand_music(track_id: str) -> str:
    """Expand a history track ID into searchable catalog metadata."""
    md = _track_meta_lookup().get(track_id)
    if md is None:
        return track_id
    parts = [f"track_id: {track_id}"]
    for field in _EXPAND_FIELDS:
        v = md.get(field)
        if v is None:
            continue
        if isinstance(v, list):
            joined = ", ".join(str(x) for x in v if x is not None and str(x))
        else:
            joined = str(v)
        if joined:
            parts.append(f"{field}: {joined.lower()}")
    return ", ".join(parts)


def chat_to_query_text(inp: InferenceInput, mode: str = "full") -> str:
    """Render either the current request or all visible conversation turns."""
    if mode == "last_user":
        return inp.user_query
    if mode != "full":
        raise ValueError(f"unknown query mode: {mode}")

    parts: list[str] = []
    for c in inp.chat_history:
        role = c.get("role", "user")
        content = c.get("content", "")
        if role == "music":
            role = "assistant"
            content = _expand_music(content)
        parts.append(f"{role}: {content}")
    parts.append(f"user: {inp.user_query}")
    return "\n".join(parts)
