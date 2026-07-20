"""Shared catalog, example, and history primitives for retrievers.

The final pipeline imports this module for track metadata, lexical indexes,
and deterministic history/entity retrieval. Component drivers define fit scope
and query fields.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import unicodedata

import bm25s

from recsys2026.data import load

MAX_TURNS = 8


# -------------------- example --------------------


@dataclass(frozen=True)
class TurnExample:
    session_id: str
    user_id: str
    turn_number: int
    chat_history: list[dict]
    user_query: str
    gold_track_id: str | None
    user_thought: str = ""


def as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").casefold()
    s = re.sub(r"[^\w]+", " ", s)
    return " ".join(s.split())


# -------------------- helpers --------------------


def _bm25_corpus_text(row: dict, fields: tuple[str, ...]) -> str:
    out: list[str] = []
    for f in fields:
        value = row.get(f)
        if value is None:
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value if v is not None and str(v))
        out.append(f"{f}: {value}")
    return "\n".join(out).lower()


_EXPAND_FIELDS = ("track_name", "artist_name", "album_name", "release_date", "tag_list")


def track_metadata_text(track_id: str, meta: dict) -> str:
    parts = [f"track_id: {track_id}"]
    for f in _EXPAND_FIELDS:
        v = meta.get(f)
        if v is None:
            continue
        if isinstance(v, list):
            joined = ", ".join(str(x) for x in v if x is not None and str(x))
        else:
            joined = str(v)
        if joined:
            parts.append(f"{f}: {joined.lower()}")
    return ", ".join(parts)


def match_catalog_names(
    text: str,
    rare_bucket: dict[str, list[str]],
    *,
    min_chars: int,
    min_tokens: int,
) -> set[str]:
    if not text:
        return set()
    padded = f" {text} "
    matched: set[str] = set()
    for token in set(text.split()):
        for name in rare_bucket.get(token, []):
            if len(name) >= min_chars and len(name.split()) >= min_tokens:
                if f" {name} " in padded:
                    matched.add(name)
    return matched


# -------------------- TrackIndex --------------------


@dataclass
class TrackIndex:
    track_ids: list[str]
    id_to_idx: dict[str, int]
    n_tracks: int
    meta_by_id: dict[str, dict]
    artist_to_idx: dict[str, list[int]]
    album_to_idx: dict[str, list[int]]
    artist_name_to_idx: dict[str, list[int]]
    album_artist_name_to_idx: dict[tuple[str, str], list[int]]
    track_name_to_idx: dict[str, list[int]]
    album_name_to_idx: dict[str, list[int]]
    track_name_rare_bucket: dict[str, list[str]]
    artist_name_rare_bucket: dict[str, list[str]]
    album_name_rare_bucket: dict[str, list[str]]
    track_artist_name_keys: list[set[str]]
    bm25_indexes: dict = field(default_factory=dict)


def build_track_index(
    bm25_variants: tuple[tuple[str, tuple[str, ...]], ...],
) -> TrackIndex:
    print("loading track metadata ...")
    meta = load("track", split="all_tracks")
    meta_by_id = {row["track_id"]: row for row in meta}

    print("loading track_emb ...")
    emb = load("track_emb", split="all_tracks")
    track_ids: list[str] = list(emb["track_id"])
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    n = len(track_ids)

    artist_to_idx: dict[str, list[int]] = defaultdict(list)
    album_to_idx: dict[str, list[int]] = defaultdict(list)
    artist_name_to_idx: dict[str, list[int]] = defaultdict(list)
    album_artist_name_to_idx: dict[tuple[str, str], list[int]] = defaultdict(list)
    track_name_to_idx: dict[str, list[int]] = defaultdict(list)
    album_name_to_idx: dict[str, list[int]] = defaultdict(list)
    track_artist_name_keys: list[set[str]] = [set() for _ in range(n)]
    for i, tid in enumerate(track_ids):
        row = meta_by_id.get(tid, {})
        track_names = {
            norm_name(str(name))
            for name in as_list(row.get("track_name"))
            if str(name or "").strip()
        }
        track_names.discard("")
        for name in track_names:
            track_name_to_idx[name].append(i)
        for aid in as_list(row.get("artist_id")):
            if aid:
                artist_to_idx[str(aid)].append(i)
        for alid in as_list(row.get("album_id")):
            if alid:
                album_to_idx[str(alid)].append(i)
        artist_names = {
            norm_name(str(name))
            for name in as_list(row.get("artist_name"))
            if str(name or "").strip()
        }
        album_names = {
            norm_name(str(name))
            for name in as_list(row.get("album_name"))
            if str(name or "").strip()
        }
        artist_names.discard("")
        album_names.discard("")
        track_artist_name_keys[i] = artist_names
        for name in artist_names:
            artist_name_to_idx[name].append(i)
        for name in album_names:
            album_name_to_idx[name].append(i)
        album_artist_keys: set[tuple[str, str]] = set()
        for album_name in album_names:
            for artist_name in artist_names:
                album_artist_keys.add((album_name, artist_name))
        for key in album_artist_keys:
            album_artist_name_to_idx[key].append(i)

    def build_rare_bucket(names: list[str]) -> dict[str, list[str]]:
        token_df: Counter[str] = Counter()
        for name in names:
            token_df.update(set(name.split()))
        buckets: dict[str, list[str]] = defaultdict(list)
        for name in names:
            toks = name.split()
            if not toks:
                continue
            rare = min(toks, key=lambda tok: (token_df[tok], tok))
            buckets[rare].append(name)
        return dict(buckets)

    track_name_rare_bucket = build_rare_bucket(list(track_name_to_idx))
    artist_name_rare_bucket = build_rare_bucket(list(artist_name_to_idx))
    album_name_rare_bucket = build_rare_bucket(list(album_name_to_idx))

    bm25_indexes: dict[str, bm25s.BM25] = {}
    for name, fields_tuple in bm25_variants:
        print(f"building BM25 index for {name} (fields={fields_tuple}) ...")
        corpus = [
            _bm25_corpus_text(meta_by_id.get(tid, {}), fields_tuple)
            for tid in track_ids
        ]
        idx = bm25s.BM25()
        idx.index(bm25s.tokenize(corpus, show_progress=False), show_progress=False)
        bm25_indexes[name] = idx

    return TrackIndex(
        track_ids=track_ids,
        id_to_idx=id_to_idx,
        n_tracks=n,
        meta_by_id=meta_by_id,
        artist_to_idx=dict(artist_to_idx),
        album_to_idx=dict(album_to_idx),
        artist_name_to_idx=dict(artist_name_to_idx),
        album_artist_name_to_idx=dict(album_artist_name_to_idx),
        track_name_to_idx=dict(track_name_to_idx),
        album_name_to_idx=dict(album_name_to_idx),
        track_name_rare_bucket=track_name_rare_bucket,
        artist_name_rare_bucket=artist_name_rare_bucket,
        album_name_rare_bucket=album_name_rare_bucket,
        track_artist_name_keys=track_artist_name_keys,
        bm25_indexes=bm25_indexes,
    )


# -------------------- Cooc --------------------


def build_examples_devset() -> list[TurnExample]:
    ds = load("dataset", split="test")
    examples: list[TurnExample] = []
    for item in ds:
        conversations = list(item["conversations"])
        for target_turn in range(1, MAX_TURNS + 1):
            current = [c for c in conversations if c["turn_number"] == target_turn]
            user_turn = next(c for c in current if c["role"] == "user")
            gold = next(c["content"] for c in current if c["role"] == "music")
            examples.append(
                TurnExample(
                    session_id=item["session_id"],
                    user_id=item["user_id"],
                    turn_number=target_turn,
                    chat_history=[
                        c for c in conversations if c["turn_number"] < target_turn
                    ],
                    user_query=user_turn["content"],
                    gold_track_id=gold,
                    user_thought=(user_turn.get("thought") or "").strip(),
                )
            )
    return examples


# -------------------- per-example helpers --------------------


def history_state(ex: TurnExample, track_index: TrackIndex):
    h_arts: set[str] = set()
    h_albs: set[str] = set()
    played: set[int] = set()
    history_idxs: list[int] = []
    for c in ex.chat_history:
        if c.get("role") != "music":
            continue
        tid = c.get("content")
        if not tid:
            continue
        idx = track_index.id_to_idx.get(tid)
        if idx is not None:
            played.add(idx)
            history_idxs.append(idx)
        md = track_index.meta_by_id.get(tid)
        if md is None:
            continue
        for x in as_list(md.get("artist_id")):
            if x:
                h_arts.add(str(x))
        for x in as_list(md.get("album_id")):
            if x:
                h_albs.add(str(x))
    return h_arts, h_albs, played, history_idxs


def history_name_counts(
    ex: TurnExample,
    track_index: TrackIndex,
    *,
    last_only: bool,
) -> tuple[Counter[str], Counter[tuple[str, str]], set[int]]:
    artist_counts: Counter[str] = Counter()
    album_artist_counts: Counter[tuple[str, str]] = Counter()
    played: set[int] = set()
    music_turns = [
        c for c in ex.chat_history if c.get("role") == "music" and c.get("content")
    ]
    if last_only:
        music_turns = music_turns[-1:]
    for c in music_turns:
        tid = c.get("content")
        idx = track_index.id_to_idx.get(tid)
        if idx is not None:
            played.add(idx)
        md = track_index.meta_by_id.get(tid)
        if md is None:
            continue
        artist_names = {
            norm_name(str(name))
            for name in as_list(md.get("artist_name"))
            if str(name or "").strip()
        }
        album_names = {
            norm_name(str(name))
            for name in as_list(md.get("album_name"))
            if str(name or "").strip()
        }
        artist_names.discard("")
        album_names.discard("")
        for artist_name in artist_names:
            artist_counts[artist_name] += 1
        for album_name in album_names:
            for artist_name in artist_names:
                album_artist_counts[(album_name, artist_name)] += 1
    return artist_counts, album_artist_counts, played


def played_set(ex: TurnExample, track_index: TrackIndex) -> set[int]:
    played: set[int] = set()
    for c in ex.chat_history:
        if c.get("role") == "music":
            idx = track_index.id_to_idx.get(c.get("content"))
            if idx is not None:
                played.add(idx)
    return played
