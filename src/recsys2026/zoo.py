"""086_retriever_zoo_v2: 080 の再作成版.

080 の経緯と現状:
- 元の 080_retriever_zoo は ~03:14 (2026-05-10) に外部要因で削除された.
- 086 でほぼ同等の機能を再構築 (cand cache + analyze_marginal は全てこの exp 配下).
- thought 関連 (current/history 両方) は使用 OK の方針 (blind_a に thought 提供あり).

ユーザの依頼:「retriever を改善したい. 各 retriever の recall と 1 user 当たりの候補数平均を
集計し, union set もまとめる. 学習が必要な retriever は train set のみを使う. 評価は test
(devset) のみ.」

含む retriever (~36):

Phase 1 cheap (no LLM, no train):
- BM25 family: 4field / 5field / 単 field 4 種 / history-boost / drop_music / user_only /
                with_thought / 5field_thought / thought_only / artist_album
- History-based: artist match / album match / artist+album / primary_tag / last music *2 /
                 release_decade
- Popularity: global
- CF: history centroid / user_emb × track cf-bpr
- Cooccurrence: track-track / artist-artist (train sessions only)

Phase 1 dense (frozen encoder):
- Qwen3-Embedding (metadata / attributes / lyrics) / CLAP audio / SigLIP image

Phase 2 LLM-augmented:
- HyDE BM25 (4field / 5field) / HyDE dense Qwen3 / intent_tag_match / album_qwen3_history

Phase 3 学習系 (別 exp):
- 081 two_tower の cand を ZOO_OUT_DIR に書き出し済み (再学習も可)

リーク防止:
- cooccurrence 行列は **train split のみ** から構築
- Qwen3 / CLAP / SigLIP は pretrained encoder (no fit on test)
- 評価は devset (test split) でのみ実施
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import unicodedata

import bm25s
import numpy as np
import torch
from tqdm import tqdm

from recsys2026.data import load
from recsys2026.paths import OUTPUT_DIR as _OUTPUT_ROOT, RESULTS_DIR as _RESULTS_ROOT

OUT_DIR = _OUTPUT_ROOT / "zoo"
RESULTS_DIR = _RESULTS_ROOT / "zoo"
TOP_K = 200
RECALL_KS = (20, 50, 100, 200)

EMB_COL = "cf-bpr"
DENSE_COLS = {
    "dense_qwen3_metadata": "metadata-qwen3_embedding_0.6b",
    "dense_qwen3_attributes": "attributes-qwen3_embedding_0.6b",
    "dense_qwen3_lyrics": "lyrics-qwen3_embedding_0.6b",
    "dense_clap_audio": "audio-laion_clap",
    "dense_siglip_image": "image-siglip2",
}

SEMANTIC_DENSE_SOURCES = {
    "attribute_query_rrf": "dense_qwen3_attributes",
    "lyrics_query_rrf": "dense_qwen3_lyrics",
    "metadata_query_rrf_nohistory": "dense_qwen3_metadata",
}

SEMANTIC_QUERY_CACHE = (
    Path(__file__).resolve().parents[2]
    / "output"
    / "093_current_thought_goal_query"
    / "dense_qfeat_devset_maxNone_last_user.npz"
)

TAG_INTENT_TERMS = (
    "acoustic", "afrobeat", "ambient", "blues", "christmas", "classical",
    "country", "dance", "disco", "drill", "drum and bass", "edm",
    "electronic", "folk", "funk", "gospel", "grunge", "hip hop", "house",
    "indie", "jazz", "k pop", "latin", "metal", "opera", "piano", "pop",
    "punk", "r b", "rap", "reggae", "rock", "salsa", "soul", "techno",
    "trap", "workout", "party", "romantic", "sad", "happy", "chill",
    "relaxing", "energetic", "upbeat", "focus", "sleep", "summer",
)

MAX_TURNS = 8


# -------------------- example --------------------


@dataclass(frozen=True)
class TurnExample:
    session_id: str
    user_id: str
    session_date: str
    turn_number: int
    chat_history: list[dict]
    user_query: str
    gold_track_id: str | None
    user_thought: str = ""
    conversation_goal: dict = field(default_factory=dict)


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


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return x / denom


def _to_dense(values: list, dim: int | None = None) -> np.ndarray:
    if dim is None:
        lengths = [len(v) for v in values if v is not None and len(v) > 0]
        if not lengths:
            raise ValueError("no non-empty embeddings found")
        dim = Counter(lengths).most_common(1)[0][0]
    out = np.zeros((len(values), dim), dtype=np.float32)
    for i, v in enumerate(values):
        if v is None or len(v) != dim:
            continue
        out[i] = np.asarray(v, dtype=np.float32)
    return out


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


def _track_metadata_str(track_id: str, meta: dict) -> str:
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


def _bm25_query_text(
    ex: TurnExample,
    track_meta: dict[str, dict],
    mode: str = "full",
) -> str:
    """chat history + user_query を BM25 query に整形."""
    if mode == "last_user":
        return ex.user_query.lower()
    if mode == "thought_only":
        thought = "" if _blind_b_safe() else (ex.user_thought or "").strip()
        return thought.lower() if thought else ex.user_query.lower()

    history = ex.chat_history
    if mode == "drop_music":
        history = [c for c in history if c.get("role") != "music"]
    elif mode == "user_only":
        history = [c for c in history if c.get("role") == "user"]

    parts: list[str] = []
    for c in history:
        role = c.get("role", "user")
        content = c.get("content", "")
        if role == "music":
            md = track_meta.get(content)
            if md is not None:
                role = "assistant"
                content = _track_metadata_str(content, md)
            else:
                continue
        parts.append(f"{role}: {content}")

    user_line = f"user: {ex.user_query}"
    if mode == "with_thought" and ex.user_thought and not _blind_b_safe():
        user_line = f"user: {ex.user_query} {ex.user_thought}"
    parts.append(user_line)

    text = "\n".join(parts).lower()

    if mode == "history_boost":
        boost_tokens = []
        for c in ex.chat_history:
            if c.get("role") == "music":
                md = track_meta.get(c.get("content", ""))
                if md is not None:
                    boost_tokens.append(_track_metadata_str(c["content"], md))
        if boost_tokens:
            text = text + "\n" + "\n".join(boost_tokens).lower()

    return text


def _blind_b_safe() -> bool:
    """Blind-B-safe fixed: conversation_goal + thought are never used in query text."""
    return True


def _goal_text_from_goal(goal: dict | None) -> str:
    if _blind_b_safe():
        return ""
    goal = goal or {}
    parts = [
        str(goal.get(k) or "").strip()
        for k in ("category", "specificity", "listener_goal")
    ]
    return " ".join(p for p in parts if p)


def _strip_prompt_artifacts(text: str) -> str:
    text = str(text or "")
    text = re.sub(r"\bthought\s*:", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bRecSys\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAchieved query example\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bas per the instruction\b", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())


def _semantic_query_text(ex: TurnExample) -> str:
    parts = [
        _goal_text_from_goal(ex.conversation_goal),
        ex.user_query,
        ex.user_thought,
    ]
    text = "\n".join(_strip_prompt_artifacts(p) for p in parts if str(p or "").strip())
    return text or "music recommendation"


def _semantic_query_key(ex: TurnExample) -> str:
    return f"{ex.session_id}:{ex.turn_number}"


def _tag_intent_query(ex: TurnExample) -> str:
    text = norm_name(" ".join([
        _goal_text_from_goal(ex.conversation_goal),
        ex.user_query,
        "" if _blind_b_safe() else ex.user_thought,
    ]))
    terms: list[str] = []
    for term in TAG_INTENT_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", text):
            terms.append(term)
    for m in re.finditer(r"\b(19[5-9]0|20[0-2]0)s?\b|\b([5-9]0)s\b|\b(00s|2000s|2010s|2020s)\b", text):
        terms.append(m.group(0))
    if "r b" in terms:
        terms.append("r&b")
    return " ".join(dict.fromkeys(terms))


def _current_request_norm(ex: TurnExample) -> str:
    return norm_name(f"{ex.user_query or ''} {ex.user_thought or ''}")


def _matched_catalog_names(
    text_norm: str,
    rare_bucket: dict[str, list[str]],
    *,
    min_chars: int,
    min_tokens: int,
) -> set[str]:
    if not text_norm:
        return set()
    text_sp = f" {text_norm} "
    matched: set[str] = set()
    for tok in set(text_norm.split()):
        for name in rare_bucket.get(tok, []):
            if len(name) < min_chars:
                continue
            if len(name.split()) < min_tokens:
                continue
            if f" {name} " in text_sp:
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
    secondary_artist_name_to_idx: dict[str, list[int]]
    track_name_rare_bucket: dict[str, list[str]]
    artist_name_rare_bucket: dict[str, list[str]]
    album_name_rare_bucket: dict[str, list[str]]
    track_artist_name_keys: list[set[str]]
    track_album_artist_name_keys: list[set[tuple[str, str]]]
    primary_tag_to_idx: dict[str, list[int]]
    primary_tag_per_track: list[str]
    popularity: np.ndarray
    popularity_order: np.ndarray
    cf: np.ndarray
    bm25_indexes: dict = field(default_factory=dict)
    dense_mats: dict[str, np.ndarray] = field(default_factory=dict)


def build_track_index(
    bm25_variants: tuple[tuple[str, tuple[str, ...]], ...],
    dense_keys: tuple[str, ...],
) -> TrackIndex:
    print("loading track metadata ...")
    meta = load("track", split="all_tracks")
    meta_by_id = {row["track_id"]: row for row in meta}

    print("loading track_emb ...")
    emb = load("track_emb", split="all_tracks")
    track_ids: list[str] = list(emb["track_id"])
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    n = len(track_ids)

    popularity = np.zeros(n, dtype=np.float32)
    artist_to_idx: dict[str, list[int]] = defaultdict(list)
    album_to_idx: dict[str, list[int]] = defaultdict(list)
    artist_name_to_idx: dict[str, list[int]] = defaultdict(list)
    album_artist_name_to_idx: dict[tuple[str, str], list[int]] = defaultdict(list)
    track_name_to_idx: dict[str, list[int]] = defaultdict(list)
    album_name_to_idx: dict[str, list[int]] = defaultdict(list)
    secondary_artist_name_to_idx: dict[str, list[int]] = defaultdict(list)
    track_artist_name_keys: list[set[str]] = [set() for _ in range(n)]
    track_album_artist_name_keys: list[set[tuple[str, str]]] = [set() for _ in range(n)]
    tag_to_idx: dict[str, list[int]] = defaultdict(list)
    primary_tag_per_track = [""] * n
    for i, tid in enumerate(track_ids):
        row = meta_by_id.get(tid, {})
        popularity[i] = float(row.get("popularity") or 0.0)
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
        artist_name_list = [
            norm_name(str(name))
            for name in as_list(row.get("artist_name"))
            if str(name or "").strip()
        ]
        artist_names = {name for name in artist_name_list if name}
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
        for name in artist_name_list[1:]:
            if name:
                secondary_artist_name_to_idx[name].append(i)
        for name in album_names:
            album_name_to_idx[name].append(i)
        album_artist_keys: set[tuple[str, str]] = set()
        for album_name in album_names:
            for artist_name in artist_names:
                album_artist_keys.add((album_name, artist_name))
        track_album_artist_name_keys[i] = album_artist_keys
        for key in album_artist_keys:
            album_artist_name_to_idx[key].append(i)
        tags = as_list(row.get("tag_list"))
        if tags:
            primary = str(tags[0]).lower().strip()
            primary_tag_per_track[i] = primary
            tag_to_idx[primary].append(i)

    popularity_order = np.argsort(-popularity).astype(np.int32)

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

    print(f"loading {EMB_COL} ({n} tracks) ...")
    cf_raw = _to_dense(emb[EMB_COL])
    cf = _normalize_rows(cf_raw)

    bm25_indexes: dict[str, bm25s.BM25] = {}
    for name, fields_tuple in bm25_variants:
        print(f"building BM25 index for {name} (fields={fields_tuple}) ...")
        corpus = [_bm25_corpus_text(meta_by_id.get(tid, {}), fields_tuple) for tid in track_ids]
        idx = bm25s.BM25()
        idx.index(bm25s.tokenize(corpus, show_progress=False), show_progress=False)
        bm25_indexes[name] = idx

    dense_mats: dict[str, np.ndarray] = {}
    for key in dense_keys:
        col = DENSE_COLS[key]
        if col not in emb.column_names:
            print(f"  [skip] {key}: column {col} missing")
            continue
        print(f"loading dense matrix for {key} ({col}) ...")
        mat = _to_dense(emb[col])
        mat = _normalize_rows(mat)
        dense_mats[key] = mat

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
        secondary_artist_name_to_idx=dict(secondary_artist_name_to_idx),
        track_name_rare_bucket=track_name_rare_bucket,
        artist_name_rare_bucket=artist_name_rare_bucket,
        album_name_rare_bucket=album_name_rare_bucket,
        track_artist_name_keys=track_artist_name_keys,
        track_album_artist_name_keys=track_album_artist_name_keys,
        primary_tag_to_idx=dict(tag_to_idx),
        primary_tag_per_track=primary_tag_per_track,
        popularity=popularity,
        popularity_order=popularity_order,
        cf=cf,
        bm25_indexes=bm25_indexes,
        dense_mats=dense_mats,
    )


# -------------------- Cooc --------------------


def collect_train_sessions() -> list[list[str]]:
    """train split のみ. valid/devset/blind は混ぜない."""
    ds = load("dataset", split="train")
    sessions: list[list[str]] = []
    for item in ds:
        tracks = [
            c["content"]
            for c in item["conversations"]
            if c["role"] == "music" and c.get("content")
        ]
        sessions.append(tracks)
    return sessions


@dataclass
class Cooc:
    track_track: dict[int, tuple[np.ndarray, np.ndarray]]
    artist_artist: dict[str, Counter]
    artist_name_artist_name: dict[str, Counter]
    transition_artist_name: dict[str, Counter]


def build_cooc(track_index: TrackIndex) -> Cooc:
    sessions = collect_train_sessions()
    print(f"build cooc from {len(sessions)} train sessions")

    track_counts: dict[int, Counter] = defaultdict(Counter)
    artist_counts: dict[str, Counter] = defaultdict(Counter)
    artist_name_counts: dict[str, Counter] = defaultdict(Counter)
    transition_artist_name_counts: dict[str, Counter] = defaultdict(Counter)

    for tracks in tqdm(sessions, desc="build cooc"):
        idxs: list[int] = []
        artists: set[str] = set()
        artist_names: set[str] = set()
        artist_names_by_turn: list[set[str]] = []
        for tid in set(tracks):
            j = track_index.id_to_idx.get(tid)
            if j is None:
                continue
            idxs.append(j)
            for aid in as_list(track_index.meta_by_id.get(tid, {}).get("artist_id")):
                if aid:
                    artists.add(str(aid))
        for tid in tracks:
            md = track_index.meta_by_id.get(tid, {})
            names = {
                norm_name(str(name))
                for name in as_list(md.get("artist_name"))
                if str(name or "").strip()
            }
            names.discard("")
            artist_names.update(names)
            artist_names_by_turn.append(names)

        for i, a in enumerate(idxs):
            ca = track_counts[a]
            for b in idxs[i + 1:]:
                ca[b] += 1
                track_counts[b][a] += 1

        a_list = list(artists)
        for i, a in enumerate(a_list):
            ca = artist_counts[a]
            for b in a_list[i + 1:]:
                ca[b] += 1
                artist_counts[b][a] += 1

        name_list = list(artist_names)
        for i, a in enumerate(name_list):
            ca = artist_name_counts[a]
            for b in name_list[i + 1:]:
                ca[b] += 1
                artist_name_counts[b][a] += 1

        for prev_names, next_names in zip(artist_names_by_turn, artist_names_by_turn[1:]):
            for a in prev_names:
                ca = transition_artist_name_counts[a]
                for b in next_names:
                    if a != b:
                        ca[b] += 1

    track_track: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k, c in track_counts.items():
        if not c:
            continue
        items = c.most_common()
        nb = np.fromiter((i for i, _ in items), dtype=np.int32, count=len(items))
        cn = np.fromiter((v for _, v in items), dtype=np.float32, count=len(items))
        track_track[k] = (nb, cn)
    print(
        f"cooc: {len(track_track)} tracks have track-neighbors, "
        f"{len(artist_counts)} artists have artist-neighbors, "
        f"{len(artist_name_counts)} artist names have name-neighbors, "
        f"{len(transition_artist_name_counts)} artist names have transitions"
    )
    return Cooc(
        track_track=track_track,
        artist_artist=dict(artist_counts),
        artist_name_artist_name=dict(artist_name_counts),
        transition_artist_name=dict(transition_artist_name_counts),
    )


# -------------------- Personal memory --------------------


@dataclass
class PersonalMemory:
    user_sessions: dict[str, list[tuple[str, list[int]]]]


def build_personal_memory(track_index: TrackIndex) -> PersonalMemory:
    """Train split only. Per-target lookup applies strict date censoring."""
    ds = load("dataset", split="train")
    user_sessions: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for item in ds:
        date = str(item.get("session_date") or "")
        tracks: list[int] = []
        for msg in item["conversations"]:
            if msg.get("role") != "music":
                continue
            tid = msg.get("content")
            idx = track_index.id_to_idx.get(tid)
            if idx is not None:
                tracks.append(idx)
        if tracks:
            user_sessions[str(item["user_id"])].append((date, tracks))
    for sessions in user_sessions.values():
        sessions.sort(key=lambda x: x[0])
    print(f"personal memory: {len(user_sessions)} train users with music history")
    return PersonalMemory(user_sessions=dict(user_sessions))


def _prior_personal_track_counts(ex: TurnExample, memory: PersonalMemory) -> Counter[int]:
    out: Counter[int] = Counter()
    if not ex.session_date:
        return out
    for date, tracks in memory.user_sessions.get(ex.user_id, []):
        if date >= ex.session_date:
            break
        out.update(tracks)
    return out


def _score_personal_exact_repeat(ex, track_index, memory: PersonalMemory):
    counts = _prior_personal_track_counts(ex, memory)
    if not counts:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for idx, cnt in counts.items():
        score[int(idx)] += float(cnt) * 10.0
    return score


def _score_personal_artist_expansion(ex, track_index, memory: PersonalMemory):
    counts = _prior_personal_track_counts(ex, memory)
    if not counts:
        return None
    artist_counts: Counter[str] = Counter()
    for idx, cnt in counts.items():
        tid = track_index.track_ids[int(idx)]
        md = track_index.meta_by_id.get(tid, {})
        for aid in as_list(md.get("artist_id")):
            if aid:
                artist_counts[str(aid)] += int(cnt)
    if not artist_counts:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for aid, cnt in artist_counts.items():
        for j in track_index.artist_to_idx.get(aid, []):
            score[j] += float(cnt)
    return score


def _score_personal_album_expansion(ex, track_index, memory: PersonalMemory):
    counts = _prior_personal_track_counts(ex, memory)
    if not counts:
        return None
    album_counts: Counter[str] = Counter()
    for idx, cnt in counts.items():
        tid = track_index.track_ids[int(idx)]
        md = track_index.meta_by_id.get(tid, {})
        for album_id in as_list(md.get("album_id")):
            if album_id:
                album_counts[str(album_id)] += int(cnt)
    if not album_counts:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for album_id, cnt in album_counts.items():
        for j in track_index.album_to_idx.get(album_id, []):
            score[j] += float(cnt)
    return score


# -------------------- User emb --------------------


def load_user_vectors_normalized() -> dict[str, np.ndarray]:
    user_emb = load("user_emb")
    out: dict[str, np.ndarray] = {}
    for split in user_emb:
        for row in user_emb[split]:
            v = row.get(EMB_COL)
            if v is None or len(v) == 0:
                continue
            vec = np.asarray(v, dtype=np.float32)
            n = float(np.linalg.norm(vec))
            if n > 0:
                out[row["user_id"]] = vec / n
    return out


# -------------------- examples --------------------


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
                    session_date=str(item.get("session_date") or ""),
                    turn_number=target_turn,
                    chat_history=[c for c in conversations if c["turn_number"] < target_turn],
                    user_query=user_turn["content"],
                    gold_track_id=gold,
                    user_thought=(user_turn.get("thought") or "").strip(),
                    conversation_goal=dict(item.get("conversation_goal") or {}),
                )
            )
    return examples


# -------------------- per-example helpers --------------------


def history_state(ex: TurnExample, track_index: TrackIndex):
    h_arts: set[str] = set()
    h_albs: set[str] = set()
    h_tags: set[str] = set()
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
            tag = track_index.primary_tag_per_track[idx]
            if tag:
                h_tags.add(tag)
        md = track_index.meta_by_id.get(tid)
        if md is None:
            continue
        for x in as_list(md.get("artist_id")):
            if x:
                h_arts.add(str(x))
        for x in as_list(md.get("album_id")):
            if x:
                h_albs.add(str(x))
    return h_arts, h_albs, h_tags, played, history_idxs


def _history_name_counts(
    ex: TurnExample,
    track_index: TrackIndex,
    *,
    last_only: bool,
) -> tuple[Counter[str], Counter[tuple[str, str]], set[int]]:
    artist_counts: Counter[str] = Counter()
    album_artist_counts: Counter[tuple[str, str]] = Counter()
    played: set[int] = set()
    music_turns = [c for c in ex.chat_history if c.get("role") == "music" and c.get("content")]
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


# -------------------- retrievers --------------------


PadResult = tuple[np.ndarray, np.ndarray]


def _pad_cands(cand_lists: list[np.ndarray], top_k: int) -> PadResult:
    n = len(cand_lists)
    cand = np.full((n, top_k), -1, dtype=np.int32)
    sizes = np.zeros(n, dtype=np.int32)
    for i, arr in enumerate(cand_lists):
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=np.int32)
        if len(arr) == 0:
            continue
        k = min(len(arr), top_k)
        cand[i, :k] = arr[:k]
        sizes[i] = k
    return cand, sizes


def _topk_argpartition(score: np.ndarray, k: int, played: set[int]) -> np.ndarray:
    if played:
        score = score.copy()
        score[np.fromiter(played, dtype=np.int32)] = -np.inf
    n = len(score)
    k = min(k, n)
    if k <= 0:
        return np.empty(0, dtype=np.int32)
    part = np.argpartition(-score, k - 1)[:k]
    order = np.argsort(-score[part])
    return part[order].astype(np.int32)


def retrieve_bm25(
    examples, track_index, bm25_name: str, query_mode: str, top_k: int,
) -> PadResult:
    bm25 = track_index.bm25_indexes[bm25_name]
    out_lists: list[np.ndarray] = []
    for ex in tqdm(examples, desc=f"bm25[{bm25_name}/{query_mode}]"):
        played = played_set(ex, track_index)
        query = _bm25_query_text(ex, track_index.meta_by_id, mode=query_mode)
        pool = min(top_k + len(played) + 16, track_index.n_tracks)
        toks = bm25s.tokenize([query], show_progress=False)
        idx_arr, _ = bm25.retrieve(toks, k=pool, show_progress=False)
        kept: list[int] = []
        for i in idx_arr[0]:
            ii = int(i)
            if ii in played:
                continue
            kept.append(ii)
            if len(kept) >= top_k:
                break
        out_lists.append(np.asarray(kept, dtype=np.int32))
    return _pad_cands(out_lists, top_k)


def _retrieve_count_score(examples, track_index, score_fn, desc: str, top_k: int) -> PadResult:
    out_lists: list[np.ndarray] = []
    for ex in tqdm(examples, desc=desc):
        played = played_set(ex, track_index)
        score = score_fn(ex, track_index)
        if score is None:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        nz = np.flatnonzero(score > 0)
        if len(nz) == 0:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        order = np.argsort(-score[nz])
        kept = nz[order]
        if played:
            played_arr = np.fromiter(played, dtype=np.int32)
            kept = kept[~np.isin(kept, played_arr)]
        out_lists.append(kept[:top_k].astype(np.int32))
    return _pad_cands(out_lists, top_k)


def _score_history_artist(ex, track_index):
    h_arts, _, _, _, _ = history_state(ex, track_index)
    if not h_arts:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for aid in h_arts:
        for j in track_index.artist_to_idx.get(aid, []):
            score[j] += 1.0
    return score


def _score_history_album(ex, track_index):
    _, h_albs, _, _, _ = history_state(ex, track_index)
    if not h_albs:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for alid in h_albs:
        for j in track_index.album_to_idx.get(alid, []):
            score[j] += 1.0
    return score


def _score_history_artist_album(ex, track_index):
    h_arts, h_albs, _, _, _ = history_state(ex, track_index)
    if not h_arts and not h_albs:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for aid in h_arts:
        for j in track_index.artist_to_idx.get(aid, []):
            score[j] += 1.0
    for alid in h_albs:
        for j in track_index.album_to_idx.get(alid, []):
            score[j] += 1.0
    return score


def _score_history_primary_tag(ex, track_index):
    _, _, h_tags, _, _ = history_state(ex, track_index)
    if not h_tags:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for tag in h_tags:
        for j in track_index.primary_tag_to_idx.get(tag, []):
            score[j] += 1.0
    return score


def _last_music_meta(ex, track_index):
    for c in reversed(ex.chat_history):
        if c.get("role") == "music" and c.get("content"):
            return track_index.meta_by_id.get(c["content"])
    return None


def _score_last_music_artist(ex, track_index):
    md = _last_music_meta(ex, track_index)
    if md is None:
        return None
    artists: set[str] = set()
    for x in as_list(md.get("artist_id")):
        if x:
            artists.add(str(x))
    if not artists:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for aid in artists:
        for j in track_index.artist_to_idx.get(aid, []):
            score[j] += 1.0
    return score


def _score_last_music_album(ex, track_index):
    md = _last_music_meta(ex, track_index)
    if md is None:
        return None
    albs: set[str] = set()
    for x in as_list(md.get("album_id")):
        if x:
            albs.add(str(x))
    if not albs:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for alid in albs:
        for j in track_index.album_to_idx.get(alid, []):
            score[j] += 1.0
    return score


def _score_history_release_decade(ex, track_index):
    decades: set[str] = set()
    for c in ex.chat_history:
        if c.get("role") != "music":
            continue
        md = track_index.meta_by_id.get(c.get("content", ""))
        if md is None:
            continue
        rd = md.get("release_date")
        if not rd:
            continue
        s = str(rd).strip()
        if len(s) >= 4 and s[:4].isdigit():
            decades.add(s[:3] + "0s")
    if not decades:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for i, tid in enumerate(track_index.track_ids):
        md = track_index.meta_by_id.get(tid, {})
        rd = md.get("release_date")
        if not rd:
            continue
        s = str(rd).strip()
        if len(s) >= 4 and s[:4].isdigit():
            d = s[:3] + "0s"
            if d in decades:
                score[i] = 1.0
    return score


def _score_history_artist_name(ex, track_index):
    artist_counts, _, _ = _history_name_counts(ex, track_index, last_only=False)
    if not artist_counts:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for name, cnt in artist_counts.items():
        for j in track_index.artist_name_to_idx.get(name, []):
            score[j] += float(cnt)
    return score


def _score_last_music_artist_name(ex, track_index):
    artist_counts, _, _ = _history_name_counts(ex, track_index, last_only=True)
    if not artist_counts:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for name, cnt in artist_counts.items():
        for j in track_index.artist_name_to_idx.get(name, []):
            score[j] += float(cnt)
    return score


def _score_history_album_artist_name(ex, track_index):
    _, album_artist_counts, _ = _history_name_counts(ex, track_index, last_only=False)
    if not album_artist_counts:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for key, cnt in album_artist_counts.items():
        for j in track_index.album_artist_name_to_idx.get(key, []):
            score[j] += float(cnt)
    return score


def _score_last_album_artist_name(ex, track_index):
    _, album_artist_counts, _ = _history_name_counts(ex, track_index, last_only=True)
    if not album_artist_counts:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for key, cnt in album_artist_counts.items():
        for j in track_index.album_artist_name_to_idx.get(key, []):
            score[j] += float(cnt)
    return score


def _score_exact_title_artist_source(ex, track_index):
    text = _current_request_norm(ex)
    titles = _matched_catalog_names(
        text,
        track_index.track_name_rare_bucket,
        min_chars=5,
        min_tokens=1,
    )
    artists = _matched_catalog_names(
        text,
        track_index.artist_name_rare_bucket,
        min_chars=3,
        min_tokens=1,
    )
    if not titles or not artists:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for title in titles:
        for j in track_index.track_name_to_idx.get(title, []):
            if track_index.track_artist_name_keys[j] & artists:
                score[j] += 10.0
    if not np.any(score > 0):
        return None
    return score


def _score_exact_album_artist_source(ex, track_index):
    text = _current_request_norm(ex)
    albums = _matched_catalog_names(
        text,
        track_index.album_name_rare_bucket,
        min_chars=10,
        min_tokens=2,
    )
    artists = _matched_catalog_names(
        text,
        track_index.artist_name_rare_bucket,
        min_chars=3,
        min_tokens=1,
    )
    if not albums or not artists:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for album in albums:
        for artist in artists:
            for j in track_index.album_artist_name_to_idx.get((album, artist), []):
                score[j] += 5.0
    if not np.any(score > 0):
        return None
    return score


def _score_current_artist_catalog_source(ex, track_index):
    text = _current_request_norm(ex)
    artists = _matched_catalog_names(
        text,
        track_index.artist_name_rare_bucket,
        min_chars=8,
        min_tokens=1,
    )
    if not artists:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for artist in artists:
        for j in track_index.artist_name_to_idx.get(artist, []):
            score[j] += 1.0
    if not np.any(score > 0):
        return None
    return score


def _score_secondary_artist_source(ex, track_index):
    text = _current_request_norm(ex)
    artists = _matched_catalog_names(
        text,
        track_index.artist_name_rare_bucket,
        min_chars=3,
        min_tokens=1,
    )
    if not artists:
        return None
    score = np.zeros(track_index.n_tracks, dtype=np.float32)
    for artist in artists:
        for j in track_index.secondary_artist_name_to_idx.get(artist, []):
            score[j] += 1.0
    if not np.any(score > 0):
        return None
    return score


def retrieve_popularity_global(examples, track_index, top_k: int) -> PadResult:
    out_lists: list[np.ndarray] = []
    pop_order = track_index.popularity_order
    for ex in tqdm(examples, desc="popularity_global"):
        played = played_set(ex, track_index)
        if played:
            kept = pop_order[~np.isin(pop_order, np.fromiter(played, dtype=np.int32))]
        else:
            kept = pop_order
        out_lists.append(kept[:top_k].astype(np.int32))
    return _pad_cands(out_lists, top_k)


def retrieve_cf_history_centroid(examples, track_index, top_k: int) -> PadResult:
    out_lists: list[np.ndarray] = []
    for ex in tqdm(examples, desc="cf_history_centroid"):
        _, _, _, played, history_idxs = history_state(ex, track_index)
        if not history_idxs:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        hist_arr = np.asarray(history_idxs, dtype=np.int32)
        centroid = track_index.cf[hist_arr].mean(axis=0)
        cn = float(np.linalg.norm(centroid))
        if cn == 0:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        centroid /= cn
        score = track_index.cf @ centroid
        kept = _topk_argpartition(score, top_k, played)
        out_lists.append(kept)
    return _pad_cands(out_lists, top_k)


def retrieve_user_emb_track_cf(examples, track_index, user_vectors, top_k: int) -> PadResult:
    out_lists: list[np.ndarray] = []
    for ex in tqdm(examples, desc="user_emb_track_cf"):
        played = played_set(ex, track_index)
        vec = user_vectors.get(ex.user_id)
        if vec is None:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        score = track_index.cf @ vec
        kept = _topk_argpartition(score, top_k, played)
        out_lists.append(kept)
    return _pad_cands(out_lists, top_k)


def retrieve_cooc_track(examples, track_index, cooc, top_k: int) -> PadResult:
    out_lists: list[np.ndarray] = []
    for ex in tqdm(examples, desc="cooc_track"):
        _, _, _, played, history_idxs = history_state(ex, track_index)
        if not history_idxs:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        score = np.zeros(track_index.n_tracks, dtype=np.float32)
        for h in history_idxs:
            nb_cn = cooc.track_track.get(h)
            if nb_cn is None:
                continue
            nb, cn = nb_cn
            score[nb] += cn
        if played:
            score[np.fromiter(played, dtype=np.int32)] = 0.0
        nz = np.flatnonzero(score > 0)
        if len(nz) == 0:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        order = np.argsort(-score[nz])
        out_lists.append(nz[order][:top_k].astype(np.int32))
    return _pad_cands(out_lists, top_k)


def retrieve_cooc_artist(examples, track_index, cooc, top_k: int) -> PadResult:
    out_lists: list[np.ndarray] = []
    for ex in tqdm(examples, desc="cooc_artist"):
        h_arts, _, _, played, _ = history_state(ex, track_index)
        if not h_arts:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        artist_score: dict[str, float] = defaultdict(float)
        for aid in h_arts:
            ca = cooc.artist_artist.get(aid)
            if ca is None:
                continue
            for nb_aid, c in ca.items():
                artist_score[nb_aid] += c
        if not artist_score:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        score = np.zeros(track_index.n_tracks, dtype=np.float32)
        for nb_aid, s in artist_score.items():
            for j in track_index.artist_to_idx.get(nb_aid, []):
                score[j] += s
        nz = np.flatnonzero(score > 0)
        if len(nz) == 0:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        order = np.argsort(-score[nz])
        kept = nz[order]
        if played:
            played_arr = np.fromiter(played, dtype=np.int32)
            kept = kept[~np.isin(kept, played_arr)]
        out_lists.append(kept[:top_k].astype(np.int32))
    return _pad_cands(out_lists, top_k)


def _retrieve_artist_name_neighbor_source(
    examples,
    track_index,
    neighbor_table: dict[str, Counter],
    *,
    last_only: bool,
    desc: str,
    top_k: int,
) -> PadResult:
    out_lists: list[np.ndarray] = []
    for ex in tqdm(examples, desc=desc):
        artist_counts, _, _ = _history_name_counts(ex, track_index, last_only=last_only)
        if not artist_counts:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        seed_names = set(artist_counts)
        neighbor_score: dict[str, float] = defaultdict(float)
        for seed_name, seed_count in artist_counts.items():
            neighbors = neighbor_table.get(seed_name)
            if neighbors is None:
                continue
            for nb_name, cooc_count in neighbors.items():
                if nb_name in seed_names:
                    continue
                neighbor_score[nb_name] += float(seed_count) * float(cooc_count)
        if not neighbor_score:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        score = np.zeros(track_index.n_tracks, dtype=np.float32)
        for nb_name, s in neighbor_score.items():
            for j in track_index.artist_name_to_idx.get(nb_name, []):
                score[j] += s
        nz = np.flatnonzero(score > 0)
        if len(nz) == 0:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        order = np.argsort(-score[nz])
        kept = nz[order]
        played = played_set(ex, track_index)
        if played:
            played_arr = np.fromiter(played, dtype=np.int32)
            kept = kept[~np.isin(kept, played_arr)]
        out_lists.append(kept[:top_k].astype(np.int32))
    return _pad_cands(out_lists, top_k)


def retrieve_cooc_artist_name(examples, track_index, cooc, top_k: int) -> PadResult:
    return _retrieve_artist_name_neighbor_source(
        examples,
        track_index,
        cooc.artist_name_artist_name,
        last_only=False,
        desc="cooc_artist_name",
        top_k=top_k,
    )


def retrieve_transition_artist_name_last(examples, track_index, cooc, top_k: int) -> PadResult:
    return _retrieve_artist_name_neighbor_source(
        examples,
        track_index,
        cooc.transition_artist_name,
        last_only=True,
        desc="transition_artist_name_last",
        top_k=top_k,
    )


def retrieve_dense(examples, track_index, dense_key: str, query_mat: np.ndarray, top_k: int) -> PadResult:
    track_mat = track_index.dense_mats[dense_key]
    out_lists: list[np.ndarray] = []
    for i, ex in enumerate(tqdm(examples, desc=f"dense[{dense_key}]")):
        played = played_set(ex, track_index)
        q = query_mat[i]
        score = track_mat @ q
        kept = _topk_argpartition(score, top_k, played)
        out_lists.append(kept)
    return _pad_cands(out_lists, top_k)


def retrieve_semantic_dense(
    examples,
    track_index,
    dense_key: str,
    query_mat: np.ndarray,
    top_k: int,
    desc: str,
) -> PadResult:
    return retrieve_dense(examples, track_index, dense_key, query_mat, top_k)


def load_or_encode_semantic_queries(examples, cache_path, batch_size: int = 16):
    keys = np.asarray([_semantic_query_key(ex) for ex in examples])

    def load_npz(path: Path):
        data = np.load(path)
        if "keys" not in data or "embeddings" not in data:
            return None
        stored_keys = data["keys"].astype(str)
        emb = data["embeddings"].astype(np.float32)
        if len(stored_keys) == len(keys) and np.array_equal(stored_keys, keys):
            print(f"  [cache] semantic dense queries: {path}")
            return _normalize_rows(emb)
        pos = {k: i for i, k in enumerate(stored_keys)}
        if all(k in pos for k in keys):
            print(f"  [cache] semantic dense queries selected from {path}")
            return _normalize_rows(emb[[pos[k] for k in keys]])
        return None

    if SEMANTIC_QUERY_CACHE.exists():
        cached = load_npz(SEMANTIC_QUERY_CACHE)
        if cached is not None:
            return cached
    if Path(cache_path).exists():
        cached = load_npz(Path(cache_path))
        if cached is not None:
            return cached

    from recsys2026.encoders import Qwen3TextEncoder

    encoder = Qwen3TextEncoder(batch_size=batch_size)
    qs = [_semantic_query_text(ex) for ex in examples]
    print(f"  encoding {len(qs)} semantic queries with Qwen3 ...")
    mat = _normalize_rows(encoder.encode(qs))
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, keys=keys, embeddings=mat)
    return mat


def retrieve_tag_intent_bm25(examples, track_index, top_k: int) -> PadResult:
    bm25 = track_index.bm25_indexes["tag_list"]
    out_lists: list[np.ndarray] = []
    for ex in tqdm(examples, desc="tag_intent_bm25"):
        query = _tag_intent_query(ex)
        if not query:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        played = played_set(ex, track_index)
        pool = min(top_k + len(played) + 16, track_index.n_tracks)
        toks = bm25s.tokenize([query], show_progress=False)
        idx_arr, _ = bm25.retrieve(toks, k=pool, show_progress=False)
        kept: list[int] = []
        for i in idx_arr[0]:
            ii = int(i)
            if ii in played:
                continue
            kept.append(ii)
            if len(kept) >= top_k:
                break
        out_lists.append(np.asarray(kept, dtype=np.int32))
    return _pad_cands(out_lists, top_k)


# -------------------- query encoders --------------------


def _build_query_texts(examples) -> list[str]:
    from recsys2026.retrieval import chat_to_query_text
    from recsys2026.submission import InferenceInput
    queries: list[str] = []
    for ex in examples:
        inp = InferenceInput(
            session_id=ex.session_id,
            user_id=ex.user_id,
            turn_number=ex.turn_number,
            chat_history=ex.chat_history,
            user_query=ex.user_query,
        )
        queries.append(chat_to_query_text(inp, mode="full"))
    return queries


def encode_queries_qwen3(examples, cache_path, batch_size: int = 16):
    if cache_path.exists():
        print(f"  [cache] {cache_path}")
        return np.load(cache_path)
    from recsys2026.encoders import Qwen3TextEncoder
    encoder = Qwen3TextEncoder(batch_size=batch_size)
    qs = _build_query_texts(examples)
    print(f"  encoding {len(qs)} queries with Qwen3 ...")
    mat = _normalize_rows(encoder.encode(qs))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mat)
    return mat


def encode_queries_clap(examples, cache_path, batch_size: int = 32):
    if cache_path.exists():
        print(f"  [cache] {cache_path}")
        return np.load(cache_path)
    try:
        from recsys2026.encoders import ClapTextEncoder
    except Exception as e:
        print(f"  [skip] CLAP unavailable: {e}")
        return None
    encoder = ClapTextEncoder(batch_size=batch_size)
    qs = _build_query_texts(examples)
    print(f"  encoding {len(qs)} queries with CLAP ...")
    mat = _normalize_rows(encoder.encode(qs))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mat)
    return mat


def encode_queries_siglip(examples, cache_path, batch_size: int = 32):
    if cache_path.exists():
        print(f"  [cache] {cache_path}")
        return np.load(cache_path)
    try:
        from recsys2026.encoders import SigLIPTextEncoder
    except Exception as e:
        print(f"  [skip] SigLIP unavailable: {e}")
        return None
    encoder = SigLIPTextEncoder(batch_size=batch_size)
    qs = _build_query_texts(examples)
    print(f"  encoding {len(qs)} queries with SigLIP ...")
    mat = _normalize_rows(encoder.encode(qs))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mat)
    return mat


# -------------------- Phase 2 LLM --------------------


def chat_to_text_for_llm(ex, track_meta) -> str:
    parts: list[str] = []
    for c in ex.chat_history:
        role = c.get("role", "user")
        content = c.get("content", "")
        if role == "music":
            md = track_meta.get(content)
            if md is None:
                continue
            role = "assistant"
            content = _track_metadata_str(content, md)
        parts.append(f"{role}: {content}")
    parts.append(f"user: {ex.user_query}")
    return "\n".join(parts)


def llm_rewrite(
    examples, track_index, cache_path, system_prompt: str,
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    batch_size: int = 16, max_new_tokens: int = 80,
) -> list[str]:
    if cache_path.exists():
        print(f"  [cache] rewrites {cache_path}")
        return json.loads(cache_path.read_text())

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"  loading {model_name} on {device} (dtype={dtype}) ...")
    tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()

    chat_texts = [chat_to_text_for_llm(ex, track_index.meta_by_id) for ex in examples]
    prompts = []
    for ct in chat_texts:
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": ct},
        ]
        prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    out: list[str] = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompts), batch_size), desc="llm rewrite"):
            chunk = prompts[i:i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=True).to(device)
            gen = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
                do_sample=False,
            )
            new_tokens = gen[:, enc["input_ids"].shape[1]:]
            decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
            for s in decoded:
                line = s.strip().splitlines()[0] if s.strip() else ""
                out.append(line.strip())

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, ensure_ascii=False))
    print(f"  saved {cache_path}")

    del model, tok
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


HYDE_PROMPT = (
    "You are a music recommendation assistant. Given a conversation between a user "
    "and an assistant, write ONE line describing the kind of song the user is asking "
    "for. Include any of: artist, genre, mood, tempo, era, lyrical theme, instruments. "
    "No greeting, no explanation — just the description."
)

INTENT_TAG_PROMPT = (
    "You are a music tagging assistant. Given a conversation between a user and an "
    "assistant, output a JSON array of 3 to 5 short music tags (e.g., genres, moods, "
    "eras, instruments, vocal styles) describing the song the user wants next. "
    "Output ONLY the JSON array, no explanation."
)


def retrieve_hyde_bm25(rewrites, examples, track_index, bm25_name, top_k):
    bm25 = track_index.bm25_indexes[bm25_name]
    out_lists: list[np.ndarray] = []
    for i, ex in enumerate(tqdm(examples, desc=f"hyde_bm25[{bm25_name}]")):
        played = played_set(ex, track_index)
        rw = rewrites[i] if i < len(rewrites) else ""
        if not rw:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        pool = min(top_k + len(played) + 16, track_index.n_tracks)
        toks = bm25s.tokenize([rw.lower()], show_progress=False)
        idx_arr, _ = bm25.retrieve(toks, k=pool, show_progress=False)
        kept: list[int] = []
        for j in idx_arr[0]:
            jj = int(j)
            if jj in played:
                continue
            kept.append(jj)
            if len(kept) >= top_k:
                break
        out_lists.append(np.asarray(kept, dtype=np.int32))
    return _pad_cands(out_lists, top_k)


def retrieve_hyde_dense(rewrites, examples, track_index, dense_key, cache_path, top_k, qwen_batch=16):
    track_mat = track_index.dense_mats[dense_key]
    if cache_path.exists():
        print(f"  [cache] hyde encoding {cache_path}")
        query_mat = np.load(cache_path)
    else:
        from recsys2026.encoders import Qwen3TextEncoder
        encoder = Qwen3TextEncoder(batch_size=qwen_batch)
        print(f"  encoding {len(rewrites)} hyde rewrites with Qwen3 ...")
        query_mat = _normalize_rows(encoder.encode([rw if rw else " " for rw in rewrites]))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, query_mat)
        print(f"  saved {cache_path}")

    out_lists: list[np.ndarray] = []
    for i, ex in enumerate(tqdm(examples, desc=f"hyde_dense[{dense_key}]")):
        played = played_set(ex, track_index)
        q = query_mat[i]
        score = track_mat @ q
        kept = _topk_argpartition(score, top_k, played)
        out_lists.append(kept)
    return _pad_cands(out_lists, top_k)


def _parse_tag_array(s: str) -> list[str]:
    s = s.strip()
    if s.startswith("["):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [str(x).strip().lower() for x in arr if str(x).strip()]
        except Exception:
            pass
    parts = [p.strip().strip("\"'`[]").lower() for p in s.split(",")]
    return [p for p in parts if p]


def retrieve_intent_tag_match(rewrites, examples, track_index, top_k):
    tag_token_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, tid in enumerate(track_index.track_ids):
        meta = track_index.meta_by_id.get(tid, {})
        tags = as_list(meta.get("tag_list"))
        for tag in tags:
            t = str(tag).lower().strip()
            if t:
                tag_token_to_idx[t].append(i)
                for tok in t.split():
                    if len(tok) > 2:
                        tag_token_to_idx[tok].append(i)

    out_lists: list[np.ndarray] = []
    for i, ex in enumerate(tqdm(examples, desc="intent_tag_match")):
        played = played_set(ex, track_index)
        rw = rewrites[i] if i < len(rewrites) else ""
        tags = _parse_tag_array(rw)
        if not tags:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        score = np.zeros(track_index.n_tracks, dtype=np.float32)
        for tag in tags:
            idxs = tag_token_to_idx.get(tag, [])
            for j in idxs:
                score[j] += 1.0
            for sub in tag.split():
                if len(sub) > 2:
                    for j in tag_token_to_idx.get(sub, []):
                        score[j] += 0.5
        nz = np.flatnonzero(score > 0)
        if len(nz) == 0:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        order = np.argsort(-score[nz])
        kept = nz[order]
        if played:
            played_arr = np.fromiter(played, dtype=np.int32)
            kept = kept[~np.isin(kept, played_arr)]
        out_lists.append(kept[:top_k].astype(np.int32))
    return _pad_cands(out_lists, top_k)


def _build_album_qwen3_matrix(track_index, cache_path, batch_size=32):
    if cache_path.exists():
        print(f"  [cache] album_qwen3 {cache_path}")
        return np.load(cache_path)

    from recsys2026.encoders import Qwen3TextEncoder
    encoder = Qwen3TextEncoder(batch_size=batch_size)
    album_texts: list[str] = []
    for tid in track_index.track_ids:
        meta = track_index.meta_by_id.get(tid, {})
        a = meta.get("album_name") or ""
        artist = meta.get("artist_name") or ""
        if isinstance(artist, list):
            artist = ", ".join(str(x) for x in artist if x)
        if isinstance(a, list):
            a = ", ".join(str(x) for x in a if x)
        album_texts.append(f"album: {a} | artist: {artist}".strip())
    print(f"  encoding {len(album_texts)} track album names with Qwen3 ...")
    mat = _normalize_rows(encoder.encode(album_texts))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mat)
    return mat


def retrieve_album_qwen3_history(examples, track_index, album_mat, top_k):
    out_lists: list[np.ndarray] = []
    for ex in tqdm(examples, desc="album_qwen3_history"):
        _, _, _, played, history_idxs = history_state(ex, track_index)
        if not history_idxs:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        hist_arr = np.asarray(history_idxs, dtype=np.int32)
        centroid = album_mat[hist_arr].mean(axis=0)
        cn = float(np.linalg.norm(centroid))
        if cn == 0:
            out_lists.append(np.empty(0, dtype=np.int32))
            continue
        centroid /= cn
        score = album_mat @ centroid
        kept = _topk_argpartition(score, top_k, played)
        out_lists.append(kept)
    return _pad_cands(out_lists, top_k)


# -------------------- BM25 catalog --------------------


BM25_RETRIEVERS: tuple[tuple[str, str, str], ...] = (
    ("bm25_4field",          "4field",       "full"),
    ("bm25_5field",          "5field",       "full"),
    ("bm25_track_name",      "track_name",   "full"),
    ("bm25_artist_name",     "artist_name",  "full"),
    ("bm25_album_name",      "album_name",   "full"),
    ("bm25_tag_list",        "tag_list",     "full"),
    ("bm25_history_boost",   "4field",       "history_boost"),
    ("bm25_drop_music",      "4field",       "drop_music"),
    ("bm25_user_only",       "4field",       "user_only"),
    ("bm25_with_thought",    "4field",       "with_thought"),
    ("bm25_thought_only",    "4field",       "thought_only"),
    ("bm25_artist_album",    "artist_album", "full"),
    ("bm25_5field_thought",  "5field",       "with_thought"),
)

BM25_VARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("4field",       ("track_name", "artist_name", "album_name", "release_date")),
    ("5field",       ("track_name", "artist_name", "album_name", "release_date", "tag_list")),
    ("track_name",   ("track_name",)),
    ("artist_name",  ("artist_name",)),
    ("album_name",   ("album_name",)),
    ("tag_list",     ("tag_list",)),
    ("artist_album", ("artist_name", "album_name")),
)


# -------------------- evaluation --------------------


def compute_metrics(cand, sizes, gold_idxs, recall_ks=RECALL_KS):
    n = len(cand)
    valid = gold_idxs >= 0
    n_v = int(valid.sum())
    metrics = {
        "n_examples": int(n),
        "n_valid_gold": n_v,
        "nonempty_rate": float((sizes > 0).mean()) if n > 0 else 0.0,
        "mean_cand_size": float(sizes.mean()) if n > 0 else 0.0,
        "median_cand_size": float(np.median(sizes)) if n > 0 else 0.0,
        "p90_cand_size": float(np.percentile(sizes, 90)) if n > 0 else 0.0,
    }
    for k in recall_ks:
        cand_k = cand[:, :k]
        if n == 0 or n_v == 0:
            metrics[f"recall@{k}"] = 0.0
            continue
        gold_col = gold_idxs[:, None]
        hit = (cand_k == gold_col).any(axis=1) & valid
        metrics[f"recall@{k}"] = float(hit.sum() / n_v)
    return metrics


def compute_union_metrics(cand_dict, sizes_dict, members, gold_idxs, top_k):
    n = len(gold_idxs)
    valid = gold_idxs >= 0
    n_v = int(valid.sum())
    union_sizes = np.zeros(n, dtype=np.int32)
    union_hits = np.zeros(n, dtype=bool)
    for i in range(n):
        u: set[int] = set()
        for m in members:
            cand = cand_dict[m][i]
            sz = int(sizes_dict[m][i])
            if sz > 0:
                u.update(int(x) for x in cand[:sz])
        union_sizes[i] = len(u)
        if gold_idxs[i] >= 0 and len(u) > 0 and gold_idxs[i] in u:
            union_hits[i] = True
    return {
        "members": list(members),
        "n_examples": int(n),
        "n_valid_gold": n_v,
        "mean_union_size": float(union_sizes.mean()),
        "median_union_size": float(np.median(union_sizes)),
        "p90_union_size": float(np.percentile(union_sizes, 90)),
        "recall_at_union": float(union_hits.sum() / n_v) if n_v > 0 else 0.0,
    }


# -------------------- main --------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["devset"], default="devset")
    parser.add_argument("--top_k", type=int, default=TOP_K)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--skip", type=str, default=None)
    parser.add_argument("--skip_dense", action="store_true")
    parser.add_argument("--skip_clap", action="store_true")
    parser.add_argument("--skip_siglip", action="store_true")
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--qwen_batch", type=int, default=16)
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--llm_batch", type=int, default=16)
    parser.add_argument("--rebuild_cache", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cand_dir = OUT_DIR / "cand"
    cand_dir.mkdir(parents=True, exist_ok=True)
    encode_dir = OUT_DIR / "encode"
    encode_dir.mkdir(parents=True, exist_ok=True)

    print("loading devset examples ...")
    examples = build_examples_devset()
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
    n_ex = len(examples)
    print(f"  {n_ex} turn examples")

    cheap_names = (
        [name for name, _, _ in BM25_RETRIEVERS]
        + [
            "history_artist", "history_album", "history_artist_album", "history_primary_tag",
            "last_music_artist", "last_music_album", "history_release_decade",
            "history_artist_name", "last_music_artist_name",
            "history_album_artist_name", "last_album_artist_name",
            "popularity_global",
            "cf_history_centroid", "user_emb_track_cf",
            "cooc_track", "cooc_artist", "cooc_artist_name", "transition_artist_name_last",
            "tag_intent_bm25",
            "exact_title_artist_source", "exact_album_artist_source",
            "current_artist_catalog_source", "secondary_artist_source",
            "personal_exact_repeat", "personal_artist_expansion", "personal_album_expansion",
        ]
    )
    dense_names = list(DENSE_COLS.keys())
    semantic_dense_names = list(SEMANTIC_DENSE_SOURCES.keys())
    llm_names = [
        "hyde_bm25_4field",
        "hyde_bm25_5field",
        "hyde_dense_qwen3_metadata",
        "intent_tag_match",
        "album_qwen3_history",
    ]

    all_names = cheap_names + dense_names + semantic_dense_names + llm_names

    if args.only:
        only_set = {x.strip() for x in args.only.split(",")}
        all_names = [n for n in all_names if n in only_set]
    if args.skip:
        skip_set = {x.strip() for x in args.skip.split(",")}
        all_names = [n for n in all_names if n not in skip_set]
    if args.skip_dense:
        all_names = [n for n in all_names if not n.startswith("dense_")]
    if args.skip_clap:
        all_names = [n for n in all_names if n != "dense_clap_audio"]
    if args.skip_siglip:
        all_names = [n for n in all_names if n != "dense_siglip_image"]
    if args.skip_llm:
        all_names = [n for n in all_names if n not in llm_names]

    print(f"running {len(all_names)} retrievers: {all_names}")

    needed_bm25 = set()
    for name, bm_name, mode in BM25_RETRIEVERS:
        if name in all_names:
            needed_bm25.add(bm_name)
    if "tag_intent_bm25" in all_names:
        needed_bm25.add("tag_list")
    bm25_variants = tuple(v for v in BM25_VARIANTS if v[0] in needed_bm25)

    needed_dense_set = {n for n in DENSE_COLS if n in all_names}
    for name, dense_key in SEMANTIC_DENSE_SOURCES.items():
        if name in all_names:
            needed_dense_set.add(dense_key)
    needed_dense = tuple(n for n in DENSE_COLS if n in needed_dense_set)
    if "hyde_dense_qwen3_metadata" in all_names and "dense_qwen3_metadata" not in needed_dense:
        needed_dense = needed_dense + ("dense_qwen3_metadata",)

    track_index = build_track_index(bm25_variants, needed_dense)

    gold_idxs = np.full(n_ex, -1, dtype=np.int64)
    for i, ex in enumerate(examples):
        if ex.gold_track_id:
            gold_idxs[i] = track_index.id_to_idx.get(ex.gold_track_id, -1)
    print(f"gold valid: {(gold_idxs >= 0).sum()} / {n_ex}")

    cooc = None
    if any(n in all_names for n in ("cooc_track", "cooc_artist", "cooc_artist_name", "transition_artist_name_last")):
        cooc = build_cooc(track_index)

    personal_memory = None
    personal_keys = {"personal_exact_repeat", "personal_artist_expansion", "personal_album_expansion"}
    if any(n in all_names for n in personal_keys):
        personal_memory = build_personal_memory(track_index)

    user_vectors: dict[str, np.ndarray] = {}
    if "user_emb_track_cf" in all_names:
        user_vectors = load_user_vectors_normalized()
        print(f"loaded user vectors: {len(user_vectors)}")

    qwen_query_mat = None
    qwen_keys = {"dense_qwen3_metadata", "dense_qwen3_attributes", "dense_qwen3_lyrics"}
    if any(n in all_names for n in qwen_keys):
        qwen_query_mat = encode_queries_qwen3(
            examples, encode_dir / f"qwen3_query_mat__n{n_ex}.npy", batch_size=args.qwen_batch,
        )

    semantic_query_mat = None
    semantic_keys = set(SEMANTIC_DENSE_SOURCES)
    if any(n in all_names for n in semantic_keys):
        semantic_query_mat = load_or_encode_semantic_queries(
            examples,
            encode_dir / f"semantic_qwen3_query_goal_current__n{n_ex}.npz",
            batch_size=args.qwen_batch,
        )

    clap_query_mat = None
    if "dense_clap_audio" in all_names:
        clap_query_mat = encode_queries_clap(examples, encode_dir / f"clap_query_mat__n{n_ex}.npy")
        if clap_query_mat is None:
            all_names = [n for n in all_names if n != "dense_clap_audio"]

    siglip_query_mat = None
    if "dense_siglip_image" in all_names:
        siglip_query_mat = encode_queries_siglip(examples, encode_dir / f"siglip_query_mat__n{n_ex}.npy")
        if siglip_query_mat is None:
            all_names = [n for n in all_names if n != "dense_siglip_image"]

    hyde_rewrites = None
    intent_rewrites = None
    album_qwen3_mat = None

    hyde_keys = {"hyde_bm25_4field", "hyde_bm25_5field", "hyde_dense_qwen3_metadata"}
    if any(n in all_names for n in hyde_keys):
        hyde_rewrites = llm_rewrite(
            examples, track_index,
            encode_dir / f"hyde_rewrites__{args.llm_model.replace('/', '_')}__n{n_ex}.json",
            HYDE_PROMPT,
            model_name=args.llm_model,
            batch_size=args.llm_batch,
        )

    if "intent_tag_match" in all_names:
        intent_rewrites = llm_rewrite(
            examples, track_index,
            encode_dir / f"intent_tags__{args.llm_model.replace('/', '_')}__n{n_ex}.json",
            INTENT_TAG_PROMPT,
            model_name=args.llm_model,
            batch_size=args.llm_batch,
            max_new_tokens=120,
        )

    if "album_qwen3_history" in all_names:
        album_qwen3_mat = _build_album_qwen3_matrix(
            track_index,
            encode_dir / "album_qwen3_matrix.npy",
            batch_size=args.qwen_batch,
        )

    cand_dict: dict[str, np.ndarray] = {}
    sizes_dict: dict[str, np.ndarray] = {}
    metrics_dict: dict[str, dict] = {}
    timing: dict[str, float] = {}

    bm25_lookup = {name: (bm_name, mode) for name, bm_name, mode in BM25_RETRIEVERS}

    for name in all_names:
        cache_path = cand_dir / f"{name}__n{n_ex}.npz"
        if cache_path.exists() and not args.rebuild_cache:
            data = np.load(cache_path)
            cand = data["cand"]
            sizes = data["sizes"]
            try:
                elapsed = float(data["elapsed"].item())
            except KeyError:
                elapsed = 0.0
            print(f"[cache] {name}: cand={cand.shape}")
        else:
            t0 = time.time()
            if name in bm25_lookup:
                bm_name, mode = bm25_lookup[name]
                cand, sizes = retrieve_bm25(examples, track_index, bm_name, mode, args.top_k)
            elif name == "history_artist":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_history_artist, "history_artist", args.top_k)
            elif name == "history_album":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_history_album, "history_album", args.top_k)
            elif name == "history_artist_album":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_history_artist_album, "history_artist_album", args.top_k)
            elif name == "history_primary_tag":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_history_primary_tag, "history_primary_tag", args.top_k)
            elif name == "last_music_artist":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_last_music_artist, "last_music_artist", args.top_k)
            elif name == "last_music_album":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_last_music_album, "last_music_album", args.top_k)
            elif name == "history_release_decade":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_history_release_decade, "history_release_decade", args.top_k)
            elif name == "history_artist_name":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_history_artist_name, "history_artist_name", args.top_k)
            elif name == "last_music_artist_name":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_last_music_artist_name, "last_music_artist_name", args.top_k)
            elif name == "history_album_artist_name":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_history_album_artist_name, "history_album_artist_name", args.top_k)
            elif name == "last_album_artist_name":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_last_album_artist_name, "last_album_artist_name", args.top_k)
            elif name == "exact_title_artist_source":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_exact_title_artist_source, "exact_title_artist_source", args.top_k)
            elif name == "exact_album_artist_source":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_exact_album_artist_source, "exact_album_artist_source", args.top_k)
            elif name == "current_artist_catalog_source":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_current_artist_catalog_source, "current_artist_catalog_source", args.top_k)
            elif name == "secondary_artist_source":
                cand, sizes = _retrieve_count_score(examples, track_index, _score_secondary_artist_source, "secondary_artist_source", args.top_k)
            elif name == "personal_exact_repeat":
                cand, sizes = _retrieve_count_score(
                    examples,
                    track_index,
                    lambda ex, ti: _score_personal_exact_repeat(ex, ti, personal_memory),
                    "personal_exact_repeat",
                    args.top_k,
                )
            elif name == "personal_artist_expansion":
                cand, sizes = _retrieve_count_score(
                    examples,
                    track_index,
                    lambda ex, ti: _score_personal_artist_expansion(ex, ti, personal_memory),
                    "personal_artist_expansion",
                    args.top_k,
                )
            elif name == "personal_album_expansion":
                cand, sizes = _retrieve_count_score(
                    examples,
                    track_index,
                    lambda ex, ti: _score_personal_album_expansion(ex, ti, personal_memory),
                    "personal_album_expansion",
                    args.top_k,
                )
            elif name == "popularity_global":
                cand, sizes = retrieve_popularity_global(examples, track_index, args.top_k)
            elif name == "cf_history_centroid":
                cand, sizes = retrieve_cf_history_centroid(examples, track_index, args.top_k)
            elif name == "user_emb_track_cf":
                cand, sizes = retrieve_user_emb_track_cf(examples, track_index, user_vectors, args.top_k)
            elif name == "cooc_track":
                cand, sizes = retrieve_cooc_track(examples, track_index, cooc, args.top_k)
            elif name == "cooc_artist":
                cand, sizes = retrieve_cooc_artist(examples, track_index, cooc, args.top_k)
            elif name == "cooc_artist_name":
                cand, sizes = retrieve_cooc_artist_name(examples, track_index, cooc, args.top_k)
            elif name == "transition_artist_name_last":
                cand, sizes = retrieve_transition_artist_name_last(examples, track_index, cooc, args.top_k)
            elif name in SEMANTIC_DENSE_SOURCES:
                dense_key = SEMANTIC_DENSE_SOURCES[name]
                cand, sizes = retrieve_semantic_dense(
                    examples, track_index, dense_key, semantic_query_mat, args.top_k, name,
                )
            elif name == "tag_intent_bm25":
                cand, sizes = retrieve_tag_intent_bm25(examples, track_index, args.top_k)
            elif name in qwen_keys:
                cand, sizes = retrieve_dense(examples, track_index, name, qwen_query_mat, args.top_k)
            elif name == "dense_clap_audio":
                cand, sizes = retrieve_dense(examples, track_index, name, clap_query_mat, args.top_k)
            elif name == "dense_siglip_image":
                cand, sizes = retrieve_dense(examples, track_index, name, siglip_query_mat, args.top_k)
            elif name in ("hyde_bm25_4field", "hyde_bm25_5field"):
                bm_name = "4field" if name == "hyde_bm25_4field" else "5field"
                cand, sizes = retrieve_hyde_bm25(hyde_rewrites, examples, track_index, bm_name, args.top_k)
            elif name == "hyde_dense_qwen3_metadata":
                cand, sizes = retrieve_hyde_dense(
                    hyde_rewrites, examples, track_index,
                    "dense_qwen3_metadata",
                    encode_dir / f"hyde_qwen3_query__n{n_ex}.npy",
                    args.top_k,
                    qwen_batch=args.qwen_batch,
                )
            elif name == "intent_tag_match":
                cand, sizes = retrieve_intent_tag_match(intent_rewrites, examples, track_index, args.top_k)
            elif name == "album_qwen3_history":
                cand, sizes = retrieve_album_qwen3_history(examples, track_index, album_qwen3_mat, args.top_k)
            else:
                print(f"  [unknown] {name}")
                continue
            elapsed = time.time() - t0
            np.savez(cache_path, cand=cand, sizes=sizes, elapsed=np.array(elapsed))

        timing[name] = elapsed
        cand_dict[name] = cand
        sizes_dict[name] = sizes
        m = compute_metrics(cand, sizes, gold_idxs)
        m["elapsed_sec"] = elapsed
        metrics_dict[name] = m
        print(
            f"  {name}: nonempty={m['nonempty_rate']:.3f}, "
            f"r@20={m['recall@20']:.4f}, r@200={m['recall@200']:.4f}, "
            f"mean_size={m['mean_cand_size']:.1f}, t={elapsed:.1f}s"
        )

    (RESULTS_DIR / "per_retriever.json").write_text(json.dumps(metrics_dict, indent=2))
    (RESULTS_DIR / "timing.json").write_text(json.dumps(timing, indent=2))

    available = list(cand_dict.keys())
    UNION_COMBOS: list[tuple[str, ...]] = []

    def add_combo(*names):
        if all(n in available for n in names):
            UNION_COMBOS.append(tuple(names))

    add_combo("bm25_4field", "cf_history_centroid")
    add_combo("bm25_4field", "history_artist_album")
    add_combo("bm25_4field", "popularity_global")
    add_combo("bm25_5field", "cf_history_centroid", "history_artist_album", "cooc_track")
    add_combo("bm25_5field_thought", "cf_history_centroid", "history_artist_album", "cooc_track")
    add_combo("bm25_5field_thought", "history_artist_album", "user_emb_track_cf",
              "dense_qwen3_attributes", "cooc_track")
    add_combo("history_artist_name", "last_music_artist_name")
    add_combo("history_album_artist_name", "last_album_artist_name")
    add_combo("history_artist_name", "last_music_artist_name",
              "history_album_artist_name", "last_album_artist_name")
    add_combo("cooc_artist", "cooc_artist_name", "transition_artist_name_last")
    add_combo("attribute_query_rrf", "lyrics_query_rrf", "metadata_query_rrf_nohistory", "tag_intent_bm25")
    add_combo("exact_title_artist_source", "exact_album_artist_source",
              "current_artist_catalog_source", "secondary_artist_source")
    add_combo("personal_exact_repeat", "personal_artist_expansion", "personal_album_expansion")
    add_combo("history_artist_album", "history_artist_name", "last_music_artist_name",
              "history_album_artist_name", "last_album_artist_name")
    top_candidates = [
        "bm25_4field", "bm25_5field", "bm25_5field_thought", "bm25_history_boost",
        "cf_history_centroid", "user_emb_track_cf",
        "history_artist_album", "history_primary_tag",
        "history_artist_name", "last_music_artist_name",
        "history_album_artist_name", "last_album_artist_name",
        "cooc_track", "cooc_artist", "cooc_artist_name", "transition_artist_name_last",
        "popularity_global",
        "attribute_query_rrf", "lyrics_query_rrf", "metadata_query_rrf_nohistory", "tag_intent_bm25",
        "exact_title_artist_source", "exact_album_artist_source",
        "current_artist_catalog_source", "secondary_artist_source",
        "personal_exact_repeat", "personal_artist_expansion", "personal_album_expansion",
        "dense_qwen3_metadata", "dense_qwen3_attributes", "dense_qwen3_lyrics",
        "dense_clap_audio", "dense_siglip_image",
        "hyde_bm25_5field", "intent_tag_match", "album_qwen3_history",
    ]
    add_combo(*[n for n in top_candidates if n in available])

    union_metrics: dict[str, dict] = {}
    for combo in UNION_COMBOS:
        key = "+".join(combo)
        union_metrics[key] = compute_union_metrics(cand_dict, sizes_dict, combo, gold_idxs, args.top_k)
        print(f"  UNION {key}: r={union_metrics[key]['recall_at_union']:.4f}, size={union_metrics[key]['mean_union_size']:.1f}")

    (RESULTS_DIR / "unions.json").write_text(json.dumps(union_metrics, indent=2))

    rows = []
    rows.append("| retriever | nonempty | mean_size | median | p90 | r@20 | r@50 | r@100 | r@200 | t(s) |")
    rows.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name in all_names:
        if name not in metrics_dict:
            continue
        m = metrics_dict[name]
        rows.append(
            f"| {name} | {m['nonempty_rate']:.3f} | {m['mean_cand_size']:.1f} | "
            f"{m['median_cand_size']:.0f} | {m['p90_cand_size']:.0f} | "
            f"{m['recall@20']:.4f} | {m['recall@50']:.4f} | "
            f"{m['recall@100']:.4f} | {m['recall@200']:.4f} | "
            f"{m['elapsed_sec']:.1f} |"
        )
    rows.append("")
    rows.append("| union | mean_size | median | p90 | recall@union |")
    rows.append("| --- | ---: | ---: | ---: | ---: |")
    for key, m in union_metrics.items():
        rows.append(
            f"| {key} | {m['mean_union_size']:.1f} | "
            f"{m['median_union_size']:.0f} | {m['p90_union_size']:.0f} | "
            f"{m['recall_at_union']:.4f} |"
        )

    (RESULTS_DIR / "summary.md").write_text("\n".join(rows))
    print("\n".join(rows))

    best_union_key = max(union_metrics, key=lambda k: union_metrics[k]["recall_at_union"]) if union_metrics else ""
    best_union_recall = union_metrics[best_union_key]["recall_at_union"] if best_union_key else 0.0
    summary = {
        "best_single_recall@200": max((m["recall@200"] for m in metrics_dict.values()), default=0.0),
        "best_single_recall@20": max((m["recall@20"] for m in metrics_dict.values()), default=0.0),
        "best_union_key": best_union_key,
        "best_union_recall": best_union_recall,
        "n_retrievers": len(all_names),
        "n_examples": n_ex,
    }
    (RESULTS_DIR / "scores.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
