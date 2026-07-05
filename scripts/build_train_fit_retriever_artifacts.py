#!/usr/bin/env python3
"""Build train-safe artifacts for label-derived retrievers.

This complements ``scripts/build_basic_retrievers.py``.  The basic builder is
for devset/inference-style artifacts.  This script emits artifacts that are
safe to use as downstream reranker training inputs:

- ``cv3_oof/public_labeled`` for fold-held-out train/statistical sources.
- ``strict_date_censored_all_rows/public_labeled`` for same-user memory.
- matching ``full_public`` or ``full_train`` blind inference artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from tqdm import tqdm

from recsys2026.artifacts import (
    component_output_dir,
    component_results_dir,
    encode_keys,
    file_ref,
    json_dump,
    save_candidate_artifact,
    utc_now,
)
from recsys2026.data import load
from recsys2026.paths import REPO_ROOT
from recsys2026.splits import read_jsonl


PUBLIC_SOURCES = ("train", "devset")
DATASET_BY_SOURCE = {"train": "train", "devset": "test"}
MAX_TURNS = 8

FOLD_SOURCES = {
    "cooc_track",
    "cooc_artist",
    "cooc_album",
    "cooc_artist_name",
    "transition_track_last",
    "transition_track_bigram_last2",
    "transition_album_last",
    "transition_artist_id_last",
    "transition_artist_name_last",
    "train_play_count_unique_users",
    "user_neighbor",
    "train_neighbor",
}
STRICT_DATE_SOURCES = {
    "personal_exact_repeat",
    "personal_artist_expansion",
    "personal_album_expansion",
}
ALL_SOURCES = tuple(sorted(FOLD_SOURCES | STRICT_DATE_SOURCES))


@dataclass(frozen=True)
class PublicExample:
    source_split: Literal["train", "devset", "blind_a", "blind_b"]
    session_id: str
    user_id: str
    session_date: str
    turn_number: int
    fold: int
    chat_history: tuple[dict[str, Any], ...]
    user_query: str
    user_thought: str
    conversation_goal: dict[str, Any]
    gold_track_id: str
    gold_idx: int


@dataclass(frozen=True)
class SessionMusic:
    source_split: Literal["train", "devset"]
    session_id: str
    user_id: str
    session_date: str
    fold: int
    track_ids: tuple[str, ...]
    track_idxs: tuple[int, ...]


@dataclass
class Cooc:
    track_track: dict[int, tuple[np.ndarray, np.ndarray]]
    artist_artist: dict[str, Counter]
    album_album: dict[str, Counter]
    transition_track: dict[int, tuple[np.ndarray, np.ndarray]]
    transition_track_bigram: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]]
    transition_album: dict[str, Counter]
    transition_artist_id: dict[str, Counter]
    artist_name_artist_name: dict[str, Counter]
    transition_artist_name: dict[str, Counter]


def load_zoo_module() -> Any:
    from recsys2026 import zoo

    return zoo


def read_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def source_entry_from_config(cfg: dict[str, Any], source: str) -> dict[str, Any]:
    """Return the source entry whose name or component matches the builder source."""
    for raw in cfg.get("sources") or []:
        if isinstance(raw, str):
            if raw == source:
                return {"name": raw}
            continue
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "")
        component = str(raw.get("component") or raw.get("retriever") or name)
        if source in {name, component}:
            return dict(raw)
    return {}


def source_policy_from_config(cfg: dict[str, Any], source: str) -> dict[str, Any]:
    defaults = dict(cfg.get("source_policy_defaults") or {})
    source_metadata = cfg.get("source_metadata") or {}
    entry = source_entry_from_config(cfg, source)
    metadata = dict(source_metadata.get(source) or {})
    entry_name = str(entry.get("name") or "")
    if entry_name and entry_name != source:
        metadata = {**dict(source_metadata.get(entry_name) or {}), **metadata}
    entry_policy = dict(entry.get("source_policy") or {})
    policy = {**defaults, **metadata, **entry_policy}
    policy.setdefault("requires_labeled_fit", source in (FOLD_SOURCES | STRICT_DATE_SOURCES))
    policy.setdefault(
        "train_row_policy",
        "requires_strict_date_censor" if source in STRICT_DATE_SOURCES else "requires_oof",
    )
    policy.setdefault(
        "fold_split_required_for_reranker_train",
        source in FOLD_SOURCES,
    )
    policy.setdefault(
        "preferred_train_row_artifact_mode",
        "strict_date_censored_all_rows" if source in STRICT_DATE_SOURCES else "cv3_oof",
    )
    policy.setdefault(
        "preferred_inference_artifact_mode",
        "full_train" if source in STRICT_DATE_SOURCES else "full_public",
    )
    return policy


def load_fold_map(split_dir: Path) -> dict[tuple[str, str], int]:
    return {
        (str(row["source_split"]), str(row["session_id"])): int(row["fold"])
        for row in read_jsonl(split_dir / "sessions.jsonl")
    }


def split_name(split_dir: Path) -> str:
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists():
        try:
            return str(json.loads(manifest_path.read_text()).get("name") or split_dir.name)
        except Exception:  # noqa: BLE001
            return split_dir.name
    return split_dir.name


def build_public_examples(zoo: Any, split_dir: Path, track_index: Any) -> list[PublicExample]:
    fold_map = load_fold_map(split_dir)
    examples: list[PublicExample] = []
    for source_split in PUBLIC_SOURCES:
        ds = load("dataset", split=DATASET_BY_SOURCE[source_split])
        for item in ds:
            conversations = list(item["conversations"])
            fold = fold_map[(source_split, item["session_id"])]
            for target_turn in range(1, MAX_TURNS + 1):
                current = [c for c in conversations if int(c["turn_number"]) == target_turn]
                user_turn = next((c for c in current if c["role"] == "user"), None)
                music_turn = next((c for c in current if c["role"] == "music"), None)
                if user_turn is None or music_turn is None:
                    continue
                gold_tid = str(music_turn.get("content") or "")
                gold_idx = track_index.id_to_idx.get(gold_tid, -1)
                examples.append(
                    PublicExample(
                        source_split=source_split,  # type: ignore[arg-type]
                        session_id=str(item["session_id"]),
                        user_id=str(item["user_id"]),
                        session_date=str(item.get("session_date") or ""),
                        turn_number=target_turn,
                        fold=fold,
                        chat_history=tuple(c for c in conversations if int(c["turn_number"]) < target_turn),
                        user_query=str(user_turn.get("content") or ""),
                        user_thought=str(user_turn.get("thought") or "").strip(),
                        conversation_goal=dict(item.get("conversation_goal") or {}),
                        gold_track_id=gold_tid,
                        gold_idx=int(gold_idx),
                    )
                )
    return examples


def build_blind_examples(zoo: Any, target: Literal["blind_a", "blind_b"], track_index: Any) -> list[PublicExample]:
    examples: list[PublicExample] = []
    for item in load(target, split="test"):
        conversations = list(item["conversations"])
        current = conversations[-1]
        target_turn = int(current["turn_number"])
        examples.append(
            PublicExample(
                source_split=target,
                session_id=str(item["session_id"]),
                user_id=str(item["user_id"]),
                session_date=str(item.get("session_date") or ""),
                turn_number=target_turn,
                fold=-1,
                chat_history=tuple(c for c in conversations if int(c["turn_number"]) < target_turn),
                user_query=str(current.get("content") or ""),
                user_thought=str(current.get("thought") or "").strip(),
                conversation_goal=dict(item.get("conversation_goal") or {}),
                gold_track_id="",
                gold_idx=-1,
            )
        )
    return examples


def build_public_sessions(split_dir: Path, track_index: Any) -> list[SessionMusic]:
    fold_map = load_fold_map(split_dir)
    sessions: list[SessionMusic] = []
    for source_split in PUBLIC_SOURCES:
        ds = load("dataset", split=DATASET_BY_SOURCE[source_split])
        for item in ds:
            tids = tuple(
                str(c["content"])
                for c in item["conversations"]
                if c.get("role") == "music" and c.get("content")
            )
            idxs = tuple(int(track_index.id_to_idx[tid]) for tid in tids if tid in track_index.id_to_idx)
            sessions.append(
                SessionMusic(
                    source_split=source_split,  # type: ignore[arg-type]
                    session_id=str(item["session_id"]),
                    user_id=str(item["user_id"]),
                    session_date=str(item.get("session_date") or ""),
                    fold=fold_map[(source_split, item["session_id"])],
                    track_ids=tids,
                    track_idxs=idxs,
                )
            )
    return sessions


def played_set(ex: PublicExample, track_index: Any) -> set[int]:
    played: set[int] = set()
    for c in ex.chat_history:
        if c.get("role") != "music":
            continue
        idx = track_index.id_to_idx.get(c.get("content"))
        if idx is not None:
            played.add(int(idx))
    return played


def history_state(zoo: Any, ex: PublicExample, track_index: Any):
    return zoo.history_state(
        zoo.TurnExample(
            session_id=ex.session_id,
            user_id=ex.user_id,
            session_date=ex.session_date,
            turn_number=ex.turn_number,
            chat_history=list(ex.chat_history),
            user_query=ex.user_query,
            gold_track_id=ex.gold_track_id or None,
            user_thought=ex.user_thought,
            conversation_goal=ex.conversation_goal,
        ),
        track_index,
    )


def select_from_score(
    score: np.ndarray | None,
    played: set[int],
    top_k: int,
    *,
    positive_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if score is None:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    score = np.asarray(score, dtype=np.float32)
    idxs = np.flatnonzero(score > 0) if positive_only else np.arange(len(score), dtype=np.int32)
    if len(idxs) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    if played:
        played_arr = np.fromiter(played, dtype=np.int32)
        idxs = idxs[~np.isin(idxs, played_arr)]
    if len(idxs) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    vals = score[idxs]
    order = np.argsort(-vals, kind="stable")
    idxs = idxs[order][:top_k].astype(np.int32, copy=False)
    vals = vals[order][:top_k].astype(np.float32, copy=False)
    return idxs, vals


def select_from_score_with_extras(
    score: np.ndarray | None,
    played: set[int],
    top_k: int,
    *,
    positive_only: bool = False,
    extras: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    idxs, vals = select_from_score(score, played, top_k, positive_only=positive_only)
    if not extras:
        return idxs, vals, {}
    return idxs, vals, {
        name: np.asarray(arr, dtype=np.float32)[idxs].astype(np.float32, copy=False)
        for name, arr in extras.items()
    }


def pad_scored(rows: list[tuple[np.ndarray, np.ndarray]], top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cand = np.full((len(rows), top_k), -1, dtype=np.int32)
    sizes = np.zeros(len(rows), dtype=np.int32)
    scores = np.full((len(rows), top_k), np.nan, dtype=np.float32)
    for i, (idxs, vals) in enumerate(rows):
        k = min(len(idxs), top_k)
        if k:
            cand[i, :k] = idxs[:k]
            scores[i, :k] = vals[:k]
            sizes[i] = k
    return cand, sizes, scores


def pad_scored_with_extras(
    rows: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    cand = np.full((len(rows), top_k), -1, dtype=np.int32)
    sizes = np.zeros(len(rows), dtype=np.int32)
    scores = np.full((len(rows), top_k), np.nan, dtype=np.float32)
    extra_keys = sorted({key for _, _, extras in rows for key in extras})
    extra_out = {key: np.full((len(rows), top_k), np.nan, dtype=np.float32) for key in extra_keys}
    for i, (idxs, vals, extras) in enumerate(rows):
        k = min(len(idxs), top_k)
        if not k:
            continue
        cand[i, :k] = idxs[:k]
        scores[i, :k] = vals[:k]
        sizes[i] = k
        for key, arr in extras.items():
            extra_out[key][i, :k] = arr[:k]
    return cand, sizes, scores, extra_out


def build_cooc_from_sessions(zoo: Any, track_index: Any, sessions: list[SessionMusic]) -> Cooc:
    track_counts: dict[int, Counter] = defaultdict(Counter)
    artist_counts: dict[str, Counter] = defaultdict(Counter)
    album_counts: dict[str, Counter] = defaultdict(Counter)
    transition_track_counts: dict[int, Counter] = defaultdict(Counter)
    transition_track_bigram_counts: dict[tuple[int, int], Counter] = defaultdict(Counter)
    transition_album_counts: dict[str, Counter] = defaultdict(Counter)
    transition_artist_id_counts: dict[str, Counter] = defaultdict(Counter)
    artist_name_counts: dict[str, Counter] = defaultdict(Counter)
    transition_artist_name_counts: dict[str, Counter] = defaultdict(Counter)

    for session in tqdm(sessions, desc="build cooc"):
        idxs = list(dict.fromkeys(session.track_idxs))
        artists: set[str] = set()
        albums: set[str] = set()
        artist_names: set[str] = set()
        artist_ids_by_turn: list[set[str]] = []
        artist_names_by_turn: list[set[str]] = []
        album_ids_by_turn: list[set[str]] = []
        for tid in session.track_ids:
            md = track_index.meta_by_id.get(tid, {})
            turn_artists: set[str] = set()
            for aid in zoo.as_list(md.get("artist_id")):
                if aid:
                    aid_s = str(aid)
                    artists.add(aid_s)
                    turn_artists.add(aid_s)
            turn_albums: set[str] = set()
            for album_id in zoo.as_list(md.get("album_id")):
                if album_id:
                    album_s = str(album_id)
                    albums.add(album_s)
                    turn_albums.add(album_s)
            names = {
                zoo.norm_name(str(name))
                for name in zoo.as_list(md.get("artist_name"))
                if str(name or "").strip()
            }
            names.discard("")
            artist_names.update(names)
            artist_ids_by_turn.append(turn_artists)
            artist_names_by_turn.append(names)
            album_ids_by_turn.append(turn_albums)

        for i, a in enumerate(idxs):
            ca = track_counts[a]
            for b in idxs[i + 1 :]:
                ca[b] += 1
                track_counts[b][a] += 1

        a_list = list(artists)
        for i, a in enumerate(a_list):
            ca = artist_counts[a]
            for b in a_list[i + 1 :]:
                ca[b] += 1
                artist_counts[b][a] += 1

        album_list = list(albums)
        for i, a in enumerate(album_list):
            ca = album_counts[a]
            for b in album_list[i + 1 :]:
                ca[b] += 1
                album_counts[b][a] += 1

        for prev_idx, next_idx in zip(session.track_idxs, session.track_idxs[1:]):
            transition_track_counts[int(prev_idx)][int(next_idx)] += 1

        for prev2_idx, prev1_idx, next_idx in zip(session.track_idxs, session.track_idxs[1:], session.track_idxs[2:]):
            transition_track_bigram_counts[(int(prev2_idx), int(prev1_idx))][int(next_idx)] += 1

        for prev_albums, next_albums in zip(album_ids_by_turn, album_ids_by_turn[1:]):
            for a in prev_albums:
                ca = transition_album_counts[a]
                for b in next_albums:
                    if a != b:
                        ca[b] += 1

        for prev_artists, next_artists in zip(artist_ids_by_turn, artist_ids_by_turn[1:]):
            for a in prev_artists:
                ca = transition_artist_id_counts[a]
                for b in next_artists:
                    if a != b:
                        ca[b] += 1

        name_list = list(artist_names)
        for i, a in enumerate(name_list):
            ca = artist_name_counts[a]
            for b in name_list[i + 1 :]:
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
        items = c.most_common()
        if not items:
            continue
        nb = np.fromiter((i for i, _ in items), dtype=np.int32, count=len(items))
        cn = np.fromiter((v for _, v in items), dtype=np.float32, count=len(items))
        track_track[int(k)] = (nb, cn)
    transition_track: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for k, c in transition_track_counts.items():
        items = c.most_common()
        if not items:
            continue
        nb = np.fromiter((i for i, _ in items), dtype=np.int32, count=len(items))
        cn = np.fromiter((v for _, v in items), dtype=np.float32, count=len(items))
        transition_track[int(k)] = (nb, cn)
    transition_track_bigram: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k, c in transition_track_bigram_counts.items():
        items = c.most_common()
        if not items:
            continue
        nb = np.fromiter((i for i, _ in items), dtype=np.int32, count=len(items))
        cn = np.fromiter((v for _, v in items), dtype=np.float32, count=len(items))
        transition_track_bigram[k] = (nb, cn)
    return Cooc(
        track_track=track_track,
        artist_artist=dict(artist_counts),
        album_album=dict(album_counts),
        transition_track=transition_track,
        transition_track_bigram=transition_track_bigram,
        transition_album=dict(transition_album_counts),
        transition_artist_id=dict(transition_artist_id_counts),
        artist_name_artist_name=dict(artist_name_counts),
        transition_artist_name=dict(transition_artist_name_counts),
    )


def score_cooc_source(
    zoo: Any,
    examples: list[PublicExample],
    track_index: Any,
    cooc: Cooc,
    source: str,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rows: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []
    for ex in tqdm(examples, desc=source):
        h_arts, h_albums, _, played, history_idxs = history_state(zoo, ex, track_index)
        score = np.zeros(track_index.n_tracks, dtype=np.float32)
        transition_prob = np.zeros(track_index.n_tracks, dtype=np.float32)
        if source == "cooc_track":
            for h in history_idxs:
                nb_cn = cooc.track_track.get(int(h))
                if nb_cn is None:
                    continue
                nb, cn = nb_cn
                score[nb] += cn
        elif source == "transition_track_last":
            if history_idxs:
                nb_cn = cooc.transition_track.get(int(history_idxs[-1]))
                if nb_cn is not None:
                    nb, cn = nb_cn
                    score[nb] += cn
                    denom = float(cn.sum())
                    if denom > 0:
                        transition_prob[nb] += cn / denom
        elif source == "transition_track_bigram_last2":
            if len(history_idxs) >= 2:
                key = (int(history_idxs[-2]), int(history_idxs[-1]))
                nb_cn = cooc.transition_track_bigram.get(key)
                if nb_cn is not None:
                    nb, cn = nb_cn
                    score[nb] += cn
                    denom = float(cn.sum())
                    if denom > 0:
                        transition_prob[nb] += cn / denom
        elif source == "cooc_artist":
            artist_score: dict[str, float] = defaultdict(float)
            for aid in h_arts:
                for nb_aid, count in (cooc.artist_artist.get(aid) or {}).items():
                    artist_score[nb_aid] += float(count)
            for nb_aid, value in artist_score.items():
                for idx in track_index.artist_to_idx.get(nb_aid, []):
                    score[idx] += value
        elif source == "cooc_album":
            album_score: dict[str, float] = defaultdict(float)
            for album_id in h_albums:
                for nb_album_id, count in (cooc.album_album.get(album_id) or {}).items():
                    album_score[nb_album_id] += float(count)
            for nb_album_id, value in album_score.items():
                for idx in track_index.album_to_idx.get(nb_album_id, []):
                    score[idx] += value
        elif source == "transition_album_last":
            last_albums: set[str] = set()
            for turn in reversed(ex.chat_history):
                if turn.get("role") != "music":
                    continue
                md = track_index.meta_by_id.get(str(turn.get("content") or ""), {})
                last_albums = {
                    str(album_id)
                    for album_id in zoo.as_list(md.get("album_id"))
                    if str(album_id or "").strip()
                }
                if last_albums:
                    break
            album_score: dict[str, float] = defaultdict(float)
            for album_id in last_albums:
                for nb_album_id, count in (cooc.transition_album.get(album_id) or {}).items():
                    album_score[nb_album_id] += float(count)
            denom = float(sum(album_score.values()))
            for nb_album_id, value in album_score.items():
                for idx in track_index.album_to_idx.get(nb_album_id, []):
                    score[idx] += value
                    if denom > 0:
                        transition_prob[idx] += float(value) / denom
        elif source == "transition_artist_id_last":
            last_artists: set[str] = set()
            for turn in reversed(ex.chat_history):
                if turn.get("role") != "music":
                    continue
                md = track_index.meta_by_id.get(str(turn.get("content") or ""), {})
                last_artists = {
                    str(artist_id)
                    for artist_id in zoo.as_list(md.get("artist_id"))
                    if str(artist_id or "").strip()
                }
                if last_artists:
                    break
            artist_score: dict[str, float] = defaultdict(float)
            for artist_id in last_artists:
                for nb_artist_id, count in (cooc.transition_artist_id.get(artist_id) or {}).items():
                    artist_score[nb_artist_id] += float(count)
            denom = float(sum(artist_score.values()))
            for nb_artist_id, value in artist_score.items():
                for idx in track_index.artist_to_idx.get(nb_artist_id, []):
                    score[idx] += value
                    if denom > 0:
                        transition_prob[idx] += float(value) / denom
        elif source in {"cooc_artist_name", "transition_artist_name_last"}:
            artist_counts, _, _ = zoo._history_name_counts(
                zoo.TurnExample(
                    session_id=ex.session_id,
                    user_id=ex.user_id,
                    session_date=ex.session_date,
                    turn_number=ex.turn_number,
                    chat_history=list(ex.chat_history),
                    user_query=ex.user_query,
                    gold_track_id=ex.gold_track_id or None,
                    user_thought=ex.user_thought,
                    conversation_goal=ex.conversation_goal,
                ),
                track_index,
                last_only=(source == "transition_artist_name_last"),
            )
            table = cooc.transition_artist_name if source == "transition_artist_name_last" else cooc.artist_name_artist_name
            name_score: dict[str, float] = defaultdict(float)
            for name, weight in artist_counts.items():
                for nb_name, count in (table.get(name) or {}).items():
                    name_score[nb_name] += float(weight) * float(count)
            for nb_name, value in name_score.items():
                for idx in track_index.artist_name_to_idx.get(nb_name, []):
                    score[idx] += value
        else:
            raise ValueError(source)
        extras = {}
        if source.startswith("transition_"):
            extras["transition_probability"] = transition_prob
        rows.append(
            select_from_score_with_extras(
                score,
                played,
                top_k,
                positive_only=True,
                extras=extras,
            )
        )
    return pad_scored_with_extras(rows, top_k)


def score_train_play_count(
    examples: list[PublicExample],
    track_index: Any,
    sessions: list[SessionMusic],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_sets: list[set[str]] = [set() for _ in range(track_index.n_tracks)]
    for session in sessions:
        for idx in set(session.track_idxs):
            user_sets[int(idx)].add(session.user_id)
    score = np.asarray([len(s) for s in user_sets], dtype=np.float32)
    rows = [select_from_score(score, played_set(ex, track_index), top_k) for ex in tqdm(examples, desc="train_play_count_unique_users")]
    return pad_scored(rows, top_k)


def build_personal_memory(track_index: Any, sessions: list[SessionMusic], *, train_only: bool) -> dict[str, list[tuple[str, str, list[int]]]]:
    memory: dict[str, list[tuple[str, str, list[int]]]] = defaultdict(list)
    for session in sessions:
        if train_only and session.source_split != "train":
            continue
        if session.track_idxs:
            memory[session.user_id].append((session.session_date, session.session_id, list(session.track_idxs)))
    for values in memory.values():
        values.sort(key=lambda x: (x[0], x[1]))
    return dict(memory)


def prior_personal_counts(ex: PublicExample, memory: dict[str, list[tuple[str, str, list[int]]]]) -> Counter[int]:
    out: Counter[int] = Counter()
    if not ex.session_date:
        return out
    for date, session_id, idxs in memory.get(ex.user_id, []):
        if date >= ex.session_date:
            break
        if session_id == ex.session_id:
            continue
        out.update(idxs)
    return out


def score_personal_source(
    zoo: Any,
    examples: list[PublicExample],
    track_index: Any,
    memory: dict[str, list[tuple[str, str, list[int]]]],
    source: str,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for ex in tqdm(examples, desc=source):
        counts = prior_personal_counts(ex, memory)
        score = np.zeros(track_index.n_tracks, dtype=np.float32)
        if counts:
            if source == "personal_exact_repeat":
                for idx, cnt in counts.items():
                    score[int(idx)] += float(cnt) * 10.0
            elif source == "personal_artist_expansion":
                artist_counts: Counter[str] = Counter()
                for idx, cnt in counts.items():
                    md = track_index.meta_by_id.get(track_index.track_ids[int(idx)], {})
                    for aid in zoo.as_list(md.get("artist_id")):
                        if aid:
                            artist_counts[str(aid)] += int(cnt)
                for aid, cnt in artist_counts.items():
                    for j in track_index.artist_to_idx.get(aid, []):
                        score[j] += float(cnt)
            elif source == "personal_album_expansion":
                album_counts: Counter[str] = Counter()
                for idx, cnt in counts.items():
                    md = track_index.meta_by_id.get(track_index.track_ids[int(idx)], {})
                    for album_id in zoo.as_list(md.get("album_id")):
                        if album_id:
                            album_counts[str(album_id)] += int(cnt)
                for album_id, cnt in album_counts.items():
                    for j in track_index.album_to_idx.get(album_id, []):
                        score[j] += float(cnt)
            else:
                raise ValueError(source)
        rows.append(select_from_score(score, played_set(ex, track_index), top_k, positive_only=True))
    return pad_scored(rows, top_k)


def score_user_neighbor(
    zoo: Any,
    examples: list[PublicExample],
    track_index: Any,
    sessions: list[SessionMusic],
    top_k: int,
    *,
    n_neigh: int,
    rank_offset: float,
    device: str,
    score_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    user_vecs = zoo.load_user_vectors_normalized()
    user_tracks: dict[str, list[int]] = defaultdict(list)
    for session in sessions:
        seen: set[int] = set()
        for idx in session.track_idxs:
            if idx not in seen:
                user_tracks[session.user_id].append(int(idx))
                seen.add(int(idx))
    train_user_ids = [uid for uid in user_tracks if uid in user_vecs]
    if not train_user_ids:
        return pad_scored([(np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)) for _ in examples], top_k)
    train_mat = zoo._normalize_rows(np.stack([user_vecs[uid] for uid in train_user_ids], axis=0).astype(np.float32))
    t_mat = torch.from_numpy(train_mat).to(device)
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for ex in tqdm(examples, desc="user_neighbor"):
        q = user_vecs.get(ex.user_id)
        if q is None:
            rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue
        sim = (torch.from_numpy(q).to(device).unsqueeze(0) @ t_mat.T).squeeze(0)
        top = torch.topk(sim, k=min(n_neigh + 1, len(train_user_ids)))
        idxs = top.indices.cpu().numpy()
        vals = top.values.cpu().numpy()
        score = np.zeros(track_index.n_tracks, dtype=np.float32)
        nbr_count = 0
        for ni, value in zip(idxs, vals, strict=True):
            nb_uid = train_user_ids[int(ni)]
            if nb_uid == ex.user_id:
                continue
            neighbor_weight = float(value) if score_mode == "sim_weighted" else 1.0
            for rank, tidx in enumerate(user_tracks.get(nb_uid, [])):
                score[int(tidx)] += neighbor_weight / (rank + rank_offset)
            nbr_count += 1
            if nbr_count >= n_neigh:
                break
        rows.append(select_from_score(score, played_set(ex, track_index), top_k, positive_only=True))
    return pad_scored(rows, top_k)


def load_public_query_embeddings() -> np.ndarray:
    train_q = np.load(REPO_ROOT / "output/081_two_tower/train_q_emb__n121592.npy").astype(np.float32)
    dev_q = np.load(REPO_ROOT / "output/086_retriever_zoo_v2/encode/qwen3_query_mat__n8000.npy").astype(np.float32)
    return np.concatenate([train_q, dev_q], axis=0)


def encode_blind_query_embeddings(zoo: Any, examples: list[PublicExample], cache_path: Path) -> np.ndarray:
    if cache_path.exists():
        return np.load(cache_path).astype(np.float32)
    from recsys2026.encoders import Qwen3TextEncoder
    from recsys2026.submission import InferenceInput
    from recsys2026.retrieval import chat_to_query_text

    texts = [
        chat_to_query_text(
            InferenceInput(
                session_id=ex.session_id,
                user_id=ex.user_id,
                turn_number=ex.turn_number,
                chat_history=list(ex.chat_history),
                user_query=ex.user_query,
            ),
            mode="full",
        )
        for ex in examples
    ]
    encoder = Qwen3TextEncoder(batch_size=16)
    mat = zoo._normalize_rows(encoder.encode(texts).astype(np.float32))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mat)
    return mat


def score_train_neighbor(
    zoo: Any,
    examples: list[PublicExample],
    track_index: Any,
    query_emb: np.ndarray,
    memory_rows: np.ndarray,
    memory_gold: np.ndarray,
    top_k: int,
    *,
    n_neigh: int,
    rank_offset: float,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = zoo._normalize_rows(query_emb.astype(np.float32))
    mem_q = torch.from_numpy(q[memory_rows]).to(device)
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for start in tqdm(range(0, len(examples), batch_size), desc="train_neighbor"):
        end = min(start + batch_size, len(examples))
        chunk = torch.from_numpy(q[start:end]).to(device)
        sims = chunk @ mem_q.T
        top = torch.topk(sims, k=min(n_neigh, mem_q.shape[0]), dim=1)
        nbr_idx = top.indices.cpu().numpy()
        nbr_score = top.values.cpu().numpy()
        for j in range(end - start):
            score = np.zeros(track_index.n_tracks, dtype=np.float32)
            for rank, (local_ni, sim) in enumerate(zip(nbr_idx[j], nbr_score[j], strict=True)):
                global_row = int(memory_rows[int(local_ni)])
                gi = int(memory_gold[global_row])
                if gi >= 0:
                    score[gi] += float(sim) / (rank + rank_offset)
            rows.append(select_from_score(score, played_set(examples[start + j], track_index), top_k, positive_only=True))
    return pad_scored(rows, top_k)


def public_metrics(examples: list[PublicExample], cand: np.ndarray, sizes: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"n_examples": len(examples), "mean_size": float(sizes.mean()) if len(sizes) else 0.0}
    gold = np.asarray([ex.gold_idx for ex in examples], dtype=np.int32)
    groups = {
        "all": np.arange(len(examples), dtype=np.int32),
        "train": np.asarray([i for i, ex in enumerate(examples) if ex.source_split == "train"], dtype=np.int32),
        "devset": np.asarray([i for i, ex in enumerate(examples) if ex.source_split == "devset"], dtype=np.int32),
    }
    for name, rows in groups.items():
        if len(rows) == 0:
            continue
        prefix = "" if name == "all" else f"{name}_"
        out[f"{prefix}n_examples"] = int(len(rows))
        out[f"{prefix}mean_size"] = float(sizes[rows].mean())
        for k in (20, 50, 100, 200, 500):
            kk = min(k, cand.shape[1])
            hit = (cand[rows, :kk] == gold[rows, None]).any(axis=1)
            out[f"{prefix}recall@{k}"] = float(hit.mean())
        hit_all = np.zeros(len(rows), dtype=bool)
        for j, row_i in enumerate(rows):
            hit_all[j] = bool((cand[row_i, : int(sizes[row_i])] == gold[row_i]).any())
        out[f"{prefix}recall@all"] = float(hit_all.mean())
    return out


def write_public_artifact(
    out_dir: Path,
    examples: list[PublicExample],
    cand: np.ndarray,
    sizes: np.ndarray,
    scores: np.ndarray,
    extra_scores: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = np.tile(np.arange(1, cand.shape[1] + 1, dtype=np.int32), (cand.shape[0], 1))
    for i, size in enumerate(sizes):
        rank[i, int(size) :] = -1
    arrays: dict[str, np.ndarray] = {
        "track_idx": cand.astype(np.int32, copy=False),
        "sizes": sizes.astype(np.int32, copy=False),
        "keys": encode_keys([(f"{ex.source_split}:{ex.session_id}", ex.turn_number) for ex in examples]),
        "source_split": np.asarray([ex.source_split.encode("utf-8") for ex in examples], dtype="S8"),
        "folds": np.asarray([ex.fold for ex in examples], dtype=np.int16),
        "rank": rank,
        "score__primary": scores.astype(np.float32, copy=False),
    }
    for name, arr in extra_scores.items():
        key = name if name.startswith("score__") else f"score__{name}"
        arrays[key] = arr.astype(np.float32, copy=False)
    np.savez_compressed(out_dir / "candidates.npz", **arrays)
    with (out_dir / "turns.jsonl").open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            f.write(
                json.dumps(
                    {
                        "row_id": i,
                        "source_split": ex.source_split,
                        "session_id": ex.session_id,
                        "user_id": ex.user_id,
                        "session_date": ex.session_date,
                        "turn_number": ex.turn_number,
                        "fold": int(ex.fold),
                        "gold_track_id": ex.gold_track_id,
                        "gold_track_idx": int(ex.gold_idx),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    json_dump(out_dir / "manifest.json", manifest)


def write_blind_artifact(
    out_dir: Path,
    target: Literal["blind_a", "blind_b"],
    cand: np.ndarray,
    sizes: np.ndarray,
    scores: np.ndarray,
    extra_scores: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> None:
    rank = np.tile(np.arange(1, cand.shape[1] + 1, dtype=np.int32), (cand.shape[0], 1))
    for i, size in enumerate(sizes):
        rank[i, int(size) :] = -1
    score_arrays = {"primary": scores, **extra_scores}
    save_candidate_artifact(
        out_dir,
        cand,
        sizes,
        target=target,
        manifest=manifest,
        rank=rank,
        score_arrays=score_arrays,
        compress=True,
    )


def base_manifest(args: argparse.Namespace, source: str, cfg: dict[str, Any], policy: dict[str, Any], elapsed: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "candidates",
        "stage": "retriever",
        "name": source,
        "config": args.config,
        "created_at": utc_now(),
        "producer": {
            "command": ["uv", "run", "python", "scripts/build_train_fit_retriever_artifacts.py", *sys.argv[1:]],
            "cwd": ".",
        },
        "source_code": {
            "script": file_ref(REPO_ROOT / "scripts/build_train_fit_retriever_artifacts.py"),
            "zoo": file_ref(REPO_ROOT / "src/recsys2026/zoo.py"),
            "config": file_ref(REPO_ROOT / args.config_file),
        },
        "params": {
            "config": args.config,
            "top_k": args.top_k,
            "n_neigh": args.n_neigh,
            "rank_offset": args.rank_offset,
            "batch_size": args.batch_size,
            "device": args.device,
        },
        "source_policy": policy,
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_blind_for_fit": False,
            "uses_target_future_turns": False,
            "same_user_memory_date_censored": source in STRICT_DATE_SOURCES,
            "popularity_tiebreaker": False,
        },
        "candidate_universe": "all_tracks",
        "retention": "top_k",
        "score_fields": ["score__primary"],
        "elapsed_sec": elapsed,
    }


def build_source_for_examples(
    source: str,
    zoo: Any,
    track_index: Any,
    examples: list[PublicExample],
    sessions: list[SessionMusic],
    args: argparse.Namespace,
    *,
    fold: int | None,
    public_query_emb: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if source in {
        "cooc_track",
        "cooc_artist",
        "cooc_album",
        "cooc_artist_name",
        "transition_track_last",
        "transition_track_bigram_last2",
        "transition_album_last",
        "transition_artist_id_last",
        "transition_artist_name_last",
    }:
        fit_sessions = [s for s in sessions if fold is None or s.fold != fold]
        cooc = build_cooc_from_sessions(zoo, track_index, fit_sessions)
        return score_cooc_source(zoo, examples, track_index, cooc, source, args.top_k)
    if source == "train_play_count_unique_users":
        fit_sessions = [s for s in sessions if fold is None or s.fold != fold]
        cand, sizes, scores = score_train_play_count(examples, track_index, fit_sessions, args.top_k)
        return cand, sizes, scores, {}
    if source == "user_neighbor":
        fit_sessions = [s for s in sessions if fold is None or s.fold != fold]
        cand, sizes, scores = score_user_neighbor(
            zoo,
            examples,
            track_index,
            fit_sessions,
            args.top_k,
            n_neigh=args.n_neigh,
            rank_offset=args.rank_offset,
            device=args.device,
            score_mode=args.user_neighbor_score_mode,
        )
        return cand, sizes, scores, {}
    if source == "train_neighbor":
        if public_query_emb is None:
            raise ValueError("train_neighbor requires public_query_emb")
        all_gold = np.asarray([ex.gold_idx for ex in examples], dtype=np.int32)
        all_rows = np.arange(len(examples), dtype=np.int32)
        memory_rows = all_rows if fold is None else all_rows[np.asarray([ex.fold for ex in examples]) != fold]
        cand, sizes, scores = score_train_neighbor(
            zoo,
            examples,
            track_index,
            public_query_emb,
            memory_rows,
            all_gold,
            args.top_k,
            n_neigh=args.n_neigh,
            rank_offset=args.rank_offset,
            device=args.device,
            batch_size=args.batch_size,
        )
        return cand, sizes, scores, {}
    if source in STRICT_DATE_SOURCES:
        memory = build_personal_memory(track_index, sessions, train_only=True)
        cand, sizes, scores = score_personal_source(zoo, examples, track_index, memory, source, args.top_k)
        return cand, sizes, scores, {}
    raise ValueError(source)


def run_public(zoo: Any, track_index: Any, public_examples: list[PublicExample], sessions: list[SessionMusic], source: str, args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    policy = source_policy_from_config(cfg, source)
    fit_mode = (
        "strict_date_censored_all_rows"
        if source in STRICT_DATE_SOURCES
        else str(args.artifact_mode or "cv3_oof")
    )
    out_dir = component_output_dir("retriever", source, args.config, fit_mode=fit_mode, target="public_labeled")
    if (out_dir / "candidates.npz").exists() and not args.force:
        print(f"[skip] {out_dir}")
        return

    t0 = time.time()
    if source in STRICT_DATE_SOURCES:
        cand, sizes, scores, extra_scores = build_source_for_examples(
            source, zoo, track_index, public_examples, sessions, args, fold=None, public_query_emb=None
        )
    else:
        width = args.top_k
        cand = np.full((len(public_examples), width), -1, dtype=np.int32)
        sizes = np.zeros(len(public_examples), dtype=np.int32)
        scores = np.full((len(public_examples), width), np.nan, dtype=np.float32)
        extra_scores: dict[str, np.ndarray] = {}
        public_query_emb = load_public_query_embeddings() if source == "train_neighbor" else None
        folds = np.asarray([ex.fold for ex in public_examples], dtype=np.int16)
        for fold in sorted(int(x) for x in np.unique(folds)):
            valid_rows = np.flatnonzero(folds == fold)
            fold_examples = [public_examples[int(i)] for i in valid_rows]
            print(f"{source}: fold {fold}, rows={len(valid_rows)}")
            if source == "train_neighbor":
                sub_query_emb = public_query_emb[valid_rows]
                fit_rows = np.flatnonzero(folds != fold).astype(np.int32)
                all_gold = np.asarray([ex.gold_idx for ex in public_examples], dtype=np.int32)
                sub_cand, sub_sizes, sub_scores = score_train_neighbor(
                    zoo,
                    fold_examples,
                    track_index,
                    np.concatenate([sub_query_emb, public_query_emb], axis=0),
                    np.arange(len(fold_examples), len(fold_examples) + len(public_examples), dtype=np.int32)[fit_rows],
                    np.concatenate([np.full(len(fold_examples), -1, dtype=np.int32), all_gold]),
                    args.top_k,
                    n_neigh=args.n_neigh,
                    rank_offset=args.rank_offset,
                    device=args.device,
                    batch_size=args.batch_size,
                )
                sub_extra_scores = {}
            else:
                sub_cand, sub_sizes, sub_scores, sub_extra_scores = build_source_for_examples(
                    source, zoo, track_index, fold_examples, sessions, args, fold=fold, public_query_emb=None
                )
            cand[valid_rows] = sub_cand
            sizes[valid_rows] = sub_sizes
            scores[valid_rows] = sub_scores
            for key, arr in sub_extra_scores.items():
                if key not in extra_scores:
                    extra_scores[key] = np.full((len(public_examples), width), np.nan, dtype=np.float32)
                extra_scores[key][valid_rows] = arr

    elapsed = time.time() - t0
    manifest = base_manifest(args, source, cfg, policy, elapsed)
    manifest.update({"artifact_mode": fit_mode, "target": "public_labeled"})
    manifest["score_fields"] = ["score__primary"] + [f"score__{key}" for key in sorted(extra_scores)]
    manifest["fit_scope"] = {
        "fit_mode": fit_mode,
        "fit_splits": ["train"] if source in STRICT_DATE_SOURCES else ["public_labeled"],
        "requires_labeled_fit": True,
        "fit_sources": list(policy.get("fit_sources") or []),
        "train_row_policy": (
            "strict_target_date_censored"
            if source in STRICT_DATE_SOURCES
            else f"out_of_fold_by_{split_name(args.split_dir)}"
        ),
        "fold_split_required_for_reranker_train": source in FOLD_SOURCES,
        "uses_devset_for_fit": source in FOLD_SOURCES,
        "uses_blind_for_fit": False,
        "target_row_excluded_from_fit": True,
    }
    write_public_artifact(out_dir, public_examples, cand, sizes, scores, extra_scores, manifest)
    metrics = public_metrics(public_examples, cand, sizes)
    metrics.update({"name": source, "config": args.config, "artifact_mode": fit_mode, "target": "public_labeled", "artifact": str(out_dir.relative_to(REPO_ROOT))})
    json_dump(component_results_dir("retriever", source, args.config, fit_mode=fit_mode, target="public_labeled") / "scores.json", metrics)
    print(json.dumps(metrics, indent=2))


def run_blind(zoo: Any, track_index: Any, public_examples: list[PublicExample], blind_examples: list[PublicExample], sessions: list[SessionMusic], source: str, args: argparse.Namespace, cfg: dict[str, Any]) -> None:
    policy = source_policy_from_config(cfg, source)
    fit_mode = "full_train" if source in STRICT_DATE_SOURCES else "full_public"
    target = args.blind_target
    out_dir = component_output_dir("retriever", source, args.config, fit_mode=fit_mode, target=target)
    if (out_dir / "candidates.npz").exists() and not args.force:
        print(f"[skip] {out_dir}")
        return
    t0 = time.time()
    if source == "train_neighbor":
        public_q = load_public_query_embeddings()
        blind_q = encode_blind_query_embeddings(zoo, blind_examples, REPO_ROOT / "artifacts/runs/retriever/train_neighbor" / args.config / "encode" / f"qwen3_query_mat__{target}.npy")
        query_emb = np.concatenate([blind_q, public_q], axis=0)
        memory_rows = np.arange(len(blind_examples), len(blind_examples) + len(public_examples), dtype=np.int32)
        memory_gold = np.concatenate([
            np.full(len(blind_examples), -1, dtype=np.int32),
            np.asarray([ex.gold_idx for ex in public_examples], dtype=np.int32),
        ])
        cand, sizes, scores = score_train_neighbor(
            zoo,
            blind_examples,
            track_index,
            query_emb,
            memory_rows,
            memory_gold,
            args.top_k,
            n_neigh=args.n_neigh,
            rank_offset=args.rank_offset,
            device=args.device,
            batch_size=args.batch_size,
        )
        extra_scores = {}
    else:
        cand, sizes, scores, extra_scores = build_source_for_examples(
            source, zoo, track_index, blind_examples, sessions, args, fold=None, public_query_emb=None
        )
    elapsed = time.time() - t0
    manifest = base_manifest(args, source, cfg, policy, elapsed)
    manifest.update({"artifact_mode": fit_mode, "target": target})
    manifest["score_fields"] = ["score__primary"] + [f"score__{key}" for key in sorted(extra_scores)]
    manifest["fit_scope"] = {
        "fit_mode": fit_mode,
        "fit_splits": ["train"] if source in STRICT_DATE_SOURCES else ["public_labeled"],
        "requires_labeled_fit": True,
        "fit_sources": list(policy.get("fit_sources") or []),
        "train_row_policy": "strict_target_date_censored" if source in STRICT_DATE_SOURCES else "inference_only",
        "fold_split_required_for_reranker_train": False,
        "uses_devset_for_fit": source in FOLD_SOURCES,
        "uses_blind_for_fit": False,
        "target_row_excluded_from_fit": None,
    }
    write_blind_artifact(out_dir, target, cand, sizes, scores, extra_scores, manifest)
    print(f"wrote {out_dir} mean_size={sizes.mean():.1f} elapsed={elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="oof3_top500")
    parser.add_argument("--config-file", type=Path, default=Path("retriever/union/configs/union_v1.yaml"))
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "artifacts/cache/splits/cv3")
    parser.add_argument("--artifact-mode", default=None, help="Public OOF artifact mode, e.g. cv3_oof or cv5_oof.")
    parser.add_argument("--source", action="append", choices=ALL_SOURCES, default=[])
    parser.add_argument("--mode", choices=("public", "blind", "both"), default="both")
    parser.add_argument("--blind-target", choices=("blind_a", "blind_b"), default="blind_a")
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--n-neigh", type=int, default=500)
    parser.add_argument("--rank-offset", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--user-neighbor-score-mode", choices=("legacy_rank", "sim_weighted"), default="sim_weighted")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    args.config_file = args.config_file if args.config_file.is_absolute() else REPO_ROOT / args.config_file
    args.split_dir = args.split_dir if args.split_dir.is_absolute() else REPO_ROOT / args.split_dir
    cfg = read_config(args.config_file)
    sources = args.source or [
        "cooc_track",
        "cooc_artist",
        "cooc_album",
        "cooc_artist_name",
        "transition_track_last",
        "transition_track_bigram_last2",
        "transition_album_last",
        "transition_artist_id_last",
        "transition_artist_name_last",
        "train_play_count_unique_users",
        "personal_exact_repeat",
        "personal_artist_expansion",
        "personal_album_expansion",
    ]

    zoo = load_zoo_module()
    print("building track index")
    track_index = zoo.build_track_index((), ())
    print("building public examples/sessions")
    public_examples = build_public_examples(zoo, args.split_dir, track_index)
    sessions = build_public_sessions(args.split_dir, track_index)
    print(f"public_examples={len(public_examples)} sessions={len(sessions)} sources={sources}")
    blind_examples: list[PublicExample] = []
    if args.mode in {"blind", "both"}:
        blind_examples = build_blind_examples(zoo, args.blind_target, track_index)
        print(f"{args.blind_target} examples={len(blind_examples)}")

    for source in sources:
        print(f"\n=== {source} ===")
        if args.mode in {"public", "both"}:
            run_public(zoo, track_index, public_examples, sessions, source, args, cfg)
        if args.mode in {"blind", "both"}:
            run_blind(zoo, track_index, public_examples, blind_examples, sessions, source, args, cfg)


if __name__ == "__main__":
    main()
