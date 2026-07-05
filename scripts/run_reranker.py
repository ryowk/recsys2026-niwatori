#!/usr/bin/env python3
"""Fit 098-rich LGBM on an independent retriever-union candidate artifact.

This experiment removes the old 098 candidate-generation boost mixture from the
model under test:

- candidates come from an existing retriever/union artifact
- 098 rich numeric/categorical features are computed on those candidates
- retriever source present/rank/score/meta features are appended from
  source_features.npz when available

The goal is to test "wide independent retriever union + feature-rich reranker"
without hand-tuned artist/album boost values in candidate generation.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal
from datetime import date, datetime

import lightgbm as lgb
import numpy as np
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import normalize as sparse_normalize

try:
    import xgboost as xgb
except Exception:  # noqa: BLE001
    xgb = None

try:
    import catboost as cb
except Exception:  # noqa: BLE001
    cb = None

from recsys2026.artifacts import encode_keys, json_dump, utc_now
from recsys2026.artifacts import file_ref, save_ranked_artifact
from recsys2026.paths import CACHE_DIR, OUTPUT_DIR, REPO_ROOT, RESULTS_DIR


from recsys2026 import fast_features as fast098
from recsys2026 import reranker_protocol as proto

_SESSION_DATE_CACHE: dict[str, date | None] | None = None
_TAG_CHAIN_CONTEXT_CACHE: dict[int, dict[str, Any]] = {}

TAG_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "i",
    "in",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "the",
    "to",
    "with",
    "you",
    "your",
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def parse_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10]).date()
    except ValueError:
        return None


def session_date_lookup(legacy: Any) -> dict[str, date | None]:
    global _SESSION_DATE_CACHE
    if _SESSION_DATE_CACHE is not None:
        return _SESSION_DATE_CACHE
    out: dict[str, date | None] = {}
    for split in ("train", "test"):
        try:
            ds = legacy.load("dataset", split=split)
        except Exception:
            continue
        for item in ds:
            out[str(item.get("session_id"))] = parse_date(item.get("session_date"))
    for target in ("blind_a", "blind_b"):
        try:
            ds = legacy.load(target, split="test")
        except Exception:
            continue
        for item in ds:
            out[str(item.get("session_id"))] = parse_date(item.get("session_date"))
    _SESSION_DATE_CACHE = out
    return out


def parse_isrc(value: object) -> tuple[str, int]:
    values = value if isinstance(value, list) else [value]
    for raw in values:
        text = str(raw or "").strip().upper()
        if len(text) < 7:
            continue
        yy_text = text[5:7]
        if not yy_text.isdigit():
            continue
        yy = int(yy_text)
        year = 2000 + yy if yy < 30 else 1900 + yy
        return text[:2], year
    return "", 0


def stable_small_bucket(value: str, modulo: int = 64) -> int:
    if not value:
        return 0
    return (sum((i + 1) * ord(ch) for i, ch in enumerate(value)) % modulo) + 1


def day_distance_in_year(d: date | None, month: int, day: int) -> float:
    if d is None:
        return 0.0
    target = date(d.year, month, day)
    raw = abs((d - target).days)
    return float(min(raw, 366 - raw))


def extra_metadata_feature_context(track_index: Any) -> dict[str, np.ndarray]:
    n = len(track_index.track_ids)
    isrc_year = np.zeros(n, dtype=np.float32)
    isrc_country_bucket = np.zeros(n, dtype=np.float32)
    isrc_missing = np.ones(n, dtype=np.float32)
    release_minus_isrc_abs = np.zeros(n, dtype=np.float32)
    duration_bucket = np.zeros(n, dtype=np.float32)
    album_mean_duration = np.zeros(n, dtype=np.float32)
    christmas_tag = np.zeros(n, dtype=np.float32)
    halloween_tag = np.zeros(n, dtype=np.float32)
    isrc_country: list[str] = [""] * n

    for i, tid in enumerate(track_index.track_ids):
        meta = track_index.meta_by_id.get(tid, {})
        country, year = parse_isrc(meta.get("ISRC"))
        isrc_country[i] = country
        if year:
            isrc_year[i] = float(year)
            isrc_country_bucket[i] = float(stable_small_bucket(country))
            isrc_missing[i] = 0.0
            rel_year = float(track_index.release_year[i])
            if rel_year > 0:
                release_minus_isrc_abs[i] = abs(rel_year - float(year))
        duration = float(track_index.duration[i])
        if duration <= 0:
            bucket = 0
        elif duration < 60_000:
            bucket = 1
        elif duration < 180_000:
            bucket = 2
        elif duration < 300_000:
            bucket = 3
        elif duration < 600_000:
            bucket = 4
        else:
            bucket = 5
        duration_bucket[i] = float(bucket)
        tag_text = " ".join(track_index.tag_tokens[i]).lower()
        christmas_tag[i] = float(any(tok in tag_text for tok in ("christmas", "xmas", "holiday")))
        halloween_tag[i] = float(any(tok in tag_text for tok in ("halloween", "spooky")))

    for idxs in track_index.album_to_idx.values():
        arr = np.asarray(idxs, dtype=np.int32)
        vals = track_index.duration[arr]
        valid = vals > 0
        mean = float(vals[valid].mean()) if np.any(valid) else 0.0
        album_mean_duration[arr] = mean

    return {
        "isrc_year": isrc_year,
        "isrc_country_bucket": isrc_country_bucket,
        "isrc_missing": isrc_missing,
        "release_minus_isrc_abs": release_minus_isrc_abs,
        "duration_bucket": duration_bucket,
        "album_mean_duration": album_mean_duration,
        "christmas_tag": christmas_tag,
        "halloween_tag": halloween_tag,
        "isrc_country": np.asarray(isrc_country, dtype=object),
    }


def _code_array(values: list[object]) -> np.ndarray:
    mapping: dict[str, int] = {}
    out = np.zeros(len(values), dtype=np.int32)
    for i, value in enumerate(values):
        text = str(value or "")
        if not text:
            continue
        out[i] = mapping.setdefault(text, len(mapping) + 1)
    return out


def extra_feedback_feature_context(track_index: Any) -> dict[str, np.ndarray]:
    primary_tags = [next(iter(tokens)) if tokens else "" for tokens in track_index.tag_tokens]
    return {
        "artist_code": _code_array(track_index.artist_name),
        "album_code": _code_array(track_index.album_name),
        "primary_tag_code": _code_array(primary_tags),
    }


def extra_hier_pop_feature_context(track_index: Any) -> dict[str, np.ndarray]:
    n = len(track_index.track_ids)
    pop = np.asarray(track_index.popularity, dtype=np.float32)
    artist_mean = np.zeros(n, dtype=np.float32)
    artist_max = np.zeros(n, dtype=np.float32)
    artist_count = np.zeros(n, dtype=np.float32)
    album_mean = np.zeros(n, dtype=np.float32)
    album_count = np.zeros(n, dtype=np.float32)

    for idxs in track_index.artist_to_idx.values():
        arr = np.asarray(idxs, dtype=np.int32)
        vals = pop[arr]
        artist_mean[arr] = float(vals.mean()) if len(vals) else 0.0
        artist_max[arr] = float(vals.max()) if len(vals) else 0.0
        artist_count[arr] = float(len(arr))
    for idxs in track_index.album_to_idx.values():
        arr = np.asarray(idxs, dtype=np.int32)
        vals = pop[arr]
        album_mean[arr] = float(vals.mean()) if len(vals) else 0.0
        album_count[arr] = float(len(arr))
    track_over_artist = np.zeros(n, dtype=np.float32)
    np.divide(pop, artist_mean, out=track_over_artist, where=artist_mean > 0)
    track_over_album = np.zeros(n, dtype=np.float32)
    np.divide(pop, album_mean, out=track_over_album, where=album_mean > 0)
    return {
        "track_log_popularity": np.log1p(pop).astype(np.float32, copy=False),
        "artist_mean_popularity": artist_mean,
        "artist_max_popularity": artist_max,
        "artist_track_count_log1p": np.log1p(artist_count).astype(np.float32, copy=False),
        "album_mean_popularity": album_mean,
        "album_track_count_log1p": np.log1p(album_count).astype(np.float32, copy=False),
        "track_over_artist_mean_popularity": track_over_artist,
        "track_over_album_mean_popularity": track_over_album,
    }


POS_RE = re.compile(r"\b(love|loved|like|liked|perfect|great|awesome|yes|yeah|yep|nice|good|exactly|works)\b", re.I)
NEG_RE = re.compile(r"\b(no|nope|not really|not quite|different|wrong|hate|dislike|too much|too slow|too fast|too sad|too loud|too mellow)\b", re.I)
SHIFT_RE = re.compile(r"\b(more|less|instead|rather|different|switch|change|another|something else|how about|what about|but)\b", re.I)
THOUGHT_STICK_RE = re.compile(r"\b(stick|sticking|continue|continuing|same|similar|another|more from|keeps?|maintain)\b", re.I)
THOUGHT_SWITCH_RE = re.compile(r"\b(switch|switching|different|change|changing|contrast|pivot|instead|new direction|fresh)\b", re.I)
THOUGHT_MOOD_RE = re.compile(r"\b(vibe|mood|energy|feel|feeling|atmosphere|tempo|chill|upbeat|dark|warm|melancholy|dance|calm)\b", re.I)
THOUGHT_REQUEST_RE = re.compile(r"\b(asked|request|requested|looking for|wants?|wanted|user|listener|fits? your|matches? your)\b", re.I)


def reaction_flags(text: object) -> tuple[float, float, float]:
    s = str(text or "")
    if not s.strip():
        return 0.0, 0.0, 0.0
    pos = float(bool(POS_RE.search(s)))
    neg = float(bool(NEG_RE.search(s)))
    shift = float(bool(SHIFT_RE.search(s)))
    return pos, neg, shift


def thought_flags(text: object) -> tuple[float, float, float, float]:
    s = str(text or "")
    if not s.strip():
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(bool(THOUGHT_STICK_RE.search(s))),
        float(bool(THOUGHT_SWITCH_RE.search(s))),
        float(bool(THOUGHT_MOOD_RE.search(s))),
        float(bool(THOUGHT_REQUEST_RE.search(s))),
    )


def prior_music_by_turn(ex: Any, track_index: Any) -> dict[int, int]:
    out: dict[int, int] = {}
    for msg in ex.chat_history:
        if msg.get("role") != "music":
            continue
        idx = track_index.id_to_idx.get(str(msg.get("content") or ""))
        if idx is None:
            continue
        try:
            turn = int(msg.get("turn_number"))
        except (TypeError, ValueError):
            continue
        out[turn] = idx
    return out


def last_music_idx(ex: Any, track_index: Any) -> int | None:
    for msg in reversed(ex.chat_history):
        if msg.get("role") != "music":
            continue
        idx = track_index.id_to_idx.get(str(msg.get("content") or ""))
        if idx is not None:
            return idx
    return None


def prior_user_reaction_counts(ex: Any) -> tuple[float, float, float, float, float, float]:
    pos_count = neg_count = shift_count = 0.0
    last_pos = last_neg = last_shift = 0.0
    for msg in ex.chat_history:
        if msg.get("role") != "user":
            continue
        pos, neg, shift = reaction_flags(msg.get("content"))
        pos_count += pos
        neg_count += neg
        shift_count += shift
        last_pos, last_neg, last_shift = pos, neg, shift
    return pos_count, neg_count, shift_count, last_pos, last_neg, last_shift


def prior_assistant_thought_counts(ex: Any) -> tuple[float, ...]:
    stick_count = switch_count = mood_count = request_count = 0.0
    last_stick = last_switch = last_mood = last_request = 0.0
    thought_count = 0.0
    total_chars = 0.0
    last_chars = 0.0
    for msg in ex.chat_history:
        if msg.get("role") != "music":
            continue
        thought = str(msg.get("thought") or "")
        if not thought.strip():
            continue
        stick, switch, mood, request = thought_flags(thought)
        stick_count += stick
        switch_count += switch
        mood_count += mood
        request_count += request
        last_stick, last_switch, last_mood, last_request = stick, switch, mood, request
        thought_count += 1.0
        total_chars += float(len(thought))
        last_chars = float(len(thought))
    mean_chars = total_chars / thought_count if thought_count > 0 else 0.0
    return (
        thought_count,
        stick_count,
        switch_count,
        mood_count,
        request_count,
        last_stick,
        last_switch,
        last_mood,
        last_request,
        mean_chars,
        last_chars,
    )


def raw_keys(arr: np.ndarray) -> list[str]:
    return [bytes(x).decode("utf-8") for x in arr]


def key_str(source: str, ex: Any) -> str:
    return f"{source}:{ex.session_id}:{int(ex.turn_number)}"


def blind_key_str(ex: Any) -> str:
    return f"{ex.session_id}:{int(ex.turn_number)}"


def row_key_str(ex: Any) -> str:
    return f"{ex.session_id}:{int(ex.turn_number)}"


def make_candidate_set(legacy: Any, indices: np.ndarray, scores: np.ndarray) -> Any:
    return legacy.CandidateSet(indices=indices.astype(np.int32, copy=False), scores=scores.astype(np.float32, copy=False))


def choose_primary_scores(
    candidate_dir: Path,
    cand_arrays: dict[str, np.ndarray],
    *,
    width: int,
    mode: Literal["zero", "bm25", "max_source"],
) -> np.ndarray:
    shape = cand_arrays["track_idx"][:, :width].shape
    if mode == "zero":
        return np.zeros(shape, dtype=np.float32)
    feature_path = candidate_dir / "source_features.npz"
    if not feature_path.exists():
        return np.zeros(shape, dtype=np.float32)
    with np.load(feature_path, allow_pickle=False) as sf:
        if mode == "bm25" and "src__bm25__score__primary" in sf:
            return np.nan_to_num(sf["src__bm25__score__primary"][:, :width], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        if mode == "max_source" and "meta__max_source_score__primary" in sf:
            return np.nan_to_num(sf["meta__max_source_score__primary"][:, :width], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return np.zeros(shape, dtype=np.float32)


def source_feature_plan(
    candidate_dir: Path,
    *,
    drop_cross_source_score_meta: bool = False,
    extra_score_transforms: bool = False,
) -> list[tuple[str, str]]:
    path = candidate_dir / "source_features.npz"
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as sf:
        keys = sorted(k for k in sf.files if np.asarray(sf[k]).ndim == 2)
    plan: list[tuple[str, str]] = []
    for key in keys:
        if drop_cross_source_score_meta and key == "meta__max_source_score__primary":
            # score__primary is only meaningful within each retriever. Taking
            # the max across heterogeneous sources mixes incomparable scales.
            # Keep per-source raw/z-scored fields instead.
            continue
        clean = key.replace("__", "_").replace("/", "_")
        if key.endswith("__rank") or key.endswith("_rank") or "rank" in key:
            plan.extend(
                [
                    (f"src_{clean}_miss0", key),
                    (f"src_{clean}_log1p", key),
                    (f"src_{clean}_inv", key),
                ]
            )
        elif any(token in key for token in ("score", "sim", "similarity", "distance", "dist", "count", "weight", "logit", "prob")):
            items = [
                (f"src_{clean}_raw0", key),
                (f"src_{clean}_row_z", key),
            ]
            if extra_score_transforms:
                items.extend(
                    [
                        (f"src_{clean}_row_max_ratio", key),
                        (f"src_{clean}_row_max_gap", key),
                    ]
                )
            plan.extend(items)
        else:
            plan.append((f"src_{clean}", key))
    return plan


def append_source_features(
    x_base: np.ndarray,
    candidate_dir: Path,
    source_rows: np.ndarray,
    valid_mask: np.ndarray,
    *,
    width: int,
    enabled: bool,
    drop_cross_source_score_meta: bool,
    extra_score_transforms: bool,
) -> tuple[np.ndarray, list[str]]:
    if not enabled:
        return x_base, []
    plan = source_feature_plan(
        candidate_dir,
        drop_cross_source_score_meta=drop_cross_source_score_meta,
        extra_score_transforms=extra_score_transforms,
    )
    if not plan:
        return x_base, []
    raw_keys_needed = sorted({key for _, key in plan})
    out = np.empty((x_base.shape[0], x_base.shape[1] + len(plan)), dtype=np.float32)
    out[:, : x_base.shape[1]] = x_base
    col = x_base.shape[1]
    names: list[str] = []
    with np.load(candidate_dir / "source_features.npz", allow_pickle=False) as sf:
        for raw_key in raw_keys_needed:
            arr = np.asarray(sf[raw_key][source_rows, :width])
            raw_float = arr.astype(np.float32, copy=False)
            finite = np.isfinite(raw_float)
            if "__rank" in raw_key or raw_key.endswith("_rank") or "rank" in raw_key:
                rank = np.where((raw_float > 0) & finite, raw_float, 0.0).astype(np.float32, copy=False)
                inv = np.zeros_like(rank, dtype=np.float32)
                np.divide(1.0, rank, out=inv, where=rank > 0)
                values_by_name = {
                    f"src_{raw_key.replace('__', '_')}_miss0": rank,
                    f"src_{raw_key.replace('__', '_')}_log1p": np.log1p(rank).astype(np.float32, copy=False),
                    f"src_{raw_key.replace('__', '_')}_inv": inv,
                }
            elif any(token in raw_key for token in ("score", "sim", "similarity", "distance", "dist", "count", "weight", "logit", "prob")):
                score = np.nan_to_num(raw_float, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                denom = np.maximum(finite.sum(axis=1, keepdims=True), 1)
                row_mean = np.where(denom > 0, np.nan_to_num(raw_float, nan=0.0, posinf=0.0, neginf=0.0).sum(axis=1, keepdims=True) / denom, 0.0)
                centered = np.where(finite, raw_float - row_mean, 0.0)
                row_std = np.sqrt((centered * centered).sum(axis=1, keepdims=True) / denom) + 1e-6
                z = np.where(finite, centered / row_std, 0.0).astype(np.float32, copy=False)
                max_input = np.where(finite, raw_float, -np.inf)
                row_max = max_input.max(axis=1, keepdims=True)
                row_max = np.where(np.isfinite(row_max), row_max, 0.0).astype(np.float32, copy=False)
                ratio = np.zeros_like(score, dtype=np.float32)
                np.divide(score, row_max, out=ratio, where=row_max > 0)
                gap = np.where(finite & (row_max > 0), row_max - raw_float, 0.0).astype(np.float32, copy=False)
                values_by_name = {
                    f"src_{raw_key.replace('__', '_')}_raw0": score,
                    f"src_{raw_key.replace('__', '_')}_row_z": z,
                    f"src_{raw_key.replace('__', '_')}_row_max_ratio": ratio,
                    f"src_{raw_key.replace('__', '_')}_row_max_gap": gap,
                }
            else:
                values_by_name = {
                    f"src_{raw_key.replace('__', '_')}": np.nan_to_num(raw_float, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
                }
            for feature_name, plan_key in plan:
                if plan_key != raw_key:
                    continue
                value = values_by_name[feature_name]
                out[:, col] = value[valid_mask]
                names.append(feature_name)
                col += 1
            del arr, raw_float, finite, values_by_name
            gc.collect()
    if col != out.shape[1]:
        raise RuntimeError(f"source feature fill mismatch col={col} width={out.shape[1]}")
    return out, names


def append_extra_candidate_feature_npz(
    x_base: np.ndarray,
    feature_paths: list[Path],
    source_rows: np.ndarray,
    valid_mask: np.ndarray,
    *,
    width: int,
) -> tuple[np.ndarray, list[str]]:
    """Append external 2D per-candidate feature arrays aligned to candidate rows."""
    if not feature_paths:
        return x_base, []
    feature_items: list[tuple[Path, str, str]] = []
    for path_raw in feature_paths:
        path = path_raw if path_raw.is_absolute() else REPO_ROOT / path_raw
        with np.load(path, allow_pickle=False) as data:
            for key in sorted(data.files):
                arr = np.asarray(data[key])
                if arr.ndim != 2:
                    continue
                clean = f"ext_{path.stem}_{key}".replace("__", "_").replace("/", "_")
                feature_items.append((path, key, clean))
    if not feature_items:
        return x_base, []

    names: list[str] = []
    values: list[np.ndarray] = []
    for path, key, clean in feature_items:
        with np.load(path, allow_pickle=False) as data:
            raw = np.asarray(data[key][source_rows, :width], dtype=np.float32)
        finite = np.isfinite(raw)
        score = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        denom = np.maximum(finite.sum(axis=1, keepdims=True), 1)
        row_mean = score.sum(axis=1, keepdims=True) / denom
        centered = np.where(finite, raw - row_mean, 0.0)
        row_std = np.sqrt((centered * centered).sum(axis=1, keepdims=True) / denom) + 1e-6
        z = np.where(finite, centered / row_std, 0.0).astype(np.float32, copy=False)
        values.append(score[valid_mask])
        names.append(f"{clean}_raw0")
        values.append(z[valid_mask])
        names.append(f"{clean}_row_z")
        del raw, finite, score, centered, z
        gc.collect()

    extra = np.column_stack(values).astype(np.float32, copy=False)
    out = np.empty((x_base.shape[0], x_base.shape[1] + extra.shape[1]), dtype=np.float32)
    out[:, : x_base.shape[1]] = x_base
    out[:, x_base.shape[1] :] = extra
    return out, names


EXTRA_METADATA_FEATURE_NAMES = [
    "extra_isrc_year",
    "extra_isrc_missing",
    "extra_isrc_country_bucket",
    "extra_isrc_country_eq_user_country",
    "extra_release_minus_isrc_year_abs",
    "extra_duration_bucket",
    "extra_duration_vs_history_mean",
    "extra_duration_vs_album_mean",
    "extra_age_release_alignment",
    "extra_specificity_track_high",
    "extra_specificity_expr_high",
    "extra_specificity_both_high",
    "extra_session_month",
    "extra_session_is_weekend",
    "extra_days_to_xmas",
    "extra_days_to_halloween",
    "extra_christmas_tag_near_xmas",
    "extra_halloween_tag_near_halloween",
]


def append_extra_metadata_features(
    x_base: np.ndarray,
    legacy: Any,
    examples: list[Any],
    candidates: Any,
    track_index: Any,
    *,
    width: int,
    enabled: bool,
) -> tuple[np.ndarray, list[str]]:
    if not enabled:
        return x_base, []
    indices = candidates.indices[:, :width]
    valid_mask = indices >= 0
    n_extra = len(EXTRA_METADATA_FEATURE_NAMES)
    extra = np.zeros((int(valid_mask.sum()), n_extra), dtype=np.float32)
    ctx = extra_metadata_feature_context(track_index)
    sess_dates = session_date_lookup(legacy)
    offset = 0
    for row_i, ex in enumerate(examples):
        pos = np.flatnonzero(valid_mask[row_i])
        if len(pos) == 0:
            continue
        cand_idx = indices[row_i, pos].astype(np.int32, copy=False)
        m = len(cand_idx)
        sl = slice(offset, offset + m)
        row = extra[sl]

        user_country = str((ex.user_profile or {}).get("country_code") or "").upper()
        cand_countries = ctx["isrc_country"][cand_idx]
        row[:, 0] = ctx["isrc_year"][cand_idx]
        row[:, 1] = ctx["isrc_missing"][cand_idx]
        row[:, 2] = ctx["isrc_country_bucket"][cand_idx]
        row[:, 3] = np.asarray([float(str(c) == user_country and bool(user_country)) for c in cand_countries], dtype=np.float32)
        row[:, 4] = ctx["release_minus_isrc_abs"][cand_idx]
        row[:, 5] = ctx["duration_bucket"][cand_idx]

        history_idx: list[int] = []
        for msg in ex.chat_history:
            if msg.get("role") != "music":
                continue
            idx = track_index.id_to_idx.get(str(msg.get("content") or ""))
            if idx is not None:
                history_idx.append(idx)
        if history_idx:
            hist_dur = track_index.duration[np.asarray(history_idx, dtype=np.int32)]
            hist_dur = hist_dur[hist_dur > 0]
            hist_mean = float(hist_dur.mean()) if len(hist_dur) else 0.0
        else:
            hist_mean = 0.0
        if hist_mean > 0:
            row[:, 6] = (track_index.duration[cand_idx] - hist_mean) / 60_000.0
        else:
            row[:, 6] = 0.0
        album_mean = ctx["album_mean_duration"][cand_idx]
        row[:, 7] = np.where(album_mean > 0, (track_index.duration[cand_idx] - album_mean) / 60_000.0, 0.0)

        age = float((ex.user_profile or {}).get("age") or 0.0)
        # Blind-B-safe: session_date is empty in Blind B → blank it so age_release_alignment /
        # session_month / weekend / xmas / halloween features become 0 (train+inference consistent).
        session_date = None  # blind-B-safe fixed: session_date is never used
        session_year = float(session_date.year if session_date else 2026)
        rel_year = track_index.release_year[cand_idx]
        row[:, 8] = np.where((age > 0) & (rel_year > 0), age - (session_year - rel_year), 0.0)

        # Blind B has no conversation_goal → specificity unavailable. Blank it so the
        # 3 specificity features become 0 (train/inference consistent).
        specificity = ""  # blind-B-safe fixed: goal specificity is never used
        track_high = float(len(specificity) >= 1 and specificity[0].upper() == "H")
        expr_high = float(len(specificity) >= 2 and specificity[1].upper() == "H")
        row[:, 9] = track_high
        row[:, 10] = expr_high
        row[:, 11] = float(track_high and expr_high)

        if session_date is not None:
            row[:, 12] = float(session_date.month)
            row[:, 13] = float(session_date.weekday() >= 5)
            xmas = day_distance_in_year(session_date, 12, 25)
            halloween = day_distance_in_year(session_date, 10, 31)
        else:
            xmas = halloween = 0.0
        row[:, 14] = xmas
        row[:, 15] = halloween
        row[:, 16] = ctx["christmas_tag"][cand_idx] * float(0 < xmas <= 45)
        row[:, 17] = ctx["halloween_tag"][cand_idx] * float(0 < halloween <= 30)
        offset += m
    if offset != len(extra):
        raise RuntimeError(f"extra metadata feature length mismatch offset={offset} total={len(extra)}")
    out = np.empty((x_base.shape[0], x_base.shape[1] + n_extra), dtype=np.float32)
    out[:, : x_base.shape[1]] = x_base
    out[:, x_base.shape[1] :] = extra
    return out, list(EXTRA_METADATA_FEATURE_NAMES)


EXTRA_FEEDBACK_FEATURE_NAMES = [
    "extra_rejected_turn_count",
    "extra_latest_goal_not_move",
    "extra_rejected_track_match",
    "extra_rejected_artist_match",
    "extra_rejected_album_match",
    "extra_rejected_primary_tag_match",
    "extra_latest_rejected_same_artist",
    "extra_latest_rejected_same_album",
    "extra_prior_user_positive_count",
    "extra_prior_user_negative_count",
    "extra_prior_user_shift_count",
    "extra_prior_user_last_positive",
    "extra_prior_user_last_negative",
    "extra_prior_user_last_shift",
    "extra_current_user_positive",
    "extra_current_user_negative",
    "extra_current_user_shift",
    "extra_current_negative_same_last_artist",
    "extra_current_negative_same_last_album",
    "extra_current_positive_same_last_artist",
    "extra_current_positive_same_last_album",
    "extra_current_shift_not_same_last_artist",
]


def append_extra_feedback_features(
    x_base: np.ndarray,
    examples: list[Any],
    candidates: Any,
    track_index: Any,
    *,
    width: int,
    gpa_enabled: bool,
    reaction_enabled: bool,
) -> tuple[np.ndarray, list[str]]:
    if not (gpa_enabled or reaction_enabled):
        return x_base, []
    indices = candidates.indices[:, :width]
    valid_mask = indices >= 0
    n_extra = len(EXTRA_FEEDBACK_FEATURE_NAMES)
    extra = np.zeros((int(valid_mask.sum()), n_extra), dtype=np.float32)
    ctx = extra_feedback_feature_context(track_index)
    artist_code = ctx["artist_code"]
    album_code = ctx["album_code"]
    tag_code = ctx["primary_tag_code"]

    offset = 0
    for row_i, ex in enumerate(examples):
        pos = np.flatnonzero(valid_mask[row_i])
        if len(pos) == 0:
            continue
        cand_idx = indices[row_i, pos].astype(np.int32, copy=False)
        m = len(cand_idx)
        sl = slice(offset, offset + m)
        row = extra[sl]

        if gpa_enabled:
            progress = [str(x or "") for x in ex.prior_goal_progress]
            rejected_turns = {
                i + 1
                for i, value in enumerate(progress)
                if value == "DOES_NOT_MOVE_TOWARD_GOAL"
            }
            music_by_turn = prior_music_by_turn(ex, track_index)
            rejected_idx = [music_by_turn[t] for t in sorted(rejected_turns) if t in music_by_turn]
            latest_rejected = rejected_idx[-1] if rejected_idx else None
            latest_not_move = float(bool(progress) and progress[-1] == "DOES_NOT_MOVE_TOWARD_GOAL")

            row[:, 0] = float(len(rejected_idx))
            row[:, 1] = latest_not_move
            if rejected_idx:
                rejected_arr = np.asarray(rejected_idx, dtype=np.int32)
                row[:, 2] = np.isin(cand_idx, rejected_arr, assume_unique=False).astype(np.float32)
                rej_artist = artist_code[rejected_arr]
                rej_album = album_code[rejected_arr]
                rej_tag = tag_code[rejected_arr]
                row[:, 3] = np.isin(artist_code[cand_idx], rej_artist[rej_artist > 0], assume_unique=False).astype(np.float32)
                row[:, 4] = np.isin(album_code[cand_idx], rej_album[rej_album > 0], assume_unique=False).astype(np.float32)
                row[:, 5] = np.isin(tag_code[cand_idx], rej_tag[rej_tag > 0], assume_unique=False).astype(np.float32)
                if latest_rejected is not None:
                    latest_artist = artist_code[latest_rejected]
                    latest_album = album_code[latest_rejected]
                    row[:, 6] = ((artist_code[cand_idx] == latest_artist) & (latest_artist > 0)).astype(np.float32)
                    row[:, 7] = ((album_code[cand_idx] == latest_album) & (latest_album > 0)).astype(np.float32)

        if reaction_enabled:
            prior_pos, prior_neg, prior_shift, prior_last_pos, prior_last_neg, prior_last_shift = prior_user_reaction_counts(ex)
            cur_pos, cur_neg, cur_shift = reaction_flags(ex.user_query)
            row[:, 8] = prior_pos
            row[:, 9] = prior_neg
            row[:, 10] = prior_shift
            row[:, 11] = prior_last_pos
            row[:, 12] = prior_last_neg
            row[:, 13] = prior_last_shift
            row[:, 14] = cur_pos
            row[:, 15] = cur_neg
            row[:, 16] = cur_shift

            last_idx = last_music_idx(ex, track_index)
            if last_idx is not None:
                same_last_artist = ((artist_code[cand_idx] == artist_code[last_idx]) & (artist_code[last_idx] > 0)).astype(np.float32)
                same_last_album = ((album_code[cand_idx] == album_code[last_idx]) & (album_code[last_idx] > 0)).astype(np.float32)
                row[:, 17] = cur_neg * same_last_artist
                row[:, 18] = cur_neg * same_last_album
                row[:, 19] = cur_pos * same_last_artist
                row[:, 20] = cur_pos * same_last_album
                row[:, 21] = cur_shift * (1.0 - same_last_artist)
        offset += m

    if offset != len(extra):
        raise RuntimeError(f"extra feedback feature length mismatch offset={offset} total={len(extra)}")
    out = np.empty((x_base.shape[0], x_base.shape[1] + n_extra), dtype=np.float32)
    out[:, : x_base.shape[1]] = x_base
    out[:, x_base.shape[1] :] = extra
    return out, list(EXTRA_FEEDBACK_FEATURE_NAMES)


EXTRA_GOAL_CLUSTER_FEATURE_NAMES = [
    "extra_goal_cluster_id",
    "extra_goal_cluster_distance",
    "extra_goal_cluster_confidence",
]


EXTRA_ASSISTANT_THOUGHT_FEATURE_NAMES = [
    "extra_assistant_thought_count",
    "extra_assistant_thought_stick_count",
    "extra_assistant_thought_switch_count",
    "extra_assistant_thought_mood_count",
    "extra_assistant_thought_request_count",
    "extra_assistant_thought_last_stick",
    "extra_assistant_thought_last_switch",
    "extra_assistant_thought_last_mood",
    "extra_assistant_thought_last_request",
    "extra_assistant_thought_mean_chars",
    "extra_assistant_thought_last_chars",
]


EXTRA_HIER_POP_FEATURE_NAMES = [
    "extra_track_log_popularity",
    "extra_artist_mean_popularity",
    "extra_artist_max_popularity",
    "extra_artist_track_count_log1p",
    "extra_album_mean_popularity",
    "extra_album_track_count_log1p",
    "extra_track_over_artist_mean_popularity",
    "extra_track_over_album_mean_popularity",
]


EXTRA_CATEGORY_TURN_FEATURE_NAMES = [
    "extra_bucket_size_log1p",
    "extra_bucket_exact_track_count",
    "extra_bucket_exact_track_prob",
    "extra_bucket_artist_count",
    "extra_bucket_artist_prob",
    "extra_bucket_album_count",
    "extra_bucket_album_prob",
    "extra_bucket_primary_tag_count",
    "extra_bucket_primary_tag_prob",
    "extra_bucket_track_popularity_vs_gold_mean",
    "extra_bucket_release_year_vs_gold_mean",
]


def score_calibration_feature_name(raw_key: str) -> str:
    return "cal_" + raw_key.replace("__", "_").replace("/", "_") + "_isotonic"


def source_score_keys(candidate_dir: Path) -> list[str]:
    path = candidate_dir / "source_features.npz"
    if not path.exists():
        return []
    with np.load(path, allow_pickle=False) as data:
        return sorted(k for k in data.files if "__score__" in k)


def fit_binned_isotonic(x: np.ndarray, y: np.ndarray, *, n_bins: int = 128) -> dict[str, np.ndarray] | None:
    finite = np.isfinite(x)
    x = np.asarray(x[finite], dtype=np.float32)
    y = np.asarray(y[finite], dtype=np.float32)
    if len(x) < 1000 or float(y.sum()) <= 0.0:
        return None
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.unique(np.quantile(x, quantiles).astype(np.float32))
    if len(edges) < 3:
        return None
    bins = np.searchsorted(edges[1:-1], x, side="right")
    counts = np.bincount(bins, minlength=len(edges) - 1).astype(np.float64)
    positives = np.bincount(bins, weights=y, minlength=len(edges) - 1).astype(np.float64)
    score_sums = np.bincount(bins, weights=x, minlength=len(edges) - 1).astype(np.float64)
    valid = counts > 0
    if int(valid.sum()) < 2:
        return None
    centers = (score_sums[valid] / counts[valid]).astype(np.float32)
    # Jeffreys-style smoothing keeps tiny bins from becoming hard 0/1 labels.
    rates = ((positives[valid] + 0.5) / (counts[valid] + 1.0)).astype(np.float32)
    weights = counts[valid]
    iso = IsotonicRegression(increasing=True, out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(centers, rates, sample_weight=weights)
    bin_values = np.zeros(len(edges) - 1, dtype=np.float32)
    bin_values[valid] = iso.predict(centers).astype(np.float32)
    if np.any(~valid):
        global_rate = float((y.sum() + 0.5) / (len(y) + 1.0))
        bin_values[~valid] = global_rate
    return {"edges": edges.astype(np.float32), "bin_values": bin_values}


def fit_score_calibration_context(
    candidate_dir: Path,
    indices: np.ndarray,
    examples: list[Any],
    source_rows: np.ndarray,
    track_index: Any,
    *,
    enabled: bool,
) -> dict[str, Any] | None:
    if not enabled:
        return None
    path = candidate_dir / "source_features.npz"
    if not path.exists():
        return None
    keys = source_score_keys(candidate_dir)
    if not keys:
        return None
    print(f"fit source score calibration features={len(keys)} rows={len(source_rows)}")
    gold_idx = np.asarray(
        [track_index.id_to_idx.get(ex.gold_track_id or "", -1) for ex in examples],
        dtype=np.int32,
    )
    calibrators: dict[str, dict[str, np.ndarray]] = {}
    chunk = 2048
    with np.load(path, allow_pickle=False) as data:
        for key in keys:
            xs: list[np.ndarray] = []
            ys: list[np.ndarray] = []
            arr = data[key]
            for start in range(0, len(source_rows), chunk):
                rows = source_rows[start : start + chunk]
                values = np.asarray(arr[rows, : indices.shape[1]], dtype=np.float32)
                finite = np.isfinite(values)
                if not np.any(finite):
                    continue
                labels = indices[rows, : indices.shape[1]] == gold_idx[rows, None]
                xs.append(values[finite].astype(np.float32, copy=False))
                ys.append(labels[finite].astype(np.float32, copy=False))
            if not xs:
                continue
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            calibrator = fit_binned_isotonic(x, y)
            if calibrator is not None:
                calibrators[key] = calibrator
                print(f"  calibrated {key}: n={len(x)} positives={int(y.sum())}")
            del xs, ys, x, y
            gc.collect()
    if not calibrators:
        return None
    return {"calibrators": calibrators}


def append_extra_score_calibration_features(
    x_base: np.ndarray,
    candidate_dir: Path,
    source_rows: np.ndarray,
    valid_mask: np.ndarray,
    *,
    width: int,
    context: dict[str, Any] | None,
) -> tuple[np.ndarray, list[str]]:
    if context is None:
        return x_base, []
    calibrators: dict[str, dict[str, np.ndarray]] = context.get("calibrators") or {}
    if not calibrators:
        return x_base, []
    path = candidate_dir / "source_features.npz"
    if not path.exists():
        return x_base, []
    n_extra = len(calibrators)
    extra = np.zeros((int(valid_mask.sum()), n_extra), dtype=np.float32)
    names: list[str] = []
    with np.load(path, allow_pickle=False) as data:
        for col, key in enumerate(sorted(calibrators)):
            if key not in data:
                names.append(score_calibration_feature_name(key))
                continue
            cal = calibrators[key]
            values = np.asarray(data[key][source_rows, :width], dtype=np.float32)
            flat = values[valid_mask]
            out = np.zeros(len(flat), dtype=np.float32)
            finite = np.isfinite(flat)
            if np.any(finite):
                edges = cal["edges"]
                bin_values = cal["bin_values"]
                bins = np.searchsorted(edges[1:-1], flat[finite], side="right")
                out[finite] = bin_values[bins]
            extra[:, col] = out
            names.append(score_calibration_feature_name(key))
            del values, flat, out
            gc.collect()
    out_matrix = np.empty((x_base.shape[0], x_base.shape[1] + n_extra), dtype=np.float32)
    out_matrix[:, : x_base.shape[1]] = x_base
    out_matrix[:, x_base.shape[1] :] = extra
    return out_matrix, names


def append_extra_hier_pop_features(
    x_base: np.ndarray,
    candidates: Any,
    track_index: Any,
    *,
    width: int,
    enabled: bool,
) -> tuple[np.ndarray, list[str]]:
    if not enabled:
        return x_base, []
    indices = candidates.indices[:, :width]
    valid_mask = indices >= 0
    cand_idx = indices[valid_mask].astype(np.int32, copy=False)
    ctx = extra_hier_pop_feature_context(track_index)
    extra = np.column_stack([ctx[name][cand_idx] for name in ctx]).astype(np.float32, copy=False)
    out = np.empty((x_base.shape[0], x_base.shape[1] + extra.shape[1]), dtype=np.float32)
    out[:, : x_base.shape[1]] = x_base
    out[:, x_base.shape[1] :] = extra
    return out, list(EXTRA_HIER_POP_FEATURE_NAMES)


def category_turn_bucket(ex: Any) -> str:
    category = str((ex.conversation_goal or {}).get("category") or "missing").strip() or "missing"
    try:
        turn = int(ex.turn_number)
    except (TypeError, ValueError):
        turn = 0
    turn_bucket = min(max(turn, 0), 8)
    return f"{category}|t{turn_bucket}"


def fit_category_turn_context(
    examples: list[Any],
    track_index: Any,
    *,
    enabled: bool,
) -> dict[str, Any] | None:
    if not enabled:
        return None
    feedback_ctx = extra_feedback_feature_context(track_index)
    artist_code = feedback_ctx["artist_code"]
    album_code = feedback_ctx["album_code"]
    tag_code = feedback_ctx["primary_tag_code"]
    pop = np.log1p(np.asarray(track_index.popularity, dtype=np.float32))
    rel = np.asarray(track_index.release_year, dtype=np.float32)

    bucket_to_id: dict[str, int] = {}
    rows_by_bucket: dict[int, list[int]] = defaultdict(list)
    row_contrib: dict[str, tuple[int, int, int, int, int, float, float]] = {}
    for row_i, ex in enumerate(examples):
        gold_idx = track_index.id_to_idx.get(ex.gold_track_id or "")
        if gold_idx is None:
            continue
        bucket = category_turn_bucket(ex)
        bucket_id = bucket_to_id.setdefault(bucket, len(bucket_to_id))
        rows_by_bucket[bucket_id].append(int(gold_idx))
        row_contrib[row_key_str(ex)] = (
            bucket_id,
            int(gold_idx),
            int(artist_code[gold_idx]),
            int(album_code[gold_idx]),
            int(tag_code[gold_idx]),
            float(pop[gold_idx]),
            float(rel[gold_idx]),
        )

    nb = max(1, len(bucket_to_id))
    n_tracks = len(track_index.track_ids)
    bucket_total = np.zeros(nb, dtype=np.float32)
    bucket_pop_sum = np.zeros(nb, dtype=np.float32)
    bucket_release_sum = np.zeros(nb, dtype=np.float32)
    exact_by_track = np.zeros((nb, n_tracks), dtype=np.float32)
    artist_by_track = np.zeros((nb, n_tracks), dtype=np.float32)
    album_by_track = np.zeros((nb, n_tracks), dtype=np.float32)
    tag_by_track = np.zeros((nb, n_tracks), dtype=np.float32)

    for bucket_id, gold_indices in rows_by_bucket.items():
        arr = np.asarray(gold_indices, dtype=np.int32)
        bucket_total[bucket_id] = float(len(arr))
        bucket_pop_sum[bucket_id] = float(pop[arr].sum())
        valid_rel = rel[arr]
        valid_rel = valid_rel[valid_rel > 0]
        bucket_release_sum[bucket_id] = float(valid_rel.sum())
        np.add.at(exact_by_track[bucket_id], arr, 1.0)
        for code_values, out in (
            (artist_code, artist_by_track),
            (album_code, album_by_track),
            (tag_code, tag_by_track),
        ):
            counts: Counter[int] = Counter(int(x) for x in code_values[arr] if int(x) > 0)
            if not counts:
                continue
            values = np.zeros(n_tracks, dtype=np.float32)
            for code, count in counts.items():
                values[code_values == code] = float(count)
            out[bucket_id] = values

    return {
        "bucket_to_id": bucket_to_id,
        "bucket_total": bucket_total,
        "bucket_pop_sum": bucket_pop_sum,
        "bucket_release_sum": bucket_release_sum,
        "exact_by_track": exact_by_track,
        "artist_by_track": artist_by_track,
        "album_by_track": album_by_track,
        "tag_by_track": tag_by_track,
        "artist_code": artist_code,
        "album_code": album_code,
        "tag_code": tag_code,
        "track_log_popularity": pop,
        "release_year": rel,
        "row_contrib": row_contrib,
    }


def append_extra_category_turn_features(
    x_base: np.ndarray,
    examples: list[Any],
    candidates: Any,
    track_index: Any,
    *,
    width: int,
    context: dict[str, Any] | None,
) -> tuple[np.ndarray, list[str]]:
    if context is None:
        return x_base, []
    indices = candidates.indices[:, :width]
    valid_mask = indices >= 0
    n_extra = len(EXTRA_CATEGORY_TURN_FEATURE_NAMES)
    extra = np.zeros((int(valid_mask.sum()), n_extra), dtype=np.float32)
    offset = 0
    for row_i, ex in enumerate(examples):
        pos = np.flatnonzero(valid_mask[row_i])
        if len(pos) == 0:
            continue
        cand_idx = indices[row_i, pos].astype(np.int32, copy=False)
        m = len(cand_idx)
        row = extra[offset : offset + m]
        bucket_id = context["bucket_to_id"].get(category_turn_bucket(ex))
        if bucket_id is None:
            offset += m
            continue

        total = float(context["bucket_total"][bucket_id])
        pop_sum = float(context["bucket_pop_sum"][bucket_id])
        rel_sum = float(context["bucket_release_sum"][bucket_id])
        own = context["row_contrib"].get(row_key_str(ex))
        own_gold = own_artist = own_album = own_tag = None
        own_pop = own_rel = 0.0
        if own is not None and int(own[0]) == int(bucket_id):
            _, own_gold, own_artist, own_album, own_tag, own_pop, own_rel = own
            total -= 1.0
            pop_sum -= own_pop
            if own_rel > 0:
                rel_sum -= own_rel

        if total <= 0.0:
            offset += m
            continue
        exact = context["exact_by_track"][bucket_id, cand_idx].copy()
        artist = context["artist_by_track"][bucket_id, cand_idx].copy()
        album = context["album_by_track"][bucket_id, cand_idx].copy()
        tag = context["tag_by_track"][bucket_id, cand_idx].copy()
        if own is not None:
            exact -= (cand_idx == int(own_gold)).astype(np.float32)
            artist -= (context["artist_code"][cand_idx] == int(own_artist)).astype(np.float32)
            album -= (context["album_code"][cand_idx] == int(own_album)).astype(np.float32)
            tag -= (context["tag_code"][cand_idx] == int(own_tag)).astype(np.float32)
            np.maximum(exact, 0.0, out=exact)
            np.maximum(artist, 0.0, out=artist)
            np.maximum(album, 0.0, out=album)
            np.maximum(tag, 0.0, out=tag)

        row[:, 0] = math.log1p(total)
        row[:, 1] = exact
        row[:, 2] = exact / total
        row[:, 3] = artist
        row[:, 4] = artist / total
        row[:, 5] = album
        row[:, 6] = album / total
        row[:, 7] = tag
        row[:, 8] = tag / total
        row[:, 9] = context["track_log_popularity"][cand_idx] - (pop_sum / total)
        rel_mean = rel_sum / total if rel_sum > 0.0 else 0.0
        if rel_mean > 0.0:
            row[:, 10] = context["release_year"][cand_idx] - rel_mean
        offset += m
    if offset != len(extra):
        raise RuntimeError(f"extra category-turn feature length mismatch offset={offset} total={len(extra)}")
    out = np.empty((x_base.shape[0], x_base.shape[1] + n_extra), dtype=np.float32)
    out[:, : x_base.shape[1]] = x_base
    out[:, x_base.shape[1] :] = extra
    return out, list(EXTRA_CATEGORY_TURN_FEATURE_NAMES)


def append_extra_assistant_thought_features(
    x_base: np.ndarray,
    examples: list[Any],
    candidates: Any,
    *,
    width: int,
    enabled: bool,
) -> tuple[np.ndarray, list[str]]:
    if not enabled:
        return x_base, []
    indices = candidates.indices[:, :width]
    valid_mask = indices >= 0
    n_extra = len(EXTRA_ASSISTANT_THOUGHT_FEATURE_NAMES)
    extra = np.zeros((int(valid_mask.sum()), n_extra), dtype=np.float32)
    offset = 0
    for row_i, ex in enumerate(examples):
        m = int(valid_mask[row_i].sum())
        if m == 0:
            continue
        values = np.asarray(prior_assistant_thought_counts(ex), dtype=np.float32)
        extra[offset : offset + m, :] = values[None, :]
        offset += m
    if offset != len(extra):
        raise RuntimeError(f"extra assistant thought feature length mismatch offset={offset} total={len(extra)}")
    out = np.empty((x_base.shape[0], x_base.shape[1] + n_extra), dtype=np.float32)
    out[:, : x_base.shape[1]] = x_base
    out[:, x_base.shape[1] :] = extra
    return out, list(EXTRA_ASSISTANT_THOUGHT_FEATURE_NAMES)


EXTRA_TAG_CHAIN_FEATURE_NAMES = [
    "extra_tag_token_overlap_with_history",
    "extra_tag_token_jaccard_with_history",
    "extra_tag_vec_cosine_with_history",
    "extra_tag_chain_neighbor_overlap",
    "extra_tag_chain_ppmi_sum",
    "extra_tag_chain_ppmi_max",
]


EXTRA_POOL_PRIOR_FEATURE_NAMES = [
    "extra_pool_hist_count",
    "extra_pool_turn_norm",
    "extra_pool_cf_max",
    "extra_pool_cf_last",
    "extra_pool_cf_mean_top3",
    "extra_pool_metadata_max",
    "extra_pool_metadata_last",
    "extra_pool_attributes_max",
    "extra_pool_attributes_last",
    "extra_pool_audio_max",
    "extra_pool_audio_last",
    "extra_pool_lyrics_max",
    "extra_pool_image_max",
    "extra_pool_same_artist_any",
    "extra_pool_same_album_any",
    "extra_pool_same_artist_last",
    "extra_pool_same_album_last",
    "extra_pool_tag_jaccard_max",
    "extra_pool_tag_jaccard_last",
    "extra_pool_release_year_absdiff_last",
    "extra_pool_duration_absdiff_last",
    "extra_pool_multimodal_max_mean",
]


EXTRA_TALKPLAY_AUX_FEATURE_NAMES = [
    "aux_spec_track_precision",
    "aux_spec_refinement",
    "aux_turn_progress",
    "aux_goal_tag_overlap",
    "aux_goal_tag_jaccard",
    "aux_goal_tag_jaccard_x_posproxy",
    "aux_profile_culture_tag_overlap",
    "aux_profile_culture_tag_jaccard",
    "aux_assistant_tag_overlap",
    "aux_assistant_tag_jaccard",
    "aux_current_goal_token_jaccard",
    "aux_pivot_flag",
]

_AUX_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _aux_tokens(text: object) -> set[str]:
    return {t for t in _AUX_TOKEN_RE.findall(str(text or "").lower()) if len(t) >= 2}


def append_extra_talkplay_aux_features(
    x_base: np.ndarray,
    examples: list[Any],
    candidates: Any,
    track_index: Any,
    *,
    width: int,
    enabled: bool,
) -> tuple[np.ndarray, list[str]]:
    """Bundled talkplay aux features (4-B profile / 4-C specificity / 4-D assistant / 7-B goal / 7-C pivot).

    All lexical/metadata, full coverage, vectorized via tag_bin sparse matrix.
    """
    if not enabled:
        return x_base, []
    indices = candidates.indices[:, :width]
    valid_mask = indices >= 0
    n_extra = len(EXTRA_TALKPLAY_AUX_FEATURE_NAMES)
    extra = np.zeros((int(valid_mask.sum()), n_extra), dtype=np.float32)
    ctx = tag_chain_feature_context(track_index)
    tag_bin = ctx["tag_bin"]
    tag_count = ctx["tag_count"]
    vocab = ctx["vocab"]

    spec_precision_map = {"HH": 2.0, "HL": 1.5, "LH": 0.8, "LL": 0.5}
    spec_refine_map = {"HH": 0.4, "HL": 0.7, "LH": 1.5, "LL": 1.2}

    def _ids(text: object) -> list[int]:
        return sorted({vocab[t] for t in _aux_tokens(text) if t in vocab})

    offset = 0
    for row_i, ex in enumerate(examples):
        pos = np.flatnonzero(valid_mask[row_i])
        m = len(pos)
        if m == 0:
            continue
        cand_idx = indices[row_i, pos].astype(np.int32, copy=False)
        row = extra[offset : offset + m]

        goal = dict(ex.conversation_goal or {})
        profile = dict(ex.user_profile or {})
        spec = str(goal.get("specificity") or "").upper()
        row[:, 0] = spec_precision_map.get(spec, 1.0)
        row[:, 1] = spec_refine_map.get(spec, 1.0)
        row[:, 2] = (float(getattr(ex, "turn_number", 1) or 1) - 1.0) / 7.0

        cur_text = f"{ex.user_query or ''} {getattr(ex, 'user_query_thought', '') or ''}"
        pos_proxy, _, _ = reaction_flags(cur_text)

        goal_text = str(goal.get("listener_goal") or "")
        goal_ids = _ids(goal_text)
        cand_bin = tag_bin[cand_idx]
        cand_tagcount = tag_count[cand_idx]
        if goal_ids:
            ov = _as_1d_float(cand_bin[:, goal_ids].sum(axis=1))
            denom = cand_tagcount + float(len(goal_ids)) - ov
            jac = np.divide(ov, denom, out=np.zeros_like(ov), where=denom > 0)
            row[:, 3] = ov
            row[:, 4] = jac
            row[:, 5] = jac * pos_proxy

        culture_text = " ".join(
            str(profile.get(k) or "")
            for k in ("preferred_musical_culture", "country_name", "preferred_language")
        )
        culture_ids = _ids(culture_text)
        if culture_ids:
            ov = _as_1d_float(cand_bin[:, culture_ids].sum(axis=1))
            denom = cand_tagcount + float(len(culture_ids)) - ov
            row[:, 6] = ov
            row[:, 7] = np.divide(ov, denom, out=np.zeros_like(ov), where=denom > 0)

        assist_parts: list[str] = []
        for msg in ex.chat_history:
            if msg.get("role") == "assistant":
                assist_parts.append(str(msg.get("content") or ""))
                assist_parts.append(str(msg.get("thought") or ""))
        assist_ids = _ids(" ".join(assist_parts))
        if assist_ids:
            ov = _as_1d_float(cand_bin[:, assist_ids].sum(axis=1))
            denom = cand_tagcount + float(len(assist_ids)) - ov
            row[:, 8] = ov
            row[:, 9] = np.divide(ov, denom, out=np.zeros_like(ov), where=denom > 0)

        # pivot: current text vs goal text raw token jaccard (low = pivot away from goal)
        cur_tok = _aux_tokens(cur_text)
        goal_tok = _aux_tokens(goal_text)
        if cur_tok and goal_tok:
            inter = len(cur_tok & goal_tok)
            union = len(cur_tok | goal_tok)
            cg_jac = inter / union if union else 0.0
        else:
            cg_jac = 0.0
        row[:, 10] = cg_jac
        row[:, 11] = float(cg_jac < 0.1 and bool(goal_tok))

        offset += m
    if offset != len(extra):
        raise RuntimeError(f"aux feature length mismatch offset={offset} total={len(extra)}")
    out = np.empty((x_base.shape[0], x_base.shape[1] + n_extra), dtype=np.float32)
    out[:, : x_base.shape[1]] = x_base
    out[:, x_base.shape[1] :] = extra
    return out, list(EXTRA_TALKPLAY_AUX_FEATURE_NAMES)


_POOL_PRIOR_CACHE: dict[str, np.ndarray] | None = None


def extra_pool_prior_feature_context(track_index: Any) -> dict[str, np.ndarray]:
    """Load shared embeddings for pool prior features (catalog-only, no labels)."""
    global _POOL_PRIOR_CACHE
    if _POOL_PRIOR_CACHE is not None:
        return _POOL_PRIOR_CACHE
    cache_dir = REPO_ROOT / "data" / "derived"
    cf = np.load(cache_dir / "track_emb_cf_v1.npy")
    metadata = np.load(cache_dir / "track_emb_metadata_v1.npy")
    attributes = np.load(cache_dir / "track_emb_attributes_v1.npy")
    audio = np.load(cache_dir / "track_emb_audio_v1.npy")
    lyrics = np.load(cache_dir / "track_emb_lyrics_v1.npy")
    image = np.load(cache_dir / "track_emb_image_v1.npy")

    def _normalize(x: np.ndarray) -> np.ndarray:
        return x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-9)

    fb_ctx = extra_feedback_feature_context(track_index)
    _POOL_PRIOR_CACHE = {
        "cf_norm": _normalize(cf).astype(np.float32),
        "metadata_norm": _normalize(metadata).astype(np.float32),
        "attributes_norm": _normalize(attributes).astype(np.float32),
        "audio_norm": _normalize(audio).astype(np.float32),
        "lyrics_norm": _normalize(lyrics).astype(np.float32),
        "image_norm": _normalize(image).astype(np.float32),
        "artist_code": fb_ctx["artist_code"],
        "album_code": fb_ctx["album_code"],
        "release_year": np.asarray(track_index.release_year, dtype=np.float32),
        "duration": np.asarray(track_index.duration, dtype=np.float32),
    }
    return _POOL_PRIOR_CACHE


def append_extra_pool_prior_features(
    x_base: np.ndarray,
    examples: list[Any],
    candidates: Any,
    track_index: Any,
    *,
    width: int,
    enabled: bool,
) -> tuple[np.ndarray, list[str]]:
    if not enabled:
        return x_base, []
    indices = candidates.indices[:, :width]
    valid_mask = indices >= 0
    n_extra = len(EXTRA_POOL_PRIOR_FEATURE_NAMES)
    extra = np.zeros((int(valid_mask.sum()), n_extra), dtype=np.float32)
    ctx = extra_pool_prior_feature_context(track_index)
    cf = ctx["cf_norm"]
    metadata = ctx["metadata_norm"]
    attributes = ctx["attributes_norm"]
    audio = ctx["audio_norm"]
    lyrics = ctx["lyrics_norm"]
    image = ctx["image_norm"]
    artist_code = ctx["artist_code"]
    album_code = ctx["album_code"]
    release_year = ctx["release_year"]
    duration = ctx["duration"]
    # tag features reuse tag_chain context
    tag_ctx = tag_chain_feature_context(track_index)
    tag_bin = tag_ctx["tag_bin"]
    tag_count = tag_ctx["tag_count"]

    offset = 0
    for row_i, ex in enumerate(examples):
        pos = np.flatnonzero(valid_mask[row_i])
        if len(pos) == 0:
            continue
        cand_idx = indices[row_i, pos].astype(np.int32, copy=False)
        m = len(cand_idx)
        row = extra[offset : offset + m]

        hist_idx: list[int] = []
        for msg in ex.chat_history:
            if msg.get("role") != "music":
                continue
            tid = str(msg.get("content") or "")
            idx = track_index.id_to_idx.get(tid)
            if idx is not None:
                hist_idx.append(int(idx))
        if not hist_idx:
            offset += m
            continue

        hist_arr = np.asarray(hist_idx, dtype=np.int32)
        last_h = int(hist_arr[-1])
        row[:, 0] = float(len(hist_arr))
        turn = float(getattr(ex, "turn_number", 1) or 1)
        row[:, 1] = (turn - 1.0) / 7.0

        # CF
        hist_cf = cf[hist_arr]
        cand_cf = cf[cand_idx]
        cf_sim = cand_cf @ hist_cf.T  # (m, h)
        row[:, 2] = cf_sim.max(axis=1)
        row[:, 3] = cand_cf @ cf[last_h]
        top3_n = min(3, cf_sim.shape[1])
        row[:, 4] = np.sort(cf_sim, axis=1)[:, -top3_n:].mean(axis=1)

        # Metadata
        hist_m = metadata[hist_arr]
        cand_m = metadata[cand_idx]
        m_sim = cand_m @ hist_m.T
        row[:, 5] = m_sim.max(axis=1)
        row[:, 6] = cand_m @ metadata[last_h]

        # Attributes
        hist_a = attributes[hist_arr]
        cand_a = attributes[cand_idx]
        a_sim = cand_a @ hist_a.T
        row[:, 7] = a_sim.max(axis=1)
        row[:, 8] = cand_a @ attributes[last_h]

        # Audio
        hist_au = audio[hist_arr]
        cand_au = audio[cand_idx]
        au_sim = cand_au @ hist_au.T
        row[:, 9] = au_sim.max(axis=1)
        row[:, 10] = cand_au @ audio[last_h]

        # Lyrics (mean cos)
        hist_l = lyrics[hist_arr]
        cand_l = lyrics[cand_idx]
        l_sim = cand_l @ hist_l.T
        row[:, 11] = l_sim.max(axis=1)

        # Image (mean cos)
        hist_i = image[hist_arr]
        cand_i = image[cand_idx]
        i_sim = cand_i @ hist_i.T
        row[:, 12] = i_sim.max(axis=1)

        # Artist/album signals
        hist_artists = artist_code[hist_arr]
        hist_albums = album_code[hist_arr]
        cand_artists = artist_code[cand_idx]
        cand_albums = album_code[cand_idx]
        row[:, 13] = np.isin(cand_artists, hist_artists[hist_artists > 0]).astype(np.float32)
        row[:, 14] = np.isin(cand_albums, hist_albums[hist_albums > 0]).astype(np.float32)
        last_art = int(artist_code[last_h])
        last_alb = int(album_code[last_h])
        row[:, 15] = ((cand_artists == last_art) & (last_art > 0)).astype(np.float32)
        row[:, 16] = ((cand_albums == last_alb) & (last_alb > 0)).astype(np.float32)

        # Tag jaccard
        hist_bin = tag_bin[hist_arr]
        hist_union_count = _as_1d_float(hist_bin.sum(axis=0))
        hist_ids = np.flatnonzero(hist_union_count > 0)
        if len(hist_ids) > 0:
            cand_bin = tag_bin[cand_idx]
            overlap = _as_1d_float(cand_bin[:, hist_ids].sum(axis=1))
            denom = tag_count[cand_idx] + float(len(hist_ids)) - overlap
            row[:, 17] = np.divide(overlap, denom, out=np.zeros_like(overlap), where=denom > 0)
        last_tag_ids = np.flatnonzero(_as_1d_float(tag_bin[last_h]))
        if len(last_tag_ids) > 0:
            cand_bin = tag_bin[cand_idx]
            overlap_last = _as_1d_float(cand_bin[:, last_tag_ids].sum(axis=1))
            denom_last = tag_count[cand_idx] + float(len(last_tag_ids)) - overlap_last
            row[:, 18] = np.divide(overlap_last, denom_last, out=np.zeros_like(overlap_last), where=denom_last > 0)

        # Release year / duration absdiff vs last
        last_year = float(release_year[last_h])
        cand_year = release_year[cand_idx]
        row[:, 19] = np.where((last_year > 0) & (cand_year > 0), np.abs(cand_year - last_year), 0.0)
        last_dur = float(duration[last_h])
        cand_dur = duration[cand_idx]
        row[:, 20] = np.where((last_dur > 0) & (cand_dur > 0), np.abs(cand_dur - last_dur) / 60000.0, 0.0)

        # Multimodal max mean (mean of CF/metadata/attributes/audio max)
        row[:, 21] = (row[:, 2] + row[:, 5] + row[:, 7] + row[:, 9]) / 4.0

        offset += m
    if offset != len(extra):
        raise RuntimeError(f"extra pool prior feature length mismatch offset={offset} total={len(extra)}")
    out = np.empty((x_base.shape[0], x_base.shape[1] + n_extra), dtype=np.float32)
    out[:, : x_base.shape[1]] = x_base
    out[:, x_base.shape[1] :] = extra
    return out, list(EXTRA_POOL_PRIOR_FEATURE_NAMES)


def tag_chain_feature_context(
    track_index: Any,
    *,
    top_neighbors: int = 20,
    max_vocab: int = 4096,
    max_track_tokens: int = 64,
) -> dict[str, Any]:
    cache_key = id(track_index)
    cached = _TAG_CHAIN_CONTEXT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    token_sets = [
        {
            str(tok)
            for tok in tokens
            if len(str(tok)) >= 2 and str(tok).lower() not in TAG_STOPWORDS
        }
        for tokens in track_index.tag_tokens
    ]
    df: Counter[str] = Counter()
    for toks in token_sets:
        df.update(toks)
    vocab = {
        tok: i
        for i, (tok, _) in enumerate(
            sorted(df.items(), key=lambda item: (-item[1], item[0]))[:max_vocab]
        )
    }
    n_tracks = max(1, len(token_sets))
    df_id = np.zeros(len(vocab), dtype=np.int32)
    for tok, idx in vocab.items():
        df_id[idx] = int(df[tok])
    rows = []
    cols = []
    token_id_rows = []
    for row, toks in enumerate(token_sets):
        ids = sorted({vocab[tok] for tok in toks if tok in vocab}, key=lambda i: (-df_id[i], i))
        ids = ids[:max_track_tokens]
        ids.sort()
        token_id_rows.append(ids)
        rows.extend([row] * len(ids))
        cols.extend(ids)
    data = np.ones(len(rows), dtype=np.float32)
    tag_bin = sparse.csr_matrix((data, (rows, cols)), shape=(n_tracks, len(vocab)), dtype=np.float32)
    tag_count = np.asarray(tag_bin.getnnz(axis=1), dtype=np.float32)
    idf = (np.log((1.0 + float(n_tracks)) / (1.0 + df_id.astype(np.float32))) + 1.0).astype(np.float32)
    tag_tfidf = tag_bin.copy().astype(np.float32)
    tag_tfidf.data *= idf[tag_tfidf.indices]
    tag_tfidf = sparse_normalize(tag_tfidf, norm="l2", axis=1, copy=False)

    pair: Counter[tuple[int, int]] = Counter()
    for ids in token_id_rows:
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                pair[(a, b)] += 1
    neighbors_tmp: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for (a, b), cnt in pair.items():
        denom = float(df_id[a]) * float(df_id[b])
        if denom <= 0:
            continue
        val = math.log((float(cnt) * float(n_tracks)) / denom)
        if val <= 0:
            continue
        neighbors_tmp[a].append((b, val))
        neighbors_tmp[b].append((a, val))
    adj_rows: list[int] = []
    adj_cols: list[int] = []
    adj_data: list[float] = []
    for tok, vals in neighbors_tmp.items():
        for nb, val in sorted(vals, key=lambda x: (-x[1], x[0]))[:top_neighbors]:
            adj_rows.append(tok)
            adj_cols.append(nb)
            adj_data.append(float(val))
    adj_ppmi = sparse.csr_matrix(
        (np.asarray(adj_data, dtype=np.float32), (adj_rows, adj_cols)),
        shape=(len(vocab), len(vocab)),
        dtype=np.float32,
    )
    adj_binary = adj_ppmi.copy()
    adj_binary.data = np.ones_like(adj_binary.data, dtype=np.float32)
    ctx = {
        "tag_bin": tag_bin,
        "tag_tfidf": tag_tfidf,
        "tag_count": tag_count,
        "adj_ppmi": adj_ppmi,
        "adj_binary": adj_binary,
        "vocab": vocab,
    }
    _TAG_CHAIN_CONTEXT_CACHE[cache_key] = ctx
    return ctx


def _as_1d_float(value: Any) -> np.ndarray:
    if sparse.issparse(value):
        return np.asarray(value.toarray(), dtype=np.float32).ravel()
    return np.asarray(value, dtype=np.float32).ravel()


def append_extra_tag_chain_features(
    x_base: np.ndarray,
    examples: list[Any],
    candidates: Any,
    track_index: Any,
    *,
    width: int,
    enabled: bool,
) -> tuple[np.ndarray, list[str]]:
    if not enabled:
        return x_base, []
    indices = candidates.indices[:, :width]
    valid_mask = indices >= 0
    n_extra = len(EXTRA_TAG_CHAIN_FEATURE_NAMES)
    extra = np.zeros((int(valid_mask.sum()), n_extra), dtype=np.float32)
    ctx = tag_chain_feature_context(track_index)
    tag_bin = ctx["tag_bin"]
    tag_tfidf = ctx["tag_tfidf"]
    tag_count = ctx["tag_count"]
    adj_ppmi = ctx["adj_ppmi"]
    adj_binary = ctx["adj_binary"]
    offset = 0
    for row_i, ex in enumerate(examples):
        pos = np.flatnonzero(valid_mask[row_i])
        if len(pos) == 0:
            continue
        cand_idx = indices[row_i, pos].astype(np.int32, copy=False)
        m = len(cand_idx)
        row = extra[offset : offset + m]
        hist_idx: list[int] = []
        for msg in ex.chat_history:
            if msg.get("role") != "music":
                continue
            idx = track_index.id_to_idx.get(str(msg.get("content") or ""))
            if idx is not None:
                hist_idx.append(idx)
        if hist_idx:
            hist_counts = _as_1d_float(tag_bin[hist_idx].sum(axis=0))
            hist_ids = np.flatnonzero(hist_counts > 0.0)
            if len(hist_ids):
                cand_bin = tag_bin[cand_idx]
                overlap = _as_1d_float(cand_bin[:, hist_ids].sum(axis=1))
                union = tag_count[cand_idx] + float(len(hist_ids)) - overlap
                row[:, 0] = overlap
                row[:, 1] = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)

                hist_tfidf = _as_1d_float(tag_tfidf[hist_idx].sum(axis=0))
                norm = float(np.linalg.norm(hist_tfidf))
                if norm > 0.0:
                    hist_tfidf /= norm
                    hist_tfidf_sparse = sparse.csr_matrix(hist_tfidf.reshape(1, -1))
                    row[:, 2] = _as_1d_float(tag_tfidf[cand_idx].dot(hist_tfidf_sparse.T))

                hist_sparse = sparse.csr_matrix(
                    (np.ones(len(hist_ids), dtype=np.float32), ([0] * len(hist_ids), hist_ids)),
                    shape=(1, tag_bin.shape[1]),
                    dtype=np.float32,
                )
                expanded_ppmi = hist_sparse.dot(adj_ppmi)
                expanded_binary = hist_sparse.dot(adj_binary)
                expanded_binary.data = np.ones_like(expanded_binary.data, dtype=np.float32)
                row[:, 3] = _as_1d_float(cand_bin.dot(expanded_binary.T))
                row[:, 4] = _as_1d_float(cand_bin.dot(expanded_ppmi.T))
                weighted = cand_bin.multiply(expanded_ppmi)
                row[:, 5] = _as_1d_float(weighted.max(axis=1))
        offset += m
    if offset != len(extra):
        raise RuntimeError(f"extra tag chain feature length mismatch offset={offset} total={len(extra)}")
    out = np.empty((x_base.shape[0], x_base.shape[1] + n_extra), dtype=np.float32)
    out[:, : x_base.shape[1]] = x_base
    out[:, x_base.shape[1] :] = extra
    return out, list(EXTRA_TAG_CHAIN_FEATURE_NAMES)


def listener_goal_text(ex: Any) -> str:
    goal = dict(ex.conversation_goal or {})
    parts = [
        str(goal.get("listener_goal") or "").strip(),
        f"category: {goal.get('category')}" if goal.get("category") else "",
        f"specificity: {goal.get('specificity')}" if goal.get("specificity") else "",
    ]
    text = "\n".join(p for p in parts if p)
    return text or str(ex.user_query or "").strip() or "music recommendation"


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return x / denom


def load_or_encode_listener_goal_embeddings(
    examples: list[Any],
    *,
    cache_path: Path,
    batch_size: int,
) -> np.ndarray:
    cache_path = cache_path if cache_path.is_absolute() else REPO_ROOT / cache_path
    keys = [row_key_str(ex) for ex in examples]
    cache_keys: list[str] = []
    cache_emb: np.ndarray | None = None
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as data:
            cache_keys = [str(x) for x in data["keys"]]
            cache_emb = np.asarray(data["embeddings"], dtype=np.float32)
    pos = {k: i for i, k in enumerate(cache_keys)}
    missing = [i for i, k in enumerate(keys) if k not in pos]
    if missing:
        from recsys2026.encoders import Qwen3TextEncoder

        print(f"  encoding listener_goal embeddings rows={len(missing)} cache={rel(cache_path)}")
        encoder = Qwen3TextEncoder(batch_size=batch_size)
        texts = [listener_goal_text(examples[i]) for i in missing]
        new_emb = _normalize_rows(encoder.encode(texts))
        new_keys = [keys[i] for i in missing]
        if cache_emb is None:
            cache_keys = new_keys
            cache_emb = new_emb
        else:
            cache_keys = cache_keys + new_keys
            cache_emb = np.concatenate([cache_emb, new_emb.astype(np.float32, copy=False)], axis=0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, keys=np.asarray(cache_keys), embeddings=cache_emb.astype(np.float32, copy=False))
        pos = {k: i for i, k in enumerate(cache_keys)}
    if cache_emb is None:
        raise RuntimeError(f"listener_goal embedding cache is empty: {cache_path}")
    return _normalize_rows(cache_emb[[pos[k] for k in keys]])


def fit_goal_cluster_context(
    examples: list[Any],
    *,
    enabled: bool,
    n_clusters: int,
    cache_path: Path,
    batch_size: int,
    seed: int,
) -> dict[str, Any] | None:
    if not enabled:
        return None
    emb = load_or_encode_listener_goal_embeddings(examples, cache_path=cache_path, batch_size=batch_size)
    n = max(2, min(int(n_clusters), len(examples)))
    clusterer = MiniBatchKMeans(
        n_clusters=n,
        random_state=seed,
        batch_size=min(8192, max(1024, len(examples))),
        n_init=3,
        max_iter=100,
        reassignment_ratio=0.01,
    )
    print(f"fit listener_goal MiniBatchKMeans rows={len(examples)} clusters={n}")
    clusterer.fit(emb)
    return {
        "clusterer": clusterer,
        "cache_path": cache_path,
        "batch_size": int(batch_size),
    }


def append_extra_goal_cluster_features(
    x_base: np.ndarray,
    examples: list[Any],
    candidates: Any,
    *,
    width: int,
    context: dict[str, Any] | None,
) -> tuple[np.ndarray, list[str]]:
    if context is None:
        return x_base, []
    indices = candidates.indices[:, :width]
    valid_mask = indices >= 0
    n_extra = len(EXTRA_GOAL_CLUSTER_FEATURE_NAMES)
    extra = np.zeros((int(valid_mask.sum()), n_extra), dtype=np.float32)
    emb = load_or_encode_listener_goal_embeddings(
        examples,
        cache_path=Path(context["cache_path"]),
        batch_size=int(context["batch_size"]),
    )
    clusterer = context["clusterer"]
    cluster_id = clusterer.predict(emb).astype(np.float32)
    dist = clusterer.transform(emb).min(axis=1).astype(np.float32)
    conf = (1.0 / (1.0 + dist)).astype(np.float32)
    offset = 0
    for row_i in range(len(examples)):
        m = int(valid_mask[row_i].sum())
        if m == 0:
            continue
        sl = slice(offset, offset + m)
        extra[sl, 0] = cluster_id[row_i]
        extra[sl, 1] = dist[row_i]
        extra[sl, 2] = conf[row_i]
        offset += m
    if offset != len(extra):
        raise RuntimeError(f"extra goal cluster feature length mismatch offset={offset} total={len(extra)}")
    out = np.empty((x_base.shape[0], x_base.shape[1] + n_extra), dtype=np.float32)
    out[:, : x_base.shape[1]] = x_base
    out[:, x_base.shape[1] :] = extra
    return out, list(EXTRA_GOAL_CLUSTER_FEATURE_NAMES)


def fit_feature_stack(
    legacy: Any,
    train_examples: list[Any],
    train_candidates: Any,
    train_dense: np.ndarray,
    track_index: Any,
    user_vectors: dict[str, np.ndarray],
    *,
    feature_chunk_examples: int,
    n_bm25_for_dense_flag: int | None,
    extra_goal_cluster_features: bool,
    extra_category_turn_features: bool,
    goal_cluster_n_clusters: int,
    goal_cluster_cache_path: Path,
    goal_cluster_batch_size: int,
    goal_cluster_seed: int,
    fit_examples: list[Any] | None = None,
) -> tuple[Any, TfidfVectorizer, Any, Any, Any, dict[str, Any] | None, dict[str, Any] | None]:
    examples_for_fit = fit_examples if fit_examples is not None else train_examples
    encoder = legacy.FeatureEncoder(track_index, user_vectors)
    encoder.fit_categories(examples_for_fit)
    vectorizer = TfidfVectorizer(
        min_df=2,
        max_features=120_000,
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
    )
    text_corpus = (
        track_index.texts
        + [legacy.goal_text(ex.conversation_goal) for ex in examples_for_fit]
        + [legacy.conversation_text(ex, track_index) for ex in examples_for_fit]
        + [legacy._query_plus_goal(ex.user_query, ex.conversation_goal, ex.user_query_thought) for ex in examples_for_fit]
    )
    print(f"fit text vectorizer rows={len(examples_for_fit)}")
    vectorizer.fit(text_corpus)
    track_tfidf = vectorizer.transform(track_index.texts)
    fast_context = fast098.make_fast_context(legacy, encoder)
    goal_cluster_context = fit_goal_cluster_context(
        examples_for_fit,
        enabled=extra_goal_cluster_features,
        n_clusters=goal_cluster_n_clusters,
        cache_path=goal_cluster_cache_path,
        batch_size=goal_cluster_batch_size,
        seed=goal_cluster_seed,
    )
    category_turn_context = fit_category_turn_context(
        examples_for_fit,
        track_index,
        enabled=extra_category_turn_features,
    )
    return encoder, vectorizer, track_tfidf, fast_context, n_bm25_for_dense_flag, goal_cluster_context, category_turn_context


def build_rich_matrix(
    legacy: Any,
    examples: list[Any],
    candidates: Any,
    dense_q: np.ndarray,
    encoder: Any,
    vectorizer: TfidfVectorizer,
    track_tfidf: Any,
    track_index: Any,
    fast_context: Any,
    candidate_dir: Path,
    source_rows: np.ndarray,
    *,
    width: int,
    feature_chunk_examples: int,
    n_bm25_for_dense_flag: int | None,
    source_features_enabled: bool,
    drop_cross_source_score_meta: bool,
    extra_source_score_transforms: bool,
    extra_metadata_features: bool,
    extra_gpa_features: bool,
    extra_reaction_features: bool,
    extra_assistant_thought_features: bool,
    extra_tag_chain_features: bool,
    extra_hier_pop_features: bool,
    extra_pool_prior_features: bool,
    extra_talkplay_aux_features: bool,
    goal_cluster_context: dict[str, Any] | None,
    category_turn_context: dict[str, Any] | None,
    score_calibration_context: dict[str, Any] | None,
    extra_candidate_feature_npz: list[Path],
    neutralize_098_features: set[str],
    labels: bool,
) -> tuple[np.ndarray, np.ndarray | None, list[int], list[str]]:
    x_base, y, groups = fast098.build_feature_matrix_fast(
        legacy,
        examples,
        candidates,
        encoder,
        vectorizer,
        track_tfidf,
        negatives_per_group=None,
        chunk_examples=feature_chunk_examples,
        query_dense_emb=dense_q,
        n_bm25=n_bm25_for_dense_flag,
        intent_lookup=None,
        fast_context=fast_context,
    )
    if neutralize_098_features:
        for col, name in enumerate(encoder.feature_names):
            if name in neutralize_098_features:
                x_base[:, col] = 0.0
    if not labels:
        y = None
    valid_mask = candidates.indices[:, :width] >= 0
    x_meta, metadata_names = append_extra_metadata_features(
        x_base,
        legacy,
        examples,
        candidates,
        track_index,
        width=width,
        enabled=extra_metadata_features,
    )
    x_extra, feedback_names = append_extra_feedback_features(
        x_meta,
        examples,
        candidates,
        track_index,
        width=width,
        gpa_enabled=extra_gpa_features,
        reaction_enabled=extra_reaction_features,
    )
    x_goal, goal_cluster_names = append_extra_goal_cluster_features(
        x_extra,
        examples,
        candidates,
        width=width,
        context=goal_cluster_context,
    )
    x_thought, thought_names = append_extra_assistant_thought_features(
        x_goal,
        examples,
        candidates,
        width=width,
        enabled=extra_assistant_thought_features,
    )
    x_pop, hier_pop_names = append_extra_hier_pop_features(
        x_thought,
        candidates,
        track_index,
        width=width,
        enabled=extra_hier_pop_features,
    )
    x_bucket, category_turn_names = append_extra_category_turn_features(
        x_pop,
        examples,
        candidates,
        track_index,
        width=width,
        context=category_turn_context,
    )
    x_tag, tag_chain_names = append_extra_tag_chain_features(
        x_bucket,
        examples,
        candidates,
        track_index,
        width=width,
        enabled=extra_tag_chain_features,
    )
    x_pool, pool_prior_names = append_extra_pool_prior_features(
        x_tag,
        examples,
        candidates,
        track_index,
        width=width,
        enabled=extra_pool_prior_features,
    )
    x_aux, aux_names = append_extra_talkplay_aux_features(
        x_pool,
        examples,
        candidates,
        track_index,
        width=width,
        enabled=extra_talkplay_aux_features,
    )
    x_cal, calibration_names = append_extra_score_calibration_features(
        x_aux,
        candidate_dir,
        source_rows,
        valid_mask,
        width=width,
        context=score_calibration_context,
    )
    x_ext, ext_names = append_extra_candidate_feature_npz(
        x_cal,
        extra_candidate_feature_npz,
        source_rows,
        valid_mask,
        width=width,
    )
    x, source_names = append_source_features(
        x_ext,
        candidate_dir,
        source_rows,
        valid_mask,
        width=width,
        enabled=source_features_enabled,
        drop_cross_source_score_meta=drop_cross_source_score_meta,
        extra_score_transforms=extra_source_score_transforms,
    )
    return x, y, groups, metadata_names + feedback_names + goal_cluster_names + thought_names + hier_pop_names + category_turn_names + tag_chain_names + pool_prior_names + aux_names + calibration_names + ext_names + source_names


def categorical_indices_for(feature_names: list[str], base_indices: list[int]) -> list[int]:
    out = list(base_indices)
    for i, name in enumerate(feature_names):
        if name == "extra_goal_cluster_id":
            out.append(i)
    return out


def positive_eval_mask_rows(
    public_sources: list[str],
    public_examples: list[Any],
    rows: np.ndarray,
) -> np.ndarray:
    """Filter rows to positive_eval mask (turn==1 OR target GPA == MOVES_TOWARD_GOAL).

    Reads artifacts/cache/positive_eval_mask.npz.
    """
    mask_path = CACHE_DIR / "positive_eval_mask.npz"
    if not mask_path.exists():
        raise FileNotFoundError(f"positive_eval mask not found: {mask_path}. Run scripts/build_positive_eval_mask.py first.")
    mask_data = np.load(mask_path, allow_pickle=False)
    mask_src = mask_data["source_split"].astype(str)
    mask_sid = mask_data["session_id"].astype(str)
    mask_turn = mask_data["turn_number"]
    mask_pos = mask_data["positive_eval_mask"]
    key_to_pos: dict[str, bool] = {}
    for i in range(len(mask_turn)):
        key_to_pos[f"{mask_src[i]}:{mask_sid[i]}:{int(mask_turn[i])}"] = bool(mask_pos[i])

    keep_rows: list[int] = []
    for r in rows:
        r_int = int(r)
        ex = public_examples[r_int]
        src = public_sources[r_int]
        key = f"{src}:{ex.session_id}:{int(ex.turn_number)}"
        if key_to_pos.get(key, False):
            keep_rows.append(r_int)
    return np.asarray(keep_rows, dtype=np.int32)


def positive_rows(examples: list[Any], indices: np.ndarray, rows: np.ndarray, track_index: Any) -> np.ndarray:
    keep: list[int] = []
    for row_raw in rows:
        row = int(row_raw)
        gold_idx = track_index.id_to_idx.get(examples[row].gold_track_id or "")
        if gold_idx is not None and bool(np.any(indices[row] == gold_idx)):
            keep.append(row)
    return np.asarray(keep, dtype=np.int32)


def positive_group_weights(examples: list[Any], positive_weight: float, negative_weight: float = 0.5) -> np.ndarray:
    """P5-C: per-example weight. turn1 or positive-proxy => positive_weight, else negative_weight.

    Aligns 1:1 with build_feature_matrix_fast group order (one group per example).
    """
    w = np.full(len(examples), negative_weight, dtype=np.float32)
    for i, ex in enumerate(examples):
        if int(ex.turn_number) == 1:
            w[i] = 1.0
            continue
        cur_text = f"{ex.user_query or ''} {getattr(ex, 'user_query_thought', '') or ''}"
        pos, _, _ = reaction_flags(cur_text)
        if pos > 0:
            w[i] = positive_weight
    return w


def textproxy_positive_rows(public_examples: list[Any], rows: np.ndarray) -> np.ndarray:
    """P5-A: keep rows where turn==1 OR current text matches positive-reaction regex proxy.

    Uses only current user text/thought (no true target GPA), so strict-safe.
    """
    keep: list[int] = []
    for r in rows:
        r_int = int(r)
        ex = public_examples[r_int]
        if int(ex.turn_number) == 1:
            keep.append(r_int)
            continue
        cur_text = f"{ex.user_query or ''} {getattr(ex, 'user_query_thought', '') or ''}"
        pos, _, _ = reaction_flags(cur_text)
        if pos > 0:
            keep.append(r_int)
    return np.asarray(keep, dtype=np.int32)


def rank_from_predictions(candidates: Any, groups: list[int], pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    width = candidates.indices.shape[1]
    ranked = np.full(candidates.indices.shape, -1, dtype=np.int32)
    scores = np.full(candidates.indices.shape, np.nan, dtype=np.float32)
    offset = 0
    for i, group_size in enumerate(groups):
        size = int(group_size)
        row_scores = pred[offset : offset + size]
        order = np.argsort(-row_scores, kind="stable")
        ranked[i, :size] = candidates.indices[i, :size][order]
        scores[i, :size] = row_scores[order]
        offset += size
    if offset != len(pred):
        raise RuntimeError("prediction/group length mismatch")
    return ranked[:, :width], scores[:, :width]


def binary_group_weights(groups: list[int]) -> np.ndarray:
    weights = np.empty(int(sum(groups)), dtype=np.float32)
    offset = 0
    for group_size_raw in groups:
        group_size = int(group_size_raw)
        if group_size <= 1:
            weights[offset : offset + group_size] = 1.0
        else:
            weights[offset] = 0.5
            weights[offset + 1 : offset + group_size] = 0.5 / float(group_size - 1)
        offset += group_size
    return weights


def group_qid(groups: list[int]) -> np.ndarray:
    return np.repeat(np.arange(len(groups), dtype=np.int32), np.asarray(groups, dtype=np.int32))


def fit_lgbm_model(
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    groups: list[int],
    *,
    fold_seed: int,
    categorical_feature: list[int],
    feature_names: list[str],
    group_weights: np.ndarray | None = None,
) -> Any:
    # P5-C: expand per-group weights to per-row sample_weight
    sample_weight = None
    if group_weights is not None:
        sample_weight = np.repeat(np.asarray(group_weights, dtype=np.float32), groups)
    model_family = str(getattr(args, "model_family", "lightgbm"))
    is_binary = args.lgbm_objective == "binary"

    if model_family == "xgboost":
        if xgb is None:
            raise RuntimeError("xgboost is not installed")
        common = dict(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            min_child_weight=args.min_child_weight,
            reg_lambda=args.reg_lambda,
            max_bin=args.max_bin,
            tree_method="hist",
            device=args.xgb_device,
            random_state=fold_seed,
            n_jobs=args.n_jobs,
            verbosity=1,
        )
        if is_binary:
            model = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
            model.fit(x, y, sample_weight=binary_group_weights(groups))
            return model
        model = xgb.XGBRanker(objective=args.xgb_rank_objective, eval_metric="ndcg@20", **common)
        model.fit(x, y, group=groups)
        return model

    if model_family == "catboost":
        if cb is None:
            raise RuntimeError("catboost is not installed")
        thread_count = args.n_jobs if args.n_jobs > 0 else -1
        common = dict(
            iterations=args.n_estimators,
            learning_rate=args.learning_rate,
            random_seed=fold_seed,
            thread_count=thread_count,
            task_type=args.catboost_task_type,
            devices=args.catboost_devices,
            allow_writing_files=False,
            verbose=50,
        )
        if is_binary:
            model = cb.CatBoostClassifier(
                loss_function="Logloss",
                depth=args.max_depth,
                l2_leaf_reg=args.reg_lambda,
                **common,
            )
            model.fit(cb.Pool(x, y, weight=binary_group_weights(groups)))
            return model
        model = cb.CatBoostRanker(
            loss_function=args.catboost_loss,
            eval_metric="NDCG:top=20",
            depth=args.max_depth,
            l2_leaf_reg=args.reg_lambda,
            **common,
        )
        model.fit(cb.Pool(x, y, group_id=group_qid(groups)))
        return model

    if model_family != "lightgbm":
        raise ValueError(f"unknown model_family={model_family}")

    if args.lgbm_objective == "binary":
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            min_child_samples=args.min_child_samples,
            random_state=fold_seed,
            n_jobs=args.n_jobs,
            verbose=-1,
        )
        model.fit(
            x,
            y,
            sample_weight=None if args.binary_no_weight else binary_group_weights(groups),
            categorical_feature=categorical_feature,
            feature_name=feature_names,
        )
        return model

    model = lgb.LGBMRanker(
        objective=args.lgbm_objective,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_samples=args.min_child_samples,
        random_state=fold_seed,
        n_jobs=args.n_jobs,
        verbose=-1,
    )
    if args.lambdarank_truncation_level is not None:
        model.set_params(lambdarank_truncation_level=args.lambdarank_truncation_level)
    model.fit(
        x, y, group=groups, sample_weight=sample_weight,
        categorical_feature=categorical_feature, feature_name=feature_names,
    )
    return model


def predict_lgbm_model(model: Any, x: np.ndarray) -> np.ndarray:
    if cb is not None and isinstance(model, (cb.CatBoostClassifier, cb.CatBoostRanker)):
        if isinstance(model, cb.CatBoostClassifier):
            return np.asarray(model.predict_proba(cb.Pool(x))[:, 1], dtype=np.float32)
        return np.asarray(model.predict(cb.Pool(x)), dtype=np.float32)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].astype(np.float32, copy=False)
    return model.predict(x).astype(np.float32, copy=False)


def save_tree_model(model: Any, path_stem: Path) -> Path:
    if cb is not None and isinstance(model, (cb.CatBoostClassifier, cb.CatBoostRanker)):
        path = path_stem.with_suffix(".cbm")
        model.save_model(str(path))
        return path
    if xgb is not None and isinstance(model, (xgb.XGBClassifier, xgb.XGBRanker)):
        path = path_stem.with_suffix(".ubj")
        model.save_model(str(path))
        return path
    path = path_stem.with_suffix(".txt")
    model.booster_.save_model(str(path))
    return path


def subset_list(values: list[Any], rows: np.ndarray) -> list[Any]:
    return [values[int(i)] for i in rows]


def load_spotify_to_track_idx(mapping_path: Path) -> dict[str, int]:
    import pyarrow.parquet as pq

    table = pq.read_table(mapping_path, columns=["spotify_id", "track_idx"])
    spotify = table.column("spotify_id").to_pylist()
    idx = table.column("track_idx").to_pylist()
    return {str(s): int(i) for s, i in zip(spotify, idx, strict=True)}


def tpd1_turn_messages_for_history(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(m) for m in history]


def load_tpd1_mix_examples(
    legacy: Any,
    track_index: Any,
    *,
    mapping_path: Path,
    max_examples: int,
    seed: int,
) -> tuple[list[Any], dict[str, Any]]:
    """Reservoir-sample TPD1 mapped music turns as BlindB-safe reranker groups."""
    from datasets import load_dataset

    if max_examples <= 0:
        raise ValueError("--tpd1-mix-max-examples must be positive when TPD1 mix is enabled")
    mapping = load_spotify_to_track_idx(mapping_path)
    rng = random.Random(seed)
    sample: list[Any] = []
    seen = 0
    n_rows = 0
    n_music = 0
    n_mapped = 0
    n_without_user = 0
    ds = load_dataset("talkpl-ai/TalkPlayData-1", split="train")
    for row_idx, item in enumerate(ds):
        n_rows += 1
        cid = str(item.get("cid") or row_idx)
        pid = str(item.get("pid") or "")
        history: list[dict[str, Any]] = []
        pending_user: dict[str, Any] | None = None
        turn_number = 0
        for conv_idx, turn in enumerate(item.get("conversations") or []):
            role = str(turn.get("role") or "")
            content = str(turn.get("content") or "")
            if role == "user":
                pending_user = {
                    "role": "user",
                    "content": content,
                    "turn_number": turn_number + 1,
                }
                continue
            if role == "music":
                n_music += 1
                turn_number += 1
                mapped_idx = mapping.get(content)
                gold_tid = track_index.track_ids[int(mapped_idx)] if mapped_idx is not None else None
                if pending_user is None:
                    n_without_user += 1
                elif gold_tid is not None:
                    n_mapped += 1
                    ex = legacy.TurnExample(
                        session_id=f"tpd1_{cid}",
                        user_id=f"tpd1_{pid}" if pid else "tpd1",
                        turn_number=turn_number,
                        user_profile={},
                        conversation_goal={},
                        chat_history=tpd1_turn_messages_for_history(history),
                        user_query=str(pending_user.get("content") or ""),
                        user_query_thought="",
                        prior_goal_progress=[],
                        gold_track_id=gold_tid,
                    )
                    seen += 1
                    if len(sample) < max_examples:
                        sample.append(ex)
                    else:
                        j = rng.randrange(seen)
                        if j < max_examples:
                            sample[j] = ex
                if pending_user is not None:
                    history.append(dict(pending_user))
                if gold_tid is not None:
                    history.append({"role": "music", "content": gold_tid, "turn_number": turn_number})
                pending_user = None
                continue
            if role == "assistant":
                if turn_number > 0:
                    history.append({"role": "assistant", "content": content, "turn_number": turn_number})

    stats = {
        "tpd1_rows": n_rows,
        "music_turns": n_music,
        "mapped_music_turns": n_mapped,
        "music_turns_without_user": n_without_user,
        "sampled_examples": len(sample),
        "max_examples": max_examples,
        "seed": seed,
        "mapping": rel(mapping_path),
        "sample_policy": "reservoir_over_mapped_music_turns",
        "blind_b_safe_fields": True,
    }
    print(f"TPD1 reranker mix examples: {json.dumps(stats, ensure_ascii=False)}")
    return sample, stats


def build_tpd1_mix_candidates(
    legacy: Any,
    examples: list[Any],
    track_index: Any,
    *,
    candidate_k: int,
    seed: int,
    cache_name: str,
) -> Any:
    generated = legacy.generate_candidates(
        examples,
        track_index,
        candidate_k=candidate_k,
        artist_boost=50.0,
        album_boost=30.0,
        exclude_history=True,
        cache_name=cache_name,
        use_cache=True,
        desc=f"tpd1_mix_cand[{cache_name}]",
        dense_query_emb=None,
        n_bm25=None,
    )
    indices = np.asarray(generated.indices, dtype=np.int32).copy()
    rng = random.Random(seed + 17)
    inserted = 0
    already = 0
    for i, ex in enumerate(examples):
        gold_idx = track_index.id_to_idx.get(ex.gold_track_id or "")
        if gold_idx is None:
            continue
        if np.any(indices[i] == int(gold_idx)):
            already += 1
            continue
        replace_pos = rng.randrange(indices.shape[1])
        indices[i, replace_pos] = int(gold_idx)
        inserted += 1
    # Keep candidate_score neutral. Public cooc500 best config also uses primary_score_mode=zero.
    scores = np.zeros(indices.shape, dtype=np.float32)
    print(
        "TPD1 reranker mix candidates: "
        f"groups={len(examples)} k={indices.shape[1]} gold_already={already} gold_inserted={inserted}"
    )
    return legacy.CandidateSet(indices=indices, scores=scores)


def append_zero_columns_to_match(
    x: np.ndarray,
    names: list[str],
    target_names: list[str],
) -> np.ndarray:
    if names == target_names:
        return x
    if target_names[: len(names)] != names:
        raise RuntimeError(
            "TPD1 feature schema prefix mismatch: "
            f"tpd1_names_tail={names[-5:]} target_prefix_tail={target_names[:len(names)][-5:]}"
        )
    missing = len(target_names) - len(names)
    if missing < 0:
        raise RuntimeError(f"TPD1 feature schema has extra columns: {len(names)} > {len(target_names)}")
    if missing == 0:
        return x
    zeros = np.zeros((x.shape[0], missing), dtype=np.float32)
    return np.hstack([x, zeros])


def save_ranked_public(
    out_dir: Path,
    keys: list[str],
    sources: list[str],
    examples: list[Any],
    folds: np.ndarray,
    ranked: np.ndarray,
    scores: np.ndarray,
    manifest: dict[str, Any],
    track_ids: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = np.asarray([(row >= 0).sum() for row in ranked], dtype=np.int32)
    np.savez_compressed(
        out_dir / "ranked.npz",
        track_idx=ranked.astype(np.int32, copy=False),
        sizes=sizes,
        keys=np.asarray([k.encode("utf-8") for k in keys], dtype="S128"),
        source_split=np.asarray([s.encode("utf-8") for s in sources], dtype="S8"),
        folds=folds.astype(np.int16, copy=False),
        scores=scores.astype(np.float32, copy=False),
    )
    with (out_dir / "ranked_top100.jsonl").open("w", encoding="utf-8") as f:
        for src, ex, row in zip(sources, examples, ranked, strict=True):
            f.write(
                json.dumps(
                    {
                        "source_split": src,
                        "session_id": ex.session_id,
                        "turn_number": int(ex.turn_number),
                        "ranked_track_ids": [track_ids[int(i)] for i in row[:100] if int(i) >= 0],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    json_dump(out_dir / "manifest.json", manifest)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="protocol_098_union_rich_lgbm")
    parser.add_argument("--config", required=True)
    parser.add_argument("--public-candidates", type=Path, required=True)
    parser.add_argument("--blind-candidates", type=Path, default=None)
    parser.add_argument("--blind-target", choices=("blind_a", "blind_b"), default="blind_a")
    parser.add_argument("--max-candidates", type=int, default=500, help="Candidate width to read; <=0 means use the full artifact width.")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--cv-folds", default="0", help="Comma-separated fold ids or 'all'.")
    parser.add_argument("--cv-artifact-mode", default="cv3_oof", help="Public OOF artifact mode for CV outputs and final-model fit candidates.")
    parser.add_argument("--skip-cv", action="store_true", help="Skip CV ranking and only fit one final model for blind ranking.")
    parser.add_argument(
        "--merge-existing-cv",
        action="store_true",
        help=(
            "When running a fold subset, merge the newly computed folds with an existing "
            "OOF public_labeled ranked artifact for the same config instead of "
            "overwriting it with only the current subset."
        ),
    )
    parser.add_argument("--primary-score-mode", choices=("zero", "bm25", "max_source"), default="bm25")
    parser.add_argument("--disable-source-features", action="store_true")
    parser.add_argument(
        "--drop-cross-source-score-meta",
        action="store_true",
        help="Drop meta__max_source_score__primary from source features. Default keeps legacy baseline behavior.",
    )
    parser.add_argument(
        "--neutralize-098-features",
        default="",
        help="Comma-separated base 098 feature names to set to zero after building the feature matrix.",
    )
    parser.add_argument("--extra-source-score-transforms", action="store_true")
    parser.add_argument("--extra-metadata-features", action="store_true")
    parser.add_argument("--extra-feedback-features", action="store_true")
    parser.add_argument("--extra-gpa-features", action="store_true")
    parser.add_argument("--extra-reaction-features", action="store_true")
    parser.add_argument("--extra-goal-cluster-features", action="store_true")
    parser.add_argument("--extra-assistant-thought-features", action="store_true")
    parser.add_argument("--extra-tag-chain-features", action="store_true")
    parser.add_argument("--extra-hier-pop-features", action="store_true")
    parser.add_argument("--extra-pool-prior-features", action="store_true")
    parser.add_argument("--extra-talkplay-aux-features", action="store_true")
    parser.add_argument("--extra-category-turn-features", action="store_true")
    parser.add_argument("--extra-score-calibration-features", action="store_true")
    parser.add_argument(
        "--extra-candidate-feature-npz",
        default="",
        help="Comma-separated npz files containing 2D per-candidate feature arrays aligned to the selected candidate artifact.",
    )
    parser.add_argument("--goal-cluster-n-clusters", type=int, default=32)
    parser.add_argument("--goal-cluster-cache-path", type=Path, default=CACHE_DIR / "listener_goal_embeddings.npz")
    parser.add_argument("--goal-cluster-batch-size", type=int, default=32)
    parser.add_argument("--train-positive-only", action="store_true")
    parser.add_argument(
        "--train-positive-eval-mask",
        action="store_true",
        help="Filter train rows to positive_eval mask (turn==1 OR target GPA == MOVES_TOWARD_GOAL)",
    )
    parser.add_argument(
        "--train-textproxy-positive",
        action="store_true",
        help="P5-A: filter train rows to turn==1 OR regex positive-reaction proxy on current text (no true GPA)",
    )
    parser.add_argument(
        "--positive-group-weight",
        type=float,
        default=None,
        help="P5-C: row weighting. positive-proxy/turn1 groups get this weight, others 0.5. No row removal.",
    )
    parser.add_argument("--feature-chunk-examples", type=int, default=512)
    parser.add_argument("--n-bm25-for-dense-flag", type=int, default=None)
    parser.add_argument("--model-family", choices=("lightgbm", "xgboost", "catboost"), default="lightgbm")
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--lgbm-objective", choices=("lambdarank", "rank_xendcg", "binary"), default="lambdarank")
    parser.add_argument("--binary-no-weight", action="store_true")
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--min-child-weight", type=float, default=1.0)
    parser.add_argument("--reg-lambda", type=float, default=3.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--xgb-device", default="cpu")
    parser.add_argument("--xgb-rank-objective", choices=("rank:ndcg", "rank:pairwise", "rank:map"), default="rank:ndcg")
    parser.add_argument("--catboost-loss", choices=("YetiRank", "PairLogit", "QuerySoftMax"), default="YetiRank")
    parser.add_argument("--catboost-task-type", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--catboost-devices", default="0")
    parser.add_argument("--lambdarank-truncation-level", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260518)
    parser.add_argument("--allow-encode-missing", action="store_true")
    parser.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help="Load a pre-trained LightGBM model file and skip the final fit (blind inference only).",
    )
    parser.add_argument(
        "--tpd1-mix-reranker-train",
        action="store_true",
        help="Append sampled TalkPlayData-1 mapped music turns as additional reranker training groups.",
    )
    parser.add_argument("--tpd1-mix-max-examples", type=int, default=0)
    parser.add_argument("--tpd1-mix-candidate-k", type=int, default=64)
    parser.add_argument("--tpd1-mix-weight", type=float, default=1.0)
    parser.add_argument("--tpd1-mix-seed", type=int, default=20260626)
    parser.add_argument("--tpd1-mix-cache-name", default="")
    parser.add_argument("--tpd1-mix-mapping", type=Path, default=CACHE_DIR / "spotify_uuid_map.parquet")
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])
    if args.load_model is not None:
        if args.blind_candidates is None:
            raise SystemExit("--load-model requires a blind target (--blind-candidates/--blind-target)")
        if args.tpd1_mix_reranker_train:
            raise SystemExit("--load-model cannot be combined with --tpd1-mix-reranker-train")
        if not args.skip_cv:
            print("--load-model: forcing --skip-cv (load-only inference)")
            args.skip_cv = True
    if args.extra_feedback_features:
        args.extra_gpa_features = True
        args.extra_reaction_features = True
    neutralize_098_features = {name.strip() for name in str(args.neutralize_098_features).split(",") if name.strip()}
    extra_candidate_feature_npz = [
        (Path(p) if Path(p).is_absolute() else REPO_ROOT / p)
        for p in str(args.extra_candidate_feature_npz).split(",")
        if p.strip()
    ]

    candidate_dir = args.public_candidates if args.public_candidates.is_absolute() else REPO_ROOT / args.public_candidates
    args.tpd1_mix_mapping = args.tpd1_mix_mapping if args.tpd1_mix_mapping.is_absolute() else REPO_ROOT / args.tpd1_mix_mapping
    legacy = proto.load_098_module()
    dense_cache_dir = CACHE_DIR / "dense_qfeat"

    print("loading tracks/users/examples")
    track_index = legacy.TrackIndex("all_tracks", corpus_fields=legacy.CORPUS_FIELDS_5, secondary_corpus_fields=None, load_dense=True)
    user_vectors = legacy.load_user_vectors()
    train_examples0 = legacy.build_examples_from_dataset("train")
    dev_examples0 = legacy.build_examples_from_dataset("test")
    base_examples = train_examples0 + dev_examples0
    base_sources = ["train"] * len(train_examples0) + ["devset"] * len(dev_examples0)

    # Blind-B-safe (fixed): dense query embeddings are message-only. Missing
    # cache rows are re-encoded on GPU.
    train_dense_caches = [dense_cache_dir / "train.npz"]
    train_dense_out = dense_cache_dir / "train.npz"
    dev_dense_caches = [dense_cache_dir / "devset.npz"]
    dev_dense_out = dense_cache_dir / "devset.npz"
    allow_encode = True

    if args.load_model is not None:
        # Load-only inference: the train/devset dense query features are never
        # consumed (fit_feature_stack ignores them and the final train matrix is
        # skipped), so do not require the train/devset dense caches here.
        print("skip train/devset dense query features (--load-model)")
        base_dense = np.zeros((len(base_examples), 1), dtype=np.float32)
    else:
        print("materializing dense query features (blind-B-safe fixed)")
        train_dense = proto.materialize_dense(
            legacy,
            train_examples0,
            train_dense_caches,
            cache_out=train_dense_out,
            batch_size=64,
            allow_encode_missing=allow_encode,
        )
        dev_dense = proto.materialize_dense(
            legacy,
            dev_examples0,
            dev_dense_caches,
            cache_out=dev_dense_out,
            batch_size=64,
            allow_encode_missing=allow_encode,
        )
        base_dense = np.concatenate([train_dense, dev_dense], axis=0)
    by_key = {key_str(src, ex): (src, ex, base_dense[i]) for i, (src, ex) in enumerate(zip(base_sources, base_examples, strict=True))}

    print("loading union candidates")
    cand_npz = np.load(candidate_dir / "candidates.npz", allow_pickle=False)
    keys = raw_keys(cand_npz["keys"])
    artifact_width = int(cand_npz["track_idx"].shape[1])
    width = artifact_width if args.max_candidates <= 0 else min(args.max_candidates, artifact_width)
    public_sources = []
    public_examples = []
    public_dense_rows = []
    for k in keys:
        src, ex, dense = by_key[k]
        public_sources.append(src)
        public_examples.append(ex)
        public_dense_rows.append(dense)
    public_dense = np.asarray(public_dense_rows, dtype=np.float32)
    indices = np.asarray(cand_npz["track_idx"], dtype=np.int32)[:, :width]
    primary_scores = choose_primary_scores(candidate_dir, cand_npz, width=width, mode=args.primary_score_mode)
    public_candidates = make_candidate_set(legacy, indices, primary_scores)
    folds = np.asarray(cand_npz["folds"], dtype=np.int16)

    tpd1_mix_examples: list[Any] = []
    tpd1_mix_candidates: Any | None = None
    tpd1_mix_dense: np.ndarray | None = None
    tpd1_mix_stats: dict[str, Any] | None = None
    if args.tpd1_mix_reranker_train:
        cache_name = args.tpd1_mix_cache_name or (
            f"reranker_mix_{args.config}_n{args.tpd1_mix_max_examples}_k{args.tpd1_mix_candidate_k}_s{args.tpd1_mix_seed}"
        )
        tpd1_mix_examples, tpd1_mix_stats = load_tpd1_mix_examples(
            legacy,
            track_index,
            mapping_path=args.tpd1_mix_mapping,
            max_examples=args.tpd1_mix_max_examples,
            seed=args.tpd1_mix_seed,
        )
        tpd1_mix_candidates = build_tpd1_mix_candidates(
            legacy,
            tpd1_mix_examples,
            track_index,
            candidate_k=args.tpd1_mix_candidate_k,
            seed=args.tpd1_mix_seed,
            cache_name=cache_name,
        )
        tpd1_mix_dense = np.zeros((len(tpd1_mix_examples), public_dense.shape[1]), dtype=np.float32)
        if "candidate_score" not in neutralize_098_features and args.primary_score_mode != "zero":
            print("warning: TPD1 mix uses zero candidate scores; public candidate_score is not neutralized")

    blind_candidate_dir = None
    blind_keys: list[str] = []
    blind_examples: list[Any] = []
    blind_dense: np.ndarray | None = None
    blind_candidates: Any | None = None
    if args.blind_candidates is not None:
        blind_candidate_dir = args.blind_candidates if args.blind_candidates.is_absolute() else REPO_ROOT / args.blind_candidates
        print(f"loading {args.blind_target} examples and candidates")
        blind_examples0 = legacy.build_examples_from_blind(args.blind_target)
        blind_dense0 = proto.materialize_dense(
            legacy,
            blind_examples0,
            [dense_cache_dir / f"{args.blind_target}.npz"],
            cache_out=dense_cache_dir / f"{args.blind_target}.npz",
            batch_size=64,
            allow_encode_missing=True,
        )
        blind_by_key = {
            blind_key_str(ex): (ex, blind_dense0[i])
            for i, ex in enumerate(blind_examples0)
        }
        blind_npz = np.load(blind_candidate_dir / "candidates.npz", allow_pickle=False)
        blind_keys = raw_keys(blind_npz["keys"])
        blind_artifact_width = int(blind_npz["track_idx"].shape[1])
        blind_width = blind_artifact_width if args.max_candidates <= 0 else min(args.max_candidates, blind_artifact_width)
        blind_examples = []
        blind_dense_rows = []
        for k in blind_keys:
            ex, dense = blind_by_key[k]
            blind_examples.append(ex)
            blind_dense_rows.append(dense)
        blind_dense = np.asarray(blind_dense_rows, dtype=np.float32)
        blind_indices = np.asarray(blind_npz["track_idx"], dtype=np.int32)[:, :blind_width]
        blind_primary_scores = choose_primary_scores(blind_candidate_dir, blind_npz, width=blind_width, mode=args.primary_score_mode)
        blind_candidates = make_candidate_set(legacy, blind_indices, blind_primary_scores)

    cand_metrics = legacy.candidate_metrics(public_examples, public_candidates, track_index)
    print(f"candidate metrics: {cand_metrics}")
    available_folds = sorted(set(int(x) for x in folds))
    eval_folds = [] if args.skip_cv else (available_folds if args.cv_folds == "all" else [int(x) for x in args.cv_folds.split(",") if x.strip()])
    all_rows = np.arange(len(public_examples), dtype=np.int32)
    fold_ranked = np.full(indices.shape, -1, dtype=np.int32)
    fold_scores = np.full(indices.shape, np.nan, dtype=np.float32)
    fold_metrics: dict[str, Any] = {}
    feature_names: list[str] | None = None
    eval_rows_parts: list[np.ndarray] = []
    cv_out_dir = OUTPUT_DIR / "reranker" / args.name / args.config / args.cv_artifact_mode / "public_labeled"
    cv_scores_dir = RESULTS_DIR / "reranker" / args.name / args.config / args.cv_artifact_mode / "public_labeled"

    for fold in eval_folds:
        print(f"CV fold {fold}: prepare")
        valid_rows = all_rows[folds == fold]
        train_rows = all_rows[folds != fold]
        if args.train_positive_eval_mask:
            before = len(train_rows)
            train_rows = positive_eval_mask_rows(public_sources, public_examples, train_rows)
            print(f"  train_positive_eval_mask: {before} → {len(train_rows)} rows")
        if args.train_textproxy_positive:
            before = len(train_rows)
            train_rows = textproxy_positive_rows(public_examples, train_rows)
            print(f"  train_textproxy_positive: {before} → {len(train_rows)} rows")
        if args.train_positive_only:
            train_rows = positive_rows(public_examples, indices, train_rows, track_index)
        eval_rows_parts.append(valid_rows)
        train_ex = subset_list(public_examples, train_rows)
        valid_ex = subset_list(public_examples, valid_rows)
        train_cand = make_candidate_set(legacy, indices[train_rows], primary_scores[train_rows])
        valid_cand = make_candidate_set(legacy, indices[valid_rows], primary_scores[valid_rows])
        train_dense_fold = public_dense[train_rows]
        valid_dense_fold = public_dense[valid_rows]
        feature_fit_ex = train_ex + tpd1_mix_examples if tpd1_mix_examples else train_ex

        encoder, vectorizer, track_tfidf, fast_context, n_bm25_flag, goal_cluster_context, category_turn_context = fit_feature_stack(
            legacy,
            train_ex,
            train_cand,
            train_dense_fold,
            track_index,
            user_vectors,
            feature_chunk_examples=args.feature_chunk_examples,
            n_bm25_for_dense_flag=args.n_bm25_for_dense_flag,
            extra_goal_cluster_features=args.extra_goal_cluster_features,
            extra_category_turn_features=args.extra_category_turn_features,
            goal_cluster_n_clusters=args.goal_cluster_n_clusters,
            goal_cluster_cache_path=args.goal_cluster_cache_path,
            goal_cluster_batch_size=args.goal_cluster_batch_size,
            goal_cluster_seed=args.seed + fold,
            fit_examples=feature_fit_ex,
        )
        score_calibration_context = fit_score_calibration_context(
            candidate_dir,
            indices,
            public_examples,
            train_rows,
            track_index,
            enabled=args.extra_score_calibration_features,
        )
        print(f"CV fold {fold}: build train matrix")
        x_train, y_train, groups, source_names = build_rich_matrix(
            legacy,
            train_ex,
            train_cand,
            train_dense_fold,
            encoder,
            vectorizer,
            track_tfidf,
            track_index,
            fast_context,
            candidate_dir,
            train_rows,
            width=width,
            feature_chunk_examples=args.feature_chunk_examples,
            n_bm25_for_dense_flag=n_bm25_flag,
            source_features_enabled=not args.disable_source_features,
            drop_cross_source_score_meta=args.drop_cross_source_score_meta,
            extra_source_score_transforms=args.extra_source_score_transforms,
            extra_metadata_features=args.extra_metadata_features,
            extra_gpa_features=args.extra_gpa_features,
            extra_reaction_features=args.extra_reaction_features,
            extra_assistant_thought_features=args.extra_assistant_thought_features,
            extra_tag_chain_features=args.extra_tag_chain_features,
            extra_hier_pop_features=args.extra_hier_pop_features,
            extra_pool_prior_features=args.extra_pool_prior_features,
            extra_talkplay_aux_features=args.extra_talkplay_aux_features,
            goal_cluster_context=goal_cluster_context,
            category_turn_context=category_turn_context,
            score_calibration_context=score_calibration_context,
            extra_candidate_feature_npz=extra_candidate_feature_npz,
            neutralize_098_features=neutralize_098_features,
            labels=True,
        )
        if y_train is None or int(y_train.sum()) == 0:
            raise RuntimeError("no positive labels")
        feature_names = encoder.feature_names + source_names
        categorical_feature = categorical_indices_for(feature_names, encoder.categorical_feature_indices)
        tpd1_groups: list[int] = []
        if tpd1_mix_examples:
            if tpd1_mix_candidates is None or tpd1_mix_dense is None:
                raise RuntimeError("TPD1 mix was enabled but candidates/dense rows are missing")
            print(f"CV fold {fold}: build TPD1 mix matrix groups={len(tpd1_mix_examples)}")
            tpd1_source_rows = np.arange(len(tpd1_mix_examples), dtype=np.int32)
            x_tpd1, y_tpd1, tpd1_groups, tpd1_names = build_rich_matrix(
                legacy,
                tpd1_mix_examples,
                tpd1_mix_candidates,
                tpd1_mix_dense,
                encoder,
                vectorizer,
                track_tfidf,
                track_index,
                fast_context,
                candidate_dir,
                tpd1_source_rows,
                width=tpd1_mix_candidates.indices.shape[1],
                feature_chunk_examples=args.feature_chunk_examples,
                n_bm25_for_dense_flag=n_bm25_flag,
                source_features_enabled=False,
                drop_cross_source_score_meta=args.drop_cross_source_score_meta,
                extra_source_score_transforms=args.extra_source_score_transforms,
                extra_metadata_features=args.extra_metadata_features,
                extra_gpa_features=args.extra_gpa_features,
                extra_reaction_features=args.extra_reaction_features,
                extra_assistant_thought_features=args.extra_assistant_thought_features,
                extra_tag_chain_features=args.extra_tag_chain_features,
                extra_hier_pop_features=args.extra_hier_pop_features,
                extra_pool_prior_features=args.extra_pool_prior_features,
                extra_talkplay_aux_features=args.extra_talkplay_aux_features,
                goal_cluster_context=goal_cluster_context,
                category_turn_context=category_turn_context,
                score_calibration_context=None,
                extra_candidate_feature_npz=[],
                neutralize_098_features=neutralize_098_features,
                labels=True,
            )
            if y_tpd1 is None or int(y_tpd1.sum()) == 0:
                raise RuntimeError("no positive labels in TPD1 mix rows")
            x_tpd1 = append_zero_columns_to_match(x_tpd1, tpd1_names, source_names)
            x_train = np.vstack([x_train, x_tpd1]).astype(np.float32, copy=False)
            y_train = np.concatenate([y_train, y_tpd1]).astype(np.int8, copy=False)
            groups = groups + tpd1_groups
            print(
                f"CV fold {fold}: appended TPD1 mix rows={len(y_tpd1)} positives={int(y_tpd1.sum())} "
                f"groups={len(tpd1_groups)} weight={args.tpd1_mix_weight}"
            )
            del x_tpd1, y_tpd1
            gc.collect()
        print(
            f"CV fold {fold}: fit {args.model_family}/{args.lgbm_objective} "
            f"rows={len(y_train)} positives={int(y_train.sum())} "
            f"groups={len(groups)} features={x_train.shape[1]}"
        )
        if tpd1_groups:
            public_group_weights = (
                positive_group_weights(train_ex, args.positive_group_weight)
                if args.positive_group_weight is not None
                else np.ones(len(train_ex), dtype=np.float32)
            )
            fold_group_weights = np.concatenate(
                [
                    public_group_weights.astype(np.float32, copy=False),
                    np.full(len(tpd1_groups), float(args.tpd1_mix_weight), dtype=np.float32),
                ]
            )
        else:
            fold_group_weights = (
                positive_group_weights(train_ex, args.positive_group_weight)
                if args.positive_group_weight is not None
                else None
            )
        model = fit_lgbm_model(
            args,
            x_train,
            y_train,
            groups,
            fold_seed=args.seed + fold,
            categorical_feature=categorical_feature,
            feature_names=feature_names,
            group_weights=fold_group_weights,
        )
        del x_train, y_train, groups
        gc.collect()

        print(f"CV fold {fold}: build valid matrix")
        x_valid, _, valid_groups, _ = build_rich_matrix(
            legacy,
            valid_ex,
            valid_cand,
            valid_dense_fold,
            encoder,
            vectorizer,
            track_tfidf,
            track_index,
            fast_context,
            candidate_dir,
            valid_rows,
            width=width,
            feature_chunk_examples=args.feature_chunk_examples,
            n_bm25_for_dense_flag=n_bm25_flag,
            source_features_enabled=not args.disable_source_features,
            drop_cross_source_score_meta=args.drop_cross_source_score_meta,
            extra_source_score_transforms=args.extra_source_score_transforms,
            extra_metadata_features=args.extra_metadata_features,
            extra_gpa_features=args.extra_gpa_features,
            extra_reaction_features=args.extra_reaction_features,
            extra_assistant_thought_features=args.extra_assistant_thought_features,
            extra_tag_chain_features=args.extra_tag_chain_features,
            extra_hier_pop_features=args.extra_hier_pop_features,
            extra_pool_prior_features=args.extra_pool_prior_features,
            extra_talkplay_aux_features=args.extra_talkplay_aux_features,
            goal_cluster_context=goal_cluster_context,
            category_turn_context=category_turn_context,
            score_calibration_context=score_calibration_context,
            extra_candidate_feature_npz=extra_candidate_feature_npz,
            neutralize_098_features=neutralize_098_features,
            labels=False,
        )
        pred = predict_lgbm_model(model, x_valid)
        ranked, ranked_scores = rank_from_predictions(valid_cand, valid_groups, pred)
        fold_ranked[valid_rows] = ranked
        fold_scores[valid_rows] = ranked_scores
        fold_sources = subset_list(public_sources, valid_rows)
        fold_metrics[f"fold{fold}"] = proto.evaluate_ranked(fold_sources, valid_ex, ranked, track_index, top_k=args.top_k)
        print(f"CV fold {fold}: {fold_metrics[f'fold{fold}']}")
        del x_valid, pred, model, encoder, vectorizer, track_tfidf, fast_context
        gc.collect()

    if args.merge_existing_cv and eval_rows_parts:
        ranked_path = cv_out_dir / "ranked.npz"
        scores_path = cv_scores_dir / "scores.json"
        if ranked_path.exists():
            print(f"merge existing CV artifact: {rel(ranked_path)}")
            with np.load(ranked_path, allow_pickle=False) as existing:
                existing_ranked = np.asarray(existing["track_idx"], dtype=np.int32)
                if existing_ranked.shape != fold_ranked.shape:
                    raise ValueError(
                        f"existing ranked shape {existing_ranked.shape} != current shape {fold_ranked.shape}"
                    )
                existing_scores = (
                    np.asarray(existing["scores"], dtype=np.float32)
                    if "scores" in existing.files
                    else np.full(existing_ranked.shape, np.nan, dtype=np.float32)
                )
            current_mask = np.zeros(len(public_examples), dtype=bool)
            for rows in eval_rows_parts:
                current_mask[rows] = True
            existing_mask = np.any(existing_ranked >= 0, axis=1)
            existing_mask[current_mask] = False
            existing_rows = np.flatnonzero(existing_mask).astype(np.int32)
            if len(existing_rows):
                fold_ranked[existing_rows] = existing_ranked[existing_rows]
                fold_scores[existing_rows] = existing_scores[existing_rows]
                eval_rows_parts.insert(0, existing_rows)
                print(f"merged existing rows={len(existing_rows)} with current rows={int(current_mask.sum())}")
            if scores_path.exists():
                try:
                    existing_scores_json = json.loads(scores_path.read_text())
                    existing_fold_metrics = dict(existing_scores_json.get("fold_metrics") or {})
                    fold_metrics = {**existing_fold_metrics, **fold_metrics}
                except Exception as exc:  # noqa: BLE001
                    print(f"warning: failed to merge existing fold_metrics from {rel(scores_path)}: {exc}")

    cv_metrics: dict[str, Any] = {}
    if eval_rows_parts:
        eval_rows = np.concatenate(eval_rows_parts).astype(np.int32)
        eval_sources = subset_list(public_sources, eval_rows)
        eval_examples = subset_list(public_examples, eval_rows)
        cv_metrics = proto.evaluate_ranked(eval_sources, eval_examples, fold_ranked[eval_rows], track_index, top_k=args.top_k)
        print(f"CV combined: {cv_metrics}")

    base_manifest = {
        "schema_version": 1,
        "artifact_type": "ranked",
        "stage": "reranker",
        "name": args.name,
        "config": args.config,
        "producer": {"command": ["uv", "run", "python", "scripts/run_protocol_098_union_rich_lgbm.py"], "cwd": "."},
        "protocol": "docs/pipeline_cv_protocol.md",
        "params": jsonable(vars(args)),
        "source_candidate_artifact": rel(candidate_dir),
        "feature_names": feature_names or [],
        "external_reranker_train": {
            "enabled": bool(args.tpd1_mix_reranker_train),
            "source": "TalkPlayData-1 train" if args.tpd1_mix_reranker_train else None,
            "stats": tpd1_mix_stats,
            "candidate_policy": (
                "fit_free_098_bm25_candidates_gold_inserted_zero_primary_score"
                if args.tpd1_mix_reranker_train
                else None
            ),
            "source_feature_policy": (
                "TPD1 training groups do not read public/TPD1 source_features; missing source feature columns are zero-filled"
                if args.tpd1_mix_reranker_train
                else None
            ),
            "dense_query_policy": (
                "TPD1 training groups use zero query dense vectors in this first-pass experiment"
                if args.tpd1_mix_reranker_train
                else None
            ),
            "group_weight": float(args.tpd1_mix_weight) if args.tpd1_mix_reranker_train else None,
        },
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_blind_for_fit": False,
            "uses_devset_score_for_tuning": False,
            "popularity_tiebreaker": False,
            "train_rows_use_oof_candidates_when_required": True,
            "tpd1_reranker_train_rows": bool(args.tpd1_mix_reranker_train),
            "tpd1_train_rows_use_tpd1_internal_oof_source_features": False,
            "tpd1_train_rows_source_features_zero_filled": bool(args.tpd1_mix_reranker_train),
            # Blind-B-safe is fixed ON: the build excludes GPA / conversation_goal /
            # current thought / session_date from query text AND base features.
            "blind_b_safe": True,
            "current_thought_allowed": False,
            "conversation_goal_allowed": False,
            "gpa_allowed": False,
            "session_date_allowed": False,
            "intent_features": False,
        },
    }
    outputs: dict[str, str] = {}
    if eval_rows_parts:
        save_ranked_public(
            cv_out_dir,
            keys,
            public_sources,
            public_examples,
            folds,
            fold_ranked,
            fold_scores,
            {
                **base_manifest,
                "artifact_mode": args.cv_artifact_mode,
                "target": "public_labeled",
                "created_at": utc_now(),
            },
            track_index.track_ids,
        )
        json_dump(
            cv_scores_dir / "scores.json",
            {
                "name": args.name,
                "config": args.config,
                "artifact_mode": args.cv_artifact_mode,
                "target": "public_labeled",
                "candidate_metrics": cand_metrics,
                "eval_folds": sorted(set(int(folds[int(i)]) for i in eval_rows)),
                "eval_rows": int(len(eval_rows)),
                "cv_metrics": cv_metrics,
                "fold_metrics": fold_metrics,
                "retriever_artifact": rel(candidate_dir),
                "reranker_artifact": rel(cv_out_dir),
                "merged_existing_cv": bool(args.merge_existing_cv),
            },
        )
        outputs["scores"] = rel(cv_scores_dir / "scores.json")
        outputs["reranker_artifact"] = rel(cv_out_dir)

    if blind_candidate_dir is not None:
        if blind_candidates is None or blind_dense is None:
            raise RuntimeError("blind candidates were not loaded")
        print(f"fit final model for {args.blind_target}")
        final_train_rows = all_rows
        if args.train_positive_eval_mask:
            before = len(final_train_rows)
            final_train_rows = positive_eval_mask_rows(public_sources, public_examples, final_train_rows)
            print(f"  final train_positive_eval_mask: {before} → {len(final_train_rows)} rows")
        if args.train_textproxy_positive:
            before = len(final_train_rows)
            final_train_rows = textproxy_positive_rows(public_examples, final_train_rows)
            print(f"  final train_textproxy_positive: {before} → {len(final_train_rows)} rows")
        if args.train_positive_only:
            final_train_rows = positive_rows(public_examples, indices, final_train_rows, track_index)
        final_ex = subset_list(public_examples, final_train_rows)
        final_cand = make_candidate_set(legacy, indices[final_train_rows], primary_scores[final_train_rows])
        final_dense = public_dense[final_train_rows]
        final_feature_fit_ex = final_ex + tpd1_mix_examples if tpd1_mix_examples else final_ex
        encoder, vectorizer, track_tfidf, fast_context, n_bm25_flag, goal_cluster_context, category_turn_context = fit_feature_stack(
            legacy,
            final_ex,
            final_cand,
            final_dense,
            track_index,
            user_vectors,
            feature_chunk_examples=args.feature_chunk_examples,
            n_bm25_for_dense_flag=args.n_bm25_for_dense_flag,
            extra_goal_cluster_features=args.extra_goal_cluster_features,
            extra_category_turn_features=args.extra_category_turn_features,
            goal_cluster_n_clusters=args.goal_cluster_n_clusters,
            goal_cluster_cache_path=args.goal_cluster_cache_path,
            goal_cluster_batch_size=args.goal_cluster_batch_size,
            goal_cluster_seed=args.seed + 100,
            fit_examples=final_feature_fit_ex,
        )
        score_calibration_context = fit_score_calibration_context(
            candidate_dir,
            indices,
            public_examples,
            final_train_rows,
            track_index,
            enabled=args.extra_score_calibration_features,
        )
        if args.load_model is not None:
            load_model_path = args.load_model if args.load_model.is_absolute() else REPO_ROOT / args.load_model
            if not load_model_path.exists():
                raise FileNotFoundError(f"--load-model model file not found: {load_model_path}")
            print(f"skip final fit: loading LightGBM model from {rel(load_model_path)}")
            final_model = lgb.Booster(model_file=str(load_model_path))
            final_feature_names = [str(name) for name in final_model.feature_name()]
        else:
            x_train, y_train, groups, source_names = build_rich_matrix(
                legacy,
                final_ex,
                final_cand,
                final_dense,
                encoder,
                vectorizer,
                track_tfidf,
                track_index,
                fast_context,
                candidate_dir,
                final_train_rows,
                width=width,
                feature_chunk_examples=args.feature_chunk_examples,
                n_bm25_for_dense_flag=n_bm25_flag,
                source_features_enabled=not args.disable_source_features,
                drop_cross_source_score_meta=args.drop_cross_source_score_meta,
                extra_source_score_transforms=args.extra_source_score_transforms,
                extra_metadata_features=args.extra_metadata_features,
                extra_gpa_features=args.extra_gpa_features,
                extra_reaction_features=args.extra_reaction_features,
                extra_assistant_thought_features=args.extra_assistant_thought_features,
                extra_tag_chain_features=args.extra_tag_chain_features,
                extra_hier_pop_features=args.extra_hier_pop_features,
                extra_pool_prior_features=args.extra_pool_prior_features,
                extra_talkplay_aux_features=args.extra_talkplay_aux_features,
                goal_cluster_context=goal_cluster_context,
                category_turn_context=category_turn_context,
                score_calibration_context=score_calibration_context,
                extra_candidate_feature_npz=extra_candidate_feature_npz,
                neutralize_098_features=neutralize_098_features,
                labels=True,
            )
            if y_train is None or int(y_train.sum()) == 0:
                raise RuntimeError("no positive labels for final model")
            final_feature_names = encoder.feature_names + source_names
            final_categorical_feature = categorical_indices_for(final_feature_names, encoder.categorical_feature_indices)
            final_tpd1_groups: list[int] = []
            if tpd1_mix_examples:
                if tpd1_mix_candidates is None or tpd1_mix_dense is None:
                    raise RuntimeError("TPD1 mix was enabled but candidates/dense rows are missing")
                print(f"final model: build TPD1 mix matrix groups={len(tpd1_mix_examples)}")
                tpd1_source_rows = np.arange(len(tpd1_mix_examples), dtype=np.int32)
                x_tpd1, y_tpd1, final_tpd1_groups, tpd1_names = build_rich_matrix(
                    legacy,
                    tpd1_mix_examples,
                    tpd1_mix_candidates,
                    tpd1_mix_dense,
                    encoder,
                    vectorizer,
                    track_tfidf,
                    track_index,
                    fast_context,
                    candidate_dir,
                    tpd1_source_rows,
                    width=tpd1_mix_candidates.indices.shape[1],
                    feature_chunk_examples=args.feature_chunk_examples,
                    n_bm25_for_dense_flag=n_bm25_flag,
                    source_features_enabled=False,
                    drop_cross_source_score_meta=args.drop_cross_source_score_meta,
                    extra_source_score_transforms=args.extra_source_score_transforms,
                    extra_metadata_features=args.extra_metadata_features,
                    extra_gpa_features=args.extra_gpa_features,
                    extra_reaction_features=args.extra_reaction_features,
                    extra_assistant_thought_features=args.extra_assistant_thought_features,
                    extra_tag_chain_features=args.extra_tag_chain_features,
                    extra_hier_pop_features=args.extra_hier_pop_features,
                    extra_pool_prior_features=args.extra_pool_prior_features,
                    extra_talkplay_aux_features=args.extra_talkplay_aux_features,
                    goal_cluster_context=goal_cluster_context,
                    category_turn_context=category_turn_context,
                    score_calibration_context=None,
                    extra_candidate_feature_npz=[],
                    neutralize_098_features=neutralize_098_features,
                    labels=True,
                )
                if y_tpd1 is None or int(y_tpd1.sum()) == 0:
                    raise RuntimeError("no positive labels in final TPD1 mix rows")
                x_tpd1 = append_zero_columns_to_match(x_tpd1, tpd1_names, source_names)
                x_train = np.vstack([x_train, x_tpd1]).astype(np.float32, copy=False)
                y_train = np.concatenate([y_train, y_tpd1]).astype(np.int8, copy=False)
                groups = groups + final_tpd1_groups
                print(
                    f"final model: appended TPD1 mix rows={len(y_tpd1)} positives={int(y_tpd1.sum())} "
                    f"groups={len(final_tpd1_groups)} weight={args.tpd1_mix_weight}"
                )
                del x_tpd1, y_tpd1
                gc.collect()
            print(
                f"final {args.model_family}/{args.lgbm_objective} "
                f"rows={len(y_train)} positives={int(y_train.sum())} "
                f"groups={len(groups)} features={x_train.shape[1]}"
            )
            if final_tpd1_groups:
                public_group_weights = (
                    positive_group_weights(final_ex, args.positive_group_weight)
                    if args.positive_group_weight is not None
                    else np.ones(len(final_ex), dtype=np.float32)
                )
                final_group_weights = np.concatenate(
                    [
                        public_group_weights.astype(np.float32, copy=False),
                        np.full(len(final_tpd1_groups), float(args.tpd1_mix_weight), dtype=np.float32),
                    ]
                )
            else:
                final_group_weights = (
                    positive_group_weights(final_ex, args.positive_group_weight)
                    if args.positive_group_weight is not None
                    else None
                )
            final_model = fit_lgbm_model(
                args,
                x_train,
                y_train,
                groups,
                fold_seed=args.seed + 100,
                categorical_feature=final_categorical_feature,
                feature_names=final_feature_names,
                group_weights=final_group_weights,
            )
            del x_train, y_train, groups
            gc.collect()

        print(f"rank {args.blind_target}")
        blind_rows = np.arange(len(blind_examples), dtype=np.int32)
        x_blind, _, blind_groups, blind_source_names = build_rich_matrix(
            legacy,
            blind_examples,
            blind_candidates,
            blind_dense,
            encoder,
            vectorizer,
            track_tfidf,
            track_index,
            fast_context,
            blind_candidate_dir,
            blind_rows,
            width=blind_candidates.indices.shape[1],
            feature_chunk_examples=args.feature_chunk_examples,
            n_bm25_for_dense_flag=n_bm25_flag,
            source_features_enabled=not args.disable_source_features,
            drop_cross_source_score_meta=args.drop_cross_source_score_meta,
            extra_source_score_transforms=args.extra_source_score_transforms,
            extra_metadata_features=args.extra_metadata_features,
            extra_gpa_features=args.extra_gpa_features,
            extra_reaction_features=args.extra_reaction_features,
            extra_assistant_thought_features=args.extra_assistant_thought_features,
            extra_tag_chain_features=args.extra_tag_chain_features,
            extra_hier_pop_features=args.extra_hier_pop_features,
            extra_pool_prior_features=args.extra_pool_prior_features,
            extra_talkplay_aux_features=args.extra_talkplay_aux_features,
            goal_cluster_context=goal_cluster_context,
            category_turn_context=category_turn_context,
            score_calibration_context=score_calibration_context,
            extra_candidate_feature_npz=extra_candidate_feature_npz,
            neutralize_098_features=neutralize_098_features,
            labels=False,
        )
        if args.load_model is not None:
            built_feature_names = encoder.feature_names + blind_source_names
            if built_feature_names != final_feature_names:
                mismatch = next(
                    (
                        f"index {i}: built={b!r} model={m!r}"
                        for i, (b, m) in enumerate(zip(built_feature_names, final_feature_names))
                        if b != m
                    ),
                    f"length built={len(built_feature_names)} model={len(final_feature_names)}",
                )
                raise RuntimeError(f"blind feature layout does not match the loaded model ({mismatch})")
        pred = predict_lgbm_model(final_model, x_blind)
        blind_ranked, blind_scores = rank_from_predictions(blind_candidates, blind_groups, pred)
        blind_sizes = np.asarray([(row >= 0).sum() for row in blind_ranked], dtype=np.int32)
        blind_out = OUTPUT_DIR / "reranker" / args.name / args.config / "full_public" / args.blind_target
        blind_out.mkdir(parents=True, exist_ok=True)
        if args.load_model is not None:
            model_path = blind_out / "model.txt"
            if load_model_path.resolve() != model_path.resolve():
                shutil.copyfile(load_model_path, model_path)
        else:
            model_path = save_tree_model(final_model, blind_out / "model")
        save_ranked_artifact(
            blind_out,
            blind_ranked,
            blind_sizes,
            target=args.blind_target,
            scores=blind_scores,
            manifest={
                **base_manifest,
                "feature_names": final_feature_names,
                "artifact_mode": "full_public",
                "target": args.blind_target,
                "created_at": utc_now(),
                "fit_candidate_artifact_mode": args.cv_artifact_mode,
                "fit_candidate_artifact": rel(candidate_dir),
                "inference_candidate_artifact_mode": "full_public",
                "inference_candidate_artifact": rel(blind_candidate_dir),
                "model": file_ref(model_path),
                "loaded_model": rel(load_model_path) if args.load_model is not None else None,
            },
            compress=True,
        )
        blind_scores_dir = RESULTS_DIR / "reranker" / args.name / args.config / "full_public" / args.blind_target
        json_dump(
            blind_scores_dir / "scores.json",
            {
                "name": args.name,
                "config": args.config,
                "artifact_mode": "full_public",
                "target": args.blind_target,
                "fit_candidate_artifact": rel(candidate_dir),
                "inference_candidate_artifact": rel(blind_candidate_dir),
                "reranker_artifact": rel(blind_out),
                "model": rel(model_path),
                "loaded_model": rel(load_model_path) if args.load_model is not None else None,
                "public_candidate_metrics": cand_metrics,
                "cv_metrics_snapshot": cv_metrics,
            },
        )
        outputs["blind_scores"] = rel(blind_scores_dir / "scores.json")
        outputs["blind_ranked"] = rel(blind_out)

    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
