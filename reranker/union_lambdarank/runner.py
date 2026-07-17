#!/usr/bin/env python3
"""Fit the final LightGBM LambdaRank model and rank one inference target."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sparse_normalize

from recsys2026.artifacts import json_dump, utc_now
from recsys2026.artifacts import file_ref, save_ranked_artifact
from recsys2026.paths import PREPROCESSED_DIR, OUTPUT_DIR, REPO_ROOT, RESULTS_DIR


from reranker.union_lambdarank import fast_features
from reranker.union_lambdarank import protocol as proto

RERANKER_NAME = "union_lambdarank"
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


def extra_metadata_feature_context(track_index: Any) -> dict[str, np.ndarray]:
    n = len(track_index.track_ids)
    isrc_year = np.zeros(n, dtype=np.float32)
    isrc_country_bucket = np.zeros(n, dtype=np.float32)
    isrc_missing = np.ones(n, dtype=np.float32)
    release_minus_isrc_abs = np.zeros(n, dtype=np.float32)
    duration_bucket = np.zeros(n, dtype=np.float32)
    album_mean_duration = np.zeros(n, dtype=np.float32)
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
        "isrc_country": np.asarray(isrc_country, dtype=object),
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
        "artist_track_count_log1p": np.log1p(artist_count).astype(
            np.float32, copy=False
        ),
        "album_mean_popularity": album_mean,
        "album_track_count_log1p": np.log1p(album_count).astype(np.float32, copy=False),
        "track_over_artist_mean_popularity": track_over_artist,
        "track_over_album_mean_popularity": track_over_album,
    }


def raw_keys(arr: np.ndarray) -> list[str]:
    return [bytes(x).decode("utf-8") for x in arr]


def key_str(source: str, ex: Any) -> str:
    return f"{source}:{ex.session_id}:{int(ex.turn_number)}"


def inference_key_str(ex: Any) -> str:
    return f"{ex.session_id}:{int(ex.turn_number)}"


def make_candidate_set(features: Any, indices: np.ndarray, scores: np.ndarray) -> Any:
    return features.CandidateSet(
        indices=indices.astype(np.int32, copy=False),
        scores=scores.astype(np.float32, copy=False),
    )


def choose_primary_scores(
    cand_arrays: dict[str, np.ndarray],
    *,
    width: int,
) -> np.ndarray:
    shape = cand_arrays["track_idx"][:, :width].shape
    return np.zeros(shape, dtype=np.float32)


def source_feature_plan(
    candidate_dir: Path,
    *,
    drop_cross_source_score_meta: bool = False,
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
        elif any(
            token in key
            for token in (
                "score",
                "sim",
                "similarity",
                "distance",
                "dist",
                "count",
                "weight",
                "logit",
                "prob",
            )
        ):
            plan.extend(
                [
                    (f"src_{clean}_raw0", key),
                    (f"src_{clean}_row_z", key),
                ]
            )
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
) -> tuple[np.ndarray, list[str]]:
    if not enabled:
        return x_base, []
    plan = source_feature_plan(
        candidate_dir,
        drop_cross_source_score_meta=drop_cross_source_score_meta,
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
                rank = np.where((raw_float > 0) & finite, raw_float, 0.0).astype(
                    np.float32, copy=False
                )
                inv = np.zeros_like(rank, dtype=np.float32)
                np.divide(1.0, rank, out=inv, where=rank > 0)
                values_by_name = {
                    f"src_{raw_key.replace('__', '_')}_miss0": rank,
                    f"src_{raw_key.replace('__', '_')}_log1p": np.log1p(rank).astype(
                        np.float32, copy=False
                    ),
                    f"src_{raw_key.replace('__', '_')}_inv": inv,
                }
            elif any(
                token in raw_key
                for token in (
                    "score",
                    "sim",
                    "similarity",
                    "distance",
                    "dist",
                    "count",
                    "weight",
                    "logit",
                    "prob",
                )
            ):
                score = np.nan_to_num(
                    raw_float, nan=0.0, posinf=0.0, neginf=0.0
                ).astype(np.float32, copy=False)
                denom = np.maximum(finite.sum(axis=1, keepdims=True), 1)
                row_mean = np.where(
                    denom > 0,
                    np.nan_to_num(raw_float, nan=0.0, posinf=0.0, neginf=0.0).sum(
                        axis=1, keepdims=True
                    )
                    / denom,
                    0.0,
                )
                centered = np.where(finite, raw_float - row_mean, 0.0)
                row_std = (
                    np.sqrt((centered * centered).sum(axis=1, keepdims=True) / denom)
                    + 1e-6
                )
                z = np.where(finite, centered / row_std, 0.0).astype(
                    np.float32, copy=False
                )
                values_by_name = {
                    f"src_{raw_key.replace('__', '_')}_raw0": score,
                    f"src_{raw_key.replace('__', '_')}_row_z": z,
                }
            else:
                values_by_name = {
                    f"src_{raw_key.replace('__', '_')}": np.nan_to_num(
                        raw_float, nan=0.0, posinf=0.0, neginf=0.0
                    ).astype(np.float32, copy=False)
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
        raise RuntimeError(
            f"source feature fill mismatch col={col} width={out.shape[1]}"
        )
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
    features: Any,
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
        row[:, 3] = np.asarray(
            [
                float(str(c) == user_country and bool(user_country))
                for c in cand_countries
            ],
            dtype=np.float32,
        )
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
        row[:, 7] = np.where(
            album_mean > 0,
            (track_index.duration[cand_idx] - album_mean) / 60_000.0,
            0.0,
        )

        age = float((ex.user_profile or {}).get("age") or 0.0)
        # Session dates are excluded; date-derived columns remain neutral.
        session_year = 2026.0
        rel_year = track_index.release_year[cand_idx]
        row[:, 8] = np.where(
            (age > 0) & (rel_year > 0), age - (session_year - rel_year), 0.0
        )

        # Goal specificity is excluded so fit and inference use the same schema.
        specificity = ""
        track_high = float(len(specificity) >= 1 and specificity[0].upper() == "H")
        expr_high = float(len(specificity) >= 2 and specificity[1].upper() == "H")
        row[:, 9] = track_high
        row[:, 10] = expr_high
        row[:, 11] = float(track_high and expr_high)

        offset += m
    if offset != len(extra):
        raise RuntimeError(
            f"extra metadata feature length mismatch offset={offset} total={len(extra)}"
        )
    out = np.empty((x_base.shape[0], x_base.shape[1] + n_extra), dtype=np.float32)
    out[:, : x_base.shape[1]] = x_base
    out[:, x_base.shape[1] :] = extra
    return out, list(EXTRA_METADATA_FEATURE_NAMES)


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
    extra = np.column_stack([ctx[name][cand_idx] for name in ctx]).astype(
        np.float32, copy=False
    )
    out = np.empty(
        (x_base.shape[0], x_base.shape[1] + extra.shape[1]), dtype=np.float32
    )
    out[:, : x_base.shape[1]] = x_base
    out[:, x_base.shape[1] :] = extra
    return out, list(EXTRA_HIER_POP_FEATURE_NAMES)


EXTRA_TAG_CHAIN_FEATURE_NAMES = [
    "extra_tag_token_overlap_with_history",
    "extra_tag_token_jaccard_with_history",
    "extra_tag_vec_cosine_with_history",
    "extra_tag_chain_neighbor_overlap",
    "extra_tag_chain_ppmi_sum",
    "extra_tag_chain_ppmi_max",
]


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
        ids = sorted(
            {vocab[tok] for tok in toks if tok in vocab}, key=lambda i: (-df_id[i], i)
        )
        ids = ids[:max_track_tokens]
        ids.sort()
        token_id_rows.append(ids)
        rows.extend([row] * len(ids))
        cols.extend(ids)
    data = np.ones(len(rows), dtype=np.float32)
    tag_bin = sparse.csr_matrix(
        (data, (rows, cols)), shape=(n_tracks, len(vocab)), dtype=np.float32
    )
    tag_count = np.asarray(tag_bin.getnnz(axis=1), dtype=np.float32)
    idf = (
        np.log((1.0 + float(n_tracks)) / (1.0 + df_id.astype(np.float32))) + 1.0
    ).astype(np.float32)
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
                row[:, 1] = np.divide(
                    overlap, union, out=np.zeros_like(overlap), where=union > 0
                )

                hist_tfidf = _as_1d_float(tag_tfidf[hist_idx].sum(axis=0))
                norm = float(np.linalg.norm(hist_tfidf))
                if norm > 0.0:
                    hist_tfidf /= norm
                    hist_tfidf_sparse = sparse.csr_matrix(hist_tfidf.reshape(1, -1))
                    row[:, 2] = _as_1d_float(
                        tag_tfidf[cand_idx].dot(hist_tfidf_sparse.T)
                    )

                hist_sparse = sparse.csr_matrix(
                    (
                        np.ones(len(hist_ids), dtype=np.float32),
                        ([0] * len(hist_ids), hist_ids),
                    ),
                    shape=(1, tag_bin.shape[1]),
                    dtype=np.float32,
                )
                expanded_ppmi = hist_sparse.dot(adj_ppmi)
                expanded_binary = hist_sparse.dot(adj_binary)
                expanded_binary.data = np.ones_like(
                    expanded_binary.data, dtype=np.float32
                )
                row[:, 3] = _as_1d_float(cand_bin.dot(expanded_binary.T))
                row[:, 4] = _as_1d_float(cand_bin.dot(expanded_ppmi.T))
                weighted = cand_bin.multiply(expanded_ppmi)
                row[:, 5] = _as_1d_float(weighted.max(axis=1))
        offset += m
    if offset != len(extra):
        raise RuntimeError(
            f"extra tag chain feature length mismatch offset={offset} total={len(extra)}"
        )
    out = np.empty((x_base.shape[0], x_base.shape[1] + n_extra), dtype=np.float32)
    out[:, : x_base.shape[1]] = x_base
    out[:, x_base.shape[1] :] = extra
    return out, list(EXTRA_TAG_CHAIN_FEATURE_NAMES)


def fit_feature_stack(
    features: Any,
    train_examples: list[Any],
    track_index: Any,
    user_vectors: dict[str, np.ndarray],
) -> tuple[Any, TfidfVectorizer, Any, Any]:
    examples_for_fit = train_examples
    encoder = features.FeatureEncoder(track_index, user_vectors)
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
        + [features.goal_text() for _ in examples_for_fit]
        + [features.conversation_text(ex, track_index) for ex in examples_for_fit]
        + [features.query_text(ex.user_query) for ex in examples_for_fit]
    )
    print(f"fit text vectorizer rows={len(examples_for_fit)}")
    vectorizer.fit(text_corpus)
    track_tfidf = vectorizer.transform(track_index.texts)
    fast_context = fast_features.make_fast_context(features, encoder)
    return encoder, vectorizer, track_tfidf, fast_context


def build_rich_matrix(
    features: Any,
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
    source_features_enabled: bool,
    drop_cross_source_score_meta: bool,
    extra_metadata_features: bool,
    extra_tag_chain_features: bool,
    extra_hier_pop_features: bool,
    neutralize_base_features: set[str],
    labels: bool,
) -> tuple[np.ndarray, np.ndarray | None, list[int], list[str]]:
    x_base, y, groups = fast_features.build_feature_matrix_fast(
        features,
        examples,
        candidates,
        encoder,
        vectorizer,
        track_tfidf,
        chunk_examples=feature_chunk_examples,
        query_dense_emb=dense_q,
        fast_context=fast_context,
    )
    if neutralize_base_features:
        for col, name in enumerate(encoder.feature_names):
            if name in neutralize_base_features:
                x_base[:, col] = 0.0
    if not labels:
        y = None
    valid_mask = candidates.indices[:, :width] >= 0
    x_meta, metadata_names = append_extra_metadata_features(
        x_base,
        features,
        examples,
        candidates,
        track_index,
        width=width,
        enabled=extra_metadata_features,
    )
    x_pop, hier_pop_names = append_extra_hier_pop_features(
        x_meta,
        candidates,
        track_index,
        width=width,
        enabled=extra_hier_pop_features,
    )
    x_tag, tag_chain_names = append_extra_tag_chain_features(
        x_pop,
        examples,
        candidates,
        track_index,
        width=width,
        enabled=extra_tag_chain_features,
    )
    x, source_names = append_source_features(
        x_tag,
        candidate_dir,
        source_rows,
        valid_mask,
        width=width,
        enabled=source_features_enabled,
        drop_cross_source_score_meta=drop_cross_source_score_meta,
    )
    return (
        x,
        y,
        groups,
        metadata_names + hier_pop_names + tag_chain_names + source_names,
    )


def select_feature_set(
    matrix: np.ndarray,
    base_names: list[str],
    appended_names: list[str],
    base_categorical: list[int],
    feature_set: str,
) -> tuple[np.ndarray, list[str], list[int]]:
    """Select the paper's full/independent/provenance feature ablation."""
    names = base_names + appended_names
    if matrix.shape[1] != len(names):
        raise ValueError(
            f"feature matrix/name mismatch: {matrix.shape[1]} != {len(names)}"
        )
    if feature_set == "provenance_only":
        provenance_columns = [
            i for i, name in enumerate(names) if name.startswith("src_")
        ]
        if not provenance_columns:
            raise ValueError("provenance_only requires source_features.npz columns")
        provenance_names = [names[i] for i in provenance_columns]
        return matrix[:, provenance_columns], provenance_names, []
    if feature_set == "no_provenance":
        independent_columns = [
            i for i, name in enumerate(names) if not name.startswith("src_")
        ]
        independent_names = [names[i] for i in independent_columns]
        categorical = [
            independent_columns.index(i)
            for i in base_categorical
            if i in independent_columns
        ]
        if len(independent_columns) == matrix.shape[1]:
            return matrix, independent_names, categorical
        return matrix[:, independent_columns], independent_names, categorical
    if feature_set != "full":
        raise ValueError(f"unknown feature_set: {feature_set}")
    return matrix, names, list(base_categorical)


def positive_rows(
    examples: list[Any], indices: np.ndarray, rows: np.ndarray, track_index: Any
) -> np.ndarray:
    keep: list[int] = []
    for row_raw in rows:
        row = int(row_raw)
        gold_idx = track_index.id_to_idx.get(examples[row].gold_track_id or "")
        if gold_idx is not None and bool(np.any(indices[row] == gold_idx)):
            keep.append(row)
    return np.asarray(keep, dtype=np.int32)


def rank_from_predictions(
    candidates: Any, groups: list[int], pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


def fit_lgbm_model(
    args: argparse.Namespace,
    x: np.ndarray,
    y: np.ndarray,
    groups: list[int],
    *,
    fold_seed: int,
    categorical_feature: list[int],
    feature_names: list[str],
) -> Any:
    model = lgb.LGBMRanker(
        objective="lambdarank",
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
        x,
        y,
        group=groups,
        categorical_feature=categorical_feature,
        feature_name=feature_names,
    )
    return model


def predict_lgbm_model(model: Any, x: np.ndarray) -> np.ndarray:
    return model.predict(x).astype(np.float32, copy=False)


def save_tree_model(model: Any, path_stem: Path) -> Path:
    path = path_stem.with_suffix(".txt")
    model.booster_.save_model(str(path))
    return path


def subset_list(values: list[Any], rows: np.ndarray) -> list[Any]:
    return [values[int(i)] for i in rows]


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--public-candidates", type=Path, required=True)
    parser.add_argument("--inference-candidates", type=Path, required=True)
    parser.add_argument(
        "--inference-target", choices=("devset", "blind_b"), required=True
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=500,
        help="Candidate width to read; <=0 means use the full artifact width.",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--cv-artifact-mode",
        default="cv5_oof",
        help="Fit-candidate artifact mode recorded in the manifest.",
    )
    parser.add_argument(
        "--feature-set",
        choices=("full", "no_provenance", "provenance_only"),
        default="full",
        help="Paper feature ablation. no_provenance is equivalent to --disable-source-features.",
    )
    parser.add_argument(
        "--drop-cross-source-score-meta",
        action="store_true",
        help="Drop meta__max_source_score__primary from source features. Default keeps features baseline behavior.",
    )
    parser.add_argument(
        "--neutralize-base-features",
        default="",
        help="Comma-separated base feature names to set to zero.",
    )
    parser.add_argument("--extra-metadata-features", action="store_true")
    parser.add_argument("--extra-tag-chain-features", action="store_true")
    parser.add_argument("--extra-hier-pop-features", action="store_true")
    parser.add_argument("--train-positive-only", action="store_true")
    parser.add_argument("--feature-chunk-examples", type=int, default=512)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--subsample", type=float, default=0.85)
    parser.add_argument("--colsample-bytree", type=float, default=0.85)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--lambdarank-truncation-level", type=int, default=None)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=20260518)
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])
    neutralize_base_features = {
        name.strip()
        for name in str(args.neutralize_base_features).split(",")
        if name.strip()
    }

    candidate_dir = (
        args.public_candidates
        if args.public_candidates.is_absolute()
        else REPO_ROOT / args.public_candidates
    )
    features = proto.load_feature_module()
    dense_dir = PREPROCESSED_DIR / "dense_qfeat"

    print("loading tracks/users/examples")
    track_index = features.TrackIndex()
    user_vectors = features.load_user_vectors()
    train_examples0 = features.build_examples_from_dataset("train")
    dev_examples0 = features.build_examples_from_dataset("test")
    base_examples = train_examples0 + dev_examples0
    base_sources = ["train"] * len(train_examples0) + ["devset"] * len(dev_examples0)

    # Dense query embeddings are message-only. Missing rows are encoded on GPU.
    train_dense_artifacts = [dense_dir / "train.npz"]
    train_dense_out = dense_dir / "train.npz"
    dev_dense_artifacts = [dense_dir / "devset.npz"]
    dev_dense_out = dense_dir / "devset.npz"

    print("materializing dense query features")
    train_dense = proto.materialize_dense(
        features,
        train_examples0,
        train_dense_artifacts,
        artifact_out=train_dense_out,
        batch_size=64,
    )
    dev_dense = proto.materialize_dense(
        features,
        dev_examples0,
        dev_dense_artifacts,
        artifact_out=dev_dense_out,
        batch_size=64,
    )
    base_dense = np.concatenate([train_dense, dev_dense], axis=0)
    by_key = {
        key_str(src, ex): (src, ex, base_dense[i])
        for i, (src, ex) in enumerate(zip(base_sources, base_examples, strict=True))
    }

    print("loading union candidates")
    cand_npz = np.load(candidate_dir / "candidates.npz", allow_pickle=False)
    keys = raw_keys(cand_npz["keys"])
    artifact_width = int(cand_npz["track_idx"].shape[1])
    width = (
        artifact_width
        if args.max_candidates <= 0
        else min(args.max_candidates, artifact_width)
    )
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
    primary_scores = choose_primary_scores(cand_npz, width=width)
    public_candidates = make_candidate_set(features, indices, primary_scores)
    inference_candidate_dir = (
        args.inference_candidates
        if args.inference_candidates.is_absolute()
        else REPO_ROOT / args.inference_candidates
    )
    print(f"loading {args.inference_target} examples and candidates")
    if args.inference_target == "devset":
        inference_examples0 = dev_examples0
        inference_dense0 = dev_dense
    else:
        inference_examples0 = features.build_examples_from_inference(
            args.inference_target
        )
        inference_dense0 = proto.materialize_dense(
            features,
            inference_examples0,
            [dense_dir / f"{args.inference_target}.npz"],
            artifact_out=dense_dir / f"{args.inference_target}.npz",
            batch_size=64,
        )
    inference_by_key = {
        inference_key_str(ex): (ex, inference_dense0[i])
        for i, ex in enumerate(inference_examples0)
    }
    inference_npz = np.load(
        inference_candidate_dir / "candidates.npz", allow_pickle=False
    )
    inference_keys = raw_keys(inference_npz["keys"])
    inference_artifact_width = int(inference_npz["track_idx"].shape[1])
    inference_width = (
        inference_artifact_width
        if args.max_candidates <= 0
        else min(args.max_candidates, inference_artifact_width)
    )
    inference_examples = []
    inference_dense_rows = []
    for key in inference_keys:
        ex, dense = inference_by_key[key]
        inference_examples.append(ex)
        inference_dense_rows.append(dense)
    inference_dense = np.asarray(inference_dense_rows, dtype=np.float32)
    inference_indices = np.asarray(inference_npz["track_idx"], dtype=np.int32)[
        :, :inference_width
    ]
    inference_primary_scores = choose_primary_scores(
        inference_npz, width=inference_width
    )
    inference_candidates = make_candidate_set(
        features, inference_indices, inference_primary_scores
    )

    cand_metrics = features.candidate_metrics(
        public_examples, public_candidates, track_index
    )
    print(f"candidate metrics: {cand_metrics}")
    all_rows = np.arange(len(public_examples), dtype=np.int32)

    base_manifest = {
        "schema_version": 1,
        "artifact_type": "ranked",
        "stage": "reranker",
        "name": RERANKER_NAME,
        "config": args.config,
        "producer": {
            "command": [
                "uv",
                "run",
                "python",
                "-m",
                "reranker.union_lambdarank.runner",
            ],
            "cwd": ".",
        },
        "protocol": "docs/folds.md",
        "params": jsonable(vars(args)),
        "source_candidate_artifact": rel(candidate_dir),
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_blind_for_fit": False,
            "uses_devset_score_for_tuning": False,
            "popularity_tiebreaker": False,
            "train_rows_use_oof_candidates_when_required": True,
            "current_thought_allowed": False,
            "conversation_goal_allowed": False,
            "gpa_allowed": False,
            "session_date_allowed": False,
            "intent_features": False,
        },
    }
    print(f"fit final model for {args.inference_target}")
    final_train_rows = all_rows
    if args.train_positive_only:
        final_train_rows = positive_rows(
            public_examples, indices, final_train_rows, track_index
        )
    final_ex = subset_list(public_examples, final_train_rows)
    final_cand = make_candidate_set(
        features, indices[final_train_rows], primary_scores[final_train_rows]
    )
    final_dense = public_dense[final_train_rows]
    encoder, vectorizer, track_tfidf, fast_context = fit_feature_stack(
        features,
        final_ex,
        track_index,
        user_vectors,
    )
    x_train, y_train, groups, source_names = build_rich_matrix(
        features,
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
        source_features_enabled=args.feature_set != "no_provenance",
        drop_cross_source_score_meta=args.drop_cross_source_score_meta,
        extra_metadata_features=args.extra_metadata_features,
        extra_tag_chain_features=args.extra_tag_chain_features,
        extra_hier_pop_features=args.extra_hier_pop_features,
        neutralize_base_features=neutralize_base_features,
        labels=True,
    )
    if y_train is None or int(y_train.sum()) == 0:
        raise RuntimeError("no positive labels for final model")
    x_train, final_feature_names, final_categorical_feature = select_feature_set(
        x_train,
        encoder.feature_names,
        source_names,
        encoder.categorical_feature_indices,
        args.feature_set,
    )
    print(
        f"final LightGBM/LambdaRank rows={len(y_train)} "
        f"positives={int(y_train.sum())} groups={len(groups)} "
        f"features={x_train.shape[1]}"
    )
    final_model = fit_lgbm_model(
        args,
        x_train,
        y_train,
        groups,
        fold_seed=args.seed + 100,
        categorical_feature=final_categorical_feature,
        feature_names=final_feature_names,
    )
    del x_train, y_train, groups
    gc.collect()

    print(f"rank {args.inference_target}")
    inference_rows = np.arange(len(inference_examples), dtype=np.int32)
    x_inference, _, inference_groups, inference_source_names = build_rich_matrix(
        features,
        inference_examples,
        inference_candidates,
        inference_dense,
        encoder,
        vectorizer,
        track_tfidf,
        track_index,
        fast_context,
        inference_candidate_dir,
        inference_rows,
        width=inference_candidates.indices.shape[1],
        feature_chunk_examples=args.feature_chunk_examples,
        source_features_enabled=args.feature_set != "no_provenance",
        drop_cross_source_score_meta=args.drop_cross_source_score_meta,
        extra_metadata_features=args.extra_metadata_features,
        extra_tag_chain_features=args.extra_tag_chain_features,
        extra_hier_pop_features=args.extra_hier_pop_features,
        neutralize_base_features=neutralize_base_features,
        labels=False,
    )
    x_inference, built_feature_names, _ = select_feature_set(
        x_inference,
        encoder.feature_names,
        inference_source_names,
        encoder.categorical_feature_indices,
        args.feature_set,
    )
    if built_feature_names != final_feature_names:
        raise RuntimeError("feature layout mismatch between fit and inference")
    pred = predict_lgbm_model(final_model, x_inference)
    inference_ranked, inference_scores = rank_from_predictions(
        inference_candidates, inference_groups, pred
    )
    inference_sizes = np.asarray(
        [(row >= 0).sum() for row in inference_ranked], dtype=np.int32
    )
    inference_mode = (
        "full_train" if args.inference_target == "devset" else "full_public"
    )
    inference_out = (
        OUTPUT_DIR
        / "reranker"
        / RERANKER_NAME
        / args.config
        / inference_mode
        / args.inference_target
    )
    inference_out.mkdir(parents=True, exist_ok=True)
    model_path = save_tree_model(final_model, inference_out / "model")
    save_ranked_artifact(
        inference_out,
        inference_ranked,
        inference_sizes,
        target=args.inference_target,
        scores=inference_scores,
        manifest={
            **base_manifest,
            "feature_names": final_feature_names,
            "artifact_mode": inference_mode,
            "target": args.inference_target,
            "created_at": utc_now(),
            "fit_candidate_artifact_mode": args.cv_artifact_mode,
            "fit_candidate_artifact": rel(candidate_dir),
            "inference_candidate_artifact_mode": inference_mode,
            "inference_candidate_artifact": rel(inference_candidate_dir),
            "model": file_ref(model_path),
        },
        compress=True,
    )
    inference_scores_dir = (
        RESULTS_DIR
        / "reranker"
        / RERANKER_NAME
        / args.config
        / inference_mode
        / args.inference_target
    )
    target_metrics = (
        proto.evaluate_ranked(
            ["devset"] * len(inference_examples),
            inference_examples,
            inference_ranked,
            track_index,
            top_k=args.top_k,
        )
        if args.inference_target == "devset"
        else {}
    )
    json_dump(
        inference_scores_dir / "scores.json",
        {
            "name": RERANKER_NAME,
            "config": args.config,
            "artifact_mode": inference_mode,
            "target": args.inference_target,
            "fit_candidate_artifact": rel(candidate_dir),
            "inference_candidate_artifact": rel(inference_candidate_dir),
            "reranker_artifact": rel(inference_out),
            "model": rel(model_path),
            "public_candidate_metrics": cand_metrics,
            "target_metrics": target_metrics,
        },
    )
    print(
        json.dumps(
            {
                "scores": rel(inference_scores_dir / "scores.json"),
                "ranked": rel(inference_out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
