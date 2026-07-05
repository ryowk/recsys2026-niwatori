#!/usr/bin/env python3
"""Vectorized feature builder for the protocol-098 reranker feature stack.

The original 098 builder emits one Python list per candidate row.  At protocol
scale this means roughly 26M Python-row constructions per CV train fold.  This
module keeps a Python loop only over query rows and fills the 300 candidate rows
with NumPy/SciPy vector operations.

Supported scope is intentionally narrow:

- negatives_per_group is None
- intent_lookup is None
- 098 FeatureEncoder / TrackIndex objects

That covers the current protocol_098 early-stopping and model-family sweeps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import sparse


def _bsafe() -> bool:
    """Blind-B-safe fixed: prior GPA + goal categoricals are neutralized in the
    098 base features (columns are kept, values are blanked)."""
    return True


@dataclass
class FastFeatureContext:
    legacy: Any
    encoder: Any
    artist_code_candidate: np.ndarray
    artist_code_history: np.ndarray
    album_code_candidate: np.ndarray
    album_code_history: np.ndarray
    primary_tag_code_candidate: np.ndarray
    primary_tag_code_history: np.ndarray
    n_artist_codes: int
    n_album_codes: int
    n_primary_tag_codes: int
    tag_token_to_id: dict[str, int]
    track_tag_csr: sparse.csr_matrix
    log_duration: np.ndarray


def _codes(candidate_values: list[str], history_values: list[str]) -> tuple[np.ndarray, np.ndarray, int]:
    values = sorted(set(candidate_values) | set(history_values))
    mapping = {value: i for i, value in enumerate(values)}
    cand = np.asarray([mapping[v] for v in candidate_values], dtype=np.int32)
    hist = np.asarray([mapping[v] for v in history_values], dtype=np.int32)
    return cand, hist, len(values)


def make_fast_context(legacy: Any, encoder: Any) -> FastFeatureContext:
    ti = encoder.track_index
    missing = legacy.MISSING_CAT

    artist_candidate = [str(v or missing) for v in ti.artist_name]
    artist_history = [str(v) for v in ti.artist_name]
    album_candidate = [str(v or missing) for v in ti.album_name]
    album_history = [str(v) for v in ti.album_name]
    tag_candidate = [str(v or missing) for v in ti.primary_tag]
    tag_history = [str(v) for v in ti.primary_tag]

    artist_code_candidate, artist_code_history, n_artist = _codes(artist_candidate, artist_history)
    album_code_candidate, album_code_history, n_album = _codes(album_candidate, album_history)
    tag_code_candidate, tag_code_history, n_tag = _codes(tag_candidate, tag_history)

    tag_token_to_id: dict[str, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for track_i, tag_tokens in enumerate(ti.tag_tokens):
        for tok in tag_tokens:
            col = tag_token_to_id.setdefault(tok, len(tag_token_to_id))
            rows.append(track_i)
            cols.append(col)
    data = np.ones(len(rows), dtype=np.float32)
    track_tag_csr = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(ti.n_tracks, len(tag_token_to_id)),
        dtype=np.float32,
    )

    return FastFeatureContext(
        legacy=legacy,
        encoder=encoder,
        artist_code_candidate=artist_code_candidate,
        artist_code_history=artist_code_history,
        album_code_candidate=album_code_candidate,
        album_code_history=album_code_history,
        primary_tag_code_candidate=tag_code_candidate,
        primary_tag_code_history=tag_code_history,
        n_artist_codes=n_artist,
        n_album_codes=n_album,
        n_primary_tag_codes=n_tag,
        tag_token_to_id=tag_token_to_id,
        track_tag_csr=track_tag_csr,
        log_duration=np.log1p(np.asarray(ti.duration, dtype=np.float32)),
    )


def _token_overlap(
    ctx: FastFeatureContext,
    cand_idx: np.ndarray,
    token_set: set[str],
    cand_tag_rows: sparse.csr_matrix | None = None,
) -> np.ndarray:
    if not token_set or not ctx.tag_token_to_id:
        return np.zeros(len(cand_idx), dtype=np.float32)
    token_ids = [ctx.tag_token_to_id[t] for t in token_set if t in ctx.tag_token_to_id]
    if not token_ids:
        return np.zeros(len(cand_idx), dtype=np.float32)
    rows = cand_tag_rows if cand_tag_rows is not None else ctx.track_tag_csr[cand_idx]
    return np.asarray(rows[:, token_ids].sum(axis=1)).ravel().astype(np.float32, copy=False)


def _history_tag_overlap(
    ctx: FastFeatureContext,
    cand_idx: np.ndarray,
    history_indices: list[int],
    cand_tag_rows: sparse.csr_matrix | None = None,
) -> np.ndarray:
    if not history_indices or not ctx.tag_token_to_id:
        return np.zeros(len(cand_idx), dtype=np.float32)
    hist_counts = np.asarray(ctx.track_tag_csr[np.asarray(history_indices, dtype=np.int32)].sum(axis=0)).ravel().astype(np.float32, copy=False)
    if not np.any(hist_counts):
        return np.zeros(len(cand_idx), dtype=np.float32)
    rows = cand_tag_rows if cand_tag_rows is not None else ctx.track_tag_csr[cand_idx]
    return np.asarray(rows.dot(hist_counts)).ravel().astype(np.float32, copy=False)


def _count_by_code(history_codes: np.ndarray, n_codes: int, candidate_codes: np.ndarray) -> np.ndarray:
    if len(history_codes) == 0:
        return np.zeros(len(candidate_codes), dtype=np.float32)
    counts = np.bincount(history_codes, minlength=n_codes)
    return counts[candidate_codes].astype(np.float32, copy=False)


def build_feature_matrix_fast(
    legacy: Any,
    examples: list[Any],
    candidates: Any,
    encoder: Any,
    vectorizer: Any,
    track_tfidf: Any,
    negatives_per_group: int | None,
    chunk_examples: int,
    query_dense_emb: np.ndarray | None = None,
    n_bm25: int | None = None,
    intent_lookup: dict[str, dict[str, str]] | None = None,
    fast_context: FastFeatureContext | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[int]]:
    if negatives_per_group is not None:
        raise NotImplementedError("fast 098 builder only supports negatives_per_group=None")
    if intent_lookup is not None:
        raise NotImplementedError("fast 098 builder only supports intent_lookup=None")

    ctx = fast_context if fast_context is not None else make_fast_context(legacy, encoder)
    ti = encoder.track_index
    n_features = len(encoder.feature_names)
    valid_mask = candidates.indices >= 0
    group_sizes = valid_mask.sum(axis=1).astype(np.int32).tolist()
    total_rows = int(valid_mask.sum())
    x = np.empty((total_rows, n_features), dtype=np.float32)
    has_labels = any(ex.gold_track_id is not None for ex in examples)
    y = np.empty(total_rows, dtype=np.int8) if has_labels else None

    rank_all = np.arange(1, candidates.indices.shape[1] + 1, dtype=np.float32)
    log_rank_all = np.log1p(rank_all).astype(np.float32)
    reciprocal_rank_all = (1.0 / rank_all).astype(np.float32)
    dense_only_all = (
        (rank_all - 1.0 >= float(n_bm25)).astype(np.float32)
        if n_bm25 is not None
        else np.zeros_like(rank_all, dtype=np.float32)
    )

    out_pos = 0
    for start in range(0, len(examples), chunk_examples):
        end = min(start + chunk_examples, len(examples))
        ex_chunk = examples[start:end]
        goal_texts = [legacy.goal_text(ex.conversation_goal) for ex in ex_chunk]
        conv_texts = [legacy.conversation_text(ex, ti) for ex in ex_chunk]
        query_texts = [
            legacy._query_plus_goal(ex.user_query, ex.conversation_goal, ex.user_query_thought)
            for ex in ex_chunk
        ]
        tfidf_vecs = vectorizer.transform(goal_texts + conv_texts + query_texts)
        chunk_len = len(ex_chunk)
        goal_vecs = tfidf_vecs[:chunk_len]
        conv_vecs = tfidf_vecs[chunk_len : 2 * chunk_len]
        query_vecs = tfidf_vecs[2 * chunk_len :]

        for local_i, ex in enumerate(ex_chunk):
            global_i = start + local_i
            positions = np.flatnonzero(valid_mask[global_i])
            if len(positions) == 0:
                continue
            cand_idx = candidates.indices[global_i, positions].astype(np.int32, copy=False)
            m = len(cand_idx)
            sl = slice(out_pos, out_pos + m)
            row = x[sl]
            cand_tag_rows = ctx.track_tag_csr[cand_idx]
            sims = (
                sparse.vstack([goal_vecs[local_i], conv_vecs[local_i], query_vecs[local_i]])
                @ track_tfidf[cand_idx].T
            ).toarray()
            goal_sim = sims[0]
            conv_sim = sims[1]
            query_sim = sims[2]

            history_tracks = [
                str(msg.get("content"))
                for msg in ex.chat_history
                if msg.get("role") == "music"
            ]
            history_indices = [ti.id_to_idx[tid] for tid in history_tracks if tid in ti.id_to_idx]
            history_arr = np.asarray(history_indices, dtype=np.int32)

            if len(history_indices):
                hist_cf_arr = ti.cf[history_arr]
                hist_centroid = hist_cf_arr.mean(axis=0)
                hcn = float(np.linalg.norm(hist_centroid))
                hist_centroid = hist_centroid / hcn if hcn > 0 else None
                history_year_mean = float(np.mean(ti.release_year[history_arr]))
            else:
                hist_centroid = None
                history_year_mean = 0.0

            query_dense_vec = query_dense_emb[global_i] if query_dense_emb is not None else None
            if query_dense_vec is not None and ti.dense_emb is not None and len(history_indices):
                hist_dense_arr = ti.dense_emb[history_arr]
                hist_dense_centroid = hist_dense_arr.mean(axis=0)
                hdn = float(np.linalg.norm(hist_dense_centroid))
                if hdn > 0:
                    hist_dense_centroid = hist_dense_centroid / hdn
                    query_history_dense_cos = float(query_dense_vec @ hist_dense_centroid)
                else:
                    query_history_dense_cos = 0.0
            else:
                query_history_dense_cos = 0.0

            context_tokens = legacy.tokens(legacy.goal_text(ex.conversation_goal) + " " + legacy.conversation_text(ex))
            profile = ex.user_profile
            # Blind-B-safe: prior GPA is unavailable in Blind B → blank it so the
            # prior_gpa_count / moves / not_move / null features become 0 (train+inference consistent).
            prior = [] if _bsafe() else [p for p in ex.prior_goal_progress]
            prior_moves = sum(1 for p in prior if str(p) == "MOVES_TOWARD_GOAL")
            prior_not = sum(1 for p in prior if str(p) == "DOES_NOT_MOVE_TOWARD_GOAL")
            prior_null = sum(1 for p in prior if str(p) in {"None", ""})
            example_cats = encoder.example_categories(ex)
            culture_tokens = legacy.tokens(str(profile.get("preferred_musical_culture") or ""))
            country_tokens = legacy.tokens(str(profile.get("country_name") or ""))
            lang_tokens = legacy.tokens(str(profile.get("preferred_language") or ""))

            cand_artist_codes = ctx.artist_code_candidate[cand_idx]
            cand_album_codes = ctx.album_code_candidate[cand_idx]
            cand_tag_codes = ctx.primary_tag_code_candidate[cand_idx]
            hist_artist_codes = ctx.artist_code_history[history_arr] if len(history_arr) else np.empty(0, dtype=np.int32)
            hist_album_codes = ctx.album_code_history[history_arr] if len(history_arr) else np.empty(0, dtype=np.int32)
            hist_tag_codes = ctx.primary_tag_code_history[history_arr] if len(history_arr) else np.empty(0, dtype=np.int32)

            pos_rank = rank_all[positions]
            row[:, 0] = pos_rank
            row[:, 1] = log_rank_all[positions]
            row[:, 2] = reciprocal_rank_all[positions]
            row[:, 3] = candidates.scores[global_i, positions]
            row[:, 4] = ti.popularity[cand_idx]
            row[:, 5] = ctx.log_duration[cand_idx]
            row[:, 6] = ti.release_year[cand_idx]
            row[:, 7] = float(profile.get("age") or 0.0)
            row[:, 8] = float(ex.turn_number)
            row[:, 9] = float(len(history_indices))
            row[:, 10] = _count_by_code(hist_artist_codes, ctx.n_artist_codes, cand_artist_codes)
            row[:, 11] = _count_by_code(hist_album_codes, ctx.n_album_codes, cand_album_codes)
            if len(history_arr):
                row[:, 12] = np.isin(cand_idx, history_arr, assume_unique=False).astype(np.float32)
            else:
                row[:, 12] = 0.0
            row[:, 13] = float(len(prior))
            row[:, 14] = float(prior_moves)
            row[:, 15] = float(prior_not)
            row[:, 16] = float(prior_null)
            row[:, 17] = goal_sim
            row[:, 18] = conv_sim
            row[:, 19] = query_sim
            row[:, 20] = _token_overlap(ctx, cand_idx, context_tokens, cand_tag_rows)
            row[:, 21] = float(ex.user_id in encoder.user_vectors)
            row[:, 22] = _count_by_code(hist_tag_codes, ctx.n_primary_tag_codes, cand_tag_codes)
            row[:, 23] = _history_tag_overlap(ctx, cand_idx, history_indices, cand_tag_rows)
            if len(history_indices):
                last_idx = history_indices[-1]
                row[:, 24] = (cand_artist_codes == ctx.artist_code_history[last_idx]).astype(np.float32)
                row[:, 25] = (cand_album_codes == ctx.album_code_history[last_idx]).astype(np.float32)
            else:
                row[:, 24] = 0.0
                row[:, 25] = 0.0
            if hist_centroid is not None:
                row[:, 26] = ti.cf[cand_idx] @ hist_centroid
            else:
                row[:, 26] = 0.0
            if len(history_indices) and history_year_mean > 0:
                cand_year = ti.release_year[cand_idx]
                row[:, 27] = np.where(cand_year > 0, np.abs(cand_year - history_year_mean), 0.0)
            else:
                row[:, 27] = 0.0
            row[:, 28] = _token_overlap(ctx, cand_idx, culture_tokens, cand_tag_rows)
            row[:, 29] = _token_overlap(ctx, cand_idx, country_tokens, cand_tag_rows)
            row[:, 30] = _token_overlap(ctx, cand_idx, lang_tokens, cand_tag_rows)
            if query_dense_vec is not None and ti.dense_emb is not None:
                row[:, 31] = ti.dense_emb[cand_idx] @ query_dense_vec
            else:
                row[:, 31] = 0.0
            row[:, 32] = query_history_dense_cos
            row[:, 33] = dense_only_all[positions]
            row[:, 34:39] = 0.0
            # Blind-B-safe: conversation_goal + GPA unavailable in Blind B → MISSING_CAT
            # (consistent train+inference; not zeroed, which would collide with a real category code).
            if _bsafe():
                _mc = legacy.MISSING_CAT
                row[:, 39] = encoder.encode_cat("goal_category", _mc)
                row[:, 40] = encoder.encode_cat("goal_specificity", _mc)
                row[:, 41] = encoder.encode_cat("latest_goal_progress", _mc)
            else:
                row[:, 39] = encoder.encode_cat("goal_category", example_cats.get("goal_category", legacy.MISSING_CAT))
                row[:, 40] = encoder.encode_cat("goal_specificity", example_cats.get("goal_specificity", legacy.MISSING_CAT))
                row[:, 41] = encoder.encode_cat("latest_goal_progress", example_cats.get("latest_goal_progress", legacy.MISSING_CAT))

            if y is not None:
                gold_idx = ti.id_to_idx.get(ex.gold_track_id or "")
                y[sl] = (cand_idx == gold_idx).astype(np.int8) if gold_idx is not None else 0
            out_pos += m

    return x, y, group_sizes
