#!/usr/bin/env python3
"""Run a protocol-complete CPU baseline.

This is not a byte-for-byte reimplementation of legacy exp098.  It is the
first end-to-end run that obeys the current pipeline protocol:

- fixed public-labeled 3-fold CV over train+devset
- retriever cv3_oof candidates for reranker training/evaluation rows
- retriever full_public candidates for blind inference
- one final reranker fitted on all public rows using cv3_oof candidates
- blind prediction JSON with a cheap template responder

The retriever is TF-IDF over track metadata and does not fit on labeled rows,
so its cv3_oof and full_public score semantics are identical.  The artifact
modes are still materialized separately because downstream rerankers should
not need to special-case fit-free retrievers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from recsys2026.artifacts import (
    encode_keys,
    file_ref,
    json_dump,
    save_candidate_artifact,
    save_ranked_artifact,
    track_id_lookup,
    utc_now,
)
from recsys2026.data import load
from recsys2026.paths import OUTPUT_DIR, REPO_ROOT, RESULTS_DIR
from recsys2026.splits import read_jsonl
from recsys2026.submission import Target, format_record, iter_inputs, validate_predictions, zip_submission


MAX_TURNS = 8


@dataclass(frozen=True)
class Example:
    source_split: Literal["train", "devset", "blind_a", "blind_b"]
    session_id: str
    user_id: str
    turn_number: int
    query_text: str
    history_track_idx: tuple[int, ...]
    history_artist_code: tuple[int, ...]
    history_album_code: tuple[int, ...]
    gold_idx: int
    fold: int


@dataclass
class TrackStore:
    track_ids: list[str]
    id_to_idx: dict[str, int]
    texts: list[str]
    display: list[str]
    artist_code: np.ndarray
    album_code: np.ndarray
    meta_by_id: dict[str, dict[str, Any]]


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def join_field(value: Any) -> str:
    return " ".join(str(x) for x in as_list(value) if x is not None)


def first_field(value: Any) -> str:
    vals = as_list(value)
    return str(vals[0]) if vals else ""


def load_tracks() -> TrackStore:
    rows = list(load("track", split="all_tracks"))
    track_ids = [r["track_id"] for r in rows]
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    artist_ids = [first_field(r.get("artist_id")) for r in rows]
    album_ids = [first_field(r.get("album_id")) for r in rows]
    artist_lookup = {v: i + 1 for i, v in enumerate(sorted({x for x in artist_ids if x}))}
    album_lookup = {v: i + 1 for i, v in enumerate(sorted({x for x in album_ids if x}))}
    texts: list[str] = []
    display: list[str] = []
    for r in rows:
        track = join_field(r.get("track_name"))
        artist = join_field(r.get("artist_name"))
        album = join_field(r.get("album_name"))
        tags = join_field(r.get("tag_list"))
        year = str(r.get("release_date") or "")[:4]
        texts.append(f"{track} {artist} {album} {tags} {year}".strip())
        if artist:
            display.append(f"{track} by {artist}")
        else:
            display.append(track)
    return TrackStore(
        track_ids=track_ids,
        id_to_idx=id_to_idx,
        texts=texts,
        display=display,
        artist_code=np.asarray([artist_lookup.get(x, 0) for x in artist_ids], dtype=np.int32),
        album_code=np.asarray([album_lookup.get(x, 0) for x in album_ids], dtype=np.int32),
        meta_by_id={r["track_id"]: r for r in rows},
    )


def music_text(track_id: str, tracks: TrackStore) -> str:
    row = tracks.meta_by_id.get(track_id)
    if row is None:
        return track_id
    track = join_field(row.get("track_name"))
    artist = join_field(row.get("artist_name"))
    album = join_field(row.get("album_name"))
    if album:
        return f"{track} by {artist} from {album}"
    return f"{track} by {artist}"


def goal_text(item: dict[str, Any]) -> str:
    # blind-B-safe fixed: conversation_goal is never used.
    return ""


def example_query(
    item: dict[str, Any],
    *,
    target_turn: int,
    tracks: TrackStore,
) -> tuple[str, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    conversations = list(item["conversations"])
    current = [c for c in conversations if int(c["turn_number"]) == target_turn]
    user_turn = next(c for c in current if c["role"] == "user")
    _thought = ""  # blind-B-safe fixed: thought is never used
    parts = [str(user_turn.get("content") or ""), _thought, goal_text(item)]
    history_track_idx: list[int] = []
    history_artist_code: list[int] = []
    history_album_code: list[int] = []
    for msg in conversations:
        turn = int(msg["turn_number"])
        if turn >= target_turn:
            continue
        role = msg.get("role")
        content = str(msg.get("content") or "")
        if role in {"user", "assistant"}:
            parts.append(content)
        elif role == "music":
            idx = tracks.id_to_idx.get(content)
            if idx is not None:
                history_track_idx.append(idx)
                history_artist_code.append(int(tracks.artist_code[idx]))
                history_album_code.append(int(tracks.album_code[idx]))
                parts.append(music_text(content, tracks))
    text = " ".join(p for p in parts if p).strip()
    return text, tuple(history_track_idx), tuple(history_artist_code), tuple(history_album_code)


def load_public_examples(split_dir: Path, tracks: TrackStore) -> list[Example]:
    fold_by_key: dict[tuple[str, str], int] = {}
    for row in read_jsonl(split_dir / "sessions.jsonl"):
        fold_by_key[(row["source_split"], row["session_id"])] = int(row["fold"])

    examples: list[Example] = []
    for source_split, dataset_split in (("train", "train"), ("devset", "test")):
        for item in load("dataset", split=dataset_split):
            gold_by_turn = {
                int(c["turn_number"]): str(c["content"])
                for c in item["conversations"]
                if c["role"] == "music"
            }
            fold = fold_by_key[(source_split, item["session_id"])]
            for turn in range(1, MAX_TURNS + 1):
                query, hist_tracks, hist_artists, hist_albums = example_query(item, target_turn=turn, tracks=tracks)
                examples.append(
                    Example(
                        source_split=source_split,  # type: ignore[arg-type]
                        session_id=item["session_id"],
                        user_id=item["user_id"],
                        turn_number=turn,
                        query_text=query,
                        history_track_idx=hist_tracks,
                        history_artist_code=hist_artists,
                        history_album_code=hist_albums,
                        gold_idx=tracks.id_to_idx[gold_by_turn[turn]],
                        fold=fold,
                    )
                )
    return examples


def load_blind_examples(target: Target, tracks: TrackStore) -> list[Example]:
    if target not in ("blind_a", "blind_b"):
        raise ValueError(target)
    examples: list[Example] = []
    for item in load(target, split="test"):
        current = item["conversations"][-1]
        query, hist_tracks, hist_artists, hist_albums = example_query(
            item,
            target_turn=int(current["turn_number"]),
            tracks=tracks,
        )
        examples.append(
            Example(
                source_split=target,  # type: ignore[arg-type]
                session_id=item["session_id"],
                user_id=item["user_id"],
                turn_number=int(current["turn_number"]),
                query_text=query,
                history_track_idx=hist_tracks,
                history_artist_code=hist_artists,
                history_album_code=hist_albums,
                gold_idx=-1,
                fold=-1,
            )
        )
    return examples


def fit_retriever(tracks: TrackStore, *, max_features: int, min_df: int) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_features=max_features,
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
        dtype=np.float32,
    )
    track_matrix = vectorizer.fit_transform(tracks.texts).tocsr()
    return vectorizer, track_matrix


def retrieve_candidates(
    examples: list[Example],
    vectorizer: TfidfVectorizer,
    track_matrix: sparse.csr_matrix,
    *,
    candidate_k: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(examples)
    track_idx = np.full((n, candidate_k), -1, dtype=np.int32)
    scores = np.full((n, candidate_k), -np.inf, dtype=np.float32)
    track_matrix_t = track_matrix.T.tocsr()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        queries = [ex.query_text for ex in examples[start:end]]
        q = vectorizer.transform(queries)
        block = (q @ track_matrix_t).toarray().astype(np.float32, copy=False)
        for local_i, ex in enumerate(examples[start:end]):
            if ex.history_track_idx:
                block[local_i, list(ex.history_track_idx)] = -np.inf
        kk = min(candidate_k, block.shape[1])
        part = np.argpartition(-block, kth=kk - 1, axis=1)[:, :kk]
        part_scores = np.take_along_axis(block, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        idx = np.take_along_axis(part, order, axis=1).astype(np.int32, copy=False)
        sc = np.take_along_axis(part_scores, order, axis=1).astype(np.float32, copy=False)
        track_idx[start:end, :kk] = idx
        scores[start:end, :kk] = sc
        if start == 0 or (start // chunk_size) % 25 == 0:
            print(f"retrieved {end}/{n}")
    sizes = np.full(n, candidate_k, dtype=np.int32)
    return track_idx, scores, sizes


def candidate_metrics(examples: list[Example], track_idx: np.ndarray, sizes: np.ndarray) -> dict[str, Any]:
    gold = np.asarray([ex.gold_idx for ex in examples], dtype=np.int32)
    out: dict[str, Any] = {
        "n_examples": len(examples),
        "mean_size": float(sizes.mean()) if len(sizes) else 0.0,
    }
    for k in (20, 50, 100):
        kk = min(k, track_idx.shape[1])
        hits = (track_idx[:, :kk] == gold[:, None]).any(axis=1)
        out[f"recall@{k}"] = float(hits.mean()) if len(hits) else 0.0
        emitted = int(np.minimum(sizes, kk).sum())
        out[f"precision@{k}"] = float(hits.sum() / emitted) if emitted else 0.0
    hits_all = np.zeros(len(examples), dtype=bool)
    for i, size_raw in enumerate(sizes):
        hits_all[i] = bool((track_idx[i, : int(size_raw)] == gold[i]).any())
    out["recall@all"] = float(hits_all.mean()) if len(hits_all) else 0.0
    return out


def build_features(
    examples: list[Example],
    track_idx: np.ndarray,
    scores: np.ndarray,
    rows: np.ndarray,
    tracks: TrackStore,
) -> np.ndarray:
    cand = track_idx[rows]
    sc = scores[rows]
    n_rows, k = cand.shape
    rank = np.broadcast_to(np.arange(1, k + 1, dtype=np.float32), (n_rows, k))
    finite = np.isfinite(sc)
    safe_score = np.where(finite, sc, 0.0).astype(np.float32, copy=False)
    row_max = safe_score[:, :1]
    row_mean = safe_score.mean(axis=1, keepdims=True)
    row_std = safe_score.std(axis=1, keepdims=True) + 1e-6
    z = (safe_score - row_mean) / row_std
    gap = row_max - safe_score
    cand_artist = tracks.artist_code[cand]
    cand_album = tracks.album_code[cand]
    hist_artist = np.zeros((n_rows, k), dtype=np.float32)
    hist_album = np.zeros((n_rows, k), dtype=np.float32)
    for out_i, row_i in enumerate(rows):
        ex = examples[int(row_i)]
        artist_codes = [c for c in ex.history_artist_code if c > 0]
        album_codes = [c for c in ex.history_album_code if c > 0]
        if artist_codes:
            hist_artist[out_i] = np.isin(cand_artist[out_i], artist_codes)
        if album_codes:
            hist_album[out_i] = np.isin(cand_album[out_i], album_codes)
    feats = np.stack(
        [
            rank,
            np.log1p(rank),
            1.0 / rank,
            safe_score,
            z.astype(np.float32, copy=False),
            gap.astype(np.float32, copy=False),
            hist_artist,
            hist_album,
        ],
        axis=2,
    )
    return feats.reshape(n_rows * k, feats.shape[2]).astype(np.float32, copy=False)


def build_lgbm_data(
    examples: list[Example],
    track_idx: np.ndarray,
    scores: np.ndarray,
    rows: np.ndarray,
    tracks: TrackStore,
) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray]:
    cand = track_idx[rows]
    gold = np.asarray([examples[int(i)].gold_idx for i in rows], dtype=np.int32)
    y_matrix = (cand == gold[:, None])
    keep = y_matrix.any(axis=1)
    kept_rows = rows[keep]
    if len(kept_rows) == 0:
        raise RuntimeError("no positive rows in selected training data")
    x = build_features(examples, track_idx, scores, kept_rows, tracks)
    y = y_matrix[keep].reshape(-1).astype(np.int32)
    group = [cand.shape[1]] * len(kept_rows)
    return x, y, group, kept_rows


def fit_ranker(
    examples: list[Example],
    track_idx: np.ndarray,
    scores: np.ndarray,
    rows: np.ndarray,
    tracks: TrackStore,
    *,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    n_jobs: int,
    seed: int,
) -> lgb.LGBMRanker:
    x, y, group, kept_rows = build_lgbm_data(examples, track_idx, scores, rows, tracks)
    print(
        f"fit ranker rows={len(rows)} positive_groups={len(kept_rows)} "
        f"lgb_rows={len(y)} positives={int(y.sum())}"
    )
    model = lgb.LGBMRanker(
        objective="lambdarank",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=-1,
    )
    model.fit(x, y, group=group)
    return model


def rank_rows(
    model: lgb.LGBMRanker,
    examples: list[Example],
    track_idx: np.ndarray,
    scores: np.ndarray,
    rows: np.ndarray,
    tracks: TrackStore,
    *,
    chunk_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    k = track_idx.shape[1]
    ranked = np.full((len(rows), k), -1, dtype=np.int32)
    ranked_scores = np.full((len(rows), k), np.nan, dtype=np.float32)
    for start in range(0, len(rows), chunk_rows):
        end = min(start + chunk_rows, len(rows))
        sub_rows = rows[start:end]
        x = build_features(examples, track_idx, scores, sub_rows, tracks)
        pred = model.predict(x).reshape(len(sub_rows), k).astype(np.float32, copy=False)
        order = np.argsort(-pred, axis=1, kind="stable")
        ranked[start:end] = np.take_along_axis(track_idx[sub_rows], order, axis=1)
        ranked_scores[start:end] = np.take_along_axis(pred, order, axis=1)
        if start == 0 or (start // chunk_rows) % 20 == 0:
            print(f"ranked {end}/{len(rows)}")
    return ranked, ranked_scores


def ndcg_at_pos(pos: int, k: int) -> float:
    if pos < 0 or pos >= k:
        return 0.0
    return 1.0 / math.log2(pos + 2)


def evaluate_ranked_public(examples: list[Example], ranked: np.ndarray, *, top_k: int = 20) -> dict[str, Any]:
    by_turn: dict[int, list[dict[str, float]]] = defaultdict(list)
    by_source: dict[str, list[float]] = defaultdict(list)
    for ex, row in zip(examples, ranked, strict=True):
        pos_arr = np.flatnonzero(row[:top_k] == ex.gold_idx)
        pos = int(pos_arr[0]) if len(pos_arr) else -1
        vals = {
            "ndcg@1": ndcg_at_pos(pos, 1),
            "ndcg@10": ndcg_at_pos(pos, 10),
            "ndcg@20": ndcg_at_pos(pos, 20),
        }
        by_turn[ex.turn_number].append(vals)
        by_source[ex.source_split].append(vals["ndcg@20"])
    turn_means = {
        turn: {name: sum(v[name] for v in vals) / len(vals) for name in ("ndcg@1", "ndcg@10", "ndcg@20")}
        for turn, vals in by_turn.items()
    }
    out = {
        name: sum(vals[name] for vals in turn_means.values()) / len(turn_means)
        for name in ("ndcg@1", "ndcg@10", "ndcg@20")
    }
    out["n_examples"] = len(examples)
    for source, vals in by_source.items():
        out[f"{source}_ndcg@20"] = float(sum(vals) / len(vals))
    return out


def save_public_candidates(
    out_dir: Path,
    examples: list[Example],
    track_idx: np.ndarray,
    scores: np.ndarray,
    sizes: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [(f"{ex.source_split}:{ex.session_id}", ex.turn_number) for ex in examples]
    folds = np.asarray([ex.fold for ex in examples], dtype=np.int16)
    np.savez_compressed(
        out_dir / "candidates.npz",
        track_idx=track_idx,
        sizes=sizes,
        keys=encode_keys(keys),
        folds=folds,
        score__tfidf=scores.astype(np.float32, copy=False),
        rank=np.broadcast_to(np.arange(1, track_idx.shape[1] + 1, dtype=np.int32), track_idx.shape),
    )
    with (out_dir / "turns.jsonl").open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            f.write(
                json.dumps(
                    {
                        "row_id": i,
                        "source_split": ex.source_split,
                        "session_id": ex.session_id,
                        "user_id": ex.user_id,
                        "turn_number": ex.turn_number,
                        "fold": ex.fold,
                        "gold_track_idx": ex.gold_idx,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    json_dump(out_dir / "manifest.json", manifest)


def save_public_ranked(
    out_dir: Path,
    examples: list[Example],
    ranked: np.ndarray,
    ranked_scores: np.ndarray,
    manifest: dict[str, Any],
    tracks: TrackStore,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [(f"{ex.source_split}:{ex.session_id}", ex.turn_number) for ex in examples]
    sizes = np.full(len(examples), ranked.shape[1], dtype=np.int32)
    np.savez_compressed(
        out_dir / "ranked.npz",
        track_idx=ranked.astype(np.int32, copy=False),
        sizes=sizes,
        keys=encode_keys(keys),
        folds=np.asarray([ex.fold for ex in examples], dtype=np.int16),
        scores=ranked_scores.astype(np.float32, copy=False),
    )
    with (out_dir / "ranked_top100.jsonl").open("w", encoding="utf-8") as f:
        for ex, row in zip(examples, ranked, strict=True):
            f.write(
                json.dumps(
                    {
                        "source_split": ex.source_split,
                        "session_id": ex.session_id,
                        "turn_number": ex.turn_number,
                        "ranked_track_ids": [tracks.track_ids[int(i)] for i in row[:100]],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    json_dump(out_dir / "manifest.json", manifest)


def save_blind_candidates(
    out_dir: Path,
    target: Target,
    track_idx: np.ndarray,
    scores: np.ndarray,
    sizes: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    save_candidate_artifact(
        out_dir,
        track_idx,
        sizes,
        target=target,
        manifest=manifest,
        score_arrays={"tfidf": scores.astype(np.float32, copy=False)},
        rank=np.broadcast_to(np.arange(1, track_idx.shape[1] + 1, dtype=np.int32), track_idx.shape),
        compress=True,
    )


def write_template_predictions(
    ranked_dir: Path,
    target: Target,
    tracks: TrackStore,
    out_json: Path,
    *,
    top_k: int,
) -> Path:
    arrays = np.load(ranked_dir / "ranked.npz")
    ranked = arrays["track_idx"]
    records: list[dict[str, Any]] = []
    for inp, row in zip(iter_inputs(target), ranked, strict=True):
        tids = [tracks.track_ids[int(i)] for i in row[:top_k]]
        top_display = tracks.display[int(row[0])]
        response = f"I picked {top_display} and related tracks that match your request."
        records.append(format_record(inp, tids, response))
    validate_predictions(records, target)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, ensure_ascii=False))
    return zip_submission(out_json)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value) if value.is_absolute() and value.is_relative_to(REPO_ROOT) else str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="protocol_tfidf_lgbm")
    parser.add_argument("--config", default="protocol_v1")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "artifacts" / "cache" / "splits" / "cv5")
    parser.add_argument("--blind-target", choices=("blind_a", "blind_b"), default="blind_a")
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--final-k", type=int, default=20)
    parser.add_argument("--max-features", type=int, default=200_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--retrieval-chunk-size", type=int, default=512)
    parser.add_argument("--rank-chunk-rows", type=int, default=4096)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    split_dir = args.split_dir if args.split_dir.is_absolute() else REPO_ROOT / args.split_dir
    if not (split_dir / "sessions.jsonl").exists():
        raise FileNotFoundError(f"missing split artifact: {split_dir}. Run `just build-public-splits` first.")

    print("loading tracks")
    tracks = load_tracks()
    print("loading public examples")
    public_examples = load_public_examples(split_dir, tracks)
    print(f"public examples={len(public_examples)}")
    print("loading blind examples")
    blind_target: Target = args.blind_target  # type: ignore[assignment]
    blind_examples = load_blind_examples(blind_target, tracks)
    print(f"{blind_target} examples={len(blind_examples)}")

    print("fitting fit-free TF-IDF retriever over track metadata")
    vectorizer, track_matrix = fit_retriever(tracks, max_features=args.max_features, min_df=args.min_df)

    base_manifest = {
        "schema_version": 1,
        "producer": {
            "command": ["uv", "run", "python", "scripts/run_protocol_tfidf_lgbm_baseline.py"],
            "cwd": ".",
        },
        "protocol": "docs/pipeline_cv_protocol.md",
        "split_artifact": rel(split_dir),
        "params": jsonable(vars(args)),
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_blind_for_fit": False,
            "popularity_tiebreaker": False,
            "retriever_fits_on_labeled_rows": False,
        },
    }
    retriever_source_policy = {
        "requires_labeled_fit": False,
        "fit_sources": ["track_metadata"],
        "train_row_policy": "safe_in_sample",
        "fold_split_required_for_reranker_train": False,
        "preferred_train_row_artifact_mode": "fit_free_all_rows",
        "preferred_inference_artifact_mode": "fit_free_all_rows",
        "reason": "TF-IDF vectorizer is fit on track metadata only, not on train/dev labels.",
    }
    retriever_fit_scope = {
        "fit_mode": "fit_free",
        "fit_splits": [],
        "requires_labeled_fit": False,
        "fit_sources": retriever_source_policy["fit_sources"],
        "train_row_policy": "safe_in_sample",
        "fold_split_required_for_reranker_train": False,
        "preferred_train_row_artifact_mode": "fit_free_all_rows",
        "preferred_inference_artifact_mode": "fit_free_all_rows",
        "uses_devset_for_fit": False,
        "uses_blind_for_fit": False,
        "note": "TF-IDF is fit on track metadata only; fold splitting is not required for reranker training.",
    }

    retriever_public_dir = OUTPUT_DIR / "retriever" / args.name / args.config / "fit_free_all_rows" / "public_labeled"
    public_npz = retriever_public_dir / "candidates.npz"
    if public_npz.exists():
        print(f"loading existing public candidates from {public_npz}")
        data = np.load(public_npz)
        public_cand = data["track_idx"]
        public_scores = data["score__tfidf"]
        public_sizes = data["sizes"]
    else:
        print("retrieving public cv3_oof candidates")
        public_cand, public_scores, public_sizes = retrieve_candidates(
            public_examples,
            vectorizer,
            track_matrix,
            candidate_k=args.candidate_k,
            chunk_size=args.retrieval_chunk_size,
        )
    save_public_candidates(
        retriever_public_dir,
        public_examples,
        public_cand,
        public_scores,
        public_sizes,
        {
            **base_manifest,
            "artifact_type": "candidates",
            "stage": "retriever",
            "name": args.name,
            "config": args.config,
            "artifact_mode": "fit_free_all_rows",
            "target": "public_labeled",
            "created_at": utc_now(),
            "fit_scope": retriever_fit_scope,
            "source_policy": retriever_source_policy,
        },
    )
    public_candidate_metrics = candidate_metrics(public_examples, public_cand, public_sizes)
    print(f"public candidate metrics: {public_candidate_metrics}")

    fold_ranked = np.full_like(public_cand, -1)
    fold_ranked_scores = np.full(public_cand.shape, np.nan, dtype=np.float32)
    folds = np.asarray([ex.fold for ex in public_examples], dtype=np.int16)
    all_rows = np.arange(len(public_examples), dtype=np.int32)
    fold_metrics: dict[str, Any] = {}
    for fold in range(3):
        print(f"CV fold {fold}: fit reranker")
        train_rows = all_rows[folds != fold]
        valid_rows = all_rows[folds == fold]
        model = fit_ranker(
            public_examples,
            public_cand,
            public_scores,
            train_rows,
            tracks,
            n_estimators=args.n_estimators,
            num_leaves=args.num_leaves,
            learning_rate=args.learning_rate,
            n_jobs=args.n_jobs,
            seed=args.seed + fold,
        )
        print(f"CV fold {fold}: rank held-out")
        ranked, ranked_scores = rank_rows(
            model,
            public_examples,
            public_cand,
            public_scores,
            valid_rows,
            tracks,
            chunk_rows=args.rank_chunk_rows,
        )
        fold_ranked[valid_rows] = ranked
        fold_ranked_scores[valid_rows] = ranked_scores
        fold_eval = evaluate_ranked_public([public_examples[int(i)] for i in valid_rows], ranked, top_k=args.final_k)
        fold_metrics[f"fold{fold}"] = fold_eval
        print(f"CV fold {fold}: {fold_eval}")

    cv_metrics = evaluate_ranked_public(public_examples, fold_ranked, top_k=args.final_k)
    print(f"CV combined: {cv_metrics}")
    reranker_cv_dir = OUTPUT_DIR / "reranker" / args.name / args.config / "cv3_oof" / "public_labeled"
    save_public_ranked(
        reranker_cv_dir,
        public_examples,
        fold_ranked,
        fold_ranked_scores,
        {
            **base_manifest,
            "artifact_type": "ranked",
            "stage": "reranker",
            "name": args.name,
            "config": args.config,
            "artifact_mode": "cv3_oof",
            "target": "public_labeled",
            "created_at": utc_now(),
            "retriever_artifact": rel(retriever_public_dir),
            "cv_metrics": cv_metrics,
        },
        tracks,
    )
    cv_results_dir = RESULTS_DIR / "reranker" / args.name / args.config / "cv3_oof" / "public_labeled"
    json_dump(
        cv_results_dir / "scores.json",
        {
            "name": args.name,
            "config": args.config,
            "artifact_mode": "cv3_oof",
            "target": "public_labeled",
            "candidate_metrics": public_candidate_metrics,
            "cv_metrics": cv_metrics,
            "fold_metrics": fold_metrics,
            "retriever_artifact": rel(retriever_public_dir),
            "reranker_artifact": rel(reranker_cv_dir),
        },
    )

    print("fit final reranker on all public rows using cv3_oof candidates")
    final_model = fit_ranker(
        public_examples,
        public_cand,
        public_scores,
        all_rows,
        tracks,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )

    retriever_blind_dir = OUTPUT_DIR / "retriever" / args.name / args.config / "fit_free_all_rows" / blind_target
    blind_npz = retriever_blind_dir / "candidates.npz"
    if blind_npz.exists():
        print(f"loading existing {blind_target} candidates from {blind_npz}")
        data = np.load(blind_npz)
        blind_cand = data["track_idx"]
        blind_scores = data["score__tfidf"]
        blind_sizes = data["sizes"]
    else:
        print(f"retrieving {blind_target} full_public candidates")
        blind_cand, blind_scores, blind_sizes = retrieve_candidates(
            blind_examples,
            vectorizer,
            track_matrix,
            candidate_k=args.candidate_k,
            chunk_size=args.retrieval_chunk_size,
        )
    save_blind_candidates(
        retriever_blind_dir,
        blind_target,
        blind_cand,
        blind_scores,
        blind_sizes,
        {
            **base_manifest,
            "artifact_type": "candidates",
            "stage": "retriever",
            "name": args.name,
            "config": args.config,
            "artifact_mode": "fit_free_all_rows",
            "target": blind_target,
            "created_at": utc_now(),
            "fit_scope": retriever_fit_scope,
            "source_policy": retriever_source_policy,
        },
    )

    print(f"ranking {blind_target}")
    blind_rows = np.arange(len(blind_examples), dtype=np.int32)
    blind_ranked, blind_ranked_scores = rank_rows(
        final_model,
        blind_examples,
        blind_cand,
        blind_scores,
        blind_rows,
        tracks,
        chunk_rows=args.rank_chunk_rows,
    )
    reranker_blind_dir = OUTPUT_DIR / "reranker" / args.name / args.config / "full_public" / blind_target
    save_ranked_artifact(
        reranker_blind_dir,
        blind_ranked,
        np.full(len(blind_examples), blind_ranked.shape[1], dtype=np.int32),
        target=blind_target,
        manifest={
            **base_manifest,
            "artifact_type": "ranked",
            "stage": "reranker",
            "name": args.name,
            "config": args.config,
            "artifact_mode": "full_public",
            "target": blind_target,
            "created_at": utc_now(),
            "fit_candidate_artifact_mode": "cv3_oof",
            "fit_candidate_artifact": rel(retriever_public_dir),
            "inference_candidate_artifact_mode": "full_public",
            "inference_candidate_artifact": rel(retriever_blind_dir),
        },
        scores=blind_ranked_scores,
        compress=True,
    )

    responder_dir = OUTPUT_DIR / "responder" / "template_top_track" / args.name / blind_target
    pred_json = responder_dir / "prediction.json"
    zip_path = write_template_predictions(reranker_blind_dir, blind_target, tracks, pred_json, top_k=args.final_k)
    json_dump(
        responder_dir / "manifest.json",
        {
            **base_manifest,
            "artifact_type": "predictions",
            "stage": "responder",
            "name": "template_top_track",
            "config": args.name,
            "target": blind_target,
            "created_at": utc_now(),
            "ranked_artifact": rel(reranker_blind_dir),
            "outputs": {"json": rel(pred_json), "zip": rel(zip_path)},
        },
    )

    pipeline_cfg = REPO_ROOT / "pipeline" / "configs" / "protocol_tfidf_lgbm_template.yaml"
    pipeline_cfg.write_text(
        "\n".join(
            [
                "description: >",
                "  Protocol-complete CPU baseline: TF-IDF metadata retriever, LGBM lambdarank reranker, template responder.",
                f"top_k: {args.final_k}",
                "split_dir: artifacts/cache/splits/cv3",
                "ranked_artifact:",
                f"  blind_a: {rel(reranker_blind_dir)}",
                "prediction_artifact:",
                f"  blind_a: {rel(pred_json)}",
                "assert_track_ids_match_ranked: true",
                "components:",
                "  retriever:",
                f"    name: {args.name}",
                f"    config: {args.config}",
                "    fit_artifact_mode: cv3_oof",
                "    inference_artifact_mode: full_public",
                "  reranker:",
                f"    name: {args.name}",
                f"    config: {args.config}",
                "    fit_mode: full_public_with_cv3_oof_candidates",
                "  responder:",
                "    name: template_top_track",
                f"    config: {args.name}",
                "protocol:",
                "  current_policy: docs/pipeline_cv_protocol.md",
                "  public_labeled_split: artifacts/cache/splits/cv3",
                "  status: protocol_complete_baseline",
                "",
            ]
        )
    )
    print(f"wrote {pipeline_cfg}")
    print(f"wrote {pred_json}")
    print(f"wrote {zip_path}")
    print(json.dumps({"cv_metrics": cv_metrics, "candidate_metrics": public_candidate_metrics}, indent=2))


if __name__ == "__main__":
    main()
