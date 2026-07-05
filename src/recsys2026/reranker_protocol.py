#!/usr/bin/env python3
"""Candidate + dense-feature materialization and evaluation helpers for the reranker.

Provides `load_098_module` (the reranker feature / example library),
`materialize_dense` (dense query-feature cache), `make_candidate_set`,
`evaluate_ranked`, and split / key helpers used by `scripts/run_reranker.py`.
Public labeled data = train + devset; supervised retriever features for train
rows use out-of-fold artifacts. Intent features are kept off.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from recsys2026.artifacts import (
    encode_keys,
    file_ref,
    json_dump,
    save_candidate_artifact,
    save_ranked_artifact,
    utc_now,
)
from recsys2026.paths import OUTPUT_DIR, REPO_ROOT, RESULTS_DIR
from recsys2026.splits import read_jsonl
from recsys2026.submission import Target, format_record, iter_inputs, validate_predictions, zip_submission


LegacyModule = Any


def load_098_module() -> LegacyModule:
    from recsys2026 import reranker_features

    return reranker_features


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(REPO_ROOT))


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return rel(value)
        except ValueError:
            return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def key(ex: Any) -> str:
    return f"{ex.session_id}:{ex.turn_number}"


def public_key(source: str, ex: Any) -> tuple[str, int]:
    return (f"{source}:{ex.session_id}", int(ex.turn_number))


def load_fold_map(split_dir: Path) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for row in read_jsonl(split_dir / "sessions.jsonl"):
        out[(str(row["source_split"]), str(row["session_id"]))] = int(row["fold"])
    return out


def load_cache_map(paths: list[Path], *, value_names: tuple[str, ...]) -> dict[str, tuple[np.ndarray, ...]]:
    out: dict[str, tuple[np.ndarray, ...]] = {}
    for path in paths:
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        keys = [str(x) for x in data["keys"]]
        values = tuple(data[name] for name in value_names)
        for i, k in enumerate(keys):
            out[k] = tuple(v[i] for v in values)
        print(f"loaded cache {path} rows={len(keys)}")
    return out


def materialize_dense(
    legacy: LegacyModule,
    examples: list[Any],
    cache_paths: list[Path],
    *,
    cache_out: Path,
    batch_size: int,
    allow_encode_missing: bool,
) -> np.ndarray:
    cache = load_cache_map(cache_paths, value_names=("embeddings",))
    missing = [ex for ex in examples if key(ex) not in cache]
    if missing:
        if not allow_encode_missing:
            raise FileNotFoundError(f"dense cache missing {len(missing)} rows, e.g. {key(missing[0])}")
        print(f"encoding missing dense rows={len(missing)}")
        from recsys2026.encoders import Qwen3TextEncoder

        encoder = Qwen3TextEncoder(batch_size=batch_size)
        emb = legacy.encode_dense_queries(
            missing,
            encoder,
            "last_user",
            cache_path=cache_out,
            use_cache=True,
            desc="dense_qfeat[missing]",
        )
        for ex, row in zip(missing, emb, strict=True):
            cache[key(ex)] = (row,)
    rows = [cache[key(ex)][0] for ex in examples]
    return np.asarray(rows, dtype=np.float32)


def materialize_candidates(
    legacy: LegacyModule,
    examples: list[Any],
    dense_q: np.ndarray,
    track_index: Any,
    cache_paths: list[Path],
    *,
    candidate_k: int,
    artist_boost: float,
    album_boost: float,
    exclude_history: bool,
    n_bm25: int,
    cache_name: str,
    allow_generate_missing: bool,
) -> Any:
    cache = load_cache_map(cache_paths, value_names=("indices", "scores"))
    missing_indices = [i for i, ex in enumerate(examples) if key(ex) not in cache]
    if missing_indices:
        if not allow_generate_missing:
            raise FileNotFoundError(f"candidate cache missing {len(missing_indices)} rows, e.g. {key(examples[missing_indices[0]])}")
        missing_examples = [examples[i] for i in missing_indices]
        print(f"generating missing candidates rows={len(missing_examples)}")
        generated = legacy.generate_candidates(
            missing_examples,
            track_index,
            candidate_k=candidate_k,
            artist_boost=artist_boost,
            album_boost=album_boost,
            exclude_history=exclude_history,
            cache_name=cache_name,
            use_cache=True,
            desc=f"cand[{cache_name}]",
            dense_query_emb=dense_q[missing_indices],
            n_bm25=n_bm25,
        )
        for ex, idx, score in zip(missing_examples, generated.indices, generated.scores, strict=True):
            cache[key(ex)] = (idx, score)
    indices = np.asarray([cache[key(ex)][0] for ex in examples], dtype=np.int32)
    scores = np.asarray([cache[key(ex)][1] for ex in examples], dtype=np.float32)
    return legacy.CandidateSet(indices=indices, scores=scores)


def save_public_candidates(
    out_dir: Path,
    sources: list[str],
    examples: list[Any],
    folds: np.ndarray,
    candidates: Any,
    manifest: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [public_key(src, ex) for src, ex in zip(sources, examples, strict=True)]
    sizes = np.full(len(examples), candidates.indices.shape[1], dtype=np.int32)
    np.savez_compressed(
        out_dir / "candidates.npz",
        track_idx=candidates.indices.astype(np.int32, copy=False),
        sizes=sizes,
        keys=encode_keys(keys),
        source_split=np.asarray([s.encode("utf-8") for s in sources], dtype="S8"),
        folds=folds.astype(np.int16, copy=False),
        rank=np.broadcast_to(np.arange(1, candidates.indices.shape[1] + 1, dtype=np.int32), candidates.indices.shape),
        score__bm25_boost=candidates.scores.astype(np.float32, copy=False),
        score__primary=candidates.scores.astype(np.float32, copy=False),
    )
    with (out_dir / "turns.jsonl").open("w", encoding="utf-8") as f:
        for i, (src, ex, fold) in enumerate(zip(sources, examples, folds, strict=True)):
            f.write(
                json.dumps(
                    {
                        "row_id": i,
                        "source_split": src,
                        "session_id": ex.session_id,
                        "user_id": ex.user_id,
                        "turn_number": ex.turn_number,
                        "fold": int(fold),
                        "gold_track_id": ex.gold_track_id,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    json_dump(out_dir / "manifest.json", manifest)


def save_public_ranked(
    out_dir: Path,
    sources: list[str],
    examples: list[Any],
    folds: np.ndarray,
    ranked: np.ndarray,
    ranked_scores: np.ndarray,
    manifest: dict[str, Any],
    track_ids: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = np.full(len(examples), ranked.shape[1], dtype=np.int32)
    keys = [public_key(src, ex) for src, ex in zip(sources, examples, strict=True)]
    np.savez_compressed(
        out_dir / "ranked.npz",
        track_idx=ranked.astype(np.int32, copy=False),
        sizes=sizes,
        keys=encode_keys(keys),
        source_split=np.asarray([s.encode("utf-8") for s in sources], dtype="S8"),
        folds=folds.astype(np.int16, copy=False),
        scores=ranked_scores.astype(np.float32, copy=False),
    )
    with (out_dir / "ranked_top100.jsonl").open("w", encoding="utf-8") as f:
        for src, ex, row in zip(sources, examples, ranked, strict=True):
            f.write(
                json.dumps(
                    {
                        "source_split": src,
                        "session_id": ex.session_id,
                        "turn_number": ex.turn_number,
                        "ranked_track_ids": [track_ids[int(i)] for i in row[:100] if int(i) >= 0],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    json_dump(out_dir / "manifest.json", manifest)


def fit_ranker(
    legacy: LegacyModule,
    train_examples: list[Any],
    train_candidates: Any,
    train_dense_q: np.ndarray,
    track_index: Any,
    user_vectors: dict[str, np.ndarray],
    *,
    candidate_k: int,
    n_bm25: int,
    feature_chunk_examples: int,
    n_estimators: int,
    num_leaves: int,
    learning_rate: float,
    n_jobs: int,
    seed: int,
) -> tuple[lgb.LGBMRanker, Any, TfidfVectorizer, Any]:
    print(f"fit feature encoder rows={len(train_examples)}")
    encoder = legacy.FeatureEncoder(track_index, user_vectors)
    encoder.fit_categories(train_examples)
    vectorizer = TfidfVectorizer(
        min_df=2,
        max_features=120_000,
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
    )
    text_corpus = (
        track_index.texts
        + [legacy.goal_text(ex.conversation_goal) for ex in train_examples]
        + [legacy.conversation_text(ex, track_index) for ex in train_examples]
        + [
            legacy._query_plus_goal(ex.user_query, ex.conversation_goal, ex.user_query_thought)
            for ex in train_examples
        ]
    )
    print("fit text vectorizer")
    vectorizer.fit(text_corpus)
    track_tfidf = vectorizer.transform(track_index.texts)
    print("build feature matrix")
    x_train, y_train, group = legacy.build_feature_matrix(
        train_examples,
        train_candidates,
        encoder,
        vectorizer,
        track_tfidf,
        negatives_per_group=None,
        chunk_examples=feature_chunk_examples,
        query_dense_emb=train_dense_q,
        n_bm25=n_bm25,
        intent_lookup=None,
    )
    if y_train is None or int(y_train.sum()) == 0:
        raise RuntimeError("no positive reranker rows")
    print(
        f"fit LGBM rows={len(y_train)} positives={int(y_train.sum())} "
        f"groups={len(group)} candidate_k={candidate_k}"
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
    model.fit(
        x_train,
        y_train,
        group=group,
        categorical_feature=encoder.categorical_feature_indices,
    )
    return model, encoder, vectorizer, track_tfidf


def rank_examples(
    legacy: LegacyModule,
    model: lgb.LGBMRanker,
    examples: list[Any],
    candidates: Any,
    dense_q: np.ndarray,
    encoder: Any,
    vectorizer: TfidfVectorizer,
    track_tfidf: Any,
    *,
    feature_chunk_examples: int,
    n_bm25: int,
) -> tuple[np.ndarray, np.ndarray]:
    ranked, score_rows = legacy.score_and_rank(
        model,
        examples,
        candidates,
        encoder,
        vectorizer,
        track_tfidf,
        chunk_examples=feature_chunk_examples,
        query_dense_emb=dense_q,
        n_bm25=n_bm25,
        intent_lookup=None,
    )
    width = candidates.indices.shape[1]
    ranked_mat = np.full((len(examples), width), -1, dtype=np.int32)
    score_mat = np.full((len(examples), width), np.nan, dtype=np.float32)
    for i, (idxs, scores) in enumerate(zip(ranked, score_rows, strict=True)):
        n = min(width, len(idxs))
        ranked_mat[i, :n] = np.asarray(idxs[:n], dtype=np.int32)
        score_mat[i, :n] = np.asarray(scores[:n], dtype=np.float32)
    return ranked_mat, score_mat


def ndcg_at_pos(pos: int, k: int) -> float:
    if pos < 0 or pos >= k:
        return 0.0
    import math

    return 1.0 / math.log2(pos + 2)


def evaluate_ranked(
    sources: list[str],
    examples: list[Any],
    ranked: np.ndarray,
    track_index: Any,
    *,
    top_k: int,
) -> dict[str, Any]:
    by_turn: dict[int, list[dict[str, float]]] = defaultdict(list)
    by_source: dict[str, list[float]] = defaultdict(list)
    for src, ex, row in zip(sources, examples, ranked, strict=True):
        gold_idx = track_index.id_to_idx.get(ex.gold_track_id or "")
        pos = -1
        if gold_idx is not None:
            hits = np.flatnonzero(row[:top_k] == gold_idx)
            if len(hits):
                pos = int(hits[0])
        vals = {
            "ndcg@1": ndcg_at_pos(pos, 1),
            "ndcg@10": ndcg_at_pos(pos, 10),
            "ndcg@20": ndcg_at_pos(pos, 20),
        }
        by_turn[int(ex.turn_number)].append(vals)
        by_source[src].append(vals["ndcg@20"])
    turn_means = {
        turn: {name: sum(v[name] for v in vals) / len(vals) for name in ("ndcg@1", "ndcg@10", "ndcg@20")}
        for turn, vals in by_turn.items()
    }
    out = {
        name: sum(v[name] for v in turn_means.values()) / len(turn_means)
        for name in ("ndcg@1", "ndcg@10", "ndcg@20")
    }
    out["n_examples"] = len(examples)
    for src, vals in by_source.items():
        out[f"{src}_ndcg@20"] = float(sum(vals) / len(vals))
    return out


def write_blind_predictions(
    target: Target,
    examples: list[Any],
    ranked: np.ndarray,
    track_index: Any,
    out_json: Path,
    *,
    top_k: int,
) -> Path:
    records: list[dict[str, Any]] = []
    inputs = list(iter_inputs(target))
    for inp, ex, row in zip(inputs, examples, ranked, strict=True):
        tids: list[str] = []
        seen: set[str] = set()
        for idx_raw in row:
            idx = int(idx_raw)
            if idx < 0:
                continue
            tid = track_index.track_ids[idx]
            if tid in seen:
                continue
            seen.add(tid)
            tids.append(tid)
            if len(tids) == top_k:
                break
        response = legacy_prediction_response(track_index, ex, int(row[0]))
        records.append(format_record(inp, tids, response))
    validate_predictions(records, target)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(records, ensure_ascii=False))
    return zip_submission(out_json)


def legacy_prediction_response(track_index: Any, ex: Any, top_idx: int) -> str:
    # Keep the same deterministic template family as exp098 without importing a
    # module global into this helper's call sites.
    track = track_index.track_name[top_idx] or "this track"
    artist = track_index.artist_name[top_idx] or "the artist"
    goal = str(ex.conversation_goal.get("listener_goal") or ex.user_query).strip()
    goal = " ".join(goal.split())[:140]
    templates = [
        'I would start with "{track}" by {artist}. It fits your current request and keeps the recommendation close to: {goal}',
        'For this turn, "{track}" by {artist} is the strongest match I found. It should line up with the mood and preferences you described.',
        'My next pick is "{track}" by {artist}. It connects your recent feedback with the direction of this session.',
        'Try "{track}" by {artist}. It gives you a focused next step based on your profile, goal, and conversation so far.',
    ]
    bucket = int.from_bytes(f"{ex.session_id}:{ex.turn_number}".encode("utf-8"), "little", signed=False) % len(templates)
    return templates[bucket].format(track=track, artist=artist, goal=goal)


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="protocol_098")
    parser.add_argument("--config", default="current_thought_profile")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "splits" / "public_labeled_v1")
    parser.add_argument("--blind-target", choices=("blind_a", "blind_b"), default="blind_a")
    parser.add_argument("--candidate-k", type=int, default=300)
    parser.add_argument("--final-k", type=int, default=20)
    parser.add_argument("--feature-chunk-examples", type=int, default=512)
    parser.add_argument("--artist-boost", type=float, default=50.0)
    parser.add_argument("--album-boost", type=float, default=30.0)
    parser.add_argument("--n-bm25", type=int, default=200)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--learning-rate", type=float, default=0.04)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--dense-encode-batch-size", type=int, default=64)
    parser.add_argument("--allow-encode-missing", action="store_true")
    parser.add_argument("--allow-generate-missing", action="store_true")
    parser.add_argument("--fold", type=int, default=None, help="debug: run only one CV fold")
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    split_dir = args.split_dir if args.split_dir.is_absolute() else REPO_ROOT / args.split_dir
    legacy = load_098_module()
    old_out = OUTPUT_DIR / "098_current_thought_profile_ablation"
    reuse_093 = OUTPUT_DIR / "093_current_thought_goal_query"

    print("loading tracks/users")
    track_index = legacy.TrackIndex(
        "all_tracks",
        corpus_fields=legacy.CORPUS_FIELDS_5,
        secondary_corpus_fields=None,
        load_dense=True,
    )
    user_vectors = legacy.load_user_vectors()

    print("loading examples")
    train_examples = legacy.build_examples_from_dataset("train")
    dev_examples = legacy.build_examples_from_dataset("test")
    blind_target: Target = args.blind_target  # type: ignore[assignment]
    blind_examples = legacy.build_examples_from_blind(blind_target)
    public_examples = train_examples + dev_examples
    public_sources = ["train"] * len(train_examples) + ["devset"] * len(dev_examples)
    fold_map = load_fold_map(split_dir)
    folds = np.asarray(
        [fold_map[(src, ex.session_id)] for src, ex in zip(public_sources, public_examples, strict=True)],
        dtype=np.int16,
    )
    print(f"public examples={len(public_examples)} train={len(train_examples)} devset={len(dev_examples)} blind={len(blind_examples)}")

    print("materializing dense query features")
    train_dense = materialize_dense(
        legacy,
        train_examples,
        [
            old_out / "dense_qfeat_train_seed0_maxNone_last_user.npz",
            old_out / "dense_qfeat_valid_seed0_maxNone_last_user.npz",
        ],
        cache_out=old_out / "dense_qfeat_protocol_missing_train_last_user.npz",
        batch_size=args.dense_encode_batch_size,
        allow_encode_missing=args.allow_encode_missing,
    )
    dev_dense = materialize_dense(
        legacy,
        dev_examples,
        [
            reuse_093 / "dense_qfeat_devset_maxNone_last_user.npz",
            old_out / "dense_qfeat_devset_maxNone_last_user.npz",
        ],
        cache_out=old_out / "dense_qfeat_devset_maxNone_last_user.npz",
        batch_size=args.dense_encode_batch_size,
        allow_encode_missing=args.allow_encode_missing,
    )
    public_dense = np.concatenate([train_dense, dev_dense], axis=0)
    blind_dense = materialize_dense(
        legacy,
        blind_examples,
        [old_out / f"dense_qfeat_{blind_target}_maxNone_last_user.npz"],
        cache_out=old_out / f"dense_qfeat_{blind_target}_maxNone_last_user.npz",
        batch_size=args.dense_encode_batch_size,
        allow_encode_missing=args.allow_encode_missing,
    )

    print("materializing candidates")
    train_candidates = materialize_candidates(
        legacy,
        train_examples,
        train_dense,
        track_index,
        [
            old_out / "candidates_train_seed0_maxNone_tags1_ms0_dr0_drop_music_un1_6886e94aba.npz",
            old_out / "candidates_valid_seed0_maxNone_tags1_ms0_dr0_drop_music_un1_0f641a66d0.npz",
            old_out / "candidates_train_seed0_maxNone_tags1_ms0_dr0_drop_music_un1_d227406628.npz",
            old_out / "candidates_valid_seed0_maxNone_tags1_ms0_dr0_drop_music_un1_2ec55399d1.npz",
        ],
        candidate_k=args.candidate_k,
        artist_boost=args.artist_boost,
        album_boost=args.album_boost,
        exclude_history=True,
        n_bm25=args.n_bm25,
        cache_name="protocol_train_missing_tags1_ms0_dr0_drop_music_un1",
        allow_generate_missing=args.allow_generate_missing,
    )
    dev_candidates = materialize_candidates(
        legacy,
        dev_examples,
        dev_dense,
        track_index,
        [
            reuse_093 / "candidates_devset_maxNone_tags1_ms0_dr0_drop_music_un1_761691dcf2.npz",
            OUTPUT_DIR / "038_query_union_candidates" / "candidates_devset_maxNone_tags1_ms0_dr0_drop_music_un1_761691dcf2.npz",
            old_out / "candidates_devset_maxNone_tags1_ms0_dr0_drop_music_un1.npz",
        ],
        candidate_k=args.candidate_k,
        artist_boost=args.artist_boost,
        album_boost=args.album_boost,
        exclude_history=True,
        n_bm25=args.n_bm25,
        cache_name="devset_maxNone_tags1_ms0_dr0_drop_music_un1",
        allow_generate_missing=args.allow_generate_missing,
    )
    public_candidates = legacy.CandidateSet(
        indices=np.vstack([train_candidates.indices, dev_candidates.indices]),
        scores=np.vstack([train_candidates.scores, dev_candidates.scores]),
    )
    blind_candidates = materialize_candidates(
        legacy,
        blind_examples,
        blind_dense,
        track_index,
        [old_out / f"candidates_{blind_target}_maxNone_tags1_ms0_dr0_drop_music_un1_a9a669a606.npz"],
        candidate_k=args.candidate_k,
        artist_boost=args.artist_boost,
        album_boost=args.album_boost,
        exclude_history=True,
        n_bm25=args.n_bm25,
        cache_name=f"{blind_target}_maxNone_tags1_ms0_dr0_drop_music_un1",
        allow_generate_missing=args.allow_generate_missing,
    )

    base_manifest = {
        "schema_version": 1,
        "producer": {
            "command": ["uv", "run", "python", "scripts/run_protocol_098_baseline.py"],
            "cwd": ".",
        },
        "protocol": "docs/pipeline_cv_protocol.md",
        "split_artifact": rel(split_dir),
        "params": jsonable(vars(args)),
        "feature_source": "recsys2026.reranker_features",
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_blind_for_fit": False,
            "popularity_tiebreaker": False,
            "intent_features": False,
            "current_thought_allowed": True,
        },
    }
    retriever_source_policy = {
        "requires_labeled_fit": False,
        "fit_sources": ["track_metadata", "target_input_context", "pretrained_query_embeddings"],
        "train_row_policy": "safe_in_sample",
        "fold_split_required_for_reranker_train": False,
        "preferred_train_row_artifact_mode": "fit_free_all_rows",
        "preferred_inference_artifact_mode": "fit_free_all_rows",
        "reason": "098 retriever candidates are BM25/dense feature retrieval and do not fit on labeled rows.",
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
        "note": "No labeled-row fit; fold splitting is not required for reranker training.",
    }
    retriever_public_dir = OUTPUT_DIR / "retriever" / args.name / args.config / "cv3_oof" / "public_labeled"
    save_public_candidates(
        retriever_public_dir,
        public_sources,
        public_examples,
        folds,
        public_candidates,
        {
            **base_manifest,
            "artifact_type": "candidates",
            "stage": "retriever",
            "name": args.name,
            "config": args.config,
            "artifact_mode": "cv3_oof",
            "target": "public_labeled",
            "created_at": utc_now(),
            "fit_scope": retriever_fit_scope,
            "source_policy": retriever_source_policy,
        },
    )
    public_cand_metrics = legacy.candidate_metrics(public_examples, public_candidates, track_index)
    print(f"public candidate metrics: {public_cand_metrics}")
    retriever_scores_dir = RESULTS_DIR / "retriever" / args.name / args.config / "cv3_oof" / "public_labeled"
    json_dump(
        retriever_scores_dir / "scores.json",
        {
            "name": args.name,
            "config": args.config,
            "artifact_mode": "cv3_oof",
            "target": "public_labeled",
            "candidate_metrics": public_cand_metrics,
            "retriever_artifact": rel(retriever_public_dir),
        },
    )

    all_rows = np.arange(len(public_examples), dtype=np.int32)
    fold_ranked = np.full(public_candidates.indices.shape, -1, dtype=np.int32)
    fold_scores = np.full(public_candidates.indices.shape, np.nan, dtype=np.float32)
    fold_metrics: dict[str, Any] = {}
    run_folds = [args.fold] if args.fold is not None else [0, 1, 2]
    for fold in run_folds:
        print(f"CV fold {fold}: train")
        train_rows = all_rows[folds != fold]
        valid_rows = all_rows[folds == fold]
        train_ex = [public_examples[int(i)] for i in train_rows]
        valid_ex = [public_examples[int(i)] for i in valid_rows]
        train_cand = legacy.CandidateSet(
            indices=public_candidates.indices[train_rows],
            scores=public_candidates.scores[train_rows],
        )
        valid_cand = legacy.CandidateSet(
            indices=public_candidates.indices[valid_rows],
            scores=public_candidates.scores[valid_rows],
        )
        model, encoder, vectorizer, track_tfidf = fit_ranker(
            legacy,
            train_ex,
            train_cand,
            public_dense[train_rows],
            track_index,
            user_vectors,
            candidate_k=args.candidate_k,
            n_bm25=args.n_bm25,
            feature_chunk_examples=args.feature_chunk_examples,
            n_estimators=args.n_estimators,
            num_leaves=args.num_leaves,
            learning_rate=args.learning_rate,
            n_jobs=args.n_jobs,
            seed=fold,
        )
        print(f"CV fold {fold}: rank")
        ranked, scores = rank_examples(
            legacy,
            model,
            valid_ex,
            valid_cand,
            public_dense[valid_rows],
            encoder,
            vectorizer,
            track_tfidf,
            feature_chunk_examples=args.feature_chunk_examples,
            n_bm25=args.n_bm25,
        )
        fold_ranked[valid_rows] = ranked
        fold_scores[valid_rows] = scores
        fold_sources = [public_sources[int(i)] for i in valid_rows]
        metrics = evaluate_ranked(fold_sources, valid_ex, ranked, track_index, top_k=args.final_k)
        fold_metrics[f"fold{fold}"] = metrics
        print(f"CV fold {fold}: {metrics}")

    cv_metrics = evaluate_ranked(public_sources, public_examples, fold_ranked, track_index, top_k=args.final_k)
    print(f"CV combined: {cv_metrics}")
    reranker_public_dir = OUTPUT_DIR / "reranker" / args.name / args.config / "cv3_oof" / "public_labeled"
    save_public_ranked(
        reranker_public_dir,
        public_sources,
        public_examples,
        folds,
        fold_ranked,
        fold_scores,
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
        track_index.track_ids,
    )
    scores_dir = RESULTS_DIR / "reranker" / args.name / args.config / "cv3_oof" / "public_labeled"
    json_dump(
        scores_dir / "scores.json",
        {
            "name": args.name,
            "config": args.config,
            "artifact_mode": "cv3_oof",
            "target": "public_labeled",
            "candidate_metrics": public_cand_metrics,
            "cv_metrics": cv_metrics,
            "fold_metrics": fold_metrics,
            "retriever_artifact": rel(retriever_public_dir),
            "reranker_artifact": rel(reranker_public_dir),
        },
    )

    print("final full-public reranker train")
    final_model, final_encoder, final_vectorizer, final_track_tfidf = fit_ranker(
        legacy,
        public_examples,
        public_candidates,
        public_dense,
        track_index,
        user_vectors,
        candidate_k=args.candidate_k,
        n_bm25=args.n_bm25,
        feature_chunk_examples=args.feature_chunk_examples,
        n_estimators=args.n_estimators,
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_jobs=args.n_jobs,
        seed=0,
    )

    retriever_blind_dir = OUTPUT_DIR / "retriever" / args.name / args.config / "full_public" / blind_target
    save_candidate_artifact(
        retriever_blind_dir,
        blind_candidates.indices,
        np.full(len(blind_examples), blind_candidates.indices.shape[1], dtype=np.int32),
        target=blind_target,
        manifest={
            **base_manifest,
            "artifact_type": "candidates",
            "stage": "retriever",
            "name": args.name,
            "config": args.config,
            "artifact_mode": "full_public",
            "target": blind_target,
            "created_at": utc_now(),
            "fit_scope": retriever_fit_scope,
            "source_policy": retriever_source_policy,
        },
        score_arrays={
            "bm25_boost": blind_candidates.scores.astype(np.float32, copy=False),
            "primary": blind_candidates.scores.astype(np.float32, copy=False),
        },
        rank=np.broadcast_to(np.arange(1, blind_candidates.indices.shape[1] + 1, dtype=np.int32), blind_candidates.indices.shape),
        compress=True,
    )

    print(f"rank {blind_target}")
    blind_ranked, blind_rank_scores = rank_examples(
        legacy,
        final_model,
        blind_examples,
        blind_candidates,
        blind_dense,
        final_encoder,
        final_vectorizer,
        final_track_tfidf,
        feature_chunk_examples=args.feature_chunk_examples,
        n_bm25=args.n_bm25,
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
        scores=blind_rank_scores,
        compress=True,
    )

    responder_dir = OUTPUT_DIR / "responder" / "legacy_098_template" / args.config / blind_target
    pred_json = responder_dir / "prediction.json"
    zip_path = write_blind_predictions(blind_target, blind_examples, blind_ranked, track_index, pred_json, top_k=args.final_k)
    json_dump(
        responder_dir / "manifest.json",
        {
            **base_manifest,
            "artifact_type": "predictions",
            "stage": "responder",
            "name": "legacy_098_template",
            "config": args.config,
            "target": blind_target,
            "created_at": utc_now(),
            "ranked_artifact": rel(reranker_blind_dir),
            "outputs": {"json": rel(pred_json), "zip": rel(zip_path)},
        },
    )
    print(f"wrote {reranker_blind_dir}")
    print(f"wrote {zip_path}")
    print(json.dumps({"candidate_metrics": public_cand_metrics, "cv_metrics": cv_metrics}, indent=2))


if __name__ == "__main__":
    main()
