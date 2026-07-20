#!/usr/bin/env python3
"""Build the fit-free TF-IDF retriever artifacts used by the final pipeline."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from recsys2026.artifacts import (
    artifact_complete,
    encode_keys,
    save_candidate_artifact,
    save_npz_artifact,
    utc_now,
)
from recsys2026.data import load
from recsys2026.paths import OUTPUT_DIR, REPO_ROOT
from recsys2026.splits import read_jsonl
from recsys2026.submission import Target


MAX_TURNS = 8
NAME = "tfidf_catalog"
CONFIG = "top300"
CANDIDATE_K = 300
MAX_FEATURES = 200_000
MIN_DF = 2
RETRIEVAL_CHUNK_SIZE = 512


@dataclass(frozen=True)
class Example:
    source_split: Literal["train", "devset", "blind_b"]
    session_id: str
    user_id: str
    turn_number: int
    query_text: str
    history_track_idx: tuple[int, ...]
    gold_idx: int
    fold: int


@dataclass
class TrackStore:
    track_ids: list[str]
    id_to_idx: dict[str, int]
    texts: list[str]
    meta_by_id: dict[str, dict[str, Any]]


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def join_field(value: Any) -> str:
    return " ".join(str(x) for x in as_list(value) if x is not None)


def load_tracks() -> TrackStore:
    rows = list(load("track", split="all_tracks"))
    track_ids = [r["track_id"] for r in rows]
    id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    texts: list[str] = []
    for r in rows:
        track = join_field(r.get("track_name"))
        artist = join_field(r.get("artist_name"))
        album = join_field(r.get("album_name"))
        tags = join_field(r.get("tag_list"))
        year = str(r.get("release_date") or "")[:4]
        texts.append(f"{track} {artist} {album} {tags} {year}".strip())
    return TrackStore(
        track_ids=track_ids,
        id_to_idx=id_to_idx,
        texts=texts,
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


def example_query(
    item: dict[str, Any],
    *,
    target_turn: int,
    tracks: TrackStore,
) -> tuple[str, tuple[int, ...]]:
    conversations = list(item["conversations"])
    current = [c for c in conversations if int(c["turn_number"]) == target_turn]
    user_turn = next(c for c in current if c["role"] == "user")
    parts = [str(user_turn.get("content") or "")]
    history_track_idx: list[int] = []
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
                parts.append(music_text(content, tracks))
    text = " ".join(p for p in parts if p).strip()
    return text, tuple(history_track_idx)


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
                query, hist_tracks = example_query(
                    item, target_turn=turn, tracks=tracks
                )
                examples.append(
                    Example(
                        source_split=source_split,  # type: ignore[arg-type]
                        session_id=item["session_id"],
                        user_id=item["user_id"],
                        turn_number=turn,
                        query_text=query,
                        history_track_idx=hist_tracks,
                        gold_idx=tracks.id_to_idx[gold_by_turn[turn]],
                        fold=fold,
                    )
                )
    return examples


def load_inference_examples(target: Target, tracks: TrackStore) -> list[Example]:
    if target != "blind_b":
        raise ValueError(target)
    examples: list[Example] = []
    for item in load(target, split="test"):
        current = item["conversations"][-1]
        query, hist_tracks = example_query(
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
                gold_idx=-1,
                fold=-1,
            )
        )
    return examples


def fit_retriever(
    tracks: TrackStore, *, max_features: int, min_df: int
) -> tuple[TfidfVectorizer, sparse.csr_matrix]:
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
        sc = np.take_along_axis(part_scores, order, axis=1).astype(
            np.float32, copy=False
        )
        track_idx[start:end, :kk] = idx
        scores[start:end, :kk] = sc
        if start == 0 or (start // chunk_size) % 25 == 0:
            print(f"retrieved {end}/{n}")
    sizes = np.full(n, candidate_k, dtype=np.int32)
    return track_idx, scores, sizes


def candidate_metrics(
    examples: list[Example], track_idx: np.ndarray, sizes: np.ndarray
) -> dict[str, Any]:
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


def save_public_candidates(
    out_dir: Path,
    examples: list[Example],
    track_idx: np.ndarray,
    scores: np.ndarray,
    sizes: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    keys = [(f"{ex.source_split}:{ex.session_id}", ex.turn_number) for ex in examples]
    folds = np.asarray([ex.fold for ex in examples], dtype=np.int16)
    arrays = {
        "track_idx": track_idx,
        "sizes": sizes,
        "keys": encode_keys(keys),
        "folds": folds,
        "score__tfidf": scores.astype(np.float32, copy=False),
        "rank": np.broadcast_to(
            np.arange(1, track_idx.shape[1] + 1, dtype=np.int32), track_idx.shape
        ),
    }
    turns = [
        {
            "row_id": i,
            "source_split": ex.source_split,
            "session_id": ex.session_id,
            "user_id": ex.user_id,
            "turn_number": ex.turn_number,
            "fold": ex.fold,
            "gold_track_idx": ex.gold_idx,
        }
        for i, ex in enumerate(examples)
    ]
    save_npz_artifact(out_dir, arrays, turns, manifest)


def save_inference_candidates(
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
        rank=np.broadcast_to(
            np.arange(1, track_idx.shape[1] + 1, dtype=np.int32), track_idx.shape
        ),
        compress=True,
    )


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", choices=("public_labeled", "blind_b"), required=True
    )
    args = parser.parse_args([arg for arg in sys.argv[1:] if arg != "--"])
    sys.stdout.reconfigure(line_buffering=True)
    inference_target: Target = "blind_b"
    retriever_public_dir = (
        OUTPUT_DIR
        / "retriever"
        / NAME
        / CONFIG
        / "fit_free_all_rows"
        / "public_labeled"
    )
    retriever_inference_dir = (
        OUTPUT_DIR
        / "retriever"
        / NAME
        / CONFIG
        / "fit_free_all_rows"
        / inference_target
    )
    output_dir = (
        retriever_public_dir
        if args.target == "public_labeled"
        else retriever_inference_dir
    )
    if artifact_complete(output_dir, "candidates.npz", "turns.jsonl"):
        print(f"[skip] {output_dir}")
        return

    split_dir = REPO_ROOT / "artifacts/preprocessed/splits/cv5"
    if not (split_dir / "sessions.jsonl").exists():
        raise FileNotFoundError(
            f"missing split artifact: {split_dir}. Run `bash run_preprocess.sh` first."
        )

    print("loading tracks")
    tracks = load_tracks()
    if args.target == "public_labeled":
        print("loading public examples")
        examples = load_public_examples(split_dir, tracks)
    else:
        print("loading inference examples")
        examples = load_inference_examples(inference_target, tracks)
    print(f"{args.target} examples={len(examples)}")

    print("fitting fit-free TF-IDF retriever over track metadata")
    vectorizer, track_matrix = fit_retriever(
        tracks, max_features=MAX_FEATURES, min_df=MIN_DF
    )

    base_manifest = {
        "schema_version": 1,
        "producer": {
            "command": ["uv", "run", "python", "-m", "retriever.tfidf_catalog.main"],
            "cwd": ".",
        },
        "protocol": "docs/folds.md",
        "split_artifact": rel(split_dir),
        "params": {
            "candidate_k": CANDIDATE_K,
            "max_features": MAX_FEATURES,
            "min_df": MIN_DF,
            "retrieval_chunk_size": RETRIEVAL_CHUNK_SIZE,
        },
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

    candidate_npz = output_dir / "candidates.npz"
    if candidate_npz.exists():
        print(f"loading existing candidates from {candidate_npz}")
        data = np.load(candidate_npz)
        candidates = data["track_idx"]
        scores = data["score__tfidf"]
        sizes = data["sizes"]
    else:
        print(f"retrieving {args.target} fit-free candidates")
        candidates, scores, sizes = retrieve_candidates(
            examples,
            vectorizer,
            track_matrix,
            candidate_k=CANDIDATE_K,
            chunk_size=RETRIEVAL_CHUNK_SIZE,
        )
    manifest = {
        **base_manifest,
        "artifact_type": "candidates",
        "stage": "retriever",
        "name": NAME,
        "config": CONFIG,
        "artifact_mode": "fit_free_all_rows",
        "target": args.target,
        "created_at": utc_now(),
        "fit_scope": retriever_fit_scope,
        "source_policy": retriever_source_policy,
    }
    if args.target == "public_labeled":
        save_public_candidates(
            retriever_public_dir,
            examples,
            candidates,
            scores,
            sizes,
            manifest,
        )
        print(
            f"public candidate metrics: {candidate_metrics(examples, candidates, sizes)}"
        )
    else:
        save_inference_candidates(
            retriever_inference_dir,
            inference_target,
            candidates,
            scores,
            sizes,
            manifest,
        )


if __name__ == "__main__":
    main()
