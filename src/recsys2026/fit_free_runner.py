#!/usr/bin/env python3
"""Shared artifact runner for fit-free retriever components."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import bm25s
import numpy as np
import yaml
from tqdm import tqdm

from recsys2026 import retriever_common as common
from recsys2026.artifacts import (
    component_output_dir,
    component_results_dir,
    encode_keys,
    file_ref,
    json_dump,
    save_candidate_artifact,
    save_npz_artifact,
    utc_now,
)
from recsys2026.data import load
from recsys2026.paths import REPO_ROOT
from recsys2026.retriever_eval import candidate_metrics, devset_gold_indices
from recsys2026.splits import read_jsonl


PUBLIC_SOURCES = (("train", "train"), ("devset", "test"))


@dataclass(frozen=True)
class FitFreeSpec:
    """Component-specific callbacks consumed by the shared artifact runner."""

    name: str
    source_path: Path
    bm25_variants: tuple[tuple[str, tuple[str, ...]], ...] = ()
    bm25_name: str | None = None
    query_fn: Callable[[Any, Any], str] | None = None
    score_fn: Callable[[Any, Any], np.ndarray | None] | None = None

    def __post_init__(self) -> None:
        if (self.query_fn is None) == (self.score_fn is None):
            raise ValueError("exactly one of query_fn or score_fn is required")
        if self.query_fn is not None and self.bm25_name is None:
            raise ValueError("BM25 query components require bm25_name")


def read_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def source_policy_from_config(cfg: dict[str, Any], source: str) -> dict[str, Any]:
    configured_sources = cfg.get("sources") or []
    if configured_sources and source not in configured_sources:
        raise ValueError(f"source {source!r} is not declared in the config")
    defaults = dict(cfg.get("source_policy_defaults") or {})
    metadata = dict((cfg.get("source_metadata") or {}).get(source) or {})
    policy = {**defaults, **metadata}
    policy.setdefault("requires_labeled_fit", False)
    policy.setdefault(
        "train_row_policy",
        "requires_oof" if policy["requires_labeled_fit"] else "safe_in_sample",
    )
    policy.setdefault(
        "fold_split_required_for_reranker_train",
        bool(policy["requires_labeled_fit"]),
    )
    policy.setdefault(
        "preferred_train_row_artifact_mode",
        "oof2_train" if policy["requires_labeled_fit"] else "fit_free_all_rows",
    )
    policy.setdefault(
        "preferred_inference_artifact_mode",
        "full_train" if policy["requires_labeled_fit"] else "fit_free_all_rows",
    )
    return policy


def load_fold_map(split_dir: Path) -> dict[tuple[str, str], int]:
    return {
        (str(row["source_split"]), str(row["session_id"])): int(row["fold"])
        for row in read_jsonl(split_dir / "sessions.jsonl")
    }


def build_public_labeled_examples(
    common_module: Any, split_dir: Path
) -> tuple[list[Any], list[dict[str, Any]]]:
    fold_map = load_fold_map(split_dir)
    examples: list[Any] = []
    rows: list[dict[str, Any]] = []
    for source_split, dataset_split in PUBLIC_SOURCES:
        for item in load("dataset", split=dataset_split):
            conversations = list(item["conversations"])
            fold = fold_map[(source_split, str(item["session_id"]))]
            for target_turn in range(1, common_module.MAX_TURNS + 1):
                current = [
                    c for c in conversations if int(c["turn_number"]) == target_turn
                ]
                user_turn = next((c for c in current if c["role"] == "user"), None)
                music_turn = next((c for c in current if c["role"] == "music"), None)
                if user_turn is None or music_turn is None:
                    continue
                gold_track_id = str(music_turn.get("content") or "")
                examples.append(
                    common_module.TurnExample(
                        session_id=str(item["session_id"]),
                        user_id=str(item["user_id"]),
                        turn_number=target_turn,
                        chat_history=[
                            c
                            for c in conversations
                            if int(c["turn_number"]) < target_turn
                        ],
                        user_query=str(user_turn.get("content") or ""),
                        gold_track_id=gold_track_id,
                        user_thought=str(user_turn.get("thought") or "").strip(),
                    )
                )
                rows.append(
                    {
                        "row_id": len(rows),
                        "source_split": source_split,
                        "session_id": str(item["session_id"]),
                        "user_id": str(item["user_id"]),
                        "turn_number": target_turn,
                        "fold": fold,
                        "gold_track_id": gold_track_id,
                    }
                )
    return examples, rows


def build_inference_examples(common_module: Any, target: str) -> list[Any]:
    examples: list[Any] = []
    for item in load(target, split="test"):
        conversations = list(item["conversations"])
        current = conversations[-1]
        target_turn = int(current["turn_number"])
        examples.append(
            common_module.TurnExample(
                session_id=str(item["session_id"]),
                user_id=str(item["user_id"]),
                turn_number=target_turn,
                chat_history=[
                    c for c in conversations if int(c["turn_number"]) < target_turn
                ],
                user_query=str(current.get("content") or ""),
                gold_track_id=None,
                user_thought=str(current.get("thought") or "").strip(),
            )
        )
    return examples


def public_labeled_metrics(
    rows: list[dict[str, Any]], track_index: Any, cand: np.ndarray, sizes: np.ndarray
) -> dict[str, Any]:
    gold = np.asarray(
        [track_index.id_to_idx.get(str(row["gold_track_id"]), -1) for row in rows],
        dtype=np.int32,
    )
    out: dict[str, Any] = {
        "n_examples": len(rows),
        "mean_size": float(sizes.mean()) if len(sizes) else 0.0,
    }
    groups = {
        "all": np.arange(len(rows), dtype=np.int32),
        "train": np.asarray(
            [i for i, row in enumerate(rows) if row["source_split"] == "train"],
            dtype=np.int32,
        ),
        "devset": np.asarray(
            [i for i, row in enumerate(rows) if row["source_split"] == "devset"],
            dtype=np.int32,
        ),
    }
    for name, idx in groups.items():
        if len(idx) == 0:
            continue
        prefix = "" if name == "all" else f"{name}_"
        out[f"{prefix}n_examples"] = int(len(idx))
        out[f"{prefix}mean_size"] = float(sizes[idx].mean())
        for k in (20, 50, 100, 200, 500):
            kk = min(k, cand.shape[1])
            hit = (cand[idx, :kk] == gold[idx, None]).any(axis=1)
            out[f"{prefix}recall@{k}"] = float(hit.mean())
        hit_all = np.zeros(len(idx), dtype=bool)
        for j, row_i in enumerate(idx):
            hit_all[j] = bool((cand[row_i, : int(sizes[row_i])] == gold[row_i]).any())
        out[f"{prefix}recall@all"] = float(hit_all.mean())
    return out


def fit_scope_from_source_policy(policy: dict[str, Any]) -> dict[str, Any]:
    requires_labeled_fit = bool(policy.get("requires_labeled_fit", False))
    fit_splits = ["train"] if requires_labeled_fit else []
    fit_mode = "train_labeled_fit" if requires_labeled_fit else "fit_free"
    return {
        "fit_mode": fit_mode,
        "fit_splits": fit_splits,
        "requires_labeled_fit": requires_labeled_fit,
        "fit_sources": list(policy.get("fit_sources") or []),
        "train_row_policy": str(policy.get("train_row_policy") or "safe_in_sample"),
        "fold_split_required_for_reranker_train": bool(
            policy.get("fold_split_required_for_reranker_train", requires_labeled_fit)
        ),
        "preferred_train_row_artifact_mode": str(
            policy.get(
                "preferred_train_row_artifact_mode",
                "oof2_train" if requires_labeled_fit else "fit_free_all_rows",
            )
        ),
        "preferred_inference_artifact_mode": str(
            policy.get(
                "preferred_inference_artifact_mode",
                "full_train" if requires_labeled_fit else "fit_free_all_rows",
            )
        ),
        "uses_devset_for_fit": False,
        "uses_blind_for_fit": False,
    }


def pad_scored(
    rows: list[tuple[np.ndarray, np.ndarray]], top_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cand = np.full((len(rows), top_k), -1, dtype=np.int32)
    sizes = np.zeros(len(rows), dtype=np.int32)
    scores = np.full((len(rows), top_k), np.nan, dtype=np.float32)
    for i, (idxs, vals) in enumerate(rows):
        if len(idxs) == 0:
            continue
        k = min(len(idxs), top_k)
        cand[i, :k] = np.asarray(idxs[:k], dtype=np.int32)
        scores[i, :k] = np.asarray(vals[:k], dtype=np.float32)
        sizes[i] = k
    return cand, sizes, scores


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
    if positive_only:
        idxs = np.flatnonzero(score > 0)
    else:
        idxs = np.arange(len(score), dtype=np.int32)
    if len(idxs) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    if played:
        played_arr = np.fromiter(played, dtype=np.int32)
        idxs = idxs[~np.isin(idxs, played_arr)]
    if len(idxs) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    vals = score[idxs]
    order = np.argsort(-vals, kind="stable")
    idxs = idxs[order][:top_k].astype(np.int32)
    vals = vals[order][:top_k].astype(np.float32)
    return idxs, vals


def bm25_queries_scored(
    common_module: Any,
    examples: list,
    track_index: Any,
    bm25_name: str,
    queries: list[str],
    top_k: int,
    desc: str,
):
    bm25 = track_index.bm25_indexes[bm25_name]
    rows = []
    for ex, query in tqdm(list(zip(examples, queries, strict=True)), desc=desc):
        if not query:
            rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue
        played = common_module.played_set(ex, track_index)
        pool = min(top_k + len(played) + 16, track_index.n_tracks)
        toks = bm25s.tokenize([str(query).lower()], show_progress=False)
        idx_arr, score_arr = bm25.retrieve(toks, k=pool, show_progress=False)
        kept: list[int] = []
        vals: list[float] = []
        for idx_raw, score_raw in zip(idx_arr[0], score_arr[0], strict=True):
            idx = int(idx_raw)
            if idx in played:
                continue
            kept.append(idx)
            vals.append(float(score_raw))
            if len(kept) >= top_k:
                break
        rows.append(
            (np.asarray(kept, dtype=np.int32), np.asarray(vals, dtype=np.float32))
        )
    return pad_scored(rows, top_k)


def count_scored(
    common_module: Any,
    examples: list,
    track_index: Any,
    score_fn: Any,
    desc: str,
    top_k: int,
):
    rows = []
    for ex in tqdm(examples, desc=desc):
        rows.append(
            select_from_score(
                score_fn(ex, track_index),
                common_module.played_set(ex, track_index),
                top_k,
                positive_only=True,
            )
        )
    return pad_scored(rows, top_k)


def save_source(
    name: str,
    config: str,
    target: str,
    cand: np.ndarray,
    sizes: np.ndarray,
    scores: np.ndarray,
    source_refs: dict[str, Any],
    elapsed: float,
    producer_command: list[str],
    run_params: dict[str, Any],
    source_policy: dict[str, Any],
    *,
    fit_mode: str | None = None,
    public_rows: list[dict[str, Any]] | None = None,
    track_index: Any | None = None,
):
    out_dir = component_output_dir(
        "retriever", name, config, target=target, fit_mode=fit_mode
    )
    rank = np.tile(np.arange(1, cand.shape[1] + 1, dtype=np.int32), (cand.shape[0], 1))
    for i, size in enumerate(sizes):
        rank[i, int(size) :] = -1
    manifest = {
        "schema_version": 1,
        "artifact_type": "candidates",
        "stage": "retriever",
        "name": name,
        "config": config,
        "target": target,
        "artifact_mode": fit_mode,
        "created_at": utc_now(),
        "producer": {
            "command": producer_command,
            "cwd": ".",
        },
        "run_params": run_params,
        "source_code": source_refs,
        "fit_scope": fit_scope_from_source_policy(source_policy),
        "source_policy": source_policy,
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_target_future_turns": False,
            "same_user_memory_date_censored": name.startswith("personal_"),
            "popularity_tiebreaker": False,
        },
        "candidate_universe": "all_tracks",
        "retention": "top_k",
        "score_fields": ["score__primary"],
        "elapsed_sec": elapsed,
    }

    if target == "public_labeled":
        if public_rows is None or track_index is None:
            raise ValueError("public_labeled save requires public_rows and track_index")
        arrays = {
            "track_idx": cand.astype(np.int32, copy=False),
            "sizes": sizes.astype(np.int32, copy=False),
            "keys": encode_keys(
                [
                    (
                        f"{row['source_split']}:{row['session_id']}",
                        int(row["turn_number"]),
                    )
                    for row in public_rows
                ]
            ),
            "source_split": np.asarray(
                [str(row["source_split"]).encode("utf-8") for row in public_rows],
                dtype="S8",
            ),
            "folds": np.asarray(
                [int(row["fold"]) for row in public_rows], dtype=np.int16
            ),
            "rank": rank,
            "score__primary": scores.astype(np.float32, copy=False),
        }
        save_npz_artifact(out_dir, arrays, public_rows, manifest)
        metrics = public_labeled_metrics(public_rows, track_index, cand, sizes)
        metrics.update(
            {
                "artifact": str(out_dir.relative_to(REPO_ROOT)),
                "name": name,
                "config": config,
                "target": target,
                "artifact_mode": fit_mode,
            }
        )
        json_dump(
            component_results_dir(
                "retriever", name, config, target=target, fit_mode=fit_mode
            )
            / "scores.json",
            metrics,
        )
    elif target == "devset":
        save_candidate_artifact(
            out_dir,
            cand,
            sizes,
            target=target,
            manifest=manifest,
            rank=rank,
            score_arrays={"primary": scores},
        )
        metrics = candidate_metrics(cand, sizes, devset_gold_indices()[: cand.shape[0]])
        metrics.update(
            {
                "artifact": str(out_dir.relative_to(REPO_ROOT)),
                "name": name,
                "config": config,
                "target": target,
            }
        )
        json_dump(
            component_results_dir("retriever", name, config, target=target)
            / "scores.json",
            metrics,
        )
    else:
        save_candidate_artifact(
            out_dir,
            cand,
            sizes,
            target=target,
            manifest=manifest,
            rank=rank,
            score_arrays={"primary": scores},
            compress=True,
        )


def main(spec: FitFreeSpec) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", type=Path, default=Path("retriever/fit_free_sources.yaml")
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--target", choices=("devset", "public_labeled", "blind_b"), required=True
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/preprocessed/splits/cv5",
    )
    parser.add_argument("--top-k", type=int, default=500)
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    cfg = read_config(REPO_ROOT / args.config_file)
    if spec.name not in set(cfg["sources"]):
        raise ValueError(f"component {spec.name!r} is absent from {args.config_file}")

    public_rows: list[dict[str, Any]] | None = None
    if args.target == "devset":
        examples = common.build_examples_devset()
    elif args.target == "public_labeled":
        split_dir = (
            args.split_dir
            if args.split_dir.is_absolute()
            else REPO_ROOT / args.split_dir
        )
        examples, public_rows = build_public_labeled_examples(common, split_dir)
    else:
        examples = build_inference_examples(common, args.target)

    print(f"building fit-free retriever {spec.name}")
    track_index = common.build_track_index(spec.bm25_variants)

    refs = {
        "runner": file_ref(REPO_ROOT / "src/recsys2026/fit_free_runner.py"),
        "common": file_ref(REPO_ROOT / "src/recsys2026/retriever_common.py"),
        "component": file_ref(spec.source_path),
        "config": file_ref(REPO_ROOT / args.config_file),
    }

    producer_command = [
        "uv",
        "run",
        "python",
        "-m",
        f"retriever.{spec.name}.main",
        *sys.argv[1:],
    ]
    run_params = {
        "config": args.config,
        "target": args.target,
        "top_k": args.top_k,
        "config_file": str(args.config_file),
    }

    t0 = time.time()
    if spec.query_fn is not None:
        queries = [spec.query_fn(ex, track_index) for ex in examples]
        cand, sizes, scores = bm25_queries_scored(
            common,
            examples,
            track_index,
            str(spec.bm25_name),
            queries,
            args.top_k,
            spec.name,
        )
    else:
        cand, sizes, scores = count_scored(
            common,
            examples,
            track_index,
            spec.score_fn,
            spec.name,
            args.top_k,
        )
    elapsed = time.time() - t0
    fit_mode = "fit_free_all_rows" if args.target != "devset" else None
    save_source(
        spec.name,
        args.config,
        args.target,
        cand,
        sizes,
        scores,
        refs,
        elapsed,
        producer_command,
        run_params,
        source_policy_from_config(cfg, spec.name),
        fit_mode=fit_mode,
        public_rows=public_rows,
        track_index=track_index,
    )
    print(f"saved {spec.name}: mean_size={sizes.mean():.1f}, elapsed={elapsed:.1f}s")
