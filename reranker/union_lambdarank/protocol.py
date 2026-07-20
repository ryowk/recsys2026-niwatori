"""Dense-artifact and evaluation helpers for the union LambdaRank reranker."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


FeatureModule = Any


def load_feature_module() -> FeatureModule:
    """Return the feature library used by the final reranker."""
    from reranker.union_lambdarank import features

    return features


def key(example: Any) -> str:
    return f"{example.session_id}:{example.turn_number}"


def load_embedding_map(
    paths: list[Path],
    *,
    value_names: tuple[str, ...],
    expected_dim: int,
) -> dict[str, tuple[np.ndarray, ...]]:
    rows_by_key: dict[str, tuple[np.ndarray, ...]] = {}
    for path in paths:
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                keys = [str(value) for value in data["keys"]]
                values = tuple(np.asarray(data[name]) for name in value_names)
            if len(keys) != len(set(keys)) or any(
                value.ndim == 0 or value.shape[0] != len(keys) for value in values
            ):
                raise ValueError("key/value row mismatch or duplicate keys")
            if any(
                value.ndim != 2
                or value.shape[1] != expected_dim
                or not np.isfinite(value).all()
                for value in values
            ):
                raise ValueError("invalid dense embedding shape or non-finite values")
        except (KeyError, OSError, ValueError) as exc:
            print(f"ignoring invalid dense artifact {path}: {exc}")
            continue
        for row, cache_key in enumerate(keys):
            rows_by_key[cache_key] = tuple(value[row] for value in values)
        print(f"loaded dense artifact {path} rows={len(keys)}")
    return rows_by_key


def materialize_dense(
    features: FeatureModule,
    examples: list[Any],
    artifact_paths: list[Path],
    *,
    artifact_out: Path,
    batch_size: int,
) -> np.ndarray:
    """Load dense query vectors and encode any missing rows."""
    rows_by_key = load_embedding_map(
        artifact_paths,
        value_names=("embeddings",),
        expected_dim=int(features.DENSE_QUERY_DIM),
    )
    missing = [example for example in examples if key(example) not in rows_by_key]
    if missing:
        print(f"encoding missing dense rows={len(missing)}")
        from recsys2026.encoders import Qwen3TextEncoder

        encoder = Qwen3TextEncoder(batch_size=batch_size)
        embeddings = features.encode_dense_queries(
            missing,
            encoder,
            "last_user",
            artifact_path=None,
            desc="dense_qfeat[missing]",
        )
        for example, row in zip(missing, embeddings, strict=True):
            rows_by_key[key(example)] = (row,)
    result = np.asarray(
        [rows_by_key[key(example)][0] for example in examples], dtype=np.float32
    )
    if missing or not artifact_out.exists():
        artifact_out.parent.mkdir(parents=True, exist_ok=True)
        temp_path = artifact_out.with_name(f".{artifact_out.name}.tmp")
        with temp_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                keys=np.asarray([key(example) for example in examples]),
                embeddings=result,
            )
        temp_path.replace(artifact_out)
    return result


def ndcg_at_pos(position: int, k: int) -> float:
    if position < 0 or position >= k:
        return 0.0
    return 1.0 / math.log2(position + 2)


def evaluate_ranked(
    sources: list[str],
    examples: list[Any],
    ranked: np.ndarray,
    track_index: Any,
    *,
    top_k: int,
) -> dict[str, Any]:
    """Compute turn-balanced nDCG and per-split nDCG@20."""
    by_turn: dict[int, list[dict[str, float]]] = defaultdict(list)
    by_source: dict[str, list[float]] = defaultdict(list)
    for source, example, row in zip(sources, examples, ranked, strict=True):
        gold_idx = track_index.id_to_idx.get(example.gold_track_id or "")
        position = -1
        if gold_idx is not None:
            hits = np.flatnonzero(row[:top_k] == gold_idx)
            if len(hits):
                position = int(hits[0])
        values = {
            "ndcg@1": ndcg_at_pos(position, 1),
            "ndcg@10": ndcg_at_pos(position, 10),
            "ndcg@20": ndcg_at_pos(position, 20),
        }
        by_turn[int(example.turn_number)].append(values)
        by_source[source].append(values["ndcg@20"])

    turn_means = {
        turn: {
            name: sum(value[name] for value in values) / len(values)
            for name in ("ndcg@1", "ndcg@10", "ndcg@20")
        }
        for turn, values in by_turn.items()
    }
    result = {
        name: sum(value[name] for value in turn_means.values()) / len(turn_means)
        for name in ("ndcg@1", "ndcg@10", "ndcg@20")
    }
    result["n_examples"] = len(examples)
    for source, values in by_source.items():
        result[f"{source}_ndcg@20"] = float(sum(values) / len(values))
    return result
