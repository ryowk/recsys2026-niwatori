"""Retriever artifact evaluation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np

from .artifacts import track_id_lookup
from .data import load

K_VALUES = (20, 50, 100, 200)


def devset_gold_indices() -> np.ndarray:
    _, id_to_idx = track_id_lookup()
    ds = load("dataset", split="test")
    gold: list[int] = []
    for item in ds:
        conversations = list(item["conversations"])
        for target_turn in range(1, 9):
            current = [c for c in conversations if c["turn_number"] == target_turn]
            gold_tid = next(c["content"] for c in current if c["role"] == "music")
            gold.append(id_to_idx.get(gold_tid, -1))
    return np.asarray(gold, dtype=np.int64)


def candidate_metrics(
    track_idx: np.ndarray,
    sizes: np.ndarray,
    gold_idx: np.ndarray,
    *,
    k_values: tuple[int, ...] = K_VALUES,
) -> dict[str, Any]:
    valid = gold_idx >= 0
    n_valid = max(int(valid.sum()), 1)
    out: dict[str, Any] = {
        "n_examples": int(track_idx.shape[0]),
        "n_valid_gold": int(valid.sum()),
        "nonempty_rate": float((sizes > 0).mean()),
        "mean_size": float(sizes.mean()),
        "median_size": float(np.median(sizes)),
        "p90_size": float(np.percentile(sizes, 90)),
    }
    for k in k_values:
        kk = min(k, track_idx.shape[1])
        emitted = np.minimum(sizes, kk)
        total_emitted = int(emitted[valid].sum())
        hits = (
            (track_idx[:, :kk] == gold_idx[:, None]).any(axis=1) & valid & (emitted > 0)
        )
        n_hits = int(hits.sum())
        out[f"hits@{k}"] = n_hits
        out[f"emitted@{k}"] = total_emitted
        out[f"recall@{k}"] = float(n_hits / n_valid)
        out[f"precision@{k}"] = float(n_hits / total_emitted) if total_emitted else 0.0
        per_query_precision = np.zeros_like(sizes, dtype=np.float64)
        mask = valid & (emitted > 0)
        per_query_precision[mask] = hits[mask].astype(np.float64) / emitted[mask]
        out[f"macro_precision@{k}"] = float(per_query_precision[valid].mean())
    total_emitted = int(sizes[valid].sum())
    hits_all = np.zeros(track_idx.shape[0], dtype=bool)
    for i, size_raw in enumerate(sizes):
        size = int(size_raw)
        if valid[i] and size:
            hits_all[i] = bool((track_idx[i, :size] == gold_idx[i]).any())
    n_hits_all = int(hits_all[valid].sum())
    out["hits@all"] = n_hits_all
    out["emitted@all"] = total_emitted
    out["recall@all"] = float(n_hits_all / n_valid)
    out["precision@all"] = float(n_hits_all / total_emitted) if total_emitted else 0.0
    return out
