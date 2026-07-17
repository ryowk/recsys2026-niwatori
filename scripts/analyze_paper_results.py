#!/usr/bin/env python3
"""Generate paper metrics from submitted-design train-only -> devset artifacts."""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import yaml

from recsys2026.artifacts import decode_keys, json_dump, npz_dump, target_keys
from recsys2026.paths import REPO_ROOT
from recsys2026.retriever_eval import devset_gold_indices

sys.path.insert(0, str(REPO_ROOT))

from retriever.union.builder import load_sources, source_args_from_config


FAMILIES = {
    "lexical": {"bm25", "tfidf", "tag_intent", "exact_album_artist", "exact_title"},
    "semantic": {"twotower"},
    "history_entity": {"history_artist", "history_album", "last_artist", "last_album"},
    "behavioral": {"cooc_track", "transition_track", "cooc_album", "cooc_artist_name"},
}
SOURCE_ORDERING = {
    "bm25": "ranked",
    "tfidf": "ranked",
    "twotower": "ranked",
    "history_artist": "coarse",
    "history_album": "coarse",
    "last_artist": "set-valued",
    "last_album": "set-valued",
    "exact_album_artist": "set-valued",
    "tag_intent": "ranked",
    "cooc_track": "ranked",
    "transition_track": "ranked",
    "cooc_album": "ranked",
    "cooc_artist_name": "ranked",
    "exact_title": "set-valued",
}
FINAL_UNION_CONFIG = (
    REPO_ROOT / "retriever/union/configs/combined_tpd1_parts_cooc500_cv5.yaml"
)
FINAL_RERANKER_CONFIG = (
    REPO_ROOT
    / "reranker/union_lambdarank/configs/combined_tpd1_parts_cooc500_t200_cv5.yaml"
)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    fields = sorted({key for row in rows for key in row})
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(value, encoding="utf-8")
    temp_path.replace(path)


def candidate_hits(
    candidates: np.ndarray, sizes: np.ndarray, gold: np.ndarray, k: int | None
) -> np.ndarray:
    width = candidates.shape[1] if k is None else min(k, candidates.shape[1])
    if k is not None:
        return (candidates[:, :width] == gold[:, None]).any(axis=1)
    return np.asarray(
        [
            bool((candidates[i, : int(sizes[i])] == gold[i]).any())
            for i in range(len(gold))
        ],
        dtype=bool,
    )


def candidate_metrics(
    candidates: np.ndarray, sizes: np.ndarray, gold: np.ndarray
) -> dict[str, Any]:
    hits_at_20 = candidate_hits(candidates, sizes, gold, 20)
    hits_all = candidate_hits(candidates, sizes, gold, None)
    denominator = int(sizes.sum())
    return {
        "rows": int(len(gold)),
        "mean_candidates": float(sizes.mean()),
        "recall@20": float(hits_at_20.mean()),
        "recall@all": float(hits_all.mean()),
        "micro_precision@all": (
            float(hits_all.sum() / denominator) if denominator else 0.0
        ),
        "gold_hits@all": int(hits_all.sum()),
    }


def ranked_metrics(
    ranked: np.ndarray,
    gold: np.ndarray,
) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": int(len(gold))}
    for k in (1, 10, 20):
        out[f"ndcg@{k}"] = float(row_ndcg(ranked, gold, k=k).mean())
    return out


def row_ndcg(ranked: np.ndarray, gold: np.ndarray, *, k: int = 20) -> np.ndarray:
    values = np.zeros(len(gold), dtype=np.float64)
    for i in range(len(gold)):
        positions = np.flatnonzero(ranked[i, :k] == gold[i])
        if len(positions):
            values[i] = 1.0 / math.log2(int(positions[0]) + 2)
    return values


def source_row(source: dict[str, Any], row_i: int) -> np.ndarray:
    arrays = source["arrays"]
    size = int(arrays["sizes"][row_i])
    if source.get("max_candidates") is not None:
        size = min(size, int(source["max_candidates"]))
    candidates = np.asarray(arrays["track_idx"][row_i, :size], dtype=np.int32)
    if source.get("min_score") is not None:
        score_key = str(source.get("score_field") or "score__primary")
        scores = np.asarray(arrays[score_key][row_i, :size], dtype=np.float32)
        keep = np.isfinite(scores) & (scores >= float(source["min_score"]))
        candidates = candidates[keep]
    return candidates[candidates >= 0]


def source_row_scored(
    source: dict[str, Any], row_i: int
) -> tuple[np.ndarray, np.ndarray]:
    arrays = source["arrays"]
    size = int(arrays["sizes"][row_i])
    if source.get("max_candidates") is not None:
        size = min(size, int(source["max_candidates"]))
    candidates = np.asarray(arrays["track_idx"][row_i, :size], dtype=np.int32)
    score_key = str(source.get("score_field") or "")
    if not score_key or score_key not in arrays:
        if "score__primary" in arrays:
            score_key = "score__primary"
        elif "score__tfidf" in arrays:
            score_key = "score__tfidf"
    scores = (
        np.asarray(arrays[score_key][row_i, :size], dtype=np.float64)
        if score_key in arrays
        else np.ones(size, dtype=np.float64)
    )
    keep = (candidates >= 0) & np.isfinite(scores)
    if source.get("min_score") is not None:
        keep &= scores >= float(source["min_score"])
    candidates = candidates[keep]
    scores = scores[keep]
    order = np.argsort(-scores, kind="stable")
    return candidates[order], scores[order]


def average_tied_ranks(scores: np.ndarray) -> np.ndarray:
    """Return one-based average ranks while preserving equal-score ties."""
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[stop] == scores[start]:
            stop += 1
        ranks[start:stop] = ((start + 1) + stop) / 2.0
        start = stop
    return ranks


def source_ranking_metrics(
    source: dict[str, Any], gold: np.ndarray, *, k: int = 20
) -> dict[str, float | str]:
    """Measure expected nDCG under random within-tie order."""
    tie_aware_total = 0.0
    for row_i, gold_track in enumerate(gold):
        candidates, scores = source_row_scored(source, row_i)
        positions = np.flatnonzero(candidates == gold_track)
        if not len(positions):
            continue
        position = int(positions[0])
        tie_start = position
        while tie_start > 0 and scores[tie_start - 1] == scores[position]:
            tie_start -= 1
        tie_stop = position + 1
        while tie_stop < len(scores) and scores[tie_stop] == scores[position]:
            tie_stop += 1
        tie_aware_total += sum(
            1.0 / math.log2(rank + 1)
            for rank in range(tie_start + 1, min(tie_stop, k) + 1)
        ) / float(tie_stop - tie_start)
    return {
        "ordering": SOURCE_ORDERING[str(source["name"])],
        "ndcg@20_tie_aware": tie_aware_total / len(gold),
    }


def materialize_source(source: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rows = [source_row(source, i) for i in range(len(source["arrays"]["sizes"]))]
    width = max((len(row) for row in rows), default=0)
    candidates = np.full((len(rows), width), -1, dtype=np.int32)
    sizes = np.asarray([len(row) for row in rows], dtype=np.int32)
    for i, row in enumerate(rows):
        candidates[i, : len(row)] = row
    return candidates, sizes


def decorate_sources(
    config_file: Path, target: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    args, config, entries = source_args_from_config(config_file, target)
    sources = load_sources(args)
    for source in sources:
        entry = entries[source["name"]]
        if entry.get("max_candidates") is not None:
            source["max_candidates"] = int(entry["max_candidates"])
        if entry.get("min_score") is not None:
            source["min_score"] = float(entry["min_score"])
            source["score_field"] = str(entry.get("score_field") or "score__primary")
    return sources, config


def validate_fit_scope(
    manifest: dict[str, Any], *, source: str, target: str
) -> dict[str, Any]:
    """Validate the fixed paper protocol encoded in one source manifest."""
    fit_scope = dict(manifest.get("fit_scope") or {})
    if manifest.get("target") != target:
        raise ValueError(
            f"{source}: manifest target={manifest.get('target')!r}, expected {target!r}"
        )
    if fit_scope.get("uses_devset_for_fit") is not False:
        raise ValueError(f"{source}/{target}: uses_devset_for_fit must be false")
    if fit_scope.get("uses_blind_for_fit") is not False:
        raise ValueError(f"{source}/{target}: uses_blind_for_fit must be false")
    requires_fit = bool(fit_scope.get("requires_labeled_fit", False))
    if requires_fit:
        expected_mode = "train5_oof" if target == "public_labeled" else "full_train"
        if manifest.get("artifact_mode") != expected_mode:
            raise ValueError(
                f"{source}/{target}: artifact_mode={manifest.get('artifact_mode')!r}, expected {expected_mode!r}"
            )
        if "devset" in set(fit_scope.get("fit_splits") or []):
            raise ValueError(f"{source}/{target}: devset appears in fit_splits")
        if target == "public_labeled":
            excluded = fit_scope.get("target_row_excluded_from_fit")
            if excluded is not True:
                excluded = (manifest.get("leak_check") or {}).get(
                    "target_row_excluded_from_fit"
                )
            if excluded is not True:
                raise ValueError(
                    f"{source}: OOF manifest does not assert target-row exclusion"
                )
    return {
        "source": source,
        "target": target,
        "artifact_mode": manifest.get("artifact_mode"),
        "requires_labeled_fit": requires_fit,
        "fit_splits": list(fit_scope.get("fit_splits") or []),
        "uses_devset_for_fit": False,
        "target_row_excluded_from_fit": (
            True if requires_fit and target == "public_labeled" else None
        ),
    }


def validate_submission_parity(paper_union_config: Path) -> dict[str, Any]:
    """Ensure the paper base is the final Blind-B system shifted to train -> devset."""
    final_union = yaml.safe_load(FINAL_UNION_CONFIG.read_text()) or {}
    paper_union = yaml.safe_load(paper_union_config.read_text()) or {}

    def source_signature(config: dict[str, Any]) -> list[tuple[str, Any, Any]]:
        return [
            (str(source["name"]), source.get("max_candidates"), source.get("min_score"))
            for source in config.get("sources") or []
        ]

    final_sources = source_signature(final_union)
    paper_sources = source_signature(paper_union)
    if paper_sources != final_sources:
        raise ValueError(
            "paper source order/caps/thresholds differ from final Blind-B config"
        )
    if paper_union.get("union_rule") != final_union.get("union_rule"):
        raise ValueError("paper union_rule differs from final Blind-B config")

    final_reranker = yaml.safe_load(FINAL_RERANKER_CONFIG.read_text()) or {}
    paper_config_dir = REPO_ROOT / "reranker/union_lambdarank/configs"
    variants = [
        "paper_train5_devset_full",
        "paper_train5_devset_no_provenance",
        "paper_train5_devset_provenance_only",
        "paper_train5_devset_without_tpd1",
    ]
    fixed_keys = [
        "max_candidates",
        "top_k",
        "train_positive_only",
        "lgbm",
        "feature_build",
    ]
    checked_variants: list[str] = []
    for variant in variants:
        paper_reranker = (
            yaml.safe_load((paper_config_dir / f"{variant}.yaml").read_text()) or {}
        )
        for key in fixed_keys:
            if paper_reranker.get(key) != final_reranker.get(key):
                raise ValueError(
                    f"{variant}: {key} differs from final Blind-B reranker config"
                )
        checked_variants.append(variant)
    return {
        "passed": True,
        "final_union_config": rel(FINAL_UNION_CONFIG),
        "final_reranker_config": rel(FINAL_RERANKER_CONFIG),
        "source_order_caps_thresholds_equal": True,
        "union_rule_equal": True,
        "reranker_fixed_keys": fixed_keys,
        "checked_variants": checked_variants,
    }


def validate_paper_protocol(
    config_file: Path,
    dev_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fail before reporting if train OOF or train-full/dev artifacts violate scope."""
    train_args, _, _ = source_args_from_config(config_file, "public_labeled")
    dev_by_name = {str(source["name"]): source for source in dev_sources}
    expected_dev_keys = target_keys("devset")
    checks: list[dict[str, Any]] = []
    reference_train_keys: list[tuple[str, int]] | None = None

    for raw in train_args:
        name, raw_path = raw.split("=", 1)
        train_dir = Path(raw_path)
        if not train_dir.is_absolute():
            train_dir = REPO_ROOT / train_dir
        train_manifest = json.loads((train_dir / "manifest.json").read_text())
        checks.append(
            validate_fit_scope(train_manifest, source=name, target="public_labeled")
        )
        with np.load(train_dir / "candidates.npz", allow_pickle=False) as data:
            train_keys = decode_keys(data["keys"])
            if "folds" not in data.files:
                raise ValueError(f"{name}: train-side artifact has no folds array")
            folds = {int(value) for value in np.asarray(data["folds"]).tolist()}
        if folds != {0, 1, 2, 3, 4}:
            raise ValueError(
                f"{name}: train-side folds={sorted(folds)}, expected [0, 1, 2, 3, 4]"
            )
        if reference_train_keys is None:
            reference_train_keys = train_keys
        elif train_keys != reference_train_keys:
            raise ValueError(
                f"{name}: train-side keys do not align with the other sources"
            )

        dev_source = dev_by_name.get(name)
        if dev_source is None:
            raise ValueError(f"{name}: missing devset source")
        checks.append(
            validate_fit_scope(dev_source["manifest"], source=name, target="devset")
        )
        if decode_keys(dev_source["arrays"]["keys"]) != expected_dev_keys:
            raise ValueError(f"{name}: devset keys do not match official target order")

    if reference_train_keys is None:
        raise ValueError("paper config has no train-side sources")
    return {
        "passed": True,
        "train_rows": len(reference_train_keys),
        "devset_rows": len(expected_dev_keys),
        "train_folds": [0, 1, 2, 3, 4],
        "source_checks": checks,
    }


def validate_union_manifest(
    manifest: dict[str, Any],
    *,
    target: str,
    expected_sources: list[str],
) -> dict[str, Any]:
    if manifest.get("target") != target:
        raise ValueError(f"union/{target}: manifest target={manifest.get('target')!r}")
    rule = dict(manifest.get("union_rule") or {})
    if rule.get("type") != "ordered_unique":
        raise ValueError(
            f"union/{target}: expected ordered_unique, got {rule.get('type')!r}"
        )
    if rule.get("max_candidates") is not None:
        raise ValueError(
            f"union/{target}: final submission union must not have a global cap"
        )
    actual_sources = [
        str(ref.get("name")) for ref in manifest.get("source_artifacts") or []
    ]
    if actual_sources != expected_sources:
        raise ValueError(
            f"union/{target}: source order differs from the final submission"
        )
    if list(rule.get("source_order") or []) != expected_sources:
        raise ValueError(
            f"union/{target}: manifest source_order differs from the fixed config"
        )
    if rule.get("tie_breaker") != "source_order_then_source_rank":
        raise ValueError(
            f"union/{target}: unexpected tie_breaker={rule.get('tie_breaker')!r}"
        )
    fit_scope = dict(manifest.get("fit_scope") or {})
    if fit_scope.get("uses_devset_for_fit") is not False:
        raise ValueError(f"union/{target}: uses_devset_for_fit must be false")
    if fit_scope.get("uses_blind_for_fit") is not False:
        raise ValueError(f"union/{target}: uses_blind_for_fit must be false")
    return {
        "target": target,
        "type": "ordered_unique",
        "source_order": actual_sources,
        "tie_breaker": "source_order_then_source_rank",
        "max_candidates": None,
    }


def validate_reranker_submission_features(
    manifest: dict[str, Any], *, config: str
) -> dict[str, Any]:
    params = dict(manifest.get("params") or {})
    neutralized_raw = params.get("neutralize_base_features") or ""
    neutralized = {
        value.strip()
        for value in (
            neutralized_raw.split(",")
            if isinstance(neutralized_raw, str)
            else neutralized_raw
        )
        if value.strip()
    }
    required = {"candidate_rank", "log_candidate_rank", "reciprocal_candidate_rank"}
    if not required.issubset(neutralized):
        raise ValueError(
            f"{config}: union-position features are active: {sorted(required - neutralized)}"
        )
    return {
        "config": config,
        "neutralized_union_position_features": sorted(required),
        "primary_candidate_score": "fixed_zero",
    }


def union_hit_mask(
    source_names: Iterable[str], hit_by_source: dict[str, np.ndarray]
) -> np.ndarray:
    names = list(source_names)
    if not names:
        return np.zeros_like(next(iter(hit_by_source.values())))
    return np.logical_or.reduce([hit_by_source[name] for name in names])


def rrf_rank(
    sources: list[dict[str, Any]],
    *,
    constant: int = 60,
    tie_aware: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = len(sources[0]["arrays"]["sizes"])
    ranked_rows: list[np.ndarray] = []
    for row_i in range(n_rows):
        scores: dict[int, float] = defaultdict(float)
        best_rank: dict[int, float] = {}
        for source in sources:
            if tie_aware:
                candidates, source_scores = source_row_scored(source, row_i)
                source_ranks = average_tied_ranks(source_scores)
            else:
                candidates = source_row(source, row_i)
                source_ranks = np.arange(1, len(candidates) + 1, dtype=np.float64)
            seen: set[int] = set()
            for tid_raw, rank_raw in zip(candidates, source_ranks, strict=True):
                tid = int(tid_raw)
                if tid in seen:
                    continue
                seen.add(tid)
                rank = float(rank_raw)
                scores[tid] += 1.0 / float(constant + rank)
                best_rank[tid] = min(best_rank.get(tid, rank), rank)
        ranked_rows.append(
            np.asarray(
                sorted(scores, key=lambda tid: (-scores[tid], best_rank[tid], tid)),
                dtype=np.int32,
            )
        )
    width = max(len(row) for row in ranked_rows)
    ranked = np.full((n_rows, width), -1, dtype=np.int32)
    sizes = np.asarray([len(row) for row in ranked_rows], dtype=np.int32)
    for i, row in enumerate(ranked_rows):
        ranked[i, : len(row)] = row
    return ranked, sizes


def error_decomposition(
    ranked: np.ndarray, pool: np.ndarray, pool_sizes: np.ndarray, gold: np.ndarray
) -> dict[str, Any]:
    retrieval_miss = np.zeros(len(gold), dtype=np.float64)
    rerank_miss = np.zeros(len(gold), dtype=np.float64)
    ordering = np.zeros(len(gold), dtype=np.float64)
    ranks = np.full(len(gold), -1, dtype=np.int32)
    pool_hits = candidate_hits(pool, pool_sizes, gold, None)
    for i in range(len(gold)):
        if not pool_hits[i]:
            retrieval_miss[i] = 1.0
            continue
        positions = np.flatnonzero(ranked[i, :20] == gold[i])
        if not len(positions):
            rerank_miss[i] = 1.0
            continue
        rank = int(positions[0]) + 1
        ranks[i] = rank
        ordering[i] = 1.0 - 1.0 / math.log2(rank + 1)
    return {
        "retrieval_miss_loss": float(retrieval_miss.mean()),
        "reranking_miss_loss": float(rerank_miss.mean()),
        "ordering_loss": float(ordering.mean()),
        "sum": float((retrieval_miss + rerank_miss + ordering).mean()),
        "retrieval_miss_rows": int(retrieval_miss.sum()),
        "reranking_miss_rows": int(rerank_miss.sum()),
        "top20_hit_rows": int((ranks > 0).sum()),
        "ordering_loss_rows": int((ranks > 1).sum()),
    }


def plot_retriever_scatter(
    source_metrics: list[dict[str, Any]],
    union_metrics: dict[str, Any],
    output_path: Path,
) -> None:
    colors = {
        "lexical": "#31688e",
        "semantic": "#35b779",
        "history_entity": "#e6ab02",
        "behavioral": "#d95f02",
    }
    labels = {
        "bm25": "BM25",
        "tfidf": "TF-IDF",
        "tag_intent": "Tag",
        "exact_album_artist": "Exact alb.",
        "exact_title": "Exact title",
        "twotower": "Two-tower",
        "history_artist": "Hist. artist",
        "history_album": "Hist. album",
        "last_artist": "Last artist",
        "last_album": "Last album",
        "cooc_track": "Track cooc.",
        "transition_track": "Transition",
        "cooc_album": "Album cooc.",
        "cooc_artist_name": "Artist cooc.",
    }
    markers = {
        "lexical": "o",
        "semantic": "s",
        "history_entity": "^",
        "behavioral": "D",
    }
    label_offsets = {
        "bm25": (13, 11, "left"),
        "tfidf": (-12, 3, "right"),
        "twotower": (13, -10, "left"),
        "history_artist": (-3, 6, "right"),
        "history_album": (-3, 8, "right"),
        "last_artist": (-3, -11, "right"),
        "last_album": (3, -6, "left"),
        "exact_album_artist": (3, 5, "left"),
        "tag_intent": (-5, 4, "right"),
        "exact_title": (3, 5, "left"),
        "cooc_track": (5, 5, "left"),
        "transition_track": (5, 8, "left"),
        "cooc_album": (5, -8, "left"),
        "cooc_artist_name": (5, 5, "left"),
    }
    fig, ax = plt.subplots(figsize=(3.35, 2.50), facecolor="white")
    ax.set_facecolor("white")
    for family, color in colors.items():
        rows = [row for row in source_metrics if row["family"] == family]
        ax.scatter(
            [row["mean_candidates"] for row in rows],
            [row["recall@all"] for row in rows],
            s=18,
            facecolors=color,
            marker=markers[family],
            edgecolors=color,
            linewidth=0.9,
            label=family.replace("_", "/"),
            zorder=3,
        )
        for row in rows:
            source = str(row["source"])
            dx, dy, horizontal = label_offsets[source]
            ax.annotate(
                labels[source],
                (float(row["mean_candidates"]), float(row["recall@all"])),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=horizontal,
                fontsize=5.3,
                arrowprops={
                    "arrowstyle": "-",
                    "color": "#555555",
                    "linewidth": 0.38,
                    "shrinkA": 0,
                    "shrinkB": 2,
                },
            )
    ax.scatter(
        [union_metrics["mean_candidates"]],
        [union_metrics["recall@all"]],
        marker="*",
        s=65,
        color="#202020",
        label="union",
        zorder=4,
    )
    ax.annotate(
        "Union",
        (float(union_metrics["mean_candidates"]), float(union_metrics["recall@all"])),
        xytext=(-16, -13),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=6,
        arrowprops={
            "arrowstyle": "-",
            "color": "#555555",
            "linewidth": 0.5,
            "shrinkA": 0,
            "shrinkB": 2,
        },
    )
    ax.set_xscale("log")
    ax.set_xlabel("Mean candidates per row", fontsize=7)
    ax.set_ylabel("Recall@all", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.margins(y=0.08)
    ax.grid(True, which="both", linewidth=0.35, alpha=0.35)
    ax.legend(fontsize=5.3, ncol=2, frameon=False, loc="upper left")
    fig.tight_layout(pad=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    fig.savefig(
        temp_path,
        bbox_inches="tight",
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)
    temp_path.replace(output_path)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def latex_retriever_rows(report: dict[str, Any]) -> str:
    labels = {
        "bm25": "BM25",
        "tfidf": "TF--IDF",
        "tag_intent": "Tag intent",
        "exact_album_artist": "Exact album/artist",
        "exact_title": "Exact title/artist",
        "twotower": "Two-tower",
        "history_artist": "History artist",
        "history_album": "History album",
        "last_artist": "Last artist",
        "last_album": "Last album",
        "cooc_track": "Track co-occurrence",
        "transition_track": "Track transition",
        "cooc_album": "Album co-occurrence",
        "cooc_artist_name": "Artist co-occurrence",
    }
    families = {
        "lexical": "Lexical",
        "semantic": "Semantic",
        "history_entity": "History/entity",
        "behavioral": "Behavioral",
    }
    family_order = {
        "lexical": 0,
        "semantic": 1,
        "history_entity": 2,
        "behavioral": 3,
    }
    ordering_order = {"ranked": 0, "coarse": 1, "set-valued": 2}
    ablations = {
        str(row["name"]): row
        for row in report["complementarity"]
        if row["kind"] == "source"
    }
    lines = ["% Generated by scripts/analyze_paper_results.py."]
    for row in sorted(
        report["sources"],
        key=lambda item: (
            family_order[str(item["family"])],
            ordering_order[str(item["ordering"])],
            -float(item["recall@all"]),
            str(item["source"]),
        ),
    ):
        ablation = ablations[str(row["source"])]
        lines.append(
            f"{labels[str(row['source'])]} & {families[str(row['family'])]} & "
            f"{row['mean_candidates']:.1f} & "
            f"{row['ndcg@20_tie_aware']:.4f} & {row['recall@20']:.4f} & {row['recall@all']:.4f} & "
            f"{row['micro_precision@all']:.4f} & {int(ablation['unique_gold_hits'])} & "
            f"{float(ablation['leave_one_out_delta_recall_all']):.4f} \\\\"
        )
    union = report["union"]
    lines.extend(
        [
            "\\midrule",
            f"Union & All & {union['mean_candidates']:.1f} & -- & -- & "
            f"{union['recall@all']:.4f} & {union['micro_precision@all']:.4f} & -- & -- \\\\",
            "\\bottomrule",
        ]
    )
    return "\n".join(lines) + "\n"


def latex_family_rows(report: dict[str, Any]) -> str:
    labels = {
        "lexical": "Lexical",
        "semantic": "Semantic",
        "history_entity": "History/entity",
        "behavioral": "Behavioral",
    }
    rows = {
        str(row["name"]): row
        for row in report["complementarity"]
        if row["kind"] == "family"
    }
    lines = ["% Generated by scripts/analyze_paper_results.py."]
    for family in ("lexical", "semantic", "history_entity", "behavioral"):
        row = rows[family]
        lines.append(
            f"{labels[family]} & {float(row['recall@all']):.4f} & "
            f"{int(row['unique_gold_hits'])} & "
            f"{float(row['leave_one_out_delta_recall_all']):.4f} \\\\"
        )
    lines.append("\\bottomrule")
    return "\n".join(lines) + "\n"


def latex_macros(report: dict[str, Any]) -> str:
    ranking = report["ranking"]
    full = ranking["full"]["all"]
    no_prov = ranking["no_provenance"]["all"]
    prov = ranking["provenance_only"]["all"]
    no_tpd1 = ranking["without_tpd1"]["all"]
    rrf = ranking["rrf"]["all"]
    feature_delta = float(full["ndcg@20"] - no_prov["ndcg@20"])
    reranker_delta = float(full["ndcg@20"] - rrf["ndcg@20"])
    tpd_delta = float(full["ndcg@20"] - no_tpd1["ndcg@20"])
    tpd_union_delta = float(report["external_union_ablation"]["delta_recall@all"])
    union = report["union"]
    error = report["error_decomposition"]
    rrf_twenty = [float(row["ndcg@20"]) for row in report["rrf_sensitivity"]]
    source_metrics = {str(row["source"]): row for row in report["sources"]}
    source_ablation = {
        str(row["name"]): row
        for row in report["complementarity"]
        if row["kind"] == "source"
    }
    family_ablation = {
        str(row["name"]): row
        for row in report["complementarity"]
        if row["kind"] == "family"
    }
    commands = {
        "DevRRFOne": fmt(rrf["ndcg@1"]),
        "DevRRFTen": fmt(rrf["ndcg@10"]),
        "DevRRFTwenty": fmt(rrf["ndcg@20"]),
        "DevNoProvOne": fmt(no_prov["ndcg@1"]),
        "DevNoProvTen": fmt(no_prov["ndcg@10"]),
        "DevNoProvTwenty": fmt(no_prov["ndcg@20"]),
        "DevProvOnlyOne": fmt(prov["ndcg@1"]),
        "DevProvOnlyTen": fmt(prov["ndcg@10"]),
        "DevProvOnlyTwenty": fmt(prov["ndcg@20"]),
        "DevNoTpdOne": fmt(no_tpd1["ndcg@1"]),
        "DevNoTpdTen": fmt(no_tpd1["ndcg@10"]),
        "DevNoTpdTwenty": fmt(no_tpd1["ndcg@20"]),
        "DevFullOne": fmt(full["ndcg@1"]),
        "DevFullTen": fmt(full["ndcg@10"]),
        "DevFullTwenty": fmt(full["ndcg@20"]),
        "DevUnionRecallAll": fmt(union["recall@all"]),
        "DevUnionMeanCandidates": f"{union['mean_candidates']:.1f}",
        "DevBMRecallAll": fmt(float(source_metrics["bm25"]["recall@all"])),
        "DevTwoTowerRecallAll": fmt(float(source_metrics["twotower"]["recall@all"])),
        "DevTwoTowerUniqueHits": str(
            int(source_ablation["twotower"]["unique_gold_hits"])
        ),
        "DevBMUniqueHits": str(int(source_ablation["bm25"]["unique_gold_hits"])),
        "DevTrackCoocUniqueHits": str(
            int(source_ablation["cooc_track"]["unique_gold_hits"])
        ),
        "DevSemanticAblation": fmt(
            float(family_ablation["semantic"]["leave_one_out_delta_recall_all"])
        ),
        "DevLexicalAblation": fmt(
            float(family_ablation["lexical"]["leave_one_out_delta_recall_all"])
        ),
        "DevBehavioralAblation": fmt(
            float(family_ablation["behavioral"]["leave_one_out_delta_recall_all"])
        ),
        "RetrievalMissLoss": fmt(error["retrieval_miss_loss"]),
        "RerankMissLoss": fmt(error["reranking_miss_loss"]),
        "OrderingLoss": fmt(error["ordering_loss"]),
        "RetrievalMissRows": f"{int(error['retrieval_miss_rows']):,}",
        "RerankMissRows": f"{int(error['reranking_miss_rows']):,}",
        "OrderingLossRows": f"{int(error['ordering_loss_rows']):,}",
        "DevRRFMinTwenty": fmt(min(rrf_twenty)),
        "DevRRFMaxTwenty": fmt(max(rrf_twenty)),
        "DevFullVsRRFGain": fmt(reranker_delta),
        "DevPerRetrieverFeatureGain": fmt(feature_delta),
        "DevTpdRecallGain": fmt(tpd_union_delta),
        "DevTpdNDCGGain": fmt(tpd_delta),
    }
    return (
        "% Generated by scripts/analyze_paper_results.py.\n"
        + "\n".join(
            f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in commands.items()
        )
        + "\n"
    )


def main() -> None:
    config_name = "paper_train5_devset"
    config_file = REPO_ROOT / "retriever/union/configs/paper_train5_devset.yaml"
    output_dir = REPO_ROOT / "artifacts/results/paper/train5_devset"
    paper_dir = REPO_ROOT / "paper"
    paper_results = paper_dir / "generated_results.tex"
    paper_retriever_rows = paper_dir / "generated_retriever_rows.tex"
    paper_family_rows = paper_dir / "generated_family_rows.tex"
    union_dir = REPO_ROOT / "artifacts/runs/retriever/union" / config_name / "devset"
    gold = devset_gold_indices()

    sources, config = decorate_sources(config_file, "devset")
    protocol_validation = validate_paper_protocol(config_file, sources)
    protocol_validation["submission_parity"] = validate_submission_parity(config_file)
    expected_source_names = [str(source["name"]) for source in sources]
    union_checks = []
    for target in ("public_labeled", "devset"):
        manifest_path = (
            REPO_ROOT
            / "artifacts/runs/retriever/union"
            / config_name
            / target
            / "manifest.json"
        )
        union_checks.append(
            validate_union_manifest(
                json.loads(manifest_path.read_text()),
                target=target,
                expected_sources=expected_source_names,
            )
        )
    protocol_validation["union_checks"] = union_checks
    source_metrics: list[dict[str, Any]] = []
    hit_by_source: dict[str, np.ndarray] = {}
    for source in sources:
        candidates, sizes = materialize_source(source)
        metrics = candidate_metrics(candidates, sizes, gold)
        metrics.update(source_ranking_metrics(source, gold))
        metrics.update(
            {
                "source": source["name"],
                "family": next(
                    name
                    for name, members in FAMILIES.items()
                    if source["name"] in members
                ),
            }
        )
        source_metrics.append(metrics)
        hit_by_source[source["name"]] = candidate_hits(candidates, sizes, gold, None)
    source_metrics.sort(
        key=lambda item: (-float(item["micro_precision@all"]), str(item["source"]))
    )

    full_hits = union_hit_mask(hit_by_source, hit_by_source)
    complementarity: list[dict[str, Any]] = []
    for source in sources:
        name = source["name"]
        without = union_hit_mask(
            (other for other in hit_by_source if other != name), hit_by_source
        )
        unique = hit_by_source[name] & ~without
        complementarity.append(
            {
                "kind": "source",
                "name": name,
                "recall@all": float(hit_by_source[name].mean()),
                "unique_gold_hits": int(unique.sum()),
                "leave_one_out_delta_recall_all": float(
                    full_hits.mean() - without.mean()
                ),
            }
        )
    for family, members in FAMILIES.items():
        without = union_hit_mask(
            (name for name in hit_by_source if name not in members), hit_by_source
        )
        family_hits = union_hit_mask(
            (name for name in hit_by_source if name in members), hit_by_source
        )
        complementarity.append(
            {
                "kind": "family",
                "name": family,
                "recall@all": float(family_hits.mean()),
                "unique_gold_hits": int((family_hits & ~without).sum()),
                "leave_one_out_delta_recall_all": float(
                    full_hits.mean() - without.mean()
                ),
            }
        )
    complementarity.sort(
        key=lambda item: (
            -int(item["unique_gold_hits"]),
            str(item["kind"]),
            str(item["name"]),
        )
    )

    with np.load(union_dir / "candidates.npz", allow_pickle=False) as data:
        union_candidates = np.asarray(data["track_idx"], dtype=np.int32)
        union_sizes = np.asarray(data["sizes"], dtype=np.int32)
        union_keys = decode_keys(data["keys"])
    if len(union_keys) != len(gold):
        raise ValueError(f"devset union row mismatch: {len(union_keys)} != {len(gold)}")
    union_metrics = candidate_metrics(union_candidates, union_sizes, gold)
    union_hits = candidate_hits(union_candidates, union_sizes, gold, None)
    if not np.array_equal(union_hits, full_hits):
        raise ValueError(
            "materialized union gold coverage does not match the set union of sources"
        )
    protocol_validation["union_set_matches_sources"] = True

    no_tpd_union_dir = (
        REPO_ROOT
        / "artifacts/runs/retriever/union/paper_train5_devset_without_tpd1/devset"
    )
    with np.load(no_tpd_union_dir / "candidates.npz", allow_pickle=False) as data:
        no_tpd_union_candidates = np.asarray(data["track_idx"], dtype=np.int32)
        no_tpd_union_sizes = np.asarray(data["sizes"], dtype=np.int32)
        no_tpd_union_keys = decode_keys(data["keys"])
    if no_tpd_union_keys != union_keys:
        raise ValueError("without-TPD1 union keys do not match the main devset union")
    no_tpd_union_metrics = candidate_metrics(
        no_tpd_union_candidates, no_tpd_union_sizes, gold
    )
    external_union_ablation = {
        "with_tpd1_mean_candidates": union_metrics["mean_candidates"],
        "without_tpd1_mean_candidates": no_tpd_union_metrics["mean_candidates"],
        "delta_mean_candidates": float(
            union_metrics["mean_candidates"] - no_tpd_union_metrics["mean_candidates"]
        ),
        "with_tpd1_recall@all": union_metrics["recall@all"],
        "without_tpd1_recall@all": no_tpd_union_metrics["recall@all"],
        "delta_recall@all": float(
            union_metrics["recall@all"] - no_tpd_union_metrics["recall@all"]
        ),
        "with_tpd1_recall@20": union_metrics["recall@20"],
        "without_tpd1_recall@20": no_tpd_union_metrics["recall@20"],
        "delta_recall@20": float(
            union_metrics["recall@20"] - no_tpd_union_metrics["recall@20"]
        ),
    }
    rrf_ranked, rrf_sizes = rrf_rank(sources, constant=60, tie_aware=True)
    rrf_sensitivity: list[dict[str, Any]] = []
    for constant in (10, 30, 60, 100):
        ranked_for_constant = (
            rrf_ranked
            if constant == 60
            else rrf_rank(sources, constant=constant, tie_aware=True)[0]
        )
        rrf_sensitivity.append(
            {
                "constant": constant,
                **ranked_metrics(ranked_for_constant, gold),
            }
        )
    npz_dump(
        output_dir / "rrf_k60_tie_aware_ranked.npz",
        {"track_idx": rrf_ranked, "sizes": rrf_sizes},
        compress=True,
    )
    plot_retriever_scatter(
        source_metrics,
        union_metrics,
        output_dir / "retriever_recall_vs_size.pdf",
    )
    paper_figure_dir = paper_dir / "figures"
    paper_figure_dir.mkdir(parents=True, exist_ok=True)
    name = "retriever_recall_vs_size.pdf"
    shutil.copy2(output_dir / name, paper_figure_dir / name)

    variants = {
        "full": "paper_train5_devset_full",
        "no_provenance": "paper_train5_devset_no_provenance",
        "provenance_only": "paper_train5_devset_provenance_only",
        "without_tpd1": "paper_train5_devset_without_tpd1",
    }
    ranking: dict[str, Any] = {
        "rrf": {
            "all": ranked_metrics(rrf_ranked, gold),
            "tie_policy": "average rank for equal source scores",
            "sources": [str(source["name"]) for source in sources],
        }
    }
    full_ranked: np.ndarray | None = None
    reranker_checks: list[dict[str, Any]] = []
    for variant, config_name in variants.items():
        ranked_dir = (
            REPO_ROOT
            / "artifacts/runs/reranker/union_lambdarank"
            / config_name
            / "full_train/devset"
        )
        manifest = json.loads((ranked_dir / "manifest.json").read_text())
        reranker_checks.append(
            validate_reranker_submission_features(
                manifest,
                config=config_name,
            )
        )
        with np.load(ranked_dir / "ranked.npz", allow_pickle=False) as data:
            ranked = np.asarray(data["track_idx"], dtype=np.int32)
            ranked_keys = decode_keys(data["keys"])
        if ranked_keys != union_keys:
            raise ValueError(f"{variant} ranked keys do not match the devset union")
        ranking[variant] = {
            "all": ranked_metrics(ranked, gold),
            "artifact": rel(ranked_dir),
        }
        if variant == "full":
            full_ranked = ranked
    protocol_validation["reranker_submission_feature_checks"] = reranker_checks
    if full_ranked is None:
        raise RuntimeError("full reranker output was not loaded")

    report = {
        "protocol": "train-only 5-fold OOF features; train-full -> official devset inference",
        "config": config,
        "rows": len(gold),
        "protocol_validation": protocol_validation,
        "union": union_metrics,
        "sources": source_metrics,
        "complementarity": complementarity,
        "ranking": ranking,
        "rrf_sensitivity": rrf_sensitivity,
        "error_decomposition": error_decomposition(
            full_ranked, union_candidates, union_sizes, gold
        ),
        "external_union_ablation": external_union_ablation,
        "artifacts": {
            "union": rel(union_dir),
            "rrf": rel(output_dir / "rrf_k60_tie_aware_ranked.npz"),
            "retriever_scatter": rel(output_dir / "retriever_recall_vs_size.pdf"),
        },
    }
    json_dump(output_dir / "report.json", report)
    write_csv(output_dir / "retriever_metrics.csv", source_metrics)
    write_csv(output_dir / "complementarity.csv", complementarity)
    write_csv(output_dir / "rrf_sensitivity.csv", rrf_sensitivity)
    write_csv(output_dir / "external_union_ablation.csv", [external_union_ablation])
    error = report["error_decomposition"]
    write_csv(
        output_dir / "error_breakdown.csv",
        [
            {
                "component": "retrieval miss",
                "mean_loss": error["retrieval_miss_loss"],
                "affected_rows": error["retrieval_miss_rows"],
            },
            {
                "component": "reranking exclusion",
                "mean_loss": error["reranking_miss_loss"],
                "affected_rows": error["reranking_miss_rows"],
            },
            {
                "component": "ordering at ranks 2-20",
                "mean_loss": error["ordering_loss"],
                "affected_rows": error["ordering_loss_rows"],
            },
        ],
    )
    write_csv(
        output_dir / "ranking_summary.csv",
        [{"ranker": name, **values["all"]} for name, values in ranking.items()],
    )
    write_text(paper_results, latex_macros(report))
    write_text(paper_retriever_rows, latex_retriever_rows(report))
    write_text(paper_family_rows, latex_family_rows(report))
    selector = r"""% Auto-generated final devset selector.
\newcommand{\LocalEvaluationName}{Devset}
\newcommand{\LocalEvaluationRows}{__N_ROWS__}
\newcommand{\LocalPreviewNotice}{}
\newcommand{\LocalScatterPath}{figures/retriever_recall_vs_size.pdf}
\newcommand{\RetrieverRowsFile}{generated_retriever_rows}
\newcommand{\FamilyRowsFile}{generated_family_rows}
\input{generated_results}
""".replace("__N_ROWS__", f"{len(gold):,}")
    write_text(paper_dir / "results_mode.tex", selector)
    summary = {
        "report": rel(output_dir / "report.json"),
        "paper_results": rel(paper_results),
        "paper_retriever_rows": rel(paper_retriever_rows),
        "paper_family_rows": rel(paper_family_rows),
    }
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
