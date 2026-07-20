#!/usr/bin/env python3
"""Shared artifact runner for challenge+TPD1 sequence retrievers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pyarrow.parquet as pq
import yaml
from datasets import load_dataset
from tqdm import tqdm

from recsys2026 import retriever_common as common
from recsys2026.artifacts import (
    artifact_complete,
    component_output_dir,
    component_results_dir,
    file_ref,
    json_dump,
    utc_now,
)
from recsys2026.paths import REPO_ROOT
from recsys2026.train_stat_runner import (
    PublicExample,
    build_cooc_from_sessions,
    build_inference_examples,
    build_public_examples,
    build_public_sessions,
    public_metrics,
    select_from_score,
    write_inference_artifact,
    write_public_artifact,
)

ScoreFunction = Callable[
    ..., tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]
]


@dataclass(frozen=True)
class TPD1Spec:
    name: str
    source_path: Path
    score_examples: ScoreFunction


def read_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def external_id_to_idx(mapping_path: Path) -> dict[str, int]:
    table = pq.read_table(mapping_path, columns=["external_track_id", "track_idx"])
    external_ids = table.column("external_track_id").to_pylist()
    idx = table.column("track_idx").to_pylist()
    return {
        str(source_id): int(i) for source_id, i in zip(external_ids, idx, strict=True)
    }


def load_tpd1_sessions(
    mapping: dict[str, int],
) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    ds = load_dataset("talkpl-ai/TalkPlayData-1", split="train")
    sessions: list[tuple[int, ...]] = []
    n_music = 0
    n_mapped_music = 0
    n_sessions_with_music = 0
    n_sessions_with_mapped_music = 0
    for item in tqdm(ds, desc="read TalkPlayData-1"):
        seq: list[int] = []
        session_music = 0
        for turn in item.get("conversations") or []:
            if turn.get("role") != "music":
                continue
            session_music += 1
            n_music += 1
            idx = mapping.get(str(turn.get("content") or ""))
            if idx is None:
                continue
            n_mapped_music += 1
            seq.append(int(idx))
        if session_music:
            n_sessions_with_music += 1
        if seq:
            n_sessions_with_mapped_music += 1
            sessions.append(tuple(seq))
    stats = {
        "tpd1_rows": len(ds),
        "music_turns": n_music,
        "mapped_music_turns": n_mapped_music,
        "unmapped_music_turns": n_music - n_mapped_music,
        "music_turn_mapping_rate": n_mapped_music / n_music if n_music else 0.0,
        "sessions_with_music": n_sessions_with_music,
        "sessions_with_mapped_music": n_sessions_with_mapped_music,
        "mapped_session_rate": n_sessions_with_mapped_music / n_sessions_with_music
        if n_sessions_with_music
        else 0.0,
    }
    return sessions, stats


def freeze_counts(
    table: dict[int, Counter[int]], *, min_count: int
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for key, counts in table.items():
        items = [(idx, cnt) for idx, cnt in counts.most_common() if cnt >= min_count]
        if not items:
            continue
        nb = np.fromiter((idx for idx, _ in items), dtype=np.int32, count=len(items))
        cn = np.fromiter((cnt for _, cnt in items), dtype=np.float32, count=len(items))
        out[int(key)] = (nb, cn)
    return out


def build_tpd1_tables(
    sessions: list[tuple[int, ...]],
    *,
    min_count: int,
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[str, Any],
]:
    cooc_counts: dict[int, Counter[int]] = defaultdict(Counter)
    transition_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for seq_raw in tqdm(sessions, desc="build TPD1 tables"):
        cooc_seq = list(dict.fromkeys(int(x) for x in seq_raw))
        for i, track_idx in enumerate(cooc_seq):
            counts = cooc_counts[track_idx]
            for neighbor_idx in cooc_seq[i + 1 :]:
                counts[neighbor_idx] += 1
                cooc_counts[neighbor_idx][track_idx] += 1
        seq = [int(x) for x in seq_raw]
        for prev_idx, next_idx in zip(seq, seq[1:], strict=False):
            transition_counts[int(prev_idx)][int(next_idx)] += 1

    cooc = freeze_counts(cooc_counts, min_count=min_count)
    transition = freeze_counts(transition_counts, min_count=min_count)
    stats = {
        "min_count": min_count,
        "cooc_anchor_tracks": len(cooc),
        "cooc_edges": int(sum(len(v[0]) for v in cooc.values())),
        "transition_anchor_tracks": len(transition),
        "transition_edges": int(sum(len(v[0]) for v in transition.values())),
    }
    return cooc, transition, stats


def add_table_score(
    score: np.ndarray, table: dict[int, tuple[np.ndarray, np.ndarray]], anchor: int
) -> None:
    nb_cn = table.get(int(anchor))
    if nb_cn is None:
        return
    nb, cn = nb_cn
    score[nb] += cn


def select_with_extras(
    score: np.ndarray,
    played: set[int],
    top_k: int,
    extras: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    idxs, values = select_from_score(score, played, top_k, positive_only=True)
    return (
        idxs,
        values,
        {
            name: np.asarray(extra, dtype=np.float32)[idxs]
            for name, extra in extras.items()
        },
    )


def pad_scored(
    rows: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    candidates = np.full((len(rows), top_k), -1, dtype=np.int32)
    sizes = np.zeros(len(rows), dtype=np.int32)
    scores = np.full((len(rows), top_k), np.nan, dtype=np.float32)
    extra_names = sorted({name for _, _, extras in rows for name in extras})
    extra_scores = {
        name: np.full((len(rows), top_k), np.nan, dtype=np.float32)
        for name in extra_names
    }
    for row_idx, (idxs, values, extras) in enumerate(rows):
        size = min(len(idxs), top_k)
        if not size:
            continue
        candidates[row_idx, :size] = idxs[:size]
        scores[row_idx, :size] = values[:size]
        sizes[row_idx] = size
        for name, extra in extras.items():
            extra_scores[name][row_idx, :size] = extra[:size]
    return candidates, sizes, scores, extra_scores


def source_policy_from_config(config: dict[str, Any]) -> dict[str, Any]:
    policy = dict(config.get("source_policy") or {})
    policy.setdefault("requires_labeled_fit", True)
    policy.setdefault("fit_sources", ["train_music_outcomes", "TalkPlayData-1"])
    policy.setdefault("train_row_policy", "cv5_oof_challenge_plus_external_fit_free")
    policy.setdefault("fold_split_required_for_reranker_train", True)
    policy.setdefault("preferred_train_row_artifact_mode", "cv5_oof")
    policy.setdefault("preferred_inference_artifact_mode", "full_public")
    return policy


def split_name(split_dir: Path) -> str:
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists():
        try:
            return str(
                json.loads(manifest_path.read_text()).get("name") or split_dir.name
            )
        except Exception:  # noqa: BLE001
            return split_dir.name
    return split_dir.name


def base_manifest(
    args: argparse.Namespace,
    spec: TPD1Spec,
    config: dict[str, Any],
    policy: dict[str, Any],
    mapping_path: Path,
    tpd1_stats: dict[str, Any],
    table_stats: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    fit_sources = (
        ["train_music_outcomes"]
        if args.disable_tpd1
        else ["train_music_outcomes", "TalkPlayData-1"]
    )
    source_code = {
        "runner": file_ref(REPO_ROOT / "src/recsys2026/tpd1_runner.py"),
        "common": file_ref(REPO_ROOT / "src/recsys2026/train_stat_runner.py"),
        "component": file_ref(spec.source_path),
        "config": file_ref(REPO_ROOT / args.config_file),
    }
    if not args.disable_tpd1:
        source_code["catalog_id_map"] = file_ref(mapping_path)
    return {
        "schema_version": 1,
        "artifact_type": "candidates",
        "stage": "retriever",
        "name": spec.name,
        "config": args.config,
        "created_at": utc_now(),
        "producer": {
            "command": [
                "uv",
                "run",
                "python",
                "-m",
                f"retriever.{spec.name}.main",
                *sys.argv[1:],
            ],
            "cwd": ".",
        },
        "source_code": source_code,
        "params": {
            "config": args.config,
            "top_k": args.top_k,
            "min_count": args.min_count,
            "combine_rule": "challenge_count_only"
            if args.disable_tpd1
            else "challenge_count_plus_tpd1_count",
        },
        "source_policy": {**policy, "fit_sources": fit_sources},
        "external_data": None
        if args.disable_tpd1
        else {
            "name": "talkpl-ai/TalkPlayData-1",
            "split": "train",
            "mapping": str(mapping_path.relative_to(REPO_ROOT)),
            "tpd1_stats": tpd1_stats,
            "table_stats": table_stats,
            "unmapped_tracks_filtered": True,
        },
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_blind_for_fit": False,
            "uses_target_future_turns": False,
            "uses_challenge_labels_for_fit": True,
            "challenge_train_rows_oof_for_public_labeled": True,
            "external_fit_source": None
            if args.disable_tpd1
            else "TalkPlayData-1 train",
            "external_unknown_tracks_filtered": not args.disable_tpd1,
            "track_only": True,
            "current_thought_allowed": False,
            "conversation_goal_allowed": False,
            "gpa_allowed": False,
            "popularity_tiebreaker": False,
        },
        "candidate_universe": "challenge_all_tracks",
        "retention": "top_k",
        "elapsed_sec": elapsed,
    }


def run_public(
    common_module: Any,
    track_index: Any,
    public_examples: list[PublicExample],
    sessions: list[Any],
    spec: TPD1Spec,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    tpd1_cooc: dict[int, tuple[np.ndarray, np.ndarray]],
    tpd1_transition: dict[int, tuple[np.ndarray, np.ndarray]],
    tpd1_stats: dict[str, Any],
    table_stats: dict[str, Any],
) -> None:
    source = spec.name
    artifact_modes = cfg.get("artifact_modes") or {}
    fit_mode = str(
        args.artifact_mode or artifact_modes.get("public_labeled") or "cv5_oof"
    )
    out_dir = component_output_dir(
        "retriever", source, args.config, fit_mode=fit_mode, target="public_labeled"
    )
    if artifact_complete(out_dir, "candidates.npz", "turns.jsonl"):
        print(f"[skip] {out_dir}")
        return

    t0 = time.time()
    width = args.top_k
    cand = np.full((len(public_examples), width), -1, dtype=np.int32)
    sizes = np.zeros(len(public_examples), dtype=np.int32)
    scores = np.full((len(public_examples), width), np.nan, dtype=np.float32)
    extra_scores: dict[str, np.ndarray] = {}
    folds = np.asarray([ex.fold for ex in public_examples], dtype=np.int16)
    for fold in sorted(int(x) for x in np.unique(folds)):
        valid_rows = np.flatnonzero(folds == fold)
        fold_examples = [public_examples[int(i)] for i in valid_rows]
        fit_sessions = [s for s in sessions if s.fold != fold]
        print(
            f"{source}: fold {fold}, rows={len(valid_rows)}, fit_sessions={len(fit_sessions)}"
        )
        challenge_cooc = build_cooc_from_sessions(
            common_module, track_index, fit_sessions
        )
        sub_cand, sub_sizes, sub_scores, sub_extra = spec.score_examples(
            common_module,
            fold_examples,
            track_index,
            challenge_cooc,
            tpd1_cooc,
            tpd1_transition,
            args.top_k,
        )
        cand[valid_rows] = sub_cand
        sizes[valid_rows] = sub_sizes
        scores[valid_rows] = sub_scores
        for key, arr in sub_extra.items():
            if key not in extra_scores:
                extra_scores[key] = np.full(
                    (len(public_examples), width), np.nan, dtype=np.float32
                )
            extra_scores[key][valid_rows] = arr

    elapsed = time.time() - t0
    policy = source_policy_from_config(cfg)
    manifest = base_manifest(
        args, spec, cfg, policy, args.mapping, tpd1_stats, table_stats, elapsed
    )
    manifest.update({"artifact_mode": fit_mode, "target": "public_labeled"})
    manifest["score_fields"] = ["score__primary"] + [
        f"score__{key}" for key in sorted(extra_scores)
    ]
    fit_splits = sorted({ex.source_split for ex in public_examples})
    external_fit_splits = [] if args.disable_tpd1 else ["TalkPlayData-1 train"]
    manifest["fit_scope"] = {
        "fit_mode": fit_mode,
        "fit_splits": [*fit_splits, *external_fit_splits],
        "requires_labeled_fit": True,
        "fit_sources": ["train_music_outcomes"]
        if args.disable_tpd1
        else list(policy.get("fit_sources") or []),
        "train_row_policy": (
            f"out_of_fold_by_{split_name(args.split_dir)}_for_challenge_counts_only"
            if args.disable_tpd1
            else f"out_of_fold_by_{split_name(args.split_dir)}_for_challenge_counts_plus_full_external_tpd1"
        ),
        "fold_split_required_for_reranker_train": True,
        "uses_devset_for_fit": "devset" in fit_splits,
        "uses_blind_for_fit": False,
        "target_row_excluded_from_fit": True,
    }
    manifest["source_policy"].update(
        {
            "fit_sources": manifest["fit_scope"]["fit_sources"],
            "train_row_policy": manifest["fit_scope"]["train_row_policy"],
            "preferred_train_row_artifact_mode": fit_mode,
            "preferred_inference_artifact_mode": "full_train"
            if fit_splits == ["train"]
            else "full_public",
        }
    )
    write_public_artifact(
        out_dir, public_examples, cand, sizes, scores, extra_scores, manifest
    )
    metrics = public_metrics(public_examples, cand, sizes)
    metrics.update(
        {
            "name": source,
            "config": args.config,
            "artifact_mode": fit_mode,
            "target": "public_labeled",
            "artifact": str(out_dir.relative_to(REPO_ROOT)),
        }
    )
    json_dump(
        component_results_dir(
            "retriever", source, args.config, fit_mode=fit_mode, target="public_labeled"
        )
        / "scores.json",
        metrics,
    )
    print(json.dumps(metrics, indent=2))


def run_inference(
    common_module: Any,
    track_index: Any,
    public_examples: list[PublicExample],
    inference_examples: list[PublicExample],
    sessions: list[Any],
    spec: TPD1Spec,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    tpd1_cooc: dict[int, tuple[np.ndarray, np.ndarray]],
    tpd1_transition: dict[int, tuple[np.ndarray, np.ndarray]],
    tpd1_stats: dict[str, Any],
    table_stats: dict[str, Any],
) -> None:
    source = spec.name
    fit_splits = sorted({ex.source_split for ex in public_examples})
    fit_mode = "full_train" if fit_splits == ["train"] else "full_public"
    out_dir = component_output_dir(
        "retriever", source, args.config, fit_mode=fit_mode, target=args.target
    )
    if artifact_complete(out_dir, "candidates.npz", "turns.jsonl"):
        print(f"[skip] {out_dir}")
        return

    t0 = time.time()
    challenge_cooc = build_cooc_from_sessions(common_module, track_index, sessions)
    cand, sizes, scores, extra_scores = spec.score_examples(
        common_module,
        inference_examples,
        track_index,
        challenge_cooc,
        tpd1_cooc,
        tpd1_transition,
        args.top_k,
    )
    elapsed = time.time() - t0
    policy = source_policy_from_config(cfg)
    manifest = base_manifest(
        args, spec, cfg, policy, args.mapping, tpd1_stats, table_stats, elapsed
    )
    manifest.update({"artifact_mode": fit_mode, "target": args.target})
    manifest["score_fields"] = ["score__primary"] + [
        f"score__{key}" for key in sorted(extra_scores)
    ]
    external_fit_splits = [] if args.disable_tpd1 else ["TalkPlayData-1 train"]
    manifest["fit_scope"] = {
        "fit_mode": fit_mode,
        "fit_splits": [*fit_splits, *external_fit_splits],
        "requires_labeled_fit": True,
        "fit_sources": ["train_music_outcomes"]
        if args.disable_tpd1
        else list(policy.get("fit_sources") or []),
        "train_row_policy": (
            f"inference_only_{fit_mode}_challenge_counts_only"
            if args.disable_tpd1
            else f"inference_only_{fit_mode}_challenge_counts_plus_full_external_tpd1"
        ),
        "fold_split_required_for_reranker_train": False,
        "uses_devset_for_fit": "devset" in fit_splits,
        "uses_blind_for_fit": False,
        "target_row_excluded_from_fit": None,
    }
    manifest["source_policy"].update(
        {
            "fit_sources": manifest["fit_scope"]["fit_sources"],
            "train_row_policy": manifest["fit_scope"]["train_row_policy"],
            "preferred_train_row_artifact_mode": (
                str(args.artifact_mode or "train5_oof")
                if fit_splits == ["train"]
                else str(
                    (cfg.get("artifact_modes") or {}).get("public_labeled") or "cv5_oof"
                )
            ),
            "preferred_inference_artifact_mode": fit_mode,
        }
    )
    write_inference_artifact(
        out_dir, args.target, cand, sizes, scores, extra_scores, manifest
    )
    if args.target == "devset":
        metrics = public_metrics(inference_examples, cand, sizes)
        metrics.update(
            {
                "name": source,
                "config": args.config,
                "artifact_mode": fit_mode,
                "target": args.target,
                "artifact": str(out_dir.relative_to(REPO_ROOT)),
            }
        )
        json_dump(
            component_results_dir(
                "retriever",
                source,
                args.config,
                fit_mode=fit_mode,
                target=args.target,
            )
            / "scores.json",
            metrics,
        )
        print(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir} mean_size={sizes.mean():.1f} elapsed={elapsed:.1f}s")


def main(spec: TPD1Spec) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="oof5_top500_parts")
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=("public_labeled", "devset", "blind_b"),
        default="public_labeled",
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/preprocessed/splits/cv5",
    )
    parser.add_argument(
        "--artifact-mode",
        default=None,
        help="Public OOF artifact mode, normally cv5_oof.",
    )
    parser.add_argument(
        "--mapping",
        type=Path,
        default=REPO_ROOT / "artifacts/preprocessed/catalog_id_map.parquet",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=None)
    parser.add_argument(
        "--disable-tpd1",
        action="store_true",
        help="Use challenge counts only while retaining the same source schema.",
    )
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    args.config_file = (
        args.config_file
        if args.config_file.is_absolute()
        else REPO_ROOT / args.config_file
    )
    args.split_dir = (
        args.split_dir if args.split_dir.is_absolute() else REPO_ROOT / args.split_dir
    )
    args.mapping = (
        args.mapping if args.mapping.is_absolute() else REPO_ROOT / args.mapping
    )
    cfg = read_config(args.config_file)
    configured_source = cfg.get("source")
    if configured_source and configured_source != spec.name:
        raise ValueError(
            f"config source {configured_source!r} does not match {spec.name!r}"
        )
    args.top_k = int(args.top_k if args.top_k is not None else cfg.get("top_k", 500))
    args.min_count = int(
        args.min_count if args.min_count is not None else cfg.get("min_count", 1)
    )
    track_index = common.build_track_index(())
    public_examples = build_public_examples(common, args.split_dir, track_index)
    sessions = build_public_sessions(args.split_dir, track_index)

    if args.disable_tpd1:
        tpd1_sessions = []
        tpd1_stats = {"enabled": False, "reason": "paper_without_tpd1_ablation"}
        tpd1_cooc, tpd1_transition = {}, {}
        table_stats = {"enabled": False}
    else:
        mapping = external_id_to_idx(args.mapping)
        tpd1_sessions, tpd1_stats = load_tpd1_sessions(mapping)
        tpd1_cooc, tpd1_transition, table_stats = build_tpd1_tables(
            tpd1_sessions,
            min_count=args.min_count,
        )

    if args.target == "public_labeled":
        run_public(
            common,
            track_index,
            public_examples,
            sessions,
            spec,
            args,
            cfg,
            tpd1_cooc,
            tpd1_transition,
            tpd1_stats,
            table_stats,
        )
    else:
        inference_examples = build_inference_examples(common, args.target, track_index)
        run_inference(
            common,
            track_index,
            public_examples,
            inference_examples,
            sessions,
            spec,
            args,
            cfg,
            tpd1_cooc,
            tpd1_transition,
            tpd1_stats,
            table_stats,
        )
