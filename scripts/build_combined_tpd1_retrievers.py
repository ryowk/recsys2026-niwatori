#!/usr/bin/env python3
"""Build challenge+TalkPlayData-1 combined cooc/transition retrievers."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import yaml
from datasets import load_dataset
from tqdm import tqdm

from recsys2026.artifacts import component_output_dir, component_results_dir, file_ref, json_dump, utc_now
from recsys2026.paths import REPO_ROOT

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_train_fit_retriever_artifacts import (
    PublicExample,
    build_blind_examples,
    build_cooc_from_sessions,
    build_public_examples,
    build_public_sessions,
    history_state,
    load_zoo_module,
    pad_scored_with_extras,
    played_set,
    public_metrics,
    select_from_score_with_extras,
    write_blind_artifact,
    write_public_artifact,
)

SourceName = Literal["cooc_track_combined_tpd1", "transition_track_combined_tpd1"]


def read_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def spotify_to_idx(mapping_path: Path) -> dict[str, int]:
    table = pq.read_table(mapping_path, columns=["spotify_id", "track_idx"])
    spotify = table.column("spotify_id").to_pylist()
    idx = table.column("track_idx").to_pylist()
    return {str(s): int(i) for s, i in zip(spotify, idx, strict=True)}


def load_tpd1_sessions(mapping: dict[str, int]) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
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
        "mapped_session_rate": n_sessions_with_mapped_music / n_sessions_with_music if n_sessions_with_music else 0.0,
    }
    return sessions, stats


def freeze_counts(table: dict[int, Counter[int]], *, min_count: int) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for key, counts in table.items():
        items = [(idx, cnt) for idx, cnt in counts.most_common() if cnt >= min_count]
        if not items:
            continue
        nb = np.fromiter((idx for idx, _ in items), dtype=np.int32, count=len(items))
        cn = np.fromiter((cnt for _, cnt in items), dtype=np.float32, count=len(items))
        out[int(key)] = (nb, cn)
    return out


def freeze_float_table(table: dict[int, dict[int, float]]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for key, scores in table.items():
        items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if not items:
            continue
        nb = np.fromiter((idx for idx, _ in items), dtype=np.int32, count=len(items))
        vals = np.fromiter((score for _, score in items), dtype=np.float32, count=len(items))
        out[int(key)] = (nb, vals)
    return out


def cooc_edge_weight(distance: int, mode: str) -> float:
    if mode == "inverse":
        return 1.0 / float(max(1, distance))
    if mode == "linear":
        return float(max(1, distance))
    if mode == "none":
        return 1.0
    raise ValueError(f"unknown TPD1 cooc distance weight: {mode}")


def add_session_cooc_edges(
    cooc_counts: dict[int, Counter[int]],
    seq_raw: tuple[int, ...],
    *,
    mode: str,
    window: int | None,
    distance_weight: str,
) -> None:
    if mode == "session_all_pairs":
        cooc_seq = list(dict.fromkeys(int(x) for x in seq_raw))
        for i, a in enumerate(cooc_seq):
            ca = cooc_counts[a]
            for b in cooc_seq[i + 1 :]:
                ca[b] += 1.0
                cooc_counts[b][a] += 1.0
        return
    if mode != "mapped_window":
        raise ValueError(f"unknown TPD1 cooc mode: {mode}")
    max_distance = int(window or 0)
    if max_distance <= 0:
        raise ValueError("tpd1_cooc_window must be positive when tpd1_cooc_mode=mapped_window")
    seq = [int(x) for x in seq_raw]
    for i, a in enumerate(seq):
        ca = cooc_counts[a]
        stop = min(len(seq), i + max_distance + 1)
        for j in range(i + 1, stop):
            b = int(seq[j])
            if a == b:
                continue
            w = cooc_edge_weight(j - i, distance_weight)
            ca[b] += w
            cooc_counts[b][a] += w


def build_tpd1_tables(
    sessions: list[tuple[int, ...]],
    *,
    min_count: int,
    cooc_mode: str = "session_all_pairs",
    cooc_window: int | None = None,
    cooc_distance_weight: str = "none",
) -> tuple[
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[int, tuple[np.ndarray, np.ndarray]],
    dict[str, Any],
]:
    cooc_counts: dict[int, Counter[int]] = defaultdict(Counter)
    transition_counts: dict[int, Counter[int]] = defaultdict(Counter)
    session_freq: Counter[int] = Counter()
    for seq_raw in tqdm(sessions, desc="build TPD1 tables"):
        cooc_seq = list(dict.fromkeys(int(x) for x in seq_raw))
        session_freq.update(cooc_seq)
        add_session_cooc_edges(
            cooc_counts,
            seq_raw,
            mode=cooc_mode,
            window=cooc_window,
            distance_weight=cooc_distance_weight,
        )
        seq = [int(x) for x in seq_raw]
        for prev_idx, next_idx in zip(seq, seq[1:]):
            transition_counts[int(prev_idx)][int(next_idx)] += 1

    cooc = freeze_counts(cooc_counts, min_count=min_count)
    transition = freeze_counts(transition_counts, min_count=min_count)
    n_sessions = max(1, len(sessions))
    pmi_scores: dict[int, dict[int, float]] = defaultdict(dict)
    for a, counts in cooc_counts.items():
        fa = float(session_freq[int(a)])
        if fa <= 0:
            continue
        for b, cnt in counts.items():
            if cnt < min_count:
                continue
            fb = float(session_freq[int(b)])
            if fb <= 0:
                continue
            pmi = np.log((float(cnt) * float(n_sessions)) / (fa * fb))
            pmi_scores[int(a)][int(b)] = float(max(0.0, pmi))
    cooc_pmi = freeze_float_table(pmi_scores)
    stats = {
        "min_count": min_count,
        "cooc_mode": cooc_mode,
        "cooc_window": cooc_window,
        "cooc_distance_weight": cooc_distance_weight,
        "cooc_anchor_tracks": len(cooc),
        "cooc_edges": int(sum(len(v[0]) for v in cooc.values())),
        "cooc_pmi_anchor_tracks": len(cooc_pmi),
        "cooc_pmi_edges": int(sum(len(v[0]) for v in cooc_pmi.values())),
        "transition_anchor_tracks": len(transition),
        "transition_edges": int(sum(len(v[0]) for v in transition.values())),
    }
    return cooc, transition, cooc_pmi, stats


def session_ngrams(seq: tuple[int, ...], *, min_n: int) -> set[tuple[int, ...]]:
    grams: set[tuple[int, ...]] = set()
    n = len(seq)
    for size in range(min_n, n + 1):
        for start in range(n - size + 1):
            grams.add(seq[start : start + size])
    return grams


def public_ngrams_by_fold(sessions: list[Any], *, min_n: int) -> tuple[dict[int, set[tuple[int, ...]]], set[tuple[int, ...]]]:
    by_fold: dict[int, set[tuple[int, ...]]] = defaultdict(set)
    all_grams: set[tuple[int, ...]] = set()
    for session in sessions:
        seq = tuple(int(x) for x in session.track_idxs)
        if len(seq) < min_n:
            continue
        grams = session_ngrams(seq, min_n=min_n)
        by_fold[int(session.fold)].update(grams)
        all_grams.update(grams)
    return dict(by_fold), all_grams


def overlaps_public_ngrams(seq: tuple[int, ...], public_grams: set[tuple[int, ...]], *, min_n: int) -> bool:
    n = len(seq)
    if n < min_n or not public_grams:
        return False
    for size in range(min_n, n + 1):
        for start in range(n - size + 1):
            if seq[start : start + size] in public_grams:
                return True
    return False


def filter_tpd1_sessions(
    sessions: list[tuple[int, ...]],
    public_grams: set[tuple[int, ...]],
    *,
    min_n: int,
) -> tuple[list[tuple[int, ...]], dict[str, Any]]:
    kept: list[tuple[int, ...]] = []
    removed = 0
    for seq in sessions:
        if overlaps_public_ngrams(seq, public_grams, min_n=min_n):
            removed += 1
            continue
        kept.append(seq)
    total = len(sessions)
    return kept, {
        "purge_ngram_min": min_n,
        "input_sessions": total,
        "removed_sessions": removed,
        "kept_sessions": len(kept),
        "removed_rate": removed / total if total else 0.0,
    }


def add_table_score(score: np.ndarray, table: dict[int, tuple[np.ndarray, np.ndarray]], anchor: int) -> None:
    nb_cn = table.get(int(anchor))
    if nb_cn is None:
        return
    nb, cn = nb_cn
    score[nb] += cn


def score_examples(
    zoo: Any,
    examples: list[PublicExample],
    track_index: Any,
    source: SourceName,
    challenge_cooc: Any,
    tpd1_cooc: dict[int, tuple[np.ndarray, np.ndarray]],
    tpd1_transition: dict[int, tuple[np.ndarray, np.ndarray]],
    tpd1_cooc_pmi: dict[int, tuple[np.ndarray, np.ndarray]],
    *,
    top_k: int,
    emit_component_scores: bool,
    emit_pmi_scores: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    rows: list[tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]] = []
    for ex in tqdm(examples, desc=source):
        _, _, _, played, history_idxs = history_state(zoo, ex, track_index)
        challenge_score = np.zeros(track_index.n_tracks, dtype=np.float32)
        tpd1_score = np.zeros(track_index.n_tracks, dtype=np.float32)
        tpd1_pmi_score = np.zeros(track_index.n_tracks, dtype=np.float32)
        transition_prob = np.zeros(track_index.n_tracks, dtype=np.float32)

        if source == "cooc_track_combined_tpd1":
            for h in history_idxs:
                add_table_score(challenge_score, challenge_cooc.track_track, int(h))
                add_table_score(tpd1_score, tpd1_cooc, int(h))
                if emit_pmi_scores:
                    add_table_score(tpd1_pmi_score, tpd1_cooc_pmi, int(h))
        elif source == "transition_track_combined_tpd1":
            if history_idxs:
                last = int(history_idxs[-1])
                add_table_score(challenge_score, challenge_cooc.transition_track, last)
                add_table_score(tpd1_score, tpd1_transition, last)
        else:
            raise ValueError(source)

        score = challenge_score + tpd1_score
        extras: dict[str, np.ndarray] = {}
        if source == "transition_track_combined_tpd1":
            denom = float(score.sum())
            if denom > 0:
                transition_prob = score / denom
            extras["transition_probability"] = transition_prob
        if emit_component_scores:
            extras["challenge"] = challenge_score
            extras["tpd1"] = tpd1_score
        if emit_pmi_scores and source == "cooc_track_combined_tpd1":
            extras["tpd1_pmi"] = tpd1_pmi_score
        rows.append(
            select_from_score_with_extras(
                score,
                played,
                top_k,
                positive_only=True,
                extras=extras,
            )
        )
    return pad_scored_with_extras(rows, top_k)


def source_policy_from_config(config: dict[str, Any]) -> dict[str, Any]:
    policy = dict(config.get("source_policy") or {})
    policy.setdefault("requires_labeled_fit", True)
    policy.setdefault("fit_sources", ["train_music_outcomes", "TalkPlayData-1"])
    policy.setdefault("train_row_policy", "cv3_oof_challenge_plus_external_fit_free")
    policy.setdefault("fold_split_required_for_reranker_train", True)
    policy.setdefault("preferred_train_row_artifact_mode", "cv3_oof")
    policy.setdefault("preferred_inference_artifact_mode", "full_public")
    return policy


def split_name(split_dir: Path) -> str:
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists():
        try:
            return str(json.loads(manifest_path.read_text()).get("name") or split_dir.name)
        except Exception:  # noqa: BLE001
            return split_dir.name
    return split_dir.name


def base_manifest(
    args: argparse.Namespace,
    source: SourceName,
    config: dict[str, Any],
    policy: dict[str, Any],
    mapping_path: Path,
    tpd1_stats: dict[str, Any],
    table_stats: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": "candidates",
        "stage": "retriever",
        "name": source,
        "config": args.config,
        "created_at": utc_now(),
        "producer": {
            "command": ["uv", "run", "python", "scripts/build_combined_tpd1_retrievers.py", *sys.argv[1:]],
            "cwd": ".",
        },
        "source_code": {
            "script": file_ref(REPO_ROOT / "scripts/build_combined_tpd1_retrievers.py"),
            "config": file_ref(REPO_ROOT / args.config_file),
            "spotify_uuid_map": file_ref(mapping_path),
        },
        "params": {
            "config": args.config,
            "top_k": args.top_k,
            "min_count": args.min_count,
            "emit_component_scores": args.emit_component_scores,
            "emit_pmi_scores": args.emit_pmi_scores,
            "tpd1_cooc_mode": args.tpd1_cooc_mode,
            "tpd1_cooc_window": args.tpd1_cooc_window,
            "tpd1_cooc_distance_weight": args.tpd1_cooc_distance_weight,
            "combine_rule": "challenge_count_plus_tpd1_count",
        },
        "source_policy": policy,
        "external_data": {
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
            "external_fit_source": "TalkPlayData-1 train",
            "external_unknown_tracks_filtered": True,
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
    zoo: Any,
    track_index: Any,
    public_examples: list[PublicExample],
    sessions: list[Any],
    source: SourceName,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    tpd1_sessions: list[tuple[int, ...]],
    tpd1_cooc: dict[int, tuple[np.ndarray, np.ndarray]],
    tpd1_transition: dict[int, tuple[np.ndarray, np.ndarray]],
    tpd1_cooc_pmi: dict[int, tuple[np.ndarray, np.ndarray]],
    tpd1_stats: dict[str, Any],
    table_stats: dict[str, Any],
) -> None:
    artifact_modes = cfg.get("artifact_modes") or {}
    fit_mode = str(args.artifact_mode or artifact_modes.get("public_labeled") or "cv3_oof")
    out_dir = component_output_dir("retriever", source, args.config, fit_mode=fit_mode, target="public_labeled")
    if (out_dir / "candidates.npz").exists() and not args.force:
        print(f"[skip] {out_dir}")
        return

    t0 = time.time()
    width = args.top_k
    cand = np.full((len(public_examples), width), -1, dtype=np.int32)
    sizes = np.zeros(len(public_examples), dtype=np.int32)
    scores = np.full((len(public_examples), width), np.nan, dtype=np.float32)
    extra_scores: dict[str, np.ndarray] = {}
    folds = np.asarray([ex.fold for ex in public_examples], dtype=np.int16)
    purge_min = int(args.external_purge_ngram_min) if args.external_purge_ngram_min else None
    purge_stats: dict[str, Any] | None = None
    purged_table_stats: dict[str, Any] | None = None
    shared_tpd1_cooc = tpd1_cooc
    shared_tpd1_transition = tpd1_transition
    shared_tpd1_cooc_pmi = tpd1_cooc_pmi
    if purge_min is not None:
        _, all_public_grams = public_ngrams_by_fold(sessions, min_n=purge_min)
        purged_sessions, purge_stats = filter_tpd1_sessions(tpd1_sessions, all_public_grams, min_n=purge_min)
        print(
            f"{source}: shared TPD1 purge removed={purge_stats['removed_sessions']} "
            f"kept={purge_stats['kept_sessions']} min_ngram={purge_min}"
        )
        shared_tpd1_cooc, shared_tpd1_transition, shared_tpd1_cooc_pmi, purged_table_stats = build_tpd1_tables(
            purged_sessions,
            min_count=args.min_count,
            cooc_mode=args.tpd1_cooc_mode,
            cooc_window=args.tpd1_cooc_window,
            cooc_distance_weight=args.tpd1_cooc_distance_weight,
        )
    for fold in sorted(int(x) for x in np.unique(folds)):
        valid_rows = np.flatnonzero(folds == fold)
        fold_examples = [public_examples[int(i)] for i in valid_rows]
        fit_sessions = [s for s in sessions if s.fold != fold]
        print(f"{source}: fold {fold}, rows={len(valid_rows)}, fit_sessions={len(fit_sessions)}")
        challenge_cooc = build_cooc_from_sessions(zoo, track_index, fit_sessions)
        sub_cand, sub_sizes, sub_scores, sub_extra = score_examples(
            zoo,
            fold_examples,
            track_index,
            source,
            challenge_cooc,
            shared_tpd1_cooc,
            shared_tpd1_transition,
            shared_tpd1_cooc_pmi,
            top_k=args.top_k,
            emit_component_scores=args.emit_component_scores,
            emit_pmi_scores=args.emit_pmi_scores,
        )
        cand[valid_rows] = sub_cand
        sizes[valid_rows] = sub_sizes
        scores[valid_rows] = sub_scores
        for key, arr in sub_extra.items():
            if key not in extra_scores:
                extra_scores[key] = np.full((len(public_examples), width), np.nan, dtype=np.float32)
            extra_scores[key][valid_rows] = arr

    elapsed = time.time() - t0
    policy = source_policy_from_config(cfg)
    manifest = base_manifest(args, source, cfg, policy, args.mapping, tpd1_stats, table_stats, elapsed)
    if purge_min is not None:
        manifest["external_data"]["purge"] = {
            "enabled": True,
            "scope": "all_public_labeled_shared_across_folds",
            "ngram_min": purge_min,
            "stats": purge_stats,
            "table_stats": purged_table_stats,
        }
        manifest["leak_check"]["external_purged_against_public_labeled"] = True
    manifest.update({"artifact_mode": fit_mode, "target": "public_labeled"})
    manifest["score_fields"] = ["score__primary"] + [f"score__{key}" for key in sorted(extra_scores)]
    manifest["fit_scope"] = {
        "fit_mode": fit_mode,
        "fit_splits": ["public_labeled", "TalkPlayData-1 train"],
        "requires_labeled_fit": True,
        "fit_sources": list(policy.get("fit_sources") or []),
        "train_row_policy": (
            f"out_of_fold_by_{split_name(args.split_dir)}_for_challenge_counts_plus_shared_public_purged_external_tpd1"
            if purge_min is not None
            else f"out_of_fold_by_{split_name(args.split_dir)}_for_challenge_counts_plus_full_external_tpd1"
        ),
        "fold_split_required_for_reranker_train": True,
        "uses_devset_for_fit": True,
        "uses_blind_for_fit": False,
        "target_row_excluded_from_fit": True,
    }
    write_public_artifact(out_dir, public_examples, cand, sizes, scores, extra_scores, manifest)
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
    json_dump(component_results_dir("retriever", source, args.config, fit_mode=fit_mode, target="public_labeled") / "scores.json", metrics)
    print(json.dumps(metrics, indent=2))


def run_blind(
    zoo: Any,
    track_index: Any,
    public_examples: list[PublicExample],
    blind_examples: list[PublicExample],
    sessions: list[Any],
    source: SourceName,
    args: argparse.Namespace,
    cfg: dict[str, Any],
    tpd1_sessions: list[tuple[int, ...]],
    tpd1_cooc: dict[int, tuple[np.ndarray, np.ndarray]],
    tpd1_transition: dict[int, tuple[np.ndarray, np.ndarray]],
    tpd1_cooc_pmi: dict[int, tuple[np.ndarray, np.ndarray]],
    tpd1_stats: dict[str, Any],
    table_stats: dict[str, Any],
) -> None:
    fit_mode = "full_public"
    out_dir = component_output_dir("retriever", source, args.config, fit_mode=fit_mode, target=args.blind_target)
    if (out_dir / "candidates.npz").exists() and not args.force:
        print(f"[skip] {out_dir}")
        return

    t0 = time.time()
    challenge_cooc = build_cooc_from_sessions(zoo, track_index, sessions)
    purge_min = int(args.external_purge_ngram_min) if args.external_purge_ngram_min else None
    blind_tpd1_cooc = tpd1_cooc
    blind_tpd1_transition = tpd1_transition
    blind_tpd1_cooc_pmi = tpd1_cooc_pmi
    purge_stats: dict[str, Any] | None = None
    purged_table_stats: dict[str, Any] | None = None
    if purge_min is not None:
        _, all_public_grams = public_ngrams_by_fold(sessions, min_n=purge_min)
        purged_sessions, purge_stats = filter_tpd1_sessions(tpd1_sessions, all_public_grams, min_n=purge_min)
        print(
            f"{source}: blind {args.blind_target}, TPD1 purge removed={purge_stats['removed_sessions']} "
            f"kept={purge_stats['kept_sessions']} min_ngram={purge_min}"
        )
        blind_tpd1_cooc, blind_tpd1_transition, blind_tpd1_cooc_pmi, purged_table_stats = build_tpd1_tables(
            purged_sessions,
            min_count=args.min_count,
            cooc_mode=args.tpd1_cooc_mode,
            cooc_window=args.tpd1_cooc_window,
            cooc_distance_weight=args.tpd1_cooc_distance_weight,
        )
    cand, sizes, scores, extra_scores = score_examples(
        zoo,
        blind_examples,
        track_index,
        source,
        challenge_cooc,
        blind_tpd1_cooc,
        blind_tpd1_transition,
        blind_tpd1_cooc_pmi,
        top_k=args.top_k,
        emit_component_scores=args.emit_component_scores,
        emit_pmi_scores=args.emit_pmi_scores,
    )
    elapsed = time.time() - t0
    policy = source_policy_from_config(cfg)
    manifest = base_manifest(args, source, cfg, policy, args.mapping, tpd1_stats, table_stats, elapsed)
    if purge_min is not None:
        manifest["external_data"]["purge"] = {
            "enabled": True,
            "scope": "all_public_labeled",
            "ngram_min": purge_min,
            "stats": purge_stats,
            "table_stats": purged_table_stats,
        }
        manifest["leak_check"]["external_purged_against_public_labeled"] = True
    manifest.update({"artifact_mode": fit_mode, "target": args.blind_target})
    manifest["score_fields"] = ["score__primary"] + [f"score__{key}" for key in sorted(extra_scores)]
    manifest["fit_scope"] = {
        "fit_mode": fit_mode,
        "fit_splits": ["public_labeled", "TalkPlayData-1 train"],
        "requires_labeled_fit": True,
        "fit_sources": list(policy.get("fit_sources") or []),
        "train_row_policy": (
            "inference_only_full_public_challenge_counts_plus_public_purged_external_tpd1"
            if purge_min is not None
            else "inference_only_full_public_challenge_counts_plus_full_external_tpd1"
        ),
        "fold_split_required_for_reranker_train": False,
        "uses_devset_for_fit": True,
        "uses_blind_for_fit": False,
        "target_row_excluded_from_fit": None,
    }
    write_blind_artifact(out_dir, args.blind_target, cand, sizes, scores, extra_scores, manifest)
    print(f"wrote {out_dir} mean_size={sizes.mean():.1f} elapsed={elapsed:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("cooc_track_combined_tpd1", "transition_track_combined_tpd1"), required=True)
    parser.add_argument("--config", default="oof3_top500")
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--target", choices=("public_labeled", "blind_a", "blind_b"), default="public_labeled")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "artifacts/cache/splits/cv5")
    parser.add_argument("--artifact-mode", default=None, help="Public OOF artifact mode, e.g. cv3_oof or cv5_oof.")
    parser.add_argument("--mapping", type=Path, default=REPO_ROOT / "artifacts/cache/spotify_uuid_map.parquet")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-count", type=int, default=None)
    parser.add_argument("--emit-component-scores", action="store_true")
    parser.add_argument("--emit-pmi-scores", action="store_true")
    parser.add_argument("--external-purge-ngram-min", type=int, default=None)
    parser.add_argument("--tpd1-cooc-mode", default=None)
    parser.add_argument("--tpd1-cooc-window", type=int, default=None)
    parser.add_argument("--tpd1-cooc-distance-weight", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    if args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args.config_file = args.config_file if args.config_file.is_absolute() else REPO_ROOT / args.config_file
    args.split_dir = args.split_dir if args.split_dir.is_absolute() else REPO_ROOT / args.split_dir
    args.mapping = args.mapping if args.mapping.is_absolute() else REPO_ROOT / args.mapping
    cfg = read_config(args.config_file)
    args.top_k = int(args.top_k if args.top_k is not None else cfg.get("top_k", 500))
    args.min_count = int(args.min_count if args.min_count is not None else cfg.get("min_count", 1))
    args.external_purge_ngram_min = (
        int(args.external_purge_ngram_min)
        if args.external_purge_ngram_min is not None
        else (int(cfg["external_purge_ngram_min"]) if cfg.get("external_purge_ngram_min") is not None else None)
    )
    if bool(cfg.get("emit_component_scores", False)):
        args.emit_component_scores = True
    if bool(cfg.get("emit_pmi_scores", False)):
        args.emit_pmi_scores = True
    args.tpd1_cooc_mode = str(args.tpd1_cooc_mode or cfg.get("tpd1_cooc_mode") or "session_all_pairs")
    args.tpd1_cooc_window = (
        int(args.tpd1_cooc_window)
        if args.tpd1_cooc_window is not None
        else (int(cfg["tpd1_cooc_window"]) if cfg.get("tpd1_cooc_window") is not None else None)
    )
    args.tpd1_cooc_distance_weight = str(
        args.tpd1_cooc_distance_weight or cfg.get("tpd1_cooc_distance_weight") or "none"
    )
    args.blind_target = args.target

    zoo = load_zoo_module()
    track_index = zoo.build_track_index((), ())
    public_examples = build_public_examples(zoo, args.split_dir, track_index)
    sessions = build_public_sessions(args.split_dir, track_index)

    mapping = spotify_to_idx(args.mapping)
    tpd1_sessions, tpd1_stats = load_tpd1_sessions(mapping)
    tpd1_cooc, tpd1_transition, tpd1_cooc_pmi, table_stats = build_tpd1_tables(
        tpd1_sessions,
        min_count=args.min_count,
        cooc_mode=args.tpd1_cooc_mode,
        cooc_window=args.tpd1_cooc_window,
        cooc_distance_weight=args.tpd1_cooc_distance_weight,
    )

    source = args.source
    if args.target == "public_labeled":
        run_public(
            zoo,
            track_index,
            public_examples,
            sessions,
            source,
            args,
            cfg,
            tpd1_sessions,
            tpd1_cooc,
            tpd1_transition,
            tpd1_cooc_pmi,
            tpd1_stats,
            table_stats,
        )
    else:
        blind_examples = build_blind_examples(zoo, args.target, track_index)
        run_blind(
            zoo,
            track_index,
            public_examples,
            blind_examples,
            sessions,
            source,
            args,
            cfg,
            tpd1_sessions,
            tpd1_cooc,
            tpd1_transition,
            tpd1_cooc_pmi,
            tpd1_stats,
            table_stats,
        )


if __name__ == "__main__":
    main()
