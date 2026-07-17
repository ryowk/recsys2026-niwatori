#!/usr/bin/env python3
"""Shared OOF/inference artifact runner for count-based retrievers."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import yaml
from tqdm import tqdm

from recsys2026 import retriever_common as common
from recsys2026.artifacts import (
    artifact_complete,
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
from recsys2026.splits import read_jsonl


PUBLIC_SOURCES = ("train", "devset")
DATASET_BY_SOURCE = {"train": "train", "devset": "test"}
MAX_TURNS = 8


@dataclass(frozen=True)
class PublicExample:
    source_split: Literal["train", "devset", "blind_b"]
    session_id: str
    user_id: str
    turn_number: int
    fold: int
    chat_history: tuple[dict[str, Any], ...]
    user_query: str
    gold_track_id: str
    gold_idx: int


@dataclass(frozen=True)
class SessionMusic:
    source_split: Literal["train", "devset"]
    session_id: str
    user_id: str
    fold: int
    track_ids: tuple[str, ...]
    track_idxs: tuple[int, ...]


@dataclass
class Cooc:
    track_track: dict[int, tuple[np.ndarray, np.ndarray]]
    transition_track: dict[int, tuple[np.ndarray, np.ndarray]]
    album_album: dict[str, Counter]
    artist_name_artist_name: dict[str, Counter]


ScoreFunction = Callable[
    [Any, list[PublicExample], Any, Cooc, int],
    tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]],
]


@dataclass(frozen=True)
class TrainStatSpec:
    name: str
    source_path: Path
    score_examples: ScoreFunction


def read_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def source_entry_from_config(cfg: dict[str, Any], source: str) -> dict[str, Any]:
    """Return the source entry whose name or component matches the builder source."""
    for raw in cfg.get("sources") or []:
        if isinstance(raw, str):
            if raw == source:
                return {"name": raw}
            continue
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name") or "")
        component = str(raw.get("component") or name)
        if source in {name, component}:
            return dict(raw)
    return {}


def source_policy_from_config(cfg: dict[str, Any], source: str) -> dict[str, Any]:
    defaults = dict(cfg.get("source_policy_defaults") or {})
    source_metadata = cfg.get("source_metadata") or {}
    entry = source_entry_from_config(cfg, source)
    metadata = dict(source_metadata.get(source) or {})
    entry_name = str(entry.get("name") or "")
    if entry_name and entry_name != source:
        metadata = {**dict(source_metadata.get(entry_name) or {}), **metadata}
    entry_policy = dict(entry.get("source_policy") or {})
    policy = {**defaults, **metadata, **entry_policy}
    policy.setdefault("requires_labeled_fit", True)
    policy.setdefault("train_row_policy", "requires_oof")
    policy.setdefault("fold_split_required_for_reranker_train", True)
    policy.setdefault("preferred_train_row_artifact_mode", "cv5_oof")
    policy.setdefault("preferred_inference_artifact_mode", "full_public")
    return policy


def load_fold_map(split_dir: Path) -> dict[tuple[str, str], int]:
    return {
        (str(row["source_split"]), str(row["session_id"])): int(row["fold"])
        for row in read_jsonl(split_dir / "sessions.jsonl")
    }


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


def build_public_examples(
    common_module: Any, split_dir: Path, track_index: Any
) -> list[PublicExample]:
    fold_map = load_fold_map(split_dir)
    examples: list[PublicExample] = []
    source_splits = tuple(
        source for source in PUBLIC_SOURCES if any(key[0] == source for key in fold_map)
    )
    for source_split in source_splits:
        ds = load("dataset", split=DATASET_BY_SOURCE[source_split])
        for item in ds:
            conversations = list(item["conversations"])
            fold = fold_map[(source_split, item["session_id"])]
            for target_turn in range(1, MAX_TURNS + 1):
                current = [
                    c for c in conversations if int(c["turn_number"]) == target_turn
                ]
                user_turn = next((c for c in current if c["role"] == "user"), None)
                music_turn = next((c for c in current if c["role"] == "music"), None)
                if user_turn is None or music_turn is None:
                    continue
                gold_tid = str(music_turn.get("content") or "")
                gold_idx = track_index.id_to_idx.get(gold_tid, -1)
                examples.append(
                    PublicExample(
                        source_split=source_split,  # type: ignore[arg-type]
                        session_id=str(item["session_id"]),
                        user_id=str(item["user_id"]),
                        turn_number=target_turn,
                        fold=fold,
                        chat_history=tuple(
                            c
                            for c in conversations
                            if int(c["turn_number"]) < target_turn
                        ),
                        user_query=str(user_turn.get("content") or ""),
                        gold_track_id=gold_tid,
                        gold_idx=int(gold_idx),
                    )
                )
    return examples


def build_inference_examples(
    common_module: Any,
    target: Literal["devset", "blind_b"],
    track_index: Any,
) -> list[PublicExample]:
    examples: list[PublicExample] = []
    dataset_name = "dataset" if target == "devset" else target
    for item in load(dataset_name, split="test"):
        conversations = list(item["conversations"])
        target_turns = (
            range(1, MAX_TURNS + 1)
            if target == "devset"
            else [int(conversations[-1]["turn_number"])]
        )
        for target_turn in target_turns:
            current_turn = [
                c for c in conversations if int(c["turn_number"]) == target_turn
            ]
            current = next((c for c in current_turn if c.get("role") == "user"), None)
            music = next((c for c in current_turn if c.get("role") == "music"), None)
            if current is None:
                continue
            gold_track_id = str(music.get("content") or "") if music is not None else ""
            examples.append(
                PublicExample(
                    source_split=target,
                    session_id=str(item["session_id"]),
                    user_id=str(item["user_id"]),
                    turn_number=target_turn,
                    fold=-1,
                    chat_history=tuple(
                        c for c in conversations if int(c["turn_number"]) < target_turn
                    ),
                    user_query=str(current.get("content") or ""),
                    gold_track_id=gold_track_id,
                    gold_idx=int(track_index.id_to_idx.get(gold_track_id, -1)),
                )
            )
    return examples


def build_public_sessions(split_dir: Path, track_index: Any) -> list[SessionMusic]:
    fold_map = load_fold_map(split_dir)
    sessions: list[SessionMusic] = []
    source_splits = tuple(
        source for source in PUBLIC_SOURCES if any(key[0] == source for key in fold_map)
    )
    for source_split in source_splits:
        ds = load("dataset", split=DATASET_BY_SOURCE[source_split])
        for item in ds:
            tids = tuple(
                str(c["content"])
                for c in item["conversations"]
                if c.get("role") == "music" and c.get("content")
            )
            idxs = tuple(
                int(track_index.id_to_idx[tid])
                for tid in tids
                if tid in track_index.id_to_idx
            )
            sessions.append(
                SessionMusic(
                    source_split=source_split,  # type: ignore[arg-type]
                    session_id=str(item["session_id"]),
                    user_id=str(item["user_id"]),
                    fold=fold_map[(source_split, item["session_id"])],
                    track_ids=tids,
                    track_idxs=idxs,
                )
            )
    return sessions


def played_set(ex: PublicExample, track_index: Any) -> set[int]:
    played: set[int] = set()
    for c in ex.chat_history:
        if c.get("role") != "music":
            continue
        idx = track_index.id_to_idx.get(c.get("content"))
        if idx is not None:
            played.add(int(idx))
    return played


def history_state(common_module: Any, ex: PublicExample, track_index: Any):
    return common_module.history_state(
        common_module.TurnExample(
            session_id=ex.session_id,
            user_id=ex.user_id,
            turn_number=ex.turn_number,
            chat_history=list(ex.chat_history),
            user_query=ex.user_query,
            gold_track_id=ex.gold_track_id or None,
        ),
        track_index,
    )


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
    idxs = (
        np.flatnonzero(score > 0)
        if positive_only
        else np.arange(len(score), dtype=np.int32)
    )
    if len(idxs) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    if played:
        played_arr = np.fromiter(played, dtype=np.int32)
        idxs = idxs[~np.isin(idxs, played_arr)]
    if len(idxs) == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)
    vals = score[idxs]
    order = np.argsort(-vals, kind="stable")
    idxs = idxs[order][:top_k].astype(np.int32, copy=False)
    vals = vals[order][:top_k].astype(np.float32, copy=False)
    return idxs, vals


def pad_scored(
    rows: list[tuple[np.ndarray, np.ndarray]], top_k: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cand = np.full((len(rows), top_k), -1, dtype=np.int32)
    sizes = np.zeros(len(rows), dtype=np.int32)
    scores = np.full((len(rows), top_k), np.nan, dtype=np.float32)
    for i, (idxs, vals) in enumerate(rows):
        k = min(len(idxs), top_k)
        if k:
            cand[i, :k] = idxs[:k]
            scores[i, :k] = vals[:k]
            sizes[i] = k
    return cand, sizes, scores


def build_cooc_from_sessions(
    common_module: Any, track_index: Any, sessions: list[SessionMusic]
) -> Cooc:
    track_counts: dict[int, Counter] = defaultdict(Counter)
    transition_counts: dict[int, Counter] = defaultdict(Counter)
    album_counts: dict[str, Counter] = defaultdict(Counter)
    artist_name_counts: dict[str, Counter] = defaultdict(Counter)

    for session in tqdm(sessions, desc="build cooc"):
        track_idxs = list(dict.fromkeys(session.track_idxs))
        for i, track_idx in enumerate(track_idxs):
            counts = track_counts[track_idx]
            for neighbor_idx in track_idxs[i + 1 :]:
                counts[neighbor_idx] += 1
                track_counts[neighbor_idx][track_idx] += 1
        for previous_idx, next_idx in zip(
            session.track_idxs, session.track_idxs[1:], strict=False
        ):
            transition_counts[int(previous_idx)][int(next_idx)] += 1

        albums: set[str] = set()
        artist_names: set[str] = set()
        for tid in session.track_ids:
            md = track_index.meta_by_id.get(tid, {})
            for album_id in common_module.as_list(md.get("album_id")):
                if album_id:
                    albums.add(str(album_id))
            names = {
                common_module.norm_name(str(name))
                for name in common_module.as_list(md.get("artist_name"))
                if str(name or "").strip()
            }
            names.discard("")
            artist_names.update(names)

        album_list = list(albums)
        for i, a in enumerate(album_list):
            ca = album_counts[a]
            for b in album_list[i + 1 :]:
                ca[b] += 1
                album_counts[b][a] += 1

        name_list = list(artist_names)
        for i, a in enumerate(name_list):
            ca = artist_name_counts[a]
            for b in name_list[i + 1 :]:
                ca[b] += 1
                artist_name_counts[b][a] += 1

    def freeze(
        table: dict[int, Counter],
    ) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        frozen: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for key, counts in table.items():
            items = counts.most_common()
            if not items:
                continue
            neighbors = np.fromiter(
                (idx for idx, _ in items), dtype=np.int32, count=len(items)
            )
            values = np.fromiter(
                (value for _, value in items), dtype=np.float32, count=len(items)
            )
            frozen[int(key)] = (neighbors, values)
        return frozen

    return Cooc(
        track_track=freeze(track_counts),
        transition_track=freeze(transition_counts),
        album_album=dict(album_counts),
        artist_name_artist_name=dict(artist_name_counts),
    )


def public_metrics(
    examples: list[PublicExample], cand: np.ndarray, sizes: np.ndarray
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_examples": len(examples),
        "mean_size": float(sizes.mean()) if len(sizes) else 0.0,
    }
    gold = np.asarray([ex.gold_idx for ex in examples], dtype=np.int32)
    groups = {
        "all": np.arange(len(examples), dtype=np.int32),
        "train": np.asarray(
            [i for i, ex in enumerate(examples) if ex.source_split == "train"],
            dtype=np.int32,
        ),
        "devset": np.asarray(
            [i for i, ex in enumerate(examples) if ex.source_split == "devset"],
            dtype=np.int32,
        ),
    }
    for name, rows in groups.items():
        if len(rows) == 0:
            continue
        prefix = "" if name == "all" else f"{name}_"
        out[f"{prefix}n_examples"] = int(len(rows))
        out[f"{prefix}mean_size"] = float(sizes[rows].mean())
        for k in (20, 50, 100, 200, 500):
            kk = min(k, cand.shape[1])
            hit = (cand[rows, :kk] == gold[rows, None]).any(axis=1)
            out[f"{prefix}recall@{k}"] = float(hit.mean())
        hit_all = np.zeros(len(rows), dtype=bool)
        for j, row_i in enumerate(rows):
            hit_all[j] = bool((cand[row_i, : int(sizes[row_i])] == gold[row_i]).any())
        out[f"{prefix}recall@all"] = float(hit_all.mean())
    return out


def write_public_artifact(
    out_dir: Path,
    examples: list[PublicExample],
    cand: np.ndarray,
    sizes: np.ndarray,
    scores: np.ndarray,
    extra_scores: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rank = np.tile(np.arange(1, cand.shape[1] + 1, dtype=np.int32), (cand.shape[0], 1))
    for i, size in enumerate(sizes):
        rank[i, int(size) :] = -1
    arrays: dict[str, np.ndarray] = {
        "track_idx": cand.astype(np.int32, copy=False),
        "sizes": sizes.astype(np.int32, copy=False),
        "keys": encode_keys(
            [(f"{ex.source_split}:{ex.session_id}", ex.turn_number) for ex in examples]
        ),
        "source_split": np.asarray(
            [ex.source_split.encode("utf-8") for ex in examples], dtype="S8"
        ),
        "folds": np.asarray([ex.fold for ex in examples], dtype=np.int16),
        "rank": rank,
        "score__primary": scores.astype(np.float32, copy=False),
    }
    for name, arr in extra_scores.items():
        key = name if name.startswith("score__") else f"score__{name}"
        arrays[key] = arr.astype(np.float32, copy=False)
    turns = [
        {
            "row_id": i,
            "source_split": ex.source_split,
            "session_id": ex.session_id,
            "user_id": ex.user_id,
            "turn_number": ex.turn_number,
            "fold": int(ex.fold),
            "gold_track_id": ex.gold_track_id,
            "gold_track_idx": int(ex.gold_idx),
        }
        for i, ex in enumerate(examples)
    ]
    save_npz_artifact(out_dir, arrays, turns, manifest)


def write_inference_artifact(
    out_dir: Path,
    target: Literal["devset", "blind_b"],
    cand: np.ndarray,
    sizes: np.ndarray,
    scores: np.ndarray,
    extra_scores: dict[str, np.ndarray],
    manifest: dict[str, Any],
) -> None:
    rank = np.tile(np.arange(1, cand.shape[1] + 1, dtype=np.int32), (cand.shape[0], 1))
    for i, size in enumerate(sizes):
        rank[i, int(size) :] = -1
    score_arrays = {"primary": scores, **extra_scores}
    save_candidate_artifact(
        out_dir,
        cand,
        sizes,
        target=target,
        manifest=manifest,
        rank=rank,
        score_arrays=score_arrays,
        compress=True,
    )


def base_manifest(
    args: argparse.Namespace,
    spec: TrainStatSpec,
    cfg: dict[str, Any],
    policy: dict[str, Any],
    elapsed: float,
) -> dict[str, Any]:
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
        "source_code": {
            "runner": file_ref(REPO_ROOT / "src/recsys2026/train_stat_runner.py"),
            "common": file_ref(REPO_ROOT / "src/recsys2026/retriever_common.py"),
            "component": file_ref(spec.source_path),
            "config": file_ref(REPO_ROOT / args.config_file),
        },
        "params": {
            "config": args.config,
            "top_k": args.top_k,
        },
        "source_policy": policy,
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_blind_for_fit": False,
            "uses_target_future_turns": False,
            "same_user_memory_date_censored": False,
            "popularity_tiebreaker": False,
        },
        "candidate_universe": "all_tracks",
        "retention": "top_k",
        "score_fields": ["score__primary"],
        "elapsed_sec": elapsed,
    }


def build_source_for_examples(
    spec: TrainStatSpec,
    common_module: Any,
    track_index: Any,
    examples: list[PublicExample],
    sessions: list[SessionMusic],
    args: argparse.Namespace,
    *,
    fold: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    fit_sessions = [s for s in sessions if fold is None or s.fold != fold]
    cooc = build_cooc_from_sessions(common_module, track_index, fit_sessions)
    return spec.score_examples(common_module, examples, track_index, cooc, args.top_k)


def run_public(
    common_module: Any,
    track_index: Any,
    public_examples: list[PublicExample],
    sessions: list[SessionMusic],
    spec: TrainStatSpec,
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> None:
    source = spec.name
    policy = source_policy_from_config(cfg, source)
    fit_mode = str(args.artifact_mode or "cv5_oof")
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
        print(f"{source}: fold {fold}, rows={len(valid_rows)}")
        sub_cand, sub_sizes, sub_scores, _ = build_source_for_examples(
            spec,
            common_module,
            track_index,
            fold_examples,
            sessions,
            args,
            fold=fold,
        )
        cand[valid_rows] = sub_cand
        sizes[valid_rows] = sub_sizes
        scores[valid_rows] = sub_scores

    elapsed = time.time() - t0
    manifest = base_manifest(args, spec, cfg, policy, elapsed)
    manifest.update({"artifact_mode": fit_mode, "target": "public_labeled"})
    manifest["score_fields"] = ["score__primary"] + [
        f"score__{key}" for key in sorted(extra_scores)
    ]
    fit_splits = sorted({ex.source_split for ex in public_examples})
    manifest["fit_scope"] = {
        "fit_mode": fit_mode,
        "fit_splits": fit_splits,
        "requires_labeled_fit": True,
        "fit_sources": list(policy.get("fit_sources") or []),
        "train_row_policy": f"out_of_fold_by_{split_name(args.split_dir)}",
        "fold_split_required_for_reranker_train": True,
        "uses_devset_for_fit": "devset" in fit_splits,
        "uses_blind_for_fit": False,
        "target_row_excluded_from_fit": True,
    }
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
    sessions: list[SessionMusic],
    spec: TrainStatSpec,
    args: argparse.Namespace,
    cfg: dict[str, Any],
) -> None:
    source = spec.name
    policy = source_policy_from_config(cfg, source)
    target = args.inference_target
    fit_splits = sorted({ex.source_split for ex in public_examples})
    fit_mode = "full_train" if fit_splits == ["train"] else "full_public"
    out_dir = component_output_dir(
        "retriever", source, args.config, fit_mode=fit_mode, target=target
    )
    if artifact_complete(out_dir, "candidates.npz", "turns.jsonl"):
        print(f"[skip] {out_dir}")
        return
    t0 = time.time()
    cand, sizes, scores, extra_scores = build_source_for_examples(
        spec,
        common_module,
        track_index,
        inference_examples,
        sessions,
        args,
        fold=None,
    )
    elapsed = time.time() - t0
    manifest = base_manifest(args, spec, cfg, policy, elapsed)
    manifest.update({"artifact_mode": fit_mode, "target": target})
    manifest["score_fields"] = ["score__primary"] + [
        f"score__{key}" for key in sorted(extra_scores)
    ]
    manifest["fit_scope"] = {
        "fit_mode": fit_mode,
        "fit_splits": fit_splits,
        "requires_labeled_fit": True,
        "fit_sources": list(policy.get("fit_sources") or []),
        "train_row_policy": "inference_only",
        "fold_split_required_for_reranker_train": False,
        "uses_devset_for_fit": "devset" in fit_splits,
        "uses_blind_for_fit": False,
        "target_row_excluded_from_fit": None,
    }
    write_inference_artifact(
        out_dir, target, cand, sizes, scores, extra_scores, manifest
    )
    if target == "devset":
        metrics = public_metrics(inference_examples, cand, sizes)
        metrics.update(
            {
                "name": source,
                "config": args.config,
                "artifact_mode": fit_mode,
                "target": target,
                "artifact": str(out_dir.relative_to(REPO_ROOT)),
            }
        )
        json_dump(
            component_results_dir(
                "retriever", source, args.config, fit_mode=fit_mode, target=target
            )
            / "scores.json",
            metrics,
        )
        print(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir} mean_size={sizes.mean():.1f} elapsed={elapsed:.1f}s")


def main(spec: TrainStatSpec) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="oof5_top500")
    parser.add_argument(
        "--config-file",
        type=Path,
        default=Path("retriever/union/configs/combined_tpd1_parts_cooc500_cv5.yaml"),
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
        "--mode", choices=("public", "inference", "both"), default="both"
    )
    parser.add_argument(
        "--inference-target", choices=("devset", "blind_b"), default="blind_b"
    )
    parser.add_argument("--top-k", type=int, default=500)
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    args.config_file = (
        args.config_file
        if args.config_file.is_absolute()
        else REPO_ROOT / args.config_file
    )
    args.split_dir = (
        args.split_dir if args.split_dir.is_absolute() else REPO_ROOT / args.split_dir
    )
    cfg = read_config(args.config_file)
    print("building track index")
    track_index = common.build_track_index(())
    print("building public examples/sessions")
    public_examples = build_public_examples(common, args.split_dir, track_index)
    sessions = build_public_sessions(args.split_dir, track_index)
    print(
        f"public_examples={len(public_examples)} sessions={len(sessions)} source={spec.name}"
    )
    inference_examples: list[PublicExample] = []
    if args.mode in {"inference", "both"}:
        inference_examples = build_inference_examples(
            common, args.inference_target, track_index
        )
        print(f"{args.inference_target} examples={len(inference_examples)}")

    if args.mode in {"public", "both"}:
        run_public(common, track_index, public_examples, sessions, spec, args, cfg)
    if args.mode in {"inference", "both"}:
        run_inference(
            common,
            track_index,
            public_examples,
            inference_examples,
            sessions,
            spec,
            args,
            cfg,
        )
