#!/usr/bin/env python3
"""Build protocol artifacts for two_tower_lora_thought.

The default mode builds the honest train-row OOF artifact for using the
supervised two-tower retriever as downstream reranker candidates/features.  For
each fixed public fold, it trains the 113 LoRA two-tower model on the other
folds and emits candidates for the held-out fold only.

The full_public mode trains one model on all public-labeled rows and emits
candidates for a blind target.  This is the inference counterpart to the OOF
artifact and should be used for submission pipelines.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pyarrow.parquet as pq
import torch
from datasets import load_dataset
from tqdm import tqdm

from recsys2026.artifacts import (
    component_output_dir,
    component_results_dir,
    encode_keys,
    file_ref,
    json_dump,
    utc_now,
)
from recsys2026.data import load
from recsys2026.paths import REPO_ROOT
from recsys2026.splits import read_jsonl
from recsys2026.submission import InferenceInput


@dataclass(frozen=True)
class PublicExample:
    source_split: Literal["train", "devset", "blind_a", "blind_b"]
    session_id: str
    user_id: str
    turn_number: int
    fold: int
    q_text: str
    gold_track_id: str
    gold_idx: int
    chat_history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class ExternalPair:
    q_text: str
    gold_track_id: str
    gold_idx: int
    source: str


def load_113_module() -> Any:
    from recsys2026 import two_tower

    return two_tower


def load_fold_map(split_dir: Path) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for row in read_jsonl(split_dir / "sessions.jsonl"):
        out[(str(row["source_split"]), str(row["session_id"]))] = int(row["fold"])
    return out


def split_name(split_dir: Path) -> str:
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists():
        try:
            return str(json.loads(manifest_path.read_text()).get("name") or split_dir.name)
        except Exception:  # noqa: BLE001
            return split_dir.name
    return split_dir.name


def spotify_to_idx(mapping_path: Path) -> dict[str, int]:
    table = pq.read_table(mapping_path, columns=["spotify_id", "track_idx"])
    spotify = table.column("spotify_id").to_pylist()
    idx = table.column("track_idx").to_pylist()
    return {str(s): int(i) for s, i in zip(spotify, idx, strict=True)}


def build_tpd1_pairs(
    tt113: Any,
    mapping_path: Path,
    track_ids: list[str],
    *,
    max_pairs: int | None,
    sample_seed: int,
) -> tuple[list[ExternalPair], dict[str, Any]]:
    mapping = spotify_to_idx(mapping_path)
    ds = load_dataset("talkpl-ai/TalkPlayData-1", split="train")
    pairs: list[ExternalPair] = []
    rng = random.Random(sample_seed)
    n_music = 0
    n_mapped = 0
    n_missing_user = 0
    n_pair_candidates = 0
    for item in tqdm(ds, desc="read TalkPlayData-1 pairs"):
        conversations = list(item.get("conversations") or [])
        for i, turn in enumerate(conversations):
            if turn.get("role") != "music":
                continue
            n_music += 1
            gold_idx = mapping.get(str(turn.get("content") or ""))
            if gold_idx is None or not (0 <= int(gold_idx) < len(track_ids)):
                continue
            n_mapped += 1
            user_pos = None
            for j in range(i - 1, -1, -1):
                if conversations[j].get("role") == "user":
                    user_pos = j
                    break
            if user_pos is None:
                n_missing_user += 1
                continue
            user_turn = conversations[user_pos]
            history: list[dict[str, Any]] = []
            for h, prev in enumerate(conversations[:user_pos]):
                history.append(
                    {
                        "role": prev.get("role"),
                        "content": prev.get("content"),
                        "turn_number": prev.get("turn_number", h + 1),
                    }
                )
            inp = InferenceInput(
                session_id=str(item.get("session_id") or item.get("cid") or item.get("pid") or ""),
                user_id=str(item.get("user_id") or item.get("pid") or ""),
                turn_number=int(user_turn.get("turn_number") or len(history) + 1),
                chat_history=history,
                user_query=str(user_turn.get("content") or ""),
            )
            q_text = tt113.query_text_with_goal_thought(inp, {}, "")
            if not q_text:
                continue
            pair = ExternalPair(
                q_text=q_text,
                gold_track_id=str(track_ids[int(gold_idx)]),
                gold_idx=int(gold_idx),
                source="TalkPlayData-1",
            )
            n_pair_candidates += 1
            if max_pairs is None or len(pairs) < max_pairs:
                pairs.append(pair)
            else:
                replace_at = rng.randrange(n_pair_candidates)
                if replace_at < max_pairs:
                    pairs[replace_at] = pair
    stats = {
        "tpd1_rows": len(ds),
        "music_turns_seen": n_music,
        "mapped_music_turns_seen": n_mapped,
        "missing_prior_user_music_turns": n_missing_user,
        "candidate_train_pairs": n_pair_candidates,
        "train_pairs": len(pairs),
        "max_pairs": max_pairs,
        "sample_mode": "all" if max_pairs is None else "deterministic_reservoir",
        "sample_seed": sample_seed,
        "mapping": str(mapping_path.relative_to(REPO_ROOT)),
        "unmapped_tracks_filtered": True,
    }
    return pairs, stats


def build_public_examples(
    tt113: Any,
    split_dir: Path,
    track_id_to_idx: dict[str, int],
) -> list[PublicExample]:
    fold_map = load_fold_map(split_dir)
    examples: list[PublicExample] = []
    for source_split, dataset_split in (("train", "train"), ("devset", "test")):
        for item in load("dataset", split=dataset_split):
            conversations = list(item["conversations"])
            fold = fold_map[(source_split, item["session_id"])]
            for target_turn in range(1, tt113.MAX_TURNS + 1):
                current = [c for c in conversations if int(c["turn_number"]) == target_turn]
                user_turn = next((c for c in current if c["role"] == "user"), None)
                music_turn = next((c for c in current if c["role"] == "music"), None)
                if user_turn is None or music_turn is None:
                    continue
                history = tuple(c for c in conversations if int(c["turn_number"]) < target_turn)
                inp = InferenceInput(
                    session_id=item["session_id"],
                    user_id=item["user_id"],
                    turn_number=target_turn,
                    chat_history=list(history),
                    user_query=user_turn["content"],
                )
                q_text = tt113.query_text_with_goal_thought(
                    inp,
                    item.get("conversation_goal") or {},
                    user_turn.get("thought") or "",
                )
                gold_tid = str(music_turn["content"])
                gold_idx = track_id_to_idx.get(gold_tid, -1)
                if gold_idx < 0:
                    continue
                examples.append(
                    PublicExample(
                        source_split=source_split,  # type: ignore[arg-type]
                        session_id=str(item["session_id"]),
                        user_id=str(item["user_id"]),
                        turn_number=target_turn,
                        fold=fold,
                        q_text=q_text,
                        gold_track_id=gold_tid,
                        gold_idx=gold_idx,
                        chat_history=history,
                    )
                )
    return examples


def build_blind_examples(
    tt113: Any,
    target: Literal["blind_a", "blind_b"],
) -> list[PublicExample]:
    examples: list[PublicExample] = []
    for item in load(target, split="test"):
        conversations = list(item["conversations"])
        current = conversations[-1]
        target_turn = int(current["turn_number"])
        history = tuple(c for c in conversations if int(c["turn_number"]) < target_turn)
        inp = InferenceInput(
            session_id=item["session_id"],
            user_id=item["user_id"],
            turn_number=target_turn,
            chat_history=list(history),
            user_query=current["content"],
        )
        q_text = tt113.query_text_with_goal_thought(
            inp,
            item.get("conversation_goal") or {},
            current.get("thought") or "",
        )
        examples.append(
            PublicExample(
                source_split=target,
                session_id=str(item["session_id"]),
                user_id=str(item["user_id"]),
                turn_number=target_turn,
                fold=-1,
                q_text=q_text,
                gold_track_id="",
                gold_idx=-1,
                chat_history=history,
            )
        )
    return examples


def infer_candidates(
    query_emb: np.ndarray,
    track_emb: np.ndarray,
    examples: list[PublicExample],
    track_id_to_idx: dict[str, int],
    *,
    top_k: int,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cand = np.full((len(examples), top_k), -1, dtype=np.int32)
    scores = np.full((len(examples), top_k), np.nan, dtype=np.float32)
    sizes = np.zeros(len(examples), dtype=np.int32)
    track_t = track_emb.T.astype(np.float32, copy=False)
    for start in tqdm(range(0, len(examples), chunk_size), desc="infer candidates"):
        end = min(start + chunk_size, len(examples))
        sim = (query_emb[start:end].astype(np.float32, copy=False) @ track_t).astype(np.float32, copy=False)
        for local_i, ex in enumerate(examples[start:end]):
            played: set[int] = set()
            for c in ex.chat_history:
                if c.get("role") == "music":
                    idx = track_id_to_idx.get(c.get("content"))
                    if idx is not None:
                        played.add(idx)
            score = sim[local_i]
            if played:
                score[list(played)] = -np.inf
            k = min(top_k, len(score))
            part = np.argpartition(-score, k - 1)[:k]
            order = np.argsort(-score[part], kind="stable")
            idx = part[order].astype(np.int32, copy=False)
            vals = score[idx].astype(np.float32, copy=False)
            cand[start + local_i, :k] = idx
            scores[start + local_i, :k] = vals
            sizes[start + local_i] = k
    return cand, sizes, scores


def save_fold_npz(
    path: Path,
    rows: np.ndarray,
    cand: np.ndarray,
    sizes: np.ndarray,
    scores: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rows=rows.astype(np.int32, copy=False),
        track_idx=cand.astype(np.int32, copy=False),
        sizes=sizes.astype(np.int32, copy=False),
        score__primary=scores.astype(np.float32, copy=False),
    )


def metrics_by_source(
    examples: list[PublicExample],
    cand: np.ndarray,
    sizes: np.ndarray,
    *,
    ks: tuple[int, ...] = (20, 50, 100, 200, 500),
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "n_examples": len(examples),
        "mean_size": float(sizes.mean()) if len(sizes) else 0.0,
    }
    groups = {
        "all": np.arange(len(examples), dtype=np.int32),
        "train": np.asarray([i for i, ex in enumerate(examples) if ex.source_split == "train"], dtype=np.int32),
        "devset": np.asarray([i for i, ex in enumerate(examples) if ex.source_split == "devset"], dtype=np.int32),
    }
    gold = np.asarray([ex.gold_idx for ex in examples], dtype=np.int32)
    for name, rows in groups.items():
        if len(rows) == 0:
            continue
        prefix = "" if name == "all" else f"{name}_"
        out[f"{prefix}n_examples"] = int(len(rows))
        out[f"{prefix}mean_size"] = float(sizes[rows].mean())
        for k in ks:
            kk = min(k, cand.shape[1])
            hits = (cand[rows, :kk] == gold[rows, None]).any(axis=1)
            out[f"{prefix}recall@{k}"] = float(hits.mean())
        hit_all = np.zeros(len(rows), dtype=bool)
        for j, row_i in enumerate(rows):
            hit_all[j] = bool((cand[row_i, : int(sizes[row_i])] == gold[row_i]).any())
        out[f"{prefix}recall@all"] = float(hit_all.mean())
    return out


def jsonable_args(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            try:
                out[key] = str(value.relative_to(REPO_ROOT))
            except ValueError:
                out[key] = str(value)
        else:
            out[key] = value
    return out


def blind_b_safe_enabled() -> bool:
    return True  # blind-B-safe fixed


def preferred_oof_mode(config: str) -> str:
    return "cv5_oof" if "oof5" in config else "cv3_oof"


def save_full_public_artifact(
    out_dir: Path,
    target: Literal["blind_a", "blind_b"],
    examples: list[PublicExample],
    track_idx: np.ndarray,
    sizes: np.ndarray,
    scores: np.ndarray,
    rank: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = [(ex.session_id, ex.turn_number) for ex in examples]
    np.savez_compressed(
        out_dir / "candidates.npz",
        track_idx=track_idx.astype(np.int32, copy=False),
        sizes=sizes.astype(np.int32, copy=False),
        keys=encode_keys(keys),
        rank=rank.astype(np.int32, copy=False),
        score__primary=scores.astype(np.float32, copy=False),
    )
    with (out_dir / "turns.jsonl").open("w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            f.write(
                json.dumps(
                    {
                        "row_id": i,
                        "session_id": ex.session_id,
                        "user_id": ex.user_id,
                        "turn_number": ex.turn_number,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    json_dump(out_dir / "manifest.json", manifest)


def run_full_public(
    tt113: Any,
    args: argparse.Namespace,
    split_dir: Path,
    device: str,
) -> None:
    target = args.blind_target
    out_dir = component_output_dir(
        "retriever",
        "two_tower_lora_thought",
        args.config,
        fit_mode="full_public",
        target=target,
    )
    model_path = out_dir / "models" / "full_public.pt"
    track_emb_path = out_dir / "models" / "track_emb_full_public.npy"
    cand_path = out_dir / "candidates.npz"
    if cand_path.exists() and not args.force:
        print(f"[skip] {cand_path}")
        return

    track_ids, track_features = tt113.build_track_features(REPO_ROOT / "artifacts/cache/two_tower/track_features.npz")
    track_id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    public_examples = build_public_examples(tt113, split_dir, track_id_to_idx)
    blind_examples = build_blind_examples(tt113, target)
    external_pairs: list[ExternalPair] = []
    external_stats: dict[str, Any] | None = None
    if args.tpd1_mix:
        external_pairs, external_stats = build_tpd1_pairs(
            tt113,
            args.tpd1_mapping,
            track_ids,
            max_pairs=args.tpd1_max_pairs,
            sample_seed=args.seed,
        )
    print(f"public train examples={len(public_examples)}")
    if args.tpd1_mix:
        print(f"tpd1 train pairs={len(external_pairs)}")
    print(f"{target} examples={len(blind_examples)}")
    q_texts = [ex.q_text for ex in public_examples] + [ex.q_text for ex in external_pairs]
    gold_idxs = np.asarray(
        [ex.gold_idx for ex in public_examples] + [ex.gold_idx for ex in external_pairs],
        dtype=np.int64,
    )

    if args.load_models_dir is not None:
        ckpt = args.load_models_dir / "full_public.pt"
        tokenizer, qwen, q_head, t_head = tt113.load_lora_two_tower(ckpt, device=device)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ckpt, model_path)  # run-local copy so the manifest file_ref resolves
    else:
        tokenizer, qwen, q_head, t_head = tt113.train_lora_two_tower(
            q_texts,
            gold_idxs,
            track_features,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            init_checkpoint=args.init_checkpoint,
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "qwen_lora": {n: p.detach().cpu() for n, p in qwen.named_parameters() if "lora" in n},
                "q_head": q_head.state_dict(),
                "t_head": t_head.state_dict(),
                "train_rows": int(len(public_examples)),
                "external_train_rows": int(len(external_pairs)),
                "target": target,
            },
            model_path,
        )
    print("encoding tracks")
    track_z = tt113.encode_all_tracks(t_head, track_features, device)
    np.save(track_emb_path, track_z)
    print(f"encoding {target} queries")
    query_z = tt113.encode_all_queries(tokenizer, qwen, q_head, [ex.q_text for ex in blind_examples], device)
    cand, sizes, scores = infer_candidates(
        query_z,
        track_z,
        blind_examples,
        track_id_to_idx,
        top_k=args.top_k,
        chunk_size=args.infer_chunk_size,
    )
    rank = np.broadcast_to(np.arange(1, args.top_k + 1, dtype=np.int32), cand.shape)
    bsafe = blind_b_safe_enabled()
    manifest = {
        "schema_version": 1,
        "artifact_type": "candidates",
        "stage": "retriever",
        "name": "two_tower_lora_thought",
        "config": args.config,
        "artifact_mode": "full_public",
        "target": target,
        "created_at": utc_now(),
        "producer": {"command": ["uv", "run", "python", "scripts/build_two_tower_lora_oof.py"], "cwd": "."},
        "split_artifact": str(split_dir.relative_to(REPO_ROOT)),
        "params": jsonable_args(args),
        "model": file_ref(model_path),
        "track_embedding": file_ref(track_emb_path),
        "fit_scope": {
            "fit_mode": "full_public",
            "fit_splits": ["public_labeled"] + (["TalkPlayData-1 train"] if args.tpd1_mix else []),
            "requires_labeled_fit": True,
            "fit_sources": (
                ["public_labeled_query_gold_pairs", "TalkPlayData-1 mapped query_gold_pairs"]
                if args.tpd1_mix
                else ["public_labeled_query_gold_pairs"]
            ),
            "train_row_policy": "inference_only",
            "fold_split_required_for_reranker_train": False,
            "uses_devset_for_fit": True,
            "uses_blind_for_fit": False,
            "note": "Submission/inference artifact: one model trained on all public-labeled rows, then applied to blind rows.",
        },
        "source_policy": {
            "requires_labeled_fit": True,
            "fit_sources": (
                [
                    "public_labeled_query_gold_pairs",
                    "TalkPlayData-1 mapped query_gold_pairs",
                    "two_tower (ex-113) architecture",
                ]
                if args.tpd1_mix
                else ["public_labeled_query_gold_pairs", "two_tower (ex-113) architecture"]
            ),
            "train_row_policy": "inference_only",
            "fold_split_required_for_reranker_train": False,
            "preferred_train_row_artifact_mode": preferred_oof_mode(args.config),
            "preferred_inference_artifact_mode": "full_public",
        },
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_blind_for_fit": False,
            "target_row_excluded_from_fit": None,
            "blind_b_safe": bsafe,
            "current_thought_allowed": not bsafe,
            "conversation_goal_allowed": not bsafe,
            "gpa_allowed": False,
        },
        "external_data": (
            {
                "name": "talkpl-ai/TalkPlayData-1",
                "split": "train",
                "stats": external_stats,
            }
            if args.tpd1_mix
            else None
        ),
    }
    save_full_public_artifact(out_dir, target, blind_examples, cand, sizes, scores, rank, manifest)
    print(f"wrote {out_dir}")


def run_tpd1_pretrain(
    tt113: Any,
    args: argparse.Namespace,
    device: str,
) -> None:
    out_dir = component_output_dir(
        "retriever",
        "two_tower_lora_thought",
        args.config,
        fit_mode="pretrain_tpd1",
        target="tpd1",
    )
    model_path = out_dir / "models" / "pretrain_tpd1.pt"
    if model_path.exists() and not args.force:
        print(f"[skip] {model_path}")
        return

    track_ids, track_features = tt113.build_track_features(REPO_ROOT / "artifacts/cache/two_tower/track_features.npz")
    external_pairs, external_stats = build_tpd1_pairs(
        tt113,
        args.tpd1_mapping,
        track_ids,
        max_pairs=args.tpd1_max_pairs,
        sample_seed=args.seed,
    )
    print(f"tpd1 pretrain pairs={len(external_pairs)}")
    q_texts = [ex.q_text for ex in external_pairs]
    gold_idxs = np.asarray([ex.gold_idx for ex in external_pairs], dtype=np.int64)

    tokenizer, qwen, q_head, t_head = tt113.train_lora_two_tower(
        q_texts,
        gold_idxs,
        track_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        init_checkpoint=args.init_checkpoint,
    )
    del tokenizer
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "qwen_lora": {n: p.detach().cpu() for n, p in qwen.named_parameters() if "lora" in n},
            "q_head": q_head.state_dict(),
            "t_head": t_head.state_dict(),
            "external_train_rows": int(len(external_pairs)),
            "source": "TalkPlayData-1",
            "stats": external_stats,
        },
        model_path,
    )
    bsafe = blind_b_safe_enabled()
    manifest = {
        "schema_version": 1,
        "artifact_type": "model_checkpoint",
        "stage": "retriever",
        "name": "two_tower_lora_thought",
        "config": args.config,
        "artifact_mode": "pretrain_tpd1",
        "target": "tpd1",
        "created_at": utc_now(),
        "producer": {"command": ["uv", "run", "python", "scripts/build_two_tower_lora_oof.py"], "cwd": "."},
        "params": jsonable_args(args),
        "model": file_ref(model_path),
        "fit_scope": {
            "fit_mode": "pretrain_tpd1",
            "fit_splits": ["TalkPlayData-1 train"],
            "requires_labeled_fit": True,
            "fit_sources": ["TalkPlayData-1 mapped query_gold_pairs"],
            "train_row_policy": "external_pretrain_only",
            "fold_split_required_for_reranker_train": False,
            "uses_devset_for_fit": False,
            "uses_blind_for_fit": False,
            "note": "External-only warm-start checkpoint. Public CV folds must still fine-tune/evaluate out of fold.",
        },
        "source_policy": {
            "requires_labeled_fit": True,
            "fit_sources": [
                "TalkPlayData-1 mapped query_gold_pairs",
                "two_tower (ex-113) architecture",
            ],
            "train_row_policy": "external_pretrain_only",
            "fold_split_required_for_reranker_train": False,
            "preferred_train_row_artifact_mode": preferred_oof_mode(args.config),
            "preferred_inference_artifact_mode": "full_public",
        },
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_blind_for_fit": False,
            "target_row_excluded_from_fit": None,
            "blind_b_safe": bsafe,
            "current_thought_allowed": not bsafe,
            "conversation_goal_allowed": not bsafe,
            "gpa_allowed": False,
        },
        "external_data": {
            "name": "talkpl-ai/TalkPlayData-1",
            "split": "train",
            "stats": external_stats,
        },
    }
    json_dump(out_dir / "manifest.json", manifest)
    print(f"wrote {model_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cv3_oof", "cv5_oof", "full_public", "tpd1_pretrain"), default="cv3_oof")
    parser.add_argument("--config", default="oof3_top500")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "splits" / "public_labeled_v1")
    parser.add_argument("--blind-target", choices=("blind_a", "blind_b"), default="blind_a")
    parser.add_argument("--fold", type=int, action="append", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--infer-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260516)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tpd1-mix", action="store_true", help="Add mapped TalkPlayData-1 query-track pairs to every training fold.")
    parser.add_argument("--tpd1-mapping", type=Path, default=REPO_ROOT / "artifacts/cache/spotify_uuid_map.parquet")
    parser.add_argument("--tpd1-max-pairs", type=int, default=None)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--load-models-dir",
        type=Path,
        default=None,
        help="Load shipped two-tower weights ({full_public,fold0..4}.pt) from this dir and "
        "encode/retrieve WITHOUT training. Reproduces candidates from weights.",
    )
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    split_dir = args.split_dir if args.split_dir.is_absolute() else REPO_ROOT / args.split_dir
    args.tpd1_mapping = args.tpd1_mapping if args.tpd1_mapping.is_absolute() else REPO_ROOT / args.tpd1_mapping
    if args.init_checkpoint is not None and not args.init_checkpoint.is_absolute():
        args.init_checkpoint = REPO_ROOT / args.init_checkpoint
    tt113 = load_113_module()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    if args.mode == "full_public":
        run_full_public(tt113, args, split_dir, device)
        return
    if args.mode == "tpd1_pretrain":
        run_tpd1_pretrain(tt113, args, device)
        return

    artifact_mode = str(args.mode)
    out_dir = component_output_dir(
        "retriever",
        "two_tower_lora_thought",
        args.config,
        fit_mode=artifact_mode,
        target="public_labeled",
    )
    res_dir = component_results_dir(
        "retriever",
        "two_tower_lora_thought",
        args.config,
        fit_mode=artifact_mode,
        target="public_labeled",
    )

    track_ids, track_features = tt113.build_track_features(REPO_ROOT / "artifacts/cache/two_tower/track_features.npz")
    track_id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    examples = build_public_examples(tt113, split_dir, track_id_to_idx)
    external_pairs: list[ExternalPair] = []
    external_stats: dict[str, Any] | None = None
    if args.tpd1_mix:
        external_pairs, external_stats = build_tpd1_pairs(
            tt113,
            args.tpd1_mapping,
            track_ids,
            max_pairs=args.tpd1_max_pairs,
            sample_seed=args.seed,
        )
    print(f"public examples={len(examples)}")
    if args.tpd1_mix:
        print(f"tpd1 train pairs per fold={len(external_pairs)}")
    folds = np.asarray([ex.fold for ex in examples], dtype=np.int16)
    run_folds = args.fold if args.fold is not None else sorted(int(x) for x in np.unique(folds))
    print(f"run folds={run_folds}")

    for fold in run_folds:
        fold_path = out_dir / "folds" / f"fold{fold}.npz"
        if fold_path.exists() and not args.force:
            print(f"[skip] {fold_path}")
            continue
        t0 = time.time()
        train_rows = np.flatnonzero(folds != fold)
        valid_rows = np.flatnonzero(folds == fold)
        print(
            f"fold {fold}: train_rows={len(train_rows)} "
            f"external_train_rows={len(external_pairs)} valid_rows={len(valid_rows)}"
        )
        q_texts = [examples[int(i)].q_text for i in train_rows] + [ex.q_text for ex in external_pairs]
        gold_idxs = np.asarray(
            [examples[int(i)].gold_idx for i in train_rows] + [ex.gold_idx for ex in external_pairs],
            dtype=np.int64,
        )
        model_path = out_dir / "models" / f"fold{fold}.pt"
        if args.load_models_dir is not None:
            ckpt = args.load_models_dir / f"fold{fold}.pt"
            tokenizer, qwen, q_head, t_head = tt113.load_lora_two_tower(ckpt, device=device)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ckpt, model_path)  # run-local copy for the fold manifest file_ref
        else:
            tokenizer, qwen, q_head, t_head = tt113.train_lora_two_tower(
                q_texts,
                gold_idxs,
                track_features,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                init_checkpoint=args.init_checkpoint,
            )
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "qwen_lora": {n: p.detach().cpu() for n, p in qwen.named_parameters() if "lora" in n},
                    "q_head": q_head.state_dict(),
                    "t_head": t_head.state_dict(),
                    "fold": fold,
                    "train_rows": int(len(train_rows)),
                    "external_train_rows": int(len(external_pairs)),
                    "valid_rows": int(len(valid_rows)),
                    "init_checkpoint": str(args.init_checkpoint.relative_to(REPO_ROOT)) if args.init_checkpoint else None,
                },
                model_path,
            )
        print(f"fold {fold}: encoding tracks")
        track_z = tt113.encode_all_tracks(t_head, track_features, device)
        np.save(out_dir / "models" / f"track_emb_fold{fold}.npy", track_z)
        valid_examples = [examples[int(i)] for i in valid_rows]
        valid_q = [ex.q_text for ex in valid_examples]
        print(f"fold {fold}: encoding held-out queries")
        query_z = tt113.encode_all_queries(tokenizer, qwen, q_head, valid_q, device)
        cand, sizes, scores = infer_candidates(
            query_z,
            track_z,
            valid_examples,
            track_id_to_idx,
            top_k=args.top_k,
            chunk_size=args.infer_chunk_size,
        )
        save_fold_npz(fold_path, valid_rows, cand, sizes, scores)
        elapsed = time.time() - t0
        print(f"fold {fold}: wrote {fold_path} elapsed={elapsed:.1f}s")
        del tokenizer, qwen, q_head, t_head
        torch.cuda.empty_cache()

    width = args.top_k
    track_idx = np.full((len(examples), width), -1, dtype=np.int32)
    sizes = np.zeros(len(examples), dtype=np.int32)
    scores = np.full((len(examples), width), np.nan, dtype=np.float32)
    missing: list[int] = []
    for fold in sorted(int(x) for x in np.unique(folds)):
        fold_path = out_dir / "folds" / f"fold{fold}.npz"
        if not fold_path.exists():
            missing.append(fold)
            continue
        data = np.load(fold_path)
        rows = data["rows"].astype(np.int32)
        track_idx[rows] = data["track_idx"]
        sizes[rows] = data["sizes"]
        scores[rows] = data["score__primary"]
    if missing:
        print(f"not combining yet; missing folds={missing}")
        return

    rank = np.broadcast_to(np.arange(1, width + 1, dtype=np.int32), track_idx.shape)
    keys = [(f"{ex.source_split}:{ex.session_id}", ex.turn_number) for ex in examples]
    np.savez_compressed(
        out_dir / "candidates.npz",
        track_idx=track_idx,
        sizes=sizes,
        keys=encode_keys(keys),
        source_split=np.asarray([ex.source_split.encode("utf-8") for ex in examples], dtype="S8"),
        folds=folds,
        rank=rank,
        score__primary=scores,
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
                        "fold": int(ex.fold),
                        "gold_track_id": ex.gold_track_id,
                        "gold_track_idx": int(ex.gold_idx),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    bsafe = blind_b_safe_enabled()
    manifest = {
        "schema_version": 1,
        "artifact_type": "candidates",
        "stage": "retriever",
        "name": "two_tower_lora_thought",
        "config": args.config,
        "artifact_mode": artifact_mode,
        "target": "public_labeled",
        "created_at": utc_now(),
        "producer": {"command": ["uv", "run", "python", "scripts/build_two_tower_lora_oof.py"], "cwd": "."},
        "split_artifact": str(split_dir.relative_to(REPO_ROOT)),
        "params": jsonable_args(args),
        "fold_models": [
            file_ref(out_dir / "models" / f"fold{fold}.pt")
            for fold in sorted(int(x) for x in np.unique(folds))
        ],
        "fit_scope": {
            "fit_mode": artifact_mode,
            "fit_splits": ["public_labeled"] + (["TalkPlayData-1 train"] if args.tpd1_mix else []),
            "requires_labeled_fit": True,
            "fit_sources": (
                ["public_labeled_query_gold_pairs", "TalkPlayData-1 mapped query_gold_pairs"]
                if args.tpd1_mix
                else ["public_labeled_query_gold_pairs"]
            ),
            "train_row_policy": f"out_of_fold_by_{split_name(split_dir)}",
            "fold_split_required_for_reranker_train": True,
            "uses_devset_for_fit": True,
            "uses_blind_for_fit": False,
            "note": "Each row is inferred by the fold model trained without that row's public fold.",
        },
        "source_policy": {
            "requires_labeled_fit": True,
            "fit_sources": (
                [
                    "public_labeled_query_gold_pairs",
                    "TalkPlayData-1 mapped query_gold_pairs",
                    "two_tower (ex-113) architecture",
                ]
                if args.tpd1_mix
                else ["public_labeled_query_gold_pairs", "two_tower (ex-113) architecture"]
            ),
            "train_row_policy": f"out_of_fold_by_{split_name(split_dir)}",
            "fold_split_required_for_reranker_train": True,
            "preferred_train_row_artifact_mode": artifact_mode,
            "preferred_inference_artifact_mode": "full_public",
        },
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_blind_for_fit": False,
            "target_row_excluded_from_fit": True,
            "blind_b_safe": bsafe,
            "current_thought_allowed": not bsafe,
            "conversation_goal_allowed": not bsafe,
            "gpa_allowed": False,
        },
        "external_data": (
            {
                "name": "talkpl-ai/TalkPlayData-1",
                "split": "train",
                "stats": external_stats,
            }
            if args.tpd1_mix
            else None
        ),
    }
    json_dump(out_dir / "manifest.json", manifest)
    metrics = metrics_by_source(examples, track_idx, sizes)
    metrics.update(
        {
            "name": "two_tower_lora_thought",
            "config": args.config,
            "artifact_mode": artifact_mode,
            "target": "public_labeled",
            "artifact": str(out_dir.relative_to(REPO_ROOT)),
        }
    )
    json_dump(res_dir / "scores.json", metrics)
    print(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
