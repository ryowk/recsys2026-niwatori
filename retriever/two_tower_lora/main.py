#!/usr/bin/env python3
"""Build OOF or full-fit two-tower retriever artifacts."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from tqdm import tqdm

from retriever.two_tower_lora import model as two_tower
from recsys2026.artifacts import (
    artifact_complete,
    component_output_dir,
    component_results_dir,
    encode_keys,
    file_ref,
    json_dump,
    npz_dump,
    save_npz_artifact,
    utc_now,
)
from recsys2026.data import load
from recsys2026.paths import REPO_ROOT
from recsys2026.splits import read_jsonl
from recsys2026.submission import InferenceInput


@dataclass(frozen=True)
class PublicExample:
    source_split: Literal["train", "devset", "blind_b"]
    session_id: str
    user_id: str
    turn_number: int
    fold: int
    q_text: str
    gold_track_id: str
    gold_idx: int
    chat_history: tuple[dict[str, Any], ...]


def load_fold_map(split_dir: Path) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for row in read_jsonl(split_dir / "sessions.jsonl"):
        out[(str(row["source_split"]), str(row["session_id"]))] = int(row["fold"])
    return out


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
    two_tower: Any,
    split_dir: Path,
    track_id_to_idx: dict[str, int],
) -> list[PublicExample]:
    fold_map = load_fold_map(split_dir)
    examples: list[PublicExample] = []
    requested = tuple(
        source
        for source in ("train", "devset")
        if any(key[0] == source for key in fold_map)
    )
    for source_split in requested:
        dataset_split = "train" if source_split == "train" else "test"
        for item in load("dataset", split=dataset_split):
            conversations = list(item["conversations"])
            fold = fold_map[(source_split, item["session_id"])]
            for target_turn in range(1, two_tower.MAX_TURNS + 1):
                current = [
                    c for c in conversations if int(c["turn_number"]) == target_turn
                ]
                user_turn = next((c for c in current if c["role"] == "user"), None)
                music_turn = next((c for c in current if c["role"] == "music"), None)
                if user_turn is None or music_turn is None:
                    continue
                history = tuple(
                    c for c in conversations if int(c["turn_number"]) < target_turn
                )
                inp = InferenceInput(
                    session_id=item["session_id"],
                    user_id=item["user_id"],
                    turn_number=target_turn,
                    chat_history=list(history),
                    user_query=user_turn["content"],
                )
                q_text = two_tower.query_text(inp)
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


def build_inference_examples(
    two_tower: Any,
    target: Literal["devset", "blind_b"],
    track_id_to_idx: dict[str, int],
) -> list[PublicExample]:
    examples: list[PublicExample] = []
    dataset_name = "dataset" if target == "devset" else target
    for item in load(dataset_name, split="test"):
        conversations = list(item["conversations"])
        target_turns = (
            range(1, two_tower.MAX_TURNS + 1)
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
            history = tuple(
                c for c in conversations if int(c["turn_number"]) < target_turn
            )
            inp = InferenceInput(
                session_id=item["session_id"],
                user_id=item["user_id"],
                turn_number=target_turn,
                chat_history=list(history),
                user_query=current["content"],
            )
            q_text = two_tower.query_text(inp)
            gold_track_id = str(music.get("content") or "") if music is not None else ""
            examples.append(
                PublicExample(
                    source_split=target,
                    session_id=str(item["session_id"]),
                    user_id=str(item["user_id"]),
                    turn_number=target_turn,
                    fold=-1,
                    q_text=q_text,
                    gold_track_id=gold_track_id,
                    gold_idx=int(track_id_to_idx.get(gold_track_id, -1)),
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
        sim = (query_emb[start:end].astype(np.float32, copy=False) @ track_t).astype(
            np.float32, copy=False
        )
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
    npz_dump(
        path,
        {
            "rows": rows.astype(np.int32, copy=False),
            "track_idx": cand.astype(np.int32, copy=False),
            "sizes": sizes.astype(np.int32, copy=False),
            "score__primary": scores.astype(np.float32, copy=False),
        },
        compress=True,
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
        "train": np.asarray(
            [i for i, ex in enumerate(examples) if ex.source_split == "train"],
            dtype=np.int32,
        ),
        "devset": np.asarray(
            [i for i, ex in enumerate(examples) if ex.source_split == "devset"],
            dtype=np.int32,
        ),
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


def preferred_oof_mode(config: str) -> str:
    if "paper_train5" in config:
        return "train5_oof"
    return "cv5_oof"


def save_full_fit_artifact(
    out_dir: Path,
    target: Literal["devset", "blind_b"],
    examples: list[PublicExample],
    track_idx: np.ndarray,
    sizes: np.ndarray,
    scores: np.ndarray,
    rank: np.ndarray,
    manifest: dict[str, Any],
) -> None:
    keys = [(ex.session_id, ex.turn_number) for ex in examples]
    arrays = {
        "track_idx": track_idx.astype(np.int32, copy=False),
        "sizes": sizes.astype(np.int32, copy=False),
        "keys": encode_keys(keys),
        "rank": rank.astype(np.int32, copy=False),
        "score__primary": scores.astype(np.float32, copy=False),
    }
    turns = [
        {
            "row_id": i,
            "session_id": ex.session_id,
            "user_id": ex.user_id,
            "turn_number": ex.turn_number,
        }
        for i, ex in enumerate(examples)
    ]
    save_npz_artifact(out_dir, arrays, turns, manifest)


def run_full_fit(
    two_tower: Any,
    args: argparse.Namespace,
    split_dir: Path,
    device: str,
) -> None:
    target = args.inference_target
    fit_mode = "full_train" if args.mode == "full_train" else "full_public"
    out_dir = component_output_dir(
        "retriever",
        "two_tower_lora",
        args.config,
        fit_mode=fit_mode,
        target=target,
    )
    model_path = out_dir / "models" / f"{fit_mode}.pt"
    track_emb_path = out_dir / "models" / f"track_emb_{fit_mode}.npy"
    if artifact_complete(out_dir, "candidates.npz", "turns.jsonl"):
        print(f"[skip] {out_dir}")
        return

    track_ids, track_features = two_tower.build_track_features(
        REPO_ROOT / "artifacts/preprocessed/two_tower/track_features.npz"
    )
    track_id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    public_examples = build_public_examples(two_tower, split_dir, track_id_to_idx)
    inference_examples = build_inference_examples(two_tower, target, track_id_to_idx)
    fit_splits = sorted({ex.source_split for ex in public_examples})
    if fit_mode == "full_train" and fit_splits != ["train"]:
        raise ValueError(
            f"full_train requires a train-only split artifact, got {fit_splits}"
        )
    print(f"public train examples={len(public_examples)}")
    print(f"{target} examples={len(inference_examples)}")
    q_texts = [ex.q_text for ex in public_examples]
    gold_idxs = np.asarray([ex.gold_idx for ex in public_examples], dtype=np.int64)

    tokenizer, qwen, q_head, t_head = two_tower.train_lora_two_tower(
        q_texts,
        gold_idxs,
        track_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "qwen_lora": {
                n: p.detach().cpu() for n, p in qwen.named_parameters() if "lora" in n
            },
            "q_head": q_head.state_dict(),
            "t_head": t_head.state_dict(),
            "train_rows": int(len(public_examples)),
            "target": target,
        },
        model_path,
    )
    print("encoding tracks")
    track_z = two_tower.encode_all_tracks(t_head, track_features, device)
    np.save(track_emb_path, track_z)
    print(f"encoding {target} queries")
    query_z = two_tower.encode_all_queries(
        tokenizer, qwen, q_head, [ex.q_text for ex in inference_examples], device
    )
    cand, sizes, scores = infer_candidates(
        query_z,
        track_z,
        inference_examples,
        track_id_to_idx,
        top_k=args.top_k,
        chunk_size=args.infer_chunk_size,
    )
    rank = np.broadcast_to(np.arange(1, args.top_k + 1, dtype=np.int32), cand.shape)
    manifest = {
        "schema_version": 1,
        "artifact_type": "candidates",
        "stage": "retriever",
        "name": "two_tower_lora",
        "config": args.config,
        "artifact_mode": fit_mode,
        "target": target,
        "created_at": utc_now(),
        "producer": {
            "command": [
                "uv",
                "run",
                "python",
                "-m",
                "retriever.two_tower_lora.main",
            ],
            "cwd": ".",
        },
        "split_artifact": str(split_dir.relative_to(REPO_ROOT)),
        "params": jsonable_args(args),
        "model": file_ref(model_path),
        "track_embedding": file_ref(track_emb_path),
        "fit_scope": {
            "fit_mode": fit_mode,
            "fit_splits": fit_splits,
            "requires_labeled_fit": True,
            "fit_sources": [f"{'+'.join(fit_splits)}_query_gold_pairs"],
            "train_row_policy": "inference_only",
            "fold_split_required_for_reranker_train": False,
            "uses_devset_for_fit": "devset" in fit_splits,
            "uses_blind_for_fit": False,
            "note": f"Inference artifact: one model trained on {'+'.join(fit_splits)}, then applied to {target} rows.",
        },
        "source_policy": {
            "requires_labeled_fit": True,
            "fit_sources": [
                f"{'+'.join(fit_splits)}_query_gold_pairs",
                "two-tower LoRA architecture",
            ],
            "train_row_policy": "inference_only",
            "fold_split_required_for_reranker_train": False,
            "preferred_train_row_artifact_mode": preferred_oof_mode(args.config),
            "preferred_inference_artifact_mode": fit_mode,
        },
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_blind_for_fit": False,
            "target_row_excluded_from_fit": None,
            "current_thought_allowed": False,
            "conversation_goal_allowed": False,
            "gpa_allowed": False,
        },
    }
    save_full_fit_artifact(
        out_dir, target, inference_examples, cand, sizes, scores, rank, manifest
    )
    if target == "devset":
        metrics = metrics_by_source(inference_examples, cand, sizes)
        metrics.update(
            {
                "name": "two_tower_lora",
                "config": args.config,
                "artifact_mode": fit_mode,
                "target": target,
                "artifact": str(out_dir.relative_to(REPO_ROOT)),
            }
        )
        json_dump(
            component_results_dir(
                "retriever",
                "two_tower_lora",
                args.config,
                fit_mode=fit_mode,
                target=target,
            )
            / "scores.json",
            metrics,
        )
        print(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("cv5_oof", "train5_oof", "full_public", "full_train"),
        default="cv5_oof",
    )
    parser.add_argument("--config", default="oof5_top500")
    parser.add_argument(
        "--split-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/preprocessed/splits/cv5",
    )
    parser.add_argument(
        "--inference-target", choices=("devset", "blind_b"), default="blind_b"
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--infer-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260516)
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    split_dir = (
        args.split_dir if args.split_dir.is_absolute() else REPO_ROOT / args.split_dir
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    if args.mode in {"full_public", "full_train"}:
        run_full_fit(two_tower, args, split_dir, device)
        return
    artifact_mode = str(args.mode)
    out_dir = component_output_dir(
        "retriever",
        "two_tower_lora",
        args.config,
        fit_mode=artifact_mode,
        target="public_labeled",
    )
    res_dir = component_results_dir(
        "retriever",
        "two_tower_lora",
        args.config,
        fit_mode=artifact_mode,
        target="public_labeled",
    )

    track_ids, track_features = two_tower.build_track_features(
        REPO_ROOT / "artifacts/preprocessed/two_tower/track_features.npz"
    )
    track_id_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    examples = build_public_examples(two_tower, split_dir, track_id_to_idx)
    print(f"public examples={len(examples)}")
    folds = np.asarray([ex.fold for ex in examples], dtype=np.int16)
    run_folds = sorted(int(x) for x in np.unique(folds))
    print(f"run folds={run_folds}")

    for fold in run_folds:
        fold_path = out_dir / "folds" / f"fold{fold}.npz"
        if fold_path.exists():
            print(f"[skip] {fold_path}")
            continue
        t0 = time.time()
        train_rows = np.flatnonzero(folds != fold)
        valid_rows = np.flatnonzero(folds == fold)
        print(f"fold {fold}: train_rows={len(train_rows)} valid_rows={len(valid_rows)}")
        q_texts = [examples[int(i)].q_text for i in train_rows]
        gold_idxs = np.asarray(
            [examples[int(i)].gold_idx for i in train_rows], dtype=np.int64
        )
        model_path = out_dir / "models" / f"fold{fold}.pt"
        tokenizer, qwen, q_head, t_head = two_tower.train_lora_two_tower(
            q_texts,
            gold_idxs,
            track_features,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "qwen_lora": {
                    n: p.detach().cpu()
                    for n, p in qwen.named_parameters()
                    if "lora" in n
                },
                "q_head": q_head.state_dict(),
                "t_head": t_head.state_dict(),
                "fold": fold,
                "train_rows": int(len(train_rows)),
                "valid_rows": int(len(valid_rows)),
            },
            model_path,
        )
        print(f"fold {fold}: encoding tracks")
        track_z = two_tower.encode_all_tracks(t_head, track_features, device)
        np.save(out_dir / "models" / f"track_emb_fold{fold}.npy", track_z)
        valid_examples = [examples[int(i)] for i in valid_rows]
        valid_q = [ex.q_text for ex in valid_examples]
        print(f"fold {fold}: encoding held-out queries")
        query_z = two_tower.encode_all_queries(tokenizer, qwen, q_head, valid_q, device)
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
    arrays = {
        "track_idx": track_idx,
        "sizes": sizes,
        "keys": encode_keys(keys),
        "source_split": np.asarray(
            [ex.source_split.encode("utf-8") for ex in examples], dtype="S8"
        ),
        "folds": folds,
        "rank": rank,
        "score__primary": scores,
    }
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
    fit_splits = sorted({ex.source_split for ex in examples})
    manifest = {
        "schema_version": 1,
        "artifact_type": "candidates",
        "stage": "retriever",
        "name": "two_tower_lora",
        "config": args.config,
        "artifact_mode": artifact_mode,
        "target": "public_labeled",
        "created_at": utc_now(),
        "producer": {
            "command": [
                "uv",
                "run",
                "python",
                "-m",
                "retriever.two_tower_lora.main",
            ],
            "cwd": ".",
        },
        "split_artifact": str(split_dir.relative_to(REPO_ROOT)),
        "params": jsonable_args(args),
        "fold_models": [
            file_ref(out_dir / "models" / f"fold{fold}.pt")
            for fold in sorted(int(x) for x in np.unique(folds))
        ],
        "fit_scope": {
            "fit_mode": artifact_mode,
            "fit_splits": fit_splits,
            "requires_labeled_fit": True,
            "fit_sources": [f"{'+'.join(fit_splits)}_query_gold_pairs"],
            "train_row_policy": f"out_of_fold_by_{split_name(split_dir)}",
            "fold_split_required_for_reranker_train": True,
            "uses_devset_for_fit": "devset" in fit_splits,
            "uses_blind_for_fit": False,
            "note": "Each row is inferred by the fold model trained without that row's public fold.",
        },
        "source_policy": {
            "requires_labeled_fit": True,
            "fit_sources": [
                f"{'+'.join(fit_splits)}_query_gold_pairs",
                "two-tower LoRA architecture",
            ],
            "train_row_policy": f"out_of_fold_by_{split_name(split_dir)}",
            "fold_split_required_for_reranker_train": True,
            "preferred_train_row_artifact_mode": artifact_mode,
            "preferred_inference_artifact_mode": "full_train"
            if fit_splits == ["train"]
            else "full_public",
        },
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_blind_for_fit": False,
            "target_row_excluded_from_fit": True,
            "current_thought_allowed": False,
            "conversation_goal_allowed": False,
            "gpa_allowed": False,
        },
    }
    save_npz_artifact(out_dir, arrays, turns, manifest)
    metrics = metrics_by_source(examples, track_idx, sizes)
    metrics.update(
        {
            "name": "two_tower_lora",
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
