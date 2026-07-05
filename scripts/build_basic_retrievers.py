#!/usr/bin/env python3
"""Build basic per-retriever component artifacts with primary source scores.

This script is a bridge from the retriever-zoo implementations to the new
component artifact schema. Unlike ``import_legacy_candidates.py``, it reruns the
retriever logic and writes ``score__primary`` for sources whose primary score
is available.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import bm25s
import numpy as np
import torch
import yaml
from tqdm import tqdm

from recsys2026.artifacts import (
    component_output_dir,
    component_results_dir,
    encode_keys,
    file_ref,
    json_dump,
    save_candidate_artifact,
    utc_now,
)
from recsys2026.data import load
from recsys2026.paths import REPO_ROOT
from recsys2026.retriever_eval import candidate_metrics, devset_gold_indices
from recsys2026.splits import read_jsonl


PUBLIC_SOURCES = (("train", "train"), ("devset", "test"))


def load_zoo_module() -> Any:
    from recsys2026 import zoo

    return zoo


def load_llm_names_module() -> Any:
    path = REPO_ROOT / "exp/114_llm_track_names/main.py"
    spec = importlib.util.spec_from_file_location("basic_llm114", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_two_tower_module() -> Any:
    from recsys2026 import two_tower

    return two_tower


def read_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def source_policy_from_config(cfg: dict[str, Any], source: str) -> dict[str, Any]:
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


def build_public_labeled_examples(zoo: Any, split_dir: Path) -> tuple[list[Any], list[dict[str, Any]]]:
    fold_map = load_fold_map(split_dir)
    examples: list[Any] = []
    rows: list[dict[str, Any]] = []
    for source_split, dataset_split in PUBLIC_SOURCES:
        for item in load("dataset", split=dataset_split):
            conversations = list(item["conversations"])
            fold = fold_map[(source_split, str(item["session_id"]))]
            for target_turn in range(1, zoo.MAX_TURNS + 1):
                current = [c for c in conversations if int(c["turn_number"]) == target_turn]
                user_turn = next((c for c in current if c["role"] == "user"), None)
                music_turn = next((c for c in current if c["role"] == "music"), None)
                if user_turn is None or music_turn is None:
                    continue
                gold_track_id = str(music_turn.get("content") or "")
                examples.append(
                    zoo.TurnExample(
                        session_id=str(item["session_id"]),
                        user_id=str(item["user_id"]),
                        session_date=str(item.get("session_date") or ""),
                        turn_number=target_turn,
                        chat_history=[c for c in conversations if int(c["turn_number"]) < target_turn],
                        user_query=str(user_turn.get("content") or ""),
                        gold_track_id=gold_track_id,
                        user_thought=str(user_turn.get("thought") or "").strip(),
                        conversation_goal=dict(item.get("conversation_goal") or {}),
                    )
                )
                rows.append(
                    {
                        "row_id": len(rows),
                        "source_split": source_split,
                        "session_id": str(item["session_id"]),
                        "user_id": str(item["user_id"]),
                        "session_date": str(item.get("session_date") or ""),
                        "turn_number": target_turn,
                        "fold": fold,
                        "gold_track_id": gold_track_id,
                    }
                )
    return examples, rows


def build_blind_examples(zoo: Any, target: str) -> list[Any]:
    examples: list[Any] = []
    for item in load(target, split="test"):
        conversations = list(item["conversations"])
        current = conversations[-1]
        target_turn = int(current["turn_number"])
        examples.append(
            zoo.TurnExample(
                session_id=str(item["session_id"]),
                user_id=str(item["user_id"]),
                session_date=str(item.get("session_date") or ""),
                turn_number=target_turn,
                chat_history=[c for c in conversations if int(c["turn_number"]) < target_turn],
                user_query=str(current.get("content") or ""),
                gold_track_id=None,
                user_thought=str(current.get("thought") or "").strip(),
                conversation_goal=dict(item.get("conversation_goal") or {}),
            )
        )
    return examples


def public_labeled_metrics(rows: list[dict[str, Any]], track_index: Any, cand: np.ndarray, sizes: np.ndarray) -> dict[str, Any]:
    gold = np.asarray([track_index.id_to_idx.get(str(row["gold_track_id"]), -1) for row in rows], dtype=np.int32)
    out: dict[str, Any] = {
        "n_examples": len(rows),
        "mean_size": float(sizes.mean()) if len(sizes) else 0.0,
    }
    groups = {
        "all": np.arange(len(rows), dtype=np.int32),
        "train": np.asarray([i for i, row in enumerate(rows) if row["source_split"] == "train"], dtype=np.int32),
        "devset": np.asarray([i for i, row in enumerate(rows) if row["source_split"] == "devset"], dtype=np.int32),
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


def load_or_encode_qwen_queries(zoo: Any, examples: list, target: str, cache_path: Path, batch_size: int) -> np.ndarray:
    if cache_path.exists():
        print(f"  [cache] {cache_path}")
        return np.load(cache_path).astype(np.float32)
    if target == "public_labeled":
        train_path = REPO_ROOT / "output/081_two_tower/train_q_emb__n121592.npy"
        dev_path = REPO_ROOT / "output/086_retriever_zoo_v2/encode/qwen3_query_mat__n8000.npy"
        if train_path.exists() and dev_path.exists():
            train_q = np.load(train_path).astype(np.float32)
            dev_q = np.load(dev_path).astype(np.float32)
            mat = np.concatenate([train_q, dev_q], axis=0)
            if len(mat) == len(examples):
                print(f"  [cache] public_labeled Qwen queries: {train_path} + {dev_path}")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, mat)
                return mat
    if target.startswith("blind_"):
        shared_path = REPO_ROOT / f"artifacts/runs/retriever/train_neighbor/oof3_top500/encode/qwen3_query_mat__{target}.npy"
        if shared_path.exists():
            mat = np.load(shared_path).astype(np.float32)
            if len(mat) == len(examples):
                print(f"  [cache] blind Qwen queries: {shared_path}")
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, mat)
                return mat
    return zoo.encode_queries_qwen3(examples, cache_path, batch_size=batch_size)


def album_qwen3_cache_path(target: str, encode_dir: Path) -> Path:
    shared = REPO_ROOT / "output/086_retriever_zoo_v2/encode/album_qwen3_matrix.npy"
    if target != "devset" and shared.exists():
        return shared
    return encode_dir / "album_qwen3_matrix.npy"


def load_or_encode_qwen_texts(zoo: Any, texts: list[str], cache_path: Path, batch_size: int, desc: str) -> np.ndarray:
    if cache_path.exists():
        print(f"  [cache] {desc}: {cache_path}")
        return np.load(cache_path).astype(np.float32)
    from recsys2026.encoders import Qwen3TextEncoder

    encoder = Qwen3TextEncoder(batch_size=batch_size)
    print(f"  encoding {len(texts)} {desc} with Qwen3 ...")
    mat = zoo._normalize_rows(encoder.encode([text if text else " " for text in texts]).astype(np.float32))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, mat)
    return mat


def load_or_generate_hyde_rewrites(
    zoo: Any,
    examples: list,
    track_index: Any,
    target: str,
    encode_dir: Path,
    model_name: str,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    model_tag = model_name.replace("/", "_")
    cache_path = encode_dir / f"hyde_rewrites__{model_tag}__n{len(examples)}.json"
    if cache_path.exists():
        print(f"  [cache] hyde rewrites {cache_path}")
        return json.loads(cache_path.read_text())
    legacy = REPO_ROOT / "output/086_retriever_zoo_v2/encode/hyde_rewrites__Qwen_Qwen2.5-1.5B-Instruct__n8000.json"
    if target == "devset" and model_name == "Qwen/Qwen2.5-1.5B-Instruct" and legacy.exists():
        print(f"  [cache] hyde rewrites {legacy}")
        return json.loads(legacy.read_text())[: len(examples)]
    return llm_rewrite_checkpointed(
        zoo,
        examples,
        track_index,
        cache_path,
        zoo.HYDE_PROMPT,
        model_name=model_name,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
    )


def llm_rewrite_checkpointed(
    zoo: Any,
    examples: list,
    track_index: Any,
    cache_path: Path,
    system_prompt: str,
    *,
    model_name: str,
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    if cache_path.exists():
        print(f"  [cache] rewrites {cache_path}")
        return json.loads(cache_path.read_text())

    partial_path = cache_path.with_suffix(cache_path.suffix + ".jsonl")
    done: dict[int, str] = {}
    if partial_path.exists():
        with partial_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                idx = int(row["index"])
                if 0 <= idx < len(examples):
                    done[idx] = str(row.get("text") or "")
        print(f"  [resume] {partial_path}: {len(done)}/{len(examples)} rewrites")

    missing = [i for i in range(len(examples)) if i not in done]
    if missing:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"  loading {model_name} on {device} (dtype={dtype}) ...")
        tok = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()

        partial_path.parent.mkdir(parents=True, exist_ok=True)
        with partial_path.open("a", encoding="utf-8") as f, torch.no_grad():
            for start in tqdm(range(0, len(missing), batch_size), desc="llm rewrite"):
                batch_indices = missing[start : start + batch_size]
                prompts = []
                for idx in batch_indices:
                    chat_text = zoo.chat_to_text_for_llm(examples[idx], track_index.meta_by_id)
                    msgs = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": chat_text},
                    ]
                    prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
                enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                gen = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tok.pad_token_id,
                    do_sample=False,
                )
                new_tokens = gen[:, enc["input_ids"].shape[1] :]
                decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
                for idx, text in zip(batch_indices, decoded, strict=True):
                    line = text.strip().splitlines()[0] if text.strip() else ""
                    done[idx] = line.strip()
                    f.write(json.dumps({"index": idx, "text": done[idx]}, ensure_ascii=False) + "\n")
                f.flush()

        del model, tok
        if device == "cuda":
            torch.cuda.empty_cache()

    if len(done) != len(examples):
        missing_count = len(examples) - len(done)
        raise RuntimeError(f"incomplete LLM rewrite cache: {missing_count} missing")
    out = [done[i] for i in range(len(examples))]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, ensure_ascii=False))
    print(f"  saved {cache_path}")
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


def pad_scored(rows: list[tuple[np.ndarray, np.ndarray]], top_k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def bm25_scored(zoo: Any, examples: list, track_index: Any, bm25_name: str, query_mode: str, top_k: int):
    bm25 = track_index.bm25_indexes[bm25_name]
    rows = []
    for ex in tqdm(examples, desc=f"bm25[{bm25_name}/{query_mode}]"):
        played = zoo.played_set(ex, track_index)
        query = zoo._bm25_query_text(ex, track_index.meta_by_id, mode=query_mode)
        pool = min(top_k + len(played) + 16, track_index.n_tracks)
        toks = bm25s.tokenize([query], show_progress=False)
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
        rows.append((np.asarray(kept, dtype=np.int32), np.asarray(vals, dtype=np.float32)))
    return pad_scored(rows, top_k)


def bm25_queries_scored(zoo: Any, examples: list, track_index: Any, bm25_name: str, queries: list[str], top_k: int, desc: str):
    bm25 = track_index.bm25_indexes[bm25_name]
    rows = []
    for ex, query in tqdm(list(zip(examples, queries, strict=True)), desc=desc):
        if not query:
            rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue
        played = zoo.played_set(ex, track_index)
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
        rows.append((np.asarray(kept, dtype=np.int32), np.asarray(vals, dtype=np.float32)))
    return pad_scored(rows, top_k)


def count_scored(zoo: Any, examples: list, track_index: Any, score_fn: Any, desc: str, top_k: int):
    rows = []
    for ex in tqdm(examples, desc=desc):
        rows.append(select_from_score(score_fn(ex, track_index), zoo.played_set(ex, track_index), top_k, positive_only=True))
    return pad_scored(rows, top_k)


def _track_artist_ids(meta: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for raw in zoo_as_list(meta.get("artist_id")):
        text = str(raw or "").strip()
        if text and text not in out:
            out.append(text)
    return out


def zoo_as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def build_catalog_collab_graph(track_index: Any) -> dict[str, Counter[str]]:
    graph: dict[str, Counter[str]] = defaultdict(Counter)
    for tid in track_index.track_ids:
        ids = _track_artist_ids(track_index.meta_by_id.get(tid, {}))
        if len(ids) < 2:
            continue
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if a == b:
                    continue
                graph[a][b] += 1
                graph[b][a] += 1
    return graph


def collab_artist_expansion_scored(zoo: Any, examples: list, track_index: Any, top_k: int):
    graph = build_catalog_collab_graph(track_index)
    rows = []
    for ex in tqdm(examples, desc="collab_artist_expansion"):
        history_counts: Counter[str] = Counter()
        for msg in ex.chat_history:
            if msg.get("role") != "music":
                continue
            meta = track_index.meta_by_id.get(str(msg.get("content") or ""), {})
            history_counts.update(_track_artist_ids(meta))
        if not history_counts:
            rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue
        seed_artists = set(history_counts)
        neighbor_score: Counter[str] = Counter()
        for artist_id, hist_count in history_counts.items():
            for nb_id, collab_count in graph.get(artist_id, {}).items():
                if nb_id in seed_artists:
                    continue
                neighbor_score[nb_id] += float(hist_count) * float(collab_count)
        if not neighbor_score:
            rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue
        score = np.zeros(track_index.n_tracks, dtype=np.float32)
        for artist_id, value in neighbor_score.items():
            for j in track_index.artist_to_idx.get(artist_id, []):
                score[j] += float(value)
        rows.append(select_from_score(score, zoo.played_set(ex, track_index), top_k, positive_only=True))
    return pad_scored(rows, top_k)


def history_release_decade_scored(zoo: Any, examples: list, track_index: Any, top_k: int):
    decade_to_idx: dict[str, np.ndarray] = {}
    tmp: dict[str, list[int]] = defaultdict(list)
    track_decade: dict[str, str] = {}
    for i, tid in enumerate(track_index.track_ids):
        md = track_index.meta_by_id.get(tid, {})
        rd = md.get("release_date")
        if not rd:
            continue
        s = str(rd).strip()
        if len(s) >= 4 and s[:4].isdigit():
            decade = s[:3] + "0s"
            tmp[decade].append(i)
            track_decade[tid] = decade
    for decade, idxs in tmp.items():
        decade_to_idx[decade] = np.asarray(idxs, dtype=np.int32)

    rows = []
    for ex in tqdm(examples, desc="history_release_decade"):
        decades = {
            track_decade[str(c.get("content") or "")]
            for c in ex.chat_history
            if c.get("role") == "music" and str(c.get("content") or "") in track_decade
        }
        if not decades:
            rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue
        idxs = np.concatenate([decade_to_idx[d] for d in sorted(decades) if d in decade_to_idx])
        if len(idxs) == 0:
            rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue
        idxs = np.unique(idxs)
        played = zoo.played_set(ex, track_index)
        if played:
            idxs = idxs[~np.isin(idxs, np.fromiter(played, dtype=np.int32))]
        idxs = idxs[:top_k].astype(np.int32, copy=False)
        rows.append((idxs, np.ones(len(idxs), dtype=np.float32)))
    return pad_scored(rows, top_k)


def dense_scored(
    zoo: Any,
    examples: list,
    track_index: Any,
    dense_key: str,
    query_mat: np.ndarray | None,
    top_k: int,
    desc: str,
    *,
    device: str,
    batch_size: int,
    extra_k: int,
    track_mat_override: np.ndarray | None = None,
    played_sets: list[set[int]] | None = None,
    valid_mask: np.ndarray | None = None,
):
    if query_mat is None:
        return pad_scored(
            [(np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)) for _ in examples],
            top_k,
        )

    track_mat = track_mat_override if track_mat_override is not None else track_index.dense_mats[dense_key]
    track_mat = np.ascontiguousarray(track_mat, dtype=np.float32)
    query_mat = np.ascontiguousarray(query_mat, dtype=np.float32)
    if played_sets is None:
        played_sets = [zoo.played_set(ex, track_index) for ex in examples]
    if valid_mask is None:
        valid_mask = np.ones(len(examples), dtype=bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)

    cand = np.full((len(examples), top_k), -1, dtype=np.int32)
    sizes = np.zeros(len(examples), dtype=np.int32)
    scores = np.full((len(examples), top_k), np.nan, dtype=np.float32)
    pool = min(top_k + extra_k + max((len(p) for p in played_sets), default=0), track_index.n_tracks)
    if pool <= 0:
        return cand, sizes, scores

    use_cuda = device == "cuda" and torch.cuda.is_available()
    if use_cuda:
        torch_device = torch.device("cuda")
        track_t = torch.as_tensor(track_mat, device=torch_device).t().contiguous()
        with torch.no_grad():
            for start in tqdm(range(0, len(examples), batch_size), desc=desc):
                end = min(start + batch_size, len(examples))
                active = valid_mask[start:end]
                if not bool(active.any()):
                    continue
                q = torch.as_tensor(query_mat[start:end], device=torch_device)
                score_t = q @ track_t
                vals_t, idx_t = torch.topk(score_t, k=pool, dim=1, largest=True, sorted=True)
                idx_np = idx_t.cpu().numpy().astype(np.int32, copy=False)
                vals_np = vals_t.cpu().numpy().astype(np.float32, copy=False)
                del q, score_t, vals_t, idx_t
                for local_i in range(end - start):
                    row_i = start + local_i
                    if not valid_mask[row_i]:
                        continue
                    idxs = idx_np[local_i]
                    vals = vals_np[local_i]
                    played = played_sets[row_i]
                    if played:
                        keep = ~np.isin(idxs, np.fromiter(played, dtype=np.int32))
                        idxs = idxs[keep]
                        vals = vals[keep]
                    k = min(len(idxs), top_k)
                    if k:
                        cand[row_i, :k] = idxs[:k]
                        scores[row_i, :k] = vals[:k]
                        sizes[row_i] = k
        return cand, sizes, scores

    for start in tqdm(range(0, len(examples), batch_size), desc=desc):
        end = min(start + batch_size, len(examples))
        score_block = query_mat[start:end] @ track_mat.T
        for local_i in range(end - start):
            row_i = start + local_i
            if not valid_mask[row_i]:
                continue
            cand_i, score_i = select_from_score(score_block[local_i], played_sets[row_i], top_k)
            k = min(len(cand_i), top_k)
            if k:
                cand[row_i, :k] = cand_i[:k]
                scores[row_i, :k] = score_i[:k]
                sizes[row_i] = k
    return cand, sizes, scores


def album_qwen3_history_scored(
    zoo: Any,
    examples: list,
    track_index: Any,
    album_mat: np.ndarray,
    top_k: int,
    *,
    device: str,
    batch_size: int,
    extra_k: int,
):
    query_mat = np.zeros((len(examples), album_mat.shape[1]), dtype=np.float32)
    valid_mask = np.zeros(len(examples), dtype=bool)
    played_sets: list[set[int]] = []
    for i, ex in enumerate(tqdm(examples, desc="album_qwen3_history/query")):
        _, _, _, played, hist = zoo.history_state(ex, track_index)
        played_sets.append(set(played))
        if not hist:
            continue
        centroid = album_mat[np.asarray(hist, dtype=np.int32)].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0:
            continue
        query_mat[i] = (centroid / norm).astype(np.float32, copy=False)
        valid_mask[i] = True
    return dense_scored(
        zoo,
        examples,
        track_index,
        "album_qwen3_history",
        query_mat,
        top_k,
        "album_qwen3_history",
        device=device,
        batch_size=batch_size,
        extra_k=extra_k,
        track_mat_override=album_mat,
        played_sets=played_sets,
        valid_mask=valid_mask,
    )


def cf_history_centroid_scored(zoo: Any, examples: list, track_index: Any, top_k: int):
    rows = []
    cf = track_index.cf
    for ex in tqdm(examples, desc="cf_history_centroid"):
        _, _, _, played, history_idxs = zoo.history_state(ex, track_index)
        if not history_idxs:
            rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue
        centroid = cf[np.asarray(history_idxs, dtype=np.int32)].mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0:
            rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue
        rows.append(select_from_score(cf @ (centroid / norm), played, top_k))
    return pad_scored(rows, top_k)


def user_emb_track_cf_scored(zoo: Any, examples: list, track_index: Any, user_vectors: dict[str, np.ndarray], top_k: int):
    rows = []
    cf = track_index.cf
    for ex in tqdm(examples, desc="user_emb_track_cf"):
        vec = user_vectors.get(ex.user_id)
        if vec is None:
            rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue
        rows.append(select_from_score(cf @ vec, zoo.played_set(ex, track_index), top_k))
    return pad_scored(rows, top_k)


def cooc_track_scored(zoo: Any, examples: list, track_index: Any, cooc: Any, top_k: int):
    rows = []
    for ex in tqdm(examples, desc="cooc_track"):
        _, _, _, played, history_idxs = zoo.history_state(ex, track_index)
        if not history_idxs:
            rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue
        score = np.zeros(track_index.n_tracks, dtype=np.float32)
        for h in history_idxs:
            nb_cn = cooc.track_track.get(int(h))
            if nb_cn is None:
                continue
            nb, cn = nb_cn
            score[nb] += cn
        rows.append(select_from_score(score, played, top_k, positive_only=True))
    return pad_scored(rows, top_k)


def cooc_artist_scored(zoo: Any, examples: list, track_index: Any, cooc: Any, top_k: int):
    rows = []
    for ex in tqdm(examples, desc="cooc_artist"):
        h_arts, _, _, played, _ = zoo.history_state(ex, track_index)
        artist_score: dict[str, float] = defaultdict(float)
        for aid in h_arts:
            for nb_aid, count in (cooc.artist_artist.get(aid) or {}).items():
                artist_score[nb_aid] += float(count)
        score = np.zeros(track_index.n_tracks, dtype=np.float32)
        for nb_aid, value in artist_score.items():
            for idx in track_index.artist_to_idx.get(nb_aid, []):
                score[idx] += value
        rows.append(select_from_score(score, played, top_k, positive_only=True))
    return pad_scored(rows, top_k)


def popularity_global_scored(zoo: Any, examples: list, track_index: Any, top_k: int):
    rows = [
        select_from_score(track_index.popularity, zoo.played_set(ex, track_index), top_k)
        for ex in tqdm(examples, desc="popularity_global")
    ]
    return pad_scored(rows, top_k)


def train_play_count_unique_users_scored(zoo: Any, examples: list, track_index: Any, top_k: int):
    user_sets: list[set[str]] = [set() for _ in range(track_index.n_tracks)]
    for item in zoo.load("dataset", split="train"):
        user_id = str(item["user_id"])
        for conv in item["conversations"]:
            if conv.get("role") != "music":
                continue
            idx = track_index.id_to_idx.get(conv.get("content"))
            if idx is not None:
                user_sets[idx].add(user_id)
    score = np.asarray([len(s) for s in user_sets], dtype=np.float32)
    rows = [
        select_from_score(score, zoo.played_set(ex, track_index), top_k)
        for ex in tqdm(examples, desc="train_play_count_unique_users")
    ]
    return pad_scored(rows, top_k)


def train_neighbor_scored(zoo: Any, examples: list, track_index: Any, top_k: int, n_neigh: int, rank_offset: float, device: str):
    train_q_path = REPO_ROOT / "output/081_two_tower/train_q_emb__n121592.npy"
    dev_q_path = REPO_ROOT / "output/086_retriever_zoo_v2/encode/qwen3_query_mat__n8000.npy"
    train_q = zoo._normalize_rows(np.load(train_q_path).astype(np.float32))
    dev_q = zoo._normalize_rows(np.load(dev_q_path).astype(np.float32))[: len(examples)]

    pairs = []
    for item in zoo.load("dataset", split="train"):
        conversations = list(item["conversations"])
        for target_turn in range(1, zoo.MAX_TURNS + 1):
            current = [c for c in conversations if c["turn_number"] == target_turn]
            music_turn = next((c for c in current if c["role"] == "music"), None)
            user_turn = next((c for c in current if c["role"] == "user"), None)
            if music_turn is not None and user_turn is not None and music_turn.get("content"):
                pairs.append(music_turn["content"])
    train_gold = np.asarray([track_index.id_to_idx.get(tid, -1) for tid in pairs], dtype=np.int32)
    train_q = train_q[: len(train_gold)]

    t_q = torch.from_numpy(train_q).to(device)
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for i in tqdm(range(0, len(examples), 256), desc="train_neighbor"):
        chunk = torch.from_numpy(dev_q[i:i + 256]).to(device)
        sims = chunk @ t_q.T
        top = torch.topk(sims, k=min(n_neigh, t_q.shape[0]), dim=1)
        nbr_idx = top.indices.cpu().numpy()
        nbr_score = top.values.cpu().numpy()
        for j in range(nbr_idx.shape[0]):
            score = np.zeros(track_index.n_tracks, dtype=np.float32)
            for rank, (ni, sim) in enumerate(zip(nbr_idx[j], nbr_score[j], strict=True)):
                gi = int(train_gold[int(ni)])
                if gi >= 0:
                    score[gi] += float(sim) / (rank + rank_offset)
            rows.append(select_from_score(score, zoo.played_set(examples[i + j], track_index), top_k, positive_only=True))
    return pad_scored(rows, top_k)


def user_neighbor_scored(
    zoo: Any,
    examples: list,
    track_index: Any,
    top_k: int,
    n_neigh: int,
    rank_offset: float,
    device: str,
    score_mode: str,
):
    user_vecs = zoo.load_user_vectors_normalized()
    user_tracks: dict[str, list[int]] = defaultdict(list)
    for item in zoo.load("dataset", split="train"):
        uid = str(item["user_id"])
        seen: set[int] = set()
        for conv in item["conversations"]:
            if conv.get("role") == "music":
                idx = track_index.id_to_idx.get(conv.get("content"))
                if idx is not None and idx not in seen:
                    user_tracks[uid].append(idx)
                    seen.add(idx)
    train_user_ids = [uid for uid in user_tracks if uid in user_vecs]
    train_mat = zoo._normalize_rows(np.stack([user_vecs[uid] for uid in train_user_ids], axis=0).astype(np.float32))
    t_mat = torch.from_numpy(train_mat).to(device)
    rows = []
    for ex in tqdm(examples, desc="user_neighbor"):
        q = user_vecs.get(ex.user_id)
        if q is None:
            rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue
        sim = (torch.from_numpy(q).to(device).unsqueeze(0) @ t_mat.T).squeeze(0)
        top = torch.topk(sim, k=min(n_neigh + 1, len(train_user_ids)))
        idxs = top.indices.cpu().numpy()
        vals = top.values.cpu().numpy()
        score = np.zeros(track_index.n_tracks, dtype=np.float32)
        nbr_count = 0
        for ni, value in zip(idxs, vals, strict=True):
            nb_uid = train_user_ids[int(ni)]
            if nb_uid == ex.user_id:
                continue
            neighbor_weight = float(value) if score_mode == "sim_weighted" else 1.0
            for rank, tidx in enumerate(user_tracks.get(nb_uid, [])):
                score[int(tidx)] += neighbor_weight / (rank + rank_offset)
            nbr_count += 1
            if nbr_count >= n_neigh:
                break
        rows.append(select_from_score(score, zoo.played_set(ex, track_index), top_k, positive_only=True))
    return pad_scored(rows, top_k)


def llm_track_names_scored(zoo: Any, examples: list, track_index: Any, top_k: int, k_per_song: int):
    llm114 = load_llm_names_module()
    path = REPO_ROOT / "output/114_llm_track_names/song_names__Qwen_Qwen2.5-1.5B-Instruct__n8000.json"
    outputs = json.loads(path.read_text())[: len(examples)]
    bm25 = track_index.bm25_indexes["4field"]
    rows = []
    for i, ex in enumerate(tqdm(examples, desc="llm_track_names")):
        played = zoo.played_set(ex, track_index)
        score: dict[int, float] = defaultdict(float)
        for name in llm114.parse_song_names(outputs[i])[:5]:
            pool = min(k_per_song + len(played) + 16, track_index.n_tracks)
            toks = bm25s.tokenize([name.lower()], show_progress=False)
            idx_arr, _ = bm25.retrieve(toks, k=pool, show_progress=False)
            for rank, idx_raw in enumerate(idx_arr[0]):
                idx = int(idx_raw)
                if idx in played:
                    continue
                score[idx] += 1.0 / (rank + 1)
        if not score:
            rows.append((np.empty(0, dtype=np.int32), np.empty(0, dtype=np.float32)))
            continue
        items = sorted(score.items(), key=lambda kv: -kv[1])[:top_k]
        rows.append((
            np.asarray([k for k, _ in items], dtype=np.int32),
            np.asarray([v for _, v in items], dtype=np.float32),
        ))
    return pad_scored(rows, top_k)


def user_culture_match_scored(zoo: Any, examples: list, track_index: Any, top_k: int):
    profile_by_key = {}
    for item in zoo.load("dataset", split="test"):
        for turn in range(1, zoo.MAX_TURNS + 1):
            profile_by_key[(item["session_id"], turn)] = dict(item.get("user_profile") or {})
    word_to_idx: dict[str, list[int]] = defaultdict(list)
    for idx, tid in enumerate(track_index.track_ids):
        md = track_index.meta_by_id.get(tid, {})
        text_parts = []
        for field in ("tag_list", "artist_name", "album_name"):
            value = md.get(field)
            if isinstance(value, list):
                text_parts.extend(str(v).lower() for v in value if v)
            elif value:
                text_parts.append(str(value).lower())
        for word in " ".join(text_parts).split():
            word = word.strip(",.()[]'\"!?-")
            if len(word) > 2:
                word_to_idx[word].append(idx)
    rows = []
    for ex in tqdm(examples, desc="user_culture_match"):
        profile = profile_by_key.get((ex.session_id, ex.turn_number), {})
        words = []
        for key in ("preferred_musical_culture", "country_name"):
            for word in str(profile.get(key) or "").lower().split():
                word = word.strip(",.()[]'\"!?-")
                if len(word) > 2:
                    words.append(word)
        score = np.zeros(track_index.n_tracks, dtype=np.float32)
        for word in words:
            for idx in word_to_idx.get(word, []):
                score[idx] += 1.0
        rows.append(select_from_score(score, zoo.played_set(ex, track_index), top_k, positive_only=True))
    return pad_scored(rows, top_k)


def two_tower_lora_thought_scored(
    zoo: Any,
    examples: list,
    track_index: Any,
    top_k: int,
    device: str,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Regenerate 113 two-tower candidates from the trained model artifact."""
    from transformers import AutoModel, AutoTokenizer

    tt113 = load_two_tower_module()
    model_path = REPO_ROOT / "artifacts/cache/two_tower/model.pt"
    track_emb_path = REPO_ROOT / "artifacts/cache/two_tower/track_emb_512d.npy"
    track_features_path = REPO_ROOT / "artifacts/cache/two_tower/track_features.npz"
    if not model_path.exists() or not track_emb_path.exists() or not track_features_path.exists():
        raise FileNotFoundError("missing 113 two-tower model artifacts under output/113_two_tower_lora_thought")

    feature_npz = np.load(track_features_path, allow_pickle=True)
    feature_track_ids = [str(x) for x in feature_npz["track_ids"]]
    if feature_track_ids != list(track_index.track_ids):
        raise ValueError("113 track feature order does not match current all_tracks order")

    devset = tt113.build_devset_examples()
    dev_keys = [(str(ex["session_id"]), int(ex["turn_number"])) for ex in devset]
    zoo_keys = [(str(ex.session_id), int(ex.turn_number)) for ex in examples]
    if dev_keys[: len(zoo_keys)] != zoo_keys:
        raise ValueError("113 devset order does not match component target order")
    q_texts = [ex["q_text"] for ex in devset[: len(examples)]]

    track_emb = np.load(track_emb_path).astype(np.float32)
    if track_emb.shape[0] != track_index.n_tracks:
        raise ValueError(f"track embedding row mismatch: {track_emb.shape[0]} vs {track_index.n_tracks}")

    query_cache = cache_dir / "query_emb_512d.npy"
    if query_cache.exists():
        query_emb = np.load(query_cache).astype(np.float32)
        if query_emb.shape[0] != len(examples):
            query_emb = None
    else:
        query_emb = None

    if query_emb is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        print(f"  loading 113 Qwen3 + LoRA on {device}")
        tokenizer = AutoTokenizer.from_pretrained(tt113.QWEN3_MODEL, padding_side="left")
        qwen = AutoModel.from_pretrained(tt113.QWEN3_MODEL, dtype=dtype).to(device)
        tt113.add_lora_to_qwen3(qwen)
        qwen.eval()

        state = torch.load(model_path, map_location=device, weights_only=False)
        qwen_sd = {k: v.float() for k, v in state["qwen_lora"].items()}
        missing, unexpected = qwen.load_state_dict(qwen_sd, strict=False)
        print(f"  qwen lora load: missing={len(missing)}, unexpected={len(unexpected)}")

        q_head = tt113.QueryHead().to(device)
        q_head.load_state_dict(state["q_head"])
        q_head.eval()

        query_emb = tt113.encode_all_queries(tokenizer, qwen, q_head, q_texts, device, batch_size=16)
        query_cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(query_cache, query_emb.astype(np.float32))
        del qwen, q_head
        if device == "cuda":
            torch.cuda.empty_cache()

    rows: list[tuple[np.ndarray, np.ndarray]] = []
    t_emb = torch.from_numpy(track_emb).to(device)
    with torch.no_grad():
        for i in tqdm(range(0, len(examples), 128), desc="two_tower_lora_thought"):
            q = torch.from_numpy(query_emb[i:i + 128]).to(device)
            sims = (q @ t_emb.T).cpu().numpy()
            for j, score in enumerate(sims):
                rows.append(select_from_score(score, zoo.played_set(examples[i + j], track_index), top_k))
    del t_emb
    if device == "cuda":
        torch.cuda.empty_cache()

    refs = {
        "model": file_ref(model_path),
        "track_emb": file_ref(track_emb_path),
        "track_features": file_ref(track_features_path),
    }
    if query_cache.exists():
        refs["query_cache"] = file_ref(query_cache)
    cand, sizes, scores = pad_scored(rows, top_k)
    return cand, sizes, scores, refs


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
    out_dir = component_output_dir("retriever", name, config, target=target, fit_mode=fit_mode)
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
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "candidates.npz",
            track_idx=cand.astype(np.int32, copy=False),
            sizes=sizes.astype(np.int32, copy=False),
            keys=encode_keys([(f"{row['source_split']}:{row['session_id']}", int(row["turn_number"])) for row in public_rows]),
            source_split=np.asarray([str(row["source_split"]).encode("utf-8") for row in public_rows], dtype="S8"),
            folds=np.asarray([int(row["fold"]) for row in public_rows], dtype=np.int16),
            rank=rank,
            score__primary=scores.astype(np.float32, copy=False),
        )
        with (out_dir / "turns.jsonl").open("w", encoding="utf-8") as f:
            for row in public_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        json_dump(out_dir / "manifest.json", manifest)
        metrics = public_labeled_metrics(public_rows, track_index, cand, sizes)
        metrics.update({"artifact": str(out_dir.relative_to(REPO_ROOT)), "name": name, "config": config, "target": target, "artifact_mode": fit_mode})
        json_dump(component_results_dir("retriever", name, config, target=target, fit_mode=fit_mode) / "scores.json", metrics)
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
        metrics.update({"artifact": str(out_dir.relative_to(REPO_ROOT)), "name": name, "config": config, "target": target})
        json_dump(component_results_dir("retriever", name, config, target=target) / "scores.json", metrics)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=Path, default=Path("retriever/union/configs/union_v1.yaml"))
    parser.add_argument("--component-config-file", type=Path, default=None)
    parser.add_argument("--config", default="basic")
    parser.add_argument("--target", choices=("devset", "public_labeled", "blind_a", "blind_b"), default="devset")
    parser.add_argument("--split-dir", type=Path, default=REPO_ROOT / "artifacts/cache/splits/cv5")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--dense-batch-size", type=int, default=512)
    parser.add_argument("--dense-extra-k", type=int, default=64)
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--llm-batch-size", type=int, default=16)
    parser.add_argument("--hyde-max-new-tokens", type=int, default=80)
    parser.add_argument("--n-neigh", type=int, default=None)
    parser.add_argument("--rank-offset", type=float, default=None)
    parser.add_argument("--user-neighbor-score-mode", choices=("legacy_rank", "sim_weighted"), default=None)
    parser.add_argument("--skip-unsupported", action="store_true")
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    cfg = read_config(REPO_ROOT / args.config_file)
    sources = list(cfg["sources"])
    if args.only:
        only = set(args.only)
        sources = [s for s in sources if s in only]

    zoo = load_zoo_module()
    public_rows: list[dict[str, Any]] | None = None
    if args.target == "devset":
        examples = zoo.build_examples_devset()
    elif args.target == "public_labeled":
        split_dir = args.split_dir if args.split_dir.is_absolute() else REPO_ROOT / args.split_dir
        examples, public_rows = build_public_labeled_examples(zoo, split_dir)
    else:
        examples = build_blind_examples(zoo, args.target)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]
        if public_rows is not None:
            public_rows = public_rows[: args.max_examples]

    bm25_lookup = {name: (bm_name, mode) for name, bm_name, mode in zoo.BM25_RETRIEVERS}
    needed_bm25 = set()
    for name in sources:
        if name in bm25_lookup:
            needed_bm25.add(bm25_lookup[name][0])
        if name in {"tag_intent_bm25", "llm_track_names"}:
            needed_bm25.add("tag_list" if name == "tag_intent_bm25" else "4field")
        if name in {"hyde_bm25_5field"}:
            needed_bm25.add("5field")
    bm25_variants = tuple(v for v in zoo.BM25_VARIANTS if v[0] in needed_bm25)

    needed_dense = {n for n in zoo.DENSE_COLS if n in sources}
    for name, dense_key in zoo.SEMANTIC_DENSE_SOURCES.items():
        if name in sources:
            needed_dense.add(dense_key)
    if "hyde_dense_qwen3_metadata" in sources:
        needed_dense.add("dense_qwen3_metadata")
    needed_dense_tuple = tuple(n for n in zoo.DENSE_COLS if n in needed_dense)

    print(f"building basic retriever artifacts: {len(sources)} sources")
    track_index = zoo.build_track_index(bm25_variants, needed_dense_tuple)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    cooc = None
    if any(s in sources for s in {"cooc_track", "cooc_artist"}):
        cooc = zoo.build_cooc(track_index)
    personal_memory = None
    if any(s.startswith("personal_") for s in sources):
        personal_memory = zoo.build_personal_memory(track_index)
    user_vectors = zoo.load_user_vectors_normalized() if "user_emb_track_cf" in sources else {}

    encode_dir = REPO_ROOT / "output/086_retriever_zoo_v2/encode" if args.target == "devset" else REPO_ROOT / "artifacts/runs/retriever/_fit_free_encode" / args.target
    qwen_query_mat = None
    if any(s in sources for s in {"dense_qwen3_attributes", "dense_qwen3_lyrics", "dense_qwen3_metadata"}):
        qwen_query_mat = load_or_encode_qwen_queries(
            zoo,
            examples,
            args.target,
            encode_dir / f"qwen3_query_mat__n{len(examples)}.npy",
            args.encode_batch_size,
        )
    semantic_query_mat = None
    if any(s in sources for s in set(zoo.SEMANTIC_DENSE_SOURCES)):
        semantic_query_mat = zoo.load_or_encode_semantic_queries(
            examples,
            encode_dir / f"semantic_qwen3_query_goal_current__n{len(examples)}.npz",
            batch_size=args.encode_batch_size,
        )
    clap_query_mat = None
    if "dense_clap_audio" in sources:
        clap_query_mat = zoo.encode_queries_clap(
            examples,
            encode_dir / f"clap_query_mat__n{len(examples)}.npy",
            batch_size=args.encode_batch_size,
        )
    siglip_query_mat = None
    if "dense_siglip_image" in sources:
        siglip_query_mat = zoo.encode_queries_siglip(
            examples,
            encode_dir / f"siglip_query_mat__n{len(examples)}.npy",
            batch_size=args.encode_batch_size,
        )
    hyde_rewrites = None
    if any(s in sources for s in {"hyde_bm25_5field", "hyde_dense_qwen3_metadata"}):
        hyde_rewrites = load_or_generate_hyde_rewrites(
            zoo,
            examples,
            track_index,
            args.target,
            encode_dir,
            args.llm_model,
            args.llm_batch_size,
            args.hyde_max_new_tokens,
        )
    album_mat = None
    if "album_qwen3_history" in sources:
        album_mat = zoo._build_album_qwen3_matrix(
            track_index,
            album_qwen3_cache_path(args.target, encode_dir),
            batch_size=args.encode_batch_size,
        )

    refs = {
        "bridge": file_ref(REPO_ROOT / "scripts/build_basic_retrievers.py"),
        "zoo": file_ref(REPO_ROOT / "src/recsys2026/zoo.py"),
        "config": file_ref(REPO_ROOT / args.config_file),
    }
    if args.component_config_file is not None:
        refs["component_config"] = file_ref(REPO_ROOT / args.component_config_file)
    producer_command = ["uv", "run", "python", "scripts/build_basic_retrievers.py", *sys.argv[1:]]
    run_params = {
        "config": args.config,
        "target": args.target,
        "top_k": args.top_k,
        "only": args.only,
        "device": device,
        "encode_batch_size": args.encode_batch_size,
        "dense_batch_size": args.dense_batch_size,
        "dense_extra_k": args.dense_extra_k,
        "llm_model": args.llm_model,
        "llm_batch_size": args.llm_batch_size,
        "hyde_max_new_tokens": args.hyde_max_new_tokens,
        "n_neigh": args.n_neigh,
        "rank_offset": args.rank_offset,
        "user_neighbor_score_mode": args.user_neighbor_score_mode,
        "config_file": str(args.config_file),
        "component_config_file": str(args.component_config_file) if args.component_config_file else None,
    }

    count_sources = {
        "history_artist": zoo._score_history_artist,
        "history_album": zoo._score_history_album,
        "last_music_artist": zoo._score_last_music_artist,
        "last_music_album": zoo._score_last_music_album,
        "history_primary_tag": zoo._score_history_primary_tag,
        "current_artist_catalog_source": zoo._score_current_artist_catalog_source,
        "exact_title_artist_source": zoo._score_exact_title_artist_source,
        "exact_album_artist_source": zoo._score_exact_album_artist_source,
        "secondary_artist_source": zoo._score_secondary_artist_source,
    }

    for name in sources:
        t0 = time.time()
        print(f"\n=== {name} ===")
        source_refs = refs
        source_policy = source_policy_from_config(cfg, name)
        if args.target != "devset" and bool(source_policy.get("requires_labeled_fit", False)):
            msg = f"{name} requires labeled fit; use train-fit/OOF builders for target={args.target}"
            if args.skip_unsupported:
                print(f"[skip] {msg}")
                continue
            raise SystemExit(msg)
        fit_mode = "fit_free_all_rows" if args.target != "devset" else None
        if name in bm25_lookup:
            cand, sizes, scores = bm25_scored(zoo, examples, track_index, *bm25_lookup[name], args.top_k)
        elif name == "history_release_decade":
            cand, sizes, scores = history_release_decade_scored(zoo, examples, track_index, args.top_k)
        elif name in count_sources:
            cand, sizes, scores = count_scored(zoo, examples, track_index, count_sources[name], name, args.top_k)
        elif name == "personal_exact_repeat":
            cand, sizes, scores = count_scored(zoo, examples, track_index, lambda ex, ti: zoo._score_personal_exact_repeat(ex, ti, personal_memory), name, args.top_k)
        elif name == "personal_artist_expansion":
            cand, sizes, scores = count_scored(zoo, examples, track_index, lambda ex, ti: zoo._score_personal_artist_expansion(ex, ti, personal_memory), name, args.top_k)
        elif name == "personal_album_expansion":
            cand, sizes, scores = count_scored(zoo, examples, track_index, lambda ex, ti: zoo._score_personal_album_expansion(ex, ti, personal_memory), name, args.top_k)
        elif name == "popularity_global":
            cand, sizes, scores = popularity_global_scored(zoo, examples, track_index, args.top_k)
        elif name == "train_play_count_unique_users":
            cand, sizes, scores = train_play_count_unique_users_scored(zoo, examples, track_index, args.top_k)
        elif name == "cf_history_centroid":
            cand, sizes, scores = cf_history_centroid_scored(zoo, examples, track_index, args.top_k)
        elif name == "user_emb_track_cf":
            cand, sizes, scores = user_emb_track_cf_scored(zoo, examples, track_index, user_vectors, args.top_k)
        elif name == "collab_artist_expansion":
            cand, sizes, scores = collab_artist_expansion_scored(zoo, examples, track_index, args.top_k)
        elif name == "cooc_track":
            cand, sizes, scores = cooc_track_scored(zoo, examples, track_index, cooc, args.top_k)
        elif name == "cooc_artist":
            cand, sizes, scores = cooc_artist_scored(zoo, examples, track_index, cooc, args.top_k)
        elif name == "train_neighbor":
            n_neigh = args.n_neigh if args.n_neigh is not None else 500
            rank_offset = args.rank_offset if args.rank_offset is not None else 10.0
            cand, sizes, scores = train_neighbor_scored(zoo, examples, track_index, args.top_k, n_neigh, rank_offset, device)
        elif name == "user_neighbor":
            n_neigh = args.n_neigh if args.n_neigh is not None else 500
            rank_offset = args.rank_offset if args.rank_offset is not None else 10.0
            score_mode = args.user_neighbor_score_mode or "sim_weighted"
            cand, sizes, scores = user_neighbor_scored(
                zoo, examples, track_index, args.top_k, n_neigh, rank_offset, device, score_mode
            )
        elif name in {"dense_qwen3_attributes", "dense_qwen3_lyrics", "dense_qwen3_metadata"}:
            cand, sizes, scores = dense_scored(
                zoo,
                examples,
                track_index,
                name,
                qwen_query_mat,
                args.top_k,
                name,
                device=device,
                batch_size=args.dense_batch_size,
                extra_k=args.dense_extra_k,
            )
        elif name == "dense_clap_audio":
            cand, sizes, scores = dense_scored(
                zoo,
                examples,
                track_index,
                name,
                clap_query_mat,
                args.top_k,
                name,
                device=device,
                batch_size=args.dense_batch_size,
                extra_k=args.dense_extra_k,
            )
        elif name == "dense_siglip_image":
            cand, sizes, scores = dense_scored(
                zoo,
                examples,
                track_index,
                name,
                siglip_query_mat,
                args.top_k,
                name,
                device=device,
                batch_size=args.dense_batch_size,
                extra_k=args.dense_extra_k,
            )
        elif name in zoo.SEMANTIC_DENSE_SOURCES:
            dense_key = zoo.SEMANTIC_DENSE_SOURCES[name]
            cand, sizes, scores = dense_scored(
                zoo,
                examples,
                track_index,
                dense_key,
                semantic_query_mat,
                args.top_k,
                name,
                device=device,
                batch_size=args.dense_batch_size,
                extra_k=args.dense_extra_k,
            )
        elif name == "tag_intent_bm25":
            queries = [zoo._tag_intent_query(ex) for ex in examples]
            cand, sizes, scores = bm25_queries_scored(zoo, examples, track_index, "tag_list", queries, args.top_k, name)
        elif name == "hyde_bm25_5field":
            cand, sizes, scores = bm25_queries_scored(zoo, examples, track_index, "5field", hyde_rewrites, args.top_k, name)
        elif name == "hyde_dense_qwen3_metadata":
            hyde_query = load_or_encode_qwen_texts(
                zoo,
                hyde_rewrites,
                encode_dir / f"hyde_qwen3_query__n{len(examples)}.npy",
                args.encode_batch_size,
                "hyde rewrites",
            )
            cand, sizes, scores = dense_scored(
                zoo,
                examples,
                track_index,
                "dense_qwen3_metadata",
                hyde_query,
                args.top_k,
                name,
                device=device,
                batch_size=args.dense_batch_size,
                extra_k=args.dense_extra_k,
            )
        elif name == "album_qwen3_history":
            cand, sizes, scores = album_qwen3_history_scored(
                zoo,
                examples,
                track_index,
                album_mat,
                args.top_k,
                device=device,
                batch_size=args.dense_batch_size,
                extra_k=args.dense_extra_k,
            )
        elif name == "llm_track_names":
            cand, sizes, scores = llm_track_names_scored(zoo, examples, track_index, args.top_k, k_per_song=50)
        elif name == "user_culture_match":
            cand, sizes, scores = user_culture_match_scored(zoo, examples, track_index, args.top_k)
        elif name == "two_tower_lora_thought":
            out_dir = component_output_dir("retriever", name, args.config, target=args.target)
            cand, sizes, scores, two_tower_refs = two_tower_lora_thought_scored(
                zoo, examples, track_index, args.top_k, device, out_dir
            )
            source_refs = {**refs, "two_tower_113_artifacts": two_tower_refs}
        else:
            raise SystemExit(f"unsupported source: {name}")
        elapsed = time.time() - t0
        save_source(
            name,
            args.config,
            args.target,
            cand,
            sizes,
            scores,
            source_refs,
            elapsed,
            producer_command,
            run_params,
            source_policy,
            fit_mode=fit_mode,
            public_rows=public_rows,
            track_index=track_index,
        )
        print(f"saved {name}: mean_size={sizes.mean():.1f}, elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
