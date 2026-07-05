#!/usr/bin/env python3
"""Build an ordered-union candidate artifact from retriever artifacts."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

from recsys2026.artifacts import (
    component_output_dir,
    component_results_dir,
    decode_keys,
    encode_keys,
    file_ref,
    json_dump,
    load_candidate_artifact,
    save_candidate_artifact,
    utc_now,
)
from recsys2026.data import load
from recsys2026.paths import REPO_ROOT
from recsys2026.retriever_eval import candidate_metrics, devset_gold_indices


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_")


def parse_source(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        name, path = raw.split("=", 1)
        return safe_name(name), REPO_ROOT / path
    path = REPO_ROOT / raw
    # artifacts/runs/retriever/<name>/<config>/<target>
    parts = path.parts
    if len(parts) >= 4:
        return safe_name(parts[-3]), path
    return safe_name(path.name), path


def target_value(value: object, target: str) -> object | None:
    if isinstance(value, dict):
        return value.get(target) or value.get("default")
    return value


def source_policy_from_config(cfg: dict, source: str) -> dict:
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


def default_artifact_mode(policy: dict, target: str) -> str:
    if target == "public_labeled":
        return str(policy.get("preferred_train_row_artifact_mode") or "cv3_oof")
    return str(policy.get("preferred_inference_artifact_mode") or "full_public")


def resolved_artifact_from_entry(
    entry: dict,
    *,
    name: str,
    source_config: str,
    target: str,
    policy: dict,
) -> tuple[str, dict]:
    """Resolve a source entry to an artifact path.

    Preferred schema:
      name: short source alias used in union features
      component: retriever component/artifact name
      config: retriever component config
      artifact_mode: optional target-specific mode override

    Legacy/escape hatch:
      artifact: explicit path template or target-specific path mapping.
    """
    artifact_template = target_value(entry.get("artifact"), target)
    if artifact_template is not None:
        return str(artifact_template).format(target=target), entry

    component = str(entry.get("component") or entry.get("retriever") or name)
    config = str(entry.get("config") or source_config)
    artifact_mode = target_value(
        entry.get("artifact_mode")
        or entry.get("mode")
        or entry.get("artifact_modes"),
        target,
    )
    if artifact_mode is None:
        artifact_mode = default_artifact_mode(policy, target)
    entry = {**entry, "component": component, "config": config, "artifact_mode": str(artifact_mode)}
    return f"artifacts/runs/retriever/{component}/{config}/{artifact_mode}/{target}", entry


def union_fit_scope(source_refs: list[dict]) -> dict:
    policies = [ref.get("source_policy") or {} for ref in source_refs]
    requires_labeled_fit = any(bool(p.get("requires_labeled_fit", False)) for p in policies)
    fold_split_required = any(
        bool(p.get("fold_split_required_for_reranker_train", False)) for p in policies
    )
    fit_splits = sorted(
        {
            split
            for ref in source_refs
            for split in ((ref.get("source_manifest_fit_scope") or {}).get("fit_splits") or [])
        }
    )
    train_row_policies = {
        ref["name"]: (ref.get("source_policy") or {}).get("train_row_policy", "safe_in_sample")
        for ref in source_refs
    }
    return {
        "fit_mode": "composed_from_sources",
        "fit_splits": fit_splits,
        "requires_labeled_fit": requires_labeled_fit,
        "train_row_policy": "mixed" if requires_labeled_fit else "safe_in_sample",
        "train_row_policies_by_source": train_row_policies,
        "fold_split_required_for_reranker_train": fold_split_required,
        "uses_devset_for_fit": any(
            bool((ref.get("source_manifest_fit_scope") or {}).get("uses_devset_for_fit", False))
            for ref in source_refs
        ),
        "uses_blind_for_fit": any(
            bool((ref.get("source_manifest_fit_scope") or {}).get("uses_blind_for_fit", False))
            for ref in source_refs
        ),
    }


def load_sources(raw_sources: list[str]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for raw in raw_sources:
        name, path = parse_source(raw)
        if name in seen:
            raise ValueError(f"duplicate source name: {name}")
        seen.add(name)
        arrays, manifest = load_candidate_artifact(path)
        out.append({"name": name, "path": path, "arrays": arrays, "manifest": manifest})
    if not out:
        raise ValueError("at least one --source is required")
    keys = decode_keys(out[0]["arrays"]["keys"])
    for source in out[1:]:
        if decode_keys(source["arrays"]["keys"]) != keys:
            raise ValueError(f"source row keys do not align: {source['name']}")
    return out


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def public_labeled_metrics(turn_rows: list[dict], cand: np.ndarray, sizes: np.ndarray) -> dict:
    tracks = list(load("track", split="all_tracks"))
    id_to_idx = {str(row["track_id"]): i for i, row in enumerate(tracks)}
    gold = np.asarray([id_to_idx.get(str(row.get("gold_track_id")), -1) for row in turn_rows], dtype=np.int32)
    out: dict[str, object] = {
        "n_examples": int(len(turn_rows)),
        "mean_size": float(sizes.mean()) if len(sizes) else 0.0,
    }
    groups = {
        "all": np.arange(len(turn_rows), dtype=np.int32),
        "train": np.asarray([i for i, row in enumerate(turn_rows) if row.get("source_split") == "train"], dtype=np.int32),
        "devset": np.asarray([i for i, row in enumerate(turn_rows) if row.get("source_split") == "devset"], dtype=np.int32),
    }
    for name, idx in groups.items():
        if len(idx) == 0:
            continue
        prefix = "" if name == "all" else f"{name}_"
        out[f"{prefix}n_examples"] = int(len(idx))
        out[f"{prefix}mean_size"] = float(sizes[idx].mean())
        for k in (20, 50, 100, 200, 500):
            kk = min(k, cand.shape[1])
            out[f"{prefix}recall@{k}"] = float((cand[idx, :kk] == gold[idx, None]).any(axis=1).mean())
        out[f"{prefix}recall@all"] = float(
            np.asarray([bool((cand[row_i, : int(sizes[row_i])] == gold[row_i]).any()) for row_i in idx], dtype=bool).mean()
        )
    return out


def split_rows_for_artifact(split_dir: Path, source_keys: np.ndarray) -> list[dict]:
    rows = read_jsonl(split_dir / "rows.jsonl")
    expected = encode_keys(
        [
            (f"{row['source_split']}:{row['session_id']}", int(row["turn_number"]))
            for row in rows
        ]
    )
    if expected.shape != source_keys.shape or not np.array_equal(expected, source_keys):
        raise ValueError(f"split rows do not align with source artifact keys: {split_dir}")
    return rows


def save_public_labeled_union(
    out_dir: Path,
    first_source: dict,
    track_idx: np.ndarray,
    sizes: np.ndarray,
    manifest: dict,
    split_rows: list[dict] | None = None,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    first_arrays = first_source["arrays"]
    turn_rows = split_rows if split_rows is not None else read_jsonl(first_source["path"] / "turns.jsonl")
    arrays: dict[str, np.ndarray] = {
        "track_idx": track_idx.astype(np.int32, copy=False),
        "sizes": sizes.astype(np.int32, copy=False),
        "keys": np.asarray(first_arrays["keys"]),
    }
    if split_rows is not None:
        arrays["source_split"] = np.asarray([str(row["source_split"]).encode("utf-8") for row in split_rows], dtype="S8")
        arrays["folds"] = np.asarray([int(row["fold"]) for row in split_rows], dtype=np.int16)
    for key in ("source_split", "folds"):
        if key in first_arrays:
            arrays.setdefault(key, np.asarray(first_arrays[key]))
    np.savez_compressed(out_dir / "candidates.npz", **arrays)
    with (out_dir / "turns.jsonl").open("w", encoding="utf-8") as f:
        for row in turn_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    json_dump(out_dir / "manifest.json", manifest)
    return turn_rows


def build_union(
    sources: list[dict],
    max_candidates: int | None,
    *,
    include_source_features: bool = True,
    merge_strategy: str = "ordered_unique",
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    n = sources[0]["arrays"]["track_idx"].shape[0]
    union_rows: list[list[int]] = []
    max_width = 0

    def source_row_size(source: dict, row_i: int) -> int:
        size = int(source["arrays"]["sizes"][row_i])
        limit = source.get("max_candidates")
        if limit is not None:
            size = min(size, int(limit))
        return size

    def source_candidate_allowed(source: dict, row_i: int, rank0: int) -> bool:
        min_score = source.get("min_score")
        if min_score is None:
            return True
        score_field = str(source.get("score_field") or "score__primary")
        score_arr = source["arrays"].get(score_field)
        if score_arr is None:
            return False
        score = float(score_arr[row_i, rank0])
        return np.isfinite(score) and score >= float(min_score)

    for i in range(n):
        row: list[int] = []
        seen: set[int] = set()
        if merge_strategy == "ordered_unique":
            for source in sources:
                arr = source["arrays"]
                cand = arr["track_idx"][i]
                size = source_row_size(source, i)
                for rank0, tid_raw in enumerate(cand[:size]):
                    if not source_candidate_allowed(source, i, rank0):
                        continue
                    tid = int(tid_raw)
                    if tid not in seen and (max_candidates is None or len(row) < max_candidates):
                        seen.add(tid)
                        row.append(tid)
        elif merge_strategy == "round_robin":
            max_source_size = max(source_row_size(source, i) for source in sources)
            for rank0 in range(max_source_size):
                for source in sources:
                    arr = source["arrays"]
                    size = source_row_size(source, i)
                    if rank0 >= size:
                        continue
                    if not source_candidate_allowed(source, i, rank0):
                        continue
                    tid = int(arr["track_idx"][i, rank0])
                    if tid not in seen and (max_candidates is None or len(row) < max_candidates):
                        seen.add(tid)
                        row.append(tid)
                if max_candidates is not None and len(row) >= max_candidates:
                    break
        else:
            raise ValueError(f"unknown merge strategy: {merge_strategy}")
        union_rows.append(row)
        max_width = max(max_width, len(row))

    track_idx = np.full((n, max_width), -1, dtype=np.int32)
    sizes = np.zeros(n, dtype=np.int32)
    for i, row in enumerate(union_rows):
        sizes[i] = len(row)
        if row:
            track_idx[i, : len(row)] = np.asarray(row, dtype=np.int32)

    if not include_source_features:
        return track_idx, sizes, {}

    source_features: dict[str, np.ndarray] = {}
    source_count = np.zeros((n, max_width), dtype=np.uint16)
    best_rank = np.full((n, max_width), -1, dtype=np.int32)
    mean_rank = np.full((n, max_width), np.nan, dtype=np.float32)
    source_field_keys: dict[str, list[str]] = {}
    for source in sources:
        arr = source["arrays"]
        source_width = arr["track_idx"].shape[1]
        fields: list[str] = []
        for key, value in arr.items():
            if key in {"track_idx", "sizes", "keys", "source_split", "folds"}:
                continue
            value_arr = np.asarray(value)
            if value_arr.ndim == 2 and value_arr.shape[0] == n and value_arr.shape[1] == source_width:
                fields.append(key)
        source_field_keys[source["name"]] = sorted(fields)
    has_primary = {source["name"]: "score__primary" in source_field_keys[source["name"]] for source in sources}
    max_primary = np.full((n, max_width), np.nan, dtype=np.float32) if any(has_primary.values()) else None

    for source in sources:
        prefix = f"src__{source['name']}"
        source_features[f"{prefix}__present"] = np.zeros((n, max_width), dtype=np.uint8)
        for field in source_field_keys[source["name"]]:
            raw = np.asarray(source["arrays"][field])
            if np.issubdtype(raw.dtype, np.integer):
                fill = -1
                dtype = raw.dtype
            elif np.issubdtype(raw.dtype, np.bool_):
                fill = 0
                dtype = np.uint8
            else:
                fill = np.nan
                dtype = np.float32
            source_features[f"{prefix}__{field}"] = np.full((n, max_width), fill, dtype=dtype)

    for i, row in enumerate(union_rows):
        pos_by_tid = {tid: j for j, tid in enumerate(row)}
        rank_sums = np.zeros(len(row), dtype=np.float32)
        for source in sources:
            arr = source["arrays"]
            cand = arr["track_idx"][i]
            size = source_row_size(source, i)
            primary = arr.get("score__primary")
            prefix = f"src__{source['name']}"
            present_arr = source_features[f"{prefix}__present"]
            field_arrays = {
                field: source_features[f"{prefix}__{field}"]
                for field in source_field_keys[source["name"]]
            }
            seen_in_source: set[int] = set()
            for rank0, tid_raw in enumerate(cand[:size]):
                if not source_candidate_allowed(source, i, rank0):
                    continue
                rank = rank0 + 1
                tid = int(tid_raw)
                if tid in seen_in_source:
                    continue
                seen_in_source.add(tid)
                j = pos_by_tid.get(tid)
                if j is None:
                    continue
                present_arr[i, j] = 1
                source_count[i, j] += 1
                rank_sums[j] += rank
                if best_rank[i, j] < 0 or rank < best_rank[i, j]:
                    best_rank[i, j] = rank
                for field, out_arr in field_arrays.items():
                    out_arr[i, j] = arr[field][i, rank - 1]
                if primary is not None:
                    score = float(primary[i, rank - 1])
                    if max_primary is not None and not np.isnan(score):
                        if np.isnan(max_primary[i, j]) or score > max_primary[i, j]:
                            max_primary[i, j] = score
        if row:
            valid = source_count[i, : len(row)] > 0
            mean_rank[i, : len(row)][valid] = rank_sums[valid] / source_count[i, : len(row)][valid]

    source_features["meta__source_count"] = source_count
    source_features["meta__best_source_rank"] = best_rank
    source_features["meta__mean_source_rank"] = mean_rank
    if max_primary is not None:
        source_features["meta__max_source_score__primary"] = max_primary
    return track_idx, sizes, source_features


def source_args_from_config(config_file: Path, target: str) -> tuple[list[str], dict, dict[str, dict]]:
    with config_file.open() as f:
        cfg = yaml.safe_load(f) or {}
    source_config = str(cfg.get("source_config", "legacy"))
    raw_sources = cfg.get("sources") or []
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"{config_file} must contain a non-empty sources list")
    source_metadata = cfg.get("source_metadata") or {}
    if source_metadata is None:
        source_metadata = {}
    if not isinstance(source_metadata, dict):
        raise TypeError(f"{config_file} source_metadata must be a mapping when present")

    out: list[str] = []
    entries: dict[str, dict] = {}
    for raw in raw_sources:
        if isinstance(raw, str):
            name = raw
            artifact = f"artifacts/runs/retriever/{name}/{source_config}/{target}"
            policy = source_policy_from_config(cfg, name)
            entry = {"name": name, "config": source_config, "source_policy": policy, **dict(source_metadata.get(name) or {})}
        elif isinstance(raw, dict):
            name = str(raw["name"])
            policy = {**source_policy_from_config(cfg, name), **dict(raw.get("source_policy") or {})}
            entry0 = {**dict(source_metadata.get(name) or {}), **dict(raw), "source_policy": policy}
            artifact, entry = resolved_artifact_from_entry(
                entry0,
                name=name,
                source_config=source_config,
                target=target,
                policy=policy,
            )
        else:
            raise TypeError(f"invalid source entry in {config_file}: {raw!r}")
        entry["artifact"] = artifact
        entries[name] = entry
        out.append(f"{name}={artifact}")
    return out, cfg, entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="union")
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-file", type=Path, default=None)
    parser.add_argument("--target", choices=("devset", "public_labeled", "blind_a", "blind_b"), default="devset")
    parser.add_argument("--source", action="append", default=[], help="name=artifact_dir or artifact_dir; may repeat")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--merge-strategy", choices=("ordered_unique", "round_robin"), default="ordered_unique")
    parser.add_argument("--no-source-features", action="store_true")
    parser.add_argument("--compress", action="store_true")
    parser.add_argument("--split-dir", type=Path, default=None, help="Override public_labeled folds/turn rows from this split artifact.")
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    config_body = None
    config_file_ref = None
    source_args = list(args.source)
    source_config_entries: dict[str, dict] = {}
    if args.config_file is not None:
        cfg_sources, config_body, source_config_entries = source_args_from_config(REPO_ROOT / args.config_file, args.target)
        source_args = cfg_sources + source_args
        config_file_ref = file_ref(REPO_ROOT / args.config_file)
        union_rule_cfg = config_body.get("union_rule") or {}
        if args.max_candidates is None and "max_candidates" in union_rule_cfg:
            args.max_candidates = union_rule_cfg["max_candidates"]
        if args.split_dir is None and config_body.get("split_dir"):
            args.split_dir = Path(str(config_body["split_dir"]))
    if not source_args:
        raise ValueError("at least one --source or --config-file source is required")

    sources = load_sources(source_args)
    for source in sources:
        entry = source_config_entries.get(source["name"]) or {}
        if "max_candidates" in entry:
            source["max_candidates"] = int(entry["max_candidates"])
        elif "source_max_candidates" in entry:
            source["max_candidates"] = int(entry["source_max_candidates"])
        threshold = entry.get("min_score")
        if threshold is None:
            threshold = entry.get("score_threshold")
        if threshold is not None:
            source["min_score"] = float(threshold)
            source["score_field"] = str(entry.get("score_field") or "score__primary")
    track_idx, sizes, source_features = build_union(
        sources,
        args.max_candidates,
        include_source_features=not args.no_source_features,
        merge_strategy=args.merge_strategy,
    )
    out_dir = component_output_dir("retriever", args.name, args.config, target=args.target)
    source_refs = []
    for source in sources:
        path = source["path"]
        ref = {
            "name": source["name"],
            "artifact": str(path.relative_to(REPO_ROOT)),
            "candidates": file_ref(path / "candidates.npz"),
        }
        source_manifest_fit_scope = source.get("manifest", {}).get("fit_scope")
        if source_manifest_fit_scope is not None:
            ref["source_manifest_fit_scope"] = source_manifest_fit_scope
        source_manifest_policy = source.get("manifest", {}).get("source_policy")
        if source_manifest_policy is not None:
            ref["source_manifest_policy"] = source_manifest_policy
        if source["name"] in source_config_entries:
            ref["config_entry"] = source_config_entries[source["name"]]
            ref["source_policy"] = source_config_entries[source["name"]].get("source_policy")
        elif source_manifest_policy is not None:
            ref["source_policy"] = source_manifest_policy
        source_refs.append(ref)
    manifest = {
        "schema_version": 1,
        "artifact_type": "candidates",
        "stage": "retriever",
        "name": args.name,
        "config": args.config,
        "target": args.target,
        "created_at": utc_now(),
        "producer": {"command": ["uv", "run", "python", "scripts/build_union_candidates.py"], "cwd": "."},
        "config_file": config_file_ref,
        "config_body": config_body,
        "split_artifact": None,
        "source_artifacts": source_refs,
        "union_rule": {
            "type": args.merge_strategy,
            "source_order": [s["name"] for s in sources],
            "max_candidates": args.max_candidates,
            "source_max_candidates": {
                s["name"]: int(s["max_candidates"])
                for s in sources
                if s.get("max_candidates") is not None
            },
            "source_score_thresholds": {
                s["name"]: {
                    "field": str(s.get("score_field") or "score__primary"),
                    "min_score": float(s["min_score"]),
                }
                for s in sources
                if s.get("min_score") is not None
            },
            "tie_breaker": "source_order_then_source_rank" if args.merge_strategy == "ordered_unique" else "round_rank_then_source_order",
        },
        "aligned_feature_pack": None if args.no_source_features else "source_features.npz",
        "aligned_feature_schema": None
        if args.no_source_features
        else {
            "format": "dense_npz_aligned_to_candidates",
            "missing_float": "nan",
            "missing_integer": -1,
            "source_prefix": "src__<source_name>__<field_name>",
            "source_field_rule": "All 2D per-candidate arrays from each source artifact are propagated, not only score__primary.",
            "meta_fields": [
                "meta__source_count",
                "meta__best_source_rank",
                "meta__mean_source_rank",
                "meta__max_source_score__primary",
            ],
        },
        "fit_scope": union_fit_scope(source_refs),
        "leak_check": {
            "uses_track_emb_test_tracks": False,
            "uses_target_future_turns": False,
            "same_user_memory_date_censored": None,
            "popularity_tiebreaker": False,
        },
        "candidate_universe": "union_of_sources",
        "retention": "standard",
    }
    if args.target == "public_labeled":
        split_rows = None
        if args.split_dir is not None:
            split_dir = args.split_dir if args.split_dir.is_absolute() else REPO_ROOT / args.split_dir
            split_rows = split_rows_for_artifact(split_dir, np.asarray(sources[0]["arrays"]["keys"]))
            manifest["split_artifact"] = str(split_dir.relative_to(REPO_ROOT))
        turn_rows = save_public_labeled_union(out_dir, sources[0], track_idx, sizes, manifest, split_rows=split_rows)
    else:
        save_candidate_artifact(
            out_dir,
            track_idx,
            sizes,
            target=args.target,
            manifest=manifest,
            compress=args.compress,
        )
    if not args.no_source_features:
        np.savez(out_dir / "source_features.npz", **source_features)
    if args.target == "devset":
        metrics = candidate_metrics(track_idx, sizes, devset_gold_indices())
        metrics.update({"artifact": str(out_dir.relative_to(REPO_ROOT)), "name": args.name, "config": args.config, "target": args.target})
        res_dir = component_results_dir("retriever", args.name, args.config, target=args.target)
        json_dump(res_dir / "scores.json", metrics)
        print(json.dumps(metrics, indent=2))
    elif args.target == "public_labeled":
        metrics = public_labeled_metrics(turn_rows, track_idx, sizes)
        metrics.update({"artifact": str(out_dir.relative_to(REPO_ROOT)), "name": args.name, "config": args.config, "target": args.target})
        res_dir = component_results_dir("retriever", args.name, args.config, target=args.target)
        json_dump(res_dir / "scores.json", metrics)
        print(json.dumps(metrics, indent=2))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
