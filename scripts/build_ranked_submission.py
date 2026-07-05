#!/usr/bin/env python3
"""Build a local Codabench zip from a component reranker ranked artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from recsys2026.artifacts import decode_keys, file_ref, json_dump
from recsys2026.data import load
from recsys2026.paths import REPO_ROOT
from recsys2026.submission import Target, format_record, iter_inputs, validate_predictions, zip_submission


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def first_text(value: object) -> str:
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value or "")


def load_track_rows() -> tuple[list[str], list[dict[str, str]]]:
    rows = list(load("track", split="all_tracks"))
    ids = [str(row["track_id"]) for row in rows]
    meta = [
        {
            "track_name": first_text(row.get("track_name")) or "this track",
            "artist_name": first_text(row.get("artist_name")) or "the artist",
        }
        for row in rows
    ]
    return ids, meta


def response_for(inp: Any, top_meta: dict[str, str]) -> str:
    query = " ".join(str(inp.user_query or "").split())[:120]
    track = top_meta["track_name"]
    artist = top_meta["artist_name"]
    if query:
        return f'I would start with "{track}" by {artist}. It matches the request you just made: {query}'
    return f'I would start with "{track}" by {artist}. It is the strongest match from the ranked track candidates.'


def build_records(ranked_dir: Path, target: Target, top_k: int) -> list[dict[str, Any]]:
    npz_path = ranked_dir / "ranked.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        ranked = np.asarray(data["track_idx"], dtype=np.int32)
        keys = decode_keys(data["keys"])
    inputs = list(iter_inputs(target))
    expected_keys = [(inp.session_id, int(inp.turn_number)) for inp in inputs]
    if keys != expected_keys:
        raise ValueError(f"ranked artifact keys do not align with target={target}")
    track_ids, track_meta = load_track_rows()
    records: list[dict[str, Any]] = []
    for inp, row in zip(inputs, ranked, strict=True):
        tids: list[str] = []
        seen: set[str] = set()
        top_meta = {"track_name": "this track", "artist_name": "the artist"}
        for idx_raw in row:
            idx = int(idx_raw)
            if idx < 0:
                continue
            tid = track_ids[idx]
            if tid in seen:
                continue
            if not tids:
                top_meta = track_meta[idx]
            seen.add(tid)
            tids.append(tid)
            if len(tids) >= top_k:
                break
        records.append(format_record(inp, tids, response_for(inp, top_meta)))
    validate_predictions(records, target)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranked-artifact", type=Path, required=True)
    parser.add_argument("--target", choices=("blind_a", "blind_b"), default="blind_b")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    ranked_dir = args.ranked_artifact if args.ranked_artifact.is_absolute() else REPO_ROOT / args.ranked_artifact
    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records = build_records(ranked_dir, args.target, args.top_k)
    pred_json = out_dir / "prediction.json"
    pred_json.write_text(json.dumps(records, ensure_ascii=False))
    zip_path = zip_submission(pred_json)
    manifest = {
        "schema_version": 1,
        "artifact_type": "predictions",
        "stage": "responder",
        "name": "ranked_template",
        "config": out_dir.name,
        "target": args.target,
        "ranked_artifact": rel(ranked_dir),
        "ranked_file": file_ref(ranked_dir / "ranked.npz"),
        "top_k": int(args.top_k),
        "outputs": {"json": rel(pred_json), "zip": rel(zip_path)},
        "submission_note": "Local submission file only; not submitted to Codabench.",
    }
    json_dump(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest["outputs"], indent=2))


if __name__ == "__main__":
    main()
