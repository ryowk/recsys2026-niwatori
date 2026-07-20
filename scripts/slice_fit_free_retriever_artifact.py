#!/usr/bin/env python3
"""Slice a fit-free public artifact into train-row and devset artifacts.

The source retrievers are independent of labeled outcomes, so their existing
train+devset candidate arrays can be reused without refitting.  This script
only changes row scope and assigns the fixed train-only folds used by the
paper protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from recsys2026.artifacts import (
    artifact_complete,
    assert_target_alignment,
    decode_keys,
    encode_keys,
    file_ref,
    save_npz_artifact,
    utc_now,
)
from recsys2026.paths import REPO_ROOT
from recsys2026.splits import read_jsonl


def raw_key(source_split: str, session_id: str, turn: int) -> str:
    return f"{source_split}:{session_id}:{turn}"


def read_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    return json.loads(manifest_path.read_text()) if manifest_path.exists() else {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-split", choices=("train", "devset"), required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    args = parser.parse_args([arg for arg in sys.argv[1:] if arg != "--"])

    input_dir = args.input if args.input.is_absolute() else REPO_ROOT / args.input
    output_dir = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    split_dir = (
        args.split_dir if args.split_dir.is_absolute() else REPO_ROOT / args.split_dir
    )
    if artifact_complete(output_dir, "candidates.npz", "turns.jsonl"):
        print(f"[skip] {output_dir}")
        return

    with np.load(input_dir / "candidates.npz", allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]) for name in source.files}
    source_keys = [bytes(value).decode("utf-8") for value in arrays["keys"]]
    prefix = f"{args.source_split}:"
    selected = np.asarray(
        [i for i, key in enumerate(source_keys) if key.startswith(prefix)],
        dtype=np.int32,
    )
    if not len(selected):
        raise ValueError(f"no rows with prefix {prefix!r} in {input_dir}")

    split_rows = [
        row
        for row in read_jsonl(split_dir / "rows.jsonl")
        if row["source_split"] == args.source_split
    ]
    if args.source_split == "train":
        expected_raw = [
            raw_key("train", str(row["session_id"]), int(row["turn_number"]))
            for row in split_rows
        ]
        selected_keys = [source_keys[int(i)] for i in selected]
        if selected_keys != expected_raw:
            raise ValueError(
                "source train rows do not align with the fixed train-only split"
            )
        output_keys = encode_keys(
            [
                (f"train:{row['session_id']}", int(row["turn_number"]))
                for row in split_rows
            ]
        )
    else:
        output_keys = encode_keys(
            [decode_keys(np.asarray([arrays["keys"][int(i)]]))[0] for i in selected]
        )
        decoded = [
            (sid.removeprefix("devset:"), turn)
            for sid, turn in decode_keys(output_keys)
        ]
        output_keys = encode_keys(decoded)
        assert_target_alignment(decoded, "devset")

    output_arrays: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        if name in {"keys", "source_split", "folds"}:
            continue
        output_arrays[name] = values[selected]
    output_arrays["keys"] = output_keys
    if args.source_split == "train":
        output_arrays["source_split"] = np.asarray(
            [b"train"] * len(selected), dtype="S8"
        )
        output_arrays["folds"] = np.asarray(
            [int(row["fold"]) for row in split_rows], dtype=np.int16
        )

    source_turns = read_jsonl(input_dir / "turns.jsonl")
    selected_turns = [
        row for row in source_turns if row.get("source_split") == args.source_split
    ]
    if len(selected_turns) != len(selected):
        raise ValueError(
            f"turn row mismatch: candidates={len(selected)} turns={len(selected_turns)}"
        )
    if args.source_split == "train":
        turns = split_rows
    else:
        turns = [{**row, "row_id": i} for i, row in enumerate(selected_turns)]
    source_manifest = read_manifest(input_dir)
    manifest = {
        **source_manifest,
        "created_at": utc_now(),
        "artifact_mode": "fit_free_train5_dev",
        "target": "public_labeled" if args.source_split == "train" else "devset",
        "producer": {
            "command": [
                "uv",
                "run",
                "python",
                "scripts/slice_fit_free_retriever_artifact.py",
                *sys.argv[1:],
            ],
            "cwd": ".",
        },
        "source_artifact": file_ref(input_dir / "candidates.npz"),
        "split_artifact": str(split_dir.relative_to(REPO_ROOT)),
        "fit_scope": {
            "fit_mode": "fit_free",
            "fit_splits": list(
                (source_manifest.get("fit_scope") or {}).get("fit_splits") or []
            ),
            "requires_labeled_fit": False,
            "train_row_policy": "safe_in_sample_fit_free",
            "fold_split_required_for_reranker_train": False,
            "uses_devset_for_fit": False,
            "uses_blind_for_fit": False,
        },
        "paper_protocol": {
            "source_split": args.source_split,
            "operation": "row_slice_only",
            "devset_labels_read": False,
        },
    }
    save_npz_artifact(output_dir, output_arrays, turns, manifest)
    print(
        f"wrote {output_dir} rows={len(selected)} width={output_arrays['track_idx'].shape[1]}"
    )


if __name__ == "__main__":
    main()
