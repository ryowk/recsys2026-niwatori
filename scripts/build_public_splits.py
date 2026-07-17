#!/usr/bin/env python3
"""Build a fixed five-fold session split for reranker-fit artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from recsys2026.artifacts import json_dump
from recsys2026.paths import REPO_ROOT
from recsys2026.splits import (
    assign_strata,
    row_records,
    session_records,
    write_jsonl,
)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-splits", type=int, required=True)
    parser.add_argument(
        "--source-splits",
        nargs="+",
        choices=("train", "devset"),
        default=["train", "devset"],
        help="Labeled source splits to partition. Use 'train' for a held-out devset evaluation protocol.",
    )
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").unlink(missing_ok=True)
    split_name = str(args.name)

    source_splits = tuple(dict.fromkeys(args.source_splits))
    sessions = session_records(source_splits)
    strata = assign_strata(sessions, n_splits=args.n_splits)
    splitter = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.seed
    )

    for s, stratum in zip(sessions, strata, strict=True):
        s["stratum"] = stratum
        s["fold"] = -1

    x = list(range(len(sessions)))
    for fold, (_, valid_idx) in enumerate(splitter.split(x, strata)):
        for idx in valid_idx:
            sessions[int(idx)]["fold"] = fold

    if any(int(s["fold"]) < 0 for s in sessions):
        raise RuntimeError("some sessions were not assigned a fold")

    session_fold = {
        (s["source_split"], s["session_id"]): int(s["fold"]) for s in sessions
    }
    rows = row_records(session_fold, source_splits)

    write_jsonl(out_dir / "sessions.jsonl", sessions)
    write_jsonl(out_dir / "rows.jsonl", rows)

    source_counts = Counter(s["source_split"] for s in sessions)
    fold_source_counts = Counter((s["fold"], s["source_split"]) for s in sessions)
    manifest = {
        "schema_version": 1,
        "name": split_name,
        "description": (
            f"Fixed session-stratified {args.n_splits}-fold split over "
            f"{'+'.join(source_splits)}."
        ),
        "seed": args.seed,
        "n_splits": args.n_splits,
        "source_splits": list(source_splits),
        "n_sessions": len(sessions),
        "n_rows": len(rows),
        "source_session_counts": dict(source_counts),
        "fold_source_session_counts": {
            f"fold{fold}_{source}": count
            for (fold, source), count in sorted(fold_source_counts.items())
        },
        "files": {
            "sessions": rel(out_dir / "sessions.jsonl"),
            "rows": rel(out_dir / "rows.jsonl"),
        },
        "stratification": {
            "session_level": True,
            "initial_fields": [
                "source_split",
                "goal_category",
                "goal_specificity",
                "user_split",
            ],
            "rare_strata_are_collapsed": True,
        },
    }
    json_dump(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
