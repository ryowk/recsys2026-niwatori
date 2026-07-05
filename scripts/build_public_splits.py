#!/usr/bin/env python3
"""Build fixed public-labeled CV and smoke split artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedKFold

from recsys2026.paths import REPO_ROOT
from recsys2026.splits import (
    assign_strata,
    row_records,
    session_records,
    weighted_nested_order,
    write_jsonl,
)


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "cache" / "splits" / "cv5")
    parser.add_argument("--name", default=None)
    parser.add_argument("--seed", type=int, default=20260515)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--smoke-sizes", type=int, nargs="+", default=[100, 300, 1000])
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    out_dir = args.out_dir if args.out_dir.is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    split_name = str(args.name or out_dir.name)

    sessions = session_records()
    strata = assign_strata(sessions, n_splits=args.n_splits)
    splitter = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for s, stratum in zip(sessions, strata, strict=True):
        s["stratum"] = stratum
        s["fold"] = -1

    x = list(range(len(sessions)))
    for fold, (_, valid_idx) in enumerate(splitter.split(x, strata)):
        for idx in valid_idx:
            sessions[int(idx)]["fold"] = fold

    if any(int(s["fold"]) < 0 for s in sessions):
        raise RuntimeError("some sessions were not assigned a fold")

    session_fold = {(s["source_split"], s["session_id"]): int(s["fold"]) for s in sessions}
    rows = row_records(session_fold)

    write_jsonl(out_dir / "sessions.jsonl", sessions)
    write_jsonl(out_dir / "rows.jsonl", rows)

    smoke_outputs: dict[str, str] = {}
    max_smoke = max(args.smoke_sizes)
    public_order = weighted_nested_order(rows, seed=args.seed)
    devset_rows = [r for r in rows if r["source_split"] == "devset"]
    devset_order = weighted_nested_order(devset_rows, seed=args.seed)
    for size in sorted(args.smoke_sizes):
        if size > max_smoke:
            raise AssertionError("unreachable")
        public_path = out_dir / f"smoke_public_{size}.jsonl"
        devset_path = out_dir / f"smoke_devset_{size}.jsonl"
        write_jsonl(public_path, public_order[: min(size, len(public_order))])
        write_jsonl(devset_path, devset_order[: min(size, len(devset_order))])
        smoke_outputs[f"smoke_public_{size}"] = rel(public_path)
        smoke_outputs[f"smoke_devset_{size}"] = rel(devset_path)

    source_counts = Counter(s["source_split"] for s in sessions)
    fold_source_counts = Counter((s["fold"], s["source_split"]) for s in sessions)
    manifest = {
        "schema_version": 1,
        "name": split_name,
        "description": (
            f"Fixed session-stratified {args.n_splits}-fold split over train+devset, "
            "plus nested smoke subsets."
        ),
        "seed": args.seed,
        "n_splits": args.n_splits,
        "source_splits": ["train", "devset"],
        "n_sessions": len(sessions),
        "n_rows": len(rows),
        "source_session_counts": dict(source_counts),
        "fold_source_session_counts": {
            f"fold{fold}_{source}": count
            for (fold, source), count in sorted(fold_source_counts.items())
        },
        "smoke_sizes": sorted(args.smoke_sizes),
        "files": {
            "sessions": rel(out_dir / "sessions.jsonl"),
            "rows": rel(out_dir / "rows.jsonl"),
            **smoke_outputs,
        },
        "stratification": {
            "session_level": True,
            "initial_fields": ["source_split", "goal_category", "goal_specificity", "user_split"],
            "rare_strata_are_collapsed": True,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
