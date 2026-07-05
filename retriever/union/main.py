#!/usr/bin/env python3
"""Build a union retriever artifact from `retriever/union/configs/*.yaml`."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from recsys2026.paths import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", choices=("devset", "public_labeled", "blind_a", "blind_b"), default="devset")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--no-source-features", action="store_true")
    parser.add_argument("--compress", action="store_true")
    args, extra = parser.parse_known_args([a for a in sys.argv[1:] if a != "--"])

    config_file = Path("retriever/union/configs") / f"{args.config}.yaml"
    builder_argv = [
        "scripts/build_union_candidates.py",
        "--name",
        "union",
        "--config",
        args.config,
        "--target",
        args.target,
        "--config-file",
        str(config_file),
    ]
    if args.max_candidates is not None:
        builder_argv.extend(["--max-candidates", str(args.max_candidates)])
    if args.no_source_features:
        builder_argv.append("--no-source-features")
    if args.compress:
        builder_argv.append("--compress")
    builder_argv.extend(extra)

    sys.path.insert(0, str(REPO_ROOT))
    from scripts import build_union_candidates

    old_argv = sys.argv
    old_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        sys.argv = builder_argv
        build_union_candidates.main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
