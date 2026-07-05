#!/usr/bin/env python3
"""Wrapper for the train-safe album co-occurrence retriever."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="oof3_top500")
    parser.add_argument("--target", choices=("public_labeled", "blind_a", "blind_b"), default="public_labeled")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args, extra = parser.parse_known_args([a for a in sys.argv[1:] if a != "--"])

    mode = "public" if args.target == "public_labeled" else "blind"
    argv = [
        sys.executable,
        "scripts/build_train_fit_retriever_artifacts.py",
        "--config",
        args.config,
        "--source",
        "cooc_album",
        "--mode",
        mode,
    ]
    if args.target != "public_labeled":
        argv.extend(["--blind-target", args.target])
    if args.top_k is not None:
        argv.extend(["--top-k", str(args.top_k)])
    if args.force:
        argv.append("--force")
    argv.extend(extra)
    subprocess.run(argv, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
