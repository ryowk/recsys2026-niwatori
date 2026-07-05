#!/usr/bin/env python3
"""Wrapper for challenge+TPD1 combined track cooc retriever."""

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
    parser.add_argument("--offline", action="store_true")
    args, extra = parser.parse_known_args([a for a in sys.argv[1:] if a != "--"])

    config_file = Path("retriever/cooc_track_combined_tpd1/configs") / f"{args.config}.yaml"
    argv = [
        sys.executable,
        "scripts/build_combined_tpd1_retrievers.py",
        "--source",
        "cooc_track_combined_tpd1",
        "--config",
        args.config,
        "--config-file",
        str(config_file),
        "--target",
        args.target,
    ]
    if args.top_k is not None:
        argv.extend(["--top-k", str(args.top_k)])
    if args.force:
        argv.append("--force")
    if args.offline:
        argv.append("--offline")
    argv.extend(extra)
    subprocess.run(argv, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
