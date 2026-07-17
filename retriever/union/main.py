#!/usr/bin/env python3
"""Build a union retriever artifact from `retriever/union/configs/*.yaml`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retriever.union import builder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--target", choices=("devset", "public_labeled", "blind_b"), required=True
    )
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    config_file = Path("retriever/union/configs") / f"{args.config}.yaml"
    builder.main(
        [
            "--config",
            args.config,
            "--target",
            args.target,
            "--config-file",
            str(config_file),
        ]
    )


if __name__ == "__main__":
    main()
