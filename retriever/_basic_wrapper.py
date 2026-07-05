#!/usr/bin/env python3
"""Shared thin wrapper for basic retriever components."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from recsys2026.component_config import load_component_config
from recsys2026.paths import REPO_ROOT


def _component_name(caller_file: str | Path) -> str:
    return Path(caller_file).resolve().parent.name


def _optional_arg(argv: list[str], flag: str, value: Any | None) -> None:
    if value is not None:
        argv.extend([flag, str(value)])


def main(caller_file: str | Path) -> None:
    """Run one source through ``scripts/build_basic_retrievers.py``."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="basic")
    parser.add_argument("--target", choices=("devset", "public_labeled", "blind_a", "blind_b"), default="devset")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--n-neigh", type=int, default=None)
    parser.add_argument("--rank-offset", type=float, default=None)
    parser.add_argument("--skip-unsupported", action="store_true")
    args, extra = parser.parse_known_args([a for a in sys.argv[1:] if a != "--"])

    cfg = load_component_config(caller_file, args.config)
    source = str(cfg.get("source") or _component_name(caller_file))
    config_file = str(cfg.get("config_file") or "retriever/union/configs/union_v1.yaml")
    component_config_file = Path(caller_file).resolve().parent / "configs" / f"{args.config}.yaml"
    component_config_file = component_config_file.relative_to(REPO_ROOT)
    top_k = args.top_k if args.top_k is not None else int(cfg.get("top_k", 200))
    n_neigh = args.n_neigh if args.n_neigh is not None else cfg.get("n_neigh")
    rank_offset = args.rank_offset if args.rank_offset is not None else cfg.get("rank_offset")
    user_neighbor_score_mode = cfg.get("user_neighbor_score_mode")

    builder_argv = [
        "scripts/build_basic_retrievers.py",
        "--config-file",
        config_file,
        "--component-config-file",
        str(component_config_file),
        "--config",
        args.config,
        "--target",
        args.target,
        "--top-k",
        str(top_k),
        "--device",
        args.device,
        "--only",
        source,
    ]
    _optional_arg(builder_argv, "--max-examples", args.max_examples)
    _optional_arg(builder_argv, "--n-neigh", n_neigh)
    _optional_arg(builder_argv, "--rank-offset", rank_offset)
    _optional_arg(builder_argv, "--user-neighbor-score-mode", user_neighbor_score_mode)
    if args.skip_unsupported:
        builder_argv.append("--skip-unsupported")
    builder_argv.extend(extra)

    sys.path.insert(0, str(REPO_ROOT))
    from scripts import build_basic_retrievers

    old_argv = sys.argv
    old_cwd = Path.cwd()
    try:
        os.chdir(REPO_ROOT)
        sys.argv = builder_argv
        build_basic_retrievers.main()
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)
