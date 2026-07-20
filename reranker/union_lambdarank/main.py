#!/usr/bin/env python3
"""Config wrapper for the final union LambdaRank reranker."""

from __future__ import annotations

import argparse
import subprocess
import sys
from typing import Any

import yaml

from recsys2026.paths import REPO_ROOT


def load_config(config: str) -> dict[str, Any]:
    path = REPO_ROOT / "reranker" / "union_lambdarank" / "configs" / f"{config}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"reranker config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def add_arg(argv: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def max_candidates_value(value: object) -> int:
    if value is None:
        return 500
    if isinstance(value, str) and value.lower() in {"all", "full", "none"}:
        return 0
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", choices=("devset", "blind_b"), required=True)
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    cfg = load_config(args.config)

    candidate_artifacts = dict(cfg.get("candidate_artifacts") or {})
    public_candidate = candidate_artifacts.get("public_labeled")
    if not public_candidate:
        raise ValueError(
            f"config={args.config} does not define candidate_artifacts.public_labeled"
        )
    inference_candidate = candidate_artifacts.get(args.target)
    if not inference_candidate:
        raise ValueError(
            f"config={args.config} does not define candidate_artifacts.{args.target}"
        )

    lgbm = dict(cfg.get("lgbm") or {})
    feature_build = dict(cfg.get("feature_build") or {})
    argv = [
        sys.executable,
        "-m",
        "reranker.union_lambdarank.runner",
        "--config",
        args.config,
        "--public-candidates",
        str(public_candidate),
        "--max-candidates",
        str(max_candidates_value(cfg.get("max_candidates"))),
        "--top-k",
        str(int(cfg.get("top_k", 20))),
        "--cv-artifact-mode",
        str(cfg.get("cv_artifact_mode", "cv5_oof")),
    ]
    add_arg(argv, "--feature-set", cfg.get("feature_set"))
    if bool(feature_build.get("drop_cross_source_score_meta", False)):
        argv.append("--drop-cross-source-score-meta")
    if bool(feature_build.get("extra_metadata_features", False)):
        argv.append("--extra-metadata-features")
    if bool(feature_build.get("extra_tag_chain_features", False)):
        argv.append("--extra-tag-chain-features")
    if bool(feature_build.get("extra_hier_pop_features", False)):
        argv.append("--extra-hier-pop-features")
    if bool(cfg.get("train_positive_only", False)):
        argv.append("--train-positive-only")
    argv.extend(
        [
            "--inference-candidates",
            str(inference_candidate),
            "--inference-target",
            args.target,
        ]
    )

    add_arg(
        argv, "--feature-chunk-examples", feature_build.get("feature_chunk_examples")
    )
    neutralized = feature_build.get("neutralize_base_features") or []
    if neutralized:
        argv.extend(
            [
                "--neutralize-base-features",
                ",".join(str(value) for value in neutralized),
            ]
        )
    add_arg(argv, "--n-estimators", lgbm.get("n_estimators"))
    add_arg(argv, "--num-leaves", lgbm.get("num_leaves"))
    add_arg(argv, "--learning-rate", lgbm.get("learning_rate"))
    add_arg(argv, "--subsample", lgbm.get("subsample"))
    add_arg(argv, "--colsample-bytree", lgbm.get("colsample_bytree"))
    add_arg(argv, "--min-child-samples", lgbm.get("min_child_samples"))
    add_arg(
        argv, "--lambdarank-truncation-level", lgbm.get("lambdarank_truncation_level")
    )
    add_arg(argv, "--n-jobs", lgbm.get("n_jobs"))
    add_arg(argv, "--seed", lgbm.get("seed"))
    subprocess.run(argv, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
