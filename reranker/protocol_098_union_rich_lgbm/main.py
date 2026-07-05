#!/usr/bin/env python3
"""Config wrapper for the 098-rich LGBM union reranker component."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from recsys2026.paths import REPO_ROOT


def load_config(config: str) -> dict[str, Any]:
    path = REPO_ROOT / "reranker" / "protocol_098_union_rich_lgbm" / "configs" / f"{config}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"reranker config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def add_arg(argv: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    argv.extend([flag, str(value)])


def add_list_arg(argv: list[str], flag: str, value: object | None) -> None:
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        joined = ",".join(str(v) for v in value)
    else:
        joined = str(value)
    if joined:
        argv.extend([flag, joined])


def max_candidates_value(value: object) -> int:
    if value is None:
        return 500
    if isinstance(value, str) and value.lower() in {"all", "full", "none"}:
        return 0
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--target", choices=("public_labeled", "blind_a", "blind_b"), default="public_labeled")
    args, extra = parser.parse_known_args([a for a in sys.argv[1:] if a != "--"])

    cfg = load_config(args.config)
    # blind-B-safe behaviour is unconditional (goal / thought / GPA are never used).

    candidate_artifacts = dict(cfg.get("candidate_artifacts") or {})
    public_candidate = candidate_artifacts.get("public_labeled")
    if not public_candidate:
        raise ValueError(f"config={args.config} does not define candidate_artifacts.public_labeled")
    blind_candidate = candidate_artifacts.get(args.target) if args.target != "public_labeled" else None
    if args.target != "public_labeled" and not blind_candidate:
        raise ValueError(f"config={args.config} does not define candidate_artifacts.{args.target}")

    lgbm = dict(cfg.get("lgbm") or {})
    feature_build = dict(cfg.get("feature_build") or {})
    argv = [
        sys.executable,
        "scripts/run_reranker.py",
        "--name",
        "protocol_098_union_rich_lgbm",
        "--config",
        args.config,
        "--public-candidates",
        str(public_candidate),
        "--max-candidates",
        str(max_candidates_value(cfg.get("max_candidates"))),
        "--top-k",
        str(int(cfg.get("top_k", 20))),
        "--cv-folds",
        str(cfg.get("cv_folds", "all")),
        "--cv-artifact-mode",
        str(cfg.get("cv_artifact_mode", "cv3_oof")),
        "--primary-score-mode",
        str(cfg.get("primary_score_mode", "bm25")),
    ]
    if not bool(cfg.get("source_features", True)):
        argv.append("--disable-source-features")
    if bool(feature_build.get("drop_cross_source_score_meta", False)):
        argv.append("--drop-cross-source-score-meta")
    if bool(feature_build.get("extra_source_score_transforms", False)):
        argv.append("--extra-source-score-transforms")
    if bool(feature_build.get("extra_metadata_features", False)):
        argv.append("--extra-metadata-features")
    if bool(feature_build.get("extra_feedback_features", False)):
        argv.append("--extra-feedback-features")
    if bool(feature_build.get("extra_gpa_features", False)):
        argv.append("--extra-gpa-features")
    if bool(feature_build.get("extra_reaction_features", False)):
        argv.append("--extra-reaction-features")
    if bool(feature_build.get("extra_goal_cluster_features", False)):
        argv.append("--extra-goal-cluster-features")
    if bool(feature_build.get("extra_assistant_thought_features", False)):
        argv.append("--extra-assistant-thought-features")
    if bool(feature_build.get("extra_tag_chain_features", False)):
        argv.append("--extra-tag-chain-features")
    if bool(feature_build.get("extra_hier_pop_features", False)):
        argv.append("--extra-hier-pop-features")
    if bool(feature_build.get("extra_pool_prior_features", False)):
        argv.append("--extra-pool-prior-features")
    if bool(feature_build.get("extra_talkplay_aux_features", False)):
        argv.append("--extra-talkplay-aux-features")
    if bool(feature_build.get("extra_category_turn_features", False)):
        argv.append("--extra-category-turn-features")
    if bool(feature_build.get("extra_score_calibration_features", False)):
        argv.append("--extra-score-calibration-features")
    if bool(cfg.get("train_positive_only", False)):
        argv.append("--train-positive-only")
    if bool(cfg.get("train_positive_eval_mask", False)):
        argv.append("--train-positive-eval-mask")
    if bool(cfg.get("train_textproxy_positive", False)):
        argv.append("--train-textproxy-positive")
    add_arg(argv, "--positive-group-weight", cfg.get("positive_group_weight"))
    if bool(cfg.get("allow_encode_missing", False)):
        argv.append("--allow-encode-missing")
    if blind_candidate is not None:
        argv.extend(["--blind-candidates", str(blind_candidate), "--blind-target", args.target, "--skip-cv"])

    tpd1_mix = dict(cfg.get("tpd1_reranker_mix") or {})
    if bool(tpd1_mix.get("enabled", False)):
        argv.append("--tpd1-mix-reranker-train")
        add_arg(argv, "--tpd1-mix-max-examples", tpd1_mix.get("max_examples"))
        add_arg(argv, "--tpd1-mix-candidate-k", tpd1_mix.get("candidate_k"))
        add_arg(argv, "--tpd1-mix-weight", tpd1_mix.get("weight"))
        add_arg(argv, "--tpd1-mix-seed", tpd1_mix.get("seed"))
        add_arg(argv, "--tpd1-mix-cache-name", tpd1_mix.get("cache_name"))
        add_arg(argv, "--tpd1-mix-mapping", tpd1_mix.get("mapping"))

    add_arg(argv, "--feature-chunk-examples", feature_build.get("feature_chunk_examples"))
    add_arg(argv, "--n-bm25-for-dense-flag", feature_build.get("n_bm25_for_dense_flag"))
    add_list_arg(argv, "--extra-candidate-feature-npz", feature_build.get("extra_candidate_feature_npz"))
    add_arg(argv, "--goal-cluster-n-clusters", feature_build.get("goal_cluster_n_clusters"))
    add_arg(argv, "--goal-cluster-cache-path", feature_build.get("goal_cluster_cache_path"))
    add_arg(argv, "--goal-cluster-batch-size", feature_build.get("goal_cluster_batch_size"))
    add_list_arg(argv, "--neutralize-098-features", feature_build.get("neutralize_098_features"))
    add_arg(argv, "--model-family", lgbm.get("model_family"))
    add_arg(argv, "--lgbm-objective", lgbm.get("objective"))
    if bool(lgbm.get("binary_no_weight", False)):
        argv.append("--binary-no-weight")
    add_arg(argv, "--n-estimators", lgbm.get("n_estimators"))
    add_arg(argv, "--num-leaves", lgbm.get("num_leaves"))
    add_arg(argv, "--max-depth", lgbm.get("max_depth"))
    add_arg(argv, "--learning-rate", lgbm.get("learning_rate"))
    add_arg(argv, "--subsample", lgbm.get("subsample"))
    add_arg(argv, "--colsample-bytree", lgbm.get("colsample_bytree"))
    add_arg(argv, "--min-child-samples", lgbm.get("min_child_samples"))
    add_arg(argv, "--min-child-weight", lgbm.get("min_child_weight"))
    add_arg(argv, "--reg-lambda", lgbm.get("reg_lambda"))
    add_arg(argv, "--max-bin", lgbm.get("max_bin"))
    add_arg(argv, "--xgb-device", lgbm.get("xgb_device"))
    add_arg(argv, "--xgb-rank-objective", lgbm.get("xgb_rank_objective"))
    add_arg(argv, "--catboost-loss", lgbm.get("catboost_loss"))
    add_arg(argv, "--catboost-task-type", lgbm.get("catboost_task_type"))
    add_arg(argv, "--catboost-devices", lgbm.get("catboost_devices"))
    add_arg(argv, "--lambdarank-truncation-level", lgbm.get("lambdarank_truncation_level"))
    add_arg(argv, "--n-jobs", lgbm.get("n_jobs"))
    add_arg(argv, "--seed", lgbm.get("seed"))
    argv.extend(extra)

    subprocess.run(argv, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
