"""Canonical paths for locally generated pipeline artifacts."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
OUTPUT_DIR = ARTIFACTS_DIR / "runs"
RESULTS_DIR = ARTIFACTS_DIR / "results"
PREPROCESSED_DIR = ARTIFACTS_DIR / "preprocessed"
