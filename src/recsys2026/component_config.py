"""Config/path helpers for top-level pipeline components.

This mirrors ``exp_config.py`` but works for ``retriever/``, ``reranker/``,
``responder/``, ``preprocess/``, and ``pipeline/`` directories.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from .artifacts import Stage, component_output_dir, component_results_dir


def _component_parts(caller_file: str | Path) -> tuple[Stage, str]:
    path = Path(caller_file).resolve()
    name = path.parent.name
    stage = path.parent.parent.name
    if stage not in {"preprocess", "retriever", "reranker", "responder", "pipeline"}:
        raise ValueError(f"{path} is not inside a component directory")
    return stage, name  # type: ignore[return-value]


def load_component_config(caller_file: str | Path, config_name: str) -> dict:
    comp_dir = Path(caller_file).resolve().parent
    path = comp_dir / "configs" / f"{config_name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (comp_dir / "configs").glob("*.yaml")) if (comp_dir / "configs").is_dir() else []
        raise FileNotFoundError(f"config not found: {path}\n  available: {available}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def component_output_dir_for_caller(
    caller_file: str | Path,
    config_name: str,
    target: str | None = None,
    fit_mode: str | None = None,
) -> Path:
    stage, name = _component_parts(caller_file)
    return component_output_dir(stage, name, config_name, target=target, fit_mode=fit_mode)


def component_results_dir_for_caller(
    caller_file: str | Path,
    config_name: str,
    target: str | None = None,
    fit_mode: str | None = None,
) -> Path:
    stage, name = _component_parts(caller_file)
    return component_results_dir(stage, name, config_name, target=target, fit_mode=fit_mode)
