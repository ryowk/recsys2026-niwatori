"""Reranker artifact evaluation utilities."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .artifacts import json_dump, load_ranked_artifact, records_from_ranked_artifact
from .eval import evaluate_devset
from .submission import Target, write_predictions


def evaluate_ranked_artifact(
    artifact_dir: Path,
    *,
    target: Target = "devset",
    top_k: int = 20,
    write_to: Path | None = None,
) -> dict[str, Any]:
    if target != "devset":
        raise ValueError("reranker metrics with gold are currently only available for target=devset")
    _, manifest = load_ranked_artifact(artifact_dir)
    records = records_from_ranked_artifact(artifact_dir, target, top_k=top_k, response="")
    with TemporaryDirectory() as tmp:
        pred_path = Path(tmp) / "devset.json"
        write_predictions(records, pred_path, target)
        scores = evaluate_devset(pred_path)
    scores["artifact"] = str(Path(artifact_dir))
    scores["name"] = manifest.get("name")
    scores["config"] = manifest.get("config")
    scores["target"] = target
    scores["top_k"] = top_k
    if write_to is not None:
        json_dump(write_to, scores)
    return scores
