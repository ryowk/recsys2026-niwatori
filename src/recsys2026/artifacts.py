"""Artifact IO for componentized retriever/reranker/responder pipelines.

The module intentionally stays small and NumPy/JSON based.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .data import load
from .paths import OUTPUT_DIR, REPO_ROOT, RESULTS_DIR
from .submission import (
    Target,
    format_record,
    iter_inputs,
)

Stage = Literal["preprocess", "retriever", "reranker", "responder", "pipeline"]


def utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temp_path.replace(path)


def json_load(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def artifact_complete(path: Path, *required_files: str) -> bool:
    """Return true only after the manifest and all declared outputs exist."""
    path = Path(path)
    return (path / "manifest.json").is_file() and all(
        (path / name).is_file() for name in required_files
    )


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    path = Path(path)
    stat = path.stat()
    try:
        rel = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        rel = str(path)
    return {
        "path": rel,
        "size": stat.st_size,
        "sha256": sha256_file(path),
    }


def component_output_dir(
    stage: Stage,
    name: str,
    config: str,
    target: str | None = None,
    fit_mode: str | None = None,
) -> Path:
    parts = [OUTPUT_DIR, stage, name, config]
    if fit_mode is not None:
        parts.append(fit_mode)
    if target is not None:
        parts.append(target)
    out = Path(*parts)
    out.mkdir(parents=True, exist_ok=True)
    return out


def component_results_dir(
    stage: Stage,
    name: str,
    config: str,
    target: str | None = None,
    fit_mode: str | None = None,
) -> Path:
    parts = [RESULTS_DIR, stage, name, config]
    if fit_mode is not None:
        parts.append(fit_mode)
    if target is not None:
        parts.append(target)
    out = Path(*parts)
    out.mkdir(parents=True, exist_ok=True)
    return out


def target_rows(target: Target) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_id, inp in enumerate(iter_inputs(target)):
        rows.append(
            {
                "row_id": row_id,
                "session_id": inp.session_id,
                "user_id": inp.user_id,
                "turn_number": int(inp.turn_number),
            }
        )
    return rows


def target_keys(target: Target) -> list[tuple[str, int]]:
    return [(r["session_id"], int(r["turn_number"])) for r in target_rows(target)]


def encode_keys(keys: list[tuple[str, int]]) -> np.ndarray:
    return np.asarray(
        [f"{sid}:{turn}".encode("utf-8") for sid, turn in keys], dtype="S96"
    )


def decode_keys(arr: np.ndarray) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for raw in arr:
        text = bytes(raw).decode("utf-8")
        sid, turn = text.rsplit(":", 1)
        out.append((sid, int(turn)))
    return out


def assert_target_alignment(keys: list[tuple[str, int]], target: Target) -> None:
    expected = target_keys(target)
    if keys != expected:
        for i, (got, exp) in enumerate(zip(keys, expected, strict=False)):
            if got != exp:
                raise ValueError(
                    f"row alignment mismatch at {i}: got={got}, expected={exp}"
                )
        raise ValueError(
            f"row count mismatch: got={len(keys)}, expected={len(expected)}"
        )


def write_turns(path: Path, target: Target) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        for row in target_rows(target):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def track_id_lookup() -> tuple[list[str], dict[str, int]]:
    tracks = load("track", split="all_tracks")
    ids = list(tracks["track_id"])
    return ids, {tid: i for i, tid in enumerate(ids)}


def _validate_rows(track_idx: np.ndarray, sizes: np.ndarray) -> None:
    if track_idx.ndim != 2:
        raise ValueError(f"track_idx must be 2D, got shape={track_idx.shape}")
    if sizes.ndim != 1 or sizes.shape[0] != track_idx.shape[0]:
        raise ValueError(
            f"sizes shape mismatch: track_idx={track_idx.shape}, sizes={sizes.shape}"
        )
    width = track_idx.shape[1]
    for i, size_raw in enumerate(sizes):
        size = int(size_raw)
        if size < 0 or size > width:
            raise ValueError(f"invalid size at row {i}: {size} for width {width}")
        vals = [int(x) for x in track_idx[i, :size] if int(x) >= 0]
        if len(vals) != size:
            raise ValueError(f"negative track idx inside valid region at row {i}")
        if len(vals) != len(set(vals)):
            raise ValueError(f"duplicate track idx at row {i}")


def _save_npz(path: Path, compress: bool, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("wb") as handle:
        if compress:
            np.savez_compressed(handle, **arrays)
        else:
            np.savez(handle, **arrays)
    temp_path.replace(path)


def jsonl_dump(path: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write JSON Lines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def npz_dump(
    path: Path, arrays: dict[str, np.ndarray], *, compress: bool = False
) -> None:
    """Atomically write a NumPy archive."""
    _save_npz(path, compress, arrays)


def save_npz_artifact(
    artifact_dir: Path,
    arrays: dict[str, np.ndarray],
    turn_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    compress: bool = True,
) -> None:
    """Write a custom candidate artifact with the manifest as completion marker."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "manifest.json").unlink(missing_ok=True)
    npz_dump(artifact_dir / "candidates.npz", arrays, compress=compress)
    jsonl_dump(artifact_dir / "turns.jsonl", turn_rows)
    json_dump(artifact_dir / "manifest.json", manifest)


def save_candidate_artifact(
    artifact_dir: Path,
    track_idx: np.ndarray,
    sizes: np.ndarray,
    *,
    target: Target,
    manifest: dict[str, Any],
    rank: np.ndarray | None = None,
    score_arrays: dict[str, np.ndarray] | None = None,
    feature_arrays: dict[str, np.ndarray] | None = None,
    mask_arrays: dict[str, np.ndarray] | None = None,
    score_fields: dict[str, Any] | None = None,
    candidate_views: dict[str, Any] | None = None,
    compress: bool = False,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "manifest.json").unlink(missing_ok=True)
    track_idx = np.asarray(track_idx, dtype=np.int32)
    sizes = np.asarray(sizes, dtype=np.int32)
    _validate_rows(track_idx, sizes)
    keys = target_keys(target)
    if len(keys) != track_idx.shape[0]:
        raise ValueError(
            f"target row count mismatch: target={len(keys)} artifact={track_idx.shape[0]}"
        )

    arrays: dict[str, np.ndarray] = {
        "track_idx": track_idx,
        "sizes": sizes,
        "keys": encode_keys(keys),
    }
    if rank is not None:
        arrays["rank"] = np.asarray(rank, dtype=np.int32)
    for prefix, values in (
        ("score__", score_arrays or {}),
        ("feat__", feature_arrays or {}),
        ("eligible_mask__", mask_arrays or {}),
    ):
        for name, arr in values.items():
            key = name if name.startswith(prefix) else f"{prefix}{name}"
            arrays[key] = np.asarray(arr)

    _save_npz(artifact_dir / "candidates.npz", compress, arrays)
    write_turns(artifact_dir / "turns.jsonl", target)
    if score_fields is not None:
        json_dump(artifact_dir / "score_fields.json", score_fields)
    if candidate_views is not None:
        json_dump(artifact_dir / "candidate_views.json", candidate_views)
    json_dump(artifact_dir / "manifest.json", manifest)


def load_candidate_artifact(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        npz_path = path / "candidates.npz"
        manifest_path = path / "manifest.json"
    else:
        npz_path = path
        manifest_path = path.with_name("manifest.json")
    data = np.load(npz_path)
    arrays = {name: data[name] for name in data.files}
    manifest = json_load(manifest_path) if manifest_path.exists() else {}
    return arrays, manifest


def save_ranked_artifact(
    artifact_dir: Path,
    track_idx: np.ndarray,
    sizes: np.ndarray,
    *,
    target: Target,
    manifest: dict[str, Any],
    scores: np.ndarray | None = None,
    source_candidate_rank: np.ndarray | None = None,
    compress: bool = False,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "manifest.json").unlink(missing_ok=True)
    track_idx = np.asarray(track_idx, dtype=np.int32)
    sizes = np.asarray(sizes, dtype=np.int32)
    _validate_rows(track_idx, sizes)
    keys = target_keys(target)
    if len(keys) != track_idx.shape[0]:
        raise ValueError(
            f"target row count mismatch: target={len(keys)} artifact={track_idx.shape[0]}"
        )

    arrays: dict[str, np.ndarray] = {
        "track_idx": track_idx,
        "sizes": sizes,
        "keys": encode_keys(keys),
    }
    if scores is not None:
        arrays["scores"] = np.asarray(scores, dtype=np.float32)
    else:
        arrays["scores"] = np.full(track_idx.shape, np.nan, dtype=np.float32)
    if source_candidate_rank is not None:
        arrays["source_candidate_rank"] = np.asarray(
            source_candidate_rank, dtype=np.int32
        )

    _save_npz(artifact_dir / "ranked.npz", compress, arrays)
    write_turns(artifact_dir / "turns.jsonl", target)
    json_dump(artifact_dir / "manifest.json", manifest)


def load_ranked_artifact(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        npz_path = path / "ranked.npz"
        manifest_path = path / "manifest.json"
    else:
        npz_path = path
        manifest_path = path.with_name("manifest.json")
    data = np.load(npz_path)
    arrays = {name: data[name] for name in data.files}
    manifest = json_load(manifest_path) if manifest_path.exists() else {}
    return arrays, manifest


def records_from_ranked_artifact(
    artifact_dir: Path,
    target: Target,
    *,
    top_k: int = 20,
    response: str = "",
) -> list[dict[str, Any]]:
    arrays, _ = load_ranked_artifact(artifact_dir)
    track_idx = arrays["track_idx"]
    sizes = arrays["sizes"]
    keys = decode_keys(arrays["keys"])
    assert_target_alignment(keys, target)
    track_ids, _ = track_id_lookup()

    records: list[dict[str, Any]] = []
    for inp, row, size_raw in zip(iter_inputs(target), track_idx, sizes, strict=True):
        size = min(int(size_raw), top_k)
        tids = [track_ids[int(idx)] for idx in row[:size] if int(idx) >= 0]
        records.append(format_record(inp, tids, response))
    return records
