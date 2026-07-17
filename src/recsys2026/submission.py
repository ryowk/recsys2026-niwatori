"""Submission input iteration, schema validation, and zip packaging."""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .data import load

Target = Literal["devset", "blind_b"]
MAX_K = 20


@dataclass(frozen=True)
class InferenceInput:
    session_id: str
    user_id: str
    turn_number: int
    chat_history: list[dict]
    user_query: str


def _iter_devset() -> Iterator[InferenceInput]:
    ds = load("dataset", split="test")
    for item in ds:
        conversations = item["conversations"]
        for target_turn in range(1, 9):
            history = [c for c in conversations if c["turn_number"] < target_turn]
            current = [c for c in conversations if c["turn_number"] == target_turn]
            user_query = next(c["content"] for c in current if c["role"] == "user")
            yield InferenceInput(
                session_id=item["session_id"],
                user_id=item["user_id"],
                turn_number=target_turn,
                chat_history=history,
                user_query=user_query,
            )


def _iter_blind_b() -> Iterator[InferenceInput]:
    ds = load("blind_b", split="test")
    for item in ds:
        conv = item["conversations"]
        yield InferenceInput(
            session_id=item["session_id"],
            user_id=item["user_id"],
            turn_number=conv[-1]["turn_number"],
            chat_history=conv[:-1],
            user_query=conv[-1]["content"],
        )


def iter_inputs(target: Target) -> Iterator[InferenceInput]:
    if target == "devset":
        yield from _iter_devset()
    elif target == "blind_b":
        yield from _iter_blind_b()
    else:
        raise ValueError(f"unknown target: {target}")


def format_record(inp: InferenceInput, track_ids: list[str], response: str) -> dict:
    """Format one session-turn prediction for the challenge schema."""
    return {
        "session_id": inp.session_id,
        "user_id": inp.user_id,
        "turn_number": inp.turn_number,
        "predicted_track_ids": list(track_ids),
        "predicted_response": response,
    }


def validate_predictions(
    records: list[dict],
    target: Target,
    *,
    require_complete: bool = True,
    allowed_keys: set[tuple[str, int]] | None = None,
) -> None:
    """Validate target coverage, top-20 uniqueness, and catalog membership."""
    required_fields = {
        "session_id",
        "user_id",
        "turn_number",
        "predicted_track_ids",
        "predicted_response",
    }
    target_inputs = list(iter_inputs(target))
    expected_users = {
        (inp.session_id, inp.turn_number): inp.user_id for inp in target_inputs
    }
    expected_all = set(expected_users)
    expected = allowed_keys if allowed_keys is not None else expected_all
    unknown_allowed = expected - expected_all
    if unknown_allowed:
        raise ValueError(
            f"allowed_keys contains keys outside target={target}, e.g. {next(iter(unknown_allowed))}"
        )
    catalog = set(load("track", split="all_tracks")["track_id"])
    seen: set[tuple[str, int]] = set()
    for r in records:
        if set(r) != required_fields:
            raise ValueError(
                f"prediction fields must be exactly {sorted(required_fields)}, "
                f"got {sorted(r)}"
            )
        key = (r["session_id"], r["turn_number"])
        if key in seen:
            raise ValueError(f"duplicate prediction for {key}")
        seen.add(key)
        if key not in expected:
            raise ValueError(
                f"unexpected (session_id, turn_number) for target={target}: {key}"
            )
        if r["user_id"] != expected_users[key]:
            raise ValueError(
                f"user_id mismatch for {key}: got={r['user_id']!r}, "
                f"expected={expected_users[key]!r}"
            )
        tids = r["predicted_track_ids"]
        if not isinstance(tids, list) or not all(isinstance(tid, str) for tid in tids):
            raise ValueError(f"predicted_track_ids must be a list of strings for {key}")
        if len(tids) != MAX_K:
            raise ValueError(f"expected {MAX_K} tracks for {key}, got {len(tids)}")
        if len(tids) != len(set(tids)):
            raise ValueError(f"duplicate track_ids in {key}")
        unknown = set(tids) - catalog
        if unknown:
            raise ValueError(f"unknown track_ids in {key}: e.g. {next(iter(unknown))}")
        if not isinstance(r["predicted_response"], str):
            raise ValueError(f"predicted_response must be a string for {key}")
    missing = expected - seen
    if require_complete and missing:
        raise ValueError(
            f"predictions missing for {len(missing)} (session_id, turn_number) pairs, "
            f"e.g. {next(iter(missing))}"
        )


def write_predictions(
    records: list[dict],
    out_path: Path,
    target: Target,
    *,
    require_complete: bool = True,
    allowed_keys: set[tuple[str, int]] | None = None,
) -> None:
    """Validate and write a prediction JSON file."""
    validate_predictions(
        records, target, require_complete=require_complete, allowed_keys=allowed_keys
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(f".{out_path.name}.tmp")
    temp_path.write_text(json.dumps(records, ensure_ascii=False))
    temp_path.replace(out_path)


def zip_submission(json_path: Path, zip_path: Path | None = None) -> Path:
    """Package JSON as the required `prediction.json` zip member."""
    json_path = Path(json_path)
    if zip_path is None:
        zip_path = json_path.with_suffix(".submission.zip")
    else:
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = zip_path.with_name(f".{zip_path.name}.tmp")
    with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="prediction.json")
    temp_path.replace(zip_path)
    return zip_path
