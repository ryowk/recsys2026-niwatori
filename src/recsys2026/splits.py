"""Fixed split helpers for the component pipeline protocol."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .data import load

PUBLIC_SOURCE_SPLITS = ("train", "devset")
DATASET_SPLIT_BY_SOURCE = {"train": "train", "devset": "test"}
MAX_TURNS = 8


def _goal(item: dict[str, Any]) -> dict[str, Any]:
    return dict(item.get("conversation_goal") or {})


def _profile(item: dict[str, Any]) -> dict[str, Any]:
    return dict(item.get("user_profile") or {})


def session_records(
    source_splits: Iterable[str] = PUBLIC_SOURCE_SPLITS,
) -> list[dict[str, Any]]:
    """Return one record per session from the requested labeled splits."""
    rows: list[dict[str, Any]] = []
    for source_split in source_splits:
        if source_split not in DATASET_SPLIT_BY_SOURCE:
            raise ValueError(f"unknown source split: {source_split}")
        ds = load("dataset", split=DATASET_SPLIT_BY_SOURCE[source_split])
        for item in ds:
            goal = _goal(item)
            profile = _profile(item)
            rows.append(
                {
                    "source_split": source_split,
                    "session_id": item["session_id"],
                    "user_id": item["user_id"],
                    "goal_category": str(goal.get("category") or "NA"),
                    "goal_specificity": str(goal.get("specificity") or "NA"),
                    "user_split": str(profile.get("user_split") or "NA"),
                }
            )
    return rows


def assign_strata(sessions: list[dict[str, Any]], *, n_splits: int) -> list[str]:
    """Build stable strata, collapsing rare combinations until every stratum is usable."""
    candidates: list[list[str]] = []
    for s in sessions:
        candidates.append(
            [
                f"{s['source_split']}|{s['goal_category']}|{s['goal_specificity']}|{s['user_split']}",
                f"{s['source_split']}|{s['goal_category']}|{s['goal_specificity']}",
                f"{s['source_split']}|{s['goal_category']}",
                str(s["source_split"]),
            ]
        )

    strata = [c[0] for c in candidates]
    for level in range(4):
        counts = Counter(strata)
        next_strata: list[str] = []
        changed = False
        for i, stratum in enumerate(strata):
            if counts[stratum] >= n_splits or level == 3:
                next_strata.append(stratum)
            else:
                next_strata.append(candidates[i][level + 1])
                changed = True
        strata = next_strata
        if not changed:
            break
    return strata


def gold_by_turn(item: dict[str, Any]) -> dict[int, str]:
    out: dict[int, str] = {}
    for c in item["conversations"]:
        if c["role"] == "music":
            out[int(c["turn_number"])] = str(c["content"])
    return out


def row_records(
    session_fold: dict[tuple[str, str], int],
    source_splits: Iterable[str] = PUBLIC_SOURCE_SPLITS,
) -> list[dict[str, Any]]:
    """Return one record per labeled turn from the requested splits."""
    rows: list[dict[str, Any]] = []
    row_id = 0
    for source_split in source_splits:
        if source_split not in DATASET_SPLIT_BY_SOURCE:
            raise ValueError(f"unknown source split: {source_split}")
        ds = load("dataset", split=DATASET_SPLIT_BY_SOURCE[source_split])
        for item in ds:
            fold = session_fold[(source_split, item["session_id"])]
            gold = gold_by_turn(item)
            for turn in range(1, MAX_TURNS + 1):
                rows.append(
                    {
                        "public_row_id": row_id,
                        "source_split": source_split,
                        "session_id": item["session_id"],
                        "user_id": item["user_id"],
                        "turn_number": turn,
                        "fold": fold,
                        "gold_track_id": gold[turn],
                    }
                )
                row_id += 1
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    temp_path.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
