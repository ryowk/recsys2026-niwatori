"""Fixed split helpers for the component pipeline protocol."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .data import load

PUBLIC_SOURCE_SPLITS = ("train", "devset")
DATASET_SPLIT_BY_SOURCE = {"train": "train", "devset": "test"}
MAX_TURNS = 8


def stable_hash(text: str, *, seed: int) -> int:
    raw = f"{seed}:{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def _goal(item: dict[str, Any]) -> dict[str, Any]:
    return dict(item.get("conversation_goal") or {})


def _profile(item: dict[str, Any]) -> dict[str, Any]:
    return dict(item.get("user_profile") or {})


def session_records() -> list[dict[str, Any]]:
    """Return one record per public labeled session."""
    rows: list[dict[str, Any]] = []
    for source_split in PUBLIC_SOURCE_SPLITS:
        ds = load("dataset", split=DATASET_SPLIT_BY_SOURCE[source_split])
        for item in ds:
            goal = _goal(item)
            profile = _profile(item)
            rows.append(
                {
                    "source_split": source_split,
                    "session_id": item["session_id"],
                    "user_id": item["user_id"],
                    "session_date": item.get("session_date"),
                    "goal_category": str(goal.get("category") or "NA"),
                    "goal_specificity": str(goal.get("specificity") or "NA"),
                    "user_split": str(profile.get("user_split") or "NA"),
                    "preferred_language": str(profile.get("preferred_language") or "NA"),
                    "preferred_musical_culture": str(profile.get("preferred_musical_culture") or "NA"),
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


def row_records(session_fold: dict[tuple[str, str], int]) -> list[dict[str, Any]]:
    """Return one record per labeled turn in train+devset."""
    rows: list[dict[str, Any]] = []
    row_id = 0
    for source_split in PUBLIC_SOURCE_SPLITS:
        ds = load("dataset", split=DATASET_SPLIT_BY_SOURCE[source_split])
        for item in ds:
            goal = _goal(item)
            profile = _profile(item)
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
                        "session_date": item.get("session_date"),
                        "goal_category": str(goal.get("category") or "NA"),
                        "goal_specificity": str(goal.get("specificity") or "NA"),
                        "user_split": str(profile.get("user_split") or "NA"),
                    }
                )
                row_id += 1
    return rows


def weighted_nested_order(rows: Iterable[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    """Return a deterministic, roughly stratified row order.

    Prefixes of this order are nested smoke subsets.  The group scheduler uses
    weighted fair queuing over source/fold/turn/goal_category groups.
    """
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{row['source_split']}|f{row['fold']}|t{row['turn_number']}|{row['goal_category']}"
        groups[key].append(row)
    for key, vals in groups.items():
        vals.sort(
            key=lambda r: stable_hash(
                f"{key}:{r['session_id']}:{r['turn_number']}:{r['public_row_id']}",
                seed=seed,
            )
        )

    total = sum(len(v) for v in groups.values())
    group_sizes = {k: len(v) for k, v in groups.items()}
    taken = {k: 0 for k in groups}
    positions = {k: 0 for k in groups}
    group_tiebreak = {k: stable_hash(k, seed=seed) for k in groups}
    order: list[dict[str, Any]] = []
    while len(order) < total:
        next_rank = len(order) + 1
        best_key = max(
            (k for k, vals in groups.items() if positions[k] < len(vals)),
            key=lambda k: (
                next_rank * group_sizes[k] / total - taken[k],
                -group_tiebreak[k],
            ),
        )
        order.append(groups[best_key][positions[best_key]])
        positions[best_key] += 1
        taken[best_key] += 1
    return order


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def subset_keys(path: Path, *, source_split: str | None = None) -> set[tuple[str, int]]:
    keys: set[tuple[str, int]] = set()
    for row in read_jsonl(path):
        if source_split is not None and row.get("source_split") != source_split:
            continue
        keys.add((str(row["session_id"]), int(row["turn_number"])))
    return keys
