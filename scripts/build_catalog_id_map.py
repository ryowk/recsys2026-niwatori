#!/usr/bin/env python3
"""Build an external-to-catalog track ID map from paired datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from recsys2026.artifacts import file_ref, json_dump, utc_now
from recsys2026.data import load
from recsys2026.paths import REPO_ROOT


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def conversation_signature(
    item: dict[str, Any],
) -> tuple[tuple[str, int | None, str], ...]:
    sig: list[tuple[str, int | None, str]] = []
    for turn in item.get("conversations") or []:
        role = str(turn.get("role") or "")
        turn_number = turn.get("turn_number")
        turn_i = int(turn_number) if turn_number is not None else None
        content = "<MUSIC>" if role == "music" else normalize_text(turn.get("content"))
        sig.append((role, turn_i, content))
    return tuple(sig)


def music_contents(item: dict[str, Any]) -> list[str]:
    return [
        str(turn.get("content") or "")
        for turn in item.get("conversations") or []
        if turn.get("role") == "music" and turn.get("content")
    ]


def paired_rows(split: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    challenge_split = "train" if split == "train" else "test"
    challenge_rows = list(load("dataset", split=challenge_split))
    tpd2_rows = list(load_dataset("talkpl-ai/TalkPlayData-2", split=split))
    if len(challenge_rows) != len(tpd2_rows):
        raise ValueError(
            f"row count mismatch split={split}: challenge={len(challenge_rows)} tpd2={len(tpd2_rows)}"
        )
    return challenge_rows, tpd2_rows


def build_mapping() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tracks = list(load("track", split="all_tracks"))
    id_to_idx = {str(row["track_id"]): i for i, row in enumerate(tracks)}

    external_to_catalog: dict[str, str] = {}
    catalog_to_external: dict[str, str] = {}
    pair_counts: Counter[tuple[str, str]] = Counter()
    split_stats: dict[str, Any] = {}

    for split in ("train", "test"):
        challenge_rows, tpd2_rows = paired_rows(split)
        n_signature_ok = 0
        n_music_pairs = 0
        for i, (challenge_item, tpd2_item) in enumerate(
            zip(challenge_rows, tpd2_rows, strict=True)
        ):
            if conversation_signature(challenge_item) != conversation_signature(
                tpd2_item
            ):
                raise ValueError(
                    f"conversation signature mismatch split={split} row={i}"
                )
            n_signature_ok += 1
            catalog_tracks = music_contents(challenge_item)
            external_tracks = music_contents(tpd2_item)
            if len(catalog_tracks) != len(external_tracks):
                raise ValueError(f"music turn count mismatch split={split} row={i}")
            for catalog_id, external_id in zip(
                catalog_tracks, external_tracks, strict=True
            ):
                if catalog_id not in id_to_idx:
                    raise ValueError(
                        f"challenge track ID missing from catalog: {catalog_id}"
                    )
                pair_counts[(external_id, catalog_id)] += 1
                n_music_pairs += 1
                old_catalog = external_to_catalog.setdefault(external_id, catalog_id)
                if old_catalog != catalog_id:
                    raise ValueError(
                        f"external ID conflict: {external_id} -> "
                        f"{old_catalog} / {catalog_id}"
                    )
                old_external = catalog_to_external.setdefault(catalog_id, external_id)
                if old_external != external_id:
                    raise ValueError(
                        f"catalog ID conflict: {catalog_id} -> "
                        f"{old_external} / {external_id}"
                    )
        split_stats[split] = {
            "rows": len(challenge_rows),
            "signature_ok": n_signature_ok,
            "music_pairs": n_music_pairs,
        }

    rows = [
        {
            "external_track_id": external_id,
            "track_id": catalog_id,
            "track_idx": int(id_to_idx[catalog_id]),
            "pair_count": int(pair_counts[(external_id, catalog_id)]),
        }
        for external_id, catalog_id in external_to_catalog.items()
    ]
    rows.sort(key=lambda row: int(row["track_idx"]))
    stats = {
        "n_mapped_tracks": len(rows),
        "catalog_tracks": len(tracks),
        "catalog_coverage": len(rows) / len(tracks),
        "split_stats": split_stats,
        "conflicts": 0,
    }
    return rows, stats


def write_mapping(
    out_path: Path, rows: list[dict[str, Any]], stats: dict[str, Any]
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "external_track_id": [row["external_track_id"] for row in rows],
            "track_id": [row["track_id"] for row in rows],
            "track_idx": [row["track_idx"] for row in rows],
            "pair_count": [row["pair_count"] for row in rows],
        }
    )
    temp_path = out_path.with_name(f".{out_path.name}.tmp")
    pq.write_table(table, temp_path)
    temp_path.replace(out_path)
    manifest = {
        "schema_version": 1,
        "artifact_type": "catalog_id_map",
        "created_at": utc_now(),
        "producer": {
            "command": [
                "uv",
                "run",
                "python",
                "scripts/build_catalog_id_map.py",
                *sys.argv[1:],
            ],
            "cwd": ".",
        },
        "source_code": {
            "script": file_ref(REPO_ROOT / "scripts/build_catalog_id_map.py")
        },
        "sources": {
            "challenge_dataset": "talkpl-ai/TalkPlayData-Challenge-Dataset train/test",
            "talkplaydata2": "talkpl-ai/TalkPlayData-2 train/test",
            "track_metadata": "talkpl-ai/TalkPlayData-Challenge-Track-Metadata all_tracks",
        },
        "method": "same split row order with non-music conversation signature validation; music turns paired by order",
        "stats": stats,
        "leak_check": {
            "uses_blind_for_fit": False,
            "uses_track_emb_test_tracks": False,
            "uses_target_future_turns": False,
            "maps_ids_only": True,
        },
        "output": file_ref(out_path),
    }
    json_dump(out_path.with_suffix(".manifest.json"), manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "artifacts/preprocessed/catalog_id_map.parquet",
    )
    args = parser.parse_args()

    out_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
    manifest_path = out_path.with_suffix(".manifest.json")
    if out_path.exists() and manifest_path.exists():
        try:
            table = pq.read_table(
                out_path, columns=["external_track_id", "track_id", "track_idx"]
            )
            manifest = json.loads(manifest_path.read_text())
            if len(table) and manifest.get("artifact_type") == "catalog_id_map":
                print(f"[skip] {out_path}")
                print(json.dumps(manifest, indent=2))
                return
        except (json.JSONDecodeError, OSError, ValueError):
            pass
        manifest_path.unlink(missing_ok=True)
        print(f"rebuilding incomplete catalog ID map: {out_path}")

    rows, stats = build_mapping()
    write_mapping(out_path, rows, stats)
    print(json.dumps(stats, indent=2))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
