"""提出 JSON の入出力 + validation。

実験ごとに制御フロー(バッチサイズ、多段推論、キャッシュ、postprocess 等)は変わるので、
共通化するのは「提出スキーマに関わる部分のみ」に限定する。

公開 API:
- Target           : "devset" | "blind_a" | "blind_b"
- iter_inputs      : target の差を吸収して InferenceInput を yield
- format_record    : 1 (session, turn) を提出 dict に整形
- write_predictions: validate してから JSON 出力
- validate_predictions: 提出スキーマ検証(全ペア充足、最大 20 件、重複なし、catalog 内 ID)
- InferenceInput / Prediction / Predictor: per-record 推論を書くときの便利な型 (optional)

実験の制御フロー本体は exp/<name>/main.py に書く。run_inference のような canonical な
ループは敢えて提供しない(多段 pipeline や全予測 postprocess の障害になるため)。
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .data import load

Target = Literal["devset", "blind_a", "blind_b"]
MAX_K = 20


@dataclass(frozen=True)
class InferenceInput:
    session_id: str
    user_id: str
    turn_number: int
    chat_history: list[dict]   # raw [{turn_number, role, content}, ...]、role=music は track_id 文字列
    user_query: str


@dataclass(frozen=True)
class Prediction:
    track_ids: list[str]       # 関連度順、最大 MAX_K、重複不可
    response: str


class Predictor(Protocol):
    """per-record 予測を書くときの便利な構造的型。実験ごとに自由なので任意採用。"""

    def predict(self, batch: list[InferenceInput]) -> list[Prediction]: ...


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


def _iter_blind(target: Literal["blind_a", "blind_b"]) -> Iterator[InferenceInput]:
    ds = load(target, split="test")
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
    elif target in ("blind_a", "blind_b"):
        yield from _iter_blind(target)
    else:
        raise ValueError(f"unknown target: {target}")


def format_record(inp: InferenceInput, track_ids: list[str], response: str) -> dict:
    """1 (session, turn) ペアを提出形式 dict に整形する。"""
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
    """提出スキーマを検証。違反があれば ValueError。

    - 全 (session_id, turn_number) ペアが target に対して埋まっているか
    - 各 record で predicted_track_ids が最大 MAX_K (20) 件、重複なし
    - 全 track_id が catalog (Track-Metadata.all_tracks) に存在する
    """
    expected_all = {(inp.session_id, inp.turn_number) for inp in iter_inputs(target)}
    expected = allowed_keys if allowed_keys is not None else expected_all
    unknown_allowed = expected - expected_all
    if unknown_allowed:
        raise ValueError(f"allowed_keys contains keys outside target={target}, e.g. {next(iter(unknown_allowed))}")
    catalog = set(load("track", split="all_tracks")["track_id"])
    seen: set[tuple[str, int]] = set()
    for r in records:
        key = (r["session_id"], r["turn_number"])
        if key in seen:
            raise ValueError(f"duplicate prediction for {key}")
        seen.add(key)
        if key not in expected:
            raise ValueError(f"unexpected (session_id, turn_number) for target={target}: {key}")
        tids = r["predicted_track_ids"]
        if len(tids) > MAX_K:
            raise ValueError(f"too many tracks for {key}: {len(tids)} > {MAX_K}")
        if len(tids) != len(set(tids)):
            raise ValueError(f"duplicate track_ids in {key}")
        unknown = set(tids) - catalog
        if unknown:
            raise ValueError(f"unknown track_ids in {key}: e.g. {next(iter(unknown))}")
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
    """validate_predictions してから JSON を書き出す。"""
    validate_predictions(records, target, require_complete=require_complete, allowed_keys=allowed_keys)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False))


def zip_submission(json_path: Path, zip_path: Path | None = None) -> Path:
    """予測 JSON を Codabench 提出用 zip に固める。

    Codabench の要件で zip 内の JSON のファイル名は exact match で `prediction.json` でなければ
    scoring fail するため、ここで arcname を rename する。元の json_path のファイル名は任意で OK。

    zip_path 省略時は元 JSON と同じディレクトリに `<stem>.submission.zip` を生成。
    """
    json_path = Path(json_path)
    if zip_path is None:
        zip_path = json_path.with_suffix(".submission.zip")
    else:
        zip_path = Path(zip_path)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(json_path, arcname="prediction.json")
    return zip_path
