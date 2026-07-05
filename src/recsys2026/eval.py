"""devset 用の評価メトリクス: nDCG@{1,10,20} + catalog/lexical diversity。

公式 music-crs-evaluator/metrics/ の振る舞いを再現する。スコア drift がないかは、
同一予測 JSON を公式 evaluator にも食わせて parity 確認する。
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from .data import load

K_VALUES: tuple[int, ...] = (1, 10, 20)


def _devset_ground_truth() -> dict[tuple[str, int], str]:
    """(session_id, turn_number) -> 正解 track_id。devset の各 conversation から抽出。"""
    ds = load("dataset", split="test")
    gt: dict[tuple[str, int], str] = {}
    for item in ds:
        sid = item["session_id"]
        for c in item["conversations"]:
            if c["role"] == "music":
                gt[(sid, c["turn_number"])] = c["content"]
    return gt


def _ndcg_at_k(preds: Sequence[str], gold: Sequence[str], k: int) -> float:
    """公式 metrics_recsys.get_ndcg と同じ式。"""
    preds = preds[:k]
    gold_set = set(gold)
    dcg = 0.0
    for i, p in enumerate(preds, start=1):
        if p in gold_set:
            dcg += 1.0 / math.log2(i + 1)
    n_rel = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0


def _catalog_diversity(all_recommended: Sequence[str], catalog_size: int) -> float:
    if catalog_size <= 0:
        return 0.0
    return len(set(all_recommended)) / float(catalog_size)


def _lexical_diversity(responses: Sequence[str], n: int = 2) -> float:
    """distinct-n。公式準拠で lower() + whitespace split、トークン < n はスキップ。"""
    ngrams: set[tuple[str, ...]] = set()
    total = 0
    for resp in responses:
        tokens = (resp or "").lower().split()
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngrams.add(tuple(tokens[i : i + n]))
            total += 1
    return len(ngrams) / float(total) if total else 0.0


def evaluate_devset_records(
    predictions: list[dict],
    *,
    require_complete: bool = True,
) -> dict:
    """devset 予測 records を評価する。

    require_complete=False の場合は、smoke subset などの部分 devset を許容する。
    その場合も、与えられた record に未知 key / duplicate があれば error にする。
    """
    gt = _devset_ground_truth()

    # per (session, turn) で nDCG を計算
    per_turn: dict[int, list[dict[str, float]]] = defaultdict(list)
    all_recommended: list[str] = []
    all_responses: list[str] = []
    seen_keys: set[tuple[str, int]] = set()

    for p in predictions:
        sid = p["session_id"]
        turn = p["turn_number"]
        key = (sid, turn)
        if key in seen_keys:
            raise ValueError(f"duplicate prediction for {key}")
        seen_keys.add(key)
        if key not in gt:
            raise KeyError(f"no ground truth for {key}")

        pred_tracks: list[str] = list(p["predicted_track_ids"])
        if len(pred_tracks) != len(set(pred_tracks)):
            raise ValueError(f"duplicate track_ids in prediction for {key}")

        per_turn[turn].append({f"ndcg@{k}": _ndcg_at_k(pred_tracks, [gt[key]], k) for k in K_VALUES})
        all_recommended.extend(pred_tracks)
        all_responses.append(p["predicted_response"])

    missing = set(gt) - seen_keys
    if require_complete and missing:
        raise KeyError(f"predictions missing for {len(missing)} (session, turn) pairs, e.g. {next(iter(missing))}")
    if not per_turn:
        raise ValueError("no devset predictions to evaluate")

    # turn ごとに平均 → turn 軸で平均 (公式 evaluate_devset.py:64-66 と同じ)
    turn_means = {
        turn: {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
        for turn, rows in per_turn.items()
    }
    n_turns = len(turn_means)
    macro: dict[str, float] = {}
    for k in (f"ndcg@{kk}" for kk in K_VALUES):
        macro[k] = sum(tm[k] for tm in turn_means.values()) / n_turns

    catalog = load("track", split="all_tracks")
    macro["catalog_diversity"] = _catalog_diversity(all_recommended, len(catalog))
    macro["lexical_diversity"] = _lexical_diversity(all_responses)
    macro["total_catalog_size"] = len(catalog)
    macro["n_examples"] = len(predictions)
    macro["require_complete"] = require_complete
    return macro


def evaluate_devset(predictions_path: Path) -> dict:
    """devset の予測 JSON を読み、マクロ平均メトリクスを dict で返す。"""
    predictions = json.loads(Path(predictions_path).read_text())
    return evaluate_devset_records(predictions, require_complete=True)
