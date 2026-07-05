"""018_dense_qwen_recoded: 003/015 と同じ metadata 特徴量を **自前で Qwen3 にエンコード**して dense retrieve.

004 (= 事前計算済 `metadata-qwen3_embedding_0.6b`) と違って、track 側を query 側と同じ
`Qwen3TextEncoder` で encode し直す。これで「track 側 embedding と query 側 encoder の空間ずれ」
仮説 (004 README 参照) が正しいかを直接検証する。

corpus は 003 と同じ 5 fields (track_name, artist_name, album_name, release_date, tag_list) を
``f"{field}: {value}\\n"`` で連結。

track 行列はキャッシュ (`output/018_dense_qwen_recoded/track_emb.npz`) して、2 回目以降は
encode をスキップする。

使い方:
    just run 018_dense_qwen_recoded --target devset
    just run 018_dense_qwen_recoded --target devset --no_cache    # 再エンコード強制
    just run 018_dense_qwen_recoded --target devset --query_mode drop_music   # 011 知見の活用
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from recsys2026.data import load
from recsys2026.encoders import Qwen3TextEncoder
from recsys2026.eval import evaluate_devset
from recsys2026.paths import CACHE_DIR, OUTPUT_DIR as _OUTPUT_ROOT, RESULTS_DIR as _RESULTS_ROOT
from recsys2026.retrieval import chat_to_query_text, topk_dot
from recsys2026.submission import (
    InferenceInput,
    Prediction,
    Target,
    format_record,
    iter_inputs,
    write_predictions,
    zip_submission,
)

OUT_DIR = _OUTPUT_ROOT / "dense_track_encoder"
RESULTS_DIR = _RESULTS_ROOT / "dense_track_encoder"
TOP_K = 20

CORPUS_FIELDS = ("track_name", "artist_name", "album_name", "release_date", "tag_list")
TRACK_EMB_CACHE = CACHE_DIR / "dense_track_emb.npz"


def _stringify(row: dict) -> str:
    """003 と同じ format. `field: value\\n` × 5 fields."""
    out = ""
    for field in CORPUS_FIELDS:
        v = row.get(field)
        if v is None:
            continue
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v if x is not None and str(x))
        out += f"{field}: {v}\n"
    return out


def _build_or_load_track_matrix(
    encoder: Qwen3TextEncoder, use_cache: bool
) -> tuple[list[str], np.ndarray]:
    if use_cache and TRACK_EMB_CACHE.exists():
        data = np.load(TRACK_EMB_CACHE, allow_pickle=False)
        return data["track_ids"].tolist(), data["embeddings"]

    print("Encoding 47k tracks with Qwen3TextEncoder (this takes a few minutes)...")
    meta = load("track", split="all_tracks")
    track_ids: list[str] = list(meta["track_id"])
    texts = [_stringify(row) for row in meta]
    # encode in chunks with progress bar
    chunk = 1024
    parts = []
    for i in tqdm(range(0, len(texts), chunk), desc="encode tracks"):
        parts.append(encoder.encode(texts[i : i + chunk]))
    track_mat = np.concatenate(parts, axis=0).astype(np.float32)
    TRACK_EMB_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        TRACK_EMB_CACHE,
        track_ids=np.array(track_ids),
        embeddings=track_mat,
    )
    return track_ids, track_mat


class DensePredictor:
    def __init__(
        self,
        encoder_batch_size: int = 64,
        query_mode: str = "full",
        use_cache: bool = True,
    ) -> None:
        self.query_mode = query_mode
        self.encoder = Qwen3TextEncoder(batch_size=encoder_batch_size)
        self.track_ids, self.track_mat = _build_or_load_track_matrix(
            self.encoder, use_cache
        )

    def predict(self, batch: list[InferenceInput]) -> list[Prediction]:
        queries = [chat_to_query_text(inp, mode=self.query_mode) for inp in batch]
        query_mat = self.encoder.encode(queries)
        idx_mat = topk_dot(query_mat, self.track_mat, k=TOP_K)
        return [
            Prediction(
                track_ids=[self.track_ids[i] for i in row], response=""
            )
            for row in idx_mat
        ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", choices=("devset", "blind_a", "blind_b"), default="devset"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_infer_inputs", type=int, default=None)
    parser.add_argument(
        "--query_mode",
        type=str,
        default="full",
        help="chat_to_query_text の mode (full / last_user / last_n:N / drop_music / user_only)",
    )
    parser.add_argument("--no_cache", action="store_true")
    args = parser.parse_args()

    target: Target = args.target
    predictor = DensePredictor(
        encoder_batch_size=args.batch_size,
        query_mode=args.query_mode,
        use_cache=not args.no_cache,
    )

    inputs = list(iter_inputs(target))
    smoke = args.max_infer_inputs is not None
    if smoke:
        inputs = inputs[: args.max_infer_inputs]

    records: list[dict] = []
    for i in tqdm(range(0, len(inputs), args.batch_size), desc=f"infer[{target}]"):
        batch = inputs[i : i + args.batch_size]
        preds = predictor.predict(batch)
        for inp, pred in zip(batch, preds, strict=True):
            records.append(format_record(inp, pred.track_ids, pred.response))

    out_path = OUT_DIR / f"{target}{'.smoke' if smoke else ''}.json"
    if smoke:
        out_path.write_text(json.dumps(records, ensure_ascii=False))
        print(f"wrote {out_path} (smoke; validation skipped)")
    else:
        write_predictions(records, out_path, target)
        print(f"wrote {out_path}")

    if target == "devset" and not smoke:
        scores = evaluate_devset(out_path)
        scores["query_mode"] = args.query_mode
        scores_path = RESULTS_DIR / "scores.json"
        scores_path.write_text(json.dumps(scores, indent=2))
        print(json.dumps(scores, indent=2))
    elif target in ("blind_a", "blind_b") and not smoke:
        zip_path = zip_submission(out_path)
        print(f"wrote {zip_path}")


if __name__ == "__main__":
    main()
