"""Retriever-only experiment 共有ユーティリティ。

numpy / datasets ベースの軽量関数のみ。torch には依存しない (encoder は別ファイル)。
retriever component 全体で共通して使う。
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .data import load
from .submission import InferenceInput

_LAST_N_RE = re.compile(r"^last_n:(\d+)$")


def _to_dense(values: list, dim: int | None = None) -> np.ndarray:
    """list[list[float]] (一部 None / 長さゼロ可) を [N, D] 行列にする。

    欠損行は 0 ベクトルで埋める。dim を渡さない場合は最頻長を採用する。
    """
    if dim is None:
        lengths = [len(v) for v in values if v is not None and len(v) > 0]
        if not lengths:
            raise ValueError("no non-empty embeddings found")
        dim = Counter(lengths).most_common(1)[0][0]
    out = np.zeros((len(values), dim), dtype=np.float32)
    for i, v in enumerate(values):
        if v is None or len(v) != dim:
            continue
        out[i] = np.asarray(v, dtype=np.float32)
    return out


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=-1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return (x / denom).astype(np.float32)


def load_track_matrix(
    column: str,
    split: str = "all_tracks",
    normalize: bool = True,
) -> tuple[list[str], np.ndarray]:
    """track_emb から指定カラムを取り出して (track_ids, [N, D]) を返す。"""
    emb = load("track_emb", split=split)
    track_ids = list(emb["track_id"])
    mat = _to_dense(emb[column])
    if normalize:
        mat = _l2_normalize(mat)
    return track_ids, mat


def topk_dot(query_mat: np.ndarray, track_mat: np.ndarray, k: int = 20) -> np.ndarray:
    """[B, D] @ [N, D].T → top-k indices [B, k]、score 降順。"""
    if query_mat.ndim == 1:
        query_mat = query_mat[None, :]
    scores = query_mat @ track_mat.T  # [B, N]
    n = scores.shape[1]
    if k >= n:
        return np.argsort(-scores, axis=1)
    part = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    rows = np.arange(scores.shape[0])[:, None]
    sub_scores = scores[rows, part]
    order = np.argsort(-sub_scores, axis=1)
    return part[rows, order]


def history_track_ids(inp: InferenceInput) -> list[str]:
    """chat_history の role=='music' content (track_id) を順序保存で抽出する。"""
    return [m["content"] for m in inp.chat_history if m.get("role") == "music"]


_EXPAND_FIELDS = ("track_name", "artist_name", "album_name", "release_date", "tag_list")
_TRACK_META_CACHE: dict[str, dict] | None = None


def _track_meta_lookup() -> dict[str, dict]:
    """track_id → metadata dict の lookup を遅延ロード + キャッシュする。"""
    global _TRACK_META_CACHE
    if _TRACK_META_CACHE is None:
        meta = load("track", split="all_tracks")
        _TRACK_META_CACHE = {row["track_id"]: row for row in meta}
    return _TRACK_META_CACHE


def _expand_music(track_id: str) -> str:
    """history の music turn の content (= track_id) を retrieve に効くテキストに展開する。

    `track_name / artist_name / album_name / release_date / tag_list` を ``", ".join`` で
    一行にまとめて lower-case する。track_id 単体だと BM25 / dense どちらにも効かないため、
    本ライブラリでは常に展開するのが default。
    """
    md = _track_meta_lookup().get(track_id)
    if md is None:
        return track_id
    parts = [f"track_id: {track_id}"]
    for field in _EXPAND_FIELDS:
        v = md.get(field)
        if v is None:
            continue
        if isinstance(v, list):
            joined = ", ".join(str(x) for x in v if x is not None and str(x))
        else:
            joined = str(v)
        if joined:
            parts.append(f"{field}: {joined.lower()}")
    return ", ".join(parts)


def chat_to_query_text(inp: InferenceInput, mode: str = "full") -> str:
    """chat_history と user_query を retriever 入力文字列に整形する。

    `role=='music'` turn は ``role='assistant'`` + track metadata 文字列に展開する
    (track_id 単体では retrieve に効かないので)。

    - ``full``       : 全 history を "role: content" で連結 + 末尾 user_query
    - ``last_user``  : 末尾 user_query 単体 (history は使わない)
    - ``last_n:N``   : 直近 N 個の history turn + 末尾 user_query
    - ``drop_music`` : history から role=='music' を捨てて連結 + 末尾 user_query
    - ``user_only``  : history の role=='user' のみ + 末尾 user_query
    """
    history = inp.chat_history
    if mode == "last_user":
        return inp.user_query
    m = _LAST_N_RE.match(mode)
    if m:
        n = int(m.group(1))
        history = history[-n:] if n > 0 else []
    elif mode == "drop_music":
        history = [c for c in history if c.get("role") != "music"]
    elif mode == "user_only":
        history = [c for c in history if c.get("role") == "user"]
    elif mode != "full":
        raise ValueError(f"unknown query mode: {mode}")

    parts: list[str] = []
    for c in history:
        role = c.get("role", "user")
        content = c.get("content", "")
        if role == "music":
            role = "assistant"
            content = _expand_music(content)
        parts.append(f"{role}: {content}")
    parts.append(f"user: {inp.user_query}")
    return "\n".join(parts)


def rrf_fuse(rankings: list[list[str]], k_const: int = 60, topk: int = 20) -> list[str]:
    """Reciprocal Rank Fusion。

    各 retriever の ranked id list を 1 つの ranked list に融合する。
    score(tid) = Σ 1 / (k_const + rank)
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, tid in enumerate(ranking, start=1):
            scores[tid] = scores.get(tid, 0.0) + 1.0 / (k_const + rank)
    return [tid for tid, _ in sorted(scores.items(), key=lambda kv: -kv[1])[:topk]]
