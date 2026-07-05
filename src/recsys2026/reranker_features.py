"""Reranker feature / example library.

Provides `TrackIndex`, `FeatureEncoder`, the example builders
(`build_examples_from_dataset` / `build_examples_from_blind`), dense query
encoding, user vectors, and candidate metrics. Loaded by the reranker
(`scripts/run_reranker.py` via `recsys2026.reranker_protocol`) to build the
176-feature LightGBM LambdaRank matrix: track / user / turn basics, history
consistency, query–metadata similarity (TF-IDF + dense cosine on
Qwen3-Embedding vectors), and intent-tag features. Blind-B-safe: goal / thought
/ GPA fields are never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import bm25s
import lightgbm as lgb
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from recsys2026.data import load
from recsys2026.eval import evaluate_devset
from recsys2026.paths import CACHE_DIR, OUTPUT_DIR as _OUTPUT_ROOT, RESULTS_DIR as _RESULTS_ROOT
from recsys2026.retrieval import chat_to_query_text
from recsys2026.submission import (
    InferenceInput,
    Target,
    format_record,
    write_predictions,
    zip_submission,
)

OUT_DIR = _OUTPUT_ROOT / "reranker_features"
RESULTS_DIR = _RESULTS_ROOT / "reranker_features"
REUSE_CACHE_OUT_DIR = _OUTPUT_ROOT / "093_current_thought_goal_query"  # legacy cache reuse (unused here)

EMB_COL = "cf-bpr"
MAX_TURNS = 8
TEXT_RE = re.compile(r"[a-z0-9]+")
MISSING_CAT = "<missing>"

# 020_bm25_history_boost と同じ corpus 構成 (= 015 公式 baseline 4 fields).
# --include_tags を渡すと 003_bm25_with_tags と同じ 5 fields に切り替わる.
CORPUS_FIELDS_4 = ("track_name", "artist_name", "album_name", "release_date")
CORPUS_FIELDS_5 = ("track_name", "artist_name", "album_name", "tag_list", "release_date")

# 018_dense_qwen_recoded で生成済の track 側 dense embedding (Qwen3-Embedding-0.6B).
DENSE_TRACK_EMB_CACHE = CACHE_DIR / "dense_track_emb.npz"


@dataclass(frozen=True)
class TurnExample:
    session_id: str
    user_id: str
    turn_number: int
    user_profile: dict
    conversation_goal: dict
    chat_history: list[dict]
    user_query: str
    user_query_thought: str
    prior_goal_progress: list[str | None]
    gold_track_id: str | None


@dataclass(frozen=True)
class CandidateSet:
    indices: np.ndarray  # [N, K] int32
    scores: np.ndarray   # [N, K] float32 (BM25+boost score)


def as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_text(value: object) -> str:
    values = [str(v) for v in as_list(value) if v is not None and str(v)]
    return values[0] if values else ""


def parse_year(value: object) -> int:
    text = str(value or "")
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return 0


def tokens(text: str) -> set[str]:
    return set(TEXT_RE.findall(text.lower()))


def stable_bucket(*parts: object, modulo: int) -> int:
    raw = "|".join(str(p) for p in parts).encode()
    return int(hashlib.md5(raw).hexdigest(), 16) % modulo


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0.0] = 1.0
    return x / denom


def dense_embedding_matrix(values: list, dim: int | None = None) -> np.ndarray:
    if dim is None:
        lengths = [len(v) for v in values if v is not None and len(v) > 0]
        if not lengths:
            raise ValueError("no non-empty embeddings found")
        dim = Counter(lengths).most_common(1)[0][0]
    out = np.zeros((len(values), dim), dtype=np.float32)
    for i, value in enumerate(values):
        if value is None or len(value) != dim:
            continue
        out[i] = np.asarray(value, dtype=np.float32)
    return out


def _stringify_corpus(row: dict, fields: tuple[str, ...]) -> str:
    out = ""
    for field in fields:
        value = row.get(field)
        if value is None:
            continue
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value if v is not None and str(v))
        out += f"{field}: {value}\n"
    return out


def _id_to_metadata_str(track_id: str, meta: dict, fields: tuple[str, ...]) -> str:
    s = f"track_id: {track_id}"
    for ct in fields:
        v = meta.get(ct) or []
        if isinstance(v, str):
            v = [v]
        s += f", {ct}: {', '.join(str(x) for x in v).lower()}"
    return s


def _blind_b_safe() -> bool:
    """Blind-B-safe is fixed ON in this repository: conversation_goal + current
    thought are excluded from all query-text construction, and target-GPA /
    goal-derived base features are blanked (Blind B does not provide these
    fields; train and inference must match)."""
    return True


def _goal_text_from_goal(goal: dict | None) -> str:
    if _blind_b_safe():
        return ""
    goal = goal or {}
    return " ".join(
        str(goal.get(k) or "")
        for k in ("category", "specificity", "listener_goal")
    ).strip()


def _query_plus_goal(user_query: str, goal: dict | None, thought: str = "") -> str:
    goal_text_value = _goal_text_from_goal(goal)
    query = str(user_query or "").strip()
    thought_text = "" if _blind_b_safe() else str(thought or "").strip()
    query_parts = [p for p in (query, thought_text) if p]
    query_text = "\n".join(query_parts)
    if not goal_text_value:
        return query_text or "music"
    if not query_text:
        return goal_text_value
    return f"{goal_text_value}\n{query_text}"


def _bm25_query_text(
    inp_or_ex, track_meta: dict[str, dict], fields: tuple[str, ...]
) -> str:
    """020_bm25_history_boost の `_official_query` と同じロジック.

    chat_history の role==music は track_id を metadata 文字列に展開して assistant 化.
    091: blind にも含まれる conversation_goal を候補生成 query に追加する.
    """
    parts: list[str] = []
    for c in inp_or_ex.chat_history:
        role = c.get("role", "user")
        content = c.get("content", "")
        if role == "music":
            md = track_meta.get(content)
            if md is not None:
                role = "assistant"
                content = _id_to_metadata_str(content, md, fields)
        parts.append(f"{role}: {content}")
    goal = _goal_text_from_goal(getattr(inp_or_ex, "conversation_goal", {}) or {})
    if goal:
        parts.append(f"goal: {goal}")
    thought = "" if _blind_b_safe() else (getattr(inp_or_ex, "user_query_thought", "") or "")
    if thought:
        parts.append(f"user: {inp_or_ex.user_query} {thought}")
    else:
        parts.append(f"user: {inp_or_ex.user_query}")
    return "\n".join(parts).lower()


class TrackIndex:
    """track 側の features (metadata + cf-bpr) と BM25 index をまとめて保持."""

    def __init__(
        self,
        universe: Literal["all_tracks"],
        corpus_fields: tuple[str, ...] = CORPUS_FIELDS_4,
        secondary_corpus_fields: tuple[str, ...] | None = None,
        load_dense: bool = False,
    ) -> None:
        self.universe = universe
        self.corpus_fields = corpus_fields
        self.secondary_corpus_fields = secondary_corpus_fields
        self.load_dense = load_dense
        meta = load("track", split="all_tracks")
        meta_by_id = {row["track_id"]: row for row in meta}

        emb = load("track_emb", split=universe)
        self.track_ids: list[str] = list(emb["track_id"])
        self.id_to_idx = {tid: i for i, tid in enumerate(self.track_ids)}
        self.cf = normalize_rows(dense_embedding_matrix(emb[EMB_COL]))

        self.track_name: list[str] = []
        self.artist_name: list[str] = []
        self.album_name: list[str] = []
        self.primary_tag: list[str] = []
        self.tag_tokens: list[set[str]] = []
        self.popularity: np.ndarray = np.zeros(len(self.track_ids), dtype=np.float32)
        self.duration: np.ndarray = np.zeros(len(self.track_ids), dtype=np.float32)
        self.release_year: np.ndarray = np.zeros(len(self.track_ids), dtype=np.float32)
        self.texts: list[str] = []
        # raw meta (BM25 query 構築 / boost id lookup 用)
        self.meta_by_id: dict[str, dict] = meta_by_id

        # 候補宇宙は all_tracks 固定. test_tracks は dev/blind 側所属情報なので使わない.
        bm25_corpus: list[str] = []

        artist_to_idx: dict[str, list[int]] = {}
        album_to_idx: dict[str, list[int]] = {}

        for i, tid in enumerate(self.track_ids):
            row = meta_by_id.get(tid, {})
            name = first_text(row.get("track_name"))
            artist = first_text(row.get("artist_name"))
            album = first_text(row.get("album_name"))
            tags = [str(t) for t in as_list(row.get("tag_list")) if t]
            tag_text = " ".join(tags)

            self.track_name.append(name)
            self.artist_name.append(artist)
            self.album_name.append(album)
            self.primary_tag.append(tags[0] if tags else "")
            self.tag_tokens.append(tokens(tag_text))
            self.popularity[i] = float(row.get("popularity") or 0.0)
            self.duration[i] = float(row.get("duration") or 0.0)
            self.release_year[i] = float(parse_year(row.get("release_date")))
            self.texts.append(" ".join([name, artist, album, tag_text]).strip())
            bm25_corpus.append(_stringify_corpus(row, corpus_fields))

            for aid in as_list(row.get("artist_id")):
                if aid:
                    artist_to_idx.setdefault(str(aid), []).append(i)
            for alid in as_list(row.get("album_id")):
                if alid:
                    album_to_idx.setdefault(str(alid), []).append(i)

        self.artist_to_idx = artist_to_idx
        self.album_to_idx = album_to_idx

        pop = self.popularity.copy()
        if pop.max() > pop.min():
            pop = (pop - pop.min()) / (pop.max() - pop.min())
        self.popularity_norm = pop.astype(np.float32)

        # build BM25 index
        print(f"building BM25 index over {len(bm25_corpus)} tracks ({corpus_fields}) ...")
        self.bm25 = bm25s.BM25()
        self.bm25.index(
            bm25s.tokenize(bm25_corpus, show_progress=False), show_progress=False
        )
        self.n_tracks = len(self.track_ids)

        # secondary BM25 index (multi-source RRF 用).
        self.bm25_b: bm25s.BM25 | None = None
        if secondary_corpus_fields is not None:
            print(
                f"building 2nd BM25 index over {self.n_tracks} tracks ({secondary_corpus_fields}) ..."
            )
            corpus_b = [
                _stringify_corpus(meta_by_id.get(tid, {}), secondary_corpus_fields)
                for tid in self.track_ids
            ]
            self.bm25_b = bm25s.BM25()
            self.bm25_b.index(
                bm25s.tokenize(corpus_b, show_progress=False), show_progress=False
            )

        # 018 で生成済の Qwen3 dense embedding を流用 (track 側).
        # query 側は実行時に Qwen3TextEncoder で encode する.
        self.dense_emb: np.ndarray | None = None
        if load_dense:
            if not DENSE_TRACK_EMB_CACHE.exists():
                raise FileNotFoundError(
                    f"dense track embedding cache not found: {DENSE_TRACK_EMB_CACHE}.\n"
                    "先に 018_dense_qwen_recoded を smoke でも良いので走らせて track_emb.npz を作成してください."
                )
            print(f"loading 018 dense track embeddings from {DENSE_TRACK_EMB_CACHE.name} ...")
            data = np.load(DENSE_TRACK_EMB_CACHE, allow_pickle=False)
            cached_ids = data["track_ids"].tolist()
            cached_emb = data["embeddings"]  # 既に L2 正規化済 (018 が正規化して保存)
            id_to_pos = {tid: i for i, tid in enumerate(cached_ids)}
            dense = np.zeros(
                (self.n_tracks, cached_emb.shape[1]), dtype=np.float32
            )
            missing = 0
            for i, tid in enumerate(self.track_ids):
                pos = id_to_pos.get(tid)
                if pos is None:
                    missing += 1
                    continue
                dense[i] = cached_emb[pos]
            if missing:
                print(
                    f"warning: {missing}/{self.n_tracks} tracks missing in 018 cache."
                )
            self.dense_emb = dense


def history_artist_album_played(
    ex: TurnExample, track_meta: dict[str, dict]
) -> tuple[set[str], set[str], set[str]]:
    """chat_history の music turn から (artist_ids, album_ids, played_track_ids)."""
    a_ids: set[str] = set()
    al_ids: set[str] = set()
    played: set[str] = set()
    for c in ex.chat_history:
        if c.get("role") != "music":
            continue
        tid = c.get("content")
        if not tid:
            continue
        played.add(tid)
        md = track_meta.get(tid)
        if md is None:
            continue
        for x in as_list(md.get("artist_id")):
            if x:
                a_ids.add(str(x))
        for x in as_list(md.get("album_id")):
            if x:
                al_ids.add(str(x))
    return a_ids, al_ids, played


# -------------------- intent extraction (Qwen2.5-1.5B-Instruct) --------------------


INTENT_SYSTEM_PROMPT = (
    "Extract music recommendation intent from the user's request as compact JSON. "
    "Use exactly these keys: genre, mood, era, artist, activity. "
    "Each value is a short string (≤4 words) or empty string if not specified. "
    "era should be a 4-digit year, decade like \"1990s\", or empty. "
    "Output ONLY a single line of valid JSON, nothing else."
)


_INTENT_KEYS = ("genre", "mood", "era", "artist", "activity")


def _parse_intent(raw: str) -> dict[str, str]:
    text = raw.strip()
    lb = text.find("{")
    rb = text.rfind("}")
    if lb < 0 or rb <= lb:
        return {}
    snippet = text[lb : rb + 1]
    try:
        obj = json.loads(snippet)
    except json.JSONDecodeError:
        return {}
    if not isinstance(obj, dict):
        return {}
    return {k: str(obj.get(k, "")).strip() for k in _INTENT_KEYS}


class IntentExtractor:
    """Qwen2.5-1.5B-Instruct で user_query → 構造化 intent JSON."""

    DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 32,
        max_new_tokens: int = 120,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
            .to(self.device)
            .eval()
        )

    def _format_one(self, user_query: str) -> str:
        msgs = [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )

    def batch_extract(self, user_queries: list[str]) -> list[dict[str, str]]:
        import torch

        out: list[dict[str, str]] = []
        for start in range(0, len(user_queries), self.batch_size):
            chunk = user_queries[start : start + self.batch_size]
            prompts = [self._format_one(t) for t in chunk]
            enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            with torch.no_grad():
                generated = self.model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=False,
                )
            new_tokens = generated[:, enc["input_ids"].shape[1] :]
            decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for s in decoded:
                out.append(_parse_intent(s))
        return out


def pre_extract_intent(
    examples: list[TurnExample],
    extractor: IntentExtractor,
    cache_path: Path,
    use_cache: bool,
    desc: str = "intent",
) -> dict[str, dict[str, str]]:
    keys = [f"{ex.session_id}:{ex.turn_number}" for ex in examples]
    if use_cache and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            cached_keys = cached.get("keys") or []
            cached_intents = cached.get("intents") or []
            if cached_keys == keys and len(cached_intents) == len(keys):
                print(f"loaded cached intents from {cache_path.name}")
                return {k: i for k, i in zip(keys, cached_intents, strict=True)}
        except (json.JSONDecodeError, OSError):
            pass

    user_queries = [
        _query_plus_goal(ex.user_query, ex.conversation_goal, ex.user_query_thought)
        for ex in examples
    ]
    chunk = extractor.batch_size
    intents: list[dict[str, str]] = []
    for i in tqdm(range(0, len(user_queries), chunk), desc=desc):
        intents.extend(extractor.batch_extract(user_queries[i : i + chunk]))
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"keys": keys, "intents": intents}))
    return {k: i for k, i in zip(keys, intents, strict=True)}


def _intent_era_year(era: str) -> int:
    s = (era or "").strip().lower()
    if not s:
        return 0
    if len(s) >= 4 and s[:4].isdigit():
        return int(s[:4])
    digits = "".join(c for c in s if c.isdigit())
    if not digits:
        return 0
    if len(digits) == 2:
        v = int(digits)
        return 2000 + v if v < 30 else 1900 + v
    if len(digits) >= 4:
        return int(digits[:4])
    return 0


def load_user_vectors() -> dict[str, np.ndarray]:
    user_emb = load("user_emb")
    vectors: dict[str, np.ndarray] = {}
    for split in user_emb:
        for row in user_emb[split]:
            if row[EMB_COL] is None or len(row[EMB_COL]) == 0:
                continue
            vec = np.asarray(row[EMB_COL], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vectors[row["user_id"]] = vec / norm
    return vectors


def goal_text(goal: dict) -> str:
    return _goal_text_from_goal(goal)


def conversation_text(ex: TurnExample, track_index: TrackIndex | None = None) -> str:
    parts = [_query_plus_goal(ex.user_query, ex.conversation_goal, ex.user_query_thought)]
    for msg in ex.chat_history:
        role = msg.get("role")
        content = str(msg.get("content") or "")
        if role in {"user", "assistant"}:
            parts.append(content)
        elif role == "music" and track_index is not None:
            idx = track_index.id_to_idx.get(content)
            if idx is not None:
                parts.append(track_index.texts[idx])
    return " ".join(parts).strip()


def build_examples_from_dataset(
    split: Literal["train", "test"],
    max_sessions: int | None = None,
) -> list[TurnExample]:
    ds = load("dataset", split=split)
    examples: list[TurnExample] = []
    for item_idx, item in enumerate(ds):
        if max_sessions is not None and item_idx >= max_sessions:
            break
        conversations = list(item["conversations"])
        progress = list(item["goal_progress_assessments"])
        for target_turn in range(1, MAX_TURNS + 1):
            current = [c for c in conversations if c["turn_number"] == target_turn]
            user_turn = next(c for c in current if c["role"] == "user")
            user_query = str(user_turn["content"])
            user_query_thought = str(user_turn.get("thought") or "").strip()
            gold = next(c["content"] for c in current if c["role"] == "music")
            examples.append(
                TurnExample(
                    session_id=item["session_id"],
                    user_id=item["user_id"],
                    turn_number=target_turn,
                    user_profile=dict(item["user_profile"] or {}),
                    conversation_goal=dict(item["conversation_goal"] or {}),
                    chat_history=[c for c in conversations if c["turn_number"] < target_turn],
                    user_query=user_query,
                    user_query_thought=user_query_thought,
                    prior_goal_progress=[
                        g.get("goal_progress_assessment")
                        for g in progress
                        if g.get("turn_number", 0) < target_turn
                    ],
                    gold_track_id=gold,
                )
            )
    return examples


def build_examples_from_blind(
    target: Literal["blind_a", "blind_b"],
    max_inputs: int | None = None,
) -> list[TurnExample]:
    ds = load(target, split="test")
    examples: list[TurnExample] = []
    for item_idx, item in enumerate(ds):
        if max_inputs is not None and item_idx >= max_inputs:
            break
        conversations = list(item["conversations"])
        current = conversations[-1]
        user_query_thought = (
            str(current.get("thought") or "").strip()
            if current.get("role") == "user"
            else ""
        )
        examples.append(
            TurnExample(
                session_id=item["session_id"],
                user_id=item["user_id"],
                turn_number=int(current["turn_number"]),
                user_profile=dict(item.get("user_profile") or {}),
                conversation_goal=dict(item.get("conversation_goal") or {}),
                chat_history=conversations[:-1],
                user_query=str(current["content"]),
                user_query_thought=user_query_thought,
                prior_goal_progress=[
                    g.get("goal_progress_assessment")
                    for g in item.get("goal_progress_assessments", [])
                    if g.get("turn_number", 0) < int(current["turn_number"])
                ],
                gold_track_id=None,
            )
        )
    return examples


def split_train_valid(
    examples: list[TurnExample],
    valid_fraction: float,
    seed: int,
) -> tuple[list[TurnExample], list[TurnExample]]:
    sessions = sorted({ex.session_id for ex in examples})
    rng = random.Random(seed)
    rng.shuffle(sessions)
    n_valid = max(1, int(round(len(sessions) * valid_fraction)))
    valid_sessions = set(sessions[:n_valid])
    train = [ex for ex in examples if ex.session_id not in valid_sessions]
    valid = [ex for ex in examples if ex.session_id in valid_sessions]
    return train, valid


# -------------------- 候補生成 (BM25 + history boost) --------------------


def _build_score_vec(
    ex: TurnExample,
    track_index: TrackIndex,
    bm25_index: "bm25s.BM25",
    corpus_fields: tuple[str, ...],
    artist_boost: float,
    album_boost: float,
    h_arts: set[str],
    h_albs: set[str],
) -> np.ndarray:
    """全 47k tracks に対する BM25+boost スコアベクトル."""
    query = _bm25_query_text(ex, track_index.meta_by_id, corpus_fields)
    toks = bm25s.tokenize([query], show_progress=False)
    idx_arr, score_arr = bm25_index.retrieve(
        toks, k=track_index.n_tracks, show_progress=False
    )
    scores = np.zeros(track_index.n_tracks, dtype=np.float32)
    scores[idx_arr[0]] = score_arr[0]

    if artist_boost > 0:
        for aid in h_arts:
            idxs = track_index.artist_to_idx.get(aid)
            if idxs:
                scores[idxs] += artist_boost
    if album_boost > 0:
        for alid in h_albs:
            idxs = track_index.album_to_idx.get(alid)
            if idxs:
                scores[idxs] += album_boost
    return scores


def _ranks_from_scores(scores: np.ndarray) -> np.ndarray:
    """score 降順での 1-origin rank を返す."""
    order = np.argsort(-scores, kind="stable")
    rank = np.empty_like(scores, dtype=np.float32)
    rank[order] = np.arange(1, len(scores) + 1, dtype=np.float32)
    return rank


def _bm25_candidate_one(
    ex: TurnExample,
    track_index: TrackIndex,
    artist_boost: float,
    album_boost: float,
    candidate_k: int,
    exclude_history: bool,
    dense_query_vec: np.ndarray | None = None,
    n_bm25: int | None = None,
    rrf_k: float = 60.0,
) -> tuple[np.ndarray, np.ndarray]:
    """1 example について 候補集合 + BM25+boost score を返す.

    038 のロジック:
      - dense_query_vec が None: 純 BM25+boost top-K (= 024 と同じ).
      - dense_query_vec が指定 + dense_emb 有: union 候補.
        BM25 top-`n_bm25` と dense top-(K - n_bm25) を合体, dedupe.
        BM25 部分が `n_bm25` 件に達しない場合は dense でその分を補う.

    score は BM25+boost score (dense-only 候補も BM25+boost score をそのまま使う,
    LGBM が `dense_only_candidate` 等の他 feature で「dense 由来」を区別する).

    Returns:
        indices: [K] int32 (track_index 内の index, 末尾は -1 の sentinel になり得る)
        scores:  [K] float32 (BM25+boost score, sentinel は -inf)
    """
    h_arts, h_albs, played = history_artist_album_played(ex, track_index.meta_by_id)

    scores_bm25 = _build_score_vec(
        ex,
        track_index,
        track_index.bm25,
        track_index.corpus_fields,
        artist_boost,
        album_boost,
        h_arts,
        h_albs,
    )

    if exclude_history and played:
        for tid in played:
            j = track_index.id_to_idx.get(tid)
            if j is not None:
                scores_bm25[j] = -np.inf

    use_union = (
        dense_query_vec is not None
        and track_index.dense_emb is not None
        and n_bm25 is not None
        and 0 < n_bm25 < candidate_k
    )

    if not use_union:
        # 024 / 035 と同じ: BM25+boost top-K のみ.
        k = min(candidate_k, track_index.n_tracks)
        top = np.argpartition(-scores_bm25, k - 1)[:k]
        top = top[np.argsort(-scores_bm25[top])]
        return top.astype(np.int32), scores_bm25[top].astype(np.float32)

    # Union path.
    n_bm25_eff = int(min(n_bm25, track_index.n_tracks))
    n_extra = candidate_k - n_bm25_eff

    bm25_top = np.argpartition(-scores_bm25, n_bm25_eff - 1)[:n_bm25_eff]
    bm25_top = bm25_top[np.argsort(-scores_bm25[bm25_top])]
    # 全 BM25 候補が -inf (履歴除外で全滅) のときは scores_bm25[bm25_top[i]]==-inf になる.
    # dense top で穴埋めする (下記のとおり).
    bm25_set = set(bm25_top.tolist())

    scores_dense = (track_index.dense_emb @ dense_query_vec).astype(np.float32)
    if exclude_history and played:
        for tid in played:
            j = track_index.id_to_idx.get(tid)
            if j is not None:
                scores_dense[j] = -np.inf

    # BM25 で取った tracks は dense では無視 (同じ track を dedupe).
    scores_dense_filt = scores_dense.copy()
    for j in bm25_set:
        scores_dense_filt[j] = -np.inf

    # dense top-n_extra を取る.
    if n_extra > 0:
        n_extra_eff = int(min(n_extra, track_index.n_tracks))
        dense_top = np.argpartition(-scores_dense_filt, n_extra_eff - 1)[:n_extra_eff]
        dense_top = dense_top[np.argsort(-scores_dense_filt[dense_top])]
    else:
        dense_top = np.array([], dtype=np.int64)

    combined = np.concatenate([bm25_top, dense_top])

    # サイズ調整.
    if len(combined) > candidate_k:
        combined = combined[:candidate_k]
    elif len(combined) < candidate_k:
        # ありえないはず (47k tracks > 300+履歴) だが念のため -1 padding.
        pad = np.full(candidate_k - len(combined), -1, dtype=combined.dtype)
        combined = np.concatenate([combined, pad])

    # 全 candidate の score = BM25+boost score (dense-only でもそれを使う).
    out_scores = np.full(candidate_k, -np.inf, dtype=np.float32)
    valid = combined >= 0
    out_scores[valid] = scores_bm25[combined[valid]]

    return combined.astype(np.int32), out_scores


def _to_inference_input_for_query(ex: TurnExample) -> InferenceInput:
    return InferenceInput(
        session_id=ex.session_id,
        user_id=ex.user_id,
        turn_number=ex.turn_number,
        chat_history=ex.chat_history,
        user_query=_query_plus_goal(
            ex.user_query,
            ex.conversation_goal,
            ex.user_query_thought,
        ),
    )


def encode_dense_queries(
    examples: list[TurnExample],
    encoder,
    query_mode: str,
    cache_path: Path,
    use_cache: bool,
    desc: str = "dense_query",
) -> np.ndarray:
    """各 example の chat_to_query_text(mode=query_mode) を Qwen3 で encode → cache."""
    keys = np.asarray([f"{ex.session_id}:{ex.turn_number}" for ex in examples])
    if use_cache:
        for path in (cache_path, REUSE_CACHE_OUT_DIR / cache_path.name):
            if not path.exists():
                continue
            cached = np.load(path, allow_pickle=False)
            if np.array_equal(cached["keys"], keys):
                print(f"loaded cached dense queries from {path}")
                return cached["embeddings"]

    queries = [
        chat_to_query_text(_to_inference_input_for_query(ex), mode=query_mode)
        for ex in examples
    ]
    chunk = max(1, getattr(encoder, "batch_size", 64))
    parts: list[np.ndarray] = []
    for i in tqdm(range(0, len(queries), chunk), desc=desc):
        parts.append(encoder.encode(queries[i : i + chunk]))
    emb = np.concatenate(parts, axis=0).astype(np.float32)
    if use_cache:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, keys=keys, embeddings=emb)
    return emb


def candidate_cache_path(
    name: str,
    examples: list[TurnExample],
    candidate_k: int,
    universe: str,
    artist_boost: float,
    album_boost: float,
    exclude_history: bool,
) -> Path:
    keys = ",".join(f"{ex.session_id}:{ex.turn_number}" for ex in examples[:10])
    digest = hashlib.md5(
        (
            f"{name}|{len(examples)}|{candidate_k}|{universe}|"
            f"{artist_boost}|{album_boost}|{exclude_history}|{keys}"
        ).encode()
    ).hexdigest()[:10]
    return OUT_DIR / f"candidates_{name}_{digest}.npz"


def generate_candidates(
    examples: list[TurnExample],
    track_index: TrackIndex,
    candidate_k: int,
    artist_boost: float,
    album_boost: float,
    exclude_history: bool,
    cache_name: str,
    use_cache: bool,
    desc: str,
    dense_query_emb: np.ndarray | None = None,
    n_bm25: int | None = None,
) -> CandidateSet:
    cache_path = candidate_cache_path(
        cache_name,
        examples,
        candidate_k,
        track_index.universe,
        artist_boost,
        album_boost,
        exclude_history,
    )
    keys = np.asarray([f"{ex.session_id}:{ex.turn_number}" for ex in examples])
    if use_cache:
        for path in (cache_path, REUSE_CACHE_OUT_DIR / cache_path.name):
            if not path.exists():
                continue
            cached = np.load(path, allow_pickle=False)
            if np.array_equal(cached["keys"], keys):
                print(f"loaded cached candidates from {path}")
                return CandidateSet(indices=cached["indices"], scores=cached["scores"])

    k = min(candidate_k, track_index.n_tracks)
    indices = np.empty((len(examples), k), dtype=np.int32)
    scores_out = np.empty((len(examples), k), dtype=np.float32)

    for i, ex in enumerate(tqdm(examples, desc=desc)):
        dvec = None
        if dense_query_emb is not None:
            dvec = dense_query_emb[i]
        idx_row, score_row = _bm25_candidate_one(
            ex,
            track_index,
            artist_boost,
            album_boost,
            candidate_k,
            exclude_history,
            dense_query_vec=dvec,
            n_bm25=n_bm25,
        )
        indices[i] = idx_row
        scores_out[i] = score_row

    if use_cache:
        np.savez_compressed(cache_path, keys=keys, indices=indices, scores=scores_out)
    return CandidateSet(indices=indices, scores=scores_out)


# -------------------- 特徴量 / 学習 (001 / 020_v2 と同等) --------------------


class FeatureEncoder:
    numeric_names = [
        "candidate_rank",
        "log_candidate_rank",
        "reciprocal_candidate_rank",
        "candidate_score",
        "track_popularity",
        "log_track_duration",
        "track_release_year",
        "user_age",
        "turn_number",
        "history_music_count",
        "same_artist_history_count",
        "same_album_history_count",
        "same_track_seen",
        "prior_gpa_count",
        "prior_moves_toward_count",
        "prior_not_move_count",
        "prior_null_gpa_count",
        "goal_track_tfidf_sim",
        "conversation_track_tfidf_sim",
        "query_track_tfidf_sim",
        "tag_token_overlap",
        "user_has_cf_embedding",
        "same_primary_tag_history_count",
        "tag_token_overlap_history",
        "last_music_same_artist",
        "last_music_same_album",
        "cf_bpr_history_cosine",
        "history_year_diff",
        # explicit user×track interaction. user.preferred_musical_culture
        # ("Western Alternative Rock" 等) を空白で token 化して track.tag_tokens
        # との set 積を取る. LGBM が user_preferred_musical_culture × track_artist
        # の交互作用を学習しているところを直接 numeric で渡す.
        "user_culture_tag_overlap",
        "user_country_tag_overlap",
        "user_lang_tag_overlap",
        # 035 追加: user_query (Qwen3) と track dense_emb (018 cache) の cosine.
        # candidates は BM25 のまま, dense は feature 側のみで使う.
        "query_dense_track_cosine",
        "query_dense_history_centroid_cosine",
        # 038 追加: union 候補で dense-only か BM25 由来かを区別.
        "dense_only_candidate",
        # LLM 抽出 intent と track 側 metadata の照合 feature.
        "intent_genre_in_track_tags",
        "intent_mood_in_track_tags",
        "intent_descriptor_in_track_tags",
        "intent_artist_token_overlap",
        "intent_era_year_diff",
    ]
    # 098: 093 current-thought query expansion + 096 profile robustness.
    # User profile categorical splits overfit session-specific priors, so keep
    # only goal/progress categories.
    categorical_names = [
        "goal_category",
        "goal_specificity",
        "latest_goal_progress",
    ]

    def __init__(self, track_index: TrackIndex, user_vectors: dict[str, np.ndarray]) -> None:
        self.track_index = track_index
        self.user_vectors = user_vectors
        self.maps: dict[str, dict[str, int]] = {}

    @property
    def feature_names(self) -> list[str]:
        return self.numeric_names + self.categorical_names

    @property
    def categorical_feature_indices(self) -> list[int]:
        offset = len(self.numeric_names)
        return list(range(offset, offset + len(self.categorical_names)))

    def fit_categories(self, examples: list[TurnExample]) -> None:
        values: dict[str, set[str]] = {name: {MISSING_CAT} for name in self.categorical_names}
        for ex in examples:
            cats = self.example_categories(ex)
            for key, value in cats.items():
                if key in values:
                    values[key].add(value)
        self.maps = {
            key: {value: j + 1 for j, value in enumerate(sorted(vals))}
            for key, vals in values.items()
        }

    def example_categories(self, ex: TurnExample) -> dict[str, str]:
        profile = ex.user_profile
        if _blind_b_safe():
            goal = {}
            latest = None
        else:
            goal = ex.conversation_goal
            latest = next((x for x in reversed(ex.prior_goal_progress) if x), None)
        return {
            "user_age_group": str(profile.get("age_group") or MISSING_CAT),
            "user_country_code": str(profile.get("country_code") or MISSING_CAT),
            "user_gender": str(profile.get("gender") or MISSING_CAT),
            "user_preferred_language": str(profile.get("preferred_language") or MISSING_CAT),
            "user_preferred_musical_culture": str(profile.get("preferred_musical_culture") or MISSING_CAT),
            "user_split": str(profile.get("user_split") or MISSING_CAT),
            "goal_category": str(goal.get("category") or MISSING_CAT),
            "goal_specificity": str(goal.get("specificity") or MISSING_CAT),
            "latest_goal_progress": str(latest or MISSING_CAT),
        }

    def encode_cat(self, name: str, value: str) -> float:
        return float(self.maps[name].get(value or MISSING_CAT, 0))


def selected_positions(
    ex: TurnExample,
    cand_indices: np.ndarray,
    track_index: TrackIndex,
    negatives_per_group: int | None,
) -> list[int]:
    # 038: -1 sentinel (padding) を除外する.
    valid_positions = [i for i in range(len(cand_indices)) if int(cand_indices[i]) >= 0]
    if negatives_per_group is None:
        return valid_positions
    gold_idx = track_index.id_to_idx.get(ex.gold_track_id or "")
    if gold_idx is None:
        return []
    pos = np.flatnonzero(cand_indices == gold_idx)
    if len(pos) == 0:
        return []
    pos_i = int(pos[0])
    negatives = [i for i in valid_positions if i != pos_i]
    chosen = negatives[:negatives_per_group] + [pos_i]
    chosen = sorted(set(chosen), key=chosen.index)
    return chosen


def build_feature_matrix(
    examples: list[TurnExample],
    candidates: CandidateSet,
    encoder: FeatureEncoder,
    vectorizer: TfidfVectorizer,
    track_tfidf,
    negatives_per_group: int | None,
    chunk_examples: int,
    query_dense_emb: np.ndarray | None = None,
    n_bm25: int | None = None,
    intent_lookup: dict[str, dict[str, str]] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[int]]:
    """035 追加: ``query_dense_emb`` は examples と同じ index で並んだ Qwen3 query 行列.

    None の場合は feature 値 0.0 で埋める.
    ``intent_lookup`` は ``f"{session}:{turn}"`` -> intent dict.
    """
    chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    group_sizes: list[int] = []
    n_features = len(encoder.feature_names)

    for start in range(0, len(examples), chunk_examples):
        end = min(start + chunk_examples, len(examples))
        ex_chunk = examples[start:end]
        goal_vecs = vectorizer.transform([goal_text(ex.conversation_goal) for ex in ex_chunk])
        conv_vecs = vectorizer.transform(
            [conversation_text(ex, encoder.track_index) for ex in ex_chunk]
        )
        query_vecs = vectorizer.transform(
            [
                _query_plus_goal(ex.user_query, ex.conversation_goal, ex.user_query_thought)
                for ex in ex_chunk
            ]
        )

        rows: list[list[float]] = []
        labels: list[int] = []
        for local_i, ex in enumerate(ex_chunk):
            global_i = start + local_i
            cand_idx = candidates.indices[global_i]
            cand_scores = candidates.scores[global_i]
            positions = selected_positions(ex, cand_idx, encoder.track_index, negatives_per_group)
            if not positions:
                continue
            group_sizes.append(len(positions))

            goal_sim = (goal_vecs[local_i] @ track_tfidf[cand_idx].T).toarray().ravel()
            conv_sim = (conv_vecs[local_i] @ track_tfidf[cand_idx].T).toarray().ravel()
            query_sim = (query_vecs[local_i] @ track_tfidf[cand_idx].T).toarray().ravel()

            history_tracks = [
                str(msg.get("content"))
                for msg in ex.chat_history
                if msg.get("role") == "music"
            ]
            history_indices = [
                encoder.track_index.id_to_idx[tid]
                for tid in history_tracks
                if tid in encoder.track_index.id_to_idx
            ]
            history_artists = Counter(encoder.track_index.artist_name[i] for i in history_indices)
            history_albums = Counter(encoder.track_index.album_name[i] for i in history_indices)
            history_primary_tags = Counter(
                encoder.track_index.primary_tag[i] for i in history_indices
            )
            seen_tracks = set(history_indices)
            last_music_idx = history_indices[-1] if history_indices else None
            last_music_artist = (
                encoder.track_index.artist_name[last_music_idx]
                if last_music_idx is not None
                else None
            )
            last_music_album = (
                encoder.track_index.album_name[last_music_idx]
                if last_music_idx is not None
                else None
            )
            history_year_mean = (
                float(np.mean([encoder.track_index.release_year[i] for i in history_indices]))
                if history_indices
                else 0.0
            )
            if history_indices:
                hist_cf_arr = encoder.track_index.cf[
                    np.asarray(history_indices, dtype=np.int32)
                ]
                hist_centroid = hist_cf_arr.mean(axis=0)
                hcn = float(np.linalg.norm(hist_centroid))
                if hcn > 0:
                    hist_centroid = hist_centroid / hcn
                else:
                    hist_centroid = None
            else:
                hist_centroid = None
            # 035: dense (Qwen3) ベースの query × track / history centroid 用意.
            query_dense_vec = (
                query_dense_emb[global_i] if query_dense_emb is not None else None
            )
            if (
                query_dense_vec is not None
                and encoder.track_index.dense_emb is not None
                and history_indices
            ):
                hist_dense_arr = encoder.track_index.dense_emb[
                    np.asarray(history_indices, dtype=np.int32)
                ]
                hist_dense_centroid = hist_dense_arr.mean(axis=0)
                hdn = float(np.linalg.norm(hist_dense_centroid))
                hist_dense_centroid = (
                    hist_dense_centroid / hdn if hdn > 0 else None
                )
                query_history_dense_cos = (
                    float(query_dense_vec @ hist_dense_centroid)
                    if hist_dense_centroid is not None
                    else 0.0
                )
            else:
                query_history_dense_cos = 0.0
            context_tokens = tokens(goal_text(ex.conversation_goal) + " " + conversation_text(ex))
            history_tag_token_lists = [
                encoder.track_index.tag_tokens[i] for i in history_indices
            ]
            profile = ex.user_profile
            prior = [] if _blind_b_safe() else [x for x in ex.prior_goal_progress]
            prior_counter = Counter(str(x) for x in prior)
            example_cats = encoder.example_categories(ex)
            age = float(profile.get("age") or 0.0)
            # explicit user×track interaction tokens.
            culture_tokens = tokens(str(profile.get("preferred_musical_culture") or ""))
            country_tokens = tokens(str(profile.get("country_name") or ""))
            lang_tokens = tokens(str(profile.get("preferred_language") or ""))
            # LLM 抽出 intent.
            intent = (
                intent_lookup.get(f"{ex.session_id}:{ex.turn_number}")
                if intent_lookup is not None
                else None
            ) or {}
            intent_genre_tokens = tokens(str(intent.get("genre") or ""))
            intent_mood_tokens = tokens(str(intent.get("mood") or ""))
            intent_activity_tokens = tokens(str(intent.get("activity") or ""))
            intent_artist_tokens = tokens(str(intent.get("artist") or ""))
            intent_descriptor_tokens = (
                intent_genre_tokens | intent_mood_tokens | intent_activity_tokens
            )
            intent_era_year = float(_intent_era_year(str(intent.get("era") or "")))

            for pos in positions:
                idx = int(cand_idx[pos])
                rank = float(pos + 1)
                artist = encoder.track_index.artist_name[idx] or MISSING_CAT
                album = encoder.track_index.album_name[idx] or MISSING_CAT
                tag = encoder.track_index.primary_tag[idx] or MISSING_CAT
                # track_artist / track_album / track_primary_tag は categorical 化すると
                # 30k+ unique values で LGBM split が壊れるので feature から外す.
                # ただし same_artist_history_count などの履歴一致系で間接的に signal
                # は残る.
                cat_values = {**example_cats}
                cand_tag_tokens = encoder.track_index.tag_tokens[idx]
                cand_release_year = float(encoder.track_index.release_year[idx])
                same_primary_tag_history = float(history_primary_tags.get(tag, 0))
                tag_overlap_hist_sum = float(
                    sum(len(cand_tag_tokens & ht) for ht in history_tag_token_lists)
                )
                last_same_artist = float(
                    last_music_artist is not None and artist == last_music_artist
                )
                last_same_album = float(
                    last_music_album is not None and album == last_music_album
                )
                if hist_centroid is not None:
                    cf_hist_cos = float(encoder.track_index.cf[idx] @ hist_centroid)
                else:
                    cf_hist_cos = 0.0
                if history_indices and cand_release_year > 0 and history_year_mean > 0:
                    year_diff = float(abs(cand_release_year - history_year_mean))
                else:
                    year_diff = 0.0

                row = [
                    rank,
                    math.log1p(rank),
                    1.0 / rank,
                    float(cand_scores[pos]),
                    float(encoder.track_index.popularity[idx]),
                    math.log1p(float(encoder.track_index.duration[idx])),
                    cand_release_year,
                    age,
                    float(ex.turn_number),
                    float(len(history_indices)),
                    float(history_artists.get(artist, 0)),
                    float(history_albums.get(album, 0)),
                    float(idx in seen_tracks),
                    float(len(prior)),
                    float(prior_counter.get("MOVES_TOWARD_GOAL", 0)),
                    float(prior_counter.get("DOES_NOT_MOVE_TOWARD_GOAL", 0)),
                    float(prior_counter.get("None", 0) + prior_counter.get("", 0)),
                    float(goal_sim[pos]),
                    float(conv_sim[pos]),
                    float(query_sim[pos]),
                    float(len(context_tokens & cand_tag_tokens)),
                    float(ex.user_id in encoder.user_vectors),
                    same_primary_tag_history,
                    tag_overlap_hist_sum,
                    last_same_artist,
                    last_same_album,
                    cf_hist_cos,
                    year_diff,
                    float(len(culture_tokens & cand_tag_tokens)),
                    float(len(country_tokens & cand_tag_tokens)),
                    float(len(lang_tokens & cand_tag_tokens)),
                    # 035 追加: query_dense × track_dense cosine (BM25 と orthogonal な signal).
                    (
                        float(
                            query_dense_vec
                            @ encoder.track_index.dense_emb[idx]
                        )
                        if query_dense_vec is not None
                        and encoder.track_index.dense_emb is not None
                        else 0.0
                    ),
                    query_history_dense_cos,
                    # 038 追加: dense-only か (BM25 top-N に入らなかった候補なら 1).
                    float(n_bm25 is not None and pos >= n_bm25),
                    # intent → track 照合 features.
                    float(len(intent_genre_tokens & cand_tag_tokens)),
                    float(len(intent_mood_tokens & cand_tag_tokens)),
                    float(len(intent_descriptor_tokens & cand_tag_tokens)),
                    float(
                        len(
                            intent_artist_tokens
                            & tokens(encoder.track_index.artist_name[idx] or "")
                        )
                    ),
                    (
                        float(abs(cand_release_year - intent_era_year))
                        if intent_era_year > 0 and cand_release_year > 0
                        else 0.0
                    ),
                ]
                row.extend(
                    encoder.encode_cat(name, cat_values.get(name, MISSING_CAT))
                    for name in encoder.categorical_names
                )
                rows.append(row)
                if ex.gold_track_id is not None:
                    labels.append(int(encoder.track_index.track_ids[idx] == ex.gold_track_id))

        if rows:
            chunks.append(np.asarray(rows, dtype=np.float32).reshape(-1, n_features))
            if labels:
                label_chunks.append(np.asarray(labels, dtype=np.int8))

    if chunks:
        x = np.vstack(chunks)
    else:
        x = np.empty((0, n_features), dtype=np.float32)
    y = np.concatenate(label_chunks) if label_chunks else None
    return x, y, group_sizes


def candidate_metrics(
    examples: list[TurnExample],
    candidates: CandidateSet,
    track_index: TrackIndex,
    ks: tuple[int, ...] = (20, 50, 100, 200),
) -> dict[str, float]:
    out: dict[str, float] = {"groups": float(len(examples))}
    for k in ks:
        if k > candidates.indices.shape[1]:
            continue
        hits = []
        for ex, cand in zip(examples, candidates.indices, strict=True):
            gold_idx = track_index.id_to_idx.get(ex.gold_track_id or "")
            hit = gold_idx is not None and gold_idx in set(cand[:k])
            hits.append(float(hit))
        recall = float(np.mean(hits)) if hits else 0.0
        out[f"candidate_recall@{k}"] = recall
        out[f"candidate_precision@{k}"] = recall / float(k)
    return out


def ndcg_from_rank(rank: int | None, k: int) -> float:
    if rank is None or rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def ranking_metrics_from_ranked_indices(
    examples: list[TurnExample],
    ranked: list[list[int]],
    track_index: TrackIndex,
    k: int,
    prefix: str,
) -> dict[str, float]:
    ndcgs: list[float] = []
    recalls: list[float] = []
    mrrs: list[float] = []
    for ex, pred_indices in zip(examples, ranked, strict=True):
        gold_idx = track_index.id_to_idx.get(ex.gold_track_id or "")
        rank = None
        if gold_idx is not None:
            for i, idx in enumerate(pred_indices[:k], start=1):
                if idx == gold_idx:
                    rank = i
                    break
        ndcgs.append(ndcg_from_rank(rank, k))
        recalls.append(float(rank is not None and rank <= k))
        mrrs.append(0.0 if rank is None or rank > k else 1.0 / rank)
    return {
        f"{prefix}_ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"{prefix}_recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"{prefix}_mrr@{k}": float(np.mean(mrrs)) if mrrs else 0.0,
    }


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def raw_ranked(candidates: CandidateSet) -> list[list[int]]:
    return [list(map(int, row)) for row in candidates.indices]


def score_and_rank(
    model: lgb.LGBMRanker,
    examples: list[TurnExample],
    candidates: CandidateSet,
    encoder: FeatureEncoder,
    vectorizer: TfidfVectorizer,
    track_tfidf,
    chunk_examples: int,
    query_dense_emb: np.ndarray | None = None,
    n_bm25: int | None = None,
    intent_lookup: dict[str, dict[str, str]] | None = None,
) -> tuple[list[list[int]], list[np.ndarray]]:
    ranked: list[list[int]] = []
    score_rows: list[np.ndarray] = []
    for start in range(0, len(examples), chunk_examples):
        end = min(start + chunk_examples, len(examples))
        sub_examples = examples[start:end]
        sub_candidates = CandidateSet(
            indices=candidates.indices[start:end],
            scores=candidates.scores[start:end],
        )
        sub_query_dense = (
            query_dense_emb[start:end] if query_dense_emb is not None else None
        )
        x, _, group_sizes = build_feature_matrix(
            sub_examples,
            sub_candidates,
            encoder,
            vectorizer,
            track_tfidf,
            negatives_per_group=None,
            chunk_examples=chunk_examples,
            n_bm25=n_bm25,
            query_dense_emb=sub_query_dense,
            intent_lookup=intent_lookup,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            scores = model.predict(x)
        offset = 0
        for local_i, group_size in enumerate(group_sizes):
            group_scores = scores[offset : offset + group_size]
            offset += group_size
            order = np.argsort(-group_scores)
            indices = sub_candidates.indices[local_i][order]
            ranked.append(list(map(int, indices)))
            score_rows.append(group_scores[order])
    return ranked, score_rows


def prediction_response(ex: TurnExample, track_index: TrackIndex, top_idx: int) -> str:
    track = track_index.track_name[top_idx] or "this track"
    artist = track_index.artist_name[top_idx] or "the artist"
    goal = str(ex.conversation_goal.get("listener_goal") or ex.user_query).strip()
    goal = " ".join(goal.split())[:140]
    templates = [
        'I would start with "{track}" by {artist}. It fits your current request and keeps the recommendation close to: {goal}',
        'For this turn, "{track}" by {artist} is the strongest match I found. It should line up with the mood and preferences you described.',
        'My next pick is "{track}" by {artist}. It connects your recent feedback with the direction of this session.',
        'Try "{track}" by {artist}. It gives you a focused next step based on your profile, goal, and conversation so far.',
    ]
    template = templates[stable_bucket(ex.session_id, ex.turn_number, modulo=len(templates))]
    return template.format(track=track, artist=artist, goal=goal)


def to_inference_input(ex: TurnExample) -> InferenceInput:
    return InferenceInput(
        session_id=ex.session_id,
        user_id=ex.user_id,
        turn_number=ex.turn_number,
        chat_history=ex.chat_history,
        user_query=ex.user_query,
    )


def write_target_predictions(
    target: Target,
    examples: list[TurnExample],
    ranked: list[list[int]],
    track_index: TrackIndex,
    final_k: int,
) -> Path:
    records: list[dict] = []
    for ex, pred_indices in zip(examples, ranked, strict=True):
        track_ids: list[str] = []
        seen: set[str] = set()
        for idx in pred_indices:
            tid = track_index.track_ids[idx]
            if tid in seen:
                continue
            seen.add(tid)
            track_ids.append(tid)
            if len(track_ids) == final_k:
                break
        response = prediction_response(ex, track_index, pred_indices[0])
        records.append(format_record(to_inference_input(ex), track_ids, response))

    out = OUT_DIR / f"{target}.json"
    write_predictions(records, out, target)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=("devset", "blind_a", "blind_b"), default="devset")
    parser.add_argument("--candidate_k", type=int, default=300)
    parser.add_argument("--final_k", type=int, default=20)
    parser.add_argument("--feature_chunk_examples", type=int, default=512)
    parser.add_argument("--artist_boost", type=float, default=50.0)
    parser.add_argument("--album_boost", type=float, default=30.0)
    parser.add_argument("--valid_fraction", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--train_negatives_per_group",
        type=int,
        default=0,
        help="<=0 で全 candidate_k 個を group として lambdarank に渡す (推奨)."
        " 正の整数を渡すと top-N 負例 + gold だけに subsample するが、"
        " negatives が top BM25 ranks に偏って 'high BM25 = neg' を学習してしまうため非推奨.",
    )
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--num_leaves", type=int, default=63)
    parser.add_argument("--learning_rate", type=float, default=0.04)
    parser.add_argument("--n_jobs", type=int, default=8)
    parser.add_argument(
        "--lambdarank_truncation_level",
        type=int,
        default=0,
        help="lambdarank の nDCG@K truncation. <=0 で LightGBM default (30). "
        "実験的には 20 (final_k 一致) は default 30 より僅かに devset 悪化したので default は 0.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_sessions", type=int, default=None)
    parser.add_argument("--max_infer_inputs", type=int, default=None)
    parser.add_argument("--candidate_universe", choices=("all_tracks",), default="all_tracks")
    parser.add_argument("--allow_history_tracks", action="store_true")
    parser.add_argument("--no_candidate_cache", action="store_true")
    parser.add_argument(
        "--include_tags",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="BM25 corpus に tag_list を含める (= 003 と同じ 5 fields). True が default."
        " --no-include_tags で 020 と同じ 4 fields に戻る.",
    )
    parser.add_argument(
        "--multi_source",
        action="store_true",
        help="2 つの BM25 (4-field と 5-field) を index して、boost 込みのランクで RRF (k=60) で"
        " マージする. include_tags は無視される (4-field がプライマリ、5-field がセカンダリ).",
    )
    parser.add_argument(
        "--dense_rrf",
        action="store_true",
        help="018_dense_qwen_recoded の Qwen3 track embedding を読み込み、query 側を実行時に"
        " encode して BM25 候補と RRF (k=60) でマージする (BM25 と orthogonal な signal).",
    )
    parser.add_argument(
        "--dense_query_mode",
        type=str,
        default="drop_music",
        help="dense retrieval 用の chat_to_query_text mode. 011 query ablation の知見で"
        " dense は 'drop_music' / 'user_only' が一番効くので default は drop_music.",
    )
    parser.add_argument(
        "--query_dense_feature_mode",
        type=str,
        default="last_user",
        help="dense feature 用の query mode. 035 では user_query 単独で意味を引き出すのが"
        " 主目的なので default は 'last_user' (= user_query のみ, chat history 抜き).",
    )
    parser.add_argument(
        "--dense_encode_batch_size",
        type=int,
        default=64,
        help="Qwen3 で query を encode する batch size (GPU memory 次第).",
    )
    parser.add_argument(
        "--n_bm25",
        type=int,
        default=200,
        help="038: union 候補のうち BM25+boost top で埋める件数. 残りは dense top で埋まる. "
        "<=0 で union を無効化し純 BM25 (= 035 互換).",
    )
    parser.add_argument(
        "--intent",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="037: LLM (Qwen2.5-1.5B) で user_query → 構造化 intent JSON を抽出して match feature 化. default=True.",
    )
    parser.add_argument(
        "--intent_lm_model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument("--intent_batch_size", type=int, default=64)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    target: Target = args.target
    use_cache = not args.no_candidate_cache
    exclude_history = not args.allow_history_tracks
    negatives_per_group: int | None = (
        args.train_negatives_per_group
        if args.train_negatives_per_group and args.train_negatives_per_group > 0
        else None
    )

    print("loading tracks/users")
    if args.multi_source:
        corpus_fields = CORPUS_FIELDS_4
        secondary_corpus_fields: tuple[str, ...] | None = CORPUS_FIELDS_5
    else:
        corpus_fields = CORPUS_FIELDS_5 if args.include_tags else CORPUS_FIELDS_4
        secondary_corpus_fields = None
    # 035 は dense を **feature 側のみ** で常時 ON. RRF 候補側は default off.
    track_index = TrackIndex(
        args.candidate_universe,
        corpus_fields=corpus_fields,
        secondary_corpus_fields=secondary_corpus_fields,
        load_dense=True,
    )
    user_vectors = load_user_vectors()

    # query_dense feature: user_query を Qwen3 で encode. mode=last_user で
    # user_query 単独 (chat history 抜き) を encode する.
    from recsys2026.encoders import Qwen3TextEncoder

    dense_encoder_feat = Qwen3TextEncoder(batch_size=args.dense_encode_batch_size)
    feat_query_mode = args.query_dense_feature_mode

    # dense_rrf を使う場合だけ別途 encoder を用意 (異なる query mode).
    dense_encoder = dense_encoder_feat if args.dense_rrf else None

    print("building train/valid examples")
    all_train_examples = build_examples_from_dataset(
        "train", max_sessions=args.max_train_sessions
    )
    train_examples, valid_examples = split_train_valid(
        all_train_examples,
        valid_fraction=args.valid_fraction,
        seed=args.seed,
    )
    print(f"train examples={len(train_examples)} valid examples={len(valid_examples)}")

    # 037: intent extraction (LLM で user_query → JSON).
    intent_extractor: IntentExtractor | None = None
    train_intent: dict[str, dict[str, str]] | None = None
    valid_intent: dict[str, dict[str, str]] | None = None
    if args.intent:
        intent_extractor = IntentExtractor(
            model_name=args.intent_lm_model, batch_size=args.intent_batch_size
        )
        train_intent = pre_extract_intent(
            train_examples,
            intent_extractor,
            cache_path=OUT_DIR
            / f"intent_train_seed{args.seed}_max{args.max_train_sessions}.json",
            use_cache=use_cache,
            desc="intent[train]",
        )
        valid_intent = pre_extract_intent(
            valid_examples,
            intent_extractor,
            cache_path=OUT_DIR
            / f"intent_valid_seed{args.seed}_max{args.max_train_sessions}.json",
            use_cache=use_cache,
            desc="intent[valid]",
        )

    # dense query embedding (cache あり, dense_rrf 有効時のみ — candidate 側).
    train_dense_q = None
    valid_dense_q = None
    if dense_encoder is not None:
        print(f"encoding train dense queries (mode={args.dense_query_mode}) ...")
        train_dense_q = encode_dense_queries(
            train_examples,
            dense_encoder,
            args.dense_query_mode,
            cache_path=OUT_DIR
            / f"dense_q_train_seed{args.seed}_max{args.max_train_sessions}_{args.dense_query_mode}.npz",
            use_cache=use_cache,
            desc="dense_q[train]",
        )
        print(f"encoding valid dense queries (mode={args.dense_query_mode}) ...")
        valid_dense_q = encode_dense_queries(
            valid_examples,
            dense_encoder,
            args.dense_query_mode,
            cache_path=OUT_DIR
            / f"dense_q_valid_seed{args.seed}_max{args.max_train_sessions}_{args.dense_query_mode}.npz",
            use_cache=use_cache,
            desc="dense_q[valid]",
        )

    # 035: feature 側 query embedding (mode=last_user で user_query 単独).
    print(f"encoding train feature dense queries (mode={feat_query_mode}) ...")
    train_dense_q_feat = encode_dense_queries(
        train_examples,
        dense_encoder_feat,
        feat_query_mode,
        cache_path=OUT_DIR
        / f"dense_qfeat_train_seed{args.seed}_max{args.max_train_sessions}_{feat_query_mode}.npz",
        use_cache=use_cache,
        desc="dense_qfeat[train]",
    )
    print(f"encoding valid feature dense queries (mode={feat_query_mode}) ...")
    valid_dense_q_feat = encode_dense_queries(
        valid_examples,
        dense_encoder_feat,
        feat_query_mode,
        cache_path=OUT_DIR
        / f"dense_qfeat_valid_seed{args.seed}_max{args.max_train_sessions}_{feat_query_mode}.npz",
        use_cache=use_cache,
        desc="dense_qfeat[valid]",
    )

    n_bm25_eff: int | None = (
        args.n_bm25 if args.n_bm25 and args.n_bm25 > 0 else None
    )
    # 038: union 候補生成では BM25 候補側にも feature 用の query embedding を使う.
    # (dense_rrf を使わずに union を作るため)
    dense_q_for_cand_train = train_dense_q_feat if n_bm25_eff is not None else train_dense_q
    dense_q_for_cand_valid = valid_dense_q_feat if n_bm25_eff is not None else valid_dense_q

    cache_suffix = (
        f"_tags{int(args.include_tags)}_ms{int(args.multi_source)}"
        f"_dr{int(args.dense_rrf)}_{args.dense_query_mode}"
        f"_un{int(n_bm25_eff is not None)}"
    )

    print("candidate generation: train")
    train_candidates = generate_candidates(
        train_examples,
        track_index,
        candidate_k=args.candidate_k,
        artist_boost=args.artist_boost,
        album_boost=args.album_boost,
        exclude_history=exclude_history,
        cache_name=f"train_seed{args.seed}_max{args.max_train_sessions}{cache_suffix}",
        use_cache=use_cache,
        desc="cand[train]",
        dense_query_emb=dense_q_for_cand_train,
        n_bm25=n_bm25_eff,
    )
    print("candidate generation: valid")
    valid_candidates = generate_candidates(
        valid_examples,
        track_index,
        candidate_k=args.candidate_k,
        artist_boost=args.artist_boost,
        album_boost=args.album_boost,
        exclude_history=exclude_history,
        cache_name=f"valid_seed{args.seed}_max{args.max_train_sessions}{cache_suffix}",
        use_cache=use_cache,
        desc="cand[valid]",
        dense_query_emb=dense_q_for_cand_valid,
        n_bm25=n_bm25_eff,
    )

    train_cand_metrics = candidate_metrics(train_examples, train_candidates, track_index)
    valid_cand_metrics = candidate_metrics(valid_examples, valid_candidates, track_index)
    print(f"train cand metrics: {train_cand_metrics}")
    print(f"valid cand metrics: {valid_cand_metrics}")

    print("fitting reranker feature encoder")
    encoder = FeatureEncoder(track_index, user_vectors)
    encoder.fit_categories(train_examples)

    text_vectorizer = TfidfVectorizer(
        min_df=2,
        max_features=120_000,
        ngram_range=(1, 2),
        strip_accents="unicode",
        lowercase=True,
    )
    text_corpus = (
        track_index.texts
        + [goal_text(ex.conversation_goal) for ex in train_examples]
        + [conversation_text(ex, track_index) for ex in train_examples]
        + [
            _query_plus_goal(ex.user_query, ex.conversation_goal, ex.user_query_thought)
            for ex in train_examples
        ]
    )
    text_vectorizer.fit(text_corpus)
    track_tfidf = text_vectorizer.transform(track_index.texts)

    print("building reranker training matrix")
    x_train, y_train, train_group_sizes = build_feature_matrix(
        train_examples,
        train_candidates,
        encoder,
        text_vectorizer,
        track_tfidf,
        negatives_per_group=negatives_per_group,
        chunk_examples=args.feature_chunk_examples,
        query_dense_emb=train_dense_q_feat,
        n_bm25=n_bm25_eff,
        intent_lookup=train_intent,
    )
    if y_train is None or int(y_train.sum()) == 0:
        raise RuntimeError(
            "no positive reranker training rows; increase candidate_k or check candidate generation"
        )
    print(
        f"reranker rows={len(y_train)} positives={int(y_train.sum())} "
        f"groups={len(train_group_sizes)}"
    )

    # group ベースの lambdarank で学習する.
    # 020_v2 系の二値分類 + class_weight=balanced + train_negatives_per_group=10 だと、
    # 負例を rank 0-9 から取っているせいで「high BM25 score = neg」の逆相関を学習して
    # raw 候補を catastrophic に degrade させる (smoke で 0.18 → 0.01 に転落)。
    # lambdarank は group 内 ranking を直接最適化するので class imbalance に強い。
    ranker_kwargs: dict[str, object] = dict(
        objective="lambdarank",
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        verbose=-1,
    )
    if args.lambdarank_truncation_level > 0:
        # nDCG@K を直接最適化させる. final_k=20 のときは 20 を渡すと nDCG@20 を最適化.
        ranker_kwargs["lambdarank_truncation_level"] = args.lambdarank_truncation_level
    model = lgb.LGBMRanker(**ranker_kwargs)
    model.fit(
        x_train,
        y_train,
        group=train_group_sizes,
        categorical_feature=encoder.categorical_feature_indices,
    )

    # feature importance: どの特徴量が学習に効いたかの参考.
    importances = model.feature_importances_
    importance_pairs = sorted(
        zip(encoder.feature_names, importances, strict=True),
        key=lambda kv: -kv[1],
    )
    top_importances = [
        {"feature": name, "importance": int(imp)}
        for name, imp in importance_pairs[:15]
    ]
    print("top-15 feature importance (gain-based):")
    for d in top_importances:
        print(f"  {d['feature']:32s} {d['importance']}")

    print("reranking valid")
    valid_ranked, _ = score_and_rank(
        model,
        valid_examples,
        valid_candidates,
        encoder,
        text_vectorizer,
        track_tfidf,
        chunk_examples=args.feature_chunk_examples,
        query_dense_emb=valid_dense_q_feat,
        n_bm25=n_bm25_eff,
        intent_lookup=valid_intent,
    )
    raw_valid_ranked = raw_ranked(valid_candidates)
    valid_raw_metrics = ranking_metrics_from_ranked_indices(
        valid_examples,
        raw_valid_ranked,
        track_index,
        k=args.final_k,
        prefix="valid_raw_candidate",
    )
    valid_rerank_metrics = ranking_metrics_from_ranked_indices(
        valid_examples,
        valid_ranked,
        track_index,
        k=args.final_k,
        prefix="valid_rerank",
    )
    print(f"valid raw: {valid_raw_metrics}")
    print(f"valid rerank: {valid_rerank_metrics}")

    if target == "devset":
        infer_examples = build_examples_from_dataset("test")
        if args.max_infer_inputs is not None:
            infer_examples = infer_examples[: args.max_infer_inputs]
    else:
        infer_examples = build_examples_from_blind(target, max_inputs=args.max_infer_inputs)

    infer_intent: dict[str, dict[str, str]] | None = None
    if intent_extractor is not None:
        infer_intent = pre_extract_intent(
            infer_examples,
            intent_extractor,
            cache_path=OUT_DIR
            / f"intent_{target}_max{args.max_infer_inputs}.json",
            use_cache=use_cache,
            desc=f"intent[{target}]",
        )

    infer_dense_q = None
    if dense_encoder is not None:
        print(f"encoding {target} dense queries (mode={args.dense_query_mode}) ...")
        infer_dense_q = encode_dense_queries(
            infer_examples,
            dense_encoder,
            args.dense_query_mode,
            cache_path=OUT_DIR
            / f"dense_q_{target}_max{args.max_infer_inputs}_{args.dense_query_mode}.npz",
            use_cache=use_cache,
            desc=f"dense_q[{target}]",
        )

    print(f"encoding {target} feature dense queries (mode={feat_query_mode}) ...")
    infer_dense_q_feat = encode_dense_queries(
        infer_examples,
        dense_encoder_feat,
        feat_query_mode,
        cache_path=OUT_DIR
        / f"dense_qfeat_{target}_max{args.max_infer_inputs}_{feat_query_mode}.npz",
        use_cache=use_cache,
        desc=f"dense_qfeat[{target}]",
    )

    dense_q_for_cand_infer = (
        infer_dense_q_feat if n_bm25_eff is not None else infer_dense_q
    )
    print(f"candidate generation: {target}")
    infer_candidates = generate_candidates(
        infer_examples,
        track_index,
        candidate_k=args.candidate_k,
        artist_boost=args.artist_boost,
        album_boost=args.album_boost,
        exclude_history=exclude_history,
        cache_name=f"{target}_max{args.max_infer_inputs}{cache_suffix}",
        use_cache=use_cache,
        dense_query_emb=dense_q_for_cand_infer,
        desc=f"cand[{target}]",
        n_bm25=n_bm25_eff,
    )

    print(f"reranking {target}")
    infer_ranked, _ = score_and_rank(
        model,
        infer_examples,
        infer_candidates,
        encoder,
        text_vectorizer,
        track_tfidf,
        chunk_examples=args.feature_chunk_examples,
        query_dense_emb=infer_dense_q_feat,
        n_bm25=n_bm25_eff,
        intent_lookup=infer_intent,
    )

    scores: dict[str, object] = {
        "candidate_k": args.candidate_k,
        "final_k": args.final_k,
        "candidate_universe": args.candidate_universe,
        "include_tags": args.include_tags,
        "multi_source": args.multi_source,
        "dense_rrf": args.dense_rrf,
        "dense_query_mode": args.dense_query_mode if args.dense_rrf else None,
        "n_bm25": args.n_bm25,
        "union_active": n_bm25_eff is not None,
        "artist_boost": args.artist_boost,
        "album_boost": args.album_boost,
        "exclude_history": exclude_history,
        "train_negatives_per_group": args.train_negatives_per_group,
        "n_estimators": args.n_estimators,
        "lambdarank_truncation_level": args.lambdarank_truncation_level,
        "seed": args.seed,
        "max_train_sessions": args.max_train_sessions,
        "max_infer_inputs": args.max_infer_inputs,
        "top_feature_importance": top_importances,
        **prefix_metrics("train", train_cand_metrics),
        **prefix_metrics("valid", valid_cand_metrics),
        **valid_raw_metrics,
        **valid_rerank_metrics,
    }

    if target == "devset" and args.max_infer_inputs is None:
        dev_candidate = candidate_metrics(infer_examples, infer_candidates, track_index)
        scores.update(prefix_metrics("devset", dev_candidate))
        out = write_target_predictions(target, infer_examples, infer_ranked, track_index, args.final_k)
        # downstream の LLM listwise rerank 用に LGBM ranked top-100 を dump.
        top100 = [
            {
                "session_id": ex.session_id,
                "turn_number": ex.turn_number,
                "candidate_track_ids": [
                    track_index.track_ids[idx] for idx in ranked[:100]
                ],
            }
            for ex, ranked in zip(infer_examples, infer_ranked)
        ]
        (OUT_DIR / "devset_ranked_top100.json").write_text(json.dumps(top100))
        print(f"wrote {OUT_DIR / 'devset_ranked_top100.json'} ({len(top100)} turns)")
        dev_scores = evaluate_devset(out)
        scores.update(dev_scores)
        print(json.dumps(dev_scores, indent=2))
    elif target == "devset":
        infer_metrics = ranking_metrics_from_ranked_indices(
            infer_examples,
            infer_ranked,
            track_index,
            k=args.final_k,
            prefix="smoke_devset_rerank",
        )
        scores.update(infer_metrics)
        out = OUT_DIR / "devset.smoke.json"
        print(f"skipped submission validation/evaluate_devset for partial devset; would write {out}")
    else:
        out = write_target_predictions(target, infer_examples, infer_ranked, track_index, args.final_k)
        zip_path = zip_submission(out)
        scores["prediction_path"] = str(out)
        scores["submission_zip_path"] = str(zip_path)
        print(f"wrote {zip_path}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    is_smoke = args.max_train_sessions is not None or args.max_infer_inputs is not None
    if is_smoke:
        scores_name = "scores.smoke.json"
    elif target == "devset":
        scores_name = "scores.json"
    else:
        scores_name = f"scores.{target}.json"
    scores_path = RESULTS_DIR / scores_name
    scores_path.write_text(json.dumps(scores, indent=2))
    print(f"wrote {scores_path}")


if __name__ == "__main__":
    main()
