"""Feature and example library for the union LambdaRank reranker.

Provides `TrackIndex`, `FeatureEncoder`, the example builders
(`build_examples_from_dataset` / `build_examples_from_inference`), dense query
encoding, user vectors, and candidate metrics. Loaded by the adjacent
`runner.py` via `protocol.py` to build the
176-feature LightGBM LambdaRank matrix: track / user / turn basics, history
consistency, query–metadata similarity (TF-IDF + dense cosine on
Qwen3-Embedding vectors), tag features, and per-retriever features.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from tqdm import tqdm

from recsys2026.data import load
from recsys2026.paths import PREPROCESSED_DIR
from recsys2026.retrieval import chat_to_query_text
from recsys2026.submission import InferenceInput

EMB_COL = "cf-bpr"
MAX_TURNS = 8
TEXT_RE = re.compile(r"[a-z0-9]+")
MISSING_CAT = "<missing>"

DENSE_TRACK_EMB_ARTIFACT = PREPROCESSED_DIR / "dense_track_emb.npz"
DENSE_QUERY_DIM = 1024


@dataclass(frozen=True)
class TurnExample:
    session_id: str
    user_id: str
    turn_number: int
    user_profile: dict
    chat_history: list[dict]
    user_query: str
    gold_track_id: str | None


@dataclass(frozen=True)
class CandidateSet:
    indices: np.ndarray  # [N, K] int32
    scores: np.ndarray  # [N, K] float32 primary candidate score


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


def query_text(user_query: str) -> str:
    """Return the inference-stable query text used by reranker features."""
    return str(user_query or "").strip() or "music"


class TrackIndex:
    """Track metadata and embedding features in final-catalog order."""

    def __init__(self) -> None:
        meta = load("track", split="all_tracks")
        meta_by_id = {row["track_id"]: row for row in meta}

        emb = load("track_emb", split="all_tracks")
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
        self.meta_by_id: dict[str, dict] = meta_by_id

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

        self.n_tracks = len(self.track_ids)

        if not DENSE_TRACK_EMB_ARTIFACT.exists():
            raise FileNotFoundError(
                f"dense track embedding artifact not found: {DENSE_TRACK_EMB_ARTIFACT}.\n"
                "Run preprocessing/dense_track_encoder.py first."
            )
        print(
            f"loading dense track embeddings from {DENSE_TRACK_EMB_ARTIFACT.name} ..."
        )
        data = np.load(DENSE_TRACK_EMB_ARTIFACT, allow_pickle=False)
        cached_ids = data["track_ids"].tolist()
        cached_emb = data["embeddings"]
        id_to_pos = {tid: i for i, tid in enumerate(cached_ids)}
        dense = np.zeros((self.n_tracks, cached_emb.shape[1]), dtype=np.float32)
        missing = 0
        for i, tid in enumerate(self.track_ids):
            pos = id_to_pos.get(tid)
            if pos is None:
                missing += 1
                continue
            dense[i] = cached_emb[pos]
        if missing:
            print(
                f"warning: {missing}/{self.n_tracks} tracks missing in dense artifact"
            )
        self.dense_emb = dense


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


def goal_text() -> str:
    """Keep goal-derived feature columns neutral across fit and inference."""
    return ""


def conversation_text(ex: TurnExample, track_index: TrackIndex | None = None) -> str:
    parts = [query_text(ex.user_query)]
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
) -> list[TurnExample]:
    ds = load("dataset", split=split)
    examples: list[TurnExample] = []
    for item in ds:
        conversations = list(item["conversations"])
        for target_turn in range(1, MAX_TURNS + 1):
            current = [c for c in conversations if c["turn_number"] == target_turn]
            user_turn = next(c for c in current if c["role"] == "user")
            user_query = str(user_turn["content"])
            gold = next(c["content"] for c in current if c["role"] == "music")
            examples.append(
                TurnExample(
                    session_id=item["session_id"],
                    user_id=item["user_id"],
                    turn_number=target_turn,
                    user_profile=dict(item["user_profile"] or {}),
                    chat_history=[
                        c for c in conversations if c["turn_number"] < target_turn
                    ],
                    user_query=user_query,
                    gold_track_id=gold,
                )
            )
    return examples


def build_examples_from_inference(
    target: Literal["blind_b"],
) -> list[TurnExample]:
    ds = load(target, split="test")
    examples: list[TurnExample] = []
    for item in ds:
        conversations = list(item["conversations"])
        current = conversations[-1]
        examples.append(
            TurnExample(
                session_id=item["session_id"],
                user_id=item["user_id"],
                turn_number=int(current["turn_number"]),
                user_profile=dict(item.get("user_profile") or {}),
                chat_history=conversations[:-1],
                user_query=str(current["content"]),
                gold_track_id=None,
            )
        )
    return examples


def _to_inference_input_for_query(ex: TurnExample) -> InferenceInput:
    return InferenceInput(
        session_id=ex.session_id,
        user_id=ex.user_id,
        turn_number=ex.turn_number,
        chat_history=ex.chat_history,
        user_query=query_text(ex.user_query),
    )


def encode_dense_queries(
    examples: list[TurnExample],
    encoder,
    query_mode: str,
    artifact_path: Path | None,
    desc: str = "dense_query",
) -> np.ndarray:
    """Encode each example's query text, reusing an exact-key artifact."""
    keys = np.asarray([f"{ex.session_id}:{ex.turn_number}" for ex in examples])
    if artifact_path is not None and artifact_path.exists():
        try:
            with np.load(artifact_path, allow_pickle=False) as existing:
                cached_keys = existing["keys"]
                cached_embeddings = existing["embeddings"]
                if (
                    np.array_equal(cached_keys, keys)
                    and cached_embeddings.ndim == 2
                    and cached_embeddings.shape[0] == len(keys)
                ):
                    print(f"loaded dense query artifact from {artifact_path}")
                    return cached_embeddings
        except (KeyError, OSError, ValueError):
            pass

    queries = [
        chat_to_query_text(_to_inference_input_for_query(ex), mode=query_mode)
        for ex in examples
    ]
    chunk = max(1, getattr(encoder, "batch_size", 64))
    parts: list[np.ndarray] = []
    for i in tqdm(range(0, len(queries), chunk), desc=desc):
        parts.append(encoder.encode(queries[i : i + chunk]))
    emb = np.concatenate(parts, axis=0).astype(np.float32)
    if artifact_path is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = artifact_path.with_name(f".{artifact_path.name}.tmp")
        with temp_path.open("wb") as handle:
            np.savez_compressed(handle, keys=keys, embeddings=emb)
        temp_path.replace(artifact_path)
    return emb


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
        # Explicit profile-token interactions with track tags.
        "user_culture_tag_overlap",
        "user_country_tag_overlap",
        "user_lang_tag_overlap",
        "query_dense_track_cosine",
        "query_dense_history_centroid_cosine",
        "dense_only_candidate",
        # Reserved zero-valued columns retained to reproduce the submitted schema.
        "intent_genre_in_track_tags",
        "intent_mood_in_track_tags",
        "intent_descriptor_in_track_tags",
        "intent_artist_token_overlap",
        "intent_era_year_diff",
    ]
    # These columns are fixed to the missing category for Blind-B compatibility.
    categorical_names = [
        "goal_category",
        "goal_specificity",
        "latest_goal_progress",
    ]

    def __init__(
        self, track_index: TrackIndex, user_vectors: dict[str, np.ndarray]
    ) -> None:
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
        values: dict[str, set[str]] = {
            name: {MISSING_CAT} for name in self.categorical_names
        }
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
        return {
            "goal_category": MISSING_CAT,
            "goal_specificity": MISSING_CAT,
            "latest_goal_progress": MISSING_CAT,
        }

    def encode_cat(self, name: str, value: str) -> float:
        return float(self.maps[name].get(value or MISSING_CAT, 0))


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
