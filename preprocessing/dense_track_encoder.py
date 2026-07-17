"""Build the shared Qwen3 track-catalog embedding artifact."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from recsys2026.data import load
from recsys2026.encoders import Qwen3TextEncoder
from recsys2026.paths import PREPROCESSED_DIR

CORPUS_FIELDS = ("track_name", "artist_name", "album_name", "release_date", "tag_list")
TRACK_EMB_ARTIFACT = PREPROCESSED_DIR / "dense_track_emb.npz"
DENSE_DIM = 1024


def _stringify(row: dict) -> str:
    """Format the five catalog fields used by the final feature encoder."""
    out = ""
    for field in CORPUS_FIELDS:
        v = row.get(field)
        if v is None:
            continue
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v if x is not None and str(x))
        out += f"{field}: {v}\n"
    return out


def build_track_embedding_artifact(batch_size: int = 64) -> None:
    meta = load("track", split="all_tracks")
    track_ids: list[str] = list(meta["track_id"])
    if TRACK_EMB_ARTIFACT.exists():
        try:
            with np.load(TRACK_EMB_ARTIFACT, allow_pickle=False) as existing:
                cached_ids = [str(value) for value in existing["track_ids"]]
                embeddings = existing["embeddings"]
                if (
                    cached_ids == track_ids
                    and embeddings.shape == (len(track_ids), DENSE_DIM)
                    and np.isfinite(embeddings).all()
                ):
                    print(f"track embedding artifact exists: {TRACK_EMB_ARTIFACT}")
                    return
        except (KeyError, OSError, ValueError):
            pass
        print(f"rebuilding incomplete track embedding artifact: {TRACK_EMB_ARTIFACT}")

    print("Encoding 47k tracks with Qwen3TextEncoder (this takes a few minutes)...")
    encoder = Qwen3TextEncoder(batch_size=batch_size)
    texts = [_stringify(row) for row in meta]
    chunk = 1024
    parts = []
    for i in tqdm(range(0, len(texts), chunk), desc="encode tracks"):
        parts.append(encoder.encode(texts[i : i + chunk]))
    track_mat = np.concatenate(parts, axis=0).astype(np.float32)
    TRACK_EMB_ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    temp_path = TRACK_EMB_ARTIFACT.with_name(f".{TRACK_EMB_ARTIFACT.name}.tmp")
    with temp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            track_ids=np.array(track_ids),
            embeddings=track_mat,
        )
    temp_path.replace(TRACK_EMB_ARTIFACT)
    print(f"wrote {TRACK_EMB_ARTIFACT}")


def main() -> None:
    build_track_embedding_artifact()


if __name__ == "__main__":
    main()
