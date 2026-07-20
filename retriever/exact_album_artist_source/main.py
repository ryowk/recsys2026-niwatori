"""Build the exact album-and-artist retriever artifact."""

from pathlib import Path

import numpy as np

from recsys2026.fit_free_runner import FitFreeSpec, main as run_component
from recsys2026.retriever_common import match_catalog_names, norm_name


def score(example, track_index):
    text = norm_name(f"{example.user_query} {example.user_thought}")
    albums = match_catalog_names(
        text, track_index.album_name_rare_bucket, min_chars=10, min_tokens=2
    )
    artists = match_catalog_names(
        text, track_index.artist_name_rare_bucket, min_chars=3, min_tokens=1
    )
    if not albums or not artists:
        return None
    values = np.zeros(track_index.n_tracks, dtype=np.float32)
    for album in albums:
        for artist in artists:
            for index in track_index.album_artist_name_to_idx.get((album, artist), []):
                values[index] += 5.0
    return values if np.any(values > 0) else None


SPEC = FitFreeSpec("exact_album_artist_source", Path(__file__), score_fn=score)


if __name__ == "__main__":
    run_component(SPEC)
