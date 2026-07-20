"""Build the exact title-and-artist retriever artifact."""

from pathlib import Path

import numpy as np

from recsys2026.fit_free_runner import FitFreeSpec, main as run_component
from recsys2026.retriever_common import match_catalog_names, norm_name


def score(example, track_index):
    text = norm_name(f"{example.user_query} {example.user_thought}")
    titles = match_catalog_names(
        text, track_index.track_name_rare_bucket, min_chars=5, min_tokens=1
    )
    artists = match_catalog_names(
        text, track_index.artist_name_rare_bucket, min_chars=3, min_tokens=1
    )
    if not titles or not artists:
        return None
    values = np.zeros(track_index.n_tracks, dtype=np.float32)
    for title in titles:
        for index in track_index.track_name_to_idx.get(title, []):
            if track_index.track_artist_name_keys[index] & artists:
                values[index] += 10.0
    return values if np.any(values > 0) else None


SPEC = FitFreeSpec("exact_title_artist_source", Path(__file__), score_fn=score)


if __name__ == "__main__":
    run_component(SPEC)
