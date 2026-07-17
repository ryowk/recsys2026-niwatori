"""Build the last-music artist retriever artifact."""

from pathlib import Path

import numpy as np

from recsys2026.fit_free_runner import FitFreeSpec, main as run_component
from recsys2026.retriever_common import as_list


def score(example, track_index):
    metadata = next(
        (
            track_index.meta_by_id.get(str(turn.get("content") or ""))
            for turn in reversed(example.chat_history)
            if turn.get("role") == "music"
        ),
        None,
    )
    artists = {
        str(value) for value in as_list((metadata or {}).get("artist_id")) if value
    }
    if not artists:
        return None
    values = np.zeros(track_index.n_tracks, dtype=np.float32)
    for artist_id in artists:
        for index in track_index.artist_to_idx.get(artist_id, []):
            values[index] += 1.0
    return values


SPEC = FitFreeSpec("last_music_artist", Path(__file__), score_fn=score)


if __name__ == "__main__":
    run_component(SPEC)
