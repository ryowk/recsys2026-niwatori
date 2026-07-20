"""Build the last-music album retriever artifact."""

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
    albums = {
        str(value) for value in as_list((metadata or {}).get("album_id")) if value
    }
    if not albums:
        return None
    values = np.zeros(track_index.n_tracks, dtype=np.float32)
    for album_id in albums:
        for index in track_index.album_to_idx.get(album_id, []):
            values[index] += 1.0
    return values


SPEC = FitFreeSpec("last_music_album", Path(__file__), score_fn=score)


if __name__ == "__main__":
    run_component(SPEC)
