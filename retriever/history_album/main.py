"""Build the observed-history album retriever artifact."""

from pathlib import Path

import numpy as np

from recsys2026.fit_free_runner import FitFreeSpec, main as run_component
from recsys2026.retriever_common import history_state


def score(example, track_index):
    _, albums, _, _ = history_state(example, track_index)
    if not albums:
        return None
    values = np.zeros(track_index.n_tracks, dtype=np.float32)
    for album_id in albums:
        for index in track_index.album_to_idx.get(album_id, []):
            values[index] += 1.0
    return values


SPEC = FitFreeSpec("history_album", Path(__file__), score_fn=score)


if __name__ == "__main__":
    run_component(SPEC)
