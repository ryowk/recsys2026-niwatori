"""Build the observed-history artist retriever artifact."""

from pathlib import Path

import numpy as np

from recsys2026.fit_free_runner import FitFreeSpec, main as run_component
from recsys2026.retriever_common import history_state


def score(example, track_index):
    artists, _, _, _ = history_state(example, track_index)
    if not artists:
        return None
    values = np.zeros(track_index.n_tracks, dtype=np.float32)
    for artist_id in artists:
        for index in track_index.artist_to_idx.get(artist_id, []):
            values[index] += 1.0
    return values


SPEC = FitFreeSpec("history_artist", Path(__file__), score_fn=score)


if __name__ == "__main__":
    run_component(SPEC)
