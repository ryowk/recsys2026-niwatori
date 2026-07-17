"""Build OOF and inference artifacts for album co-occurrence."""

from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from recsys2026.train_stat_runner import (
    TrainStatSpec,
    history_state,
    main as run_component,
    pad_scored,
    select_from_score,
)


def score_examples(common, examples, track_index, cooc, top_k):
    rows = []
    for example in tqdm(examples, desc="cooc_album"):
        _, history_albums, played, _ = history_state(common, example, track_index)
        album_scores: dict[str, float] = defaultdict(float)
        for album_id in history_albums:
            for neighbor, count in (cooc.album_album.get(album_id) or {}).items():
                album_scores[neighbor] += float(count)
        scores = np.zeros(track_index.n_tracks, dtype=np.float32)
        for album_id, value in album_scores.items():
            for index in track_index.album_to_idx.get(album_id, []):
                scores[index] += value
        rows.append(select_from_score(scores, played, top_k, positive_only=True))
    candidates, sizes, scores = pad_scored(rows, top_k)
    return candidates, sizes, scores, {}


SPEC = TrainStatSpec("cooc_album", Path(__file__), score_examples)


if __name__ == "__main__":
    run_component(SPEC)
