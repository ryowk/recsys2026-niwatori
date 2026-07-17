"""Build OOF and inference artifacts for artist-name co-occurrence."""

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
    for example in tqdm(examples, desc="cooc_artist_name"):
        _, _, played, _ = history_state(common, example, track_index)
        turn_example = common.TurnExample(
            session_id=example.session_id,
            user_id=example.user_id,
            turn_number=example.turn_number,
            chat_history=list(example.chat_history),
            user_query=example.user_query,
            gold_track_id=example.gold_track_id or None,
        )
        artist_counts, _, _ = common.history_name_counts(
            turn_example, track_index, last_only=False
        )
        neighbor_scores: dict[str, float] = defaultdict(float)
        for name, weight in artist_counts.items():
            for neighbor, count in (
                cooc.artist_name_artist_name.get(name) or {}
            ).items():
                neighbor_scores[neighbor] += float(weight) * float(count)
        scores = np.zeros(track_index.n_tracks, dtype=np.float32)
        for name, value in neighbor_scores.items():
            for index in track_index.artist_name_to_idx.get(name, []):
                scores[index] += value
        rows.append(select_from_score(scores, played, top_k, positive_only=True))
    candidates, sizes, scores = pad_scored(rows, top_k)
    return candidates, sizes, scores, {}


SPEC = TrainStatSpec("cooc_artist_name", Path(__file__), score_examples)


if __name__ == "__main__":
    run_component(SPEC)
