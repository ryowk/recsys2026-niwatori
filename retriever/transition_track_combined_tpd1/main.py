"""Build challenge+TPD1 next-track transition artifacts."""

from pathlib import Path

import numpy as np
from tqdm import tqdm

from recsys2026.tpd1_runner import (
    TPD1Spec,
    add_table_score,
    main as run_component,
    pad_scored,
    select_with_extras,
)
from recsys2026.train_stat_runner import history_state


def score_examples(
    common,
    examples,
    track_index,
    challenge_cooc,
    _tpd1_cooc,
    tpd1_transition,
    top_k,
):
    rows = []
    for example in tqdm(examples, desc="transition_track_combined_tpd1"):
        _, _, played, history_indices = history_state(common, example, track_index)
        challenge = np.zeros(track_index.n_tracks, dtype=np.float32)
        external = np.zeros(track_index.n_tracks, dtype=np.float32)
        if history_indices:
            last_index = history_indices[-1]
            add_table_score(challenge, challenge_cooc.transition_track, last_index)
            add_table_score(external, tpd1_transition, last_index)
        score = challenge + external
        probability = np.zeros(track_index.n_tracks, dtype=np.float32)
        denominator = float(score.sum())
        if denominator > 0:
            probability = score / denominator
        rows.append(
            select_with_extras(
                score,
                played,
                top_k,
                {
                    "challenge": challenge,
                    "tpd1": external,
                    "transition_probability": probability,
                },
            )
        )
    return pad_scored(rows, top_k)


SPEC = TPD1Spec("transition_track_combined_tpd1", Path(__file__), score_examples)


if __name__ == "__main__":
    run_component(SPEC)
