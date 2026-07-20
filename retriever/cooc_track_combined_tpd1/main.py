"""Build challenge+TPD1 track co-occurrence artifacts."""

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
    tpd1_cooc,
    _tpd1_transition,
    top_k,
):
    rows = []
    for example in tqdm(examples, desc="cooc_track_combined_tpd1"):
        _, _, played, history_indices = history_state(common, example, track_index)
        challenge = np.zeros(track_index.n_tracks, dtype=np.float32)
        external = np.zeros(track_index.n_tracks, dtype=np.float32)
        for history_index in history_indices:
            add_table_score(challenge, challenge_cooc.track_track, history_index)
            add_table_score(external, tpd1_cooc, history_index)
        score = challenge + external
        rows.append(
            select_with_extras(
                score,
                played,
                top_k,
                {"challenge": challenge, "tpd1": external},
            )
        )
    return pad_scored(rows, top_k)


SPEC = TPD1Spec("cooc_track_combined_tpd1", Path(__file__), score_examples)


if __name__ == "__main__":
    run_component(SPEC)
