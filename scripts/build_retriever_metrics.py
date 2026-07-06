#!/usr/bin/env python3
"""Per-source retriever recall on the public_labeled rows -> docs/retriever_metrics.md.

Loads each union source's public_labeled candidate artifact, aligns the gold
music track per (split:session, turn) key, and reports recall@{20,50,100,200}.
Run after the public retriever sources + union exist (run_retriever_cv5.sh).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from recsys2026.artifacts import decode_keys, load_candidate_artifact, track_id_lookup
from recsys2026.data import load
from recsys2026.retriever_eval import candidate_metrics

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "artifacts/runs/retriever"
K = (20, 50, 100, 200)

# (display name, component, config, artifact_mode)
SOURCES = [
    ("bm25", "bm25_5field_thought", "top500_bsafe", "fit_free_all_rows"),
    ("tfidf", "protocol_tfidf_lgbm_k300", "protocol_v1_bsafe", "fit_free_all_rows"),
    ("two_tower", "two_tower_lora_thought", "oof5_top500_bsafe", "cv5_oof"),
    ("history_artist", "history_artist", "top500", "fit_free_all_rows"),
    ("history_album", "history_album", "top500", "fit_free_all_rows"),
    ("last_music_artist", "last_music_artist", "top500", "fit_free_all_rows"),
    ("last_music_album", "last_music_album", "top500", "fit_free_all_rows"),
    ("exact_album_artist", "exact_album_artist_source", "top500", "fit_free_all_rows"),
    ("tag_intent", "tag_intent_bm25", "top500_bsafe", "fit_free_all_rows"),
    ("cooc_track_tpd1", "cooc_track_combined_tpd1", "oof5_top500_parts", "cv5_oof"),
    ("transition_track_tpd1", "transition_track_combined_tpd1", "oof5_top500_prob_parts", "cv5_oof"),
    ("cooc_album", "cooc_album", "oof5_top500", "cv5_oof"),
    ("cooc_artist_name", "cooc_artist_name", "oof5_top500", "cv5_oof"),
    ("exact_title", "exact_title_artist_source", "top500", "fit_free_all_rows"),
]
UNION = ("union (all 14 sources)", "union", "blind_b_safe_combined_tpd1_parts_cooc500_cv5", None)


def build_gold_by_key() -> dict[tuple[str, int], int]:
    _, id_to_idx = track_id_lookup()
    gold: dict[tuple[str, int], int] = {}
    for prefix, split in (("train", "train"), ("devset", "test")):
        for item in load("dataset", split=split):
            sid = item["session_id"]
            conv = list(item["conversations"])
            for turn in range(1, 9):
                tid = next(
                    (c["content"] for c in conv if c["role"] == "music" and int(c["turn_number"]) == turn),
                    None,
                )
                if tid is not None:
                    gold[(f"{prefix}:{sid}", int(turn))] = id_to_idx.get(tid, -1)
    return gold


def eval_source(artifact_dir: Path, gold_by_key: dict) -> dict | None:
    if not (artifact_dir / "candidates.npz").exists():
        return None
    arrays, _ = load_candidate_artifact(artifact_dir)
    keys = decode_keys(arrays["keys"])
    gold = np.asarray([gold_by_key.get(tuple(k), -1) for k in keys], dtype=np.int64)
    return candidate_metrics(arrays["track_idx"], arrays["sizes"], gold, k_values=K)


def main() -> None:
    gold_by_key = build_gold_by_key()
    rows = []
    for disp, comp, cfg, mode in SOURCES:
        d = RUNS / comp / cfg / mode / "public_labeled"
        m = eval_source(d, gold_by_key)
        rows.append((disp, m))
    ud = RUNS / UNION[1] / UNION[2] / "public_labeled"
    union_m = eval_source(ud, gold_by_key)

    lines = [
        "# Per-source retriever recall (public_labeled, 5-fold OOF)",
        "",
        "Recall of the gold music track among each source's candidates, over the 129,592 public-labeled rows (train + devset). Supervised sources (two-tower, cooc/transition) use their `cv5_oof` artifacts; fit-free sources use `fit_free_all_rows`. Computed by `scripts/build_retriever_metrics.py`.",
        "",
        "| source | mean cands | recall@20 | recall@50 | recall@100 | recall@200 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def fmt(disp, m):
        if m is None:
            return f"| {disp} | — | (missing) | | | |"
        return (
            f"| {disp} | {m['mean_size']:.0f} | {m['recall@20']:.4f} | {m['recall@50']:.4f} "
            f"| {m['recall@100']:.4f} | {m['recall@200']:.4f} |"
        )

    for disp, m in rows:
        lines.append(fmt(disp, m))
    lines.append(fmt(*UNION[:1], union_m) if False else fmt(UNION[0], union_m))
    lines.append("")
    lines.append(
        "The union's recall@20 = candidate recall@20 in the reranker CV report. Individual sources are intentionally narrow (each returns candidates only where its signal fires); the value is the orthogonal coverage they add to the union, not standalone recall."
    )
    out = REPO / "docs/retriever_metrics.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
