"""Diversity-aware response selection used by the Qwen3.6 responder."""

from __future__ import annotations

import random
from typing import Any, cast


def tokenize(text: str) -> list[str]:
    """Match the public lexical-diversity tokenization."""

    return text.lower().split()


def bigrams(tokens: list[str]) -> list[str]:
    return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]


def select_diverse(
    candidates_per_record: list[list[dict[str, Any]]],
    seed: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int | float]]:
    """Greedily select one generated response per row.

    The submitted objective is corpus-level unigram diversity plus
    half-weighted bigram diversity after each candidate is added.
    """

    rng = random.Random(seed)
    order = list(range(len(candidates_per_record)))
    rng.shuffle(order)
    seen_unigrams: set[str] = set()
    seen_bigrams: set[str] = set()
    total_tokens = 0
    selected: list[dict[str, Any] | None] = [None] * len(candidates_per_record)

    for row_index in order:
        candidates = candidates_per_record[row_index]
        if not candidates:
            raise ValueError(f"row {row_index} has no response candidates")

        best_gain = float("-inf")
        best = candidates[0]
        for candidate in candidates:
            tokens = tokenize(str(candidate["predicted_response"]))
            new_unigrams = set(tokens) - seen_unigrams
            new_bigrams = set(bigrams(tokens)) - seen_bigrams
            length = len(tokens)

            new_total = total_tokens + length
            if not new_total:
                gain = -1e9
            else:
                unigram_diversity = (len(seen_unigrams) + len(new_unigrams)) / new_total
                bigram_diversity = (len(seen_bigrams) + len(new_bigrams)) / max(
                    new_total - 1, 1
                )
                gain = unigram_diversity + 0.5 * bigram_diversity

            if gain > best_gain:
                best_gain = gain
                best = candidate

        selected[row_index] = best
        tokens = tokenize(str(best["predicted_response"]))
        seen_unigrams.update(tokens)
        seen_bigrams.update(bigrams(tokens))
        total_tokens += len(tokens)

    return [cast(dict[str, Any], row) for row in selected], {
        "unique_unigrams": len(seen_unigrams),
        "unique_bigrams": len(seen_bigrams),
        "total_tokens": total_tokens,
        "lexdiv_unigram": len(seen_unigrams) / max(total_tokens, 1),
    }
