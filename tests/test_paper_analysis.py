from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np

from scripts.analyze_paper_results import (
    candidate_metrics,
    error_decomposition,
    ranked_metrics,
    rrf_rank,
    source_ranking_metrics,
    validate_fit_scope,
    validate_reranker_submission_features,
    validate_submission_parity,
    validate_union_manifest,
)
from retriever.union.builder import build_union
from reranker.union_lambdarank.runner import (
    select_feature_set,
)
from responder.qwen36_27b.ensemble import select_diverse


class PaperAnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gold = np.asarray([1, 2, 3, 4], dtype=np.int32)
        self.pool = np.asarray(
            [
                [1, 9, -1],
                [8, 2, -1],
                [8, 9, -1],
                [4, 7, -1],
            ],
            dtype=np.int32,
        )
        self.sizes = np.asarray([2, 2, 2, 2], dtype=np.int32)

    def test_candidate_metrics_uses_emitted_denominator(self) -> None:
        metrics = candidate_metrics(self.pool, self.sizes, self.gold)
        self.assertEqual(metrics["gold_hits@all"], 3)
        self.assertAlmostEqual(metrics["recall@all"], 0.75)
        self.assertAlmostEqual(metrics["micro_precision@all"], 3 / 8)

    def test_responder_uses_submitted_lexical_diversity_objective(self) -> None:
        candidates = [
            [
                {"predicted_response": "a b"},
                {"predicted_response": "a a"},
            ],
            [
                {"predicted_response": "a b"},
                {"predicted_response": "c d"},
            ],
        ]
        selected, stats = select_diverse(candidates, seed=0)
        self.assertEqual(
            [row["predicted_response"] for row in selected],
            ["a b", "c d"],
        )
        self.assertEqual(stats["unique_unigrams"], 4)

    def test_ndcg_and_error_components_share_definition(self) -> None:
        ranked = np.asarray(
            [
                [1, 9, -1],  # rank 1
                [8, 2, -1],  # rank 2
                [8, 9, -1],  # retrieval miss
                [7, 6, -1],  # retrieved, but reranking miss
            ],
            dtype=np.int32,
        )
        metrics = ranked_metrics(ranked, self.gold)
        error = error_decomposition(ranked, self.pool, self.sizes, self.gold)
        expected_ndcg = (1.0 + 1.0 / math.log2(3)) / 4
        self.assertAlmostEqual(metrics["ndcg@20"], expected_ndcg)
        self.assertAlmostEqual(error["sum"], 1.0 - expected_ndcg)
        self.assertAlmostEqual(error["retrieval_miss_loss"], 0.25)
        self.assertAlmostEqual(error["reranking_miss_loss"], 0.25)

    def test_feature_ablation_selects_only_src_columns_as_provenance(self) -> None:
        matrix = np.arange(15, dtype=np.float32).reshape(3, 5)
        base_names = ["query_sim", "artist_match"]
        appended_names = ["extra_duration", "src_bm25_present", "src_tfidf_rank_inv"]

        provenance, provenance_names, provenance_cat = select_feature_set(
            matrix, base_names, appended_names, [1], "provenance_only"
        )
        np.testing.assert_array_equal(provenance, matrix[:, [3, 4]])
        self.assertEqual(provenance_names, ["src_bm25_present", "src_tfidf_rank_inv"])
        self.assertEqual(provenance_cat, [])

        independent, independent_names, independent_cat = select_feature_set(
            matrix, base_names, appended_names, [1], "no_provenance"
        )
        np.testing.assert_array_equal(independent, matrix[:, [0, 1, 2]])
        self.assertEqual(
            independent_names, ["query_sim", "artist_match", "extra_duration"]
        )
        self.assertEqual(independent_cat, [1])

    def test_fit_scope_rejects_devset_fit(self) -> None:
        manifest = {
            "target": "devset",
            "artifact_mode": "full_train",
            "fit_scope": {
                "requires_labeled_fit": True,
                "fit_splits": ["train", "devset"],
                "uses_devset_for_fit": True,
                "uses_blind_for_fit": False,
            },
        }
        with self.assertRaisesRegex(ValueError, "uses_devset_for_fit"):
            validate_fit_scope(manifest, source="learned", target="devset")

    def test_fit_scope_accepts_oof_exclusion_in_leak_check(self) -> None:
        manifest = {
            "target": "public_labeled",
            "artifact_mode": "train5_oof",
            "fit_scope": {
                "requires_labeled_fit": True,
                "fit_splits": ["train"],
                "uses_devset_for_fit": False,
                "uses_blind_for_fit": False,
            },
            "leak_check": {"target_row_excluded_from_fit": True},
        }
        check = validate_fit_scope(
            manifest, source="two_tower", target="public_labeled"
        )
        self.assertTrue(check["target_row_excluded_from_fit"])

    def test_rrf_combines_duplicate_candidates_and_uses_stable_ties(self) -> None:
        sources = [
            {
                "arrays": {
                    "track_idx": np.asarray([[1, 2]], dtype=np.int32),
                    "sizes": np.asarray([2], dtype=np.int32),
                }
            },
            {
                "arrays": {
                    "track_idx": np.asarray([[2, 3]], dtype=np.int32),
                    "sizes": np.asarray([2], dtype=np.int32),
                }
            },
        ]
        ranked, sizes = rrf_rank(sources, constant=60)
        self.assertEqual(sizes.tolist(), [3])
        self.assertEqual(ranked[0, :3].tolist(), [2, 1, 3])

    def test_tie_aware_rrf_does_not_use_artifact_order_inside_ties(self) -> None:
        left = {
            "arrays": {
                "track_idx": np.asarray([[1, 2]], dtype=np.int32),
                "sizes": np.asarray([2], dtype=np.int32),
                "score__primary": np.asarray([[1.0, 1.0]], dtype=np.float32),
            }
        }
        right = {
            "arrays": {
                "track_idx": np.asarray([[2, 3]], dtype=np.int32),
                "sizes": np.asarray([2], dtype=np.int32),
                "score__primary": np.asarray([[1.0, 1.0]], dtype=np.float32),
            }
        }
        ranked, _ = rrf_rank([left, right], constant=60, tie_aware=True)
        self.assertEqual(ranked[0, :3].tolist(), [2, 1, 3])

        left["arrays"]["track_idx"] = np.asarray([[2, 1]], dtype=np.int32)
        right["arrays"]["track_idx"] = np.asarray([[3, 2]], dtype=np.int32)
        permuted, _ = rrf_rank([left, right], constant=60, tie_aware=True)
        self.assertEqual(permuted[0, :3].tolist(), [2, 1, 3])

    def test_source_ndcg_averages_positions_inside_score_ties(self) -> None:
        source = {
            "name": "last_album",
            "arrays": {
                "track_idx": np.asarray([[9, 1]], dtype=np.int32),
                "sizes": np.asarray([2], dtype=np.int32),
                "score__primary": np.asarray([[1.0, 1.0]], dtype=np.float32),
            },
        }
        metrics = source_ranking_metrics(source, np.asarray([1], dtype=np.int32))
        expected = (1.0 + 1.0 / math.log2(3)) / 2.0
        self.assertEqual(metrics["ordering"], "set-valued")
        self.assertAlmostEqual(metrics["ndcg@20_tie_aware"], expected)

    def test_ordered_union_reproduces_submitted_source_order(self) -> None:
        def source(name: str, candidates: list[int]) -> dict:
            return {
                "name": name,
                "arrays": {
                    "track_idx": np.asarray([candidates], dtype=np.int32),
                    "sizes": np.asarray([len(candidates)], dtype=np.int32),
                },
            }

        left = source("left", [9, 2, 7])
        right = source("right", [7, 3])
        union, sizes, _ = build_union([left, right], None)
        np.testing.assert_array_equal(union, np.asarray([[9, 2, 7, 3]], dtype=np.int32))
        np.testing.assert_array_equal(sizes, np.asarray([4], dtype=np.int32))

    def test_protocol_validation_requires_submitted_source_order(self) -> None:
        union_manifest = {
            "target": "devset",
            "union_rule": {
                "type": "ordered_unique",
                "source_order": ["left", "right"],
                "tie_breaker": "source_order_then_source_rank",
                "max_candidates": None,
            },
            "source_artifacts": [{"name": "left"}, {"name": "right"}],
            "fit_scope": {"uses_devset_for_fit": False, "uses_blind_for_fit": False},
        }
        check = validate_union_manifest(
            union_manifest, target="devset", expected_sources=["left", "right"]
        )
        self.assertEqual(check["source_order"], ["left", "right"])
        union_manifest["union_rule"]["source_order"] = ["right", "left"]
        with self.assertRaisesRegex(ValueError, "source_order differs"):
            validate_union_manifest(
                union_manifest, target="devset", expected_sources=["left", "right"]
            )

        reranker_manifest = {
            "params": {
                "neutralize_base_features": (
                    "candidate_rank,log_candidate_rank,reciprocal_candidate_rank"
                )
            }
        }
        validate_reranker_submission_features(reranker_manifest, config="paper")
        reranker_manifest["params"]["neutralize_base_features"] = "candidate_rank"
        with self.assertRaisesRegex(ValueError, "union-position features are active"):
            validate_reranker_submission_features(reranker_manifest, config="paper")

    def test_paper_base_matches_final_submission_config(self) -> None:
        check = validate_submission_parity(
            Path("retriever/union/configs/paper_train5_devset.yaml")
        )
        self.assertTrue(check["passed"])


if __name__ == "__main__":
    unittest.main()
