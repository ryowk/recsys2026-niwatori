from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from reranker.union_lambdarank import protocol
from recsys2026.artifacts import artifact_complete
from recsys2026.submission import (
    validate_predictions,
    write_predictions,
    zip_submission,
)
from retriever.bm25_5field.main import SPEC as BM25_SPEC


def test_bm25_component_resolves_to_complete_index_name() -> None:
    assert BM25_SPEC.bm25_name == "5field"
    assert BM25_SPEC.bm25_variants[0][0] == "5field"


def test_artifact_complete_requires_manifest_and_outputs(tmp_path) -> None:
    (tmp_path / "candidates.npz").touch()
    assert not artifact_complete(tmp_path, "candidates.npz", "turns.jsonl")
    (tmp_path / "manifest.json").touch()
    assert not artifact_complete(tmp_path, "candidates.npz", "turns.jsonl")
    (tmp_path / "turns.jsonl").touch()
    assert artifact_complete(tmp_path, "candidates.npz", "turns.jsonl")


def test_partial_dense_cache_is_merged_before_replacement(tmp_path) -> None:
    cache = tmp_path / "dense.npz"
    np.savez_compressed(
        cache,
        keys=np.asarray(["session-a:1"]),
        embeddings=np.asarray([[1.0, 1.5]], dtype=np.float32),
    )
    examples = [
        SimpleNamespace(session_id="session-a", turn_number=1),
        SimpleNamespace(session_id="session-b", turn_number=2),
    ]

    class Features:
        DENSE_QUERY_DIM = 2

        @staticmethod
        def encode_dense_queries(examples, encoder, query_mode, artifact_path, desc):
            assert query_mode == "last_user"
            assert artifact_path is None
            return np.asarray(
                [[float(example.turn_number), 2.5] for example in examples],
                dtype=np.float32,
            )

    with patch("recsys2026.encoders.Qwen3TextEncoder", return_value=object()):
        result = protocol.materialize_dense(
            Features,
            examples,
            [cache],
            artifact_out=cache,
            batch_size=2,
        )

    np.testing.assert_array_equal(
        result, np.asarray([[1.0, 1.5], [2.0, 2.5]], dtype=np.float32)
    )
    with np.load(cache, allow_pickle=False) as merged:
        assert list(merged["keys"]) == ["session-a:1", "session-b:2"]
        np.testing.assert_array_equal(merged["embeddings"], result)


def test_complete_alternate_dense_cache_materializes_missing_output(tmp_path) -> None:
    source = tmp_path / "source.npz"
    output = tmp_path / "output.npz"
    expected = np.asarray([[1.0, 1.5]], dtype=np.float32)
    np.savez_compressed(
        source,
        keys=np.asarray(["session-a:1"]),
        embeddings=expected,
    )
    examples = [SimpleNamespace(session_id="session-a", turn_number=1)]

    class Features:
        DENSE_QUERY_DIM = 2

        @staticmethod
        def encode_dense_queries(*_args, **_kwargs):
            raise AssertionError("a complete alternate cache must not be re-encoded")

    result = protocol.materialize_dense(
        Features,
        examples,
        [source],
        artifact_out=output,
        batch_size=2,
    )

    np.testing.assert_array_equal(result, expected)
    with np.load(output, allow_pickle=False) as materialized:
        assert list(materialized["keys"]) == ["session-a:1"]
        np.testing.assert_array_equal(materialized["embeddings"], expected)


def test_prediction_writer_validates_and_zip_uses_required_member(tmp_path) -> None:
    records = [{"session_id": "s", "turn_number": 1}]
    json_path = tmp_path / "prediction.json"
    with patch("recsys2026.submission.validate_predictions") as validate:
        write_predictions(records, json_path, "blind_b")
    validate.assert_called_once_with(
        records, "blind_b", require_complete=True, allowed_keys=None
    )

    zip_path = zip_submission(json_path)
    import zipfile

    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist() == ["prediction.json"]


def test_prediction_validator_requires_exact_top20_schema() -> None:
    track_ids = [f"track-{i}" for i in range(20)]
    record = {
        "session_id": "session",
        "user_id": "user",
        "turn_number": 1,
        "predicted_track_ids": track_ids,
        "predicted_response": "response",
    }
    expected = SimpleNamespace(session_id="session", user_id="user", turn_number=1)
    with (
        patch(
            "recsys2026.submission.iter_inputs",
            side_effect=lambda _target: iter([expected]),
        ),
        patch("recsys2026.submission.load", return_value={"track_id": track_ids}),
    ):
        validate_predictions([record], "blind_b")
        with pytest.raises(ValueError, match="expected 20 tracks"):
            validate_predictions(
                [{**record, "predicted_track_ids": track_ids[:-1]}], "blind_b"
            )
        with pytest.raises(ValueError, match="fields must be exactly"):
            validate_predictions([{**record, "extra": True}], "blind_b")
        with pytest.raises(ValueError, match="user_id mismatch"):
            validate_predictions([{**record, "user_id": "wrong"}], "blind_b")
