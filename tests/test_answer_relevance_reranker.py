import argparse

import pytest

from run_answer_relevance_reranker import (
    CUSTOM_JSONL_DATASET_KEY,
    DEFAULT_RERANKER_COMPRESS_LAYERS,
    DEFAULT_RERANKER_COMPRESS_RATIO,
    DEFAULT_RERANKER_CUTOFF_LAYERS,
    DEFAULT_RERANKER_MAX_LENGTH,
    build_output_row,
    build_custom_output_row,
    custom_failed_jsonl_name,
    custom_output_jsonl_name,
    parse_int_list,
    resolve_input_output_paths,
    resolve_reranker_runtime_params,
    sigmoid,
)


def test_parse_int_list_accepts_space_and_comma_separated_values() -> None:
    assert parse_int_list(["24,40", "56"], default=[1]) == [24, 40, 56]


def test_parse_int_list_uses_default_when_missing() -> None:
    assert parse_int_list(None, default=[28]) == [28]


def test_parse_int_list_rejects_empty_after_split() -> None:
    with pytest.raises(ValueError, match="at least one integer"):
        parse_int_list([" , "], default=[1])


def test_sigmoid_matches_expected_midpoint_and_monotonicity() -> None:
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert sigmoid(2.0) > sigmoid(1.0) > sigmoid(0.0)
    assert sigmoid(-2.0) < sigmoid(-1.0) < sigmoid(0.0)


def test_build_output_row_adds_reranker_payload() -> None:
    args = argparse.Namespace(
        reranker_prompt="Prompt",
        reranker_cutoff_layers=[28],
        reranker_compress_ratio=2,
        reranker_compress_layers=[24, 40],
        reranker_max_length=1024,
        reranker_model="BAAI/model",
        reranker_dtype="float16",
        batch_size=2,
    )
    row = {
        "id": "nq_qa_1",
        "question": "Jakie to miasto?",
        "answer": "Warszawa.",
    }

    out_row = build_output_row(
        row,
        raw_score=3.25,
        sigmoid_score=0.9626731127,
        args=args,
    )

    assert out_row["id"] == "nq_qa_1"
    assert out_row["answer_relevance_reranker"]["raw_score"] == 3.25
    assert out_row["answer_relevance_reranker"]["sigmoid_score"] == pytest.approx(0.9626731127)
    assert out_row["answer_relevance_reranker"]["cutoff_layers"] == [28]
    assert out_row["answer_relevance_reranker_model"] == "BAAI/model"
    assert out_row["answer_relevance_reranker_dtype"] == "float16"
    assert out_row["answer_relevance_reranker_batch_size"] == 2
    assert isinstance(out_row["answer_relevance_reranker_timestamp_unix"], int)


def test_resolve_reranker_runtime_params_uses_new_defaults() -> None:
    args = argparse.Namespace(
        reranker_preset=None,
        reranker_max_length=None,
        reranker_cutoff_layers=None,
        reranker_compress_ratio=None,
        reranker_compress_layers=None,
    )

    resolved = resolve_reranker_runtime_params(args)

    assert resolved.reranker_max_length == DEFAULT_RERANKER_MAX_LENGTH
    assert resolved.reranker_cutoff_layers == DEFAULT_RERANKER_CUTOFF_LAYERS
    assert resolved.reranker_compress_ratio == DEFAULT_RERANKER_COMPRESS_RATIO
    assert resolved.reranker_compress_layers == DEFAULT_RERANKER_COMPRESS_LAYERS


def test_resolve_reranker_runtime_params_applies_preset_when_explicit() -> None:
    args = argparse.Namespace(
        reranker_preset="quality",
        reranker_max_length=2048,
        reranker_cutoff_layers=["16"],
        reranker_compress_ratio=3,
        reranker_compress_layers=["8,12"],
    )

    resolved = resolve_reranker_runtime_params(args)

    assert resolved.reranker_max_length == 1024
    assert resolved.reranker_cutoff_layers == [40]
    assert resolved.reranker_compress_ratio == 1
    assert resolved.reranker_compress_layers == []


def test_build_custom_output_row_adds_pair_payload() -> None:
    args = argparse.Namespace(
        reranker_model="BAAI/model",
        reranker_dtype="float16",
        batch_size=2,
    )
    row = {"anchor": ["P1"], "response": ["O1", "O2"]}
    out_row = build_custom_output_row(
        row,
        pairs=[
            {"pair_index": 0, "question": "P1", "answer": "O1", "answer_relevance_reranker": {"raw_score": 1.0, "sigmoid_score": 0.7}},
            {"pair_index": 1, "question": "P1", "answer": "O2", "answer_relevance_reranker": {"raw_score": 2.0, "sigmoid_score": 0.8}},
        ],
        args=args,
    )
    assert len(out_row["answer_relevance_reranker_pairs"]) == 2
    assert out_row["answer_relevance_reranker_pairs"][1]["pair_index"] == 1
    assert out_row["answer_relevance_reranker_model"] == "BAAI/model"


def test_custom_reranker_output_paths_are_next_to_input_file() -> None:
    input_path = r"c:\tmp\sample.jsonl"
    assert custom_output_jsonl_name(input_path) == r"c:\tmp\sample.answer_relevance_reranker.jsonl"
    assert custom_failed_jsonl_name(input_path) == r"c:\tmp\sample.answer_relevance_reranker_failed_rows.jsonl"


def test_resolve_reranker_input_output_paths_uses_custom_jsonl_paths() -> None:
    args = argparse.Namespace(
        input_jsonl_path=r"c:\tmp\sample.jsonl",
        out_jsonl_name=None,
        failed_jsonl_name=None,
        out_dir="out_pl",
        dataset_key=CUSTOM_JSONL_DATASET_KEY,
        input_jsonl_name="translated.jsonl",
    )
    input_jsonl, out_jsonl, failed_jsonl = resolve_input_output_paths(args)
    assert input_jsonl == r"c:\tmp\sample.jsonl"
    assert out_jsonl == r"c:\tmp\sample.answer_relevance_reranker.jsonl"
    assert failed_jsonl == r"c:\tmp\sample.answer_relevance_reranker_failed_rows.jsonl"
