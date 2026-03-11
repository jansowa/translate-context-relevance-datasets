import argparse

import pytest

from run_answer_relevance_vllm import (
    ANSWER_RELEVANCE_LABELS,
    build_answer_relevance_schema,
    build_output_row,
    extract_question_answer,
    selected_dataset_keys,
)


def test_extract_question_answer_for_nq_qa() -> None:
    row = {
        "id": "nq_qa_1",
        "question": "Kto gra Madame Gazelle?",
        "answer": "Morwenna Banks.",
    }

    question, answer = extract_question_answer(row, "nq_qa")

    assert question == "Kto gra Madame Gazelle?"
    assert answer == "Morwenna Banks."


def test_extract_question_answer_for_hotpotqa() -> None:
    row = {
        "id": "hotpotqa_1",
        "anchor": "Ktory zawodnik gral dla East Bengal?",
        "positive": "Bhaichung Bhutia gral dla East Bengal.",
    }

    question, answer = extract_question_answer(row, "hotpotqa")

    assert question == "Ktory zawodnik gral dla East Bengal?"
    assert answer == "Bhaichung Bhutia gral dla East Bengal."


def test_extract_question_answer_requires_non_empty_text() -> None:
    with pytest.raises(RuntimeError, match="missing non-empty 'question'"):
        extract_question_answer({"question": "", "answer": "x"}, "nq_qa")


def test_build_answer_relevance_schema_keeps_explanation_before_label() -> None:
    schema = build_answer_relevance_schema()

    assert list(schema["properties"].keys()) == ["explanation", "label"]
    assert schema["properties"]["label"]["enum"] == list(ANSWER_RELEVANCE_LABELS)


def test_build_output_row_preserves_source_fields_and_adds_score() -> None:
    args = argparse.Namespace(
        model="model-x",
        inference_source="offline",
        base_url=None,
        _api_key_last6="OFFLINE",
    )
    row = {
        "id": "nq_qa_1",
        "question": "Jakie to miasto?",
        "answer": "Warszawa.",
        "translation_model": "old-model",
    }

    out_row = build_output_row(
        row=row,
        explanation="Odpowiedz ma poprawny typ i temat.",
        label="answers_question",
        args=args,
    )

    assert out_row["id"] == "nq_qa_1"
    assert out_row["translation_model"] == "old-model"
    assert list(out_row["answer_relevance"].keys()) == ["explanation", "label"]
    assert out_row["answer_relevance"]["label"] == "answers_question"
    assert out_row["answer_relevance_model"] == "model-x"
    assert out_row["answer_relevance_source"] == "offline"
    assert out_row["answer_relevance_key_last6"] == "OFFLINE"
    assert out_row["answer_relevance_base_url"] is None
    assert isinstance(out_row["answer_relevance_timestamp_unix"], int)


def test_selected_dataset_keys_expands_all_without_duplicates() -> None:
    assert selected_dataset_keys(["all", "nq_qa"]) == ["nq_qa", "hotpotqa"]
