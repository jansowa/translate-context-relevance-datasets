import argparse

import pytest

from run_answer_relevance_vllm import (
    ANSWER_RELEVANCE_LABELS,
    BAD_ANSWER_FILTER_LABELS,
    build_answer_relevance_schema,
    build_high_precision_bad_answer_prompt,
    build_bad_answer_filter_schema,
    build_output_row,
    extract_question_answer,
    selected_dataset_keys,
    task_failed_jsonl_name,
    task_output_jsonl_name,
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


def test_build_bad_answer_filter_schema_keeps_label_before_reason() -> None:
    schema = build_bad_answer_filter_schema()

    assert list(schema["properties"].keys()) == ["label", "reason"]
    assert schema["properties"]["label"]["enum"] == list(BAD_ANSWER_FILTER_LABELS)


def test_build_high_precision_bad_answer_prompt_uses_anchor_and_answer() -> None:
    prompt = build_high_precision_bad_answer_prompt(
        anchor="Kto gra Madame Gazelle?",
        answer="Morwenna Banks.",
    )

    assert "ewidentnie zły rekord" in prompt
    assert 'anchor: "Kto gra Madame Gazelle?"' in prompt
    assert 'answer: "Morwenna Banks."' in prompt
    assert '"label":"keep|reject"' in prompt


def test_build_output_row_preserves_source_fields_and_adds_score() -> None:
    args = argparse.Namespace(
        task="answer_relevance",
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
        label="answers_question",
        reason_text="Odpowiedz ma poprawny typ i temat.",
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


def test_build_output_row_supports_bad_answer_filter_task() -> None:
    args = argparse.Namespace(
        task="bad_answer_filter",
        model="model-x",
        inference_source="offline",
        base_url=None,
        _api_key_last6="OFFLINE",
    )
    row = {
        "id": "hotpotqa_1",
        "anchor": "Ktory zawodnik gral dla East Bengal?",
        "positive": "East Bengal",
    }

    out_row = build_output_row(
        row=row,
        label="reject",
        reason_text="To praktycznie sam tytul bez informacji.",
        args=args,
    )

    assert list(out_row["bad_answer_filter"].keys()) == ["label", "reason"]
    assert out_row["bad_answer_filter"]["label"] == "reject"
    assert out_row["bad_answer_filter_model"] == "model-x"
    assert out_row["bad_answer_filter_source"] == "offline"
    assert out_row["bad_answer_filter_key_last6"] == "OFFLINE"
    assert out_row["bad_answer_filter_base_url"] is None
    assert isinstance(out_row["bad_answer_filter_timestamp_unix"], int)


def test_selected_dataset_keys_expands_all_without_duplicates() -> None:
    assert selected_dataset_keys(["all", "nq_qa"]) == ["nq_qa", "hotpotqa"]


def test_task_specific_output_names() -> None:
    assert task_output_jsonl_name("answer_relevance") == "answer_relevance.jsonl"
    assert task_output_jsonl_name("bad_answer_filter") == "bad_answer_filter.jsonl"
    assert task_failed_jsonl_name("answer_relevance") == "answer_relevance_failed_rows.jsonl"
    assert task_failed_jsonl_name("bad_answer_filter") == "bad_answer_filter_failed_rows.jsonl"
