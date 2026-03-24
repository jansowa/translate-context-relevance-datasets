import argparse

import pytest

from run_answer_relevance_vllm import (
    ANSWER_RELEVANCE_LABELS,
    BAD_ANSWER_FILTER_STAGES,
    build_answer_relevance_schema,
    build_bad_answer_filter_entity_integrity_prompt,
    build_bad_answer_filter_schema,
    build_bad_answer_filter_meaning_drift_prompt,
    build_bad_answer_filter_naturalness_prompt,
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


def test_build_bad_answer_filter_schema_contains_all_stage_outputs() -> None:
    schema = build_bad_answer_filter_schema()
    assert list(schema["properties"].keys()) == [stage.output_key for stage in BAD_ANSWER_FILTER_STAGES]
    assert schema["required"] == [stage.output_key for stage in BAD_ANSWER_FILTER_STAGES]


def test_build_bad_answer_filter_prompts_embed_texts() -> None:
    naturalness = build_bad_answer_filter_naturalness_prompt("Przykladowy tekst.")
    entity = build_bad_answer_filter_entity_integrity_prompt("East Bengal wygral 4-1.")
    drift = build_bad_answer_filter_meaning_drift_prompt("Kto gral?", "East Bengal wygral 4-1.")

    assert '"Przykladowy tekst."' in naturalness
    assert '"East Bengal wygral 4-1."' in entity
    assert '"Kto gral?"' in drift
    assert '"East Bengal wygral 4-1."' in drift


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
        result_obj={"explanation": "Odpowiedz ma poprawny typ i temat.", "label": "answers_question"},
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
    result_obj = {
        "question_language_naturalness": {"reason": "OK", "score": 5},
        "answer_language_naturalness": {"reason": "OK", "score": 4},
        "answer_entity_integrity": {"reason": "OK", "suspicious_items": [], "score": 6},
        "answer_semantic_coherence": {"reason": "OK", "problem_fragments": [], "score": 4},
        "question_answer_meaning_drift": {"reason": "OK", "shared_meaning_elements": ["East Bengal"], "score": 5},
    }
    out_row = build_output_row(
        row=row,
        label="",
        result_obj=result_obj,
        args=args,
    )
    assert list(out_row["bad_answer_filter"].keys()) == [stage.output_key for stage in BAD_ANSWER_FILTER_STAGES]
    assert out_row["bad_answer_filter"]["answer_entity_integrity"]["score"] == 6
    assert out_row["bad_answer_filter_model"] == "model-x"
    assert out_row["bad_answer_filter_source"] == "offline"
    assert out_row["bad_answer_filter_key_last6"] == "OFFLINE"
    assert out_row["bad_answer_filter_base_url"] is None
    assert isinstance(out_row["bad_answer_filter_timestamp_unix"], int)


def test_selected_dataset_keys_expands_all_without_duplicates() -> None:
    assert selected_dataset_keys(["all", "nq_qa"]) == ["nq_qa", "hotpotqa"]


def test_task_specific_output_names() -> None:
    assert task_output_jsonl_name("answer_relevance") == "answer_relevance.jsonl"
    assert task_output_jsonl_name("bad_answer_filter") == "bad_answer_filter_evaluations.jsonl"
    assert task_failed_jsonl_name("answer_relevance") == "answer_relevance_failed_rows.jsonl"
    assert task_failed_jsonl_name("bad_answer_filter") == "bad_answer_filter_evaluations_failed_rows.jsonl"
