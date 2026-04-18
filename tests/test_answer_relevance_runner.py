import argparse
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from run_answer_relevance_vllm import (
    ANSWER_RELEVANCE_LABELS,
    BAD_ANSWER_FILTER_STAGES,
    CUSTOM_JSONL_DATASET_KEY,
    build_answer_relevance_schema,
    build_custom_jsonl_pairs,
    build_bad_answer_filter_entity_integrity_prompt,
    build_bad_answer_filter_schema,
    build_bad_answer_filter_meaning_drift_prompt,
    build_bad_answer_filter_naturalness_prompt,
    build_output_row,
    custom_bad_answer_filter_failed_jsonl_name,
    custom_bad_answer_filter_output_jsonl_name,
    extract_custom_jsonl_questions_answers,
    extract_question_answer,
    merge_bad_answer_filter_results,
    merge_custom_bad_answer_filter_results,
    missing_bad_answer_filter_final_stages,
    parse_args,
    read_jsonl_rows,
    resolve_input_output_paths,
    selected_bad_answer_filter_stages,
    selected_dataset_keys,
    task_failed_jsonl_name,
    task_output_jsonl_name,
    write_bad_answer_filter_final_snapshot,
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


def test_extract_custom_jsonl_questions_answers_supports_strings() -> None:
    row = {"anchor": "Jak masz na imie?", "positive": "Jan"}
    questions, answers = extract_custom_jsonl_questions_answers(row)
    assert questions == ["Jak masz na imie?"]
    assert answers == ["Jan"]


def test_extract_custom_jsonl_questions_answers_supports_lists_and_priority() -> None:
    row = {
        "queries": ["Drugie pytanie"],
        "query": "Pierwsze pytanie",
        "responses": ["Odpowiedz A", "Odpowiedz B"],
    }
    questions, answers = extract_custom_jsonl_questions_answers(row)
    assert questions == ["Pierwsze pytanie"]
    assert answers == ["Odpowiedz A", "Odpowiedz B"]


def test_extract_custom_jsonl_questions_answers_supports_question_answers_fields() -> None:
    row = {
        "question": "Pytanie z gotowego polskiego zbioru",
        "answers": ["Odpowiedz 1", "Odpowiedz 2"],
    }
    questions, answers = extract_custom_jsonl_questions_answers(row)
    assert questions == ["Pytanie z gotowego polskiego zbioru"]
    assert answers == ["Odpowiedz 1", "Odpowiedz 2"]


def test_extract_custom_jsonl_questions_answers_filters_empty_strings() -> None:
    row = {
        "anchors": ["", "  ", "Pytanie"],
        "answers": [" ", "Odpowiedz"],
    }
    questions, answers = extract_custom_jsonl_questions_answers(row)
    assert questions == ["Pytanie"]
    assert answers == ["Odpowiedz"]


def test_extract_custom_jsonl_questions_answers_requires_supported_question_field() -> None:
    with pytest.raises(RuntimeError, match="missing non-empty question field"):
        extract_custom_jsonl_questions_answers({"answer": "x"})


def test_extract_custom_jsonl_questions_answers_requires_supported_answer_field() -> None:
    with pytest.raises(RuntimeError, match="missing non-empty answer field"):
        extract_custom_jsonl_questions_answers({"anchor": "x"})


def test_build_custom_jsonl_pairs_uses_cartesian_product() -> None:
    row = {
        "queries": ["P1", "P2"],
        "responses": ["O1", "O2", "O3"],
    }
    pairs = build_custom_jsonl_pairs(row)
    assert len(pairs) == 6
    assert pairs[0] == {"pair_index": 0, "question": "P1", "answer": "O1"}
    assert pairs[-1] == {"pair_index": 5, "question": "P2", "answer": "O3"}


def test_build_answer_relevance_schema_keeps_explanation_before_label() -> None:
    schema = build_answer_relevance_schema()
    assert list(schema["properties"].keys()) == ["explanation", "label"]
    assert schema["properties"]["label"]["enum"] == list(ANSWER_RELEVANCE_LABELS)


def test_build_bad_answer_filter_schema_contains_all_stage_outputs() -> None:
    schema = build_bad_answer_filter_schema()
    assert list(schema["properties"].keys()) == [stage.output_key for stage in BAD_ANSWER_FILTER_STAGES]
    assert schema["required"] == [stage.output_key for stage in BAD_ANSWER_FILTER_STAGES]


def test_selected_bad_answer_filter_stages_skips_entity_integrity_by_default() -> None:
    args = argparse.Namespace(enable_entity_integrity=False)
    output_keys = [stage.output_key for stage in selected_bad_answer_filter_stages(args)]
    assert "answer_entity_integrity" not in output_keys


def test_selected_bad_answer_filter_stages_can_enable_entity_integrity() -> None:
    args = argparse.Namespace(enable_entity_integrity=True)
    output_keys = [stage.output_key for stage in selected_bad_answer_filter_stages(args)]
    assert "answer_entity_integrity" in output_keys


def test_build_bad_answer_filter_schema_can_use_selected_stage_subset() -> None:
    args = argparse.Namespace(enable_entity_integrity=False)
    stages = selected_bad_answer_filter_stages(args)
    schema = build_bad_answer_filter_schema(stages)
    assert list(schema["properties"].keys()) == [stage.output_key for stage in stages]
    assert "answer_entity_integrity" not in schema["properties"]


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
        enable_entity_integrity=False,
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
    assert list(out_row["bad_answer_filter"].keys()) == [stage.output_key for stage in selected_bad_answer_filter_stages(args)]
    assert "answer_entity_integrity" not in out_row["bad_answer_filter"]
    assert out_row["bad_answer_filter_model"] == "model-x"
    assert out_row["bad_answer_filter_source"] == "offline"
    assert out_row["bad_answer_filter_key_last6"] == "OFFLINE"
    assert out_row["bad_answer_filter_base_url"] is None
    assert isinstance(out_row["bad_answer_filter_timestamp_unix"], int)


def test_build_output_row_supports_bad_answer_filter_with_entity_integrity_enabled() -> None:
    args = argparse.Namespace(
        task="bad_answer_filter",
        model="model-x",
        inference_source="offline",
        base_url=None,
        _api_key_last6="OFFLINE",
        enable_entity_integrity=True,
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
    assert list(out_row["bad_answer_filter"].keys()) == [stage.output_key for stage in selected_bad_answer_filter_stages(args)]
    assert out_row["bad_answer_filter"]["answer_entity_integrity"]["score"] == 6


def test_merge_custom_bad_answer_filter_results_builds_pair_list() -> None:
    args = argparse.Namespace(
        task="bad_answer_filter",
        model="model-x",
        inference_source="offline",
        base_url=None,
        _api_key_last6="OFFLINE",
        enable_entity_integrity=False,
    )
    row = {"anchor": ["P1", "P2"], "response": ["O1"]}
    stage_outputs = {
        "question_language_naturalness": [
            {"pair_index": 0, "question": "P1", "answer": "O1", "question_language_naturalness": {"reason": "OK", "score": 5}},
            {"pair_index": 1, "question": "P2", "answer": "O1", "question_language_naturalness": {"reason": "OK", "score": 4}},
        ],
        "answer_language_naturalness": [
            {"pair_index": 0, "question": "P1", "answer": "O1", "answer_language_naturalness": {"reason": "OK", "score": 6}},
            {"pair_index": 1, "question": "P2", "answer": "O1", "answer_language_naturalness": {"reason": "OK", "score": 6}},
        ],
        "answer_semantic_coherence": [
            {"pair_index": 0, "question": "P1", "answer": "O1", "answer_semantic_coherence": {"reason": "OK", "problem_fragments": [], "score": 5}},
            {"pair_index": 1, "question": "P2", "answer": "O1", "answer_semantic_coherence": {"reason": "OK", "problem_fragments": [], "score": 5}},
        ],
        "question_answer_meaning_drift": [
            {"pair_index": 0, "question": "P1", "answer": "O1", "question_answer_meaning_drift": {"reason": "OK", "shared_meaning_elements": ["x"], "score": 4}},
            {"pair_index": 1, "question": "P2", "answer": "O1", "question_answer_meaning_drift": {"reason": "OK", "shared_meaning_elements": ["y"], "score": 3}},
        ],
    }
    out_row = merge_custom_bad_answer_filter_results(row, stage_outputs, args)
    assert len(out_row["bad_answer_filter_pairs"]) == 2
    assert out_row["bad_answer_filter_pairs"][0]["pair_index"] == 0
    assert out_row["bad_answer_filter_pairs"][0]["question"] == "P1"
    assert out_row["bad_answer_filter_pairs"][0]["answer"] == "O1"
    assert "answer_entity_integrity" not in out_row["bad_answer_filter_pairs"][0]["bad_answer_filter"]
    assert out_row["bad_answer_filter_model"] == "model-x"


def test_merge_bad_answer_filter_results_uses_none_for_missing_stage() -> None:
    args = argparse.Namespace(
        task="bad_answer_filter",
        model="model-x",
        inference_source="offline",
        base_url=None,
        _api_key_last6="OFFLINE",
        enable_entity_integrity=False,
    )
    row = {
        "id": "hotpotqa_1",
        "anchor": "Ktory zawodnik gral dla East Bengal?",
        "positive": "East Bengal",
    }
    stage_outputs = {
        "question_language_naturalness": {"reason": "OK", "score": 5},
        "answer_language_naturalness": {"reason": "OK", "score": 4},
        "answer_semantic_coherence": {"reason": "OK", "problem_fragments": [], "score": 5},
    }

    out_row = merge_bad_answer_filter_results(row, stage_outputs, args)

    assert out_row["bad_answer_filter"]["question_answer_meaning_drift"] is None
    assert out_row["bad_answer_filter"]["answer_semantic_coherence"]["score"] == 5


def test_merge_custom_bad_answer_filter_results_uses_none_for_missing_stage() -> None:
    args = argparse.Namespace(
        task="bad_answer_filter",
        model="model-x",
        inference_source="offline",
        base_url=None,
        _api_key_last6="OFFLINE",
        enable_entity_integrity=False,
    )
    row = {"anchor": ["P1", "P2"], "response": ["O1"]}
    stage_outputs = {
        "question_language_naturalness": [
            {"pair_index": 0, "question": "P1", "answer": "O1", "question_language_naturalness": {"reason": "OK", "score": 5}},
            {"pair_index": 1, "question": "P2", "answer": "O1", "question_language_naturalness": {"reason": "OK", "score": 4}},
        ],
        "answer_language_naturalness": [
            {"pair_index": 0, "question": "P1", "answer": "O1", "answer_language_naturalness": {"reason": "OK", "score": 6}},
            {"pair_index": 1, "question": "P2", "answer": "O1", "answer_language_naturalness": {"reason": "OK", "score": 6}},
        ],
        "answer_semantic_coherence": [
            {"pair_index": 0, "question": "P1", "answer": "O1", "answer_semantic_coherence": {"reason": "OK", "problem_fragments": [], "score": 5}},
        ],
    }

    out_row = merge_custom_bad_answer_filter_results(row, stage_outputs, args)

    assert len(out_row["bad_answer_filter_pairs"]) == 2
    assert out_row["bad_answer_filter_pairs"][0]["bad_answer_filter"]["question_answer_meaning_drift"] is None
    assert out_row["bad_answer_filter_pairs"][1]["bad_answer_filter"]["answer_semantic_coherence"] is None


def test_missing_bad_answer_filter_final_stages_detects_null_stage_output() -> None:
    args = argparse.Namespace(enable_entity_integrity=False, input_jsonl_path=None)
    final_row = {
        "bad_answer_filter": {
            "question_language_naturalness": {"reason": "OK", "score": 5},
            "answer_language_naturalness": {"reason": "OK", "score": 5},
            "answer_semantic_coherence": None,
            "question_answer_meaning_drift": {"reason": "OK", "shared_meaning_elements": [], "score": 4},
        }
    }

    assert missing_bad_answer_filter_final_stages(final_row, args) == ["answer_semantic_coherence"]


def test_missing_bad_answer_filter_final_stages_detects_custom_pair_null_stage_output() -> None:
    args = argparse.Namespace(enable_entity_integrity=False, input_jsonl_path="custom.jsonl")
    final_row = {
        "bad_answer_filter_pairs": [
            {
                "pair_index": 0,
                "bad_answer_filter": {
                    "question_language_naturalness": {"reason": "OK", "score": 5},
                    "answer_language_naturalness": {"reason": "OK", "score": 5},
                    "answer_semantic_coherence": {"reason": "OK", "problem_fragments": [], "score": 5},
                    "question_answer_meaning_drift": {"reason": "OK", "shared_meaning_elements": [], "score": 4},
                },
            },
            {
                "pair_index": 1,
                "bad_answer_filter": {
                    "question_language_naturalness": {"reason": "OK", "score": 5},
                    "answer_language_naturalness": {"reason": "OK", "score": 5},
                    "answer_semantic_coherence": {"reason": "OK", "problem_fragments": [], "score": 5},
                    "question_answer_meaning_drift": None,
                },
            },
        ]
    }

    assert missing_bad_answer_filter_final_stages(final_row, args) == ["question_answer_meaning_drift"]


def test_write_bad_answer_filter_final_snapshot_deduplicates_by_input_order() -> None:
    root = Path("out_pl") / f"snapshot_test_{uuid4().hex}"
    out_path = root / "bad_answer_filter_evaluations.jsonl"
    try:
        rows = [
            {"id": "hotpotqa_1", "anchor": "Q1", "positive": "A1"},
            {"id": "hotpotqa_2", "anchor": "Q2", "positive": "A2"},
        ]
        final_rows_by_id = {
            "hotpotqa_2": {"id": "hotpotqa_2", "bad_answer_filter": {"score": 2}},
            "hotpotqa_1": {"id": "hotpotqa_1", "bad_answer_filter": {"score": 1}},
        }

        write_bad_answer_filter_final_snapshot(str(out_path), rows, "hotpotqa", final_rows_by_id)

        snapshot_rows = read_jsonl_rows(str(out_path))
        assert [row["id"] for row in snapshot_rows] == ["hotpotqa_1", "hotpotqa_2"]
        assert snapshot_rows[0]["bad_answer_filter"]["score"] == 1
        assert snapshot_rows[1]["bad_answer_filter"]["score"] == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_custom_bad_answer_filter_output_paths_are_next_to_input_file() -> None:
    input_path = r"c:\tmp\sample.jsonl"
    assert custom_bad_answer_filter_output_jsonl_name(input_path) == r"c:\tmp\sample.bad_answer_filter_evaluations.jsonl"
    assert custom_bad_answer_filter_failed_jsonl_name(input_path) == r"c:\tmp\sample.bad_answer_filter_evaluations_failed_rows.jsonl"


def test_resolve_input_output_paths_uses_custom_jsonl_paths() -> None:
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
    assert out_jsonl == r"c:\tmp\sample.bad_answer_filter_evaluations.jsonl"
    assert failed_jsonl == r"c:\tmp\sample.bad_answer_filter_evaluations_failed_rows.jsonl"


def test_selected_dataset_keys_expands_all_without_duplicates() -> None:
    assert selected_dataset_keys(["all", "nq_qa"]) == ["nq_qa", "hotpotqa"]


def test_selected_dataset_keys_accepts_explicit_clarin_ms_marco() -> None:
    assert selected_dataset_keys(["clarin-ms-marco", "nq_qa"]) == ["clarin-ms-marco", "nq_qa"]


def test_task_specific_output_names() -> None:
    assert task_output_jsonl_name("answer_relevance") == "answer_relevance.jsonl"
    assert task_output_jsonl_name("bad_answer_filter") == "bad_answer_filter_evaluations.jsonl"
    assert task_failed_jsonl_name("answer_relevance") == "answer_relevance_failed_rows.jsonl"
    assert task_failed_jsonl_name("bad_answer_filter") == "bad_answer_filter_evaluations_failed_rows.jsonl"


def test_parse_args_accepts_input_jsonl_path(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["prog", "--model", "model-x", "--task", "bad_answer_filter", "--input-jsonl-path", "custom.jsonl"])
    args = parse_args()
    assert args.task == "bad_answer_filter"
    assert args.input_jsonl_path == "custom.jsonl"
