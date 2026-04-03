import argparse
import os
import shutil
import sys
import types
import uuid

import pytest

from translation_core import (
    append_jsonl,
    build_hotpotqa_few_shot_messages,
    build_hotpotqa_zero_shot_messages,
    build_nq_qa_few_shot_messages,
    build_nq_qa_zero_shot_messages,
    build_toxic_comment_prompt,
    build_wildguard_prompt,
    load_done_ids_from_jsonl,
    normalize_wildguard_subcategories,
)


if "openai" not in sys.modules:
    openai_stub = types.ModuleType("openai")

    class _RateLimitError(Exception):
        pass

    class _AsyncOpenAI:
        pass

    openai_stub.RateLimitError = _RateLimitError
    openai_stub.AsyncOpenAI = _AsyncOpenAI
    sys.modules["openai"] = openai_stub

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = lambda *args, **kwargs: None
    sys.modules["datasets"] = datasets_stub

if "tqdm" not in sys.modules:
    tqdm_stub = types.ModuleType("tqdm")
    tqdm_stub.tqdm = lambda *args, **kwargs: None
    sys.modules["tqdm"] = tqdm_stub

from run_translation import (  # noqa: E402
    TOXIC_LABEL_COLUMNS,
    build_shared_few_shot_examples_by_rid,
    build_out_row_from_state_hotpotqa,
    build_out_row_from_state_nq_qa,
    build_out_row_from_state_toxic,
    build_out_row_from_state_wildguard,
    load_dataset_for_run,
    pair_prompt_uses_few_shot,
    parse_args,
    reorder_candidates_for_prompt_cache,
    resolve_api_connection,
    resolve_row_id,
    row_cache_group_key,
    selected_dataset_keys,
)


def _make_test_dir() -> str:
    path = os.path.join(os.getcwd(), f".test_tmp_{uuid.uuid4().hex}")
    os.makedirs(path, exist_ok=False)
    return path


def test_parse_args_accepts_toxic_dataset(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setattr(sys, "argv", ["prog", "--datasets", "toxic"])
    args = parse_args()
    assert args.datasets == ["toxic"]


def test_parse_args_accepts_wildguard_dataset(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setattr(sys, "argv", ["prog", "--datasets", "wildguard"])
    args = parse_args()
    assert args.datasets == ["wildguard"]


def test_parse_args_accepts_nq_qa_dataset(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setattr(sys, "argv", ["prog", "--datasets", "nq_qa"])
    args = parse_args()
    assert args.datasets == ["nq_qa"]


def test_parse_args_accepts_hotpotqa_dataset(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setattr(sys, "argv", ["prog", "--datasets", "hotpotqa"])
    args = parse_args()
    assert args.datasets == ["hotpotqa"]


def test_parse_args_accepts_few_shot_shared_requests(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setattr(sys, "argv", ["prog", "--datasets", "nq_qa", "--few-shot-shared-requests", "10"])
    args = parse_args()
    assert args.few_shot_shared_requests == 10


def test_parse_args_accepts_pair_prompt_mode(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setattr(sys, "argv", ["prog", "--datasets", "hotpotqa", "--pair-prompt-mode", "no-few-shot"])
    args = parse_args()
    assert args.pair_prompt_mode == "no-few-shot"


def test_selected_dataset_keys_all_excludes_toxic() -> None:
    assert selected_dataset_keys(["all"]) == ["nq", "msmarco"]
    assert selected_dataset_keys(["toxic"]) == ["toxic"]
    assert selected_dataset_keys(["wildguard"]) == ["wildguard"]
    assert selected_dataset_keys(["nq_qa"]) == ["nq_qa"]
    assert selected_dataset_keys(["hotpotqa"]) == ["hotpotqa"]


def test_parse_args_accepts_multiple_datasets(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setattr(sys, "argv", ["prog", "--datasets", "toxic", "wildguard"])
    args = parse_args()
    assert args.datasets == ["toxic", "wildguard"]


def test_parse_args_accepts_offline_inference_source(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setattr(sys, "argv", ["prog", "--inference-source", "offline", "--datasets", "nq"])
    args = parse_args()
    assert args.inference_source == "offline"


def test_parse_args_accepts_offline_micro_batch_size(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--inference-source", "offline", "--offline-micro-batch-size", "150", "--datasets", "nq"],
    )
    args = parse_args()
    assert args.offline_micro_batch_size == 150


def test_resolve_api_connection_offline_requires_no_base_url_or_key() -> None:
    args = argparse.Namespace(
        inference_source="offline",
        base_url=None,
        api_key=None,
    )
    base_url, api_key = resolve_api_connection(args)
    assert base_url is None
    assert api_key is None


def test_selected_dataset_keys_expands_all_and_deduplicates() -> None:
    assert selected_dataset_keys(["all", "toxic", "wildguard", "nq"]) == ["nq", "msmarco", "toxic", "wildguard"]


def test_load_dataset_for_run_maps_wildguard_train_to_config(monkeypatch) -> None:
    called = {}

    def fake_load_dataset(dataset_hf_id, **kwargs):
        called["dataset_hf_id"] = dataset_hf_id
        called["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr("run_translation.load_dataset", fake_load_dataset)

    out = load_dataset_for_run("allenai/wildguardmix", "wildguard", "train", "hf_xxx")
    assert out == "ok"
    assert called["dataset_hf_id"] == "allenai/wildguardmix"
    assert called["kwargs"]["name"] == "wildguardtrain"
    assert called["kwargs"]["split"] == "train"
    assert called["kwargs"]["token"] == "hf_xxx"


def test_load_dataset_for_run_maps_wildguard_test_to_config(monkeypatch) -> None:
    called = {}

    def fake_load_dataset(dataset_hf_id, **kwargs):
        called["dataset_hf_id"] = dataset_hf_id
        called["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr("run_translation.load_dataset", fake_load_dataset)

    out = load_dataset_for_run("allenai/wildguardmix", "wildguard", "test", None)
    assert out == "ok"
    assert called["kwargs"]["name"] == "wildguardtest"
    assert called["kwargs"]["split"] == "train"
    assert called["kwargs"]["token"] is None


def test_load_dataset_for_run_rejects_wildguard_validation() -> None:
    with pytest.raises(RuntimeError, match="does not support split='validation'"):
        load_dataset_for_run("allenai/wildguardmix", "wildguard", "validation", None)


def test_load_dataset_for_run_maps_hotpotqa_to_default_triplet_config(monkeypatch) -> None:
    called = {}

    def fake_load_dataset(dataset_hf_id, **kwargs):
        called["dataset_hf_id"] = dataset_hf_id
        called["kwargs"] = kwargs
        return "ok"

    monkeypatch.setattr("run_translation.load_dataset", fake_load_dataset)

    out = load_dataset_for_run("sentence-transformers/hotpotqa", "hotpotqa", "train", "hf_xxx")
    assert out == "ok"
    assert called["dataset_hf_id"] == "sentence-transformers/hotpotqa"
    assert called["kwargs"]["name"] == "triplet"
    assert called["kwargs"]["split"] == "train"
    assert called["kwargs"]["token"] == "hf_xxx"


def test_toxic_prompt_for_non_toxic_comment() -> None:
    prompt = build_toxic_comment_prompt("Sample comment", [])
    assert "NOT toxic" in prompt
    assert "non-toxic" in prompt


def test_toxic_prompt_puts_stable_output_format_before_variable_instructions() -> None:
    prompt = build_toxic_comment_prompt("Sample comment", ["threat"])
    output_pos = prompt.find("Output format:")
    toxicity_pos = prompt.find("The source comment is toxic.")
    assert output_pos != -1
    assert toxicity_pos != -1
    assert output_pos < toxicity_pos


def test_toxic_prompt_for_multiple_toxicity_types() -> None:
    prompt = build_toxic_comment_prompt("Sample comment", ["threat", "insult", "identity_hate"])
    assert "- threat:" in prompt
    assert "- insult:" in prompt
    assert "- identity_hate:" in prompt
    assert "Preserve the same toxicity types" in prompt


def test_toxic_output_row_contains_expected_fields() -> None:
    row = {
        "id": "abc",
        "comment_text": "text en",
        "toxic": 1,
        "severe_toxic": 0,
        "obscene": 1,
        "threat": 0,
        "insult": 1,
        "identity_hate": 0,
    }
    state = {"comment_text_pl": "tekst pl", "active_model": "model-x", "active_key_last6": "123456"}
    args = argparse.Namespace(dataset_key="toxic", dataset="hf/toxic", base_url="http://base")

    out = build_out_row_from_state_toxic(state, row, ds_idx=7, args=args)

    assert out["id"] == "abc"
    assert out["comment_text"] == "text en"
    assert out["comment_text_pl"] == "tekst pl"
    assert out["translation_model"] == "model-x"
    assert out["translation_key_last6"] == "123456"
    assert out["translation_base_url"] == "http://base"
    assert out["dataset_index"] == 7
    for label in TOXIC_LABEL_COLUMNS:
        assert label in out


def test_normalize_wildguard_subcategories_from_list() -> None:
    out = normalize_wildguard_subcategories(["benign", "cyberattack", "benign"])
    assert out == ["benign", "cyberattack"]


def test_normalize_wildguard_subcategories_from_string() -> None:
    out = normalize_wildguard_subcategories("benign")
    assert out == ["benign"]


def test_normalize_wildguard_subcategories_rejects_empty_or_unknown() -> None:
    with pytest.raises(ValueError, match="empty"):
        normalize_wildguard_subcategories([])
    with pytest.raises(ValueError, match="unknown"):
        normalize_wildguard_subcategories(["unknown_category"])


def test_wildguard_prompt_benign_and_ordering() -> None:
    prompt = build_wildguard_prompt("hello", ["benign"])
    assert "benign and does not request harm" in prompt
    output_pos = prompt.find("Output format:")
    dynamic_pos = prompt.find("The source prompt is benign")
    assert output_pos != -1
    assert dynamic_pos != -1
    assert output_pos < dynamic_pos


def test_wildguard_prompt_multi_risk_contains_all_subcategories() -> None:
    prompt = build_wildguard_prompt("hello", ["cyberattack", "fraud_assisting_illegal_activities"])
    assert "- cyberattack:" in prompt
    assert "- fraud_assisting_illegal_activities:" in prompt
    assert "Do not sanitize or neutralize" in prompt


def test_wildguard_output_row_contains_original_data_and_prompt_pl() -> None:
    row = {
        "prompt": "EN prompt",
        "response": "EN response",
        "subcategory": "cyberattack",
        "extra_col": 123,
    }
    state = {"id": "wg-1", "prompt_pl": "PL prompt", "active_model": "m1", "active_key_last6": "abcdef"}
    args = argparse.Namespace(dataset_key="wildguard", dataset="allenai/wildguardmix", base_url="http://base")

    out = build_out_row_from_state_wildguard(state, row, ds_idx=3, args=args)

    assert out["id"] == "wg-1"
    assert out["prompt"] == "EN prompt"
    assert out["prompt_pl"] == "PL prompt"
    assert out["response"] == "EN response"
    assert out["subcategory"] == "cyberattack"
    assert out["extra_col"] == 123
    assert out["translation_model"] == "m1"
    assert out["translation_key_last6"] == "abcdef"
    assert out["translation_base_url"] == "http://base"
    assert out["dataset_index"] == 3


def test_nq_qa_output_row_contains_translations_and_optional_originals() -> None:
    row = {
        "question": "What is amber urine?",
        "answer": "Amber urine is a dark yellow urine color.",
    }
    state = {"id": "nqqa-1", "question_pl": "Co to jest bursztynowy mocz?", "answer_pl": "Bursztynowy mocz to ciemnozolty kolor moczu.", "active_model": "m2", "active_key_last6": "fedcba"}
    args = argparse.Namespace(dataset_key="nq_qa", dataset="sentence-transformers/natural-questions", base_url=None, keep_original_columns=True)

    out = build_out_row_from_state_nq_qa(state, row, ds_idx=11, args=args)

    assert out["id"] == "nqqa-1"
    assert out["question"] == "Co to jest bursztynowy mocz?"
    assert out["answer"] == "Bursztynowy mocz to ciemnozolty kolor moczu."
    assert out["question_en"] == "What is amber urine?"
    assert out["answer_en"] == "Amber urine is a dark yellow urine color."
    assert out["translation_model"] == "m2"
    assert out["translation_key_last6"] == "fedcba"
    assert out["translation_base_url"] is None
    assert out["dataset_index"] == 11


def test_nq_qa_output_row_accepts_query_alias_from_dataset() -> None:
    row = {
        "query": "What is amber urine?",
        "answer": "Amber urine is a dark yellow urine color.",
    }
    state = {"id": "nqqa-2", "question_pl": "Co to jest bursztynowy mocz?", "answer_pl": "Bursztynowy mocz to ciemnozolty kolor moczu."}
    args = argparse.Namespace(dataset_key="nq_qa", dataset="sentence-transformers/natural-questions", base_url=None, keep_original_columns=True)

    out = build_out_row_from_state_nq_qa(state, row, ds_idx=12, args=args)

    assert out["question_en"] == "What is amber urine?"
    assert out["answer_en"] == "Amber urine is a dark yellow urine color."


def test_hotpotqa_output_row_translates_anchor_and_positive_only() -> None:
    row = {
        "anchor": "who wrote the iliad",
        "positive": "The Iliad is an ancient Greek epic poem attributed to Homer.",
        "negative": "Paris is the capital city of France.",
    }
    state = {"id": "hp-1", "anchor_pl": "kto napisal Iliade", "positive_pl": "Iliada to starozytny grecki poemat epicki przypisywany Homerowi.", "active_model": "m3", "active_key_last6": "654321"}
    args = argparse.Namespace(dataset_key="hotpotqa", dataset="sentence-transformers/hotpotqa", base_url=None, keep_original_columns=True)

    out = build_out_row_from_state_hotpotqa(state, row, ds_idx=13, args=args)

    assert out["anchor"] == "kto napisal Iliade"
    assert out["positive"] == "Iliada to starozytny grecki poemat epicki przypisywany Homerowi."
    assert out["negative"] == "Paris is the capital city of France."
    assert out["anchor_en"] == "who wrote the iliad"
    assert out["positive_en"] == "The Iliad is an ancient Greek epic poem attributed to Homer."
    assert out["translation_model"] == "m3"
    assert out["translation_key_last6"] == "654321"
    assert out["dataset_index"] == 13


def test_build_nq_qa_few_shot_messages_uses_three_examples_and_current_user(monkeypatch) -> None:
    tmp_dir = _make_test_dir()
    try:
        csv_path = os.path.join(tmp_dir, "fewshot.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(
                "\n".join(
                    [
                        "query_text,phrase_pl,doc_text,document_pl",
                        "q1,p1,d1,a1",
                        "q2,p2,d2,a2",
                        "q3,p3,d3,a3",
                        "q4,p4,d4,a4",
                    ]
                )
            )

        monkeypatch.setattr("translation_core.random.sample", lambda seq, k: list(seq)[:k])

        messages = build_nq_qa_few_shot_messages(
            question_en="target question",
            answer_en="target answer",
            examples_path=csv_path,
            example_count=3,
        )

        assert len(messages) == 7
        assert messages[0]["role"] == "user"
        assert "QUESTION (EN)" in messages[0]["content"]
        assert "q1" in messages[0]["content"]
        assert messages[1]["role"] == "assistant"
        assert '"question_pl": "p1"' in messages[1]["content"]
        assert '"answer_pl": "a1"' in messages[1]["content"]
        assert messages[-1]["role"] == "user"
        assert "target question" in messages[-1]["content"]
        assert "target answer" in messages[-1]["content"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_hotpotqa_few_shot_messages_uses_anchor_and_positive_labels(monkeypatch) -> None:
    tmp_dir = _make_test_dir()
    try:
        csv_path = os.path.join(tmp_dir, "fewshot.csv")
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            f.write(
                "\n".join(
                    [
                        "query_text,phrase_pl,doc_text,document_pl",
                        "q1,p1,d1,a1",
                        "q2,p2,d2,a2",
                        "q3,p3,d3,a3",
                        "q4,p4,d4,a4",
                    ]
                )
            )

        monkeypatch.setattr("translation_core.random.sample", lambda seq, k: list(seq)[:k])

        messages = build_hotpotqa_few_shot_messages(
            anchor_en="target anchor",
            positive_en="target positive",
            examples_path=csv_path,
            example_count=3,
        )

        assert len(messages) == 7
        assert messages[0]["role"] == "user"
        assert "ANCHOR (EN)" in messages[0]["content"]
        assert "POSITIVE (EN)" in messages[0]["content"]
        assert messages[1]["role"] == "assistant"
        assert '"anchor_pl": "p1"' in messages[1]["content"]
        assert '"positive_pl": "a1"' in messages[1]["content"]
        assert messages[-1]["role"] == "user"
        assert "target anchor" in messages[-1]["content"]
        assert "target positive" in messages[-1]["content"]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_build_nq_qa_zero_shot_messages_uses_single_user_turn() -> None:
    messages = build_nq_qa_zero_shot_messages(
        question_en="target question",
        answer_en="target answer",
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Do not summarize, shorten, or answer the question." in messages[0]["content"]
    assert "QUESTION (EN)" in messages[0]["content"]
    assert "ANSWER (EN)" in messages[0]["content"]
    assert "target question" in messages[0]["content"]
    assert "target answer" in messages[0]["content"]


def test_build_hotpotqa_zero_shot_messages_uses_single_user_turn() -> None:
    messages = build_hotpotqa_zero_shot_messages(
        anchor_en="target anchor",
        positive_en="target positive",
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert "Preserve the full content of the second text even when it is long." in messages[0]["content"]
    assert "ANCHOR (EN)" in messages[0]["content"]
    assert "POSITIVE (EN)" in messages[0]["content"]
    assert "target anchor" in messages[0]["content"]
    assert "target positive" in messages[0]["content"]


def test_pair_prompt_uses_few_shot_only_for_pair_datasets_in_few_shot_mode() -> None:
    assert pair_prompt_uses_few_shot("nq_qa", "few-shot") is True
    assert pair_prompt_uses_few_shot("hotpotqa", "few-shot") is True
    assert pair_prompt_uses_few_shot("nq_qa", "no-few-shot") is False
    assert pair_prompt_uses_few_shot("nq", "few-shot") is False


def test_build_shared_few_shot_examples_by_rid_reuses_examples_for_blocks_of_ten(monkeypatch) -> None:
    calls = []

    def fake_sample_few_shot_translation_examples(*, examples_path, example_count):
        group_no = len(calls) + 1
        calls.append((examples_path, example_count, group_no))
        return [{"query_text": f"q{group_no}", "phrase_pl": f"p{group_no}", "doc_text": f"d{group_no}", "document_pl": f"a{group_no}"}]

    monkeypatch.setattr("run_translation.sample_few_shot_translation_examples", fake_sample_few_shot_translation_examples)

    candidates = [(idx, {"id": f"row-{idx}"}) for idx in range(23)]
    out = build_shared_few_shot_examples_by_rid(
        candidates=candidates,
        dataset_key="nq_qa",
        prompt_mode="few-shot",
        examples_path="examples.csv",
        example_count=3,
        shared_requests=10,
    )

    assert len(calls) == 3
    assert out["row-0"] == out["row-9"]
    assert out["row-10"] == out["row-19"]
    assert out["row-0"] != out["row-10"]
    assert out["row-20"] == out["row-22"]


def test_build_shared_few_shot_examples_by_rid_returns_empty_for_no_few_shot(monkeypatch) -> None:
    called = {"count": 0}

    def fake_sample_few_shot_translation_examples(*, examples_path, example_count):
        called["count"] += 1
        return [{"query_text": "q", "phrase_pl": "p", "doc_text": "d", "document_pl": "a"}]

    monkeypatch.setattr("run_translation.sample_few_shot_translation_examples", fake_sample_few_shot_translation_examples)

    out = build_shared_few_shot_examples_by_rid(
        candidates=[(0, {"id": "row-0"})],
        dataset_key="nq_qa",
        prompt_mode="no-few-shot",
        examples_path="examples.csv",
        example_count=1,
        shared_requests=10,
    )

    assert out == {}
    assert called["count"] == 0


def test_resume_ids_loaded_from_toxic_jsonl() -> None:
    tmp_dir = _make_test_dir()
    try:
        out_path = os.path.join(tmp_dir, "translated.jsonl")
        append_jsonl(out_path, {"id": "toxic-row-1", "comment_text_pl": "ok"})
        done_ids = load_done_ids_from_jsonl(out_path)
        assert "toxic-row-1" in done_ids
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_resume_ids_loaded_from_wildguard_jsonl() -> None:
    tmp_dir = _make_test_dir()
    try:
        out_path = os.path.join(tmp_dir, "translated.jsonl")
        append_jsonl(out_path, {"id": "wg-row-1", "prompt_pl": "ok"})
        done_ids = load_done_ids_from_jsonl(out_path)
        assert "wg-row-1" in done_ids
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_resolve_row_id_for_wildguard_without_source_id_is_stable() -> None:
    row = {
        "prompt": "hello",
        "response": "resp",
        "subcategory": "benign",
        "adversarial": False,
    }
    rid1 = resolve_row_id(row, ds_idx=10, dataset_key="wildguard")
    rid2 = resolve_row_id(row, ds_idx=10, dataset_key="wildguard")
    assert rid1 == rid2
    assert rid1.startswith("wildguard_")


def test_row_cache_group_key_for_toxic_uses_active_labels_order() -> None:
    row = {
        "toxic": 1,
        "severe_toxic": 0,
        "obscene": 1,
        "threat": 0,
        "insult": 1,
        "identity_hate": 0,
    }
    assert row_cache_group_key("toxic", row) == ("toxic", "obscene", "insult")


def test_row_cache_group_key_for_wildguard_normalizes_subcategories() -> None:
    row = {"subcategory": ["cyberattack", "benign", "cyberattack"]}
    assert row_cache_group_key("wildguard", row) == ("cyberattack", "benign")


def test_reorder_candidates_for_prompt_cache_toxic_groups_by_key_stably() -> None:
    candidates = [
        (0, {"id": "a", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}),
        (1, {"id": "b", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}),
        (2, {"id": "c", "toxic": 1, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}),
        (3, {"id": "d", "toxic": 0, "severe_toxic": 0, "obscene": 1, "threat": 0, "insult": 0, "identity_hate": 0}),
        (4, {"id": "e", "toxic": 0, "severe_toxic": 0, "obscene": 0, "threat": 0, "insult": 0, "identity_hate": 0}),
    ]

    reordered = reorder_candidates_for_prompt_cache("toxic", candidates)
    assert [row["id"] for _, row in reordered] == ["a", "c", "b", "e", "d"]


def test_reorder_candidates_for_prompt_cache_wildguard_groups_by_normalized_subcategories() -> None:
    candidates = [
        (0, {"id": "a", "subcategory": ["benign"]}),
        (1, {"id": "b", "subcategory": ["cyberattack"]}),
        (2, {"id": "c", "subcategory": ["benign"]}),
        (3, {"id": "d", "subcategory": ["cyberattack", "benign"]}),
        (4, {"id": "e", "subcategory": ["cyberattack"]}),
    ]

    reordered = reorder_candidates_for_prompt_cache("wildguard", candidates)
    assert [row["id"] for _, row in reordered] == ["a", "c", "b", "e", "d"]
