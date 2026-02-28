import argparse
import sys
import types

from translation_core import append_jsonl, build_toxic_comment_prompt, load_done_ids_from_jsonl


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

from run_translation_vllm import (  # noqa: E402
    TOXIC_LABEL_COLUMNS,
    build_out_row_from_state_toxic,
    parse_args,
    selected_dataset_keys,
)


def test_parse_args_accepts_toxic_dataset(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setattr(sys, "argv", ["prog", "--datasets", "toxic"])
    args = parse_args()
    assert args.datasets == "toxic"


def test_selected_dataset_keys_all_excludes_toxic() -> None:
    assert selected_dataset_keys("all") == ["nq", "msmarco"]
    assert selected_dataset_keys("toxic") == ["toxic"]


def test_toxic_prompt_for_non_toxic_comment() -> None:
    prompt = build_toxic_comment_prompt("Sample comment", [])
    assert "NOT toxic" in prompt
    assert "non-toxic" in prompt


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


def test_resume_ids_loaded_from_toxic_jsonl(tmp_path) -> None:
    out_path = tmp_path / "translated.jsonl"
    append_jsonl(str(out_path), {"id": "toxic-row-1", "comment_text_pl": "ok"})
    done_ids = load_done_ids_from_jsonl(str(out_path))
    assert "toxic-row-1" in done_ids
