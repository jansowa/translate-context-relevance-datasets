#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from translation_core import append_jsonl

CLARIN_MS_MARCO_DATASET_KEY = "clarin-ms-marco"
CLARIN_MS_MARCO_DATASET_HF = "clarin-knext/msmarco-pl"
CLARIN_MS_MARCO_QRELS_HF = "clarin-knext/msmarco-pl-qrels"
CLARIN_MS_MARCO_INPUT_JSONL_NAME = "translated.jsonl"


def clarin_ms_marco_input_jsonl_path(out_dir: str) -> str:
    return str(Path(out_dir) / CLARIN_MS_MARCO_DATASET_KEY / CLARIN_MS_MARCO_INPUT_JSONL_NAME)


def _hf_token() -> str | None:
    token = os.getenv("HF_TOKEN")
    return token.strip() if token and token.strip() else None


def _normalize_id(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise RuntimeError("Encountered an empty Hugging Face identifier while building clarin-ms-marco JSONL")
    return text


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _load_dataset_split(dataset_name: str, *, subset_name: str | None, preferred_splits: list[str], token: str | None) -> Any:
    from datasets import load_dataset, load_dataset_builder

    builder = load_dataset_builder(dataset_name, name=subset_name, token=token)
    available_splits = list(builder.info.splits.keys())
    if not available_splits:
        raise RuntimeError(f"Dataset {dataset_name} subset={subset_name!r} exposes no splits")

    for split_name in preferred_splits:
        if split_name in available_splits:
            logging.info(
                "Loading Hugging Face dataset=%s subset=%s split=%s",
                dataset_name,
                subset_name or "default",
                split_name,
            )
            return load_dataset(dataset_name, name=subset_name, split=split_name, token=token)

    if len(available_splits) == 1:
        only_split = available_splits[0]
        logging.info(
            "Loading Hugging Face dataset=%s subset=%s using only available split=%s",
            dataset_name,
            subset_name or "default",
            only_split,
        )
        return load_dataset(dataset_name, name=subset_name, split=only_split, token=token)

    raise RuntimeError(
        f"Could not resolve split for dataset={dataset_name} subset={subset_name!r}. "
        f"Preferred splits={preferred_splits}, available splits={available_splits}"
    )


def ensure_clarin_ms_marco_jsonl(out_dir: str) -> str:
    output_path = Path(clarin_ms_marco_input_jsonl_path(out_dir))
    if output_path.exists():
        logging.info("Using existing clarin-ms-marco JSONL: %s", output_path)
        return str(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_path = output_path.with_name(f"{output_path.name}.tmp")
    if tmp_output_path.exists():
        tmp_output_path.unlink()
    token = _hf_token()

    logging.info("Preparing %s from Hugging Face into %s", CLARIN_MS_MARCO_DATASET_KEY, output_path)
    queries_ds = _load_dataset_split(
        CLARIN_MS_MARCO_DATASET_HF,
        subset_name="queries",
        preferred_splits=["train", "queries"],
        token=token,
    )
    corpus_ds = _load_dataset_split(
        CLARIN_MS_MARCO_DATASET_HF,
        subset_name="corpus",
        preferred_splits=["train", "corpus"],
        token=token,
    )
    qrels_ds = _load_dataset_split(
        CLARIN_MS_MARCO_QRELS_HF,
        subset_name=None,
        preferred_splits=["train", "default"],
        token=token,
    )

    answer_ids_by_question_id: dict[str, list[str]] = defaultdict(list)
    used_answer_ids: set[str] = set()
    qrels_rows = 0
    for row in qrels_ds:
        qrels_rows += 1
        question_id = _normalize_id(row["query-id"])
        answer_id = _normalize_id(row["corpus-id"])
        answer_ids_by_question_id[question_id].append(answer_id)
        used_answer_ids.add(answer_id)

    answers_by_id: dict[str, str] = {}
    corpus_rows = 0
    for row in corpus_ds:
        corpus_rows += 1
        answer_id = _normalize_id(row["_id"])
        if answer_id not in used_answer_ids:
            continue
        answer_text = _normalize_text(row.get("text"))
        if answer_text:
            answers_by_id[answer_id] = answer_text

    written_rows = 0
    skipped_questions_without_links = 0
    skipped_questions_without_text = 0
    skipped_questions_without_answers = 0
    try:
        for row in queries_ds:
            question_id = _normalize_id(row["_id"])
            linked_answer_ids = answer_ids_by_question_id.get(question_id)
            if not linked_answer_ids:
                skipped_questions_without_links += 1
                continue

            question_text = _normalize_text(row.get("text"))
            if not question_text:
                skipped_questions_without_text += 1
                continue

            answers: list[str] = []
            answer_ids: list[str] = []
            seen_answer_ids: set[str] = set()
            for answer_id in linked_answer_ids:
                if answer_id in seen_answer_ids:
                    continue
                answer_text = answers_by_id.get(answer_id)
                if not answer_text:
                    continue
                seen_answer_ids.add(answer_id)
                answer_ids.append(answer_id)
                answers.append(answer_text)

            if not answers:
                skipped_questions_without_answers += 1
                continue

            append_jsonl(
                str(tmp_output_path),
                {
                    "id": f"{CLARIN_MS_MARCO_DATASET_KEY}_{question_id}",
                    "source_dataset": CLARIN_MS_MARCO_DATASET_KEY,
                    "source_dataset_hf": CLARIN_MS_MARCO_DATASET_HF,
                    "question": question_text,
                    "question_id": question_id,
                    "answers": answers,
                    "answer_ids": answer_ids,
                },
            )
            written_rows += 1
    except Exception:
        if tmp_output_path.exists():
            tmp_output_path.unlink()
        raise

    if written_rows == 0:
        if tmp_output_path.exists():
            tmp_output_path.unlink()
        raise RuntimeError(
            "No rows were written for clarin-ms-marco. "
            f"qrels_rows={qrels_rows} corpus_rows={corpus_rows} linked_questions={len(answer_ids_by_question_id)}"
        )

    tmp_output_path.replace(output_path)

    logging.info(
        "Prepared %s JSONL: rows=%d linked_questions=%d qrels_rows=%d corpus_candidates=%d "
        "skipped_no_links=%d skipped_no_question=%d skipped_no_answers=%d",
        CLARIN_MS_MARCO_DATASET_KEY,
        written_rows,
        len(answer_ids_by_question_id),
        qrels_rows,
        len(answers_by_id),
        skipped_questions_without_links,
        skipped_questions_without_text,
        skipped_questions_without_answers,
    )
    return str(output_path)


def configure_custom_jsonl_run_args(args: Any, input_jsonl_path: str, custom_dataset_key: str) -> Any:
    args.dataset_key = custom_dataset_key
    args.input_jsonl_path = input_jsonl_path
    if hasattr(args, "out_jsonl_name"):
        args.out_jsonl_name = None
    if hasattr(args, "failed_jsonl_name"):
        args.failed_jsonl_name = None
    return args
