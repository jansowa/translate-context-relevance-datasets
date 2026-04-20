#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import hashlib
import json
import logging
import os
import openai
import random
import sys
import time
import copy
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Callable

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm

from translation_core import (
    DEFAULT_FEW_SHOT_EXAMPLES_PATH,
    SYSTEM_PAIR_TRANSLATION,
    SYSTEM_PAIR_TRANSLATION_NO_FEW_SHOT,
    SYSTEM_QUERY,
    SYSTEM_NQ_QA,
    SYSTEM_NQ_QA_NO_FEW_SHOT,
    SYSTEM_TEXT_SIMPLE,
    SYSTEM_TEXT,
    TOXIC_LABEL_DESCRIPTIONS,
    WILDGUARD_SUBCATEGORY_DESCRIPTIONS,
    RateLimitReached,
    append_jsonl,
    build_hotpotqa_few_shot_messages,
    build_hotpotqa_zero_shot_messages,
    build_nq_qa_few_shot_messages,
    build_nq_qa_zero_shot_messages,
    build_toxic_comment_prompt,
    build_wildguard_prompt,
    build_text_prompt,
    build_text_prompt_dictforced,
    build_text_prompt_strict,
    build_query_prompt,
    checkpoint_stem_from_id,
    escape_control_chars_in_json_strings,
    extract_first_json_object,
    load_done_ids_from_jsonl,
    sample_few_shot_translation_examples,
    read_json,
    rebuild_text_and_spans,
    normalize_wildguard_subcategories,
    spans_to_pieces,
    write_json_atomic,
)

DATASET_PRESETS: dict[str, str] = {
    "nq": "zilliz/natural_questions-context-relevance-with-think",
    "nq_qa": "sentence-transformers/natural-questions",
    "gooaq": "sentence-transformers/gooaq",
    "hotpotqa": "sentence-transformers/hotpotqa",
    "msmarco": "zilliz/msmarco-context-relevance-with-think",
    "toxic": "thesofakillers/jigsaw-toxic-comment-classification-challenge",
    "wildguard": "allenai/wildguardmix",
}

REQUIRED_CONTEXT_RELEVANCE_COLUMNS = (
    "id",
    "query",
    "texts",
    "context_spans",
    "context_spans_relevance",
    "labels",
    "think_process",
)

TOXIC_LABEL_COLUMNS = tuple(TOXIC_LABEL_DESCRIPTIONS.keys())

REQUIRED_TOXIC_COLUMNS = (
    "id",
    "comment_text",
    *TOXIC_LABEL_COLUMNS,
)

REQUIRED_NQ_QA_PRIMARY_COLUMNS = (
    "answer",
)

REQUIRED_HOTPOTQA_COLUMNS = (
    "anchor",
    "positive",
    "negative",
)

WILDGUARD_SUBCATEGORY_COLUMNS = tuple(WILDGUARD_SUBCATEGORY_DESCRIPTIONS.keys())

REQUIRED_WILDGUARD_COLUMNS = (
    "prompt",
    "subcategory",
)

WILDGUARD_CONFIG_BY_SPLIT = {
    "train": "wildguardtrain",
    "test": "wildguardtest",
}

HOTPOTQA_DEFAULT_CONFIG = "triplet"
GOOAQ_DEFAULT_CONFIG = "pair"


def format_seconds(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def quiet_external_loggers() -> None:
    # Keep translator progress visible, but suppress per-request noise from dependencies.
    noisy = (
        "httpx",
        "httpcore",
        "openai",
        "datasets",
        "huggingface_hub",
        "fsspec",
        "urllib3",
    )
    for name in noisy:
        logging.getLogger(name).setLevel(logging.WARNING)


@dataclass
class RowResult:
    rid: str
    ckpt_path: str | None
    out_row: dict[str, Any]


class OfflineVllmClient:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Offline mode requires the 'vllm' package in the current Python environment."
            ) from exc

        structured_outputs_cls = None
        guided_decoding_cls = None
        try:
            from vllm.sampling_params import StructuredOutputsParams as _StructuredOutputsParams

            structured_outputs_cls = _StructuredOutputsParams
        except Exception:  # noqa: BLE001
            structured_outputs_cls = None
        try:
            from vllm.sampling_params import GuidedDecodingParams as _GuidedDecodingParams

            guided_decoding_cls = _GuidedDecodingParams
        except Exception:  # noqa: BLE001
            guided_decoding_cls = None

        llm_kwargs: dict[str, Any] = {
            "model": args.model,
            "tensor_parallel_size": max(1, int(args.offline_tensor_parallel_size)),
            "dtype": args.offline_dtype,
            "gpu_memory_utilization": float(args.offline_gpu_memory_utilization),
        }
        if int(args.offline_max_model_len) > 0:
            llm_kwargs["max_model_len"] = int(args.offline_max_model_len)
        if int(args.offline_max_num_seqs) > 0:
            llm_kwargs["max_num_seqs"] = int(args.offline_max_num_seqs)
        if int(args.offline_max_num_batched_tokens) > 0:
            llm_kwargs["max_num_batched_tokens"] = int(args.offline_max_num_batched_tokens)
        if args.offline_enforce_eager:
            llm_kwargs["enforce_eager"] = True

        self._SamplingParams = SamplingParams
        self._StructuredOutputsParams = structured_outputs_cls
        self._GuidedDecodingParams = guided_decoding_cls
        self._llm = LLM(**llm_kwargs)
        self._max_output_tokens = max(64, int(args.offline_max_output_tokens))
        self._micro_batch_size = max(1, int(args.offline_micro_batch_size))
        tokenizer_getter = getattr(self._llm, "get_tokenizer", None)
        self._tokenizer = tokenizer_getter() if callable(tokenizer_getter) else None
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._stop_sentinel = object()
        self._supports_json_schema = self._StructuredOutputsParams is not None or self._GuidedDecodingParams is not None
        if not self._supports_json_schema:
            logging.warning(
                "Offline vLLM structured output params are unavailable in this vLLM build; "
                "JSON schema enforcement is disabled for offline mode."
            )

    def _render_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        if self._tokenizer is not None and hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return str(
                    self._tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            except Exception:  # noqa: BLE001
                pass
        rendered: list[str] = ["You are a chat assistant."]
        for message in messages:
            role = str(message.get("role", "user")).upper()
            content = str(message.get("content", "") or "")
            rendered.append(f"[{role}]\n{content}")
        rendered.append("[ASSISTANT]\n")
        return "\n\n".join(rendered)

    @property
    def supports_json_schema(self) -> bool:
        return self._supports_json_schema

    def _sampling_params_for_request(self, temperature: float, response_schema: dict[str, Any] | None) -> Any:
        kwargs: dict[str, Any] = {
            "temperature": float(temperature),
            "top_p": 1.0,
            "max_tokens": self._max_output_tokens,
        }
        if response_schema and self._supports_json_schema:
            if self._StructuredOutputsParams is not None:
                kwargs["structured_outputs"] = self._StructuredOutputsParams(json=response_schema)
            elif self._GuidedDecodingParams is not None:
                kwargs["guided_decoding"] = self._GuidedDecodingParams(json=response_schema)
        return self._SamplingParams(**kwargs)

    def _generate_batch_once(
        self,
        prompts: list[str],
        temperatures: list[float],
        response_schemas: list[dict[str, Any] | None],
    ) -> list[str]:
        sampling_params_list = [
            self._sampling_params_for_request(temp, schema)
            for temp, schema in zip(temperatures, response_schemas)
        ]
        outputs = self._llm.generate(prompts, sampling_params_list, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"Offline vLLM returned {len(outputs)} outputs for {len(prompts)} prompts"
            )

        out_texts: list[str] = []
        for output in outputs:
            candidates = getattr(output, "outputs", None) or []
            if not candidates:
                raise RuntimeError("Offline vLLM returned empty candidate list")
            out_texts.append(str(getattr(candidates[0], "text", "") or ""))
        return out_texts

    async def _ensure_worker_started(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._batch_worker())

    async def _batch_worker(self) -> None:
        while True:
            item = await self._queue.get()
            if item is self._stop_sentinel:
                self._queue.task_done()
                return

            batch = [item]
            # Small coalescing window to form larger micro-batches.
            while len(batch) < self._micro_batch_size:
                try:
                    nxt = await asyncio.wait_for(self._queue.get(), timeout=0.01)
                except asyncio.TimeoutError:
                    break
                if nxt is self._stop_sentinel:
                    await self._queue.put(self._stop_sentinel)
                    break
                batch.append(nxt)

            prompts = [req["prompt"] for req in batch]
            temperatures = [req["temperature"] for req in batch]
            response_schemas = [req.get("response_schema") for req in batch]
            futures = [req["future"] for req in batch]
            try:
                contents = await asyncio.to_thread(
                    self._generate_batch_once,
                    prompts,
                    temperatures,
                    response_schemas,
                )
            except Exception as exc:  # noqa: BLE001
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)
            else:
                for fut, content in zip(futures, contents):
                    if not fut.done():
                        fut.set_result(content)
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def chat_completion_content(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        await self._ensure_worker_started()
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[str] = loop.create_future()
        prompt = self._render_chat_prompt(messages)
        await self._queue.put(
            {
                "prompt": prompt,
                "temperature": float(temperature),
                "response_schema": response_schema,
                "future": fut,
            }
        )
        return await fut

    async def aclose(self) -> None:
        if self._worker_task is None:
            return
        await self._queue.put(self._stop_sentinel)
        await self._queue.join()
        await self._worker_task
        self._worker_task = None


def build_chat_messages(
    system_prompt: str,
    user_prompt: str | None = None,
    extra_messages: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system_prompt}]
    if extra_messages:
        messages.extend(extra_messages)
    elif user_prompt is not None:
        messages.append({"role": "user", "content": user_prompt})
    else:
        raise ValueError("Either user_prompt or extra_messages must be provided")
    return messages


def _is_rate_limited(exc: BaseException) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True
    msg = str(exc).lower()
    if "429" in msg and ("rate" in msg or "limit" in msg or "quota" in msg):
        return True
    if "rate limit" in msg or "quota" in msg:
        return True
    return False


def _is_transient_llm_error(exc: BaseException) -> bool:
    if _is_rate_limited(exc):
        return True
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code >= 500:
        return True
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    msg = str(exc).lower()
    transient_markers = (
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "connection refused",
        "server disconnected",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
    )
    return any(marker in msg for marker in transient_markers)


def _is_json_format_error(exc: BaseException) -> bool:
    if isinstance(exc, json.JSONDecodeError):
        return True
    msg = str(exc).lower()
    markers = (
        "unterminated string",
        "expecting value",
        "invalid control character",
        "extra data",
        "failed to extract a complete json object",
        "no '{' found in the model response",
    )
    return any(marker in msg for marker in markers)


def checkpoint_is_complete(state: dict[str, Any], expected_texts: int) -> bool:
    texts_pl = state.get("texts_pl")
    spans_pl = state.get("context_spans_pl")
    think_pl = state.get("think_process_pl")
    if not state.get("query_pl"):
        return False
    if not isinstance(texts_pl, list) or not isinstance(spans_pl, list) or not isinstance(think_pl, list):
        return False
    if len(texts_pl) != expected_texts or len(spans_pl) != expected_texts or len(think_pl) != expected_texts:
        return False
    if any(x is None for x in texts_pl):
        return False
    if any(x is None for x in spans_pl):
        return False
    if any(x is None for x in think_pl):
        return False
    return True


def resolve_row_id(row: dict[str, Any], ds_idx: int, dataset_key: str) -> str:
    rid = row.get("id")
    if rid is not None:
        rid_s = str(rid).strip()
        if rid_s:
            return rid_s

    if dataset_key == "wildguard":
        stable_payload = "|".join([f"{k}={row.get(k)!r}" for k in sorted(row.keys())])
        digest = hashlib.sha1(stable_payload.encode("utf-8")).hexdigest()[:20]
        return f"wildguard_{digest}"

    return f"{dataset_key}_{ds_idx}"


def active_toxic_types_from_row(row: dict[str, Any]) -> list[str]:
    active = []
    for label in TOXIC_LABEL_COLUMNS:
        if int(row[label]) == 1:
            active.append(label)
    return active


def row_cache_group_key(dataset_key: str, row: dict[str, Any]) -> tuple[str, ...] | None:
    if dataset_key == "toxic":
        return tuple(active_toxic_types_from_row(row))
    if dataset_key == "wildguard":
        return tuple(normalize_wildguard_subcategories(row.get("subcategory")))
    return None


def reorder_candidates_for_prompt_cache(
    dataset_key: str,
    candidates: list[tuple[int, dict[str, Any]]],
) -> list[tuple[int, dict[str, Any]]]:
    if dataset_key not in ("toxic", "wildguard"):
        return candidates

    # Stable grouping: keep first-seen group order and preserve order inside each group.
    grouped: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = {}
    ordered_keys: list[tuple[str, ...]] = []
    for ds_idx, row in candidates:
        key = row_cache_group_key(dataset_key, row)
        if key is None:
            return candidates
        if key not in grouped:
            grouped[key] = []
            ordered_keys.append(key)
        grouped[key].append((ds_idx, row))

    reordered: list[tuple[int, dict[str, Any]]] = []
    for key in ordered_keys:
        reordered.extend(grouped[key])
    return reordered


def build_out_row_from_state(
    state: dict[str, Any],
    row: dict[str, Any],
    ds_idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_row = {
        "id": row["id"],
        "source_dataset": args.dataset_key,
        "source_dataset_hf": args.dataset,
        "query": state["query_pl"],
        "texts": state["texts_pl"],
        "context_spans": state["context_spans_pl"],
        "context_spans_relevance": row["context_spans_relevance"],
        "labels": row["labels"],
        "think_process": state["think_process_pl"],
        "translation_model": state.get("active_model"),
        "translation_source": getattr(args, "inference_source", "vllm"),
        "translation_key_last6": state.get("active_key_last6"),
        "translation_base_url": (args.base_url or None),
        "dataset_index": ds_idx,
    }
    if args.keep_original_columns:
        out_row.update(
            {
                "query_en": row["query"],
                "texts_en": row["texts"],
                "context_spans_en": row["context_spans"],
                "think_process_en": row["think_process"],
            }
        )
    return out_row


def build_out_row_from_state_toxic(
    state: dict[str, Any],
    row: dict[str, Any],
    ds_idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_row: dict[str, Any] = {
        "id": row["id"],
        "source_dataset": args.dataset_key,
        "source_dataset_hf": args.dataset,
        "comment_text": row["comment_text"],
        "comment_text_pl": state["comment_text_pl"],
        "translation_model": state.get("active_model"),
        "translation_source": getattr(args, "inference_source", "vllm"),
        "translation_key_last6": state.get("active_key_last6"),
        "translation_base_url": (args.base_url or None),
        "dataset_index": ds_idx,
    }
    for label in TOXIC_LABEL_COLUMNS:
        out_row[label] = int(row[label])
    return out_row


def build_out_row_from_state_wildguard(
    state: dict[str, Any],
    row: dict[str, Any],
    ds_idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_row = dict(row)
    out_row.update(
        {
            "id": state["id"],
            "source_dataset": args.dataset_key,
            "source_dataset_hf": args.dataset,
            "prompt_pl": state["prompt_pl"],
            "translation_model": state.get("active_model"),
            "translation_source": getattr(args, "inference_source", "vllm"),
            "translation_key_last6": state.get("active_key_last6"),
            "translation_base_url": (args.base_url or None),
            "dataset_index": ds_idx,
        }
    )
    return out_row


def build_out_row_from_state_nq_qa(
    state: dict[str, Any],
    row: dict[str, Any],
    ds_idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    question_en = get_nq_qa_question_en(row)
    out_row: dict[str, Any] = {
        "id": state["id"],
        "source_dataset": args.dataset_key,
        "source_dataset_hf": args.dataset,
        "question": state["question_pl"],
        "answer": state["answer_pl"],
        "translation_model": state.get("active_model"),
        "translation_source": getattr(args, "inference_source", "vllm"),
        "translation_key_last6": state.get("active_key_last6"),
        "translation_base_url": (args.base_url or None),
        "dataset_index": ds_idx,
    }
    if args.keep_original_columns:
        out_row.update(
            {
                "question_en": question_en,
                "answer_en": row["answer"],
            }
        )
    return out_row


def build_out_row_from_state_hotpotqa(
    state: dict[str, Any],
    row: dict[str, Any],
    ds_idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_row: dict[str, Any] = {
        "id": state["id"],
        "source_dataset": args.dataset_key,
        "source_dataset_hf": args.dataset,
        "anchor": state["anchor_pl"],
        "positive": state["positive_pl"],
        "negative": row["negative"],
        "translation_model": state.get("active_model"),
        "translation_source": getattr(args, "inference_source", "vllm"),
        "translation_key_last6": state.get("active_key_last6"),
        "translation_base_url": (args.base_url or None),
        "dataset_index": ds_idx,
    }
    if args.keep_original_columns:
        out_row.update(
            {
                "anchor_en": row["anchor"],
                "positive_en": row["positive"],
            }
        )
    return out_row


def get_nq_qa_question_en(row: dict[str, Any]) -> str:
    question_en = row.get("question")
    if question_en is None:
        question_en = row.get("query")
    question_en = str(question_en or "").strip()
    if not question_en:
        raise RuntimeError("QA pair row is missing a non-empty question/query value")
    return question_en


def pair_prompt_uses_few_shot(dataset_key: str, prompt_mode: str) -> bool:
    return dataset_key in ("nq_qa", "gooaq", "hotpotqa") and prompt_mode == "few-shot"


def build_shared_few_shot_examples_by_rid(
    candidates: list[tuple[int, dict[str, Any]]],
    dataset_key: str,
    prompt_mode: str,
    examples_path: str,
    example_count: int,
    shared_requests: int,
) -> dict[str, list[dict[str, str]]]:
    if not pair_prompt_uses_few_shot(dataset_key, prompt_mode):
        return {}

    group_size = max(1, int(shared_requests))
    examples_by_rid: dict[str, list[dict[str, str]]] = {}
    for start in range(0, len(candidates), group_size):
        sampled_examples = sample_few_shot_translation_examples(
            examples_path=examples_path,
            example_count=example_count,
        )
        for ds_idx, row in candidates[start : start + group_size]:
            rid = resolve_row_id(row, ds_idx, dataset_key)
            examples_by_rid[rid] = list(sampled_examples)
    return examples_by_rid


async def llm_call_json_async(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str | None,
    temperature: float,
    max_retries: int,
    delay_seconds: float,
    response_schema: dict[str, Any] | None = None,
    extra_messages: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    last_err: BaseException | None = None
    schema_enabled = response_schema is not None
    if isinstance(client, OfflineVllmClient) and response_schema is not None and not client.supports_json_schema:
        schema_enabled = False
    messages = build_chat_messages(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extra_messages=extra_messages,
    )

    for attempt in range(max_retries):
        try:
            if isinstance(client, OfflineVllmClient):
                content = await client.chat_completion_content(
                    messages=messages,
                    temperature=temperature,
                    response_schema=response_schema if schema_enabled else None,
                )
            else:
                kwargs = dict(
                    model=model,
                    temperature=temperature,
                    messages=messages,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
                )

                if schema_enabled and response_schema is not None:
                    kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "translation_response",
                            "schema": response_schema,
                        },
                    }
                else:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = await client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content or ""

            try:
                obj = extract_first_json_object(content)
            except Exception:
                fixed = escape_control_chars_in_json_strings(content)
                obj = extract_first_json_object(fixed)

            if delay_seconds and delay_seconds > 0:
                await asyncio.sleep(float(delay_seconds))

            return obj

        except Exception as e:  # noqa: BLE001
            last_err = e
            if _is_rate_limited(e):
                raise RateLimitReached(str(e)) from e
            if schema_enabled:
                msg = str(e).lower()
                if "json_schema" in msg or "response_format" in msg:
                    schema_enabled = False
                    continue
                if isinstance(client, OfflineVllmClient) and (
                    "structured_outputs" in msg or "guided_decoding" in msg
                ):
                    schema_enabled = False
                    continue
            if _is_json_format_error(e):
                # Retry immediately on malformed JSON output from the model.
                continue
            if _is_transient_llm_error(e):
                await asyncio.sleep(min(60, (2 ** attempt) + random.random()))
                continue
            # Non-transient errors are unlikely to improve with repeated attempts.
            break

    raise RuntimeError(f"LLM call failed after retries: {last_err}") from last_err


def build_translation_schema_list(n: int) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "translated_spans": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": n,
                "maxItems": n,
            },
            "translated_think_process": {"type": "string"},
        },
        "required": ["translated_spans", "translated_think_process"],
        "additionalProperties": False,
    }


def build_translation_schema_dict(n: int) -> dict[str, Any]:
    keys = [str(i) for i in range(1, n + 1)]
    span_properties = {k: {"type": "string"} for k in keys}
    return {
        "type": "object",
        "properties": {
            "translated_spans_dict": {
                "type": "object",
                "properties": span_properties,
                "required": keys,
                "additionalProperties": False,
            },
            "translated_think_process": {"type": "string"},
        },
        "required": ["translated_spans_dict", "translated_think_process"],
        "additionalProperties": False,
    }


async def translate_text_with_span_repair_async(
    client: AsyncOpenAI,
    model: str,
    query_en: str,
    query_pl: str,
    doc_label: int,
    span_texts_en: list[str],
    spans_rel_i: list[int],
    think_process_en: str,
    *,
    delay_seconds: float,
    temperature: float,
    max_retries: int,
    max_attempts: int = 3,
) -> tuple[list[str], str]:
    n = len(span_texts_en)

    schema_list = build_translation_schema_list(n)
    schema_dict = build_translation_schema_dict(n)

    prompt_specs = [
        (
            build_text_prompt(query_en, query_pl, doc_label, span_texts_en, spans_rel_i, think_process_en),
            schema_list,
        ),
        (
            build_text_prompt_strict(query_en, query_pl, doc_label, span_texts_en, spans_rel_i, think_process_en),
            schema_list,
        ),
        (
            build_text_prompt_dictforced(query_en, query_pl, doc_label, span_texts_en, spans_rel_i, think_process_en),
            schema_dict,
        ),
    ]
    effective_attempts = max(1, min(max_attempts, len(prompt_specs)))
    prompt_specs = prompt_specs[:effective_attempts]

    last_problem = None
    for attempt_idx, (prompt, response_schema) in enumerate(prompt_specs, start=1):
        t_json = await llm_call_json_async(
            client=client,
            model=model,
            system_prompt=SYSTEM_TEXT,
            user_prompt=prompt,
            temperature=temperature,
            max_retries=max_retries,
            delay_seconds=delay_seconds,
            response_schema=response_schema,
        )

        if "translated_spans" in t_json:
            translated_spans = t_json.get("translated_spans")
            translated_tp = t_json.get("translated_think_process")
            if isinstance(translated_spans, list) and len(translated_spans) == n and isinstance(translated_tp, str):
                translated_spans = [str(s).replace("\r\n", "\n") for s in translated_spans]
                translated_tp = translated_tp.replace("\r\n", "\n")
                return translated_spans, translated_tp

            last_problem = (
                f"attempt {attempt_idx}: expected {n} spans, "
                f"got {type(translated_spans)} len={len(translated_spans) if isinstance(translated_spans, list) else 'N/A'}"
            )

        if "translated_spans_dict" in t_json:
            d = t_json.get("translated_spans_dict")
            translated_tp = t_json.get("translated_think_process")
            if isinstance(d, dict) and isinstance(translated_tp, str):
                ok = True
                out = []
                for k in range(1, n + 1):
                    ks = str(k)
                    if ks not in d:
                        ok = False
                        break
                    out.append(str(d[ks]))
                if ok:
                    out = [s.replace("\r\n", "\n") for s in out]
                    translated_tp = translated_tp.replace("\r\n", "\n")
                    return out, translated_tp

            last_problem = f"attempt {attempt_idx}: translated_spans_dict missing keys 1..{n} or bad types"

    raise RuntimeError(
        f"Failed to obtain the correct number of spans after {len(prompt_specs)} attempts. Last issue: {last_problem}"
    )


async def process_row(
    row: dict[str, Any],
    ds_idx: int,
    args: argparse.Namespace,
    api_key_last6: str,
    client: Any,
    unit_done_callback: Callable[[int], None] | None = None,
) -> RowResult:
    rid = row["id"]
    stem = checkpoint_stem_from_id(rid)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{stem}.json")

    state = await asyncio.to_thread(read_json, ckpt_path) or {}
    if not state:
        state = {
            "id": rid,
            "query_en": row["query"],
            "query_pl": None,
            "texts_pl": [None] * len(row["texts"]),
            "context_spans_pl": [None] * len(row["texts"]),
            "think_process_pl": [None] * len(row["think_process"]),
            "done_text_idxs": [],
            "status": "in_progress",
            "active_model": None,
            "active_key_last6": None,
            "dataset_index": ds_idx,
        }
        await asyncio.to_thread(write_json_atomic, ckpt_path, state)

    query_en = row["query"]
    texts: list[str] = row["texts"]
    context_spans: list[list[list[int]]] = row["context_spans"]
    context_spans_rel: list[list[int]] = row["context_spans_relevance"]
    labels: list[int] = row["labels"]
    think_process: list[str] = row["think_process"]

    if not state.get("query_pl"):
        try:
            pos_idx = labels.index(1)
        except ValueError:
            pos_idx = 0

        pos_text = texts[pos_idx]
        pos_spans = context_spans[pos_idx]
        pos_rel = context_spans_rel[pos_idx]

        _, pos_span_texts = spans_to_pieces(pos_text, pos_spans)
        rel_frags = [t for t, r in zip(pos_span_texts, pos_rel) if int(r) == 1]
        if not rel_frags:
            rel_frags = pos_span_texts[:3]

        q_prompt = build_query_prompt(query_en, rel_frags)
        q_json = await llm_call_json_async(
            client=client,
            model=args.model,
            system_prompt=SYSTEM_QUERY,
            user_prompt=q_prompt,
            temperature=args.temperature,
            max_retries=args.max_retries,
            delay_seconds=args.delay_seconds,
        )
        state["query_pl"] = (q_json.get("query_pl") or "").strip()
        if not state["query_pl"]:
            raise RuntimeError("Empty query_pl from model")

        state["active_model"] = args.model
        state["active_key_last6"] = api_key_last6
        await asyncio.to_thread(write_json_atomic, ckpt_path, state)
        if unit_done_callback:
            unit_done_callback(1)

    query_pl = state["query_pl"]
    done_idxs = set(state.get("done_text_idxs", []))

    for i in range(len(texts)):
        if i in done_idxs:
            continue

        text_i = texts[i]
        spans_i = context_spans[i]
        spans_rel_i = context_spans_rel[i]

        gaps, span_texts_en = spans_to_pieces(text_i, spans_i)

        if len(span_texts_en) != len(spans_rel_i):
            m = min(len(span_texts_en), len(spans_rel_i))
            span_texts_en = span_texts_en[:m]
            spans_rel_i = spans_rel_i[:m]
            gaps = gaps[: m + 1]

        translated_spans, translated_tp = await translate_text_with_span_repair_async(
            client=client,
            model=args.model,
            query_en=query_en,
            query_pl=query_pl,
            doc_label=int(labels[i]),
            span_texts_en=span_texts_en,
            spans_rel_i=[int(x) for x in spans_rel_i],
            think_process_en=think_process[i],
            max_attempts=args.max_prompt_attempts,
            delay_seconds=args.delay_seconds,
            temperature=args.temperature,
            max_retries=args.max_retries,
        )

        text_pl, new_spans_pl = rebuild_text_and_spans(gaps, translated_spans)

        state["texts_pl"][i] = text_pl
        state["context_spans_pl"][i] = new_spans_pl
        state["think_process_pl"][i] = translated_tp
        state["done_text_idxs"].append(i)
        state["active_model"] = args.model
        state["active_key_last6"] = api_key_last6
        await asyncio.to_thread(write_json_atomic, ckpt_path, state)
        if unit_done_callback:
            unit_done_callback(1)

    out_row = {
        "id": rid,
        "source_dataset": args.dataset_key,
        "source_dataset_hf": args.dataset,
        "query": state["query_pl"],
        "texts": state["texts_pl"],
        "context_spans": state["context_spans_pl"],
        "context_spans_relevance": row["context_spans_relevance"],
        "labels": row["labels"],
        "think_process": state["think_process_pl"],
        "translation_model": state.get("active_model"),
        "translation_source": getattr(args, "inference_source", "vllm"),
        "translation_key_last6": state.get("active_key_last6"),
        "translation_base_url": (args.base_url or None),
        "dataset_index": ds_idx,
    }
    if args.keep_original_columns:
        out_row.update(
            {
                "query_en": row["query"],
                "texts_en": row["texts"],
                "context_spans_en": row["context_spans"],
                "think_process_en": row["think_process"],
            }
        )

    return RowResult(rid=rid, ckpt_path=ckpt_path, out_row=out_row)


async def process_row_toxic(
    row: dict[str, Any],
    ds_idx: int,
    args: argparse.Namespace,
    api_key_last6: str,
    client: Any,
    unit_done_callback: Callable[[int], None] | None = None,
) -> RowResult:
    rid = row["id"]
    state = {
        "id": rid,
        "comment_text_pl": None,
        "active_model": None,
        "active_key_last6": None,
    }
    active_toxic_types = active_toxic_types_from_row(row)
    prompt = build_toxic_comment_prompt(
        comment_text_en=row["comment_text"],
        active_toxic_types=active_toxic_types,
    )
    schema = {
        "type": "object",
        "properties": {"comment_text_pl": {"type": "string"}},
        "required": ["comment_text_pl"],
        "additionalProperties": False,
    }
    translated_obj = await llm_call_json_async(
        client=client,
        model=args.model,
        system_prompt=SYSTEM_TEXT_SIMPLE,
        user_prompt=prompt,
        temperature=args.temperature,
        max_retries=args.max_retries,
        delay_seconds=args.delay_seconds,
        response_schema=schema,
    )
    state["comment_text_pl"] = (translated_obj.get("comment_text_pl") or "").strip()
    if not state["comment_text_pl"]:
        raise RuntimeError("Empty comment_text_pl from model")
    state["active_model"] = args.model
    state["active_key_last6"] = api_key_last6
    if unit_done_callback:
        unit_done_callback(1)

    out_row = build_out_row_from_state_toxic(state, row, ds_idx, args)
    return RowResult(rid=rid, ckpt_path=None, out_row=out_row)


async def process_row_wildguard(
    row: dict[str, Any],
    ds_idx: int,
    args: argparse.Namespace,
    api_key_last6: str,
    client: Any,
    unit_done_callback: Callable[[int], None] | None = None,
) -> RowResult:
    rid = resolve_row_id(row, ds_idx, args.dataset_key)
    subcategories = normalize_wildguard_subcategories(row.get("subcategory"))
    state = {
        "id": rid,
        "prompt_pl": None,
        "active_model": None,
        "active_key_last6": None,
    }
    prompt = build_wildguard_prompt(
        prompt_en=row["prompt"],
        subcategories=subcategories,
    )
    schema = {
        "type": "object",
        "properties": {"prompt_pl": {"type": "string"}},
        "required": ["prompt_pl"],
        "additionalProperties": False,
    }
    translated_obj = await llm_call_json_async(
        client=client,
        model=args.model,
        system_prompt=SYSTEM_TEXT_SIMPLE,
        user_prompt=prompt,
        temperature=args.temperature,
        max_retries=args.max_retries,
        delay_seconds=args.delay_seconds,
        response_schema=schema,
    )
    state["prompt_pl"] = (translated_obj.get("prompt_pl") or "").strip()
    if not state["prompt_pl"]:
        raise RuntimeError("Empty prompt_pl from model")
    state["active_model"] = args.model
    state["active_key_last6"] = api_key_last6
    if unit_done_callback:
        unit_done_callback(1)

    out_row = build_out_row_from_state_wildguard(state, row, ds_idx, args)
    return RowResult(rid=rid, ckpt_path=None, out_row=out_row)


async def process_row_nq_qa(
    row: dict[str, Any],
    ds_idx: int,
    args: argparse.Namespace,
    api_key_last6: str,
    client: Any,
    unit_done_callback: Callable[[int], None] | None = None,
) -> RowResult:
    rid = resolve_row_id(row, ds_idx, args.dataset_key)
    question_en = get_nq_qa_question_en(row)
    sampled_examples = (getattr(args, "_few_shot_examples_by_rid", {}) or {}).get(rid)
    state = {
        "id": rid,
        "question_pl": None,
        "answer_pl": None,
        "active_model": None,
        "active_key_last6": None,
    }
    if args.pair_prompt_mode == "no-few-shot":
        messages = build_nq_qa_zero_shot_messages(
            question_en=question_en,
            answer_en=row["answer"],
        )
        system_prompt = SYSTEM_NQ_QA_NO_FEW_SHOT
    else:
        messages = build_nq_qa_few_shot_messages(
            question_en=question_en,
            answer_en=row["answer"],
            examples_path=args.few_shot_examples_path,
            example_count=args.few_shot_example_count,
            sampled_examples=sampled_examples,
        )
        system_prompt = SYSTEM_NQ_QA
    schema = {
        "type": "object",
        "properties": {
            "question_pl": {"type": "string"},
            "answer_pl": {"type": "string"},
        },
        "required": ["question_pl", "answer_pl"],
        "additionalProperties": False,
    }
    translated_obj = await llm_call_json_async(
        client=client,
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=None,
        temperature=args.temperature,
        max_retries=args.max_retries,
        delay_seconds=args.delay_seconds,
        response_schema=schema,
        extra_messages=messages,
    )
    state["question_pl"] = (translated_obj.get("question_pl") or "").strip()
    state["answer_pl"] = (translated_obj.get("answer_pl") or "").strip()
    if not state["question_pl"]:
        raise RuntimeError("Empty question_pl from model")
    if not state["answer_pl"]:
        raise RuntimeError("Empty answer_pl from model")
    state["active_model"] = args.model
    state["active_key_last6"] = api_key_last6
    if unit_done_callback:
        unit_done_callback(1)

    out_row = build_out_row_from_state_nq_qa(state, row, ds_idx, args)
    return RowResult(rid=rid, ckpt_path=None, out_row=out_row)


async def process_row_hotpotqa(
    row: dict[str, Any],
    ds_idx: int,
    args: argparse.Namespace,
    api_key_last6: str,
    client: Any,
    unit_done_callback: Callable[[int], None] | None = None,
) -> RowResult:
    rid = resolve_row_id(row, ds_idx, args.dataset_key)
    sampled_examples = (getattr(args, "_few_shot_examples_by_rid", {}) or {}).get(rid)
    state = {
        "id": rid,
        "anchor_pl": None,
        "positive_pl": None,
        "active_model": None,
        "active_key_last6": None,
    }
    if args.pair_prompt_mode == "no-few-shot":
        messages = build_hotpotqa_zero_shot_messages(
            anchor_en=row["anchor"],
            positive_en=row["positive"],
        )
        system_prompt = SYSTEM_PAIR_TRANSLATION_NO_FEW_SHOT
    else:
        messages = build_hotpotqa_few_shot_messages(
            anchor_en=row["anchor"],
            positive_en=row["positive"],
            examples_path=args.few_shot_examples_path,
            example_count=args.few_shot_example_count,
            sampled_examples=sampled_examples,
        )
        system_prompt = SYSTEM_PAIR_TRANSLATION
    schema = {
        "type": "object",
        "properties": {
            "anchor_pl": {"type": "string"},
            "positive_pl": {"type": "string"},
        },
        "required": ["anchor_pl", "positive_pl"],
        "additionalProperties": False,
    }
    translated_obj = await llm_call_json_async(
        client=client,
        model=args.model,
        system_prompt=system_prompt,
        user_prompt=None,
        temperature=args.temperature,
        max_retries=args.max_retries,
        delay_seconds=args.delay_seconds,
        response_schema=schema,
        extra_messages=messages,
    )
    state["anchor_pl"] = (translated_obj.get("anchor_pl") or "").strip()
    state["positive_pl"] = (translated_obj.get("positive_pl") or "").strip()
    if not state["anchor_pl"]:
        raise RuntimeError("Empty anchor_pl from model")
    if not state["positive_pl"]:
        raise RuntimeError("Empty positive_pl from model")
    state["active_model"] = args.model
    state["active_key_last6"] = api_key_last6
    if unit_done_callback:
        unit_done_callback(1)

    out_row = build_out_row_from_state_hotpotqa(state, row, ds_idx, args)
    return RowResult(rid=rid, ckpt_path=None, out_row=out_row)


async def writer_loop(
    q: asyncio.Queue[RowResult | None],
    out_jsonl: str,
    done_ids: set,
    write_errors: list[BaseException],
) -> None:
    try:
        while True:
            item = await q.get()
            try:
                if item is None:
                    return

                if item.rid not in done_ids:
                    await asyncio.to_thread(append_jsonl, out_jsonl, item.out_row)
                    done_ids.add(item.rid)

                if item.ckpt_path:
                    state = await asyncio.to_thread(read_json, item.ckpt_path) or {"id": item.rid}
                    state["status"] = "done"
                    await asyncio.to_thread(write_json_atomic, item.ckpt_path, state)
            finally:
                q.task_done()
    except BaseException as exc:  # noqa: BLE001
        logging.exception("Writer loop failed")
        write_errors.append(exc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Translate dataset with checkpoints against OpenAI-compatible API "
            "(local vLLM server or external provider) or vLLM offline inference."
        )
    )
    p.add_argument(
        "--inference-source",
        default=os.getenv("INFERENCE_SOURCE", "vllm"),
        choices=["vllm", "external", "offline"],
        help="Inference source: local vLLM API server, external OpenAI-compatible API, or offline vLLM engine.",
    )
    p.add_argument(
        "--base-url",
        default=None,
        help="Override API base URL. If omitted, resolves from mode-specific env vars.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Override API key. If omitted, resolves from mode-specific env vars.",
    )
    p.add_argument("--model", default=os.getenv("MODEL_NAME"), required=os.getenv("MODEL_NAME") is None)
    p.add_argument("--parallel-requests", type=int, default=int(os.getenv("PARALLEL_REQUESTS", "2")))
    p.add_argument(
        "--offline-tensor-parallel-size",
        type=int,
        default=int(os.getenv("OFFLINE_TENSOR_PARALLEL_SIZE", os.getenv("GPU_COUNT", "1"))),
        help="Offline mode only: vLLM tensor parallel size.",
    )
    p.add_argument(
        "--offline-gpu-memory-utilization",
        type=float,
        default=float(os.getenv("OFFLINE_GPU_MEMORY_UTILIZATION", os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))),
        help="Offline mode only: vLLM GPU memory utilization.",
    )
    p.add_argument(
        "--offline-max-model-len",
        type=int,
        default=int(os.getenv("OFFLINE_MAX_MODEL_LEN", os.getenv("MAX_MODEL_LEN", "0"))),
        help="Offline mode only: max model length. 0 means vLLM default.",
    )
    p.add_argument(
        "--offline-max-num-seqs",
        type=int,
        default=int(os.getenv("OFFLINE_MAX_NUM_SEQS", "0")),
        help="Offline mode only: set vLLM max_num_seqs. 0 means default.",
    )
    p.add_argument(
        "--offline-max-num-batched-tokens",
        type=int,
        default=int(os.getenv("OFFLINE_MAX_NUM_BATCHED_TOKENS", "0")),
        help="Offline mode only: set vLLM max_num_batched_tokens. 0 means default.",
    )
    p.add_argument(
        "--offline-enforce-eager",
        action="store_true",
        default=os.getenv("OFFLINE_ENFORCE_EAGER", "0") == "1",
        help="Offline mode only: enable vLLM enforce_eager.",
    )
    p.add_argument(
        "--offline-dtype",
        default=os.getenv("OFFLINE_DTYPE", "auto"),
        help="Offline mode only: vLLM dtype (for example auto, float16, bfloat16).",
    )
    p.add_argument(
        "--offline-max-output-tokens",
        type=int,
        default=int(os.getenv("OFFLINE_MAX_OUTPUT_TOKENS", "2048")),
        help="Offline mode only: max generated tokens per LLM call.",
    )
    p.add_argument(
        "--offline-micro-batch-size",
        type=int,
        default=int(os.getenv("OFFLINE_MICRO_BATCH_SIZE", "150")),
        help="Offline mode only: target micro-batch size for in-process vLLM generate().",
    )

    p.add_argument("--delay-seconds", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=1)
    p.add_argument("--max-prompt-attempts", type=int, default=1)
    p.add_argument(
        "--few-shot-examples-path",
        default=os.getenv("FEW_SHOT_EXAMPLES_PATH", DEFAULT_FEW_SHOT_EXAMPLES_PATH),
        help="CSV file with EN->PL few-shot examples used for pair-style datasets such as nq_qa, gooaq, and hotpotqa.",
    )
    p.add_argument(
        "--few-shot-example-count",
        type=int,
        default=int(os.getenv("FEW_SHOT_EXAMPLE_COUNT", "3")),
        help="How many random few-shot examples to prepend for each pair-style prompt.",
    )
    p.add_argument(
        "--few-shot-shared-requests",
        type=int,
        default=int(os.getenv("FEW_SHOT_SHARED_REQUESTS", "10")),
        help="How many consecutive pair-style requests should reuse the same sampled few-shot examples.",
    )
    p.add_argument(
        "--pair-prompt-mode",
        default=os.getenv("PAIR_PROMPT_MODE", "few-shot"),
        choices=["few-shot", "no-few-shot"],
        help="Prompt mode for pair-style datasets (`nq_qa`, `gooaq`, `hotpotqa`). Default preserves the existing few-shot path.",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop whole run on first row-level translation error.",
    )

    p.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "nq", "nq_qa", "gooaq", "hotpotqa", "msmarco", "toxic", "wildguard"],
        help="Dataset selection: pass one or more keys. 'all' expands to NQ+MS MARCO.",
    )
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--out-dir", default="out_pl")
    p.add_argument("--out-jsonl-name", default="translated.jsonl")
    p.add_argument("--failed-jsonl-name", default="failed_rows.jsonl")
    p.add_argument(
        "--retry-failed-rows",
        action="store_true",
        help="Include rows previously present in failed_rows JSONL when resuming.",
    )
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--max-rows", type=int, default=0, help="0 = all")
    p.add_argument("--skip-rows", type=int, default=0)
    p.add_argument(
        "--keep-original-columns",
        dest="keep_original_columns",
        action="store_true",
        default=True,
        help="Keep original EN columns in the output JSONL (default).",
    )
    p.add_argument(
        "--drop-original-columns",
        dest="keep_original_columns",
        action="store_false",
        help="Exclude original EN columns from output JSONL.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-every", type=int, default=10, help="Log progress every N completed rows in non-TTY mode")
    p.add_argument(
        "--progress-bar",
        default=os.getenv("PROGRESS_BAR", "on"),
        choices=["auto", "on", "off"],
        help="Progress bar mode: auto=TTY only, on=always, off=disable tqdm",
    )
    p.add_argument(
        "--progress-metric",
        default=os.getenv("PROGRESS_METRIC", "checkpoints"),
        choices=["checkpoints", "rows", "both"],
        help="What tqdm should display: checkpoints (query+text units), rows, or both",
    )
    return p.parse_args()


def resolve_api_connection(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.inference_source == "offline":
        return None, None

    if args.inference_source == "external":
        base_url = (
            args.base_url
            or os.getenv("OPENAI_COMPAT_BASE_URL")
            or os.getenv("EXTERNAL_OPENAI_BASE_URL")
        )
        api_key = (
            args.api_key
            or os.getenv("OPENAI_COMPAT_API_KEY")
            or os.getenv("EXTERNAL_OPENAI_API_KEY")
        )
        if not base_url:
            raise RuntimeError(
                "External API mode requires base URL. Set OPENAI_COMPAT_BASE_URL (or EXTERNAL_OPENAI_BASE_URL) "
                "or pass --base-url."
            )
        if not api_key:
            raise RuntimeError(
                "External API mode requires API key. Set OPENAI_COMPAT_API_KEY (or EXTERNAL_OPENAI_API_KEY) "
                "or pass --api-key."
            )
        return base_url, api_key

    base_url = args.base_url or os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1")
    api_key = args.api_key if args.api_key is not None else os.getenv("OPENAI_API_KEY", "EMPTY")
    if not api_key:
        api_key = "EMPTY"
    return base_url, api_key


def validate_dataset_schema(ds: Any, dataset_label: str, dataset_key: str) -> None:
    cols = set(getattr(ds, "column_names", []) or [])
    if dataset_key == "toxic":
        required = REQUIRED_TOXIC_COLUMNS
    elif dataset_key in ("nq_qa", "gooaq"):
        required = REQUIRED_NQ_QA_PRIMARY_COLUMNS
        if "question" not in cols and "query" not in cols:
            raise RuntimeError(
                f"Dataset '{dataset_label}' is missing required question column. "
                f"Expected one of: ['question', 'query']. Available columns: {sorted(cols)}"
            )
    elif dataset_key == "hotpotqa":
        required = REQUIRED_HOTPOTQA_COLUMNS
    elif dataset_key == "wildguard":
        required = REQUIRED_WILDGUARD_COLUMNS
    else:
        required = REQUIRED_CONTEXT_RELEVANCE_COLUMNS
    missing = [c for c in required if c not in cols]
    if missing:
        raise RuntimeError(
            f"Dataset '{dataset_label}' is missing required columns: {missing}. Available columns: {sorted(cols)}"
        )


def load_dataset_for_run(dataset_hf_id: str, dataset_key: str, split: str, hf_token: str | None) -> Any:
    if dataset_key == "wildguard":
        config_name = WILDGUARD_CONFIG_BY_SPLIT.get(split)
        if config_name is None:
            supported = ", ".join(sorted(WILDGUARD_CONFIG_BY_SPLIT.keys()))
            raise RuntimeError(
                f"Dataset '{dataset_hf_id}' does not support split='{split}'. "
                f"Use one of: {supported}."
            )
        # WildGuard uses config names for train/test variants; each config exposes a train split.
        return load_dataset(dataset_hf_id, name=config_name, split="train", token=hf_token)

    if dataset_key == "hotpotqa":
        return load_dataset(dataset_hf_id, name=HOTPOTQA_DEFAULT_CONFIG, split=split, token=hf_token)

    if dataset_key == "gooaq":
        return load_dataset(dataset_hf_id, name=GOOAQ_DEFAULT_CONFIG, split=split, token=hf_token)

    return load_dataset(dataset_hf_id, split=split, token=hf_token)


async def run_single_dataset_async(args: argparse.Namespace) -> int:
    args.base_url, args.api_key = resolve_api_connection(args)
    use_checkpoints = args.dataset_key not in ("toxic", "wildguard", "nq_qa", "gooaq", "hotpotqa")

    if use_checkpoints and args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.out_dir, "checkpoints")

    os.makedirs(args.out_dir, exist_ok=True)
    if use_checkpoints and args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    out_jsonl = os.path.join(args.out_dir, args.out_jsonl_name)
    failed_jsonl = os.path.join(args.out_dir, args.failed_jsonl_name)
    done_ids = load_done_ids_from_jsonl(out_jsonl)
    failed_ids = set() if args.retry_failed_rows else load_done_ids_from_jsonl(failed_jsonl)

    hf_token = os.getenv("HF_TOKEN") or None
    ds = load_dataset_for_run(args.dataset, args.dataset_key, args.split, hf_token)
    validate_dataset_schema(ds, args.dataset, args.dataset_key)
    total = len(ds)

    skip = max(0, int(args.skip_rows))
    if skip >= total:
        print(f"--skip-rows={skip} >= dataset size={total}. Nothing to do.")
        return 0

    start_idx = skip
    end_idx = min(total, start_idx + int(args.max_rows)) if args.max_rows and args.max_rows > 0 else total

    recovered_from_ckpt = 0
    skipped_failed = 0
    candidates_with_ckpt: list[tuple[int, dict[str, Any]]] = []
    candidates_fresh: list[tuple[int, dict[str, Any]]] = []
    for ds_idx in range(start_idx, end_idx):
        row = ds[ds_idx]
        rid = resolve_row_id(row, ds_idx, args.dataset_key)
        if rid in done_ids:
            continue
        if rid in failed_ids:
            skipped_failed += 1
            continue

        if use_checkpoints:
            stem = checkpoint_stem_from_id(rid)
            ckpt_path = os.path.join(args.checkpoint_dir, f"{stem}.json")
            state = read_json(ckpt_path) or {}

            if state and checkpoint_is_complete(state, expected_texts=len(row["texts"])):
                append_jsonl(out_jsonl, build_out_row_from_state(state, row, ds_idx, args))
                done_ids.add(rid)
                state["status"] = "done"
                write_json_atomic(ckpt_path, state)
                recovered_from_ckpt += 1
                continue

            if state:
                candidates_with_ckpt.append((ds_idx, row))
            else:
                candidates_fresh.append((ds_idx, row))
        else:
            candidates_fresh.append((ds_idx, row))

    candidates_with_ckpt = reorder_candidates_for_prompt_cache(args.dataset_key, candidates_with_ckpt)
    candidates_fresh = reorder_candidates_for_prompt_cache(args.dataset_key, candidates_fresh)
    candidates = candidates_with_ckpt + candidates_fresh
    args._few_shot_examples_by_rid = build_shared_few_shot_examples_by_rid(
        candidates=candidates,
        dataset_key=args.dataset_key,
        prompt_mode=args.pair_prompt_mode,
        examples_path=args.few_shot_examples_path,
        example_count=args.few_shot_example_count,
        shared_requests=args.few_shot_shared_requests,
    )

    if not candidates:
        if skipped_failed:
            print(
                "Nothing to translate (rows already done or skipped because they are present in failed_rows). "
                "Use --retry-failed-rows to include failed rows."
            )
        else:
            print("Nothing to translate (all rows already done in selected window).")
        return 0

    effective_parallel = max(1, args.parallel_requests)
    if args.inference_source == "offline":
        effective_parallel = max(effective_parallel, int(args.offline_micro_batch_size))

    logging.info(
        "Translation run: source=%s dataset_key=%s dataset=%s split=%s model=%s base_url=%s parallel=%d offline_micro_batch=%d range=%d..%d total_in_range=%d pending=%d done_before=%d skipped_failed=%d retry_failed_rows=%s recovered_from_checkpoints=%d pending_with_checkpoints=%d pending_new=%d",
        args.inference_source,
        args.dataset_key,
        args.dataset,
        args.split,
        args.model,
        args.base_url,
        effective_parallel,
        int(args.offline_micro_batch_size),
        start_idx,
        end_idx - 1,
        end_idx - start_idx,
        len(candidates),
        (end_idx - start_idx) - len(candidates),
        skipped_failed,
        bool(args.retry_failed_rows),
        recovered_from_ckpt,
        len(candidates_with_ckpt),
        len(candidates_fresh),
    )
    if pair_prompt_uses_few_shot(args.dataset_key, args.pair_prompt_mode):
        logging.info(
            "Few-shot grouping: shared_requests=%d example_count=%d groups=%d",
            int(args.few_shot_shared_requests),
            int(args.few_shot_example_count),
            (len(candidates) + max(1, int(args.few_shot_shared_requests)) - 1) // max(1, int(args.few_shot_shared_requests)),
        )

    api_key_last6 = "OFFLINE" if args.inference_source == "offline" else (args.api_key[-6:] if args.api_key else "EMPTY")
    result_queue: asyncio.Queue[RowResult | None] = asyncio.Queue(
        maxsize=max(4, args.parallel_requests * 2)
    )
    write_errors: list[BaseException] = []

    writer = asyncio.create_task(
        writer_loop(result_queue, out_jsonl, done_ids, write_errors)
    )
    logging.info("Writer task started. Output: %s", out_jsonl)
    logging.info("Failed rows will be appended to: %s", failed_jsonl)

    total_units = 0
    done_units_before = 0
    for ds_idx, row in candidates:
        row_units = 1 if args.dataset_key in ("toxic", "wildguard", "nq_qa", "gooaq", "hotpotqa") else 1 + len(row["texts"])
        total_units += row_units

        if use_checkpoints:
            rid = resolve_row_id(row, ds_idx, args.dataset_key)
            stem = checkpoint_stem_from_id(rid)
            ckpt_path = os.path.join(args.checkpoint_dir, f"{stem}.json")
            state = read_json(ckpt_path) or {}
            if state.get("query_pl"):
                done_units_before += 1

            done_idxs = {int(x) for x in state.get("done_text_idxs", []) if isinstance(x, int)}
            done_units_before += len([i for i in done_idxs if 0 <= i < len(row["texts"])])

    if total_units > 0:
        done_units_before = min(done_units_before, total_units)
    logging.info(
        "Checkpoint units: %d/%d done at start",
        done_units_before,
        total_units,
    )

    is_tty = sys.stderr.isatty()
    show_pbar = args.progress_bar == "on" or (args.progress_bar == "auto" and is_tty)
    non_tty_progress = not is_tty
    show_rows_bar = args.progress_metric in ("rows", "both")
    show_units_bar = args.progress_metric in ("checkpoints", "both")
    pbar_rows = tqdm(
        total=len(candidates),
        desc="Completed rows",
        unit="row",
        dynamic_ncols=True,
        mininterval=0.5,
        ascii=not is_tty,
        disable=(not show_pbar) or (not show_rows_bar),
    )
    pbar_units = tqdm(
        total=total_units,
        initial=done_units_before,
        desc="Checkpoint units",
        unit="unit",
        dynamic_ncols=True,
        mininterval=0.5,
        ascii=not is_tty,
        disable=(not show_pbar) or (not show_units_bar),
    )
    started_at = time.time()
    completed = 0
    log_every = max(1, int(args.log_every))
    completed_units = done_units_before
    units_lock = asyncio.Lock()

    def mark_units_done(increment: int) -> None:
        nonlocal completed_units
        if increment <= 0:
            return
        completed_units += increment
        pbar_units.update(increment)

    sem = asyncio.Semaphore(effective_parallel)

    @asynccontextmanager
    async def build_inference_client():
        if args.inference_source == "offline":
            offline_client = OfflineVllmClient(args)
            try:
                yield offline_client
            finally:
                await offline_client.aclose()
            return

        async with AsyncOpenAI(api_key=args.api_key, base_url=args.base_url) as api_client:
            yield api_client

    async with build_inference_client() as client:
        async def process_with_limit(ds_idx: int, row: dict[str, Any]) -> RowResult:
            async with sem:
                if args.dataset_key == "toxic":
                    return await process_row_toxic(row, ds_idx, args, api_key_last6, client, mark_units_done)
                if args.dataset_key in ("nq_qa", "gooaq"):
                    return await process_row_nq_qa(row, ds_idx, args, api_key_last6, client, mark_units_done)
                if args.dataset_key == "hotpotqa":
                    return await process_row_hotpotqa(row, ds_idx, args, api_key_last6, client, mark_units_done)
                if args.dataset_key == "wildguard":
                    return await process_row_wildguard(row, ds_idx, args, api_key_last6, client, mark_units_done)
                return await process_row(row, ds_idx, args, api_key_last6, client, mark_units_done)

        tasks = [asyncio.create_task(process_with_limit(ds_idx, row)) for ds_idx, row in candidates]
        task_meta: dict[asyncio.Task[RowResult], tuple[int, str]] = {
            task: (ds_idx, resolve_row_id(row, ds_idx, args.dataset_key)) for task, (ds_idx, row) in zip(tasks, candidates)
        }
        failed_rows = 0
        try:
            pending_tasks = set(tasks)
            while pending_tasks:
                if write_errors:
                    raise RuntimeError(f"Writer task failed: {write_errors[0]}") from write_errors[0]
                done_now, pending_tasks = await asyncio.wait(
                    pending_tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for fut in done_now:
                    if write_errors:
                        raise RuntimeError(f"Writer task failed: {write_errors[0]}") from write_errors[0]
                    try:
                        result = await fut
                    except RateLimitReached:
                        await asyncio.sleep(1 + random.random())
                        raise
                    except Exception as exc:  # noqa: BLE001
                        ds_idx, rid = task_meta.get(fut, (-1, "unknown"))
                        failed_rows += 1
                        if args.fail_fast:
                            raise

                        logging.exception("Row failed (dataset_index=%s id=%s): %s", ds_idx, rid, exc)
                        failed_obj = {
                            "id": rid,
                            "source_dataset": args.dataset_key,
                            "source_dataset_hf": args.dataset,
                            "dataset_index": ds_idx,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "timestamp_unix": int(time.time()),
                        }
                        await asyncio.to_thread(append_jsonl, failed_jsonl, failed_obj)

                        if use_checkpoints and rid != "unknown":
                            stem = checkpoint_stem_from_id(rid)
                            ckpt_path = os.path.join(args.checkpoint_dir, f"{stem}.json")
                            state = await asyncio.to_thread(read_json, ckpt_path) or {"id": rid, "dataset_index": ds_idx}
                            state["status"] = "failed"
                            state["last_error"] = str(exc)
                            state["failed_at_unix"] = int(time.time())
                            await asyncio.to_thread(write_json_atomic, ckpt_path, state)

                        pbar_rows.update(1)
                        completed += 1
                        continue

                    await result_queue.put(result)
                    pbar_rows.update(1)
                    completed += 1
                    if (non_tty_progress and not show_pbar) and (
                        completed == 1 or completed % log_every == 0 or completed == len(candidates)
                    ):
                        elapsed = time.time() - started_at
                        rate = completed / elapsed if elapsed > 0 else 0.0
                        eta_seconds = (len(candidates) - completed) / rate if rate > 0 else 0.0
                        async with units_lock:
                            units_done_now = completed_units
                        logging.info(
                            "Progress: %d/%d rows (%.1f%%), units=%d/%d, rate=%.2f row/s, elapsed=%s, eta=%s",
                            completed,
                            len(candidates),
                            100.0 * completed / len(candidates),
                            units_done_now,
                            total_units,
                            rate,
                            format_seconds(elapsed),
                            format_seconds(eta_seconds),
                        )

            await result_queue.join()
            if write_errors:
                raise RuntimeError(f"Writer task failed: {write_errors[0]}") from write_errors[0]
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            pbar_rows.close()
            pbar_units.close()
            await result_queue.put(None)
            try:
                await asyncio.wait_for(writer, timeout=10)
            except asyncio.TimeoutError:
                writer.cancel()
                await asyncio.gather(writer, return_exceptions=True)

    if failed_rows:
        logging.warning("Done with row failures: %d failed rows. See %s", failed_rows, failed_jsonl)
    else:
        logging.info("Done. Output: %s", out_jsonl)
    return 0


async def run_async(args: argparse.Namespace) -> int:
    selected_keys = selected_dataset_keys(args.datasets)
    runs: list[tuple[str, str]] = [(k, DATASET_PRESETS[k]) for k in selected_keys]

    logging.info(
        "Selected dataset runs: %s",
        ", ".join([f"{k} -> {hf}" for k, hf in runs]),
    )

    for dataset_key, dataset_hf_id in runs:
        run_args = copy.deepcopy(args)
        run_args.dataset_key = dataset_key
        run_args.dataset = dataset_hf_id
        run_args.out_dir = os.path.join(args.out_dir, dataset_key)
        run_args.checkpoint_dir = None
        logging.info(
            "Starting dataset run: key=%s hf_id=%s out_dir=%s split=%s",
            dataset_key,
            dataset_hf_id,
            run_args.out_dir,
            run_args.split,
        )
        rc = await run_single_dataset_async(run_args)
        if rc != 0:
            return rc

    return 0


def selected_dataset_keys(datasets_arg: list[str]) -> list[str]:
    out: list[str] = []
    for key in datasets_arg:
        expanded = ["nq", "msmarco"] if key == "all" else [key]
        for item in expanded:
            if item not in out:
                out.append(item)
    return out


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    quiet_external_loggers()
    return asyncio.run(run_async(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted (CTRL+C).", file=sys.stderr)
        raise SystemExit(130)
