#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import logging
import os
import openai
import random
import sys
import time
import copy
from dataclasses import dataclass
from typing import Any, Callable

from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm import tqdm

from translation_core import (
    SYSTEM_QUERY,
    SYSTEM_TEXT,
    RateLimitReached,
    append_jsonl,
    build_text_prompt,
    build_text_prompt_dictforced,
    build_text_prompt_strict,
    build_query_prompt,
    checkpoint_stem_from_id,
    escape_control_chars_in_json_strings,
    extract_first_json_object,
    load_done_ids_from_jsonl,
    read_json,
    rebuild_text_and_spans,
    spans_to_pieces,
    write_json_atomic,
)

DATASET_PRESETS: dict[str, str] = {
    "nq": "zilliz/natural_questions-context-relevance-with-think",
    "msmarco": "zilliz/msmarco-context-relevance-with-think",
}

REQUIRED_DATASET_COLUMNS = (
    "id",
    "query",
    "texts",
    "context_spans",
    "context_spans_relevance",
    "labels",
    "think_process",
)


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
    ckpt_path: str
    out_row: dict[str, Any]


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


async def llm_call_json_async(
    client: AsyncOpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_retries: int,
    delay_seconds: float,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    last_err: BaseException | None = None
    schema_enabled = response_schema is not None

    for attempt in range(max_retries):
        try:
            kwargs = dict(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
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
            await asyncio.sleep(min(60, (2 ** attempt) + random.random()))

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
    ][:max_attempts]

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
    client: AsyncOpenAI,
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
        description="Translate dataset with checkpoints against OpenAI-compatible vLLM server."
    )
    p.add_argument("--base-url", default=os.getenv("VLLM_BASE_URL", "http://vllm:8000/v1"))
    p.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--model", default=os.getenv("MODEL_NAME"), required=os.getenv("MODEL_NAME") is None)
    p.add_argument("--parallel-requests", type=int, default=int(os.getenv("PARALLEL_REQUESTS", "2")))

    p.add_argument("--delay-seconds", type=float, default=0.0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--max-prompt-attempts", type=int, default=4)
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop whole run on first row-level translation error.",
    )

    p.add_argument(
        "--datasets",
        default="all",
        choices=["all", "nq", "msmarco"],
        help="Dataset selection: all=run NQ then MS MARCO, or a single dataset key.",
    )
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--out-dir", default="out_pl")
    p.add_argument("--out-jsonl-name", default="translated.jsonl")
    p.add_argument("--failed-jsonl-name", default="failed_rows.jsonl")
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


def validate_dataset_schema(ds: Any, dataset_label: str) -> None:
    cols = set(getattr(ds, "column_names", []) or [])
    missing = [c for c in REQUIRED_DATASET_COLUMNS if c not in cols]
    if missing:
        raise RuntimeError(
            f"Dataset '{dataset_label}' is missing required columns: {missing}. Available columns: {sorted(cols)}"
        )


async def run_single_dataset_async(args: argparse.Namespace) -> int:
    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.out_dir, "checkpoints")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    out_jsonl = os.path.join(args.out_dir, args.out_jsonl_name)
    failed_jsonl = os.path.join(args.out_dir, args.failed_jsonl_name)
    done_ids = load_done_ids_from_jsonl(out_jsonl)

    ds = load_dataset(args.dataset, split=args.split)
    validate_dataset_schema(ds, args.dataset)
    total = len(ds)

    skip = max(0, int(args.skip_rows))
    if skip >= total:
        print(f"--skip-rows={skip} >= dataset size={total}. Nothing to do.")
        return 0

    start_idx = skip
    end_idx = min(total, start_idx + int(args.max_rows)) if args.max_rows and args.max_rows > 0 else total

    recovered_from_ckpt = 0
    candidates_with_ckpt: list[tuple[int, dict[str, Any]]] = []
    candidates_fresh: list[tuple[int, dict[str, Any]]] = []
    for ds_idx in range(start_idx, end_idx):
        row = ds[ds_idx]
        rid = row["id"]
        if rid in done_ids:
            continue

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

    candidates = candidates_with_ckpt + candidates_fresh

    if not candidates:
        print("Nothing to translate (all rows already done in selected window).")
        return 0

    logging.info(
        "Translation run: dataset_key=%s dataset=%s split=%s model=%s parallel=%d range=%d..%d total_in_range=%d pending=%d done_before=%d recovered_from_checkpoints=%d pending_with_checkpoints=%d pending_new=%d",
        args.dataset_key,
        args.dataset,
        args.split,
        args.model,
        max(1, args.parallel_requests),
        start_idx,
        end_idx - 1,
        end_idx - start_idx,
        len(candidates),
        (end_idx - start_idx) - len(candidates),
        recovered_from_ckpt,
        len(candidates_with_ckpt),
        len(candidates_fresh),
    )

    api_key_last6 = args.api_key[-6:] if args.api_key else "EMPTY"
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
        row_units = 1 + len(row["texts"])
        total_units += row_units

        rid = row["id"]
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
        "Checkpoint units: %d/%d done at start (unit = query + one text)",
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

    sem = asyncio.Semaphore(max(1, args.parallel_requests))

    async with AsyncOpenAI(api_key=args.api_key, base_url=args.base_url) as client:
        async def process_with_limit(ds_idx: int, row: dict[str, Any]) -> RowResult:
            async with sem:
                return await process_row(row, ds_idx, args, api_key_last6, client, mark_units_done)

        tasks = [asyncio.create_task(process_with_limit(ds_idx, row)) for ds_idx, row in candidates]
        task_meta: dict[asyncio.Task[RowResult], tuple[int, str]] = {
            task: (ds_idx, row["id"]) for task, (ds_idx, row) in zip(tasks, candidates)
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

                        if rid != "unknown":
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
    selected_keys = ["nq", "msmarco"] if args.datasets == "all" else [args.datasets]
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
