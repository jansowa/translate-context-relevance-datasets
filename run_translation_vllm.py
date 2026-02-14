#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from translate_context_relevance_dataset import (
    SYSTEM_QUERY,
    RateLimitReached,
    append_jsonl,
    build_query_prompt,
    checkpoint_stem_from_id,
    llm_call_json,
    load_done_ids_from_jsonl,
    read_json,
    rebuild_text_and_spans,
    spans_to_pieces,
    translate_text_with_span_repair,
    write_json_atomic,
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
    out_row: Dict[str, Any]


def checkpoint_is_complete(state: Dict[str, Any], expected_texts: int) -> bool:
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
    state: Dict[str, Any],
    row: Dict[str, Any],
    ds_idx: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out_row = {
        "id": row["id"],
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


def process_row(
    row: Dict[str, Any],
    ds_idx: int,
    args: argparse.Namespace,
    api_key_last6: str,
    unit_done_callback: Optional[Callable[[int], None]] = None,
) -> RowResult:
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)

    rid = row["id"]
    stem = checkpoint_stem_from_id(rid)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{stem}.json")

    state = read_json(ckpt_path) or {}
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
        write_json_atomic(ckpt_path, state)

    query_en = row["query"]
    texts: List[str] = row["texts"]
    context_spans: List[List[List[int]]] = row["context_spans"]
    context_spans_rel: List[List[int]] = row["context_spans_relevance"]
    labels: List[int] = row["labels"]
    think_process: List[str] = row["think_process"]

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
        q_json = llm_call_json(
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
        write_json_atomic(ckpt_path, state)
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

        translated_spans, translated_tp = translate_text_with_span_repair(
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
        write_json_atomic(ckpt_path, state)
        if unit_done_callback:
            unit_done_callback(1)

    out_row = {
        "id": rid,
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


def writer_loop(
    q: "queue.Queue[Optional[RowResult]]",
    out_jsonl: str,
    done_ids: set,
    done_lock: threading.Lock,
    write_errors: List[BaseException],
) -> None:
    try:
        while True:
            item = q.get()
            if item is None:
                q.task_done()
                return

            if item.rid not in done_ids:
                append_jsonl(out_jsonl, item.out_row)
                with done_lock:
                    done_ids.add(item.rid)

            state = read_json(item.ckpt_path) or {"id": item.rid}
            state["status"] = "done"
            write_json_atomic(item.ckpt_path, state)
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

    p.add_argument("--dataset", default="zilliz/natural_questions-context-relevance-with-think")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--out-dir", default="out_pl")
    p.add_argument("--out-jsonl-name", default="translated.jsonl")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--max-rows", type=int, default=0, help="0 = all")
    p.add_argument("--skip-rows", type=int, default=0)
    p.add_argument("--keep-original-columns", default=True, action="store_true")
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


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    quiet_external_loggers()

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.out_dir, "checkpoints")

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    out_jsonl = os.path.join(args.out_dir, args.out_jsonl_name)
    done_ids = load_done_ids_from_jsonl(out_jsonl)
    done_lock = threading.Lock()

    ds = load_dataset(args.dataset, split=args.split)
    total = len(ds)

    skip = max(0, int(args.skip_rows))
    if skip >= total:
        print(f"--skip-rows={skip} >= dataset size={total}. Nothing to do.")
        return 0

    start_idx = skip
    end_idx = min(total, start_idx + int(args.max_rows)) if args.max_rows and args.max_rows > 0 else total

    recovered_from_ckpt = 0
    candidates_with_ckpt: List[Tuple[int, Dict[str, Any]]] = []
    candidates_fresh: List[Tuple[int, Dict[str, Any]]] = []
    for ds_idx in range(start_idx, end_idx):
        row = ds[ds_idx]
        rid = row["id"]
        if rid in done_ids:
            continue

        stem = checkpoint_stem_from_id(rid)
        ckpt_path = os.path.join(args.checkpoint_dir, f"{stem}.json")
        state = read_json(ckpt_path) or {}

        # If checkpoint already contains a full row, flush it to JSONL and mark as done.
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

    # Prefer closing already-started rows before opening new checkpoints.
    candidates = candidates_with_ckpt + candidates_fresh

    if not candidates:
        print("Nothing to translate (all rows already done in selected window).")
        return 0

    logging.info(
        "Translation run: dataset=%s split=%s model=%s parallel=%d range=%d..%d total_in_range=%d pending=%d done_before=%d recovered_from_checkpoints=%d pending_with_checkpoints=%d pending_new=%d",
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
    result_queue: "queue.Queue[Optional[RowResult]]" = queue.Queue(maxsize=max(4, args.parallel_requests * 2))
    write_errors: List[BaseException] = []

    writer = threading.Thread(
        target=writer_loop,
        args=(result_queue, out_jsonl, done_ids, done_lock, write_errors),
        daemon=True,
    )
    writer.start()
    logging.info("Writer thread started. Output: %s", out_jsonl)

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
    units_lock = threading.Lock()

    def mark_units_done(increment: int) -> None:
        nonlocal completed_units
        if increment <= 0:
            return
        with units_lock:
            completed_units += increment
            pbar_units.update(increment)

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.parallel_requests)) as executor:
            futures = [
                executor.submit(process_row, row, ds_idx, args, api_key_last6, mark_units_done)
                for ds_idx, row in candidates
            ]

            for fut in as_completed(futures):
                if write_errors:
                    raise RuntimeError(f"Writer thread failed: {write_errors[0]}") from write_errors[0]
                try:
                    result = fut.result()
                except RateLimitReached:
                    time.sleep(1 + random.random())
                    raise

                result_queue.put(result)
                pbar_rows.update(1)
                completed += 1
                if (non_tty_progress and not show_pbar) and (
                    completed == 1 or completed % log_every == 0 or completed == len(candidates)
                ):
                    elapsed = time.time() - started_at
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta_seconds = (len(candidates) - completed) / rate if rate > 0 else 0.0
                    with units_lock:
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

        result_queue.join()
        if write_errors:
            raise RuntimeError(f"Writer thread failed: {write_errors[0]}") from write_errors[0]
    finally:
        pbar_rows.close()
        pbar_units.close()
        result_queue.put(None)
        writer.join(timeout=10)

    logging.info("Done. Output: %s", out_jsonl)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted (CTRL+C).", file=sys.stderr)
        raise SystemExit(130)
