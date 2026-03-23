#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

from tqdm import tqdm

from qa_rule_based_filter import evaluate_row
from run_answer_relevance_vllm import (
    extract_question_answer,
    read_jsonl_rows,
    resolve_row_id,
    selected_dataset_keys,
)
from translation_core import append_jsonl, load_done_ids_from_jsonl

FILTER_PREFIX = "bad_answer_filter_rules"


@dataclass
class RuleFilterResult:
    rid: str
    out_row: dict[str, Any]


def task_output_jsonl_name() -> str:
    return f"{FILTER_PREFIX}.jsonl"


def task_failed_jsonl_name() -> str:
    return f"{FILTER_PREFIX}_failed_rows.jsonl"


def build_output_row(row: dict[str, Any], *, evaluation: dict[str, Any]) -> dict[str, Any]:
    out_row = dict(row)
    out_row[FILTER_PREFIX] = {
        "is_good": bool(evaluation["is_good"]),
        "reasons": list(evaluation["reasons"]),
        "reasons_str": str(evaluation["reasons_str"]),
    }
    out_row[f"{FILTER_PREFIX}_source"] = "rule_based"
    out_row[f"{FILTER_PREFIX}_timestamp_unix"] = int(time.time())
    return out_row


def process_row(row: dict[str, Any], row_idx: int, dataset_key: str) -> RuleFilterResult:
    rid = resolve_row_id(row, dataset_key, row_idx)
    question, answer = extract_question_answer(row, dataset_key)
    evaluation = evaluate_row(question, answer)
    return RuleFilterResult(rid=rid, out_row=build_output_row(row, evaluation=evaluation))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Filter translated QA pairs with deterministic rule-based heuristics and write resumable JSONL outputs."
        )
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", "nq_qa", "hotpotqa"],
        help="Dataset selection. 'all' expands to nq_qa and hotpotqa.",
    )
    p.add_argument("--out-dir", default="out_pl")
    p.add_argument("--input-jsonl-name", default="translated.jsonl")
    p.add_argument("--out-jsonl-name", default=task_output_jsonl_name())
    p.add_argument("--failed-jsonl-name", default=task_failed_jsonl_name())
    p.add_argument(
        "--retry-failed-rows",
        action="store_true",
        help="Include rows previously present in failed_rows JSONL when resuming.",
    )
    p.add_argument("--max-rows", type=int, default=0, help="0 = all")
    p.add_argument("--skip-rows", type=int, default=0)
    p.add_argument("--max-workers", type=int, default=max(1, min(32, (os.cpu_count() or 1))))
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-every", type=int, default=100, help="Log progress every N completed rows in non-TTY mode")
    p.add_argument(
        "--progress-bar",
        default="on",
        choices=["auto", "on", "off"],
        help="Progress bar mode: auto=TTY only, on=always, off=disable tqdm",
    )
    return p.parse_args()


def run_single_dataset(args: argparse.Namespace) -> int:
    dataset_dir = os.path.join(args.out_dir, args.dataset_key)
    input_jsonl = os.path.join(dataset_dir, args.input_jsonl_name)
    out_jsonl = os.path.join(dataset_dir, args.out_jsonl_name)
    failed_jsonl = os.path.join(dataset_dir, args.failed_jsonl_name)
    os.makedirs(dataset_dir, exist_ok=True)

    rows = read_jsonl_rows(input_jsonl)
    total = len(rows)
    skip = max(0, int(args.skip_rows))
    if skip >= total:
        print(f"--skip-rows={skip} >= dataset size={total}. Nothing to do.")
        return 0

    end_idx = min(total, skip + int(args.max_rows)) if args.max_rows and args.max_rows > 0 else total
    done_ids = load_done_ids_from_jsonl(out_jsonl)
    failed_ids = set() if args.retry_failed_rows else load_done_ids_from_jsonl(failed_jsonl)

    candidates: list[tuple[int, dict[str, Any]]] = []
    skipped_failed = 0
    for row_idx in range(skip, end_idx):
        row = rows[row_idx]
        rid = resolve_row_id(row, args.dataset_key, row_idx)
        if rid in done_ids:
            continue
        if rid in failed_ids:
            skipped_failed += 1
            continue
        candidates.append((row_idx, row))

    if not candidates:
        if skipped_failed:
            print(
                "Nothing to score (rows already done or skipped because they are present in failed_rows). "
                "Use --retry-failed-rows to include failed rows."
            )
        else:
            print("Nothing to score (all rows already done in selected window).")
        return 0

    logging.info(
        "Rule-based QA filter run: dataset_key=%s input=%s output=%s range=%d..%d total_in_range=%d "
        "pending=%d done_before=%d skipped_failed=%d retry_failed_rows=%s max_workers=%d",
        args.dataset_key,
        input_jsonl,
        out_jsonl,
        skip,
        end_idx - 1,
        end_idx - skip,
        len(candidates),
        (end_idx - skip) - len(candidates),
        skipped_failed,
        bool(args.retry_failed_rows),
        int(args.max_workers),
    )

    is_tty = sys.stderr.isatty()
    show_pbar = args.progress_bar == "on" or (args.progress_bar == "auto" and is_tty)
    non_tty_progress = not is_tty
    pbar_rows = tqdm(
        total=len(candidates),
        desc="Completed rows",
        unit="row",
        dynamic_ncols=True,
        mininterval=0.5,
        ascii=not is_tty,
        disable=not show_pbar,
    )
    started_at = time.time()
    completed = 0
    failed_rows = 0
    log_every = max(1, int(args.log_every))

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        future_map = {
            ex.submit(process_row, row, row_idx, args.dataset_key): (row_idx, resolve_row_id(row, args.dataset_key, row_idx))
            for row_idx, row in candidates
        }
        for future in as_completed(future_map):
            row_idx, rid = future_map[future]
            try:
                result = future.result()
                if result.rid not in done_ids:
                    append_jsonl(out_jsonl, result.out_row)
                    done_ids.add(result.rid)
            except Exception as exc:  # noqa: BLE001
                failed_rows += 1
                logging.exception("Row failed (row_idx=%s id=%s): %s", row_idx, rid, exc)
                append_jsonl(
                    failed_jsonl,
                    {
                        "id": rid,
                        "source_dataset": args.dataset_key,
                        "row_idx": row_idx,
                        "error": str(exc),
                        "error_type": type(exc).__name__,
                        "timestamp_unix": int(time.time()),
                    },
                )
            completed += 1
            pbar_rows.update(1)
            if (non_tty_progress and not show_pbar) and (
                completed == 1 or completed % log_every == 0 or completed == len(candidates)
            ):
                elapsed = time.time() - started_at
                rate = completed / elapsed if elapsed > 0 else 0.0
                logging.info(
                    "Progress: %d/%d rows (%.1f%%), rate=%.2f row/s, elapsed=%.1fs",
                    completed,
                    len(candidates),
                    100.0 * completed / len(candidates),
                    rate,
                    elapsed,
                )

    pbar_rows.close()

    if failed_rows:
        logging.warning("Done with row failures: %d failed rows. See %s", failed_rows, failed_jsonl)
    else:
        logging.info("Done. Output: %s", out_jsonl)
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    selected_keys = selected_dataset_keys(args.datasets)
    logging.info("Selected dataset runs: %s", ", ".join(selected_keys))

    for dataset_key in selected_keys:
        run_args = copy.deepcopy(args)
        run_args.dataset_key = dataset_key
        logging.info(
            "Starting rule-based QA filter run: key=%s out_dir=%s input_jsonl=%s",
            dataset_key,
            os.path.join(args.out_dir, dataset_key),
            args.input_jsonl_name,
        )
        rc = run_single_dataset(run_args)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
