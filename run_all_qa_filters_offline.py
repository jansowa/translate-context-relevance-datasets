#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from run_answer_relevance_reranker import custom_output_jsonl_name as reranker_output_jsonl_name
from run_answer_relevance_vllm import custom_bad_answer_filter_output_jsonl_name
from run_merge_qa_outputs import merge_rows_by_id, write_jsonl
from run_qa_rule_based_filter import custom_output_jsonl_name as rule_filter_output_jsonl_name
from run_answer_relevance_vllm import read_jsonl_rows


def merged_filters_output_path(input_jsonl_path: str) -> str:
    input_path = Path(input_jsonl_path)
    return str(input_path.with_name(f"{input_path.stem}-filters.jsonl"))


def intermediate_output_paths(input_jsonl_path: str) -> list[str]:
    return [
        rule_filter_output_jsonl_name(input_jsonl_path),
        reranker_output_jsonl_name(input_jsonl_path),
        custom_bad_answer_filter_output_jsonl_name(input_jsonl_path),
    ]


def merge_custom_filter_outputs(input_jsonl_path: str, out_jsonl_path: str | None = None) -> str:
    base_path = Path(input_jsonl_path)
    output_path = out_jsonl_path or merged_filters_output_path(input_jsonl_path)
    extra_rows_by_file: list[tuple[str, list[dict[str, Any]]]] = []
    for path in intermediate_output_paths(input_jsonl_path):
        if Path(path).exists():
            extra_rows_by_file.append((path, read_jsonl_rows(path)))

    if not extra_rows_by_file:
        raise RuntimeError(f"No intermediate filter outputs found next to {base_path}")

    merged_rows, _ = merge_rows_by_id(read_jsonl_rows(str(base_path)), extra_rows_by_file)
    write_jsonl(output_path, merged_rows)
    return output_path


def cleanup_intermediate_outputs(input_jsonl_path: str) -> None:
    for path in intermediate_output_paths(input_jsonl_path):
        if Path(path).exists():
            os.remove(path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run rule-based filter, reranker, and bad-answer filter on a custom JSONL file, then merge outputs."
    )
    p.add_argument("--input-jsonl-path", required=True, help="Path to a custom JSONL file.")
    p.add_argument(
        "--merged-out-jsonl-path",
        default=None,
        help="Optional output path for the merged final JSONL. Default: <stem>-filters.jsonl next to the input.",
    )
    p.add_argument(
        "--enable-entity-integrity",
        action="store_true",
        help="Forwarded to qa-bad-answer-filter-offline stage.",
    )
    return p.parse_args()


def run_command(command: list[str]) -> None:
    logging.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    base_command = [sys.executable]
    input_args = ["--input-jsonl-path", args.input_jsonl_path]

    run_command(base_command + ["run_qa_rule_based_filter.py", *input_args])
    run_command(base_command + ["run_answer_relevance_reranker.py", *input_args])

    bad_answer_command = [
        *base_command,
        "run_answer_relevance_vllm.py",
        "--inference-source",
        "offline",
        "--task",
        "bad_answer_filter",
        *input_args,
    ]
    if args.enable_entity_integrity:
        bad_answer_command.append("--enable-entity-integrity")
    run_command(bad_answer_command)

    merged_path = merge_custom_filter_outputs(args.input_jsonl_path, args.merged_out_jsonl_path)
    cleanup_intermediate_outputs(args.input_jsonl_path)
    logging.info("Merged filter output written to %s", merged_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
