#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import copy
import json
import logging
import os
import random
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from translation_core import RateLimitReached, append_jsonl, load_done_ids_from_jsonl

ANSWER_RELEVANCE_LABELS = (
    "answers_question",
    "not_answering_question",
    "unclear",
)

ANSWER_RELEVANCE_SYSTEM_PROMPT = """Jesteś klasyfikatorem dopasowania pytanie–odpowiedź dla danych QA.

Masz ocenić, czy tekst z pola answer jest sensownym kandydatem odpowiedzi lub poprawnym kontekstem dla tekstu z pola question.
Nie oceniaj prawdziwości faktograficznej.
Nie próbuj ustalać, jaki jest faktyczny bieżący rok, dzień, stan świata lub aktualna sytuacja.
Nie korzystaj z wiedzy zewnętrznej poza tym, co wynika z samego pytania i odpowiedzi.

Oceniaj przede wszystkim:
- zgodność tematu;
- zgodność typu oczekiwanej informacji;
- zgodność głównej encji i relacji;
- czy odpowiedź nie pochodzi z wyraźnie innego kontekstu.

Ważne:
- Krótkie odpowiedzi, pojedyncze nazwy, daty, liczby i frazy nominalne mogą być poprawne.
- Odpowiedź nie musi być idealną odpowiedzią końcową; może być także poprawnym i użytecznym kontekstem.
- Różnice w poziomie szczegółowości nie oznaczają błędu.
- Wyrażenia względne, takie jak „w tym roku”, „obecnie”, „teraz”, „dzisiaj”, nie powinny same w sobie powodować etykiety negatywnej.
- Jeśli odpowiedź dotyczy właściwego tematu i relacji, ale może być nie do końca literalnie dopasowana, preferuj `answers_question`.
- `not_answering_question` wybieraj tylko wtedy, gdy odpowiedź jest wyraźnie z innego tematu, ma zły typ informacji albo myli główną encję/relację.
- `unclear` używaj rzadko.

Etykiety:
- answers_question: odpowiedź jest sensownym kandydatem odpowiedzi lub poprawnym kontekstem dla pytania;
- not_answering_question: odpowiedź wyraźnie nie pasuje do pytania;
- unclear: nie da się wiarygodnie rozstrzygnąć na podstawie samego tekstu.

Zwracaj wyłącznie JSON zgodny ze schematem.
Pole explanation musi być pierwsze, a pole label drugie."""


@dataclass(frozen=True)
class DatasetSpec:
    question_field: str
    answer_field: str


@dataclass
class RowResult:
    rid: str
    out_row: dict[str, Any]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "nq_qa": DatasetSpec(question_field="question", answer_field="answer"),
    "hotpotqa": DatasetSpec(question_field="anchor", answer_field="positive"),
}


def runtime_dependencies() -> dict[str, Any]:
    from run_translation_vllm import (
        OfflineVllmClient,
        format_seconds,
        llm_call_json_async,
        quiet_external_loggers,
        resolve_api_connection,
    )

    return {
        "OfflineVllmClient": OfflineVllmClient,
        "format_seconds": format_seconds,
        "llm_call_json_async": llm_call_json_async,
        "quiet_external_loggers": quiet_external_loggers,
        "resolve_api_connection": resolve_api_connection,
    }


def build_answer_relevance_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "explanation": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
            "label": {
                "type": "string",
                "enum": list(ANSWER_RELEVANCE_LABELS),
            },
        },
        "required": ["explanation", "label"],
        "additionalProperties": False,
    }


def build_answer_relevance_prompt(question: str, answer: str) -> str:
    return (
        "Oceń, czy `answer` jest sensownym kandydatem odpowiedzi lub poprawnym kontekstem dla `question`.\n"
        "Nie oceniaj prawdziwości faktograficznej ani aktualności względem świata.\n"
        "Sprawdź zgodność tematu, typu informacji, głównej encji i relacji.\n"
        "Nie odrzucaj odpowiedzi tylko dlatego, że pytanie zawiera wyrażenia względne, takie jak `w tym roku`, `obecnie`, `teraz`.\n"
        "Jeśli odpowiedź jest semantycznie bliska i użyteczna jako para treningowa, wybierz `answers_question`.\n"
        "Wybierz `not_answering_question` tylko wtedy, gdy odpowiedź jest wyraźnie z innego kontekstu albo ma zły typ informacji.\n"
        "Użyj `unclear` rzadko.\n\n"
        f"question: {json.dumps(question, ensure_ascii=False)}\n"
        f"answer: {json.dumps(answer, ensure_ascii=False)}"
    )


def selected_dataset_keys(datasets_arg: list[str]) -> list[str]:
    out: list[str] = []
    for key in datasets_arg:
        expanded = list(DATASET_SPECS.keys()) if key == "all" else [key]
        for item in expanded:
            if item not in out:
                out.append(item)
    return out


def read_jsonl_rows(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        raise RuntimeError(f"Input JSONL not found: {path}")

    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL in {path} at line {line_no}: {exc}") from exc
    return rows


def resolve_row_id(row: dict[str, Any], dataset_key: str, row_idx: int) -> str:
    rid = str(row.get("id") or "").strip()
    if rid:
        return rid
    return f"{dataset_key}_{row_idx}"


def extract_question_answer(row: dict[str, Any], dataset_key: str) -> tuple[str, str]:
    spec = DATASET_SPECS[dataset_key]
    question = str(row.get(spec.question_field) or "").strip()
    answer = str(row.get(spec.answer_field) or "").strip()
    if not question:
        raise RuntimeError(f"Row is missing non-empty '{spec.question_field}'")
    if not answer:
        raise RuntimeError(f"Row is missing non-empty '{spec.answer_field}'")
    return question, answer


def build_output_row(
    row: dict[str, Any],
    explanation: str,
    label: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_row = dict(row)
    out_row["answer_relevance"] = {
        "explanation": explanation,
        "label": label,
    }
    out_row["answer_relevance_model"] = args.model
    out_row["answer_relevance_source"] = args.inference_source
    out_row["answer_relevance_key_last6"] = args._api_key_last6
    out_row["answer_relevance_base_url"] = args.base_url or None
    out_row["answer_relevance_timestamp_unix"] = int(time.time())
    return out_row


async def process_row(
    row: dict[str, Any],
    row_idx: int,
    dataset_key: str,
    args: argparse.Namespace,
    client: Any,
) -> RowResult:
    deps = runtime_dependencies()
    rid = resolve_row_id(row, dataset_key, row_idx)
    question, answer = extract_question_answer(row, dataset_key)
    result_obj = await deps["llm_call_json_async"](
        client=client,
        model=args.model,
        system_prompt=ANSWER_RELEVANCE_SYSTEM_PROMPT,
        user_prompt=build_answer_relevance_prompt(question=question, answer=answer),
        temperature=args.temperature,
        max_retries=args.max_retries,
        delay_seconds=args.delay_seconds,
        response_schema=build_answer_relevance_schema(),
    )
    explanation = str(result_obj.get("explanation") or "").strip()
    label = str(result_obj.get("label") or "").strip()
    if not explanation:
        raise RuntimeError("Empty explanation from model")
    if label not in ANSWER_RELEVANCE_LABELS:
        raise RuntimeError(f"Unsupported label from model: {label!r}")
    return RowResult(rid=rid, out_row=build_output_row(row, explanation, label, args))


async def writer_loop(
    q: asyncio.Queue[RowResult | None],
    out_jsonl: str,
    done_ids: set[str],
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
            finally:
                q.task_done()
    except BaseException as exc:  # noqa: BLE001
        logging.exception("Writer loop failed")
        write_errors.append(exc)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Score whether translated QA answers match translated questions using an OpenAI-compatible API "
            "(local vLLM server or external provider) or vLLM offline inference."
        )
    )
    p.add_argument(
        "--inference-source",
        default=os.getenv("INFERENCE_SOURCE", "offline"),
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
        default=int(os.getenv("OFFLINE_MAX_OUTPUT_TOKENS", "512")),
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
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop whole run on first row-level scoring error.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", *DATASET_SPECS.keys()],
        help="Dataset selection. 'all' expands to nq_qa and hotpotqa.",
    )
    p.add_argument("--out-dir", default="out_pl")
    p.add_argument("--input-jsonl-name", default="translated.jsonl")
    p.add_argument("--out-jsonl-name", default="answer_relevance.jsonl")
    p.add_argument("--failed-jsonl-name", default="answer_relevance_failed_rows.jsonl")
    p.add_argument(
        "--retry-failed-rows",
        action="store_true",
        help="Include rows previously present in failed_rows JSONL when resuming.",
    )
    p.add_argument("--max-rows", type=int, default=0, help="0 = all")
    p.add_argument("--skip-rows", type=int, default=0)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-every", type=int, default=10, help="Log progress every N completed rows in non-TTY mode")
    p.add_argument(
        "--progress-bar",
        default=os.getenv("PROGRESS_BAR", "on"),
        choices=["auto", "on", "off"],
        help="Progress bar mode: auto=TTY only, on=always, off=disable tqdm",
    )
    return p.parse_args()


async def run_single_dataset_async(args: argparse.Namespace) -> int:
    from tqdm import tqdm

    deps = runtime_dependencies()
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

    effective_parallel = max(1, int(args.parallel_requests))
    if args.inference_source == "offline":
        effective_parallel = max(effective_parallel, int(args.offline_micro_batch_size))

    logging.info(
        "Answer relevance run: source=%s dataset_key=%s input=%s output=%s model=%s parallel=%d offline_micro_batch=%d "
        "range=%d..%d total_in_range=%d pending=%d done_before=%d skipped_failed=%d retry_failed_rows=%s",
        args.inference_source,
        args.dataset_key,
        input_jsonl,
        out_jsonl,
        args.model,
        effective_parallel,
        int(args.offline_micro_batch_size),
        skip,
        end_idx - 1,
        end_idx - skip,
        len(candidates),
        (end_idx - skip) - len(candidates),
        skipped_failed,
        bool(args.retry_failed_rows),
    )

    result_queue: asyncio.Queue[RowResult | None] = asyncio.Queue(
        maxsize=max(4, int(args.parallel_requests) * 2)
    )
    write_errors: list[BaseException] = []

    writer = asyncio.create_task(writer_loop(result_queue, out_jsonl, done_ids, write_errors))
    logging.info("Writer task started. Output: %s", out_jsonl)
    logging.info("Failed rows will be appended to: %s", failed_jsonl)

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
    log_every = max(1, int(args.log_every))
    sem = asyncio.Semaphore(effective_parallel)

    @asynccontextmanager
    async def build_inference_client():
        if args.inference_source == "offline":
            offline_client = deps["OfflineVllmClient"](args)
            try:
                yield offline_client
            finally:
                await offline_client.aclose()
            return

        from openai import AsyncOpenAI

        async with AsyncOpenAI(api_key=args.api_key, base_url=args.base_url) as api_client:
            yield api_client

    async with build_inference_client() as client:
        async def process_with_limit(row_idx: int, row: dict[str, Any]) -> RowResult:
            async with sem:
                return await process_row(row, row_idx, args.dataset_key, args, client)

        tasks = [asyncio.create_task(process_with_limit(row_idx, row)) for row_idx, row in candidates]
        task_meta: dict[asyncio.Task[RowResult], tuple[int, str]] = {
            task: (row_idx, resolve_row_id(row, args.dataset_key, row_idx))
            for task, (row_idx, row) in zip(tasks, candidates)
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
                        row_idx, rid = task_meta.get(fut, (-1, "unknown"))
                        failed_rows += 1
                        if args.fail_fast:
                            raise

                        logging.exception("Row failed (row_idx=%s id=%s): %s", row_idx, rid, exc)
                        failed_obj = {
                            "id": rid,
                            "source_dataset": args.dataset_key,
                            "row_idx": row_idx,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "timestamp_unix": int(time.time()),
                        }
                        await asyncio.to_thread(append_jsonl, failed_jsonl, failed_obj)
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
                        logging.info(
                            "Progress: %d/%d rows (%.1f%%), rate=%.2f row/s, elapsed=%s, eta=%s",
                            completed,
                            len(candidates),
                            100.0 * completed / len(candidates),
                            rate,
                            deps["format_seconds"](elapsed),
                            deps["format_seconds"](eta_seconds),
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
    deps = runtime_dependencies()
    args.base_url, args.api_key = deps["resolve_api_connection"](args)
    args._api_key_last6 = "OFFLINE" if args.inference_source == "offline" else (args.api_key[-6:] if args.api_key else "EMPTY")

    selected_keys = selected_dataset_keys(args.datasets)
    logging.info("Selected dataset runs: %s", ", ".join(selected_keys))

    for dataset_key in selected_keys:
        run_args = copy.deepcopy(args)
        run_args.dataset_key = dataset_key
        logging.info(
            "Starting answer relevance run: key=%s out_dir=%s input_jsonl=%s",
            dataset_key,
            os.path.join(args.out_dir, dataset_key),
            args.input_jsonl_name,
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
    runtime_dependencies()["quiet_external_loggers"]()
    return asyncio.run(run_async(args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted (CTRL+C).", file=sys.stderr)
        raise SystemExit(130)
