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
from typing import Any, Callable

from translation_core import RateLimitReached, append_jsonl, load_done_ids_from_jsonl

ANSWER_RELEVANCE_LABELS = (
    "answers_question",
    "not_answering_question",
    "unclear",
)

SCORE_RANGE = (1, 2, 3, 4, 5, 6)

BAD_ANSWER_FILTER_SYSTEM_PROMPT = (
    "Jestes modulem oceny jakosci tekstu. "
    "Zwracaj wylacznie JSON zgodny ze schematem odpowiedzi."
)

ANSWER_RELEVANCE_SYSTEM_PROMPT = """Jestes klasyfikatorem dopasowania pytanie-odpowiedz dla danych QA.

Masz ocenic, czy tekst z pola answer jest sensownym kandydatem odpowiedzi lub poprawnym kontekstem dla tekstu z pola question.
Nie oceniaj prawdziwosci faktograficznej.
Nie probuj ustalac, jaki jest faktyczny biezacy rok, dzien, stan swiata lub aktualna sytuacja.
Nie korzystaj z wiedzy zewnetrznej poza tym, co wynika z samego pytania i odpowiedzi.

Oceniaj przede wszystkim:
- zgodnosc tematu;
- zgodnosc typu oczekiwanej informacji;
- zgodnosc glownej encji i relacji;
- czy odpowiedz nie pochodzi z wyraznie innego kontekstu.

Wazne:
- Krotkie odpowiedzi, pojedyncze nazwy, daty, liczby i frazy nominalne moga byc poprawne.
- Odpowiedz nie musi byc idealna odpowiedzia koncowa; moze byc takze poprawnym i uzytecznym kontekstem.
- Roznice w poziomie szczegolowosci nie oznaczaja bledu.
- Wyrazenia wzgledne, takie jak "w tym roku", "obecnie", "teraz", "dzisiaj", nie powinny same w sobie powodowac etykiety negatywnej.
- Jesli odpowiedz dotyczy wlasciwego tematu i relacji, ale moze byc nie do konca literalnie dopasowana, preferuj `answers_question`.
- `not_answering_question` wybieraj tylko wtedy, gdy odpowiedz jest wyraznie z innego tematu, ma zly typ informacji albo myli glowna encje/relacje.
- `unclear` uzywaj rzadko.

Etykiety:
- answers_question: odpowiedz jest sensownym kandydatem odpowiedzi lub poprawnym kontekstem dla pytania;
- not_answering_question: odpowiedz wyraznie nie pasuje do pytania;
- unclear: nie da sie wiarygodnie rozstrzygnac na podstawie samego tekstu.

Zwracaj wylacznie JSON zgodny ze schematem.
Pole explanation musi byc pierwsze, a pole label drugie."""


@dataclass(frozen=True)
class DatasetSpec:
    question_field: str
    answer_field: str


@dataclass
class RowResult:
    rid: str
    out_row: dict[str, Any]


@dataclass(frozen=True)
class BadAnswerFilterStage:
    key: str
    output_key: str
    build_prompt: Callable[[str, str], str]
    build_schema: Callable[[], dict[str, Any]]


DATASET_SPECS: dict[str, DatasetSpec] = {
    "nq_qa": DatasetSpec(question_field="question", answer_field="answer"),
    "hotpotqa": DatasetSpec(question_field="anchor", answer_field="positive"),
}


def runtime_dependencies() -> dict[str, Any]:
    from run_translation import (
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


def build_bad_answer_filter_naturalness_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
            "score": {
                "type": "integer",
                "enum": list(SCORE_RANGE),
            },
        },
        "required": ["reason", "score"],
        "additionalProperties": False,
    }


def build_bad_answer_filter_entity_integrity_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
            "suspicious_items": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 120,
                },
                "maxItems": 12,
            },
            "score": {
                "type": "integer",
                "enum": list(SCORE_RANGE),
            },
        },
        "required": ["reason", "suspicious_items", "score"],
        "additionalProperties": False,
    }


def build_bad_answer_filter_semantic_coherence_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
            "problem_fragments": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 160,
                },
                "maxItems": 12,
            },
            "score": {
                "type": "integer",
                "enum": list(SCORE_RANGE),
            },
        },
        "required": ["reason", "problem_fragments", "score"],
        "additionalProperties": False,
    }


def build_bad_answer_filter_meaning_drift_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
            },
            "shared_meaning_elements": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 120,
                },
                "maxItems": 12,
            },
            "score": {
                "type": "integer",
                "enum": list(SCORE_RANGE),
            },
        },
        "required": ["reason", "shared_meaning_elements", "score"],
        "additionalProperties": False,
    }


def build_bad_answer_filter_naturalness_prompt(text: str) -> str:
    return (
        f"""Oceń naturalność języka polskiego w poniższym tekście.
Zignoruj prawdziwość informacji i fakty. Oceniaj wyłącznie poprawność i płynność językową.

Skala ocen (1-6):
1 - Tekst całkowicie niezrozumiały, połamany językowo.
2 - Bardzo nienaturalny, wygląda jak surowe tłumaczenie maszynowe.
3 - Wyraźnie nienaturalny (kalki językowe), ale w miarę zrozumiały.
4 - Zrozumiały, choć zawiera drobne błędy lub sztuczne sformułowania.
5 - W większości naturalny, sporadyczne potknięcia.
6 - Całkowicie płynny i naturalny język polski.

Zwróć odpowiedź w formacie JSON (bez dodatkowego tekstu):
{{
  "reason": "Krótkie uzasadnienie oceny",
  "score": <liczba 1-6>
}}

TEKST:
{json.dumps(text, ensure_ascii=False)}"""
    )


def build_bad_answer_filter_entity_integrity_prompt(text: str) -> str:
    return (
        f"""Oceń poprawność zapisu nazw własnych, skrótów, symboli i terminów technicznych w tekście.
Szukaj błędów wynikających ze złego tłumaczenia lub formatowania (np. przetłumaczona nazwa własna, skrót zamieniony w zwykłe słowo, uszkodzony wzór). Zignoruj ogólną jakość języka.

Skala ocen (1-6):
1 - Krytyczne i liczne zniekształcenia najważniejszych terminów/nazw.
2 - Wyraźne błędy w kluczowych terminach, mocno utrudniające czytanie.
3 - Zauważalne, podejrzane elementy, uszkodzone pojedyncze nazwy.
4 - Drobne wątpliwości lub nieścisłości, ale bez silnego wpływu na tekst.
5 - Niemal brak problemów z terminologią i nazwami.
6 - Brak jakichkolwiek zniekształceń, idealna terminologia.

Zwróć odpowiedź w formacie JSON:
{{
  "reason": "Krótkie uzasadnienie",
  "suspicious_items": ["element1", "element2"],
  "score": <liczba 1-6>
}}

TEKST:
{json.dumps(text, ensure_ascii=False)}"""
    )


def build_bad_answer_filter_semantic_coherence_prompt(text: str) -> str:
    return (
        f"""Oceń wewnętrzną spójność i logikę poniższego tekstu.
Sprawdź, czy tekst ma sens i nie zaprzecza sam sobie. Nie oceniaj zgodności z zewnętrznymi faktami.

Skala ocen (1-6):
1 - Tekst pozbawiony sensu, rozpadnięty, pełen sprzeczności.
2 - Bardzo niespójny, zdania nie łączą się logicznie.
3 - Wyraźne uszkodzenia sensu i luki logiczne w wielu miejscach.
4 - Ogólnie zrozumiały, ale zawiera podejrzane lub dziwne fragmenty.
5 - Spójny tekst, co najwyżej drobne, pomijalne potknięcia.
6 - W pełni spójny, logiczny i konsekwentny tekst.

Zwróć odpowiedź w formacie JSON:
{{
  "reason": "Krótkie uzasadnienie",
  "problem_fragments": ["fragment 1", "fragment 2"],
  "score": <liczba 1-6>
}}

TEKST:
{json.dumps(text, ensure_ascii=False)}"""
    )


def build_bad_answer_filter_meaning_drift_prompt(anchor: str, answer: str) -> str:
    return (
        f"""Oceń powiązanie znaczeniowe (semantyczne) między DOKUMENT_A a DOKUMENT_B.
Zbadaj, czy tematyka i sens obu tekstów są zgodne. DOKUMENT_B nie musi być pełną odpowiedzią na DOKUMENT_A. Oceniaj wyłącznie, czy doszło do "dryfu" (zmiany tematu, odchylenia znaczenia).

Skala ocen (1-6):
1 - Zupełnie inne tematy, całkowity rozpad powiązania (silny dryf).
2 - Słaby związek, tekst B mocno "odpływa" od sensu tekstu A.
3 - Zauważalna zmiana znaczenia, temat zachowany tylko częściowo.
4 - Luźny, ale zachowany związek znaczeniowy (niewielki dryf).
5 - Dobra zgodność głównych myśli i tematów.
6 - Idealna zgodność znaczeniowa, brak dryfu.

Zwróć odpowiedź w formacie JSON:
{{
  "reason": "Krótkie uzasadnienie",
  "shared_meaning_elements": ["element1", "element2"],
  "score": <liczba 1-6>
}}

DOKUMENT_A:
{json.dumps(anchor, ensure_ascii=False)}

DOKUMENT_B:
{json.dumps(answer, ensure_ascii=False)}"""
    )


BAD_ANSWER_FILTER_STAGES: tuple[BadAnswerFilterStage, ...] = (
    BadAnswerFilterStage(
        key="question_language_naturalness",
        output_key="question_language_naturalness",
        build_prompt=lambda question, answer: build_bad_answer_filter_naturalness_prompt(question),
        build_schema=lambda: build_bad_answer_filter_naturalness_schema(),
    ),
    BadAnswerFilterStage(
        key="answer_language_naturalness",
        output_key="answer_language_naturalness",
        build_prompt=lambda question, answer: build_bad_answer_filter_naturalness_prompt(answer),
        build_schema=lambda: build_bad_answer_filter_naturalness_schema(),
    ),
    BadAnswerFilterStage(
        key="answer_entity_integrity",
        output_key="answer_entity_integrity",
        build_prompt=lambda question, answer: build_bad_answer_filter_entity_integrity_prompt(answer),
        build_schema=lambda: build_bad_answer_filter_entity_integrity_schema(),
    ),
    BadAnswerFilterStage(
        key="answer_semantic_coherence",
        output_key="answer_semantic_coherence",
        build_prompt=lambda question, answer: build_bad_answer_filter_semantic_coherence_prompt(answer),
        build_schema=lambda: build_bad_answer_filter_semantic_coherence_schema(),
    ),
    BadAnswerFilterStage(
        key="question_answer_meaning_drift",
        output_key="question_answer_meaning_drift",
        build_prompt=lambda question, answer: build_bad_answer_filter_meaning_drift_prompt(question, answer),
        build_schema=lambda: build_bad_answer_filter_meaning_drift_schema(),
    ),
)


def build_bad_answer_filter_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "question_language_naturalness": build_bad_answer_filter_naturalness_schema(),
            "answer_language_naturalness": build_bad_answer_filter_naturalness_schema(),
            "answer_entity_integrity": build_bad_answer_filter_entity_integrity_schema(),
            "answer_semantic_coherence": build_bad_answer_filter_semantic_coherence_schema(),
            "question_answer_meaning_drift": build_bad_answer_filter_meaning_drift_schema(),
        },
        "required": [stage.output_key for stage in BAD_ANSWER_FILTER_STAGES],
        "additionalProperties": False,
    }


def build_answer_relevance_prompt(question: str, answer: str) -> str:
    return (
        "Ocen, czy `answer` jest sensownym kandydatem odpowiedzi lub poprawnym kontekstem dla `question`.\n"
        "Nie oceniaj prawdziwosci faktograficznej ani aktualnosci wzgledem swiata.\n"
        "Sprawdz zgodnosc tematu, typu informacji, glownej encji i relacji.\n"
        "Nie odrzucaj odpowiedzi tylko dlatego, ze pytanie zawiera wyrazenia wzgledne, takie jak `w tym roku`, `obecnie`, `teraz`.\n"
        "Jesli odpowiedz jest semantycznie bliska i uzyteczna jako para treningowa, wybierz `answers_question`.\n"
        "Wybierz `not_answering_question` tylko wtedy, gdy odpowiedz jest wyraznie z innego kontekstu albo ma zly typ informacji.\n"
        "Uzyj `unclear` rzadko.\n\n"
        f"question: {json.dumps(question, ensure_ascii=False)}\n"
        f"answer: {json.dumps(answer, ensure_ascii=False)}"
    )


def task_output_jsonl_name(task: str) -> str:
    return "bad_answer_filter_evaluations.jsonl" if task == "bad_answer_filter" else "answer_relevance.jsonl"


def task_failed_jsonl_name(task: str) -> str:
    return "bad_answer_filter_evaluations_failed_rows.jsonl" if task == "bad_answer_filter" else "answer_relevance_failed_rows.jsonl"


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


def read_jsonl_by_id(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl_rows(path):
        rid = str(row.get("id") or "").strip()
        if rid:
            out[rid] = row
    return out


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


def bad_answer_filter_stage_jsonl_name(stage_key: str) -> str:
    return f"bad_answer_filter_evaluations.{stage_key}.jsonl"


def validate_score_result(result_obj: dict[str, Any], list_field: str | None = None) -> dict[str, Any]:
    reason = str(result_obj.get("reason") or "").strip()
    if not reason:
        raise RuntimeError("Empty reason from model")

    score = result_obj.get("score")
    if score not in SCORE_RANGE:
        raise RuntimeError(f"Unsupported score from model: {score!r}")

    if list_field is not None:
        items = result_obj.get(list_field)
        if not isinstance(items, list):
            raise RuntimeError(f"Field {list_field!r} is not a list: {type(items)}")
        result_obj[list_field] = [str(item).strip() for item in items]

    result_obj["reason"] = reason
    return result_obj


def build_output_row(
    row: dict[str, Any],
    label: str,
    result_obj: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_row = dict(row)

    if args.task == "bad_answer_filter":
        prefix = "bad_answer_filter"
        out_row[prefix] = {
            stage.output_key: result_obj[stage.output_key]
            for stage in BAD_ANSWER_FILTER_STAGES
        }
    else:
        prefix = "answer_relevance"
        out_row[prefix] = {
            "explanation": result_obj.get("explanation") or result_obj.get("reason"),
            "label": label,
        }

    out_row[f"{prefix}_model"] = args.model
    out_row[f"{prefix}_source"] = args.inference_source
    out_row[f"{prefix}_key_last6"] = args._api_key_last6
    out_row[f"{prefix}_base_url"] = args.base_url or None
    out_row[f"{prefix}_timestamp_unix"] = int(time.time())
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
    label = str(result_obj.get("label") or "").strip()
    reason_text = str(result_obj.get("explanation") or "").strip()
    if not reason_text:
        raise RuntimeError("Empty explanation from model")
    if label not in ANSWER_RELEVANCE_LABELS:
        raise RuntimeError(f"Unsupported label from model: {label!r}")
    return RowResult(rid=rid, out_row=build_output_row(row, label, result_obj, args))


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


@asynccontextmanager
async def build_inference_client(args: argparse.Namespace):
    deps = runtime_dependencies()
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


async def score_bad_answer_filter_stage_row(
    row: dict[str, Any],
    row_idx: int,
    dataset_key: str,
    args: argparse.Namespace,
    client: Any,
    stage: BadAnswerFilterStage,
) -> RowResult:
    deps = runtime_dependencies()
    rid = resolve_row_id(row, dataset_key, row_idx)
    question, answer = extract_question_answer(row, dataset_key)
    result_obj = await deps["llm_call_json_async"](
        client=client,
        model=args.model,
        system_prompt=BAD_ANSWER_FILTER_SYSTEM_PROMPT,
        user_prompt=stage.build_prompt(question, answer),
        temperature=args.temperature,
        max_retries=args.max_retries,
        delay_seconds=args.delay_seconds,
        response_schema=stage.build_schema(),
    )

    if stage.output_key in ("question_language_naturalness", "answer_language_naturalness"):
        validated_obj = validate_score_result(result_obj)
    elif stage.output_key == "answer_entity_integrity":
        validated_obj = validate_score_result(result_obj, "suspicious_items")
    elif stage.output_key == "answer_semantic_coherence":
        validated_obj = validate_score_result(result_obj, "problem_fragments")
    elif stage.output_key == "question_answer_meaning_drift":
        validated_obj = validate_score_result(result_obj, "shared_meaning_elements")
    else:
        raise RuntimeError(f"Unsupported bad-answer-filter stage: {stage.output_key}")

    return RowResult(
        rid=rid,
        out_row={
            "id": rid,
            "source_dataset": dataset_key,
            "row_idx": row_idx,
            stage.output_key: validated_obj,
        },
    )


def merge_bad_answer_filter_results(
    row: dict[str, Any],
    stage_outputs: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for stage in BAD_ANSWER_FILTER_STAGES:
        if stage.output_key not in stage_outputs:
            raise RuntimeError(f"Missing stage output for {stage.output_key}")
        aggregate[stage.output_key] = stage_outputs[stage.output_key]
    return build_output_row(row=row, label="", result_obj=aggregate, args=args)


async def run_bad_answer_filter_stage_async(
    args: argparse.Namespace,
    client: Any,
    stage: BadAnswerFilterStage,
    candidates: list[tuple[int, dict[str, Any]]],
    dataset_dir: str,
) -> None:
    from tqdm import tqdm

    deps = runtime_dependencies()
    stage_jsonl = os.path.join(dataset_dir, bad_answer_filter_stage_jsonl_name(stage.key))
    failed_jsonl = os.path.join(dataset_dir, f"bad_answer_filter_evaluations.{stage.key}.failed_rows.jsonl")
    done_ids = load_done_ids_from_jsonl(stage_jsonl)
    pending = [
        (row_idx, row)
        for row_idx, row in candidates
        if resolve_row_id(row, args.dataset_key, row_idx) not in done_ids
    ]
    if not pending:
        logging.info("Stage %s: nothing to score.", stage.key)
        return

    effective_parallel = max(1, int(args.parallel_requests))
    if args.inference_source == "offline":
        effective_parallel = max(effective_parallel, int(args.offline_micro_batch_size))

    result_queue: asyncio.Queue[RowResult | None] = asyncio.Queue(maxsize=max(4, int(args.parallel_requests) * 2))
    write_errors: list[BaseException] = []
    writer = asyncio.create_task(writer_loop(result_queue, stage_jsonl, done_ids, write_errors))

    is_tty = sys.stderr.isatty()
    show_pbar = args.progress_bar == "on" or (args.progress_bar == "auto" and is_tty)
    non_tty_progress = not is_tty
    pbar_rows = tqdm(
        total=len(pending),
        desc=f"Stage {stage.key}",
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

    async def process_with_limit(row_idx: int, row: dict[str, Any]) -> RowResult:
        async with sem:
            return await score_bad_answer_filter_stage_row(row, row_idx, args.dataset_key, args, client, stage)

    tasks = [asyncio.create_task(process_with_limit(row_idx, row)) for row_idx, row in pending]
    task_meta: dict[asyncio.Task[RowResult], tuple[int, str]] = {
        task: (row_idx, resolve_row_id(row, args.dataset_key, row_idx))
        for task, (row_idx, row) in zip(tasks, pending)
    }

    try:
        pending_tasks = set(tasks)
        while pending_tasks:
            if write_errors:
                raise RuntimeError(f"Writer task failed: {write_errors[0]}") from write_errors[0]
            done_now, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
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
                    if args.fail_fast:
                        raise
                    logging.exception("Stage %s row failed (row_idx=%s id=%s): %s", stage.key, row_idx, rid, exc)
                    await asyncio.to_thread(
                        append_jsonl,
                        failed_jsonl,
                        {
                            "id": rid,
                            "source_dataset": args.dataset_key,
                            "row_idx": row_idx,
                            "stage": stage.key,
                            "error": str(exc),
                            "error_type": type(exc).__name__,
                            "timestamp_unix": int(time.time()),
                        },
                    )
                    pbar_rows.update(1)
                    completed += 1
                    continue

                await result_queue.put(result)
                pbar_rows.update(1)
                completed += 1
                if (non_tty_progress and not show_pbar) and (
                    completed == 1 or completed % log_every == 0 or completed == len(pending)
                ):
                    elapsed = time.time() - started_at
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta_seconds = (len(pending) - completed) / rate if rate > 0 else 0.0
                    logging.info(
                        "Stage %s progress: %d/%d rows (%.1f%%), rate=%.2f row/s, elapsed=%s, eta=%s",
                        stage.key,
                        completed,
                        len(pending),
                        100.0 * completed / len(pending),
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


async def run_bad_answer_filter_dataset_async(args: argparse.Namespace) -> int:
    dataset_dir = os.path.join(args.out_dir, args.dataset_key)
    input_jsonl = os.path.join(dataset_dir, args.input_jsonl_name)
    out_jsonl = os.path.join(dataset_dir, args.out_jsonl_name)
    os.makedirs(dataset_dir, exist_ok=True)

    rows = read_jsonl_rows(input_jsonl)
    total = len(rows)
    skip = max(0, int(args.skip_rows))
    if skip >= total:
        print(f"--skip-rows={skip} >= dataset size={total}. Nothing to do.")
        return 0

    end_idx = min(total, skip + int(args.max_rows)) if args.max_rows and args.max_rows > 0 else total
    final_done_ids = load_done_ids_from_jsonl(out_jsonl)
    candidates: list[tuple[int, dict[str, Any]]] = []
    for row_idx in range(skip, end_idx):
        row = rows[row_idx]
        rid = resolve_row_id(row, args.dataset_key, row_idx)
        if rid in final_done_ids:
            continue
        candidates.append((row_idx, row))

    if not candidates:
        print("Nothing to score (all rows already done in selected window).")
        return 0

    async with build_inference_client(args) as client:
        for stage in BAD_ANSWER_FILTER_STAGES:
            await run_bad_answer_filter_stage_async(args, client, stage, candidates, dataset_dir)

    stage_results: dict[str, dict[str, dict[str, Any]]] = {}
    for stage in BAD_ANSWER_FILTER_STAGES:
        stage_path = os.path.join(dataset_dir, bad_answer_filter_stage_jsonl_name(stage.key))
        stage_rows = read_jsonl_by_id(stage_path)
        stage_results[stage.output_key] = {
            rid: row[stage.output_key]
            for rid, row in stage_rows.items()
            if stage.output_key in row
        }

    final_done_ids = load_done_ids_from_jsonl(out_jsonl)
    for row_idx, row in candidates:
        rid = resolve_row_id(row, args.dataset_key, row_idx)
        if rid in final_done_ids:
            continue
        merged_stage_outputs: dict[str, dict[str, Any]] = {}
        missing_stage = None
        for stage in BAD_ANSWER_FILTER_STAGES:
            stage_output = stage_results.get(stage.output_key, {}).get(rid)
            if stage_output is None:
                missing_stage = stage.output_key
                break
            merged_stage_outputs[stage.output_key] = stage_output
        if missing_stage is not None:
            logging.warning("Skipping final merge for id=%s because stage %s is missing.", rid, missing_stage)
            continue
        append_jsonl(out_jsonl, merge_bad_answer_filter_results(row=row, stage_outputs=merged_stage_outputs, args=args))
        final_done_ids.add(rid)

    logging.info("Bad-answer evaluation merge finished. Output: %s", out_jsonl)
    return 0


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
    p.add_argument("--base-url", default=None, help="Override API base URL. If omitted, resolves from mode-specific env vars.")
    p.add_argument("--api-key", default=None, help="Override API key. If omitted, resolves from mode-specific env vars.")
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
    p.add_argument("--fail-fast", action="store_true", help="Stop whole run on first row-level scoring error.")
    p.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=["all", *DATASET_SPECS.keys()],
        help="Dataset selection. 'all' expands to nq_qa and hotpotqa.",
    )
    p.add_argument(
        "--task",
        default="answer_relevance",
        choices=["answer_relevance", "bad_answer_filter"],
        help="Scoring task: answer relevance classifier or multi-stage bad-answer evaluation.",
    )
    p.add_argument("--out-dir", default="out_pl")
    p.add_argument("--input-jsonl-name", default="translated.jsonl")
    p.add_argument("--out-jsonl-name", default=None)
    p.add_argument("--failed-jsonl-name", default=None)
    p.add_argument("--retry-failed-rows", action="store_true", help="Include rows previously present in failed_rows JSONL when resuming.")
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

    if args.task == "bad_answer_filter":
        return await run_bad_answer_filter_dataset_async(args)

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

    result_queue: asyncio.Queue[RowResult | None] = asyncio.Queue(maxsize=max(4, int(args.parallel_requests) * 2))
    write_errors: list[BaseException] = []
    writer = asyncio.create_task(writer_loop(result_queue, out_jsonl, done_ids, write_errors))

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

    async with build_inference_client(args) as client:
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
                done_now, pending_tasks = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
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
                        await asyncio.to_thread(
                            append_jsonl,
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
    if args.out_jsonl_name is None:
        args.out_jsonl_name = task_output_jsonl_name(args.task)
    if args.failed_jsonl_name is None:
        args.failed_jsonl_name = task_failed_jsonl_name(args.task)

    for dataset_key in selected_dataset_keys(args.datasets):
        run_args = copy.deepcopy(args)
        run_args.dataset_key = dataset_key
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
