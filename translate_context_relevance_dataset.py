#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from datasets import load_dataset
from openai import OpenAI
import openai


# ---------------------------
# Exceptions / rate-limit detection
# ---------------------------

class RateLimitReached(Exception):
    """Raised when provider returns status indicating quota/rate limit (typically 429)."""


def _is_rate_limited(exc: BaseException) -> bool:
    # OpenAI SDK v1:
    # - openai.RateLimitError (usually 429)
    # - openai.APIStatusError with .status_code
    if isinstance(exc, openai.RateLimitError):
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    # Some OpenAI-compatible providers may wrap errors differently
    msg = str(exc).lower()
    if "429" in msg and ("rate" in msg or "limit" in msg or "quota" in msg):
        return True
    if "rate limit" in msg or "quota" in msg:
        return True

    return False


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


# ---------------------------
# JSON fixing
# ---------------------------

def escape_control_chars_in_json_strings(s: str) -> str:
    """
    Fixes the most common LLM error: raw control characters inside JSON strings.
    Replaces e.g. a real newline character inside "..." with the \\n escape sequence.
    """
    out = []
    in_string = False
    escape = False

    for ch in s:
        c = ord(ch)

        if in_string:
            if escape:
                out.append(ch)
                escape = False
                continue

            if ch == "\\":
                out.append(ch)
                escape = True
                continue

            if ch == '"':
                out.append(ch)
                in_string = False
                continue

            if c < 0x20:
                if ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                else:
                    out.append(f"\\u{c:04x}")
                continue

            out.append(ch)
            continue

        else:
            if ch == '"':
                out.append(ch)
                in_string = True
                continue
            out.append(ch)

    return "".join(out)


def extract_first_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' found in the model response.")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start:i + 1]
                return json.loads(chunk)
    raise ValueError("Failed to extract a complete JSON object from the response.")


# ---------------------------
# Checkpoint filename safety
# ---------------------------

_WINDOWS_FORBIDDEN = re.compile(r'[<>:"/\\|?*\x00-\x1F]')


def checkpoint_stem_from_id(rid: str) -> str:
    cleaned = _WINDOWS_FORBIDDEN.sub("_", rid).strip(" .")
    cleaned = cleaned[:60] if cleaned else "id"
    h = hashlib.sha1(rid.encode("utf-8")).hexdigest()[:16]
    return f"{cleaned}__{h}"


# ---------------------------
# Simple I/O helpers
# ---------------------------

def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_done_ids_from_jsonl(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if "id" in o:
                    done.add(o["id"])
            except Exception:
                pass
    return done


# ---------------------------
# Spans utilities
# ---------------------------

def spans_to_pieces(text: str, spans: List[List[int]]) -> Tuple[List[str], List[str]]:
    if not spans:
        return [text], []

    gaps: List[str] = []
    span_texts: List[str] = []
    prev_end = 0

    for (s, e) in spans:
        s = int(s)
        e = int(e)
        if s < prev_end:
            raise ValueError(f"Spans overlap: prev_end={prev_end}, start={s}")
        if s > len(text) or e > len(text) or s < 0 or e < 0 or e < s:
            raise ValueError(f"Invalid span [{s},{e}) for text length {len(text)}")
        gaps.append(text[prev_end:s])
        span_texts.append(text[s:e])
        prev_end = e

    gaps.append(text[prev_end:])
    return gaps, span_texts


def rebuild_text_and_spans(gaps: List[str], translated_spans: List[str]) -> Tuple[str, List[List[int]]]:
    if len(gaps) != len(translated_spans) + 1:
        raise ValueError("Length mismatch: gaps must have len(spans)+1")

    out_parts: List[str] = []
    new_spans: List[List[int]] = []
    cursor = 0

    for span_pl in translated_spans:
        gap = gaps[len(new_spans)]
        out_parts.append(gap)
        cursor += len(gap)

        start = cursor
        out_parts.append(span_pl)
        cursor += len(span_pl)
        end = cursor
        new_spans.append([start, end])

    out_parts.append(gaps[-1])
    return "".join(out_parts), new_spans


# ---------------------------
# Prompts
# ---------------------------

SYSTEM_QUERY = (
    "You are an EN→PL translator. Return ONLY valid JSON. "
    "Do not add any comments or markdown. "
    "Use natural Polish, preserve proper names, numbers, dates, and meaning."
)

SYSTEM_TEXT = (
    "You are an EN→PL translator for NLP data. Return ONLY valid JSON. "
    "Do not add any comments or markdown. "
    "Do not change the number of spans: each span must have its own translated text. "
    "Preserve meaning, proper names, numbers, and quotations. "
    "In 'think_process', keep indices like 'Sentence [1]' etc. (translate only the surrounding text)."
)


def build_query_prompt(query_en: str, positive_fragments: List[str]) -> str:
    frags = "\n".join([f"[{i + 1}] {t}" for i, t in enumerate(positive_fragments)]) or "(none)"
    return (
        "Translate the query below into Polish so that it is semantically consistent and sounds natural. "
        "Also adapt the phrasing to the context using the relevant fragments.\n\n"
        "Return JSON exactly in the following format:\n"
        '{"query_pl": "..."}\n\n'
        "INPUT DATA:\n"
        f"QUERY (EN): {query_en}\n\n"
        "Relevant context fragments (EN):\n"
        f"{frags}"
    )
def build_text_prompt(
    query_en: str,
    query_pl: str,
    doc_label: int,
    spans_texts_en: List[str],
    spans_relevance: List[int],
    think_process_en: str,
) -> str:
    spans_block = []
    for i, (s, r) in enumerate(zip(spans_texts_en, spans_relevance), start=1):
        spans_block.append(f"SPAN {i} | relevance={int(r)}\n{s}")
    spans_joined = "\n\n".join(spans_block) if spans_block else "(no spans)"

    return (
        "Your task: translate into Polish the data for ONE candidate document.\n\n"
        "The text is provided as a list of SPANS (in order). "
        "Do NOT merge and do NOT split spans. Translate each span separately.\n\n"
        "Requirements:\n"
        "1) Return a 'translated_spans' list of the same length as the number of spans.\n"
        "2) Translate 'think_process' into Polish as 'translated_think_process'.\n"
        "   - If the text contains the quoted query (e.g. in quotes), replace it with the Polish query (exactly as above).\n"
        "   - Preserve indices like 'Sentence [1]' (do not change numbers or brackets).\n\n"
        "Return JSON exactly in the following format:\n"
        '{"translated_spans": ["...", "..."], "translated_think_process": "..."}\n\n'
        "INPUT DATA:\n"
        f"QUERY (PL) - use this wording consistently: {query_pl}\n"
        f"QUERY (EN) - for reference only: {query_en}\n\n"
        f"DOCUMENT LABEL (labels): {int(doc_label)}  (1 = relevant, 0 = not relevant)\n\n"
        "SPANS (EN):\n"
        f"{spans_joined}\n\n"
        "THINK_PROCESS (EN) associated with this document:\n"
        f"{think_process_en}"
    )
def build_text_prompt_strict(
    query_en: str,
    query_pl: str,
    doc_label: int,
    spans_texts_en: List[str],
    spans_relevance: List[int],
    think_process_en: str,
) -> str:
    n = len(spans_texts_en)
    spans_block = []
    for idx, (s, r) in enumerate(zip(spans_texts_en, spans_relevance), start=1):
        spans_block.append(f"SPAN {idx}/{n} | relevance={int(r)}\n{s}")
    spans_joined = "\n\n".join(spans_block) if spans_block else "(no spans)"

    return (
        "EN->PL TRANSLATION. This is a structural task.\n"
        "You must return ONLY valid JSON.\n"
        "KEY CONSTRAINT: the number of elements in 'translated_spans' MUST be exactly N (provided in INPUT DATA).\n"
        "You must not remove, merge, or split spans. One EN span -> one PL span.\n\n"
        "Return JSON exactly in the following format:\n"
        '{"translated_spans": ["SPAN1_PL", "SPAN2_PL", "..."], "translated_think_process": "..."}\n\n'
        "Note: if you want to use a newline inside a JSON string, use the \\n escape sequence (do not insert a raw newline).\n\n"
        "INPUT DATA:\n"
        f"N_SPANS: {n}\n"
        f"QUERY (PL): {query_pl}\n"
        f"QUERY (EN): {query_en}\n"
        f"DOCUMENT LABEL: {int(doc_label)}\n\n"
        "SPANS (EN) (numbered):\n"
        f"{spans_joined}\n\n"
        "THINK_PROCESS (EN):\n"
        f"{think_process_en}"
    )
def build_text_prompt_dictforced(
    query_en: str,
    query_pl: str,
    doc_label: int,
    spans_texts_en: List[str],
    spans_relevance: List[int],
    think_process_en: str,
) -> str:
    n = len(spans_texts_en)
    spans_block = []
    for idx, (s, r) in enumerate(zip(spans_texts_en, spans_relevance), start=1):
        spans_block.append(f"SPAN {idx}/{n} | relevance={int(r)}\n{s}")
    spans_joined = "\n\n".join(spans_block) if spans_block else "(no spans)"

    return (
        "EN->PL TRANSLATION. You must return ONLY valid JSON.\n"
        "You must return exactly N spans (provided in INPUT DATA).\n"
        "Return them as a 'translated_spans_dict' object with string keys from \"1\" to \"N\".\n"
        "You MUST NOT skip any number.\n\n"
        "Return JSON exactly in the following format:\n"
        "{"
        '"translated_spans_dict": {"1": "...", "2": "...", "3": "..."}'
        ', "translated_think_process": "..."'
        "}\n\n"
        "Note: use \\n in JSON strings instead of raw newlines.\n\n"
        "INPUT DATA:\n"
        f"N_SPANS: {n}\n"
        f"QUERY (PL): {query_pl}\n"
        f"QUERY (EN): {query_en}\n"
        f"DOCUMENT LABEL: {int(doc_label)}\n\n"
        "SPANS (EN):\n"
        f"{spans_joined}\n\n"
        "THINK_PROCESS (EN):\n"
        f"{think_process_en}"
    )
# ---------------------------
# LLM call wrapper (key/model aware)
# ---------------------------

def llm_call_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_retries: int,
    delay_seconds: float,
) -> Dict[str, Any]:
    last_err: Optional[BaseException] = None

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

            # If supported: JSON mode
            try:
                kwargs["response_format"] = {"type": "json_object"}
            except Exception:
                pass

            resp = client.chat.completions.create(**kwargs)
            content = resp.choices[0].message.content or ""

            # Parse JSON (with repair)
            try:
                obj = extract_first_json_object(content)
            except json.JSONDecodeError:
                fixed = escape_control_chars_in_json_strings(content)
                obj = extract_first_json_object(fixed)

            if delay_seconds and delay_seconds > 0:
                time.sleep(float(delay_seconds))

            return obj

        except Exception as e:
            last_err = e

            # IMPORTANT: if this is rate-limit/quota, stop for this (key, model) combo
            if _is_rate_limited(e):
                raise RateLimitReached(str(e)) from e

            # otherwise retry with backoff
            time.sleep(min(60, (2 ** attempt) + random.random()))

    raise RuntimeError(f"LLM call failed after retries: {last_err}") from last_err


def translate_text_with_span_repair(
    client: OpenAI,
    model: str,
    query_en: str,
    query_pl: str,
    doc_label: int,
    span_texts_en: List[str],
    spans_rel_i: List[int],
    think_process_en: str,
    *,
    delay_seconds: float,
    temperature: float,
    max_retries: int,
    max_attempts: int = 3,
) -> Tuple[List[str], str]:
    n = len(span_texts_en)

    prompts = [
        build_text_prompt(query_en, query_pl, doc_label, span_texts_en, spans_rel_i, think_process_en),
        build_text_prompt_strict(query_en, query_pl, doc_label, span_texts_en, spans_rel_i, think_process_en),
        build_text_prompt_dictforced(query_en, query_pl, doc_label, span_texts_en, spans_rel_i, think_process_en),
    ][:max_attempts]

    last_problem = None

    for attempt_idx, prompt in enumerate(prompts, start=1):
        t_json = llm_call_json(
            client=client,
            model=model,
            system_prompt=SYSTEM_TEXT,
            user_prompt=prompt,
            temperature=temperature,
            max_retries=max_retries,
            delay_seconds=delay_seconds,
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
        f"Failed to obtain the correct number of spans after {len(prompts)} attempts. Last issue: {last_problem}"
    )


# ---------------------------
# Key/Model combo manager
# ---------------------------

@dataclass
class Combo:
    api_key: str
    model: str


class ComboRunner:
    def __init__(self, api_keys: List[str], models: List[str], base_url: Optional[str]):
        if not api_keys:
            raise ValueError("You must provide at least one api_key.")
        if not models:
            raise ValueError("You must provide at least one model.")
        self.base_url = base_url
        self.combos: List[Combo] = [Combo(k, m) for k in api_keys for m in models]
        self.idx = 0
        self.exhausted: List[Combo] = []
        self.active_client: Optional[OpenAI] = None
        self.active_combo: Optional[Combo] = None

    def has_next(self) -> bool:
        return self.idx < len(self.combos)

    def next_combo(self) -> Tuple[Combo, OpenAI]:
        if not self.has_next():
            raise StopIteration("No more api_key × model combinations.")
        combo = self.combos[self.idx]
        self.idx += 1
        client = OpenAI(api_key=combo.api_key, base_url=self.base_url or None)
        self.active_client = client
        self.active_combo = combo
        return combo, client

    def mark_exhausted(self, combo: Combo) -> None:
        self.exhausted.append(combo)

    def remaining_count(self) -> int:
        return len(self.combos) - self.idx


# ---------------------------
# Mode: TRANSLATE (rotate combos on 429)
# ---------------------------

def run_translate(args: argparse.Namespace, runner: ComboRunner) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    out_jsonl = os.path.join(args.out_dir, args.out_jsonl_name)
    done_ids = load_done_ids_from_jsonl(out_jsonl)

    ds = load_dataset(args.dataset, split=args.split)
    total = len(ds)

    skip = max(0, int(args.skip_rows))
    if skip >= total:
        print(f"--skip-rows={skip} >= number of examples in dataset={total}. Nothing to do.")
        return

    start_idx = skip
    if args.max_rows and args.max_rows > 0:
        end_idx = min(total, start_idx + int(args.max_rows))
    else:
        end_idx = total

    window_count = end_idx - start_idx  # how many records we consider in this session (after skip)

    out_jsonl = os.path.join(args.out_dir, args.out_jsonl_name)
    done_ids = load_done_ids_from_jsonl(out_jsonl)

    # Count the progress bar only for the window [start_idx, end_idx)
    target_ids = set(ds[i]["id"] for i in range(start_idx, end_idx))
    already_done = len(done_ids.intersection(target_ids))

    non_tty_progress = not sys.stderr.isatty()
    pbar_rows = tqdm(
        total=window_count,
        initial=min(already_done, window_count),
        desc=f"Completed rows (idx {start_idx}..{end_idx - 1})",
        unit="row",
        dynamic_ncols=True,
        mininterval=0.5,
        disable=non_tty_progress,
    )
    completed_rows = min(already_done, window_count)
    started_at = time.time()
    log_every = max(1, int(args.log_every))
    total_pending = max(0, window_count - already_done)

    logging.info(
        "Translation run: dataset=%s split=%s range=%d..%d total_in_range=%d pending=%d done_before=%d combos=%d",
        args.dataset,
        args.split,
        start_idx,
        end_idx - 1,
        window_count,
        total_pending,
        already_done,
        len(runner.combos),
    )

    combo, client = None, None
    if runner.has_next():
        combo, client = runner.next_combo()
        logging.info("Start combo: model=%s key=***%s", combo.model, combo.api_key[-6:])
    else:
        raise RuntimeError("No available api_key × model combinations.")

    # Iterate over indices to easily skip the first N
    for ds_idx in range(start_idx, end_idx):
        row = ds[ds_idx]

        rid = row["id"]
        if rid in done_ids:
            # already translated — treat as "completed" in this window
            continue

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
                "dataset_index": ds_idx,  # helper, so we know where the record came from
            }
            write_json_atomic(ckpt_path, state)

        query_en = row["query"]
        texts: List[str] = row["texts"]
        context_spans: List[List[List[int]]] = row["context_spans"]
        context_spans_rel: List[List[int]] = row["context_spans_relevance"]
        labels: List[int] = row["labels"]
        think_process: List[str] = row["think_process"]

        inner_total = len(texts) + 1
        inner_done = (1 if state.get("query_pl") else 0) + len(set(state.get("done_text_idxs", [])))
        pbar_inner = tqdm(
            total=inner_total,
            initial=min(inner_done, inner_total),
            desc=f"id={rid} (ds_idx={ds_idx})",
            unit="step",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
            disable=non_tty_progress,
        )

        def ensure_combo():
            nonlocal combo, client
            if combo is None or client is None:
                if not runner.has_next():
                    raise RuntimeError("Ran out of api_key × model combinations (rate-limited or errors everywhere).")
                combo, client = runner.next_combo()
                logging.info("Switch combo: model=%s key=***%s", combo.model, combo.api_key[-6:])

        # --- translate query once
        while not state.get("query_pl"):
            ensure_combo()
            try:
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
                pbar_inner.set_postfix({"stage": "query", "model": combo.model})

                q_json = llm_call_json(
                    client=client,
                    model=combo.model,
                    system_prompt=SYSTEM_QUERY,
                    user_prompt=q_prompt,
                    temperature=args.temperature,
                    max_retries=args.max_retries,
                    delay_seconds=args.delay_seconds,
                )
                state["query_pl"] = (q_json.get("query_pl") or "").strip()
                if not state["query_pl"]:
                    raise RuntimeError("Model did not return query_pl.")

                state["active_model"] = combo.model
                state["active_key_last6"] = combo.api_key[-6:]
                write_json_atomic(ckpt_path, state)

                pbar_inner.set_postfix({"stage": "query", "status": "ok"})
                pbar_inner.update(1)

            except RateLimitReached:
                runner.mark_exhausted(combo)
                combo, client = None, None
                continue

        query_pl = state["query_pl"]

        # --- translate texts
        done_set = set(state.get("done_text_idxs", []))
        for i in range(len(texts)):
            if i in done_set:
                continue

            while True:
                ensure_combo()
                try:
                    gaps, span_texts_en = spans_to_pieces(texts[i], context_spans[i])
                    spans_rel_i = context_spans_rel[i]
                    if len(spans_rel_i) != len(span_texts_en):
                        m = min(len(spans_rel_i), len(span_texts_en))
                        span_texts_en = span_texts_en[:m]
                        spans_rel_i = spans_rel_i[:m]
                        gaps = gaps[:m + 1]

                    pbar_inner.set_postfix({"stage": f"text[{i}]", "model": combo.model})

                    translated_spans, translated_tp = translate_text_with_span_repair(
                        client=client,
                        model=combo.model,
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
                    state["active_model"] = combo.model
                    state["active_key_last6"] = combo.api_key[-6:]

                    write_json_atomic(ckpt_path, state)

                    pbar_inner.set_postfix({"stage": f"text[{i}]", "status": "ok"})
                    pbar_inner.update(1)
                    break

                except RateLimitReached:
                    runner.mark_exhausted(combo)
                    combo, client = None, None
                    continue

        pbar_inner.close()

        # --- finalize row
        if (
            all(x is not None for x in state["texts_pl"])
            and all(x is not None for x in state["context_spans_pl"])
            and all(x is not None for x in state["think_process_pl"])
        ):
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

            append_jsonl(out_jsonl, out_row)
            done_ids.add(rid)

            state["status"] = "done"
            write_json_atomic(ckpt_path, state)

            pbar_rows.update(1)
            completed_rows += 1
            if non_tty_progress and (
                completed_rows == already_done + 1
                or (completed_rows - already_done) % log_every == 0
                or completed_rows == window_count
            ):
                elapsed = time.time() - started_at
                rate = (completed_rows - already_done) / elapsed if elapsed > 0 else 0.0
                eta_seconds = (window_count - completed_rows) / rate if rate > 0 else 0.0
                logging.info(
                    "Progress: %d/%d rows (%.1f%%), rate=%.2f row/s, elapsed=%s, eta=%s",
                    completed_rows,
                    window_count,
                    100.0 * completed_rows / window_count,
                    rate,
                    format_seconds(elapsed),
                    format_seconds(eta_seconds),
                )

    pbar_rows.close()
    logging.info("Done. Output: %s", out_jsonl)
    if runner.exhausted:
        logging.warning("Exhausted combinations (429):")
        for c in runner.exhausted:
            logging.warning(" - model=%s key=***%s", c.model, c.api_key[-6:])


# ---------------------------
# CLI
# ---------------------------

def parse_csv_list(s: str) -> List[str]:
    # supports comma-separated or semicolon-separated lists
    items = []
    for part in re.split(r"[;,]", s.strip()):
        part = part.strip()
        if part:
            items.append(part)
    return items


def main() -> int:
    p = argparse.ArgumentParser(
        description="Translate a dataset (with checkpoints) using api_key×model rotation on 429."
    )
    p.add_argument("--base-url", default=None, help="Base URL for an OpenAI-compatible API (optional).")
    p.add_argument("--delay-seconds", type=float, default=2.0, help="Delay after each SUCCESSFUL request.")
    p.add_argument("--temperature", type=float, default=0.0, help="LLM temperature.")
    p.add_argument("--max-retries", type=int, default=2, help="Retries for errors other than 429.")
    p.add_argument("--api-keys", required=True, help="Key list, e.g. 'k1,k2,k3'.")
    p.add_argument("--models", required=True, help="Model list, e.g. 'm1,m2'.")

    # translate options
    p.add_argument("--dataset", default="zilliz/natural_questions-context-relevance-with-think")
    p.add_argument("--split", default="train", choices=["train", "validation", "test"])
    p.add_argument("--out-dir", default="out_pl")
    p.add_argument("--out-jsonl-name", default="translated.jsonl")
    p.add_argument("--checkpoint-dir", default=None, help="Default: <out-dir>/checkpoints")
    p.add_argument("--max-rows", type=int, default=0, help="0 = all")
    p.add_argument(
        "--keep-original-columns",
        dest="keep_original_columns",
        action="store_true",
        default=True,
        help="Keep original EN columns in the JSONL (default).",
    )
    p.add_argument(
        "--drop-original-columns",
        dest="keep_original_columns",
        action="store_false",
        help="Exclude original EN columns from the JSONL output.",
    )
    p.add_argument("--max-prompt-attempts", type=int, default=4, help="How many increasingly strict prompts to try for spans.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--log-every", type=int, default=10, help="Log progress every N completed rows in non-TTY mode")

    p.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        help="Skip the first N dataset examples (e.g. 5000 => start from example 5001).",
    )

    args = p.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    quiet_external_loggers()

    api_keys = parse_csv_list(args.api_keys)
    models = parse_csv_list(args.models)

    if args.checkpoint_dir is None:
        args.checkpoint_dir = os.path.join(args.out_dir, "checkpoints")

    runner = ComboRunner(api_keys=api_keys, models=models, base_url=args.base_url)

    run_translate(args=args, runner=runner)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted (CTRL+C).", file=sys.stderr)
        raise SystemExit(130)

