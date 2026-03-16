#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Iterator

from tqdm import tqdm

from run_answer_relevance_vllm import (
    extract_question_answer,
    read_jsonl_rows,
    resolve_row_id,
    selected_dataset_keys,
)
from translation_core import load_done_ids_from_jsonl

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2.5-gemma2-lightweight"
DEFAULT_RERANKER_PROMPT = "Predict whether passage B contains an answer to query A."
DEFAULT_RERANKER_MAX_LENGTH = 1024
DEFAULT_RERANKER_CUTOFF_LAYERS = [28]
DEFAULT_RERANKER_COMPRESS_RATIO = 1
DEFAULT_RERANKER_COMPRESS_LAYERS: list[int] = []
DEFAULT_JSONL_WRITE_BUFFER = 128

RERANKER_PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "reranker_max_length": 1024,
        "reranker_cutoff_layers": [28],
        "reranker_compress_ratio": 1,
        "reranker_compress_layers": [],
    },
    "balanced": {
        "reranker_max_length": 1024,
        "reranker_cutoff_layers": [32],
        "reranker_compress_ratio": 1,
        "reranker_compress_layers": [],
    },
    "quality": {
        "reranker_max_length": 1024,
        "reranker_cutoff_layers": [40],
        "reranker_compress_ratio": 1,
        "reranker_compress_layers": [],
    },
}


@dataclass
class ScoredRow:
    rid: str
    out_row: dict[str, Any]


@dataclass
class CandidateRow:
    row_idx: int
    row: dict[str, Any]
    rid: str
    query: str
    passage: str
    estimated_tokens: int


def parse_int_list(raw_values: list[str] | None, *, default: list[int]) -> list[int]:
    if not raw_values:
        return list(default)
    parsed: list[int] = []
    for raw in raw_values:
        for item in str(raw).split(","):
            value = item.strip()
            if not value:
                continue
            parsed.append(int(value))
    if not parsed:
        raise ValueError("Expected at least one integer value")
    return parsed


def resolve_reranker_runtime_params(args: argparse.Namespace) -> argparse.Namespace:
    cutoff_env = os.getenv("RERANKER_CUTOFF_LAYERS")
    compress_env = os.getenv("RERANKER_COMPRESS_LAYERS")
    max_length = (
        int(args.reranker_max_length)
        if args.reranker_max_length is not None
        else int(os.getenv("RERANKER_MAX_LENGTH", str(DEFAULT_RERANKER_MAX_LENGTH)))
    )
    compress_ratio = (
        int(args.reranker_compress_ratio)
        if args.reranker_compress_ratio is not None
        else int(os.getenv("RERANKER_COMPRESS_RATIO", str(DEFAULT_RERANKER_COMPRESS_RATIO)))
    )
    cutoff_layers = (
        parse_int_list(args.reranker_cutoff_layers, default=DEFAULT_RERANKER_CUTOFF_LAYERS)
        if args.reranker_cutoff_layers is not None
        else parse_int_list(
            [cutoff_env] if cutoff_env is not None else [",".join(str(x) for x in DEFAULT_RERANKER_CUTOFF_LAYERS)],
            default=DEFAULT_RERANKER_CUTOFF_LAYERS,
        )
    )
    compress_layers = (
        parse_int_list(args.reranker_compress_layers, default=DEFAULT_RERANKER_COMPRESS_LAYERS)
        if args.reranker_compress_layers is not None
        else (
            DEFAULT_RERANKER_COMPRESS_LAYERS
            if compress_env is None or not str(compress_env).strip()
            else parse_int_list([compress_env], default=DEFAULT_RERANKER_COMPRESS_LAYERS)
        )
    )

    args.reranker_max_length = max_length
    args.reranker_cutoff_layers = cutoff_layers
    args.reranker_compress_ratio = compress_ratio
    args.reranker_compress_layers = compress_layers

    if args.reranker_preset:
        preset = RERANKER_PRESETS[args.reranker_preset]
        args.reranker_max_length = int(preset["reranker_max_length"])
        args.reranker_cutoff_layers = list(preset["reranker_cutoff_layers"])
        args.reranker_compress_ratio = int(preset["reranker_compress_ratio"])
        args.reranker_compress_layers = list(preset["reranker_compress_layers"])

    return args


def resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
    normalized = str(dtype_name or "float16").strip().lower()
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    if normalized not in mapping:
        supported = ", ".join(sorted(mapping))
        raise RuntimeError(f"Unsupported reranker dtype '{dtype_name}'. Supported values: {supported}")
    return mapping[normalized]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def detect_cuda_available() -> bool:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return False
    return bool(torch.cuda.is_available())


def last_logit_pool(torch_module: Any, logits: Any, attention_mask: Any) -> Any:
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return logits[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_indices = torch_module.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, sequence_lengths]


def build_model_inputs(
    pairs: list[tuple[str, str]],
    tokenizer: Any,
    *,
    prompt_inputs: list[int],
    sep_inputs: list[int],
    max_length: int,
) -> tuple[Any, list[int], list[int]]:
    query_texts = [f"A: {query}" for query, _ in pairs]
    passage_texts = [f"B: {passage}" for _, passage in pairs]

    query_batch = tokenizer(
        query_texts,
        return_tensors=None,
        add_special_tokens=False,
        max_length=max_length * 3 // 4,
        truncation=True,
    )
    passage_batch = tokenizer(
        passage_texts,
        return_tensors=None,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
    )

    inputs: list[dict[str, Any]] = []
    query_lengths: list[int] = []
    prompt_lengths: list[int] = []
    bos_token_id = tokenizer.bos_token_id

    for query_ids, passage_ids in zip(query_batch["input_ids"], passage_batch["input_ids"]):
        query_with_bos = [bos_token_id] + query_ids
        item = tokenizer.prepare_for_model(
            query_with_bos,
            sep_inputs + passage_ids,
            truncation="only_second",
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            add_special_tokens=False,
        )
        item["input_ids"] = item["input_ids"] + sep_inputs + prompt_inputs
        item["attention_mask"] = [1] * len(item["input_ids"])
        inputs.append(item)
        query_lengths.append(len(query_with_bos) + len(sep_inputs))
        prompt_lengths.append(len(sep_inputs) + len(prompt_inputs))

    return (
        tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors="pt",
        ),
        query_lengths,
        prompt_lengths,
    )


def build_output_row(
    row: dict[str, Any],
    *,
    raw_score: float,
    sigmoid_score: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_row = dict(row)
    out_row["answer_relevance_reranker"] = {
        "raw_score": raw_score,
        "sigmoid_score": sigmoid_score,
        "prompt": args.reranker_prompt,
        "cutoff_layers": list(args.reranker_cutoff_layers),
        "compress_ratio": int(args.reranker_compress_ratio),
        "compress_layers": list(args.reranker_compress_layers),
        "max_length": int(args.reranker_max_length),
    }
    out_row["answer_relevance_reranker_model"] = args.reranker_model
    out_row["answer_relevance_reranker_dtype"] = args.reranker_dtype
    out_row["answer_relevance_reranker_batch_size"] = int(args.batch_size)
    out_row["answer_relevance_reranker_timestamp_unix"] = int(time.time())
    return out_row


def estimate_pair_tokens_heuristic(query: str, passage: str, max_length: int) -> int:
    # Prosta heurystyka do sortowania i batchowania po podobnej długości.
    # Nie musi być idealna — ma tylko ograniczyć padding.
    # ~4 znaki na token to użyteczny przybliżony przelicznik.
    estimated = (len(query) + len(passage)) // 4 + 32
    estimated = max(32, estimated)
    return min(max_length + 64, estimated)


def append_jsonl_many(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def flush_jsonl_buffer(path: str, buffer_rows: list[dict[str, Any]]) -> None:
    if not buffer_rows:
        return
    append_jsonl_many(path, buffer_rows)
    buffer_rows.clear()


def build_candidates(
    rows: list[dict[str, Any]],
    *,
    dataset_key: str,
    skip: int,
    end_idx: int,
    done_ids: set[str],
    failed_ids: set[str],
    max_length: int,
) -> tuple[list[CandidateRow], int]:
    candidates: list[CandidateRow] = []
    skipped_failed = 0

    for row_idx in range(skip, end_idx):
        row = rows[row_idx]
        rid = resolve_row_id(row, dataset_key, row_idx)
        if rid in done_ids:
            continue
        if rid in failed_ids:
            skipped_failed += 1
            continue

        query, passage = extract_question_answer(row, dataset_key)
        estimated_tokens = estimate_pair_tokens_heuristic(query, passage, max_length)
        candidates.append(
            CandidateRow(
                row_idx=row_idx,
                row=row,
                rid=rid,
                query=query,
                passage=passage,
                estimated_tokens=estimated_tokens,
            )
        )

    return candidates, skipped_failed


def maybe_sort_candidates(candidates: list[CandidateRow], mode: str) -> list[CandidateRow]:
    if mode == "off":
        return candidates
    if mode == "length":
        return sorted(candidates, key=lambda item: item.estimated_tokens)
    raise ValueError(f"Unsupported length bucketing mode: {mode}")


def iter_candidate_batches(
    candidates: list[CandidateRow],
    *,
    max_batch_size: int,
    max_batch_tokens: int,
) -> Iterator[list[CandidateRow]]:
    batch: list[CandidateRow] = []
    batch_tokens = 0

    for item in candidates:
        item_tokens = max(1, int(item.estimated_tokens))
        would_exceed_size = len(batch) >= max_batch_size
        would_exceed_tokens = bool(batch) and max_batch_tokens > 0 and (batch_tokens + item_tokens) > max_batch_tokens

        if would_exceed_size or would_exceed_tokens:
            yield batch
            batch = []
            batch_tokens = 0

        batch.append(item)
        batch_tokens += item_tokens

        flush_due_to_size = len(batch) >= max_batch_size
        flush_due_to_tokens = max_batch_tokens > 0 and batch_tokens >= max_batch_tokens
        if flush_due_to_size or flush_due_to_tokens:
            yield batch
            batch = []
            batch_tokens = 0

    if batch:
        yield batch


class GemmaLightweightReranker:
    def __init__(self, args: argparse.Namespace) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Reranker mode requires 'torch' and 'transformers' in the current Python environment."
            ) from exc

        self._torch = torch
        self._prompt = args.reranker_prompt
        self._max_length = int(args.reranker_max_length)
        self._cutoff_layers = list(args.reranker_cutoff_layers)
        self._compress_ratio = int(args.reranker_compress_ratio)
        self._compress_layers = list(args.reranker_compress_layers)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_quantized_8bit = False
        self._load_in_8bit = bool(args.reranker_load_in_8bit) and self._device.type == "cuda"

        logging.info(
            "Reranker parameters: preset=%s max_length=%d cutoff_layers=%s compress_ratio=%d compress_layers=%s",
            args.reranker_preset or "none",
            self._max_length,
            self._cutoff_layers,
            self._compress_ratio,
            self._compress_layers,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.reranker_model,
            trust_remote_code=True,
        )
        tokenizer.padding_side = "right"

        self._sep_inputs = tokenizer("\n", return_tensors=None, add_special_tokens=False)["input_ids"]
        self._prompt_inputs = tokenizer(self._prompt, return_tensors=None, add_special_tokens=False)["input_ids"]

        if self._load_in_8bit:
            missing_runtime: list[str] = []
            if importlib.util.find_spec("bitsandbytes") is None:
                missing_runtime.append("bitsandbytes")
            if importlib.util.find_spec("accelerate") is None:
                missing_runtime.append("accelerate")
            if missing_runtime:
                missing = ", ".join(missing_runtime)
                raise RuntimeError(
                    "8-bit reranker loading requires the following packages to be installed: "
                    f"{missing}. Install them before running CUDA int8 scoring."
                )

            # quantization_config = BitsAndBytesConfig(
            #     load_in_8bit=True,
            #     llm_int8_enable_fp32_cpu_offload=bool(args.reranker_int8_cpu_offload),
            # )
            # try:
            #     model = AutoModelForCausalLM.from_pretrained(
            #         args.reranker_model,
            #         trust_remote_code=True,
            #         quantization_config=quantization_config,
            #         device_map="auto",
            #         torch_dtype="auto",
            #         low_cpu_mem_usage=True,
            #     )
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=False,
            )

            try:
                model = AutoModelForCausalLM.from_pretrained(
                    args.reranker_model,
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    device_map=0,  # cały model na GPU 0
                    torch_dtype="auto",
                    low_cpu_mem_usage=True,
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    "Failed to load reranker in 8-bit mode. Ensure both bitsandbytes and accelerate are installed "
                    f"and compatible with your CUDA environment. Original error: {exc}"
                ) from exc
            self._is_quantized_8bit = True
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.reranker_model,
                trust_remote_code=True,
                torch_dtype=resolve_torch_dtype(torch, args.reranker_dtype),
                low_cpu_mem_usage=True,
            )
            model = model.to(self._device)

        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._input_device = self._get_input_device(self._model)
        if hasattr(model, "hf_device_map"):
            logging.info("hf_device_map=%s", model.hf_device_map)

        logging.info(
            "Reranker model loaded: model=%s load_in_8bit=%s int8_cpu_offload=%s scoring_device=%s",
            args.reranker_model,
            self._is_quantized_8bit,
            bool(args.reranker_int8_cpu_offload),
            self._input_device,
        )

    # def _get_input_device(self, model: Any) -> Any:
    #     if self._is_quantized_8bit:
    #         try:
    #             return next(model.parameters()).device
    #         except StopIteration:
    #             return self._torch.device(self._device)
    #
    #     if hasattr(model, "device"):
    #         try:
    #             return self._torch.device(model.device)
    #         except Exception:  # noqa: BLE001
    #             pass
    #
    #     try:
    #         return next(model.parameters()).device
    #     except StopIteration:
    #         return self._torch.device(self._device)
    def _get_input_device(self, model: Any) -> Any:
        if hasattr(model, "device"):
            try:
                return self._torch.device(model.device)
            except Exception:
                pass

        try:
            return next(model.parameters()).device
        except StopIteration:
            return self._torch.device(self._device)

    def _move_batch_to_device(self, batch: Any, device: Any) -> Any:
        if self._torch.is_tensor(batch):
            return batch.to(device, non_blocking=True)
        if hasattr(batch, "to"):
            try:
                return batch.to(device)
            except TypeError:
                return batch.to(device=device)
        if isinstance(batch, dict):
            return {key: self._move_batch_to_device(value, device) for key, value in batch.items()}
        if isinstance(batch, list):
            return [self._move_batch_to_device(value, device) for value in batch]
        if isinstance(batch, tuple):
            return tuple(self._move_batch_to_device(value, device) for value in batch)
        return batch

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float]]:
        if not pairs:
            return []

        with self._torch.inference_mode():
            inputs, query_lengths, prompt_lengths = build_model_inputs(
                pairs,
                self._tokenizer,
                prompt_inputs=self._prompt_inputs,
                sep_inputs=self._sep_inputs,
                max_length=self._max_length,
            )
            input_device = self._get_input_device(self._model)
            inputs = self._move_batch_to_device(inputs, input_device)
            outputs = self._model(
                **inputs,
                return_dict=True,
                cutoff_layers=self._cutoff_layers,
                compress_ratio=self._compress_ratio,
                compress_layer=self._compress_layers,
                query_lengths=query_lengths,
                prompt_lengths=prompt_lengths,
            )
            pooled = last_logit_pool(self._torch, outputs.logits[-1], outputs.attention_masks[-1])
            raw_scores = pooled.detach().cpu().float().tolist()
            return [(float(raw_score), sigmoid(float(raw_score))) for raw_score in raw_scores]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Score translated QA pairs with the BAAI lightweight Gemma2 reranker and write resumable JSONL outputs."
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
    p.add_argument("--out-jsonl-name", default="answer_relevance_reranker.jsonl")
    p.add_argument("--failed-jsonl-name", default="answer_relevance_reranker_failed_rows.jsonl")
    p.add_argument(
        "--retry-failed-rows",
        action="store_true",
        help="Include rows previously present in failed_rows JSONL when resuming.",
    )
    p.add_argument("--max-rows", type=int, default=0, help="0 = all")
    p.add_argument("--skip-rows", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--max-batch-tokens",
        type=int,
        default=0,
        help="Optional heuristic token budget per batch. 0 disables token-budget batching.",
    )
    p.add_argument(
        "--length-bucketing",
        choices=["off", "length"],
        default="length",
        help="Batch samples with similar estimated length together. Default: length.",
    )
    p.add_argument(
        "--jsonl-write-buffer",
        type=int,
        default=DEFAULT_JSONL_WRITE_BUFFER,
        help="Number of output rows buffered before JSONL flush.",
    )
    p.add_argument("--reranker-model", default=os.getenv("RERANKER_MODEL_NAME", DEFAULT_RERANKER_MODEL))
    p.add_argument("--reranker-prompt", default=os.getenv("RERANKER_PROMPT", DEFAULT_RERANKER_PROMPT))
    p.add_argument(
        "--reranker-preset",
        choices=sorted(RERANKER_PRESETS.keys()),
        default=None,
        help="Optional preset for reranker runtime parameters.",
    )
    p.add_argument("--reranker-max-length", type=int, default=None)
    p.add_argument("--reranker-compress-ratio", type=int, default=None)
    p.add_argument(
        "--reranker-cutoff-layers",
        nargs="*",
        default=None,
        help="One or more integers, optionally comma-separated. Default: 28.",
    )
    p.add_argument(
        "--reranker-compress-layers",
        nargs="*",
        default=None,
        help="One or more integers, optionally comma-separated. Default: empty.",
    )
    p.add_argument("--reranker-dtype", default=os.getenv("RERANKER_DTYPE", "float16"))
    p.add_argument(
        "--reranker-load-in-8bit",
        dest="reranker_load_in_8bit",
        action="store_true",
        default=None,
        help="Load the reranker in 8-bit with bitsandbytes on CUDA. Default: on for CUDA, off for CPU.",
    )
    p.add_argument(
        "--no-reranker-load-in-8bit",
        dest="reranker_load_in_8bit",
        action="store_false",
        help="Disable 8-bit reranker loading even when CUDA is available.",
    )
    p.add_argument(
        "--reranker-int8-cpu-offload",
        action="store_true",
        default=env_flag("RERANKER_INT8_CPU_OFFLOAD", False),
        help="Enable bitsandbytes fp32 CPU offload for the 8-bit reranker path.",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument(
        "--progress-bar",
        default=os.getenv("PROGRESS_BAR", "on"),
        choices=["auto", "on", "off"],
        help="Progress bar mode: auto=TTY only, on=always, off=disable tqdm",
    )
    args = p.parse_args()
    if args.reranker_load_in_8bit is None:
        args.reranker_load_in_8bit = env_flag("RERANKER_LOAD_IN_8BIT", detect_cuda_available())
    return resolve_reranker_runtime_params(args)


def score_rows_or_fallback(
    reranker: GemmaLightweightReranker,
    batch: list[CandidateRow],
    dataset_key: str,
    args: argparse.Namespace,
) -> tuple[list[ScoredRow], list[dict[str, Any]]]:
    pairs = [(item.query, item.passage) for item in batch]

    try:
        scores = reranker.score_pairs(pairs)
        results: list[ScoredRow] = []
        for item, (raw_score, sigmoid_score) in zip(batch, scores):
            results.append(
                ScoredRow(
                    rid=item.rid,
                    out_row=build_output_row(
                        item.row,
                        raw_score=raw_score,
                        sigmoid_score=sigmoid_score,
                        args=args,
                    ),
                )
            )
        return results, []
    except Exception as exc:  # noqa: BLE001
        if len(batch) == 1:
            item = batch[0]
            return [], [
                {
                    "id": item.rid,
                    "source_dataset": dataset_key,
                    "row_idx": item.row_idx,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "timestamp_unix": int(time.time()),
                }
            ]

        batch_estimated_tokens = sum(item.estimated_tokens for item in batch)
        logging.warning(
            "Batch scoring failed for dataset=%s batch_size=%d estimated_tokens=%d. Retrying rows individually. Error: %s",
            dataset_key,
            len(batch),
            batch_estimated_tokens,
            exc,
        )
        ok_rows: list[ScoredRow] = []
        failed_rows: list[dict[str, Any]] = []
        for item in batch:
            partial_ok, partial_failed = score_rows_or_fallback(reranker, [item], dataset_key, args)
            ok_rows.extend(partial_ok)
            failed_rows.extend(partial_failed)
        return ok_rows, failed_rows


def run_single_dataset(args: argparse.Namespace, reranker: GemmaLightweightReranker) -> int:
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

    candidates, skipped_failed = build_candidates(
        rows,
        dataset_key=args.dataset_key,
        skip=skip,
        end_idx=end_idx,
        done_ids=done_ids,
        failed_ids=failed_ids,
        max_length=int(args.reranker_max_length),
    )
    candidates = maybe_sort_candidates(candidates, args.length_bucketing)

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
        "Reranker run: dataset_key=%s input=%s output=%s model=%s batch_size=%d max_batch_tokens=%d "
        "length_bucketing=%s range=%d..%d total_in_range=%d pending=%d done_before=%d skipped_failed=%d "
        "retry_failed_rows=%s",
        args.dataset_key,
        input_jsonl,
        out_jsonl,
        args.reranker_model,
        int(args.batch_size),
        int(args.max_batch_tokens),
        args.length_bucketing,
        skip,
        end_idx - 1,
        end_idx - skip,
        len(candidates),
        (end_idx - skip) - len(candidates),
        skipped_failed,
        bool(args.retry_failed_rows),
    )

    is_tty = sys.stderr.isatty()
    show_pbar = args.progress_bar == "on" or (args.progress_bar == "auto" and is_tty)
    pbar = tqdm(
        total=len(candidates),
        desc="Completed rows",
        unit="row",
        dynamic_ncols=True,
        mininterval=0.5,
        ascii=not is_tty,
        disable=not show_pbar,
    )

    failed_count = 0
    out_buffer: list[dict[str, Any]] = []
    failed_buffer: list[dict[str, Any]] = []
    batch_iter = iter_candidate_batches(
        candidates,
        max_batch_size=max(1, int(args.batch_size)),
        max_batch_tokens=max(0, int(args.max_batch_tokens)),
    )

    try:
        for batch in batch_iter:
            ok_rows, failed_rows = score_rows_or_fallback(reranker, batch, args.dataset_key, args)

            for item in ok_rows:
                if item.rid not in done_ids:
                    out_buffer.append(item.out_row)
                    done_ids.add(item.rid)

            for failed_obj in failed_rows:
                failed_buffer.append(failed_obj)
                failed_count += 1

            if len(out_buffer) >= int(args.jsonl_write_buffer):
                flush_jsonl_buffer(out_jsonl, out_buffer)
            if len(failed_buffer) >= int(args.jsonl_write_buffer):
                flush_jsonl_buffer(failed_jsonl, failed_buffer)

            pbar.update(len(batch))
    finally:
        pbar.close()
        flush_jsonl_buffer(out_jsonl, out_buffer)
        flush_jsonl_buffer(failed_jsonl, failed_buffer)

    if failed_count:
        logging.warning("Done with row failures: %d failed rows. See %s", failed_count, failed_jsonl)
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

    reranker = GemmaLightweightReranker(args)
    for dataset_key in selected_keys:
        run_args = argparse.Namespace(**vars(args))
        run_args.dataset_key = dataset_key
        logging.info(
            "Starting reranker run: key=%s out_dir=%s input_jsonl=%s",
            dataset_key,
            os.path.join(args.out_dir, dataset_key),
            args.input_jsonl_name,
        )
        rc = run_single_dataset(run_args, reranker)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted (CTRL+C).", file=sys.stderr)
        raise SystemExit(130)