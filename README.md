# Translate context relevance dataset (EN -> PL)

This repository provides a tool for translating context-relevance Hugging Face datasets from English to Polish.

Supported datasets:
- [`zilliz/natural_questions-context-relevance-with-think`](https://huggingface.co/datasets/zilliz/natural_questions-context-relevance-with-think) (`nq`) - A Natural Questions-based context-relevance dataset with queries and candidate passages.
- [`sentence-transformers/natural-questions`](https://huggingface.co/datasets/sentence-transformers/natural-questions) (`nq_qa`) - A Natural Questions QA-style dataset used here for question-answer translation and answer relevance scoring.
- [`sentence-transformers/hotpotqa`](https://huggingface.co/datasets/sentence-transformers/hotpotqa) (`hotpotqa`) - A HotpotQA-derived triplet dataset for question-answer relevance and retrieval-style evaluation.
- [`zilliz/msmarco-context-relevance-with-think`](https://huggingface.co/datasets/zilliz/msmarco-context-relevance-with-think) (`msmarco`) - An MS MARCO-based context-relevance dataset with queries, passages, and rationale-style annotations.
- [`thesofakillers/jigsaw-toxic-comment-classification-challenge`](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge) (`toxic`, opt-in only) - A toxic comment classification dataset translated with toxicity labels preserved.
- [`allenai/wildguardmix`](https://huggingface.co/datasets/allenai/wildguardmix) (`wildguard`, opt-in only) - A safety dataset with benign and harmful prompts translated while preserving their safety intent.

The pipeline can run in three modes:
- local mode with **vLLM (OpenAI-compatible API)**
- external mode with any **OpenAI-compatible API** (for example OpenAI, Groq, OpenRouter)
- local mode with **vLLM offline inference** (no API server)

All modes use the same translator script and output/checkpoint format.
Results are written to JSONL, and progress is persisted with checkpoints so the process can be safely resumed.

## Requirements

- Docker + Docker Compose
- NVIDIA GPU with a working GPU container runtime (required only for local vLLM mode)

## Configuration

Copy an environment profile:

```bash
cp .env.example .env
```

Key variables in `.env`:

- `MODEL_NAME` - model name used for translation (vLLM or external provider)
- `INFERENCE_SOURCE` - translation backend: `vllm` (default), `external`, or `offline`
- `OPENAI_COMPAT_BASE_URL` - external OpenAI-compatible endpoint base URL (used in `external` mode)
- `OPENAI_COMPAT_API_KEY` - external OpenAI-compatible API key (used in `external` mode)
- `PARALLEL_REQUESTS` - number of parallel translation tasks on the translator side (`asyncio` + semaphore)
- `PROGRESS_BAR` - translation progress display mode: `on` (default), `auto` (TTY only), `off`
- `PROGRESS_METRIC` - progress metric for `tqdm`: `checkpoints` (default), `rows`, `both`
- `GPU_COUNT` - number of GPUs used by vLLM (`--tensor-parallel-size`)
- `VLLM_QUANTIZATION` (optional) - vLLM quantization mode; leave empty to let vLLM auto-detect model quantization
- `VLLM_MAX_NUM_SEQS` (optional) - passed to `--max-num-seqs` only when set
- `VLLM_MAX_NUM_BATCHED_TOKENS` (optional) - passed to `--max-num-batched-tokens` only when set
- `VLLM_ENFORCE_EAGER` (optional) - if set to `1`, enables `--enforce-eager`
- `HF_TOKEN` (optional/required for gated datasets) - Hugging Face token used by `load_dataset`
- `FEW_SHOT_EXAMPLES_PATH` (optional) - path to CSV with EN->PL few-shot examples for pair-style datasets (`nq_qa`, `hotpotqa`)
- `FEW_SHOT_EXAMPLE_COUNT` (optional) - number of random few-shot examples prepended for each pair-style prompt (default: `3`)
- `FEW_SHOT_SHARED_REQUESTS` (optional) - how many consecutive pair-style requests reuse the same sampled few-shot examples (default: `10`)

Available profiles:

- `.env.example` - lightweight profile (defaults to `Qwen/Qwen2.5-0.5B-Instruct`, `PARALLEL_REQUESTS=2`, `GPU_COUNT=1`)
- `.env.gptoss` - multi-GPU profile (`openai/gpt-oss-120b`, `PARALLEL_REQUESTS=16`, `GPU_COUNT=4`)
- `.env.bielikq4` - Bielik 11B AWQ profile with 16 GB-oriented tuning (`speakleash/Bielik-11B-v3.0-Instruct-awq`, `VLLM_QUANTIZATION=`, `PARALLEL_REQUESTS=4`)

## Running

### Local vLLM mode

1. Start vLLM:

```bash
docker compose up -d --build vllm
```

2. Start translation:

```bash
docker compose run --rm translator
```

### External API mode (OpenAI-compatible)

Set `INFERENCE_SOURCE=external` in `.env` and provide:
- `OPENAI_COMPAT_BASE_URL`
- `OPENAI_COMPAT_API_KEY`

Then run translation without the vLLM dependency:

```bash
docker compose run --rm translator-external --inference-source external
```

Alternative (same service, skip dependencies explicitly):

```bash
docker compose run --rm --no-deps translator --inference-source external
```

### Offline vLLM mode (no server)

Use this mode when you want to run translation directly through the vLLM Python engine (offline inference), fully via Docker Compose.

Build and run:

```bash
docker compose build translator-offline
docker compose run --rm translator-offline --datasets nq --split train
```

Example:

```bash
docker compose run --rm translator-offline --datasets toxic --split train
```

QA answer relevance scoring on already translated Polish files:

```bash
docker compose run --rm qa-relevance-offline --datasets nq_qa hotpotqa
```

This reads `out_pl/<dataset>/translated.jsonl` and writes `out_pl/<dataset>/answer_relevance.jsonl`.
The output row keeps the original translated record and adds an `answer_relevance` object with `explanation` and `label`.
Completed rows are skipped on resume, so you can stop the run and continue later.

Reranker scoring on the same translated Polish files:

```bash
docker compose run --rm qa-reranker-offline --datasets nq_qa hotpotqa
```

This uses `BAAI/bge-reranker-v2.5-gemma2-lightweight` and writes `out_pl/<dataset>/answer_relevance_reranker.jsonl`.
The output row keeps the source record and adds `answer_relevance_reranker.raw_score` plus `answer_relevance_reranker.sigmoid_score`.
Completed rows are skipped on resume here as well.
The reranker service uses a dedicated GPU image with its own pinned Hugging Face stack, separate from the `vllm` image.

Optional offline tuning flags:
- `--offline-tensor-parallel-size`
- `--offline-gpu-memory-utilization`
- `--offline-max-model-len`
- `--offline-max-num-seqs`
- `--offline-max-num-batched-tokens`
- `--offline-enforce-eager`
- `--offline-dtype`
- `--offline-max-output-tokens`
- `--offline-micro-batch-size` (default: `150`)

By default, the translator runs both context-relevance datasets sequentially (`nq` then `msmarco`).
The `toxic` and `wildguard` datasets are not included in `all` and run only when explicitly selected.
The `nq_qa` and `hotpotqa` datasets are also opt-in and keep their own output folders to avoid collisions with existing names.
Use `--datasets` to limit the run:

```bash
docker compose run --rm translator --datasets nq
docker compose run --rm translator --datasets nq_qa
docker compose run --rm translator --datasets hotpotqa
docker compose run --rm translator --datasets msmarco
docker compose run --rm translator --datasets toxic --split train
docker compose run --rm translator --datasets wildguard --split train
docker compose run --rm translator --datasets nq_qa hotpotqa toxic wildguard --split train
```

You can pass multiple dataset keys in one run; duplicates are ignored.

## First test on a small GPU (e.g. 8 GB VRAM)

Recommended quick end-to-end test:

```bash
cp .env.example .env
docker compose up -d --build vllm
docker compose run --rm translator --max-rows 5
```

If you hit GPU OOM:

- set `PARALLEL_REQUESTS=1`
- reduce `MAX_MODEL_LEN` (for example to `1024`)
- set `VLLM_MAX_NUM_SEQS=1`
- set `VLLM_MAX_NUM_BATCHED_TOKENS` to your `MAX_MODEL_LEN`
- set `VLLM_ENFORCE_EAGER=1`
- make sure you are using the small model profile from `.env.example`

## Output and checkpoints

Output files are written inside the repository directory, in separate subfolders per dataset:

- `out_pl/nq/translated.jsonl`, `out_pl/nq/failed_rows.jsonl`, `out_pl/nq/checkpoints/*.json`
- `out_pl/nq_qa/translated.jsonl`, `out_pl/nq_qa/failed_rows.jsonl`
- `out_pl/nq_qa/answer_relevance.jsonl`, `out_pl/nq_qa/answer_relevance_failed_rows.jsonl`
- `out_pl/nq_qa/answer_relevance_reranker.jsonl`, `out_pl/nq_qa/answer_relevance_reranker_failed_rows.jsonl`
- `out_pl/hotpotqa/translated.jsonl`, `out_pl/hotpotqa/failed_rows.jsonl`
- `out_pl/hotpotqa/answer_relevance.jsonl`, `out_pl/hotpotqa/answer_relevance_failed_rows.jsonl`
- `out_pl/hotpotqa/answer_relevance_reranker.jsonl`, `out_pl/hotpotqa/answer_relevance_reranker_failed_rows.jsonl`
- `out_pl/msmarco/translated.jsonl`, `out_pl/msmarco/failed_rows.jsonl`, `out_pl/msmarco/checkpoints/*.json`
- `out_pl/toxic/translated.jsonl`, `out_pl/toxic/failed_rows.jsonl`, `out_pl/toxic/checkpoints/*.json`
- `out_pl/wildguard/translated.jsonl`, `out_pl/wildguard/failed_rows.jsonl`, `out_pl/wildguard/checkpoints/*.json`

You can resume processing by running the translator again with the same parameters.
Already completed records are skipped.
For correct interactive `tqdm` rendering, run the translator with `docker compose run --rm translator`.

Resume behavior after interruption:
- for context-relevance datasets (`nq`, `msmarco`), checkpoints are written after query translation and after every translated text in a row
- if the script is interrupted, rerun with the same arguments and it continues from the last completed checkpoint unit
- completed rows are deduplicated by `id` in output JSONL, so resumed runs do not re-append finished records

Runtime behavior:

- the translator uses structured output (`response_format=json_schema` when supported, with fallback to `json_object`) to enforce translation shape
- in offline vLLM mode, the translator also uses vLLM structured decoding (`structured_outputs`/`guided_decoding`) when available
- for `nq_qa` and `hotpotqa`, prompts reuse one sampled few-shot set for every 10 consecutive requests by default, with 3 random EN->PL examples from [`prompt_examples/ms_marco_translation_examples.csv`](/c:/work/translate-context-relevance-datasets/prompt_examples/ms_marco_translation_examples.csv)
- for `hotpotqa`, only `anchor` and `positive` are translated; `negative` is copied through unchanged in English
- `hotpotqa` is loaded with the default Hugging Face config `triplet`
- row-level failures do not stop the whole run by default; they are logged to `<out-dir>/<dataset_key>/failed_rows.jsonl`
- use `--fail-fast` to stop the entire run on the first failed row
- use `--failed-jsonl-name <name>` to change the failed-rows file name
- answer relevance scoring is available for `nq_qa` and `hotpotqa`; it reads translated Polish JSONL files and adds an `answer_relevance` object with structured output fields ordered as `explanation`, then `label`
- use `qa-relevance`, `qa-relevance-external`, or `qa-relevance-offline` services for the QA scoring stage
- reranker scoring is also available for `nq_qa` and `hotpotqa`; it uses `BAAI/bge-reranker-v2.5-gemma2-lightweight` to compute numeric pair scores and writes them to `answer_relevance_reranker.jsonl`
- if the reranker model download is blocked on Hugging Face, provide `HF_TOKEN` in the environment before running the offline service

## Architecture

`docker-compose.yml` defines these services:

- `vllm` - OpenAI-compatible endpoint at `http://vllm:8000/v1`
- `translator` - client service (depends on `vllm`) for local mode
- `translator-external` - client service without `vllm` dependency for external mode
- `translator-offline` - translator running vLLM offline inference in-process (no API server dependency)
- `qa-relevance` - answer relevance scorer using the local `vllm` server
- `qa-relevance-external` - answer relevance scorer against an external OpenAI-compatible API
- `qa-relevance-offline` - answer relevance scorer running vLLM offline inference in-process
- `qa-reranker-offline` - answer relevance scorer using a dedicated GPU Hugging Face reranker image in-process

`vllm` and `translator-offline` share model/cache volumes (`hf-cache`, `vllm-cache`), so model files are downloaded once and reused by both services.

The translator service:
  - reads dataset rows,
  - translates queries and documents,
  - preserves output/checkpoint format compatible with the existing workflow,
  - runs row processing concurrently via `asyncio` tasks (bounded by a semaphore),
  - serializes disk writes through a dedicated writer task (queue) to reduce worker blocking.

## Dependencies (translator vs reranker)

In the translator image, dependencies are installed via `uv` from `requirements.txt`.
The source dependency file is `requirements.in`.

Update pinned dependencies:

```bash
uv pip compile requirements.in -o requirements.txt
```

The standalone reranker image uses `requirements-reranker.txt` with a separately pinned GPU/Hugging Face stack so it can track a `transformers` version compatible with `BAAI/bge-reranker-v2.5-gemma2-lightweight` without affecting the `vllm` services.

## Development

Run unit tests:

```bash
pytest -q
```

CI runs tests automatically on push and pull requests.

## Original script

The repository still includes the original script:

- `translate_context_relevance_dataset.py`

It is kept as a legacy/compatibility runner alongside `run_translation_vllm.py`.
