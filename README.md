# Translate context relevance dataset (EN -> PL)

This repository provides a tool for translating context-relevance HuggingFace datasets from English to Polish.
Currently supported:
- `zilliz/natural_questions-context-relevance-with-think` (`nq`)
- `zilliz/msmarco-context-relevance-with-think` (`msmarco`)
- `thesofakillers/jigsaw-toxic-comment-classification-challenge` (`toxic`, opt-in only)
- `allenai/wildguardmix` (`wildguard`, opt-in only)

The pipeline runs locally using **vLLM (OpenAI-compatible API)** and a separate **translator** container.
Results are written to JSONL, and progress is persisted with checkpoints so the process can be safely resumed.

## Requirements

- Docker + Docker Compose
- NVIDIA GPU (for vLLM) with a working GPU container runtime

## Configuration

Copy an environment profile:

```bash
cp .env.example .env
```

Key variables in `.env`:

- `MODEL_NAME` - model name served by vLLM
- `PARALLEL_REQUESTS` - number of parallel translation tasks on the translator side (`asyncio` + semaphore)
- `PROGRESS_BAR` - translation progress display mode: `on` (default), `auto` (TTY only), `off`
- `PROGRESS_METRIC` - progress metric for `tqdm`: `checkpoints` (default), `rows`, `both`
- `GPU_COUNT` - number of GPUs used by vLLM (`--tensor-parallel-size`)
- `VLLM_QUANTIZATION` (optional) - vLLM quantization mode; leave empty to let vLLM auto-detect model quantization
- `VLLM_MAX_NUM_SEQS` (optional) - passed to `--max-num-seqs` only when set
- `VLLM_MAX_NUM_BATCHED_TOKENS` (optional) - passed to `--max-num-batched-tokens` only when set
- `VLLM_ENFORCE_EAGER` (optional) - if set to `1`, enables `--enforce-eager`
- `HF_TOKEN` (optional/required for gated datasets) - Hugging Face token used by `load_dataset`

Available profiles:

- `.env.example` - lightweight profile (defaults to `Qwen/Qwen2.5-0.5B-Instruct`, `PARALLEL_REQUESTS=2`, `GPU_COUNT=1`)
- `.env.gptoss` - multi-GPU profile (`openai/gpt-oss-120b`, `PARALLEL_REQUESTS=16`, `GPU_COUNT=4`)
- `.env.bielikq4` - Bielik 11B AWQ profile with 16 GB-oriented tuning (`speakleash/Bielik-11B-v3.0-Instruct-awq`, `VLLM_QUANTIZATION=`, `PARALLEL_REQUESTS=4`)

## Running

1. Start vLLM:

```bash
docker compose up -d --build vllm
```

2. Start translation:

```bash
docker compose run --rm translator
```

By default, the translator runs both context-relevance datasets sequentially (`nq` then `msmarco`).
The `toxic` and `wildguard` datasets are not included in `all` and run only when explicitly selected.
Use `--datasets` to limit the run:

```bash
docker compose run --rm translator --datasets nq
docker compose run --rm translator --datasets msmarco
docker compose run --rm translator --datasets toxic --split train
docker compose run --rm translator --datasets wildguard --split train
docker compose run --rm translator --datasets toxic wildguard --split train
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
- `out_pl/msmarco/translated.jsonl`, `out_pl/msmarco/failed_rows.jsonl`, `out_pl/msmarco/checkpoints/*.json`
- `out_pl/toxic/translated.jsonl`, `out_pl/toxic/failed_rows.jsonl`, `out_pl/toxic/checkpoints/*.json`
- `out_pl/wildguard/translated.jsonl`, `out_pl/wildguard/failed_rows.jsonl`, `out_pl/wildguard/checkpoints/*.json`

You can resume processing by running the translator again with the same parameters.
Already completed records are skipped.
For correct interactive `tqdm` rendering, run the translator with `docker compose run --rm translator`.

Runtime behavior:

- the translator uses structured output (`response_format=json_schema` when supported, with fallback to `json_object`) to enforce translation shape
- row-level failures do not stop the whole run by default; they are logged to `<out-dir>/<dataset_key>/failed_rows.jsonl`
- use `--fail-fast` to stop the entire run on the first failed row
- use `--failed-jsonl-name <name>` to change the failed-rows file name

## Architecture

`docker-compose.yml` starts two services:

- `vllm` - OpenAI-compatible endpoint at `http://vllm:8000/v1`
- `translator` - client service that:
  - reads dataset rows,
  - translates queries and documents,
  - preserves output/checkpoint format compatible with the existing workflow,
  - runs row processing concurrently via `asyncio` tasks (bounded by a semaphore),
  - serializes disk writes through a dedicated writer task (queue) to reduce worker blocking.

## Dependencies (uv + requirements)

In the translator image, dependencies are installed via `uv` from `requirements.txt`.
The source dependency file is `requirements.in`.

Update pinned dependencies:

```bash
uv pip compile requirements.in -o requirements.txt
```

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
