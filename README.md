# Translate context relevance dataset (EN → PL)

This repository provides a tool for translating the HuggingFace dataset `zilliz/natural_questions-context-relevance-with-think` from English to Polish.

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

- `MODEL_NAME` – model name served by vLLM
- `PARALLEL_REQUESTS` – number of parallel requests on the translator side (`ThreadPoolExecutor`)
- `PROGRESS_BAR` – translation progress display mode: `on` (default), `auto` (TTY only), `off`
- `PROGRESS_METRIC` – progress metric for `tqdm`: `checkpoints` (default), `rows`, `both`
- `GPU_COUNT` – number of GPUs used by vLLM (`--tensor-parallel-size`)
- `VLLM_QUANTIZATION` (optional) – vLLM quantization mode; leave empty for full precision (for example `awq`)

Available profiles:

- `.env.example` – lightweight profile (defaults to `Qwen/Qwen2.5-0.5B-Instruct`, `PARALLEL_REQUESTS=2`, `GPU_COUNT=1`)
- `.env.gptoss` – multi-GPU profile (`openai/gpt-oss-120b`, `PARALLEL_REQUESTS=16`, `GPU_COUNT=4`)
- `.env.bielikq4` – Bielik 11B in 4-bit AWQ quantization (`speakleash/Bielik-11B-v3.0-Instruct-AWQ`, `VLLM_QUANTIZATION=awq`)


## Running

1. Start vLLM:

```bash
docker compose up -d --build vllm
```

2. Start translation:

```bash
docker compose run --rm translator
```

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
- make sure you are using the small model profile from `.env.example`

## Output and checkpoints

Output files are written inside the repository directory:

- `out_pl/translated.jsonl` – final output (1 record per line)
- `out_pl/checkpoints/*.json` – per-`id` checkpoints

You can resume processing by running the translator again with the same parameters.
Already completed records are skipped.
For correct interactive `tqdm` rendering, run the translator with `docker compose run --rm translator`.

## Architecture

`docker-compose.yml` starts two services:

- `vllm` – OpenAI-compatible endpoint at `http://vllm:8000/v1`
- `translator` – client service that:
  - reads dataset rows,
  - translates queries and documents,
  - preserves output/checkpoint format compatible with the existing workflow,
  - sends requests in parallel via `ThreadPoolExecutor`,
  - serializes disk writes through a dedicated writer thread (queue) to reduce worker blocking.

## Dependencies (uv + requirements)

In the translator image, dependencies are installed via `uv` from `requirements.txt`.
The source dependency file is `requirements.in`.

Update pinned dependencies:

```bash
uv pip compile requirements.in -o requirements.txt
```

## Original script

The repository still includes the original script:

- `translate_context_relevance_dataset.py`

It was not removed or modified.
