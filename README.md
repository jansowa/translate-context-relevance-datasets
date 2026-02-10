# Translate context relevance dataset

Repo contains the original script `translate_context_relevance_dataset.py` (left unchanged) and a new Docker Compose setup for local EN→PL translation through an OpenAI-compatible vLLM server.

## What is new

- Two services in `docker-compose.yml`:
  - `vllm` – OpenAI-compatible inference server (single model from `.env`),
  - `translator` – lightweight Python worker that reads data, translates, and saves JSONL/checkpoints.
- Single-model configuration (`MODEL_NAME`) instead of lists of API keys/models.
- Parallel requests configured from `.env` via `PARALLEL_REQUESTS`.
- NVIDIA multi-GPU support through `.env` `GPU_COUNT` (mapped to vLLM tensor parallel size).
- Dependencies managed with `uv`, `requirements.in`, `requirements.txt`.

## Files added for the new flow

- `run_translation_vllm.py` – threaded translator client (OpenAI-compatible).
- `docker-compose.yml` – `vllm` + `translator` services.
- `Dockerfile.translator` – small Python image for the translator.
- `requirements.in`, `requirements.txt` – dependency inputs/pins.
- `.env.example` – tiny model profile (`Qwen2.5-0.5B-Instruct`, 2 workers).
- `.env.gptoss` – large profile (`gpt-oss-120b`, 16 workers).

## Quick start (RTX 3060 Ti / 8GB VRAM)

1. Copy defaults:

```bash
cp .env.example .env
```

2. Start vLLM:

```bash
docker compose up -d vllm
```

3. Run translator:

```bash
docker compose up translator
```

Output will be written the same way as before:

- `out_pl/translated.jsonl`
- `out_pl/checkpoints/*.json`

## Environment variables (`.env`)

Required:

- `MODEL_NAME` – one model name used by both services.
- `PARALLEL_REQUESTS` – number of parallel translation requests from `translator`.
- `GPU_COUNT` – number of NVIDIA GPUs used by vLLM (`--tensor-parallel-size`).

Optional:

- `OPENAI_API_KEY` – token passed to OpenAI-compatible client (`EMPTY` is fine for local vLLM in many setups).
- `VLLM_PORT` – published host port for vLLM (default `8000`).
- `GPU_MEMORY_UTILIZATION` – vLLM memory fraction.
- `MAX_MODEL_LEN` – vLLM max sequence length.

## Translator behavior

`run_translation_vllm.py` preserves the original translation logic:

- same prompt structure,
- same span repair strategy,
- same checkpointing pattern,
- same JSONL row format.

Parallelization is done with `ThreadPoolExecutor`, and writes are serialized in a dedicated writer thread for better throughput under load.

## Dependency workflow with uv

Update pins after editing `requirements.in`:

```bash
uv pip compile requirements.in -o requirements.txt
```

Install locally:

```bash
uv pip install -r requirements.txt
```

## Original script

The existing script is intentionally kept as-is:

- `translate_context_relevance_dataset.py`

It still supports API key/model list rotation (`api_key × model`) and can be used independently if needed.
