# Translate context relevance dataset

A script for translating the HuggingFace dataset `zilliz/natural_questions-context-relevance-with-think` from EN→PL.

It writes the output to JSONL and uses checkpoints so the process can be resumed.
It also supports a list of **API keys** and a list of **models**: for each `api_key × model` combination,
it sends requests until it receives a response associated with hitting the limit (most commonly HTTP **429**),
then automatically switches to the next combination.

> Note: The exact meaning of “limits” depends on the API provider. The script treats HTTP **429** (and closely related rate/quota errors) as “limit reached”.

---

## Requirements

- Python 3.10+ (recommended)
- Internet access (to download the dataset from HuggingFace)
- Libraries: `openai`, `datasets`, `tqdm`

---

## Setup (venv)

```bash
python -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install openai==1.64.0 datasets==3.3.2 tqdm==4.67.1
````

Optionally freeze dependencies:

```bash
pip freeze > requirements.txt
```

---

## Usage (translate dataset with checkpoints)

Minimal example:

```bash
python translate_context_relevance_dataset.py \
  --base-url "https://your-provider.example/v1" \
  --api-keys "KEY_1,KEY_2" \
  --models "modelA,modelB"
```

What you get:

* `out_pl/translated.jsonl` – final output (one record per line),
* `out_pl/checkpoints/*.json` – per-`id` checkpoints (allow resuming).

Resume:

* Run again with the same parameters; the script will skip `id`s already present in `translated.jsonl`
  and continue unfinished work based on checkpoints.

---

## Key CLI parameters

* `--api-keys` – list of keys (separator: `,` or `;`)
* `--models` – list of models (separator: `,` or `;`)
* `--base-url` – API base URL (optional)
* `--delay-seconds` – delay after each successful request
* `--max-retries` – retries for errors other than 429
* `--skip-rows` – skip the first N dataset examples (e.g., 5000 => start from example #5001)
* `--max-rows` – number of examples to process after skipping (0 = all remaining)
* `--max-prompt-attempts` – how many increasingly strict prompts to try when the model breaks span structure

---

## Notes

* The script detects “limit reached” mainly via HTTP 429 status / related exceptions.
* For non-limit errors, it retries with exponential backoff.
* When a given `api_key × model` hits a limit, the script switches to the next combination and continues from checkpoints.
* Existing lines in `translated.jsonl` are **not modified or removed**; new translations are appended as new lines.

```
::contentReference[oaicite:0]{index=0}
```
