# ComBench

Code release for ComBench, an evaluation framework for Olympiad-level
combinatorics reasoning with rubric-guided proof judging and executable
construction verification.

This repository contains the code needed to run the evaluation pipeline and
the data-building utilities. The full benchmark data, human review materials,
model outputs, and large experimental caches are not included in this code-only
release.

## Contents

- `src/`: generation, parsing, judging, scoring, and verifier execution code.
- `pipeline/`: utilities and prompt templates for building ComBench-style JSONL
  records.
- `data_process/`: utilities for inspecting JSONL records and checking reference
  constructions.
- `profiles/`: model profile examples. Profiles reference API keys through
  environment variable names only.
- `examples/`: a small toy JSONL record for local smoke tests.
- `tests/`: unit tests for the evaluation framework.

## Setup

This project currently targets Python 3.10 or earlier.

```bash
python -m pip install -r requirements.txt
```

## Run Tests

```bash
python -m pytest tests
```

## Smoke Test

The mock profile does not call an external API. This command checks the
generation path only:

```bash
python -m src.main \
  --dataset examples/example.jsonl \
  --model-profile mock \
  --profiles-dir profiles \
  --output-root outputs_smoke \
  --generate \
  --verbose summary
```

The local verifier can be smoke-tested separately:

```bash
python data_process/check_ref_construction.py --file examples/example.jsonl --line 1
```

`outputs_smoke/` is ignored by git.

## Using LLM Profiles

LLM profiles are OpenAI-compatible chat-completions configurations. They should
not contain real API keys. Instead, set `api_key_env` to the name of an
environment variable:

```yaml
name: openai-compatible-example
type: llm
model_name: your-model-name
base_url: "https://api.example.com/v1/"
api_key_env: OPENAI_API_KEY
temperature: 0.6
timeout: 1200
max_retries: 8
```

Then set the key in your shell before running:

```bash
export OPENAI_API_KEY="..."
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="..."
```

## Dataset Format

Each JSONL record contains an IMO-style problem and optional construction
metadata. Plain proof-only records use fields such as:

- `id`
- `query`
- `ref_answer`
- `grading_guidelines`
- `ref_solution`

Construction-centric records additionally include:

- `instruction`
- `ref_construction`
- `verify_code`

When both `instruction` and `verify_code` are present, the evaluator expects a
two-part response and checks the construction payload with the verifier.

## Notes

- The complete ComBench dataset and paper result artifacts are intentionally not
  part of this repository.
- Large model outputs and evaluation caches should stay outside git.
- The local construction verifier uses the vendored `src/prime_code/` runtime.

