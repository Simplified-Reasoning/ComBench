# Data Builder Pipeline

This directory contains utilities and prompt templates for building
ComBench-style JSONL records.

## Inputs

The staged prompt inputs live under:

- `pipeline/prompt_inputs/step1/`
- `pipeline/prompt_inputs/step2/`
- `pipeline/prompt_inputs/step3/`

Each stage uses plain text files for the problem id, query, reference answer,
grading guidelines, construction instruction, reference construction, and
verifier code as applicable.

## Prompt Templates

Prompt templates live under `pipeline/prompt_template/`:

- `step1_prompt_template.txt`
- `step2_prompt_template.txt`
- `verification_prompt_template.txt`

The CLI fills template placeholders from the staged input files.

## CLI

Run from the repository root:

```bash
python pipeline/data_builder_cli.py --help
```

Typical modes:

```bash
python pipeline/data_builder_cli.py --inst
python pipeline/data_builder_cli.py --code
python pipeline/data_builder_cli.py --verify
python pipeline/data_builder_cli.py --jsonl examples/example.jsonl
```

LLM-backed modes require an OpenAI-compatible profile under `profiles/` and an
API key provided through the environment variable named by `api_key_env`.

## Local Reference Check

After producing a JSONL record with `ref_construction` and `verify_code`, check
that the reference construction passes:

```bash
python data_process/check_ref_construction.py --file examples/example.jsonl --line 1
```

