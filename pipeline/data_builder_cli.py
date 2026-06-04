"""Data builder + LLM runner for dataset construction workflows.

This CLI reads prepared Markdown inputs, writes prompt Markdown files, calls an
LLM via src/ llm client/profile wiring, auto-populates next-step inputs, and
can append finalized step3 data into a target JSONL file:
1) step1 -> build prompt -> call LLM -> parse instruction/ref_construction/
   verify_code_informal_plan -> write step2/*.md
2) step2 -> build prompt -> call LLM -> parse verify_code -> write step3 inputs
   (`*.md` plus `verify_code.py`) -> run verification -> write verification
   prompt/result files
3) jsonl -> read step3 input files -> normalize fields -> append one record to
   JSONL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import shlex
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any, Callable, TypeVar

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.construction_text import is_json_or_python_text
from src.llm.client import DEFAULT_MAX_RETRIES
from src.models.llm.model import build_llm_client
from src.models.profile import load_profile

INPUT_ROOT = Path("pipeline/prompt_inputs")
OUTPUT_ROOT = Path("pipeline/prompts")
TEMPLATE_ROOT = Path("pipeline/prompt_template")
DEFAULT_MODEL_PROFILE = "gemini"
DEFAULT_PROFILES_DIR = Path("profiles")
DEFAULT_STEP3_INPUT_DIR = INPUT_ROOT / "step3"
DEFAULT_DATA_DIR = ROOT / "data"
VERIFICATION_DECISIONS = {"pass", "step1_issue", "step2_issue"}

STEP1_INPUT_FILES = {
    "id": INPUT_ROOT / "step1/id.md",
    "grading_guidelines": INPUT_ROOT / "step1/grading_guidelines.md",
    "query": INPUT_ROOT / "step1/query.md",
    "ref_answer": INPUT_ROOT / "step1/ref_answer.md",
    "intent": INPUT_ROOT / "step1/intent.md",
}
STEP1_VERIFICATION_INPUT_FILES = {
    "construction_goal": INPUT_ROOT / "step1/intent.md",
}
STEP2_INPUT_FILES = {
    "id": INPUT_ROOT / "step2/id.md",
    "grading_guidelines": INPUT_ROOT / "step2/grading_guidelines.md",
    "query": INPUT_ROOT / "step2/query.md",
    "ref_answer": INPUT_ROOT / "step2/ref_answer.md",
    "instruction": INPUT_ROOT / "step2/instruction.md",
    "ref_construction": INPUT_ROOT / "step2/ref_construction.md",
    "intent": INPUT_ROOT / "step2/intent.md",
}

STEP2_AUTOFILL_FILES = {
    "id": INPUT_ROOT / "step2/id.md",
    "grading_guidelines": INPUT_ROOT / "step2/grading_guidelines.md",
    "query": INPUT_ROOT / "step2/query.md",
    "ref_answer": INPUT_ROOT / "step2/ref_answer.md",
    "instruction": INPUT_ROOT / "step2/instruction.md",
    "ref_construction": INPUT_ROOT / "step2/ref_construction.md",
    "intent": INPUT_ROOT / "step2/intent.md",
}
STEP3_AUTOFILL_FILES = {
    "id": INPUT_ROOT / "step3/id.md",
    "grading_guidelines": INPUT_ROOT / "step3/grading_guidelines.md",
    "query": INPUT_ROOT / "step3/query.md",
    "ref_answer": INPUT_ROOT / "step3/ref_answer.md",
    "instruction": INPUT_ROOT / "step3/instruction.md",
    "ref_construction": INPUT_ROOT / "step3/ref_construction.md",
    "verify_code": INPUT_ROOT / "step3/verify_code.py",
}
STEP3_FIELD_FILES: dict[str, str] = {
    "id": "id.md",
    "query": "query.md",
    "ref_answer": "ref_answer.md",
    "instruction": "instruction.md",
    "ref_construction": "ref_construction.md",
    "verify_code": "verify_code.py",
}

STEP1_OUTPUT_FILE = OUTPUT_ROOT / "step1_prompt.md"
STEP2_OUTPUT_FILE = OUTPUT_ROOT / "step2_prompt.md"
VERIFICATION_OUTPUT_FILE = OUTPUT_ROOT / "verification_prompt.md"
VERIFICATION_RESULT_FILE = OUTPUT_ROOT / "verification_result.md"
STEP1_TEMPLATE_FILE = TEMPLATE_ROOT / "step1_prompt_template.txt"
STEP2_TEMPLATE_FILE = TEMPLATE_ROOT / "step2_prompt_template.txt"
VERIFICATION_TEMPLATE_FILE = TEMPLATE_ROOT / "verification_prompt_template.txt"
NO_FEW_SHOT_PLACEHOLDER = "(no few-shot examples)"
T = TypeVar("T")
console = Console()


def _fmt_path(path: Path | str) -> str:
    return f"[cyan]{path}[/cyan]"


def _print_stage_banner(title: str, subtitle: str | None = None, style: str = "blue") -> None:
    content = Text(title, style=f"bold {style}")
    if subtitle:
        content.append("\n")
        content.append_text(Text.from_markup(f"[dim]{subtitle}[/dim]"))
    console.print(Panel.fit(content, border_style=style))


def _print_kv(label: str, value: str, value_style: str = "default") -> None:
    console.print(f"[bold]{label}:[/bold] [{value_style}]{value}[/{value_style}]")


def _print_note(message: str) -> None:
    console.print(f"[dim]{message}[/dim]")


def _print_success(message: str) -> None:
    console.print(f"[green]{message}[/green]")


def _print_warning(message: str) -> None:
    console.print(f"[yellow]{message}[/yellow]")


def _print_error(message: str) -> None:
    console.print(f"[bold red]{message}[/bold red]")


def _build_file_status_table(title: str) -> Table:
    table = Table(title=title, header_style="bold magenta", box=None, pad_edge=False)
    table.add_column("Field", style="bold")
    table.add_column("Path", overflow="fold")
    table.add_column("Status", no_wrap=True)
    table.add_column("Info", style="dim")
    return table


def _read_required_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing file: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"not a regular file: {path}")
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"empty file: {path}")
    return content


def _read_template_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"missing template file: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"template path is not a regular file: {path}")
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError(f"empty template file: {path}")
    return content.strip()


def _apply_template(text: str, mapping: dict[str, str], template_path: Path) -> str:
    rendered = text
    for key, value in mapping.items():
        token = f"<<{key.upper()}>>"
        if token not in rendered:
            raise ValueError(f"template missing token {token}: {template_path}")
        rendered = rendered.replace(token, value)
    return rendered


def _validate_id_text(item_id: str, path: Path) -> None:
    if "\n" in item_id:
        raise ValueError(f"id must be single-line: {path}")


def _load_step_inputs(step_name: str, mapping: dict[str, Path]) -> dict[str, str]:
    _print_stage_banner(
        f"Read {step_name.upper()} Inputs",
        "Checking required input files before running the stage.",
        style="blue",
    )
    result: dict[str, str] = {}
    errors: list[str] = []
    table = _build_file_status_table(f"{step_name} input files")

    for key, path in mapping.items():
        try:
            value = _read_required_file(path)
            if key == "id":
                _validate_id_text(value, path)
        except (FileNotFoundError, IsADirectoryError, OSError, ValueError) as exc:
            table.add_row(key, _fmt_path(path), "[red]ERROR[/red]", str(exc))
            errors.append(str(exc))
            continue

        table.add_row(key, _fmt_path(path), "[green]OK[/green]", f"{len(value)} chars")
        result[key] = value

    console.print(table)
    if errors:
        raise ValueError(
            "Input files are not ready. Please fix the files shown above and retry."
        )
    return result


def _parse_few_shot_spec(spec: str) -> tuple[Path, str]:
    path_part, sep, item_id = spec.rpartition(":")
    if sep == "":
        raise ValueError(
            "few-shot spec must be in format <jsonl_path>:<id>; "
            f"got: {spec!r}"
        )

    path_text = path_part.strip()
    item_id = item_id.strip()
    if not path_text:
        raise ValueError(f"few-shot jsonl path is empty in spec: {spec!r}")
    if not item_id:
        raise ValueError(f"few-shot id is empty in spec: {spec!r}")
    return Path(path_text), item_id


def _resolve_few_shot_specs(extra_spec_groups: list[list[str]]) -> list[str]:
    ordered_specs: list[str] = []
    for group in extra_spec_groups:
        for spec in group:
            text = spec.strip()
            if text:
                ordered_specs.append(text)

    deduped: list[str] = []
    seen: set[str] = set()
    for spec in ordered_specs:
        if spec in seen:
            continue
        seen.add(spec)
        deduped.append(spec)

    return deduped


def _require_non_empty_field(
    item: dict[str, object],
    key: str,
    item_id: str,
    jsonl_path: Path,
) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"few-shot item {item_id!r} has missing/empty {key!r}: {jsonl_path}"
        )
    return value.strip()


def _coerce_non_empty_text(
    item: dict[str, object],
    key: str,
    item_id: str,
    jsonl_path: Path,
) -> str:
    value = item.get(key)
    if isinstance(value, str):
        text = value.strip()
    elif value is None:
        text = ""
    else:
        # Keep non-string JSON values usable in prompts (e.g. list construction).
        text = json.dumps(value, ensure_ascii=False)
    if not text:
        raise ValueError(
            f"few-shot item {item_id!r} has missing/empty {key!r}: {jsonl_path}"
        )
    return text


def _load_few_shot_item(spec: str) -> dict[str, str]:
    jsonl_path, target_id = _parse_few_shot_spec(spec)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"missing few-shot jsonl file: {jsonl_path}")
    if not jsonl_path.is_file():
        raise IsADirectoryError(f"few-shot path is not a regular file: {jsonl_path}")

    found: dict[str, object] | None = None
    for line_no, raw_line in enumerate(
        jsonl_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSONL at {jsonl_path}:{line_no}: {exc.msg}"
            ) from exc

        if not isinstance(item, dict):
            raise ValueError(
                f"invalid JSON object at {jsonl_path}:{line_no}; expected object"
            )

        if str(item.get("id")) != target_id:
            continue

        if found is not None:
            raise ValueError(
                f"duplicate id {target_id!r} found in few-shot jsonl: {jsonl_path}"
            )
        found = item

    if found is None:
        raise ValueError(
            f"few-shot id {target_id!r} not found in jsonl file: {jsonl_path}"
        )

    query = _require_non_empty_field(found, "query", target_id, jsonl_path)
    ref_answer = _require_non_empty_field(found, "ref_answer", target_id, jsonl_path)
    instruction = _require_non_empty_field(found, "instruction", target_id, jsonl_path)
    ref_construction = _coerce_non_empty_text(
        found,
        "ref_construction",
        target_id,
        jsonl_path,
    )

    console.print(
        f"[bold blue]Few-shot[/bold blue] {_fmt_path(spec)} "
        f"[green]OK[/green] [dim]"
        f"query={len(query)} chars, ref_answer={len(ref_answer)} chars, "
        f"instruction={len(instruction)} chars, "
        f"ref_construction={len(ref_construction)} chars[/dim]"
    )

    return {
        "item_id": target_id,
        "query": query,
        "ref_answer": ref_answer,
        "instruction": instruction,
        "ref_construction": ref_construction,
    }


def _load_few_shot_items(specs: list[str]) -> list[dict[str, str]]:
    if not specs:
        return []
    return [_load_few_shot_item(spec) for spec in specs]


def _format_few_shot_examples_xml(items: list[dict[str, str]]) -> str:
    if not items:
        return NO_FEW_SHOT_PLACEHOLDER

    chunks: list[str] = []
    for index, item in enumerate(items, start=1):
        chunk = (
            f"### Example {index}\n"
            "<query>\n"
            f"{item['query']}\n"
            "</query>\n"
            "<ref_answer>\n"
            f"{item['ref_answer']}\n"
            "</ref_answer>\n"
            "<instruction>\n"
            f"{item['instruction']}\n"
            "</instruction>\n"
            "<ref_construction>\n"
            f"{item['ref_construction']}\n"
            "</ref_construction>"
        )
        chunks.append(chunk)

    return "\n\n".join(chunks)


def _build_step1_prompt(
    query: str,
    ref_answer: str,
    intent: str,
    few_shot_examples: str,
) -> str:
    template = _read_template_file(STEP1_TEMPLATE_FILE)
    return _apply_template(
        template,
        {
            "query": query,
            "ref_answer": ref_answer,
            "intent": intent,
            "few_shot_examples": few_shot_examples,
        },
        STEP1_TEMPLATE_FILE,
    )


def _build_step2_prompt(
    query: str,
    ref_answer: str,
    instruction: str,
    ref_construction: str,
    intent: str,
    few_shot_examples: str,
) -> str:
    template = _read_template_file(STEP2_TEMPLATE_FILE)
    return _apply_template(
        template,
        {
            "query": query,
            "ref_answer": ref_answer,
            "instruction": instruction,
            "ref_construction": ref_construction,
            "intent": intent,
            "few_shot_examples": few_shot_examples,
        },
        STEP2_TEMPLATE_FILE,
    )


def _build_verification_prompt(
    query: str,
    ref_answer: str,
    construction_goal: str,
    instruction: str,
    ref_construction: str,
    verify_code_plan: str,
    verify_code: str,
) -> str:
    template = _read_template_file(VERIFICATION_TEMPLATE_FILE)
    return _apply_template(
        template,
        {
            "query": query,
            "ref_answer": ref_answer,
            "construction_goal": construction_goal,
            "instruction": instruction,
            "ref_construction": ref_construction,
            "verify_code_plan": verify_code_plan,
            "verify_code": verify_code,
        },
        VERIFICATION_TEMPLATE_FILE,
    )


def _render_markdown(prompt_text: str) -> str:
    # Output file should be directly usable as a complete prompt.
    return f"{prompt_text.strip()}\n"


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_fields(mapping: dict[str, Path], payload: dict[str, str], title: str) -> None:
    _print_stage_banner(
        f"Write {title}",
        "Persisting generated artifacts to output files.",
        style="green",
    )
    table = _build_file_status_table(title)
    for key, path in mapping.items():
        if key not in payload:
            raise ValueError(f"missing payload field {key!r} for writing {title}")
        text = payload[key].strip()
        if not text:
            raise ValueError(f"empty payload field {key!r} for writing {title}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{text}\n", encoding="utf-8")
        table.add_row(key, _fmt_path(path), "[green]OK[/green]", f"{len(text)} chars")
    console.print(table)


def _unwrap_exact_xml_block(text: str, tag: str) -> str:
    pattern = re.compile(rf"^\s*<{tag}>\s*(.*?)\s*</{tag}>\s*$", re.IGNORECASE | re.DOTALL)
    match = pattern.match(text)
    if not match:
        return text
    inner = match.group(1).strip()
    if not inner:
        raise ValueError(f"empty <{tag}>...</{tag}> block")
    return inner


def _extract_construct_block(text: str) -> str:
    pattern = re.compile(r"<construct>\s*(.*?)\s*</construct>", re.IGNORECASE | re.DOTALL)
    matches = list(pattern.finditer(text))
    if not matches:
        return text.strip()
    if len(matches) > 1:
        warnings.warn(
            "multiple <construct>...</construct> blocks found; using the first one.",
            stacklevel=2,
        )
    return matches[0].group(1).strip()


def _strip_markdown_code_fence(text: str) -> str:
    pattern = re.compile(r"^\s*```(?:python|json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)
    match = pattern.match(text)
    if not match:
        return text
    return match.group(1).strip()


def _normalize_ref_construction(text: str) -> str:
    # Accept either raw payload or wrapped XML blocks and remove markup.
    payload = _unwrap_exact_xml_block(text, "construct")
    payload = _extract_construct_block(payload)
    payload = _strip_markdown_code_fence(payload).strip()
    if is_json_or_python_text(payload):
        return payload
    warnings.warn(
        "ref_construction parse check failed: not valid JSON nor Python literal; "
        "it will still be written as a raw string.",
        stacklevel=2,
    )
    return payload


def _load_step3_record(input_dir: Path) -> dict[str, str]:
    _print_stage_banner(
        "Read STEP3 Inputs",
        f"Loading finalized step3 input files from {_fmt_path(input_dir)}.",
        style="blue",
    )

    raw: dict[str, str] = {}
    table = _build_file_status_table("step3 input files")
    for key, filename in STEP3_FIELD_FILES.items():
        path = input_dir / filename
        value = _read_required_file(path)
        raw[key] = value
        table.add_row(key, _fmt_path(path), "[green]OK[/green]", f"{len(value)} chars")
    console.print(table)

    item_id = raw["id"].strip()
    _validate_id_text(item_id, input_dir / STEP3_FIELD_FILES["id"])

    instruction = _unwrap_exact_xml_block(raw["instruction"], "instruction")
    ref_construction = _unwrap_exact_xml_block(
        raw["ref_construction"], "ref_construction"
    )
    ref_construction = _normalize_ref_construction(ref_construction)
    verify_code = _unwrap_exact_xml_block(raw["verify_code"], "verify_code")

    return {
        "id": item_id,
        "query": raw["query"],
        "ref_answer": raw["ref_answer"],
        "instruction": instruction,
        "ref_construction": ref_construction,
        "verify_code": verify_code,
    }


def _resolve_output_path(output_jsonl: str) -> Path:
    out = Path(output_jsonl)
    if out.suffix.lower() != ".jsonl":
        raise ValueError("output file must end with .jsonl")

    if out.is_absolute():
        return out

    if out.parent == Path("."):
        return DEFAULT_DATA_DIR / out

    return out


def _check_duplicate_id(path: Path, target_id: str) -> None:
    if not path.exists():
        return

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc.msg}") from exc

        if not isinstance(item, dict):
            raise ValueError(f"invalid JSON object at {path}:{line_no}; expected object")

        if str(item.get("id")) == target_id:
            raise ValueError(f"id {target_id!r} already exists in: {path}")


def _append_record(path: Path, record: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


def _handle_jsonl(output_jsonl: str, input_dir: Path, allow_duplicate_id: bool) -> None:
    _print_stage_banner(
        "Assemble JSONL Record",
        "Reading step3 fields, normalizing them, and appending one dataset row.",
        style="blue",
    )
    output_path = _resolve_output_path(output_jsonl)
    record = _load_step3_record(input_dir)

    if not allow_duplicate_id:
        _check_duplicate_id(output_path, str(record["id"]))

    existed = output_path.exists()
    _append_record(output_path, record)
    if not existed:
        _print_success(f"Created new JSONL file: {output_path}")
    _print_success(f"Appended 1 item to: {output_path}")
    _print_kv("id", str(record["id"]), "green")


def _resolve_retry_budget(value: Any) -> int:
    retry_budget = DEFAULT_MAX_RETRIES if value is None else value
    if not isinstance(retry_budget, int):
        raise ValueError("llm profile field max_retries must be an integer")
    if retry_budget < 1:
        raise ValueError("llm profile field max_retries must be >= 1")
    return retry_budget


def _build_llm_client_from_profile(profile_name: str) -> tuple[str, Any, int]:
    loaded_name, model_type, params = load_profile(profile_name, DEFAULT_PROFILES_DIR)
    if model_type != "llm":
        raise ValueError(
            f"profile {loaded_name!r} must have type=llm for pipeline automation"
        )

    required = {"model_name", "api_key_env", "base_url"}
    missing = [key for key in required if key not in params]
    if missing:
        raise ValueError(f"llm profile missing required fields: {missing}")
    if not params.get("model_name") or not params.get("api_key_env"):
        raise ValueError("llm profile requires non-empty model_name and api_key_env")

    retry_budget = _resolve_retry_budget(params.get("max_retries"))
    kwargs = {k: v for k, v in params.items() if k not in required | {"max_retries"}}
    client = build_llm_client(
        model_name=params["model_name"],
        api_key_env=params["api_key_env"],
        base_url=params["base_url"],
        max_retries=1,
        **kwargs,
    )
    return loaded_name, client, retry_budget


def _run_chat(client: Any, prompt: str) -> dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    return asyncio.run(client.chat(messages))


def _coerce_message_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
                    continue
        return "\n".join(p for p in parts if p).strip()
    return ""


def _extract_chat_text(raw: dict[str, Any]) -> str:
    if not isinstance(raw, dict):
        raise ValueError("LLM response is not a JSON object")

    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("LLM response missing choices")

    first = choices[0]
    if not isinstance(first, dict):
        raise ValueError("LLM response has invalid first choice")

    message = first.get("message")
    if not isinstance(message, dict):
        raise ValueError("LLM response missing choice.message")

    text = _coerce_message_content(message.get("content"))
    if not text:
        raise ValueError("LLM response has empty choice.message.content")
    return text


def _extract_single_tag(text: str, tag: str) -> str:
    pattern = re.compile(rf"<{tag}>\s*(.*?)\s*</{tag}>", re.IGNORECASE | re.DOTALL)
    matches = [m.group(1).strip() for m in pattern.finditer(text or "")]
    if not matches:
        raise ValueError(f"LLM output missing <{tag}>...</{tag}> block")
    if len(matches) > 1:
        raise ValueError(f"LLM output has multiple <{tag}>...</{tag}> blocks")
    if not matches[0]:
        raise ValueError(f"LLM output has empty <{tag}>...</{tag}> block")
    return matches[0]


def _parse_step1_result(text: str) -> dict[str, str]:
    instruction = _extract_single_tag(text, "instruction")
    ref_construction = _normalize_ref_construction(
        _extract_single_tag(text, "ref_construction")
    )
    verify_plan = _extract_single_tag(text, "verify_code_informal_plan")
    return {
        "instruction": instruction,
        "ref_construction": ref_construction,
        "verify_code_informal_plan": verify_plan,
    }


def _extract_verify_code(text: str) -> str:
    code_pattern = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
    matches = [m.group(1).strip() for m in code_pattern.finditer(text or "") if m.group(1).strip()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError("LLM output has multiple fenced code blocks for verify_code")

    # Fallback for occasional XML wrapping.
    return _extract_single_tag(text, "verify_code")


def _parse_verification_result(text: str) -> dict[str, str]:
    decision = _extract_single_tag(text, "decision")
    if decision not in VERIFICATION_DECISIONS:
        raise ValueError(
            "verification decision must be one of "
            f"{sorted(VERIFICATION_DECISIONS)}; got: {decision!r}"
        )

    return {
        "decision": decision,
        "summary": _extract_single_tag(text, "summary"),
        "details": _extract_single_tag(text, "details"),
        "suggestion": _extract_single_tag(text, "suggestion"),
    }


def _call_llm_with_retry_and_parse_with_text(
    prompt_text: str,
    llm_client: Any,
    profile_name: str,
    stage_name: str,
    parse_fn: Callable[[str], T],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[str, T]:
    if max_retries < 1:
        raise ValueError("max_retries must be >= 1")

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        _print_stage_banner(
            f"LLM Call: {stage_name}",
            f"profile={profile_name} | attempt {attempt}/{max_retries}",
            style="magenta",
        )
        try:
            raw = _run_chat(llm_client, prompt_text)
            text = _extract_chat_text(raw)
            return text, parse_fn(text)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                _print_warning(
                    f"LLM call/parse failed for {stage_name}: {exc}. Retrying..."
                )
                continue
            break

    assert last_exc is not None
    raise RuntimeError(
        f"LLM call/parse failed for {stage_name} after {max_retries} attempts: {last_exc}"
    ) from last_exc


def _call_llm_with_retry_and_parse(
    prompt_text: str,
    llm_client: Any,
    profile_name: str,
    stage_name: str,
    parse_fn: Callable[[str], T],
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> T:
    _, parsed = _call_llm_with_retry_and_parse_with_text(
        prompt_text=prompt_text,
        llm_client=llm_client,
        profile_name=profile_name,
        stage_name=stage_name,
        parse_fn=parse_fn,
        max_retries=max_retries,
    )
    return parsed


def _autofill_step2_from_step1(step1_payload: dict[str, str], parsed: dict[str, str]) -> None:
    payload = {
        "id": step1_payload["id"],
        "grading_guidelines": step1_payload["grading_guidelines"],
        "query": step1_payload["query"],
        "ref_answer": step1_payload["ref_answer"],
        "instruction": parsed["instruction"],
        "ref_construction": parsed["ref_construction"],
        "intent": parsed["verify_code_informal_plan"],
    }
    _write_fields(STEP2_AUTOFILL_FILES, payload, "step2 autofill files")


def _autofill_step3_from_step2(step2_payload: dict[str, str], verify_code: str) -> None:
    payload = {
        "id": step2_payload["id"],
        "grading_guidelines": step2_payload["grading_guidelines"],
        "query": step2_payload["query"],
        "ref_answer": step2_payload["ref_answer"],
        "instruction": step2_payload["instruction"],
        "ref_construction": step2_payload["ref_construction"],
        "verify_code": verify_code,
    }
    _write_fields(STEP3_AUTOFILL_FILES, payload, "step3 autofill files")


def _load_step1_verification_inputs() -> dict[str, str]:
    _print_stage_banner(
        "Read Verification Context",
        "Loading original construction goal from step1 for end-to-end auditing.",
        style="blue",
    )
    result: dict[str, str] = {}
    table = _build_file_status_table("verification context")
    for key, path in STEP1_VERIFICATION_INPUT_FILES.items():
        value = _read_required_file(path)
        table.add_row(key, _fmt_path(path), "[green]OK[/green]", f"{len(value)} chars")
        result[key] = value
    console.print(table)
    return result


def _print_verification_summary(parsed: dict[str, str]) -> None:
    style = {
        "pass": "green",
        "step1_issue": "yellow",
        "step2_issue": "yellow",
    }.get(parsed["decision"], "white")
    summary = Table.grid(padding=(0, 1))
    summary.add_column(style="bold")
    summary.add_column()
    summary.add_row("Decision", f"[{style}]{parsed['decision']}[/{style}]")
    summary.add_row("Summary", parsed["summary"])
    summary.add_row("Details", parsed["details"])
    summary.add_row("Suggestion", parsed["suggestion"])
    console.print(Panel(summary, title="Verification", border_style=style))


def _print_help() -> None:
    step1_id = STEP1_INPUT_FILES["id"]
    step1_grading_guidelines = STEP1_INPUT_FILES["grading_guidelines"]
    step1_query = STEP1_INPUT_FILES["query"]
    step1_ref_answer = STEP1_INPUT_FILES["ref_answer"]
    step1_intent = STEP1_INPUT_FILES["intent"]

    step2_id = STEP2_INPUT_FILES["id"]
    step2_grading_guidelines = STEP2_INPUT_FILES["grading_guidelines"]
    step2_query = STEP2_INPUT_FILES["query"]
    step2_ref_answer = STEP2_INPUT_FILES["ref_answer"]
    step2_instruction = STEP2_INPUT_FILES["instruction"]
    step2_ref_construction = STEP2_INPUT_FILES["ref_construction"]
    step2_intent = STEP2_INPUT_FILES["intent"]

    body = textwrap.dedent(
        f"""
        [bold]Commands[/bold]
          [cyan]help[/cyan]
            Show this help message.

          [cyan]step1[/cyan]
            Read step1 fixed files, generate prompt at {_fmt_path(STEP1_OUTPUT_FILE)},
            call LLM, parse XML, and autofill step2/*.md.

          [cyan]step2[/cyan]
            Read step2 fixed files, generate prompt at {_fmt_path(STEP2_OUTPUT_FILE)},
            call LLM, parse verify_code, autofill step3 inputs, then run verification and write:
              - {_fmt_path(VERIFICATION_OUTPUT_FILE)}
              - {_fmt_path(VERIFICATION_RESULT_FILE)}

          [cyan]exit[/cyan] | [cyan]quit[/cyan]
            Leave the interactive CLI.

        [bold]Fixed input files for step1[/bold]
          - {_fmt_path(step1_id)}
          - {_fmt_path(step1_grading_guidelines)}
          - {_fmt_path(step1_query)}
          - {_fmt_path(step1_ref_answer)}
          - {_fmt_path(step1_intent)}

        [bold]Fixed input files for step2[/bold]
          - {_fmt_path(step2_id)}
          - {_fmt_path(step2_grading_guidelines)}
          - {_fmt_path(step2_query)}
          - {_fmt_path(step2_ref_answer)}
          - {_fmt_path(step2_instruction)}
          - {_fmt_path(step2_ref_construction)}
          - {_fmt_path(step2_intent)}

        [bold]Prompt template files[/bold]
          - {_fmt_path(STEP1_TEMPLATE_FILE)}
          - {_fmt_path(STEP2_TEMPLATE_FILE)}
          - {_fmt_path(VERIFICATION_TEMPLATE_FILE)}
        """
    ).strip()
    console.print(Panel.fit(body, title="Data Builder Help", border_style="blue"))


def _handle_step1(
    few_shot_specs: list[str],
    llm_client: Any,
    profile_name: str,
    retry_budget: int = DEFAULT_MAX_RETRIES,
) -> None:
    _print_stage_banner(
        "STEP1",
        "Generate instruction, reference construction, and verifier design notes.",
        style="blue",
    )
    payload = _load_step_inputs("step1", STEP1_INPUT_FILES)
    few_shot_items = _load_few_shot_items(few_shot_specs)
    few_shot_examples = _format_few_shot_examples_xml(few_shot_items)

    prompt = _build_step1_prompt(
        query=payload["query"],
        ref_answer=payload["ref_answer"],
        intent=payload["intent"],
        few_shot_examples=few_shot_examples,
    )
    markdown = _render_markdown(prompt)
    _write_markdown(STEP1_OUTPUT_FILE, markdown)
    _print_success(f"Prompt written to: {STEP1_OUTPUT_FILE}")

    parsed = _call_llm_with_retry_and_parse(
        prompt_text=prompt,
        llm_client=llm_client,
        profile_name=profile_name,
        stage_name="step1",
        parse_fn=_parse_step1_result,
        max_retries=retry_budget,
    )
    _autofill_step2_from_step1(payload, parsed)
    console.print(
        Panel.fit(
            "Step1 completed. You can now manually adjust files under "
            f"{_fmt_path(INPUT_ROOT / 'step2')}.",
            border_style="green",
            title="STEP1 Complete",
        )
    )


def _handle_step2(
    few_shot_specs: list[str],
    llm_client: Any,
    profile_name: str,
    retry_budget: int = DEFAULT_MAX_RETRIES,
) -> None:
    _print_stage_banner(
        "STEP2",
        "Generate verify_code, populate step3 artifacts, then run semantic verification.",
        style="blue",
    )
    payload = _load_step_inputs("step2", STEP2_INPUT_FILES)
    verification_context = _load_step1_verification_inputs()
    few_shot_items = _load_few_shot_items(few_shot_specs)
    few_shot_examples = _format_few_shot_examples_xml(few_shot_items)

    prompt = _build_step2_prompt(
        query=payload["query"],
        ref_answer=payload["ref_answer"],
        instruction=payload["instruction"],
        ref_construction=payload["ref_construction"],
        intent=payload["intent"],
        few_shot_examples=few_shot_examples,
    )
    markdown = _render_markdown(prompt)
    _write_markdown(STEP2_OUTPUT_FILE, markdown)
    _print_success(f"Prompt written to: {STEP2_OUTPUT_FILE}")

    verify_code = _call_llm_with_retry_and_parse(
        prompt_text=prompt,
        llm_client=llm_client,
        profile_name=profile_name,
        stage_name="step2",
        parse_fn=_extract_verify_code,
        max_retries=retry_budget,
    )
    _autofill_step3_from_step2(payload, verify_code)

    verification_prompt = _build_verification_prompt(
        query=payload["query"],
        ref_answer=payload["ref_answer"],
        construction_goal=verification_context["construction_goal"],
        instruction=payload["instruction"],
        ref_construction=payload["ref_construction"],
        verify_code_plan=payload["intent"],
        verify_code=verify_code,
    )
    verification_markdown = _render_markdown(verification_prompt)
    _write_markdown(VERIFICATION_OUTPUT_FILE, verification_markdown)
    _print_success(f"Prompt written to: {VERIFICATION_OUTPUT_FILE}")

    verification_text, verification_result = _call_llm_with_retry_and_parse_with_text(
        prompt_text=verification_prompt,
        llm_client=llm_client,
        profile_name=profile_name,
        stage_name="verification",
        parse_fn=_parse_verification_result,
        max_retries=retry_budget,
    )
    _write_markdown(VERIFICATION_RESULT_FILE, _render_markdown(verification_text))
    _print_success(f"Verification result written to: {VERIFICATION_RESULT_FILE}")
    _print_verification_summary(verification_result)
    console.print(
        Panel.fit(
            "Step2 completed. Step3 files are ready under "
            f"{_fmt_path(INPUT_ROOT / 'step3')}.",
            border_style="green",
            title="STEP2 Complete",
        )
    )


def _normalize_command(parts: list[str]) -> str:
    if not parts:
        return ""
    head = parts[0].lower()
    if head in {"step1", "step2", "help", "exit", "quit"}:
        return head
    if head == "step" and len(parts) >= 2:
        if parts[1] == "1":
            return "step1"
        if parts[1] == "2":
            return "step2"
    return head


def _run_repl(
    llm_client: Any,
    profile_name: str,
    default_specs: list[str],
    retry_budget: int,
) -> None:
    console.print(
        Panel.fit(
            "Dataset data builder interactive CLI\nType [cyan]help[/cyan] to see commands.",
            border_style="blue",
            title="Pipeline CLI",
        )
    )

    while True:
        try:
            line = input("\npipeline> ").strip()
        except EOFError:
            _print_note("Bye.")
            return
        except KeyboardInterrupt:
            _print_warning("Interrupted. Type exit to quit.")
            continue

        if not line:
            continue

        try:
            parts = shlex.split(line)
        except ValueError as exc:
            _print_error(f"Failed to parse command: {exc}")
            continue

        command = _normalize_command(parts)

        if command in {"exit", "quit"}:
            _print_note("Bye.")
            return
        if command == "help":
            _print_help()
            continue

        try:
            if command == "step1":
                _handle_step1(default_specs, llm_client, profile_name, retry_budget=retry_budget)
            elif command == "step2":
                _handle_step2(default_specs, llm_client, profile_name, retry_budget=retry_budget)
            else:
                _print_error(f"Unknown command: {line}")
                _print_note("Type help to see available commands.")
        except Exception as exc:  # pragma: no cover - defensive for interactive CLI
            _print_error(f"Command failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CLI that reads fixed input files, writes LLM prompts, calls an "
            "LLM, auto-populates next-step input files, and can append one "
            "step3 item into JSONL."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--inst",
        action="store_true",
        help=(
            "Generate step1 prompt from fixed step1 input files, call LLM, "
            "and autofill step2 files (no interactive CLI)."
        ),
    )
    mode.add_argument(
        "--code",
        action="store_true",
        help=(
            "Generate step2 prompt from fixed step2 input files, call LLM, "
            "autofill step3 files, and run verification (no interactive CLI)."
        ),
    )
    mode.add_argument(
        "--jsonl",
        metavar="OUTPUT_JSONL",
        help=(
            "Read step3 input files, build one JSON object, and append it to "
            "OUTPUT_JSONL (no interactive CLI)."
        ),
    )
    parser.add_argument(
        "--few-shot",
        action="append",
        nargs="+",
        default=[],
        help=(
            "Add one or more few-shot specs in format <jsonl_path>:<id>. "
            "By default no few-shot examples are used. You can pass multiple "
            "values in one flag or repeat the flag."
        ),
    )
    parser.add_argument(
        "--model-profile",
        default=DEFAULT_MODEL_PROFILE,
        help=(
            "LLM profile name used for automatic generation "
            f"(default: {DEFAULT_MODEL_PROFILE})."
        ),
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_STEP3_INPUT_DIR),
        help=(
            "Directory containing step3 input files for --jsonl mode "
            f"(default: {DEFAULT_STEP3_INPUT_DIR})."
        ),
    )
    parser.add_argument(
        "--allow-duplicate-id",
        action="store_true",
        help="Allow --jsonl mode to append even when the same id already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.jsonl:
        try:
            _handle_jsonl(
                output_jsonl=args.jsonl,
                input_dir=Path(args.input_dir),
                allow_duplicate_id=args.allow_duplicate_id,
            )
        except Exception as exc:
            _print_error(f"Command failed: {exc}")
            raise SystemExit(1) from exc
        return

    few_shot_specs = _resolve_few_shot_specs(args.few_shot)
    profile_name, llm_client, retry_budget = _build_llm_client_from_profile(
        profile_name=args.model_profile,
    )

    if args.inst:
        try:
            _handle_step1(few_shot_specs, llm_client, profile_name, retry_budget=retry_budget)
        except Exception as exc:
            _print_error(f"Command failed: {exc}")
            raise SystemExit(1) from exc
        return

    if args.code:
        try:
            _handle_step2(few_shot_specs, llm_client, profile_name, retry_budget=retry_budget)
        except Exception as exc:
            _print_error(f"Command failed: {exc}")
            raise SystemExit(1) from exc
        return

    _run_repl(llm_client, profile_name, few_shot_specs, retry_budget)


if __name__ == "__main__":
    main()
