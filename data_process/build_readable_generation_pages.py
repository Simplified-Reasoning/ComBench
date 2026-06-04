#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATASET = ROOT.parent / "CB_100" / "construction_CB100_ref_sol.jsonl"
DEFAULT_GENERATION_CACHE = (
    ROOT
    / "outputs"
    / "construction_CB100_ref_sol"
    / "gemini"
    / "gemini"
    / "generation"
    / "cache.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "construction_CB100_ref_sol"
    / "gemini"
    / "gemini"
    / "readable_cases"
)
DEFAULT_EVALUATION_CASES_DIR = (
    ROOT
    / "outputs"
    / "construction_CB100_ref_sol"
    / "gemini"
    / "gemini"
    / "evaluation"
    / "gemini"
    / "cases"
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"expected JSON object at {path}:{line_no}")
        yield obj


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip()
    cleaned = cleaned.rstrip(". ")
    return cleaned or "unnamed"


def _text_or_na(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        text = value.strip()
        return text if text else "N/A"
    return json.dumps(value, ensure_ascii=False, indent=2)


def _build_markdown(record: dict[str, Any], response: str | None, evaluation: dict[str, Any] | None) -> str:
    record_id = _text_or_na(record.get("id"))
    response_text = response.strip() if response and response.strip() else "MISSING_GENERATION_RESPONSE"
    proof = (evaluation or {}).get("proof") or {}

    sections = [
        f"# {record_id}",
        "",
        "## Problem",
        _text_or_na(record.get("query")),
        "",
        "## Construction Instruction",
        _text_or_na(record.get("instruction")),
        "",
        "## Reference Solution",
        _text_or_na(record.get("ref_solution")),
        "",
        "## Gemini Response",
        response_text,
        "",
        "## Grading Guidelines",
        _text_or_na(record.get("grading_guidelines")),
        "",
        "## Proof Evaluation",
        f"raw_score: {_text_or_na(proof.get('raw_score'))}",
        "",
        "### raw",
        _text_or_na(proof.get("raw")),
        "",
    ]
    return "\n".join(sections)


def _load_generation_responses(path: Path) -> dict[str, str]:
    responses: dict[str, str] = {}
    duplicate_ids: list[str] = []
    for item in _read_jsonl(path):
        item_id = str(item.get("id", "")).strip()
        if not item_id:
            continue
        if item_id in responses:
            duplicate_ids.append(item_id)
        responses[item_id] = str(item.get("response", ""))
    if duplicate_ids:
        raise ValueError(f"duplicate id(s) in generation cache: {sorted(set(duplicate_ids))}")
    return responses


def _load_evaluations(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    evaluations: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for item in path.glob("*.json"):
        item_id = item.stem
        if item_id in evaluations:
            duplicate_ids.append(item_id)
        data = json.loads(item.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"expected JSON object in evaluation case: {item}")
        evaluations[item_id] = data
    if duplicate_ids:
        raise ValueError(f"duplicate evaluation id(s): {sorted(set(duplicate_ids))}")
    return evaluations


def build_pages(
    dataset_path: Path,
    generation_cache_path: Path,
    output_dir: Path,
    evaluation_cases_dir: Path,
) -> tuple[int, int, int, list[str], list[str]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset not found: {dataset_path}")
    if not generation_cache_path.exists():
        raise FileNotFoundError(f"generation cache not found: {generation_cache_path}")

    records = list(_read_jsonl(dataset_path))
    responses = _load_generation_responses(generation_cache_path)
    evaluations = _load_evaluations(evaluation_cases_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    missing_response_ids: list[str] = []
    missing_evaluation_ids: list[str] = []
    written = 0

    for index, record in enumerate(records, start=1):
        record_id = str(record.get("id", "")).strip() or f"line-{index}"
        response = responses.get(record_id)
        if response is None:
            missing_response_ids.append(record_id)
        evaluation = evaluations.get(record_id)
        if evaluation is None:
            missing_evaluation_ids.append(record_id)

        output_path = output_dir / f"{_safe_filename(record_id)}.md"
        output_path.write_text(_build_markdown(record, response, evaluation), encoding="utf-8")
        written += 1

    return written, len(responses), len(evaluations), missing_response_ids, missing_evaluation_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one human-readable Markdown page per generated math response."
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the source JSONL dataset.",
    )
    parser.add_argument(
        "--generation-cache",
        default=str(DEFAULT_GENERATION_CACHE),
        help="Path to generation/cache.jsonl containing model responses.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where per-problem Markdown files will be written.",
    )
    parser.add_argument(
        "--evaluation-cases-dir",
        default=str(DEFAULT_EVALUATION_CASES_DIR),
        help="Directory containing evaluation case JSON files.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    generation_cache_path = Path(args.generation_cache)
    output_dir = Path(args.output_dir)
    evaluation_cases_dir = Path(args.evaluation_cases_dir)

    written, response_count, evaluation_count, missing_response_ids, missing_evaluation_ids = build_pages(
        dataset_path=dataset_path,
        generation_cache_path=generation_cache_path,
        output_dir=output_dir,
        evaluation_cases_dir=evaluation_cases_dir,
    )

    print(f"dataset records: {written}")
    print(f"generation responses: {response_count}")
    print(f"evaluation cases: {evaluation_count}")
    print(f"markdown files written: {written}")
    print(f"output dir: {output_dir}")
    if missing_response_ids:
        print("missing generation responses:")
        for item_id in missing_response_ids:
            print(f"- {item_id}")
    if missing_evaluation_ids:
        print("missing evaluation cases:")
        for item_id in missing_evaluation_ids:
            print(f"- {item_id}")


if __name__ == "__main__":
    main()
