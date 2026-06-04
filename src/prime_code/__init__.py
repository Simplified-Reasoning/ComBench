import json
import traceback
from typing import Any


def _extract_solution(completion: str) -> str:
    if "```python" in completion:
        return completion.split("```python")[-1].split("```")[0]
    if "```" in completion:
        parts = completion.split("```")
        if len(parts) >= 2:
            return parts[-2]
    return completion


def _load_test_cases(test_cases: dict[str, Any] | str) -> dict[str, Any]:
    if isinstance(test_cases, dict):
        return test_cases
    return json.loads(test_cases)


def _load_check_correctness():
    from .utils import check_correctness

    return check_correctness


def _normalize_status(results: list[Any], metadata: dict[str, Any]) -> str:
    metadata_status = metadata.get("status")
    if results and all(result is True for result in results):
        return "passed"
    if metadata_status and str(metadata_status) != "passed":
        return str(metadata_status)
    if any(result == -2 for result in results):
        return "compilation_error"
    if any(result == -1 for result in results):
        return "runtime_error"
    if any(result is False for result in results):
        return "wrong_answer"
    return "failed"


def _normalize_metadata(results: list[Any], metadata_list: list[Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if metadata_list:
        first = metadata_list[0]
        if isinstance(first, dict):
            metadata.update(first)
    metadata["raw_results"] = list(results)
    metadata["status"] = _normalize_status(results, metadata)
    return metadata


def compute_score(
    completion: str,
    test_cases: dict[str, Any] | str,
    continuous: bool = False,
    timeout: int = 5,
) -> tuple[float, dict[str, Any]]:
    solution = _extract_solution(completion)
    if not solution.strip():
        return 0.0, {"status": "invalid_completion", "error": "missing code block"}

    try:
        parsed_cases = _load_test_cases(test_cases)
    except Exception as exc:
        return 0.0, {
            "status": "invalid_test_cases",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

    if "inputs" not in parsed_cases or "outputs" not in parsed_cases:
        return 0.0, {
            "status": "invalid_test_cases",
            "error": "test_cases must include inputs and outputs",
        }

    check_correctness = _load_check_correctness()

    try:
        raw_results, raw_metadata_list = check_correctness(
            in_outs=parsed_cases,
            generation=solution,
            timeout=timeout,
            debug=False,
        )
    except ModuleNotFoundError:
        raise
    except Exception as exc:
        return 0.0, {
            "status": "internal_error",
            "error": repr(exc),
            "traceback": traceback.format_exc(),
        }

    results = list(raw_results) if isinstance(raw_results, (list, tuple)) else [raw_results]
    metadata_list = list(raw_metadata_list) if raw_metadata_list is not None else []
    metadata = _normalize_metadata(results, metadata_list)

    if continuous:
        total = len(results) or 1
        score = sum(result is True for result in results) / total
    else:
        score = 1.0 if results and all(result is True for result in results) else 0.0
    metadata["score"] = score
    return float(score), metadata
