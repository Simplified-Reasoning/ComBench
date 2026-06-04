from dataclasses import dataclass
from typing import Any

from src.prime_code import compute_score


class PrimeCodeDependencyError(RuntimeError):
    pass


@dataclass(frozen=True)
class PrimeCodeExecutionResult:
    passed: bool
    score: float
    info: str
    metadata: dict[str, Any]


def run_prime_code_verifier(verify_code: str, stdin_text: str, timeout: int) -> PrimeCodeExecutionResult:
    test_cases = {"inputs": [stdin_text], "outputs": ["True"]}
    try:
        score, metadata = compute_score(
            completion=f"```python\n{verify_code}\n```",
            test_cases=test_cases,
            continuous=False,
            timeout=timeout,
        )
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown"
        raise PrimeCodeDependencyError(
            f"prime_code backend is unavailable because dependency '{missing}' is missing"
        ) from exc

    normalized_metadata = dict(metadata or {})
    normalized_score = float(score)
    passed = normalized_score == 1.0
    info = _build_info(passed, normalized_metadata)
    return PrimeCodeExecutionResult(
        passed=passed,
        score=normalized_score,
        info=info,
        metadata=normalized_metadata,
    )


def _build_info(passed: bool, metadata: dict[str, Any]) -> str:
    if passed:
        return "True"
    for key in ("status", "error_message", "error", "traceback"):
        value = metadata.get(key)
        if value:
            return str(value)
    return "unknown"
