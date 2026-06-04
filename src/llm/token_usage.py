from typing import Any, Dict, Iterable


TOKEN_USAGE_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "response_tokens",
    "total_tokens",
)


def normalize_token_usage(raw_usage: Dict[str, Any] | None) -> Dict[str, Any]:
    raw = raw_usage if isinstance(raw_usage, dict) else {}
    completion_details = _dict_value(raw, "completion_tokens_details")
    output_details = _dict_value(raw, "output_tokens_details")
    usage_metadata = _dict_value(raw, "usageMetadata")

    input_tokens = _first_number(
        raw,
        usage_metadata,
        keys=(
            "input_tokens",
            "prompt_tokens",
            "promptTokenCount",
            "inputTokenCount",
        ),
    )
    output_tokens = _first_number(
        raw,
        usage_metadata,
        keys=(
            "output_tokens",
            "completion_tokens",
            "candidatesTokenCount",
            "outputTokenCount",
        ),
    )
    reasoning_tokens = _first_number(
        raw,
        completion_details,
        output_details,
        usage_metadata,
        keys=(
            "reasoning_tokens",
            "reasoningTokenCount",
            "thinking_tokens",
            "thinkingTokenCount",
            "thoughtsTokenCount",
        ),
    )
    total_tokens = _first_number(
        raw,
        usage_metadata,
        keys=(
            "total_tokens",
            "totalTokenCount",
        ),
    )

    if output_tokens is None and total_tokens is not None and input_tokens is not None:
        output_tokens = max(total_tokens - input_tokens, 0)
    if total_tokens is None:
        total_tokens = _sum_optional(input_tokens, output_tokens)
    if reasoning_tokens is None:
        reasoning_tokens = 0

    response_tokens = None
    if output_tokens is not None:
        response_tokens = max(output_tokens - reasoning_tokens, 0)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "response_tokens": response_tokens,
        "total_tokens": total_tokens,
        "raw": raw,
    }


def sum_normalized_token_usage(usages: Iterable[Dict[str, Any] | None]) -> Dict[str, Any]:
    totals = {field: 0 for field in TOKEN_USAGE_FIELDS}
    for usage in usages:
        normalized = normalize_token_usage(usage)
        for field in TOKEN_USAGE_FIELDS:
            value = normalized.get(field)
            if isinstance(value, (int, float)):
                totals[field] += value
    return totals


def sum_pre_normalized_token_usage(usages: Iterable[Dict[str, Any] | None]) -> Dict[str, Any]:
    totals = {field: 0 for field in TOKEN_USAGE_FIELDS}
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for field in TOKEN_USAGE_FIELDS:
            value = usage.get(field)
            if isinstance(value, (int, float)):
                totals[field] += value
    return totals


def _dict_value(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key)
    return value if isinstance(value, dict) else {}


def _first_number(*sources: Dict[str, Any], keys: tuple[str, ...]) -> int | float | None:
    for source in sources:
        for key in keys:
            value = source.get(key)
            if isinstance(value, (int, float)):
                return value
    return None


def _sum_optional(*values: int | float | None) -> int | float | None:
    numeric = [value for value in values if isinstance(value, (int, float))]
    return sum(numeric) if numeric else None
