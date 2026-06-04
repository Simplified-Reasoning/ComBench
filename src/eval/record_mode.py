from __future__ import annotations

import warnings
from typing import Any, Literal


RecordMode = Literal["construction", "plain", "inconsistent"]


def resolve_record_mode(record: dict[str, Any], warn: bool = False) -> RecordMode:
    has_instruction = _has_text(record.get("instruction"))
    has_verify_code = _has_text(record.get("verify_code"))

    if has_instruction and has_verify_code:
        return "construction"
    if not has_instruction and not has_verify_code:
        return "plain"

    if warn:
        record_id = record.get("id", "<unknown>")
        warnings.warn(
            (
                f"record {record_id!r} has inconsistent construction metadata: "
                f"instruction={has_instruction}, verify_code={has_verify_code}; "
                "falling back to plain-mode evaluation"
            ),
            stacklevel=2,
        )
    return "inconsistent"


def uses_construction_prompt(mode: RecordMode) -> bool:
    return mode == "construction"


def _has_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())
