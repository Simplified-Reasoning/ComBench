from typing import Any, Dict

from src.eval.record_mode import resolve_record_mode, uses_construction_prompt


def _maybe_box_answer(ref_answer: str | None) -> str | None:
    if not ref_answer:
        return None
    return f"\\boxed{{{ref_answer}}}"


class MockResponseModel:
    name = "mock"

    def generate(self, prompt: str, record: Dict[str, Any]) -> str:
        mode = resolve_record_mode(record, warn=False)
        pieces = []
        boxed = _maybe_box_answer(record.get("ref_answer"))
        if uses_construction_prompt(mode):
            pieces.append("## Solution to Question 1")
            if boxed:
                pieces.append(f"Answer: {boxed}")
            else:
                pieces.append("No answer available.")

            pieces.append("")
            pieces.append("## Solution to Question 2")
            construction = record.get("ref_construction")
            if construction:
                pieces.append("<construct>")
                pieces.append(construction.strip())
                pieces.append("</construct>")
            else:
                pieces.append("No construction available.")
        else:
            pieces.append("## Solution")
            if boxed:
                pieces.append(f"Answer: {boxed}")
            else:
                pieces.append("No answer available.")
        return "\n".join(pieces)
