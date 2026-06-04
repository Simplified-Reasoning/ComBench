from dataclasses import dataclass
import re
from typing import Optional, Tuple

from src.eval.record_mode import RecordMode, uses_construction_prompt


CONSTRUCT_PATTERN = re.compile(r"<construct>(.*?)</construct>", re.S | re.I)
OPEN_CONSTRUCT_TAG_PATTERN = re.compile(r"<construct>", re.I)
CLOSE_CONSTRUCT_TAG_PATTERN = re.compile(r"</construct>", re.I)
QUESTION_1_HEADER_PATTERN = re.compile(r"^## Solution to Question 1[ \t]*$", re.M)
QUESTION_2_HEADER_PATTERN = re.compile(r"^## Solution to Question 2[ \t]*$", re.M)
SOLUTION_HEADER_PATTERN = re.compile(r"^## Solution[ \t]*$", re.M)


@dataclass(frozen=True)
class ParsedConstructionResponseSections:
    question_1: str
    question_2: str


@dataclass(frozen=True)
class PreparedResponse:
    mode: RecordMode
    solution_text: Optional[str]
    solution_error: Optional[str]
    construction: Optional[str]
    construction_error: Optional[str]


def parse_response_sections(
    text: str,
    mode: RecordMode,
) -> Tuple[Optional[ParsedConstructionResponseSections], Optional[str]]:
    if not uses_construction_prompt(mode):
        return None, "plain mode does not use question 1/question 2 sections"
    return _parse_construction_response_sections(text)


def prepare_response(text: str, mode: RecordMode) -> PreparedResponse:
    if uses_construction_prompt(mode):
        return _prepare_construction_response(text, mode)
    return _prepare_plain_response(text, mode)


def validate_question_1(text: str) -> Optional[str]:
    if OPEN_CONSTRUCT_TAG_PATTERN.search(text or "") or CLOSE_CONSTRUCT_TAG_PATTERN.search(text or ""):
        return "construct block is not allowed in question 1 solution"
    return None


def extract_construction(text: str) -> Tuple[Optional[str], Optional[str]]:
    text = text or ""
    open_construct_tags = len(OPEN_CONSTRUCT_TAG_PATTERN.findall(text))
    close_construct_tags = len(CLOSE_CONSTRUCT_TAG_PATTERN.findall(text))
    if open_construct_tags != 1 or close_construct_tags != 1:
        return None, "question 2 solution must contain exactly one construction block"
    construct_matches = CONSTRUCT_PATTERN.findall(text)
    if len(construct_matches) != 1:
        return None, "question 2 solution must contain exactly one construction block"
    return construct_matches[0].strip(), None


def extract_answer(text: str, mode: RecordMode) -> Optional[str]:
    prepared = prepare_response(text, mode)
    if prepared.solution_error:
        return None
    return prepared.solution_text


def _parse_construction_response_sections(text: str) -> Tuple[Optional[ParsedConstructionResponseSections], Optional[str]]:
    text = text or ""

    q1_matches = list(QUESTION_1_HEADER_PATTERN.finditer(text))
    q2_matches = list(QUESTION_2_HEADER_PATTERN.finditer(text))
    if not q1_matches:
        return None, "missing question 1 solution heading"
    if not q2_matches:
        return None, "missing question 2 solution heading"
    if len(q1_matches) != 1:
        return None, "duplicate question 1 solution heading"
    if len(q2_matches) != 1:
        return None, "duplicate question 2 solution heading"

    q1_match = q1_matches[0]
    q2_match = q2_matches[0]
    if text[: q1_match.start()].strip():
        return None, "unexpected content before question 1 heading"
    if q1_match.start() >= q2_match.start():
        return None, "solution headings out of order"

    question_1 = text[q1_match.end() : q2_match.start()].strip()
    question_2 = text[q2_match.end() :].strip()
    if not question_1:
        return None, "empty question 1 solution"
    if not question_2:
        return None, "empty question 2 solution"

    return ParsedConstructionResponseSections(question_1=question_1, question_2=question_2), None


def _prepare_construction_response(text: str, mode: RecordMode) -> PreparedResponse:
    parsed, error = _parse_construction_response_sections(text)
    if not parsed:
        return PreparedResponse(
            mode=mode,
            solution_text=None,
            solution_error=error,
            construction=None,
            construction_error=error,
        )

    solution_error = validate_question_1(parsed.question_1)
    construction, construction_error = extract_construction(parsed.question_2)
    return PreparedResponse(
        mode=mode,
        solution_text=parsed.question_1,
        solution_error=solution_error,
        construction=construction,
        construction_error=construction_error,
    )


def _prepare_plain_response(text: str, mode: RecordMode) -> PreparedResponse:
    solution_text = (text or "").strip()
    if not solution_text:
        error = "empty solution"
        return PreparedResponse(mode=mode, solution_text=None, solution_error=error, construction=None, construction_error=None)
    if OPEN_CONSTRUCT_TAG_PATTERN.search(solution_text) or CLOSE_CONSTRUCT_TAG_PATTERN.search(solution_text):
        error = "construct block is not allowed in plain solution"
        return PreparedResponse(mode=mode, solution_text=solution_text, solution_error=error, construction=None, construction_error=None)

    return PreparedResponse(
        mode=mode,
        solution_text=solution_text,
        solution_error=None,
        construction=None,
        construction_error=None,
    )
