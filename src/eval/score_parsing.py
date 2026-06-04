import re
from typing import Optional


THINKING_PATTERN = re.compile(r"<thinking>.*?</thinking>", re.S | re.I)
ANSWER_BOX_PATTERN = re.compile(r"(?is)^\\boxed\{\s*(Correct|Incorrect)\s*\}\s*$")
POINTS_PATTERN = re.compile(r"<points>(.*?)</points>", re.S | re.I)


def parse_answer_score(text: str) -> Optional[int]:
    content = THINKING_PATTERN.sub("", text or "").strip()
    match = ANSWER_BOX_PATTERN.fullmatch(content)
    if not match:
        return None
    return 1 if match.group(1).lower() == "correct" else 0


def parse_proof_score(text: str) -> Optional[int]:
    match = POINTS_PATTERN.search(text or "")
    if not match:
        return None
    normalized = " ".join(match.group(1).split())
    allowed = {
        "0 out of 7": 0,
        "1 out of 7": 1,
        "6 out of 7": 6,
        "7 out of 7": 7,
    }
    return allowed.get(normalized)
