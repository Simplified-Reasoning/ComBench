ANSWER_MAX_SCORE = 1
PROOF_MAX_SCORE = 7
CONSTRUCTION_MAX_SCORE = 1
TOTAL_MAX_SCORE = 1.0
EVALUATION_SCHEMA_VERSION = 3


def normalize_answer_score(raw_score: int) -> float:
    return raw_score / ANSWER_MAX_SCORE


def normalize_proof_score(raw_score: int) -> float:
    return raw_score / PROOF_MAX_SCORE


def normalize_construction_score(raw_score: int) -> float:
    return raw_score / CONSTRUCTION_MAX_SCORE
