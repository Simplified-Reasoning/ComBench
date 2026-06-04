import ast
import json


def is_json_or_python_text(text: str) -> bool:
    content = text.strip()
    if not content:
        return False

    try:
        json.loads(content)
        return True
    except json.JSONDecodeError:
        pass

    try:
        ast.literal_eval(content)
        return True
    except (SyntaxError, ValueError):
        return False


def normalize_python_literal_text(text: str) -> str:
    content = text.strip()
    if not content:
        return ""
    return content


def make_python_literal_stdin(text: str) -> str:
    content = normalize_python_literal_text(text)
    return f"{content}\n" if content else ""
