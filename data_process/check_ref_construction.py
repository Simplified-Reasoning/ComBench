#!/usr/bin/env python3
import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.construction_text import make_python_literal_stdin
from src.eval.prime_code_executor import PrimeCodeDependencyError, run_prime_code_verifier


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _stringify_ref_construction(ref_construction: Any) -> str:
    if isinstance(ref_construction, str):
        return ref_construction
    return repr(ref_construction)


def _to_json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    except Exception:
        return repr(value)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head
    return text[:head] + f"\n... [truncated {len(text) - max_chars} chars] ...\n" + text[-tail:]


def _print_failure_details(line_no: int, detail: dict[str, Any], max_log_chars: int) -> None:
    print(f"----- failure details (line {line_no}) -----")
    print(f"id: {detail.get('record_id')}")
    print(f"reason: {detail.get('info')}")
    print(f"verify_code_type: {detail.get('verify_code_type')}")
    print(f"ref_construction_type: {detail.get('ref_construction_type')}")

    metadata = detail.get("metadata") or {}
    location = metadata.get("prime_code_error_location")
    if isinstance(location, dict):
        file_path = location.get("file")
        line = location.get("line")
        function_name = location.get("function")
        code = location.get("code")
        print(
            "prime_code_error_location: "
            f"{file_path}:{line}"
            + (f" in {function_name}" if function_name else "")
        )
        if code:
            print(f"prime_code_error_code: {code}")

    print("metadata_summary:")
    print(
        _to_json_text(
            {
                "status": metadata.get("status"),
                "score": metadata.get("score"),
                "error": metadata.get("error"),
                "error_message": metadata.get("error_message"),
                "exception_type": metadata.get("exception_type"),
                "prime_code_error_location": metadata.get("prime_code_error_location"),
                "process_exitcode": metadata.get("process_exitcode"),
                "timeout_seconds": metadata.get("timeout_seconds"),
                "output": metadata.get("output"),
                "expected": metadata.get("expected"),
                "inputs": metadata.get("inputs"),
            }
        )
    )

    if detail.get("metadata") is not None:
        print("\nmetadata_full:")
        print(_truncate(_to_json_text(detail["metadata"]), max_log_chars))

    stdin_text = detail.get("stdin_text")
    if isinstance(stdin_text, str):
        print("\nstdin_sent:")
        print(_truncate(stdin_text, max_log_chars))

    verify_code = detail.get("verify_code")
    if isinstance(verify_code, str):
        print("\nverify_code:")
        print(_truncate(verify_code, max_log_chars))

    ref_raw = detail.get("ref_construction_text")
    if isinstance(ref_raw, str):
        print("\nref_construction_text:")
        print(_truncate(ref_raw, max_log_chars))

    exception_tb = detail.get("exception_traceback")
    if isinstance(exception_tb, str):
        print("\nexception_traceback:")
        print(_truncate(exception_tb, max_log_chars))
    print("----- end failure details -----")


def _check_record(
    obj: dict[str, Any],
    line_no: int,
    timeout: int,
) -> tuple[bool, str, str, int, int, dict[str, Any]]:
    record_id = obj.get("id")
    verify_code_obj = obj.get("verify_code")
    ref_construction_obj = obj.get("ref_construction")
    verify_code = verify_code_obj if isinstance(verify_code_obj, str) else None
    ref_construction_text = _stringify_ref_construction(ref_construction_obj)
    verify_len = len(verify_code) if isinstance(verify_code, str) else 0
    construction_len = len(ref_construction_text)
    detail: dict[str, Any] = {
        "record_id": str(record_id),
        "line_no": line_no,
        "verify_code_type": type(verify_code_obj).__name__,
        "ref_construction_type": type(ref_construction_obj).__name__,
        "verify_code": verify_code,
        "ref_construction_text": ref_construction_text,
        "stdin_text": None,
        "metadata": None,
        "exception_traceback": None,
        "info": None,
    }

    if not isinstance(verify_code, str) or not verify_code.strip():
        detail["info"] = "missing verify_code"
        return False, "missing verify_code", str(record_id), verify_len, construction_len, detail
    if not ref_construction_text.strip():
        detail["info"] = "missing ref_construction"
        return False, "missing ref_construction", str(record_id), verify_len, construction_len, detail

    stdin_text = make_python_literal_stdin(ref_construction_text)
    detail["stdin_text"] = stdin_text

    try:
        result = run_prime_code_verifier(
            verify_code=verify_code,
            stdin_text=stdin_text,
            timeout=timeout,
        )
        detail["metadata"] = result.metadata
        detail["info"] = result.info
        return result.passed, result.info, str(record_id), verify_len, construction_len, detail
    except PrimeCodeDependencyError:
        raise
    except Exception as exc:
        detail["info"] = f"verify_code exception: {exc}"
        detail["exception_traceback"] = traceback.format_exc()
        return False, detail["info"], str(record_id), verify_len, construction_len, detail


def main() -> None:
    parser = argparse.ArgumentParser(description="Check ref_construction in data/*.jsonl via verify_code.")
    parser.add_argument("--file", required=True, help="Path to a .jsonl file under data/")
    parser.add_argument("--line", type=int, help="1-based line number; if omitted, check all lines")
    parser.add_argument("--timeout", type=int, default=15, help="Timeout in seconds")
    parser.add_argument(
        "--max-log-chars",
        type=int,
        default=0,
        help="Max characters for each failure log block. Use 0 for unlimited (default).",
    )
    args = parser.parse_args()

    path = Path(args.file)
    lines = _read_lines(path)

    if args.line is not None:
        if args.line < 1 or args.line > len(lines):
            raise ValueError(f"line must be in [1, {len(lines)}], got {args.line}")
        indices = [args.line - 1]
    else:
        indices = list(range(len(lines)))

    all_ok = True
    for i in indices:
        raw = lines[i]
        try:
            obj = json.loads(raw)
        except Exception as exc:
            print(f"line {i+1}: invalid JSON ({exc})")
            print(f"raw line: {_truncate(raw, args.max_log_chars)}")
            all_ok = False
            continue

        ok, info, record_id, verify_len, construction_len, detail = _check_record(
            obj=obj,
            line_no=i + 1,
            timeout=args.timeout,
        )
        status = "PASS" if ok else "FAIL"
        print(
            f"line {i+1} id={record_id} "
            f"verify_len={verify_len} construction_len={construction_len}: "
            f"{status} ({info})"
        )
        if not ok:
            all_ok = False
            _print_failure_details(i + 1, detail, args.max_log_chars)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except PrimeCodeDependencyError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
