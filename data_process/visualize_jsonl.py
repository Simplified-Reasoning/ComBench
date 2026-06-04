#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table


def _read_line(path: Path, line_no: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if line_no < 1 or line_no > len(lines):
        raise ValueError(f"line must be in [1, {len(lines)}], got {line_no}")
    return lines[line_no - 1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize one JSONL record from data/*.jsonl with rich."
    )
    parser.add_argument("--file", required=True, help="Path to a .jsonl file under data/")
    parser.add_argument("--line", type=int, required=True, help="1-based line number")
    args = parser.parse_args()

    path = Path(args.file)
    raw = _read_line(path, args.line)
    obj = json.loads(raw)

    console = Console()

    table = Table(title=f"{path} (line {args.line})", show_lines=True)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    for key, value in obj.items():
        if key == "verify_code":
            continue
        if isinstance(value, str):
            rendered = value
        else:
            rendered = json.dumps(value, ensure_ascii=False, indent=2)
        table.add_row(escape(str(key)), escape(rendered))

    console.print(table)

    if "verify_code" in obj:
        code = obj["verify_code"]
        syntax = Syntax(code, "python", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="verify_code", expand=True))


if __name__ == "__main__":
    main()
