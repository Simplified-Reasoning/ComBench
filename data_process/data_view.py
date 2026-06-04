"""Generate a single-page HTML viewer for JSONL files under data/."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "data_process" / "data.html"


def _safe_id(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                records.append(
                    {
                        "id": f"line-{line_no}-parse-error",
                        "line_no": line_no,
                        "error": f"JSON parse error: {exc}",
                        "raw": raw,
                    }
                )
                continue

            if isinstance(parsed, dict):
                record_id = _safe_id(parsed.get("id"), f"line-{line_no}")
            else:
                record_id = f"line-{line_no}"

            records.append(
                {
                    "id": record_id,
                    "line_no": line_no,
                    "record": parsed,
                }
            )
    return records


def scan_data_dir(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    file_records: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(data_dir.glob("*.jsonl")):
        file_records[path.name] = _load_jsonl_records(path)
    return file_records


def build_html(file_records: dict[str, list[dict[str, Any]]]) -> str:
    files = []
    records_map: dict[str, list[dict[str, Any]]] = {}
    for file_name, records in file_records.items():
        files.append({"name": file_name, "count": len(records)})
        records_map[file_name] = records

    payload = json.dumps({"files": files, "records": records_map}, ensure_ascii=False)
    payload_b64 = base64.b64encode(payload.encode("utf-8")).decode("ascii")

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Data JSONL Viewer</title>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [["$", "$"], ["\\\\(", "\\\\)"]],
        displayMath: [["$$", "$$"], ["\\\\[", "\\\\]"]],
        processEscapes: true
      }},
      options: {{
        skipHtmlTags: ["script", "noscript", "style", "textarea", "pre", "code"]
      }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; background: #0f0f0f; color: #e5e7eb; }}
    .layout {{ display: flex; height: 100vh; overflow: hidden; }}
    .sidebar {{ width: 360px; border-right: 1px solid #2a2a2a; overflow-y: auto; flex-shrink: 0; background: #121212; }}
    .sidebar h2 {{ margin: 0; padding: 14px 16px; font-size: 14px; color: #94a3b8; border-bottom: 1px solid #2a2a2a; }}
    .file-group {{ border-bottom: 1px solid #1f2937; }}
    .file-head {{ padding: 10px 16px; font-size: 13px; font-weight: 600; color: #93c5fd; background: #151515; }}
    .file-meta {{ margin-top: 4px; font-size: 12px; color: #64748b; font-weight: 500; }}
    .data-item {{ padding: 8px 16px 8px 28px; cursor: pointer; border-top: 1px solid #1a1a1a; }}
    .data-item:hover {{ background: #1a1a1a; }}
    .data-item.active {{ background: #1e3a5f; color: #bfdbfe; }}
    .data-item .id {{ font-size: 13px; font-weight: 600; }}
    .data-item .line {{ margin-top: 2px; font-size: 12px; color: #64748b; }}
    .main {{ flex: 1; overflow-y: auto; padding: 16px; }}
    .title {{ margin: 0 0 12px 0; font-size: 14px; color: #94a3b8; }}
    .card {{ background: #151515; border: 1px solid #303030; border-radius: 8px; overflow: hidden; }}
    .card-head {{ padding: 12px 14px; background: #1f1f1f; border-bottom: 1px solid #303030; font-size: 13px; display: flex; justify-content: space-between; gap: 12px; }}
    .card-body {{ padding: 14px; }}
    .field {{ margin-bottom: 14px; }}
    .field:last-child {{ margin-bottom: 0; }}
    .field-name {{ margin: 0 0 6px 0; font-size: 12px; color: #93c5fd; text-transform: uppercase; letter-spacing: 0.04em; }}
    pre {{ margin: 0; padding: 10px; background: #0f0f0f; border: 1px solid #262626; border-radius: 6px; font-size: 12px; line-height: 1.5; white-space: pre-wrap; word-break: break-word; overflow-x: auto; }}
    .text-block {{ margin: 0; padding: 10px; background: #0f0f0f; border: 1px solid #262626; border-radius: 6px; font-size: 12px; line-height: 1.6; white-space: pre-wrap; word-break: break-word; }}
    .empty {{ padding: 32px; color: #6b7280; text-align: center; }}
    .error {{ color: #fecaca; }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h2>Data / JSONL</h2>
      <div id="file-list"></div>
    </aside>
    <main class="main">
      <h3 id="main-title" class="title">请选择左侧数据项</h3>
      <div id="content"></div>
    </main>
  </div>
  <script>
    (function() {{
      const b64 = "{payload_b64}";
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      const DATA = JSON.parse(new TextDecoder().decode(bytes));

      const fileList = document.getElementById("file-list");
      const mainTitle = document.getElementById("main-title");
      const content = document.getElementById("content");

      function escapeHtml(text) {{
        const node = document.createElement("div");
        node.textContent = text;
        return node.innerHTML;
      }}

      function renderFileList() {{
        fileList.innerHTML = DATA.files.map(file => {{
          const items = DATA.records[file.name] || [];
          return `
            <div class="file-group">
              <div class="file-head">
                <div>${{escapeHtml(file.name)}}</div>
                <div class="file-meta">${{items.length}} item(s)</div>
              </div>
              ${{items.map((item, idx) => `
                <div class="data-item" data-file="${{file.name}}" data-index="${{idx}}">
                  <div class="id">${{escapeHtml(item.id)}}</div>
                  <div class="line">line ${{item.line_no}}</div>
                </div>
              `).join("") || `<div class="data-item"><div class="id">(empty)</div></div>`}}
            </div>
          `;
        }}).join("");

        fileList.querySelectorAll(".data-item[data-file]").forEach(node => {{
          node.addEventListener("click", () => {{
            selectRecord(node.dataset.file, Number(node.dataset.index));
          }});
        }});
      }}

      function orderedKeys(obj) {{
        const priority = ["id", "query", "instruction", "ref_answer", "ref_construction", "verify_code"];
        const keys = Object.keys(obj || {{}});
        const front = priority.filter(key => keys.includes(key));
        const rest = keys.filter(key => !priority.includes(key)).sort((a, b) => a.localeCompare(b));
        return front.concat(rest);
      }}

      function renderLiteral(value) {{
        if (typeof value === "string") return escapeHtml(value);
        return escapeHtml(JSON.stringify(value, null, 2));
      }}

      function normalizeMathInText(text) {{
        if (typeof text !== "string" || text.indexOf("\\\\") === -1) {{
          return text;
        }}

        function fixExpr(expr) {{
          return expr.replace(/\\\\\\\\/g, "\\\\");
        }}

        return text
          .replace(/\\$\\$([\\s\\S]*?)\\$\\$/g, (_, expr) => `$$${{fixExpr(expr)}}$$`)
          .replace(/\\$([^$\\n]+?)\\$/g, (_, expr) => `$${{fixExpr(expr)}}$`)
          .replace(/\\\\\\(([\\s\\S]*?)\\\\\\)/g, (_, expr) => `\\\\(${{fixExpr(expr)}}\\\\)`)
          .replace(/\\\\\\[([\\s\\S]*?)\\\\\\]/g, (_, expr) => `\\\\[${{fixExpr(expr)}}\\\\]`);
      }}

      function renderFieldValue(key, value) {{
        const mathFields = new Set(["query", "instruction", "ref_answer"]);
        const codeLikeFields = new Set(["verify_code", "ref_construction"]);

        if (typeof value !== "string") {{
          return `<pre>${{renderLiteral(value)}}</pre>`;
        }}
        if (codeLikeFields.has(key)) {{
          return `<pre>${{escapeHtml(value)}}</pre>`;
        }}
        if (mathFields.has(key)) {{
          const normalized = normalizeMathInText(value);
          return `<div class="text-block">${{escapeHtml(normalized)}}</div>`;
        }}
        return `<pre>${{escapeHtml(value)}}</pre>`;
      }}

      function typesetMath(target) {{
        if (!window.MathJax || !window.MathJax.typesetPromise) {{
          return;
        }}
        window.MathJax.typesetPromise([target]).catch(err => {{
          console.error("MathJax typeset failed:", err);
        }});
      }}

      function selectRecord(fileName, index) {{
        fileList.querySelectorAll(".data-item").forEach(node => {{
          node.classList.toggle(
            "active",
            node.dataset.file === fileName && Number(node.dataset.index) === index
          );
        }});

        const record = (DATA.records[fileName] || [])[index];
        if (!record) {{
          mainTitle.textContent = "记录不存在";
          content.innerHTML = '<div class="empty">无法加载该记录</div>';
          return;
        }}

        mainTitle.textContent = `${{fileName}} / ${{record.id}}`;

        if (record.error) {{
          content.innerHTML = `
            <div class="card">
              <div class="card-head">
                <span class="error">解析失败</span>
                <span>line ${{record.line_no}}</span>
              </div>
              <div class="card-body">
                <div class="field">
                  <h4 class="field-name">error</h4>
                  <pre class="error">${{escapeHtml(record.error)}}</pre>
                </div>
                <div class="field">
                  <h4 class="field-name">raw</h4>
                  <pre>${{escapeHtml(record.raw || "")}}</pre>
                </div>
              </div>
            </div>
          `;
          return;
        }}

        const obj = record.record;
        const keys = (obj && typeof obj === "object" && !Array.isArray(obj)) ? orderedKeys(obj) : [];
        const fieldsHtml = keys.map(key => `
          <div class="field">
            <h4 class="field-name">${{escapeHtml(key)}}</h4>
            ${{renderFieldValue(key, obj[key])}}
          </div>
        `).join("");

        const fallback = keys.length === 0
          ? `<div class="field"><h4 class="field-name">value</h4>${{renderFieldValue("value", obj)}}</div>`
          : "";

        content.innerHTML = `
          <div class="card">
            <div class="card-head">
              <span>ID: ${{escapeHtml(record.id)}}</span>
              <span>line ${{record.line_no}}</span>
            </div>
            <div class="card-body">
              ${{fieldsHtml || fallback || '<div class="empty">空记录</div>'}}
            </div>
          </div>
        `;
        typesetMath(content);
      }}

      renderFileList();
      if (DATA.files.length > 0) {{
        const firstFile = DATA.files[0].name;
        const firstRecords = DATA.records[firstFile] || [];
        if (firstRecords.length > 0) {{
          selectRecord(firstFile, 0);
        }} else {{
          content.innerHTML = '<div class="empty">未发现可展示的记录</div>';
        }}
      }} else {{
        content.innerHTML = '<div class="empty">data 目录下未找到 .jsonl 文件</div>';
      }}

      if (window.MathJax && window.MathJax.startup && window.MathJax.startup.promise) {{
        window.MathJax.startup.promise.then(() => {{
          typesetMath(content);
        }});
      }}
    }})();
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 data/ 目录中的 .jsonl 文件生成到 data_process/data.html"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="JSONL 数据目录，默认仓库根目录下的 data/",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 HTML 路径，默认 data_process/data.html",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if not data_dir.exists() or not data_dir.is_dir():
        raise SystemExit(f"无效 data 目录: {data_dir}")

    file_records = scan_data_dir(data_dir)
    output_path = args.output.resolve() if args.output else DEFAULT_OUTPUT

    html = build_html(file_records)
    output_path.write_text(html, encoding="utf-8")

    total_files = len(file_records)
    total_items = sum(len(items) for items in file_records.values())
    print(f"已生成: {output_path} | files={total_files}, items={total_items}")


if __name__ == "__main__":
    main()
