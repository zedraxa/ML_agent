#bu hali olmamış diye tagleyip release ettiğim write_file hatası veren halinin güncellenmiş hali 
import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import ollama

WEB_SEARCH_DEFAULT_ENABLED = False
DEFAULT_PROJECT = "scratch_project"

SYSTEM_PROMPT = """
You are a local Bioengineering ML Project Agent running on Linux.

GOAL
Turn the user's request into a reproducible ML project inside workspace/<project>/.

HARD RULES
- Use ONLY the tool protocol when performing actions (create files, run commands, web search/open).
- All files MUST be under workspace/<project>/.
- WRITE_FILE MUST use:
  path: relative/path.ext
  ---
  file content...
- WEB_SEARCH is disabled unless user message includes: ALLOW_WEB_SEARCH

WORKFLOW
1) Clarify I/O + metrics (brief).
2) Find 2-5 candidate datasets (name+link+license/terms).
3) Pick dataset; download into data/raw/.
4) Create project structure and requirements.txt.
5) Implement baseline model in src/train.py.
6) Run training and save results/metrics.json.
7) Write report.md and README.md.

Output language: Turkish (unless user asks otherwise).

TOOL PROTOCOL (ONE BLOCK ONLY):
<PYTHON>...</PYTHON>
<BASH>...</BASH>
<WEB_SEARCH>...</WEB_SEARCH>
<WEB_OPEN>...</WEB_OPEN>
<READ_FILE>...</READ_FILE>
<WRITE_FILE>...</WRITE_FILE>
<TODO>...</TODO>
"""

TOOL_TAGS = ["PYTHON", "BASH", "WEB_SEARCH", "WEB_OPEN", "READ_FILE", "WRITE_FILE", "TODO"]
TOOL_RE = re.compile(
    r"<(" + "|".join(TOOL_TAGS) + r")>\s*(.*?)\s*</\1>",
    re.DOTALL | re.IGNORECASE,
)

# Accept fenced code in ANY case, with optional language labels
FENCED_BASH_RE = re.compile(r"```(?:bash)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
FENCED_PY_RE = re.compile(r"```(?:python|py)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)

DENY_PATTERNS = [
    r"\brm\b.*\b-rf\b\s+/\b",
    r":\(\)\s*{\s*:\s*\|\s*:\s*&\s*}\s*;\s*:",
    r"\bdd\b\s+if=/dev/zero\b",
    r"\bmkfs\.",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bkill\b\s+-9\s+1\b",
]


def is_dangerous_bash(cmd: str) -> Optional[str]:
    for pat in DENY_PATTERNS:
        if re.search(pat, cmd.strip()):
            return f"Blocked by denylist pattern: {pat}"
    return None


def safe_relpath(path: str) -> str:
    p = Path(path).expanduser()
    if p.is_absolute():
        raise ValueError("Absolute paths are not allowed. Use relative paths inside workspace.")
    norm = Path(os.path.normpath(str(p)))
    if str(norm).startswith(".."):
        raise ValueError("Path traversal is not allowed.")
    return str(norm)


def current_project() -> str:
    return os.getenv("AGENT_PROJECT", DEFAULT_PROJECT)


def run_python(code: str, workspace: Path, timeout_s: int = 180) -> str:
    code = textwrap.dedent(code).strip() + "\n"
    tmp = workspace / "_tmp_run.py"
    tmp.write_text(code, encoding="utf-8")
    try:
        res = subprocess.run(
            [sys.executable, str(tmp)],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        out = (res.stdout or "") + (res.stderr or "")
        return out.strip() if out.strip() else f"[python exit code: {res.returncode}] (no output)"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] Python exceeded {timeout_s} seconds."
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


def run_bash(cmd: str, workspace: Path, timeout_s: int = 180) -> str:
    reason = is_dangerous_bash(cmd)
    if reason:
        return f"[BLOCKED] {reason}\nCommand: {cmd}"
    try:
        res = subprocess.run(
            cmd,
            cwd=str(workspace),
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            executable="/bin/bash",
        )
        out = (res.stdout or "") + (res.stderr or "")
        return out.strip() if out.strip() else f"[bash exit code: {res.returncode}] (no output)"
    except subprocess.TimeoutExpired:
        return f"[TIMEOUT] Bash exceeded {timeout_s} seconds."


def web_search(query: str) -> str:
    query = query.strip()
    if not query:
        return "[ERROR] Empty query."
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=10):
                results.append(
                    {"title": r.get("title"), "href": r.get("href"), "body": r.get("body")}
                )
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as e:
        return (
            "[ERROR] WEB_SEARCH failed. Install inside venv:\n"
            "  python -m pip install -U ddgs\n"
            f"Details: {e}"
        )


def web_open(url: str) -> str:
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return "[ERROR] Provide a full URL starting with http:// or https://"
    try:
        import requests
        from bs4 import BeautifulSoup

        r = requests.get(url, timeout=25, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.extract()
        text = soup.get_text("\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return (text[:12000] + "\n\n[TRUNCATED]") if len(text) > 12000 else text
    except Exception as e:
        return f"[ERROR] WEB_OPEN failed: {e}"


def read_file(payload: str, workspace: Path) -> str:
    rel = safe_relpath(payload.strip())
    p = workspace / rel
    if not p.exists():
        return f"[ERROR] File not found: {rel}"
    if p.is_dir():
        return f"[ERROR] Path is a directory: {rel}"
    data = p.read_text(encoding="utf-8", errors="replace")
    return (data[:20000] + "\n\n[TRUNCATED]") if len(data) > 20000 else data


def sanitize_content(content: str) -> str:
    content_clean = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*$", "", content, flags=re.MULTILINE)
    content_clean = re.sub(r"^\s*```\s*$", "", content_clean, flags=re.MULTILINE)
    return content_clean.lstrip("\n")


def write_file(payload: str, workspace: Path) -> str:
    raw = payload.strip()
    if "---" not in raw:
        return "[ERROR] WRITE_FILE format:\npath: somefile.md\n---\ncontent..."
    head, content = raw.split("---", 1)
    m = re.search(r"^\s*path:\s*(.+)\s*$", head.strip(), re.MULTILINE)
    if not m:
        return "[ERROR] WRITE_FILE missing 'path: ...' line."

    rel = safe_relpath(m.group(1).strip())
    proj = current_project()
    if not rel.startswith(proj + "/") and rel != proj:
        rel = f"{proj}/{rel}"

    p = workspace / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(sanitize_content(content), encoding="utf-8")
    return f"[OK] Wrote {rel} ({p.stat().st_size} bytes)"


def append_todo(payload: str, workspace: Path) -> str:
    todo = workspace / f"{current_project()}/todo.md"
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    entry = payload.strip()
    if not entry:
        return "[ERROR] TODO block is empty."
    todo.parent.mkdir(parents=True, exist_ok=True)
    with todo.open("a", encoding="utf-8") as f:
        f.write(f"\n\n## {ts}\n{entry}\n")
    return f"[OK] Appended to {todo.relative_to(workspace)}"


@dataclass
class AgentConfig:
    model: str
    workspace: Path
    timeout: int
    max_steps: int


def llm_chat(model: str, messages: List[Dict[str, str]]) -> str:
    resp = ollama.chat(model=model, messages=messages)
    return resp["message"]["content"].strip()


def extract_tool(text: str) -> Tuple[Optional[str], Optional[str], str]:
    m = TOOL_RE.search(text or "")
    if not m:
        return None, None, text or ""
    tool = m.group(1).upper()
    payload = m.group(2)
    outside = TOOL_RE.sub("", text).strip()
    return tool, payload, outside


def normalize_user_message(s: str) -> str:
    s = s.replace("\r\n", "\n")
    lines = [ln.strip() for ln in s.split("\n")]
    lines = [ln for ln in lines if ln]
    return "\n".join(lines)


def autosave_web_outputs(cfg: AgentConfig, tool: str, out: str) -> None:
    proj = current_project()
    log_dir = cfg.workspace / proj / "datasets"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{tool.lower()}_{stamp}.json" if tool == "WEB_SEARCH" else f"{tool.lower()}_{stamp}.txt"
    (log_dir / fname).write_text(out, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct"))
    parser.add_argument("--workspace", default=os.getenv("AGENT_WORKSPACE", "workspace"))
    parser.add_argument("--timeout", type=int, default=int(os.getenv("AGENT_TIMEOUT", "180")))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("AGENT_MAX_STEPS", "50")))
    args = parser.parse_args()

    cfg = AgentConfig(
        model=args.model,
        workspace=Path(args.workspace).expanduser().resolve(),
        timeout=args.timeout,
        max_steps=args.max_steps,
    )
    cfg.workspace.mkdir(parents=True, exist_ok=True)

    print(f"🧠 Bio-ML Agent ready | model={cfg.model} | workspace={cfg.workspace}")
    print("Çıkmak için: exit / quit\n")

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nÇıkılıyor.")
            break

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            break

        user = normalize_user_message(user)
        allow_web = ("ALLOW_WEB_SEARCH" in user.upper())

        mproj = re.search(r"(?i)\bPROJECT\s*:\s*([a-z0-9_\-]+)", user)
        project = mproj.group(1) if mproj else DEFAULT_PROJECT
        os.environ["AGENT_PROJECT"] = project
        (cfg.workspace / project).mkdir(parents=True, exist_ok=True)

        messages.append({"role": "user", "content": user})

        for _ in range(cfg.max_steps):
            assistant = llm_chat(cfg.model, messages)

            tool, payload, outside = extract_tool(assistant)

            if tool is None:
                py_m = FENCED_PY_RE.search(assistant)
                bash_m = FENCED_BASH_RE.search(assistant)
                if py_m and (not bash_m or len(py_m.group(1)) >= len(bash_m.group(1))):
                    tool, payload = "PYTHON", py_m.group(1)
                    outside = FENCED_PY_RE.sub("", assistant).strip()
                elif bash_m:
                    tool, payload = "BASH", bash_m.group(1)
                    outside = FENCED_BASH_RE.sub("", assistant).strip()
                else:
                    print("\n🤖 Agent:\n", assistant)
                    messages.append({"role": "assistant", "content": assistant})
                    break

            if outside:
                print("\n⚠️ Uyarı: Tool bloğu dışında metin vardı; yine de tool çalıştırılıyor.\n")

            if tool == "PYTHON":
                out = run_python(payload, cfg.workspace, timeout_s=cfg.timeout)
            elif tool == "BASH":
                out = run_bash(payload, cfg.workspace, timeout_s=cfg.timeout)
            elif tool == "WEB_SEARCH":
                if not allow_web and not WEB_SEARCH_DEFAULT_ENABLED:
                    out = "[BLOCKED] WEB_SEARCH is disabled. To enable for this request, include: ALLOW_WEB_SEARCH"
                else:
                    out = web_search(payload)
            elif tool == "WEB_OPEN":
                out = web_open(payload)
            elif tool == "READ_FILE":
                out = read_file(payload, cfg.workspace)
            elif tool == "WRITE_FILE":
                out = write_file(payload, cfg.workspace)
                if out.startswith("[ERROR] WRITE_FILE"):
                    messages.append({"role": "assistant", "content": assistant})
                    messages.append({"role": "user", "content": "WRITE_FILE was invalid. Resend ONLY ONE <WRITE_FILE> block with correct format (path: ... --- ...)."})
                    continue
            elif tool == "TODO":
                out = append_todo(payload, cfg.workspace)
            else:
                out = f"[ERROR] Unknown tool: {tool}"

            if tool in {"WEB_SEARCH", "WEB_OPEN"} and not out.startswith("[BLOCKED]"):
                autosave_web_outputs(cfg, tool, out)

            print(f"\n🛠️ {tool} output:\n{out}\n")

            messages.append({"role": "assistant", "content": assistant})
            messages.append({
                "role": "user",
                "content": f"TOOL_OUTPUT ({tool}):\n{out}\n\nContinue. If done, answer normally (no tool)."
            })
        else:
            print("\n⚠️ Max steps reached. Task may be incomplete.\n")


if __name__ == "__main__":
    main()
