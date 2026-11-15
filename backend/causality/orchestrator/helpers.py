"""Tool `custom_checks`: a cheap model writes a short Python script for one narrow statistical
question, our code runs it on a copy of the table and returns the numbers.

    results = helpers.run(df, {"questions": [{"id": "q1", "question": "..."}]})

The writing model reports evidence only (the result keys follow `.claude/agents/check-writer.md`);
it never judges. The script is model-written code, so it runs in its own temp folder, with a
timeout, a minimal environment (no keys or tokens), and its whole process tree is killed when the
time runs out. `run` never raises.
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

MAX_QUESTIONS = 4
RESULT_KEYS = ["question", "method", "statistic", "estimate", "std_error", "ci95_low", "ci95_high",
               "p_value", "n", "detectable", "alternatives", "limitations", "error"]

TOOL_SPEC = {
    "name": "custom_checks",
    "description": (
        "Answer narrow statistical questions that no other tool covers (for example the same "
        "comparison with another control group, a placebo outcome, a balance check on a subset). "
        "For each question a small model writes a Python script that runs on the uploaded table and "
        "reports numbers only: the method, the main statistic, estimate with standard error and 95% "
        "interval, p-value, rows used, the smallest effect it could detect, every alternative "
        "specification it ran, and its limitations. It does not judge the numbers; you do. Ask one "
        "precise question per item, naming the columns. One to four questions per call; they run in "
        "parallel and take up to a few minutes."),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Short id, for example c1."},
                        "question": {"type": "string",
                                     "description": "One precise statistical question naming the columns."},
                    },
                    "required": ["id", "question"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
}

SYSTEM = """You write one Python script that answers one statistical question about a table. Reply with the script only, inside a single ```python code fence.

The script:
- reads data.csv from the current folder (pandas), and may use pandas, numpy, scipy and statsmodels only;
- never uses the network and never touches files outside the current folder;
- must itself write result.json in the current folder, with exactly these keys:
  question (the question as you understood it, one sentence),
  method (what was computed, one or two sentences, including how standard errors were computed),
  statistic (the main number with its name, e.g. "first-stage F = 12.3"),
  estimate, std_error, ci95_low, ci95_high, p_value (numbers, or null when they do not apply),
  n (rows, and groups if any, the calculation used),
  detectable (the smallest effect or violation this test would likely detect with this much data, or null),
  alternatives (list of other reasonable specifications you ran, each {"method": ..., "value": ...}; report all of them, not the most favourable),
  limitations (what this calculation cannot show, one or two sentences),
  error (null, or what went wrong);
- wraps the work in try/except so that on failure it still writes result.json with error filled in;
- converts numpy numbers to plain floats/ints before json.dump.

Report evidence only. Never say whether an assumption holds, passes or fails, and never apply a significance cutoff."""


# ----------------------------------------------------------------------------------------------
# small utilities

def strip_fence(text: str) -> str:
    """Python source from a model reply: the longest fenced block, or the text itself when bare."""
    text = (text or "").replace("\r\n", "\n").strip()
    blocks = re.findall(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)(?:\n[ \t]*```|\Z)", text, flags=re.S)
    if blocks:
        py = [b for lang, b in blocks if lang.lower() in ("python", "py", "python3", "")]
        return max(py or [b for _, b in blocks], key=len).strip() + "\n"
    return text + "\n"


def _child_env(folder: str) -> dict:
    """Only what Python needs; no keys, tokens or user settings."""
    env = {}
    for k in ("SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS",
              "PROCESSOR_ARCHITECTURE", "PATH", "LANG", "LC_ALL"):
        if k in os.environ:
            env[k] = os.environ[k]
    env.update({"TEMP": folder, "TMP": folder, "TMPDIR": folder, "HOME": folder, "USERPROFILE": folder,
                "MPLCONFIGDIR": folder, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1",
                "LOKY_MAX_CPU_COUNT": "4", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2"})
    return env


def _kill_tree(p: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(p.pid)], capture_output=True, timeout=20)
        else:
            import signal
            os.killpg(p.pid, signal.SIGKILL)
    except Exception:  # noqa: BLE001
        pass
    try:
        p.kill()
    except Exception:  # noqa: BLE001
        pass


def _run_proc(cmd: list, cwd: str, env: dict, timeout: float, stdin_text: str | None = None):
    """(returncode, stdout, stderr, timed_out); the whole process tree is killed on timeout."""
    kw = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                         errors="replace", **kw)
    try:
        out, err = p.communicate(input=stdin_text, timeout=timeout)
        return p.returncode, out or "", err or "", False
    except subprocess.TimeoutExpired:
        _kill_tree(p)
        try:
            out, err = p.communicate(timeout=10)
        except Exception:  # noqa: BLE001
            out, err = "", ""
        return None, out or "", err or "", True


def _clean(x, depth=0):
    """JSON-safe copy: NaN/inf -> None, long lists and strings cut."""
    if depth > 6:
        return None
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {str(k): _clean(v, depth + 1) for k, v in list(x.items())[:50]}
    if isinstance(x, list):
        return [_clean(v, depth + 1) for v in x[:50]]
    if isinstance(x, str):
        return x[:2000]
    return x


def _fmt(v) -> str:
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.4g}"
    return str(v)


# ----------------------------------------------------------------------------------------------
# the default writer: headless Claude Code, every tool off

class HeadlessWriter:
    def __init__(self, model: str = "claude-haiku-5-5", timeout: float = 300, claude_bin: str | None = None):
        self.model, self.timeout, self.claude_bin = model, timeout, claude_bin

    def __call__(self, system: str, prompt: str) -> str:
        from . import headless
        exe = self.claude_bin or headless.find_claude()
        if not exe:
            raise RuntimeError("Claude Code was not found. Install it, or set CLAUDE_CODE_BIN to its path.")
        work = tempfile.mkdtemp(prefix="causalityai_writer_")
        try:
            sys_path = os.path.join(work, "system.txt")
            with open(sys_path, "w", encoding="utf-8") as f:
                f.write(system)
            cmd = [exe, "-p", "--model", self.model, "--tools", "", "--strict-mcp-config",
                   "--system-prompt-file", sys_path, "--output-format", "json"]
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            rc, out, err, timed_out = _run_proc(cmd, work, env, self.timeout, stdin_text=prompt)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if timed_out:
            raise RuntimeError(f"the script writer did not answer within {self.timeout:g} s")
        try:
            reply = json.loads(out)
        except Exception:  # noqa: BLE001
            raise RuntimeError(f"the script writer gave no readable reply (exit {rc}): {(err or out)[:300]}") from None
        if reply.get("is_error"):
            raise RuntimeError(f"the script writer reported an error: {str(reply.get('result'))[:300]}")
        return str(reply.get("result") or "")


# ----------------------------------------------------------------------------------------------
# one question

def build_prompt(df, question: str) -> str:
    cols = "\n".join(f"- {c} ({df[c].dtype}, {df[c].nunique(dropna=True)} distinct, "
                     f"{int(df[c].isna().sum())} missing)" for c in df.columns)
    sample = df.head(5).to_csv(index=False)
    return (f"Question: {question}\n\nThe table data.csv has {len(df)} rows and these columns:\n{cols}\n\n"
            f"First 5 rows:\n{sample}\nWrite the script.")


def _read_result(folder: str):
    """(numbers, problem): numbers is the parsed result.json, or None with a plain reason."""
    path = os.path.join(folder, "result.json")
    if not os.path.exists(path):
        return None, "the script did not write result.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        return None, f"result.json is not valid JSON ({e})"
    if not isinstance(data, dict):
        return None, "result.json is not a JSON object"
    missing = [k for k in RESULT_KEYS if k not in data]
    if len(missing) > len(RESULT_KEYS) // 2:
        return None, "result.json is missing most keys: " + ", ".join(missing)
    data = _clean(data)
    for k in missing:
        data[k] = None
    if data.get("error"):
        return data, f"the script reported an error: {data['error']}"
    return data, None


def _attempt(code: str, df_csv: str, timeout: float):
    """Run one script in a fresh folder. (numbers, problem, error_tail)."""
    folder = tempfile.mkdtemp(prefix="causalityai_check_")
    try:
        with open(os.path.join(folder, "data.csv"), "w", encoding="utf-8", newline="") as f:
            f.write(df_csv)
        with open(os.path.join(folder, "check.py"), "w", encoding="utf-8") as f:
            f.write(code)
        rc, out, err, timed_out = _run_proc([sys.executable, "-I", "-X", "utf8", "check.py"], folder,
                                            _child_env(folder), timeout)
        if timed_out:
            return None, f"the script did not finish within {timeout:g} s", f"Timed out after {timeout:g} s.\n{err[-1500:]}"
        numbers, problem = _read_result(folder)
        tail = (err or out)[-2000:]
        if rc != 0 and problem is None:
            problem = f"the script exited with code {rc}"
        if rc != 0 and numbers is None:
            problem = f"the script failed (exit {rc}): " + (tail.strip().splitlines() or ["no output"])[-1][:300]
        return numbers, problem, tail if problem else ""
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def make_lines(r: dict) -> list:
    lines = []
    if r.get("method"):
        lines.append(f"Method: {r['method']}")
    if r.get("statistic") is not None:
        lines.append(f"Statistic: {_fmt(r['statistic'])}.")
    if r.get("estimate") is not None:
        s = f"Estimate {_fmt(r['estimate'])}"
        if r.get("std_error") is not None:
            s += f" (standard error {_fmt(r['std_error'])})"
        if r.get("ci95_low") is not None and r.get("ci95_high") is not None:
            s += f", 95% interval [{_fmt(r['ci95_low'])}, {_fmt(r['ci95_high'])}]"
        if r.get("p_value") is not None:
            s += f", p = {_fmt(r['p_value'])}"
        lines.append(s + ".")
    elif r.get("p_value") is not None:
        lines.append(f"p = {_fmt(r['p_value'])}.")
    if r.get("n") is not None:
        lines.append(f"Rows used: {_fmt(r['n'])}.")
    if r.get("detectable") is not None:
        lines.append(f"Smallest effect it would likely detect: {_fmt(r['detectable'])}.")
    alts = r.get("alternatives") or []
    if isinstance(alts, list) and alts:
        parts = []
        for a in alts[:10]:
            if isinstance(a, dict):
                name = a.get("method") or a.get("name") or "alternative"
                vals = [f"{k} {_fmt(v)}" for k, v in a.items() if k not in ("method", "name")]
                parts.append(f"{name}: {', '.join(vals)}" if vals else str(name))
            else:
                parts.append(_fmt(a))
        lines.append("Other specifications: " + "; ".join(parts) + ".")
    if r.get("limitations"):
        lines.append(f"Limitations: {r['limitations']}")
    if r.get("error"):
        lines.append(f"The script reported an error: {r['error']}")
    return lines


def _one(df_csv: str, df, qid, question, writer, timeout: float) -> dict:
    t0 = time.time()
    res = {"id": qid, "question": question, "ok": False, "lines": [], "numbers": {}, "error": None,
           "code": "", "seconds": 0.0}
    try:
        prompt = build_prompt(df, question)
        code = strip_fence(writer(SYSTEM, prompt))
        res["code"] = code
        numbers, problem, tail = _attempt(code, df_csv, timeout)
        if problem:
            repair = (f"{prompt}\n\nThis script was written for the question:\n```python\n{code}```\n\n"
                      f"It did not work: {problem}\nEnd of the error output:\n{tail[-1500:]}\n\n"
                      "Write the corrected full script.")
            code = strip_fence(writer(SYSTEM, repair))
            res["code"] = code
            numbers, problem, _ = _attempt(code, df_csv, timeout)
        if numbers is not None:
            res["numbers"] = numbers
            res["lines"] = make_lines(numbers)
        res["ok"] = problem is None and numbers is not None
        res["error"] = problem
        if problem and not res["lines"]:
            res["lines"] = [f"No numbers: {problem}."]
    except Exception as e:  # noqa: BLE001
        res["ok"], res["error"] = False, f"{type(e).__name__}: {e}"[:500]
        res["lines"] = res["lines"] or [f"No numbers: {res['error']}"]
    res["seconds"] = round(time.time() - t0, 2)
    return res


# ----------------------------------------------------------------------------------------------
# public

def run(df, args, writer=None, model: str = "claude-haiku-5-5", timeout: float = 150,
        max_workers: int = 4) -> list:
    """One result dict per question, in order. Never raises."""
    try:
        qs = (args or {}).get("questions") if isinstance(args, dict) else None
        if not isinstance(qs, list) or not qs:
            return [{"id": None, "question": "", "ok": False, "lines": ["No numbers: no questions were given."],
                     "numbers": {}, "error": "no questions were given", "code": "", "seconds": 0.0}]
        items = []
        for i, q in enumerate(qs):
            q = q if isinstance(q, dict) else {"question": str(q)}
            items.append((str(q.get("id") or f"q{i + 1}"), str(q.get("question") or "").strip()))
        if writer is None:
            from . import gemini
            writer = gemini.GeminiWriter() if gemini.available() else HeadlessWriter(model=model)
        df_csv = df.to_csv(index=False)

        def task(i_item):
            i, (qid, question) = i_item
            if i >= MAX_QUESTIONS:
                err = f"at most {MAX_QUESTIONS} questions per call"
            elif not question:
                err = "the question is empty"
            else:
                return _one(df_csv, df, qid, question, writer, timeout)
            return {"id": qid, "question": question, "ok": False, "lines": [f"No numbers: {err}."],
                    "numbers": {}, "error": err, "code": "", "seconds": 0.0}

        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
            return list(ex.map(task, enumerate(items)))
    except Exception as e:  # noqa: BLE001
        return [{"id": None, "question": "", "ok": False, "lines": [f"No numbers: {e}"], "numbers": {},
                 "error": f"{type(e).__name__}: {e}"[:500], "code": "", "seconds": 0.0}]
