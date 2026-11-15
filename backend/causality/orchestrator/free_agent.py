"""The inverted design on Gemini: the model analyses the table itself, by writing and running its own
Python, and CausalityAI's calculators are optional helpers it may reach for.

    from causality.orchestrator import free_agent
    result = free_agent.run(folder, instructions, context=facts_text, helpers=True)

`folder` holds `data.csv`. The model works through three actions:

    run_python       {"code": "..."}            our code runs the script in `folder` and returns what it printed
    layout_facts, design_tests, estimate_effect, general_checks, sensitivity
                                                this software's own analysis tools (only when helpers=True)
    write_answer     {"answer_json": "{...}"}   writes answer.json in `folder` and ends the run

With helpers=False and no context this is the plain baseline: the same model, the same loop, nothing of ours.

The model call goes through gemini.generate (pacing, retries, plain errors). Scripts the model writes run on
this machine, inside `folder`, with a time limit and an environment that holds no keys or tokens.
Nothing here raises: a failure comes back as {"status": "error", "error": "..."}.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from . import gemini

MAX_OUTPUT = 6000            # characters of script output sent back to the model
SCRIPT_TIMEOUT = 120
HELPER_TIMEOUT = 180
HELPER_COMMANDS = ("layout", "design-tests", "estimate", "checks", "sensitivity")
_SECRET_WORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH")

PROTOCOL = """

# How to act in this session

Reply only with one JSON object that has two fields:
- `text`: a short note on what you are doing (may be an empty string).
- `tool_calls`: a list of calls, each {"name": <tool name>, "input_json": <the arguments object for that
  tool, written as a JSON string>}.

Make one call per reply unless the calls are independent. The result of each call comes back in the next
message. You cannot see the table except through what your code prints, so print what you need, briefly.
Always finish by calling `write_answer`; a reply with no call does not end the task.

The tools:

"""

RUN_PYTHON = {
    "name": "run_python",
    "description": ("Run a Python script in the folder that holds data.csv and get back what it printed "
                    "(standard output and errors, shortened if long). pandas, numpy, scipy and statsmodels "
                    f"are installed. Each call is a fresh process with a {SCRIPT_TIMEOUT} second limit; "
                    "nothing is kept between calls except files you write."),
    "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
}
_COL = {"type": "string", "description": "A column name exactly as it appears in data.csv."}
_OPT_COL = {"type": ["string", "null"], "description": "A column name, or null when this design does not use it."}
_COVS = {"type": ["array", "null"], "items": {"type": "string"},
         "description": "Columns to adjust for. null = every column that is not given a role."}
_DESIGN_PROPS = {
    "design": {"type": "string", "enum": ["experiment", "did", "iv", "rd", "observational"],
               "description": "experiment = randomized assignment; did = difference-in-differences on units "
                              "observed over time; iv = instrumental variable; rd = regression discontinuity at a "
                              "cutoff; observational = adjustment for measured columns."},
    "treatment": _COL, "outcome": _COL,
    "unit": _OPT_COL, "time": _OPT_COL, "instrument": _OPT_COL, "running": _OPT_COL,
    "cutoff": {"type": ["number", "null"], "description": "The threshold on the running column (rd only)."},
    "covariates": _COVS,
}
_DESIGN_SCHEMA = {"type": "object", "properties": _DESIGN_PROPS, "required": ["design", "treatment", "outcome"]}

# This software's own analysis tools. Each description says exactly what the tool computes, what comes
# back, and what it is good for. Nothing here tells the model how to work or what to conclude.
SOFTWARE_TOOLS = [
    {"name": "layout_facts",
     "command": "layout",
     "description": (
         "CausalityAI's table scanner. It examines every column of data.csv against the treatment column and "
         "reports, with numbers: (1) whether treatment switches from 0 to 1 at one exact value of some numeric "
         "column, and what that value is; (2) whether the rows are repeated observations of the same units over "
         "time, how many units switch treatment on, and in which periods; (3) how well all the other columns "
         "together predict treatment; (4) columns that shift treatment while nothing else in the table predicts "
         "them; (5) columns strongly tied to treatment. It runs in a few seconds and needs only the treatment and "
         "outcome column names. It is useful because these patterns show how treatment was assigned in this "
         "table, they do not appear in a regression table or in summary statistics, and finding them by hand "
         "means testing every column."),
     "input_schema": {"type": "object", "properties": {"treatment": _COL, "outcome": _COL},
                      "required": ["treatment", "outcome"]}},
    {"name": "design_tests",
     "command": "design-tests",
     "description": (
         "CausalityAI's diagnostic tests for one study design, run on data.csv with tested code. It returns each "
         "test's statistic and p-value. For did: coefficients for each period before and after treatment starts, "
         "a joint test of the pre-treatment periods, and a placebo test that pretends treatment started earlier. "
         "For rd: a density test for rows piling up on one side of the cutoff, and tests for jumps in the other "
         "columns at the cutoff. For iv: the first-stage F statistic, balance of the other columns across "
         "instrument values, and a Durbin-Wu-Hausman comparison with plain regression. For experiment and "
         "observational: standardized differences between treated and untreated rows on every other column and a "
         "joint permutation test. It also lists which requirements of the design no data can test. It is useful "
         "because these tests take a lot of code to write correctly and their results show whether the numbers "
         "from that design can be relied on in this table."),
     "input_schema": _DESIGN_SCHEMA},
    {"name": "estimate_effect",
     "command": "estimate",
     "description": (
         "CausalityAI's effect estimators, run on data.csv with tested code. It returns the estimated effect of "
         "treatment on outcome with its standard error, p-value and intervals at 90, 95, 98 and 99 percent, and "
         "the same effect from one or two alternative methods for comparison. For did: Callaway-Sant'Anna, which "
         "handles units that start treatment in different periods, next to two-way fixed effects. For rd: local "
         "linear regression at the cutoff with a robust bias-corrected interval, at several window widths. For "
         "iv: two-stage least squares with an Anderson-Rubin set that stays valid when the instrument is weak, "
         "next to plain regression. For experiment: adjusted difference in means. For observational: a doubly "
         "robust estimator with cross-fitting. It is useful as the estimate itself or as a second number to hold "
         "against your own."),
     "input_schema": _DESIGN_SCHEMA},
    {"name": "general_checks",
     "command": "checks",
     "description": (
         "CausalityAI's four general checks on a treated-versus-untreated comparison in data.csv: balance (how "
         "different the two groups are on every other column), overlap (whether each treated row has comparable "
         "untreated rows, from two propensity models), missing outcomes (how many, and whether more in one "
         "group), and sensitivity to an unmeasured cause. Returns the numbers for each. It is useful for seeing "
         "at once how alike the two groups are before any estimate is trusted."),
     "input_schema": {"type": "object", "properties": {"treatment": _COL, "outcome": _COL, "covariates": _COVS},
                      "required": ["treatment", "outcome"]}},
    {"name": "sensitivity",
     "command": "sensitivity",
     "description": (
         "CausalityAI's sensitivity analysis for an estimate from data.csv. It returns the robustness value (how "
         "strongly an unmeasured cause would have to be tied to both treatment and outcome to erase the effect), "
         "the same strength compared with the strongest measured column, and the E-value. It is useful for "
         "saying how fragile an estimate is to something the table does not contain."),
     "input_schema": _DESIGN_SCHEMA},
]
_SOFTWARE = {t["name"]: t for t in SOFTWARE_TOOLS}


def _cli_args(tool: dict, args: dict) -> list:
    """Turn a software tool's arguments into free_cli.py options."""
    out = []
    if "design" in tool["input_schema"]["properties"]:
        out += ["--design", str(args.get("design") or "observational")]
    for k in ("treatment", "outcome", "unit", "time", "instrument", "running"):
        if k in tool["input_schema"]["properties"] and args.get(k) not in (None, ""):
            out += [f"--{k}", str(args[k])]
    if "cutoff" in tool["input_schema"]["properties"] and args.get("cutoff") not in (None, ""):
        out += ["--cutoff", str(args["cutoff"])]
    cov = args.get("covariates")
    if isinstance(cov, list) and cov:
        out += ["--covariates", ",".join(str(c) for c in cov)]
    return out
WRITE_ANSWER = {
    "name": "write_answer",
    "description": "Write answer.json and end the task. answer_json is the whole answer object as a JSON string.",
    "input_schema": {"type": "object", "properties": {"answer_json": {"type": "string"}},
                     "required": ["answer_json"]},
}


def child_env() -> dict:
    """The environment for scripts the model writes: what Python needs, and no keys or tokens."""
    env = {k: v for k, v in os.environ.items() if not any(w in k.upper() for w in _SECRET_WORDS)}
    env["LOKY_MAX_CPU_COUNT"] = "4"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _shorten(text: str, limit: int = MAX_OUTPUT) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    half = limit // 2
    return text[:half] + f"\n... [{len(text) - limit} characters left out] ...\n" + text[-half:]


def _run(cmd: list, folder: str, timeout: int) -> str:
    try:
        p = subprocess.run(cmd, cwd=folder, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout, env=child_env())
    except subprocess.TimeoutExpired:
        return f"The run was stopped after {timeout} seconds. Make it faster or do less in one call."
    except Exception as e:  # noqa: BLE001
        return f"The run could not start: {type(e).__name__}: {e}"
    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    if p.returncode != 0:
        return _shorten((out + "\n" if out else "") + f"[exit code {p.returncode}]\n" + err)
    return _shorten(out or "(the script printed nothing)")


def _run_python(folder: str, code: str, n: int) -> str:
    if not isinstance(code, str) or not code.strip():
        return "No code was given."
    path = os.path.join(folder, f"step_{n:02d}.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(code)
    return _run([sys.executable, "-I", "-X", "utf8", os.path.basename(path)], folder, SCRIPT_TIMEOUT)


def _run_helper(folder: str, command, args) -> str:
    if command not in HELPER_COMMANDS:
        return f"Unknown helper command. Use one of: {', '.join(HELPER_COMMANDS)}."
    if not isinstance(args, list) or not all(isinstance(a, (str, int, float)) for a in args):
        return "args must be a list of strings."
    cli = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "free_cli.py")
    return _run([sys.executable, "-X", "utf8", cli, command, "data.csv", *[str(a) for a in args]], folder,
                HELPER_TIMEOUT)


def _write_answer(folder: str, raw):
    """(ok, message). The answer must be one JSON object."""
    try:
        obj = raw if isinstance(raw, dict) else json.loads(raw)
    except Exception as e:  # noqa: BLE001
        return False, f"answer_json was not valid JSON ({e}). Call write_answer again with a valid JSON object."
    if not isinstance(obj, dict):
        return False, "answer_json must be a JSON object. Call write_answer again."
    with open(os.path.join(folder, "answer.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    return True, "answer.json written."


def run(folder: str, instructions: str, context: str | None = None, helpers: bool = False,
        model: str | None = None, max_turns: int = 12, transport=None, sleep=time.sleep) -> dict:
    """Run one audit. Returns {"status": "done"|"error", "error", "turns", "requests", "input_tokens",
    "output_tokens", "seconds", "python_runs", "helper_calls": [commands], "model"} and writes
    answer.json (when done) and transcript.json in `folder`."""
    t0 = time.time()
    model = model or gemini.model_name()
    out = {"status": "error", "error": None, "turns": 0, "requests": 0, "input_tokens": 0, "output_tokens": 0,
           "seconds": 0.0, "python_runs": 0, "helper_calls": [], "model": model, "helpers_offered": bool(helpers),
           "calls": []}                      # every action in order: {"tool", "args", "result"}
    offered = [{k: v for k, v in t.items() if k != "command"} for t in SOFTWARE_TOOLS] if helpers else []
    tools = [RUN_PYTHON] + offered + [WRITE_ANSWER]
    system = instructions.strip() + PROTOCOL + json.dumps(tools, indent=1)
    first = "The table is data.csv in your folder."
    if context:
        first += "\n\n" + context.strip()
    log = [{"role": "user", "text": first}]
    nudged = False
    try:
        for _ in range(max_turns):
            convo = "\n\n".join(f"=== {m['role'].upper()} ===\n{m['text']}" for m in log)
            body = {"systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": convo + "\n\n=== YOUR NEXT REPLY (JSON only) ==="}]}],
                    "generationConfig": {"responseMimeType": "application/json",
                                         "responseSchema": gemini.REPLY_SCHEMA}}
            payload = gemini.generate(model, body, 300, transport, sleep)
            out["requests"] += 1
            out["turns"] += 1
            u = payload.get("usageMetadata") or {}
            out["input_tokens"] += int(u.get("promptTokenCount") or 0)
            out["output_tokens"] += int(u.get("candidatesTokenCount") or 0) + int(u.get("thoughtsTokenCount") or 0)
            reply = gemini._parse_reply(gemini._text_of(payload))
            calls = [c for c in (reply.get("tool_calls") or []) if isinstance(c, dict) and c.get("name")]
            said = str(reply.get("text") or "").strip()
            log.append({"role": "assistant", "text": json.dumps(
                {"text": said, "tool_calls": [{"name": c.get("name"), "input_json": c.get("input_json", c.get("input"))}
                                              for c in calls]}, ensure_ascii=False)})
            if not calls:
                if nudged:
                    out["error"] = "The model stopped without writing an answer."
                    break
                nudged = True
                log.append({"role": "user", "text": "No tool was called. Continue the analysis, or call "
                                                    "write_answer with your answer."})
                continue
            results, done = [], False
            for c in calls:
                args, problem = gemini._call_input(c)
                name = str(c["name"])
                if problem:
                    results.append(f"[{name}] {problem}")
                elif name == "run_python":
                    out["python_runs"] += 1
                    results.append("[run_python]\n" + _run_python(folder, args.get("code"), out["python_runs"]))
                elif name in _SOFTWARE and helpers:
                    tool = _SOFTWARE[name]
                    out["helper_calls"].append(tool["command"])
                    results.append(f"[{name}]\n" + _run_helper(folder, tool["command"], _cli_args(tool, args)))
                elif name == "write_answer":
                    # small models often pass the answer object itself instead of wrapping it as a string
                    raw = args.get("answer_json") if "answer_json" in args else (args or None)
                    ok, msg = _write_answer(folder, raw)
                    results.append("[write_answer] " + msg)
                    done = done or ok
                else:
                    results.append(f"[{name}] There is no such tool.")
            for c, res in zip(calls, results):
                c_args, _problem = gemini._call_input(c)
                out["calls"].append({"tool": str(c["name"]), "args": c_args,
                                     "result": res.split("\n", 1)[-1][:4000]})
            log.append({"role": "user", "text": "\n\n".join(results)})
            if done:
                out["status"] = "done"
                break
        else:
            out["error"] = f"No answer after {max_turns} turns."
    except gemini.GeminiError as e:
        out["error"] = str(e)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    out["seconds"] = round(time.time() - t0, 1)
    try:
        with open(os.path.join(folder, "transcript.json"), "w", encoding="utf-8") as f:
            json.dump({"result": out, "messages": log}, f, indent=1, ensure_ascii=False)
    except OSError:
        pass
    return out
