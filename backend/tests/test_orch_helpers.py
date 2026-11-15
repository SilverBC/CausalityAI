import os
import threading
import time

import numpy as np
import pandas as pd
import pytest

from causality.orchestrator import helpers

GOOD = '''Here is the script.
```python
import json
import pandas as pd
df = pd.read_csv("data.csv")
d = df.groupby("t")["y"].mean()
est = float(d[1] - d[0])
json.dump({"question": "QUESTION", "method": "difference in means", "statistic": f"diff = {est:.3f}",
           "estimate": est, "std_error": 0.1, "ci95_low": est - 0.2, "ci95_high": est + 0.2,
           "p_value": 0.01, "n": len(df), "detectable": None,
           "alternatives": [{"method": "median difference", "value": 1.0}],
           "limitations": "Only a comparison of means.", "error": None}, open("result.json", "w"))
```
'''

FAIL = '''```python
x = 1 / 0
```'''

SLEEP = '''```python
import time
time.sleep(60)
```'''


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    t = rng.integers(0, 2, 200)
    return pd.DataFrame({"t": t, "y": 2.0 * t + rng.normal(size=200)})


class Recorder:
    def __init__(self, replies):
        self.replies, self.prompts, self.lock = list(replies), [], threading.Lock()

    def __call__(self, system, prompt):
        with self.lock:
            self.prompts.append(prompt)
            return self.replies[min(len(self.prompts) - 1, len(self.replies) - 1)]


def test_tool_spec_shape():
    s = helpers.TOOL_SPEC
    assert s["name"] == "custom_checks" and s["strict"] is True
    assert s["input_schema"]["additionalProperties"] is False
    item = s["input_schema"]["properties"]["questions"]["items"]
    assert set(item["required"]) == set(item["properties"]) and item["additionalProperties"] is False


def test_strip_fence_variants():
    assert helpers.strip_fence("```python\nprint(1)\n```") == "print(1)\n"
    assert helpers.strip_fence("```py\r\nprint(1)\r\n```\nthanks") == "print(1)\n"
    assert helpers.strip_fence("print(2)") == "print(2)\n"
    assert helpers.strip_fence("text\n```python\nprint(3)\n") == "print(3)\n"          # unclosed fence
    two = "```\na=1\n```\nand\n```python\nimport os\nprint(os.getcwd())\n```"
    assert helpers.strip_fence(two) == "import os\nprint(os.getcwd())\n"


def test_good_script(df):
    w = Recorder([GOOD])
    [r] = helpers.run(df, {"questions": [{"id": "c1", "question": "Difference in y by t?"}]}, writer=w)
    assert r["ok"] and r["error"] is None and r["id"] == "c1"
    assert 1.5 < r["numbers"]["estimate"] < 2.5 and r["numbers"]["n"] == 200
    assert any("95% interval" in line for line in r["lines"])
    assert any("median difference" in line for line in r["lines"])
    assert "read_csv" in r["code"] and "```" not in r["code"]
    assert len(w.prompts) == 1 and "Difference in y by t?" in w.prompts[0] and "t (int" in w.prompts[0]


def test_fail_then_repaired(df):
    w = Recorder([FAIL, GOOD])
    [r] = helpers.run(df, {"questions": [{"id": "c1", "question": "q"}]}, writer=w)
    assert r["ok"] and len(w.prompts) == 2
    assert "ZeroDivisionError" in w.prompts[1] and "1 / 0" in w.prompts[1]


def test_never_works(df):
    w = Recorder([FAIL])
    [r] = helpers.run(df, {"questions": [{"id": "c1", "question": "q"}]}, writer=w)
    assert not r["ok"] and "ZeroDivisionError" in r["error"] and len(w.prompts) == 2
    assert r["lines"] and r["numbers"] == {}


def test_script_reported_error_is_not_ok(df):
    bad = GOOD.replace('"error": None', '"error": "column missing"')
    [r] = helpers.run(df, {"questions": [{"id": "c1", "question": "q"}]}, writer=Recorder([bad]))
    assert not r["ok"] and "column missing" in r["error"]


def test_timeout_kills(df):
    t0 = time.time()
    [r] = helpers.run(df, {"questions": [{"id": "c1", "question": "q"}]}, writer=Recorder([SLEEP]), timeout=2)
    assert not r["ok"] and "within 2 s" in r["error"]
    assert time.time() - t0 < 40


def test_writer_exception_does_not_escape(df):
    def boom(system, prompt):
        raise RuntimeError("no model")
    [r] = helpers.run(df, {"questions": [{"id": "c1", "question": "q"}]}, writer=boom)
    assert not r["ok"] and "no model" in r["error"]


def test_bad_args_never_raise(df):
    assert helpers.run(df, None, writer=Recorder([GOOD]))[0]["ok"] is False
    assert helpers.run(df, {"questions": []}, writer=Recorder([GOOD]))[0]["ok"] is False
    rs = helpers.run(df, {"questions": [{"id": f"c{i}", "question": "q"} for i in range(5)]},
                     writer=Recorder([GOOD]))
    assert [r["ok"] for r in rs] == [True] * 4 + [False]


def test_three_in_parallel_keep_order(df):
    def writer(system, prompt):
        tag = prompt.split("Question: ")[1].split("\n")[0]
        delay = {"A": 1.5, "B": 0.0, "C": 0.7}[tag]
        return GOOD.replace("import json", f"import json, time\ntime.sleep({delay})").replace("QUESTION", tag)
    t0 = time.time()
    rs = helpers.run(df, {"questions": [{"id": x, "question": x} for x in "ABC"]}, writer=writer)
    assert [r["id"] for r in rs] == ["A", "B", "C"]
    assert [r["numbers"]["question"] for r in rs] == ["A", "B", "C"]
    assert all(r["ok"] for r in rs)
    assert time.time() - t0 < 15


def test_child_env_has_no_secrets(df, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    monkeypatch.setenv("MY_TOKEN", "tok")
    script = GOOD.replace('"QUESTION"', 'sorted(__import__("os").environ)').replace(
        '"limitations": "Only a comparison of means."',
        '"limitations": __import__("os").environ.get("LOKY_MAX_CPU_COUNT")')
    [r] = helpers.run(df, {"questions": [{"id": "c1", "question": "q"}]}, writer=Recorder([script]))
    assert r["ok"]
    keys = r["numbers"]["question"]
    assert "ANTHROPIC_API_KEY" not in keys and "MY_TOKEN" not in keys
    assert r["numbers"]["limitations"] == "4"


@pytest.mark.skipif(not os.environ.get("CAUSALITYAI_LIVE"), reason="set CAUSALITYAI_LIVE=1 to call the real writer")
def test_live_default_writer(df):
    [r] = helpers.run(df, {"questions": [{"id": "c1", "question": "What is the difference in mean y between t=1 and t=0?"}]})
    assert r["ok"], r["error"]
    assert 1.5 < r["numbers"]["estimate"] < 2.5
