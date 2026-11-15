import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import warnings
warnings.filterwarnings("ignore")

import pytest


@pytest.fixture(autouse=True)
def _no_real_model_keys(monkeypatch):
    """No test may reach a real model: hide any Gemini key on this machine and never read a .env file.
    (test_orch_gemini.py sets its own fake key on top of this.)"""
    from causality.orchestrator import gemini
    monkeypatch.setattr(gemini, "_env_loaded", True)
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL", "GEMINI_HELPER_MODEL", "GEMINI_RPM"):
        monkeypatch.delenv(k, raising=False)
