"""Settings from the environment and the project's gitignored .env file. Secrets are never logged."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ENV_FILE = ROOT / ".env"
SAMPLES_DIR = ROOT / "samples"
FRONTEND_DIST = ROOT / "frontend" / "dist"   # the built React app (npm run build)
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _load_env_file() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file()


def data_dir() -> Path:
    """Where the database and blobs live. Read on every call so tests can point CAUSAL_DATA_DIR elsewhere."""
    d = Path(os.environ.get("CAUSAL_DATA_DIR") or ROOT / "data")
    d.mkdir(parents=True, exist_ok=True)
    return d
