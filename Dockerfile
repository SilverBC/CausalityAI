# Causal Audit in one image: the React app is built in the first stage, then served by FastAPI.
#   docker compose up --build        (see compose.yaml)  →  http://localhost:8000

# ---- web app ---------------------------------------------------------------------------------------------
FROM node:20-slim AS web
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- server ----------------------------------------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CAUSAL_DATA_DIR=/data
# libgomp1: the OpenMP runtime lightgbm (pulled in by EconML) needs on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY backend/ ./backend/
COPY samples/ ./samples/
COPY --from=web /app/frontend/dist ./frontend/dist
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["python", "-m", "uvicorn", "app:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]
