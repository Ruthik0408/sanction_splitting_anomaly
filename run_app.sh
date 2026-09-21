#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"

# Production startup uses the already-downloaded, validated model artifacts.
# Override either variable with 0 only when intentionally downloading models.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

cd "$ROOT_DIR"

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  npm --prefix "$FRONTEND_DIR" install
fi

echo "Building frontend..."
npm --prefix "$FRONTEND_DIR" run build

if [[ ! -x "$ROOT_DIR/venv/bin/python" ]] || \
   ! "$ROOT_DIR/venv/bin/python" -c 'import uvicorn' >/dev/null 2>&1; then
  echo "Missing Python virtual environment or uvicorn. Create/activate the venv and install requirements first." >&2
  echo "Expected setup: source venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

OLLAMA_PID=""
if grep -Eq '^SEMANTIC_API_URL=http://(127\.0\.0\.1|localhost):11434/' .env 2>/dev/null; then
  if ! curl --silent --fail --max-time 2 "http://127.0.0.1:11434/api/version" >/dev/null; then
    if ! command -v ollama >/dev/null 2>&1; then
      echo "SEMANTIC_API_URL uses local Ollama, but the ollama command is not installed." >&2
      exit 1
    fi
    echo "Starting local Ollama on CPU..."
    CUDA_VISIBLE_DEVICES="" ollama serve > /tmp/sanction-splitting-ollama.log 2>&1 &
    OLLAMA_PID=$!
    for _ in {1..30}; do
      if curl --silent --fail --max-time 2 "http://127.0.0.1:11434/api/version" >/dev/null; then
        break
      fi
      if ! kill -0 "$OLLAMA_PID" 2>/dev/null; then
        echo "Local Ollama failed to start. See /tmp/sanction-splitting-ollama.log" >&2
        exit 1
      fi
      sleep 1
    done
  fi
  SEMANTIC_MODEL_NAME="$(awk -F= '/^SEMANTIC_MODEL=/{sub(/^[^=]*=/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, ""); print; exit}' .env)"
  if [[ -z "$SEMANTIC_MODEL_NAME" ]] || ! ollama show "$SEMANTIC_MODEL_NAME" >/dev/null 2>&1; then
    echo "Configured Ollama model is not installed: ${SEMANTIC_MODEL_NAME:-<empty>}" >&2
    echo "Install it with: ollama pull ${SEMANTIC_MODEL_NAME:-MODEL_NAME}" >&2
    exit 1
  fi
fi

echo "Starting backend and UI at http://localhost:${PORT}/ui"
"$ROOT_DIR/venv/bin/python" -m uvicorn scripts.api_server:app --host "$HOST" --port "$PORT" &
SERVER_PID=$!
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
  if [[ -n "$OLLAMA_PID" ]]; then
    kill "$OLLAMA_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

for _ in {1..30}; do
  if curl --silent --fail "http://127.0.0.1:${PORT}/health" >/dev/null; then
    google-chrome "http://localhost:${PORT}/ui" >/dev/null 2>&1 &
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    wait "$SERVER_PID"
    exit 1
  fi
  sleep 1
done

wait "$SERVER_PID"
