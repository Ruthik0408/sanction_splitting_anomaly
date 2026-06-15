#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"

cd "$ROOT_DIR"

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Installing frontend dependencies..."
  npm --prefix "$FRONTEND_DIR" install
fi

echo "Building frontend..."
npm --prefix "$FRONTEND_DIR" run build

if [[ ! -x "$ROOT_DIR/venv/bin/uvicorn" ]]; then
  echo "Missing venv/bin/uvicorn. Activate/create the Python venv and install requirements first." >&2
  echo "Expected setup: source venv/bin/activate && pip install -r requirements.txt" >&2
  exit 1
fi

echo "Starting backend and UI at http://localhost:${PORT}/ui"
exec "$ROOT_DIR/venv/bin/uvicorn" scripts.api_server:app --host "$HOST" --port "$PORT"
