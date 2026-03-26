#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-7860}"
DEVICE="${DEVICE:-0}"
CKPT_DIR="${WAN_TI2V_CKPT_DIR:-/mnt/nas10_shared/models/Wan2.2-TI2V-5B}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed or not on PATH" >&2
  exit 1
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Missing .venv. Run: uv venv .venv --python 3.10 && uv sync --python .venv/bin/python --extra dev --extra demo" >&2
  exit 1
fi

if ss -ltn "( sport = :${PORT} )" | grep -q ":${PORT}"; then
  echo "Port ${PORT} is already in use" >&2
  exit 1
fi

echo "Starting Wan2.2 TI2V demo on http://${HOST}:${PORT}"
echo "Checkpoint: ${CKPT_DIR}"
echo "GPU: ${DEVICE}"

export PYTHONUNBUFFERED=1
export WAN_TI2V_CKPT_DIR="$CKPT_DIR"

exec .venv/bin/python app_ti2v.py \
  --host "$HOST" \
  --port "$PORT" \
  --device "$DEVICE" \
  --ckpt-dir "$CKPT_DIR" \
  --no-open-browser