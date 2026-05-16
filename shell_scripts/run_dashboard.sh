#!/usr/bin/env bash
set -euo pipefail

STREAMLIT_BIN="${STREAMLIT_BIN:-/playpen-ssd/smerrill/conda_envs/deception/bin/streamlit}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMITMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PORT="${PORT:-8765}"
ADDRESS="${ADDRESS:-0.0.0.0}"

exec "$STREAMLIT_BIN" run "$COMMITMENT_ROOT/src/app.py" \
  --server.headless true \
  --server.address "$ADDRESS" \
  --server.port "$PORT"
