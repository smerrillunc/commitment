#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMITMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/_env.sh"
PYTHON_BIN="$(resolve_commitment_python_or_die)"

PORT="${PORT:-8765}"
ADDRESS="${ADDRESS:-0.0.0.0}"

print_commitment_env_notice

exec "$PYTHON_BIN" -m streamlit run "$COMMITMENT_ROOT/src/app.py" \
  --server.headless true \
  --server.address "$ADDRESS" \
  --server.port "$PORT"
