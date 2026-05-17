#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMMITMENT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/_env.sh"

ENV_DIR="${COMMITMENT_ENV_DIR:-$COMMITMENT_ROOT/.venv}"
REQUIREMENTS_PATH="${REQUIREMENTS_PATH:-$COMMITMENT_ROOT/requirements.txt}"
BOOTSTRAP_PYTHON_BIN="${BOOTSTRAP_PYTHON:-}"
RECREATE=0
UPGRADE_PIP=1

usage() {
  cat <<'EOF'
Usage:
  install_requirements.sh [options]

This creates a local virtual environment for commitment under `commitment/.venv`
and installs `commitment/requirements.txt` into it.

Optional:
  --env_dir DIR             Override the target virtualenv directory
  --requirements FILE       Override the requirements file path
  --bootstrap_python PATH   Python used to create the virtualenv
  --recreate                Delete and recreate the virtualenv
  --no_upgrade_pip          Skip `pip/setuptools/wheel` upgrade
  --help                    Show this message

Examples:
  bash install_requirements.sh
  bash install_requirements.sh --bootstrap_python "$(command -v python3)"
  bash install_requirements.sh --recreate
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env_dir)
      ENV_DIR="$2"
      shift 2
      ;;
    --requirements)
      REQUIREMENTS_PATH="$2"
      shift 2
      ;;
    --bootstrap_python)
      BOOTSTRAP_PYTHON_BIN="$2"
      shift 2
      ;;
    --recreate)
      RECREATE=1
      shift
      ;;
    --no_upgrade_pip)
      UPGRADE_PIP=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "$REQUIREMENTS_PATH" ]]; then
  echo "Requirements file not found: $REQUIREMENTS_PATH" >&2
  exit 1
fi

if [[ -z "$BOOTSTRAP_PYTHON_BIN" ]]; then
  BOOTSTRAP_PYTHON_BIN="$(resolve_bootstrap_python || true)"
fi

BOOTSTRAP_PYTHON_BIN="$(resolve_executable_path "$BOOTSTRAP_PYTHON_BIN" || true)"

if [[ -z "$BOOTSTRAP_PYTHON_BIN" ]]; then
  echo "Could not find a usable bootstrap Python." >&2
  echo "Set --bootstrap_python PATH or BOOTSTRAP_PYTHON." >&2
  exit 1
fi

if [[ "$RECREATE" == "1" && -d "$ENV_DIR" ]]; then
  echo "Recreating virtualenv: $ENV_DIR"
  rm -rf "$ENV_DIR"
fi

if [[ ! -d "$ENV_DIR" ]]; then
  echo "Creating virtualenv: $ENV_DIR"
  "$BOOTSTRAP_PYTHON_BIN" -m venv "$ENV_DIR"
else
  echo "Using existing virtualenv: $ENV_DIR"
fi

VENV_PYTHON="$ENV_DIR/bin/python"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Virtualenv Python not found after creation: $VENV_PYTHON" >&2
  exit 1
fi

echo "Bootstrap Python: $BOOTSTRAP_PYTHON_BIN"
echo "Virtualenv Python: $VENV_PYTHON"
echo "Requirements file: $REQUIREMENTS_PATH"

if [[ "$UPGRADE_PIP" == "1" ]]; then
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
fi

"$VENV_PYTHON" -m pip install -r "$REQUIREMENTS_PATH"

echo "Verifying key imports..."
"$VENV_PYTHON" - <<'PY'
modules = [
    "datasets",
    "transformers",
    "torch",
    "numpy",
    "pandas",
    "scipy",
    "streamlit",
    "matplotlib",
    "tqdm",
]
for name in modules:
    __import__(name)
print("Import verification passed.")
PY

echo
echo "Commitment environment is ready."
echo "Environment dir: $ENV_DIR"
echo "Python: $VENV_PYTHON"
echo "Streamlit: $ENV_DIR/bin/streamlit"
echo
echo "The commitment shell scripts will use this environment automatically."
