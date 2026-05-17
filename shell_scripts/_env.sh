#!/usr/bin/env bash
# Shared environment resolution helpers for commitment shell scripts.

commitment_shell_dir() {
  cd "$(dirname "${BASH_SOURCE[0]}")" && pwd
}

commitment_root_dir() {
  cd "$(commitment_shell_dir)/.." && pwd
}

resolve_executable_path() {
  local value="${1:-}"
  if [[ -z "$value" ]]; then
    return 1
  fi
  if [[ "$value" == */* ]]; then
    [[ -x "$value" ]] || return 1
    printf '%s\n' "$value"
    return 0
  fi
  command -v "$value" 2>/dev/null
}

resolve_commitment_env_dir() {
  if [[ -n "${COMMITMENT_ENV_DIR:-}" ]]; then
    printf '%s\n' "$COMMITMENT_ENV_DIR"
    return 0
  fi
  printf '%s/.venv\n' "$(commitment_root_dir)"
}

resolve_bootstrap_python() {
  local candidates=()

  if [[ -n "${BOOTSTRAP_PYTHON:-}" ]]; then
    local explicit_bootstrap
    explicit_bootstrap="$(resolve_executable_path "$BOOTSTRAP_PYTHON" || true)"
    if [[ -n "$explicit_bootstrap" ]]; then
      candidates+=("$explicit_bootstrap")
    fi
  fi
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    candidates+=("${CONDA_PREFIX}/bin/python")
  fi
  if command -v python3 >/dev/null 2>&1; then
    candidates+=("$(command -v python3)")
  fi
  if command -v python >/dev/null 2>&1; then
    candidates+=("$(command -v python)")
  fi

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

resolve_commitment_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    local explicit_python
    explicit_python="$(resolve_executable_path "$PYTHON_BIN" || true)"
    if [[ -z "$explicit_python" ]]; then
      echo "Configured PYTHON_BIN is not executable: $PYTHON_BIN" >&2
      return 1
    fi
    printf '%s\n' "$explicit_python"
    return 0
  fi

  local env_dir
  env_dir="$(resolve_commitment_env_dir)"
  if [[ -x "$env_dir/bin/python" ]]; then
    printf '%s\n' "$env_dir/bin/python"
    return 0
  fi

  resolve_bootstrap_python
}

resolve_commitment_python_or_die() {
  local python_bin
  python_bin="$(resolve_commitment_python || true)"
  if [[ -z "$python_bin" ]]; then
    echo "Could not find a usable Python interpreter." >&2
    echo "Run shell_scripts/install_requirements.sh first, or set PYTHON_BIN/BOOTSTRAP_PYTHON." >&2
    exit 1
  fi
  printf '%s\n' "$python_bin"
}

commitment_env_exists() {
  local env_dir
  env_dir="$(resolve_commitment_env_dir)"
  [[ -x "$env_dir/bin/python" ]]
}

print_commitment_env_notice() {
  local python_bin env_dir
  python_bin="$(resolve_commitment_python_or_die)"
  env_dir="$(resolve_commitment_env_dir)"
  if [[ "$python_bin" == "$env_dir/bin/python" ]]; then
    echo "Using commitment environment: $env_dir"
  else
    echo "Using fallback Python interpreter: $python_bin" >&2
    echo "Managed commitment environment not found at: $env_dir" >&2
    echo "To create it, run: bash $(commitment_shell_dir)/install_requirements.sh" >&2
  fi
}
