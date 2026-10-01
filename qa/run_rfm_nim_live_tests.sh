#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENGINE_ROOT="$WORKSPACE_ROOT/packages/sdm-relational-engine"
VENV_DIR="${RFM_NIM_LIVE_VENV:-$ENGINE_ROOT/.tmp/rfm-nim-live-venv}"
BASE_URL="${KUMO_RELATIONAL_NIM_BASE_URL:-${RFM_NIM_BASE_URL:-}}"
MODE=smoke
REBUILD_ENV=0

usage() {
  cat <<'USAGE'
Run URL-driven KumoRelational NIM live validation.

Usage:
  qa/run_rfm_nim_live_tests.sh --url <base-url> [--full] [pytest args...]

Options:
  --url URL       KumoRelational NIM service root. KUMO_RELATIONAL_NIM_BASE_URL
                  (or the older RFM_NIM_BASE_URL) is also accepted.
  --full          Run smoke and extended live validation. Default: smoke only.
  --rebuild-env   Recreate the script-owned virtualenv before running.
  -h, --help      Show this help.

Optional environment:
  KUMO_RELATIONAL_NIM_API_KEY  X-API-Key value; never printed by this script.
                           RFM_NIM_API_KEY is still honoured.
  RFM_NIM_VERIFY_SSL       Set 0/false/no to disable TLS verification.
  RFM_NIM_TIMEOUT_SECONDS  Per-request timeout. Default: 30.
  RFM_NIM_PYTHON           Existing Python interpreter to use instead of the
                           script-owned virtualenv.
USAGE
}

pytest_args=()
while (($#)); do
  case "$1" in
    --url)
      [[ $# -ge 2 ]] || { echo 'error: --url requires a value' >&2; exit 2; }
      BASE_URL="$2"
      shift 2
      ;;
    --full)
      MODE=full
      shift
      ;;
    --rebuild-env)
      REBUILD_ENV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      pytest_args+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$BASE_URL" ]]; then
  echo 'error: pass --url or set KUMO_RELATIONAL_NIM_BASE_URL' >&2
  exit 2
fi

if [[ -n "${RFM_NIM_PYTHON:-}" ]]; then
  PYTHON="$RFM_NIM_PYTHON"
  [[ -x "$PYTHON" ]] || { echo "error: Python is not executable: $PYTHON" >&2; exit 2; }
  "$PYTHON" -c 'import pytest, requests, sdm_relational_engine' >/dev/null || {
    echo 'error: RFM_NIM_PYTHON is missing the client live-test dependencies' >&2
    exit 2
  }
else
  PYTHON="$VENV_DIR/bin/python"
  STAMP="$VENV_DIR/.rfm-sdk-live-dependencies"
  fingerprint="$(
    cksum "$ENGINE_ROOT/pyproject.toml" "$ENGINE_ROOT/setup.py" |
      cksum |
      awk '{print $1 ":" $2}'
  )"
  installed_fingerprint=''
  [[ -f "$STAMP" ]] && installed_fingerprint="$(<"$STAMP")"

  needs_bootstrap="$REBUILD_ENV"
  if [[ ! -x "$PYTHON" || "$installed_fingerprint" != "$fingerprint" ]]; then
    needs_bootstrap=1
  elif ! "$PYTHON" -c 'import pytest, requests, sdm_relational_engine' >/dev/null 2>&1; then
    needs_bootstrap=1
  fi

  if [[ "$needs_bootstrap" == 1 ]]; then
    command -v python3 >/dev/null || { echo 'error: python3 is required' >&2; exit 2; }
    mkdir -p "$(dirname "$VENV_DIR")"
    python3 -m venv --clear "$VENV_DIR"
    "$PYTHON" -m pip install --upgrade pip
    WITH_RELATIONALLIB=0 \
      "$PYTHON" -m pip install --editable "${ENGINE_ROOT}[test]"
    printf '%s\n' "$fingerprint" >"$STAMP"
  fi
fi

if ! PYTHONPATH="$ENGINE_ROOT/tests/client${PYTHONPATH:+:$PYTHONPATH}" \
  RFM_NIM_BASE_URL="$BASE_URL" \
  "$PYTHON" -c '
import os

from rfm_nim_live_harness import normalize_base_url

try:
    normalize_base_url(os.environ["RFM_NIM_BASE_URL"])
except ValueError as exc:
    raise SystemExit(f"error: {exc}") from None
'; then
  exit 2
fi

report_dir="$ENGINE_ROOT/.tmp/rfm-nim-live-results"
mkdir -p "$report_dir"
if [[ "$MODE" == full ]]; then
  marker='live_nim_smoke or live_nim_full'
else
  marker='live_nim_smoke'
fi

echo "Running RFM NIM $MODE validation"
pytest_command=(
  "$PYTHON" -m pytest
  "$ENGINE_ROOT/tests/client/test_rfm_nim_live.py"
  --strict-markers
  -m "$marker"
  --junitxml="$report_dir/$MODE.xml"
)
if ((${#pytest_args[@]})); then
  pytest_command+=("${pytest_args[@]}")
fi
RFM_NIM_BASE_URL="$BASE_URL" \
  KUMO_RELATIONAL_NIM_API_KEY="${KUMO_RELATIONAL_NIM_API_KEY:-${RFM_NIM_API_KEY:-}}" \
  exec "${pytest_command[@]}"
