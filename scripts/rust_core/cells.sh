#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds the local runtime cells the contract gate runs in, the way CI's
# "Install the cell" step does, from wheels rather than source trees:
#
#   cells.sh [--rev REV] [OUT]
#
# OUT is the output directory, from the argument or $SDM_CELLS_DIR; one of
# the two is required. REV defaults to HEAD. Everything is keyed by git tree
# hash, so a rerun on the same tree only checks what is there:
#
#   tools/               maturin (native_wheel.py's MATURIN pin) and build
#   src-<tree12>/        a detached worktree of REV, the build source only
#   client-<tree12>/     the local client wheel (native_wheel.py, without
#                        --manylinux) and its identity file; the identity
#                        names the first commit built with this tree
#   sib/<pkg>-<tree>/    api, pql and connectors (pure) and the relational
#                        engine (per Python version), keyed by the package's
#                        own tree, so an unchanged package is never rebuilt
#   cell-<name>/         one venv per cell, named, pinned and on the Python
#                        version its ci.yml job gives (CELL, CELL_PINS,
#                        python-version), with DuckDB at the version the
#                        install step pins; every first-party package is a
#                        non-editable wheel install
#   cargo-target/        the Cargo target directory every tree shares, so a
#                        new tree builds incrementally ($CARGO_TARGET_DIR
#                        wins when set)
#
# A cell is checked after install: run_python.py's _cell() must name it from
# its versions, and sdm_client._native must load. It is reinstalled only
# when its tree, Python or pins change (its .cells-stamp file). The source
# worktrees accumulate: `git worktree remove OUT/src-<tree12>` drops one; a
# reused one must still be at its tree and clean, or the run stops.
# Nothing here deletes anything or touches the network beyond uv's and
# cargo's package downloads.
set -euo pipefail

usage() { sed -n '/^# Builds/,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

REV=HEAD
OUT=${SDM_CELLS_DIR:-}
while [ $# -gt 0 ]; do
  case $1 in
    -h | --help) usage; exit 0 ;;
    --rev) REV=${2:?--rev needs a revision}; shift 2 ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) OUT=$1; shift ;;
  esac
done
[ -n "$OUT" ] || { echo 'give OUT or set SDM_CELLS_DIR' >&2; exit 2; }

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
COMMIT=$(git -C "$REPO" rev-parse "$REV^{commit}")
TREE=$(git -C "$REPO" rev-parse "$COMMIT^{tree}")
T12=${TREE:0:12}
mkdir -p "$OUT/sib" "$OUT/logs"
OUT=$(cd "$OUT" && pwd)
SRC=$OUT/src-$T12
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$OUT/cargo-target}
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# The cells as REV's ci.yml defines them: one line per job that sets CELL
# and CELL_PINS, "name|python|pins".
CI_YML=$(git -C "$REPO" show "$COMMIT:.github/workflows/ci.yml")
CELLS=$(printf '%s\n' "$CI_YML" | awk '
  function flush() { if (cell != "" && pins != "") print cell "|" py "|" pins }
  /^  [A-Za-z0-9_.-]+:[[:space:]]*$/ { flush(); cell = ""; pins = ""; py = "" }
  /^      CELL: / { cell = $2 }
  /^      CELL_PINS: / { sub(/^      CELL_PINS: /, ""); pins = $0 }
  /python-version:/ { py = $2; gsub(/["\047]/, "", py) }
  END { flush() }
')
[ -n "$CELLS" ] || { echo "no CELL/CELL_PINS jobs in $REV's ci.yml" >&2; exit 3; }
DUCKDB=$(printf '%s\n' "$CI_YML" | grep -o 'CELL_PINS duckdb==[0-9.]*' | sort -u | awk '{print $2}')
[ "$(printf '%s\n' "$DUCKDB" | wc -l | tr -d ' ')" = 1 ] && [ -n "$DUCKDB" ] \
  || { echo "ci.yml's install step pins DuckDB as '$DUCKDB', not once" >&2; exit 3; }
MATURIN=$(git -C "$REPO" show "$COMMIT:scripts/rust_core/native_wheel.py" \
  | sed -n "s/^MATURIN = '\(.*\)'$/\1/p")
[ -n "$MATURIN" ] || { echo "native_wheel.py names no MATURIN pin" >&2; exit 3; }
while IFS='|' read -r cell py pins; do
  [ -n "$py" ] || { echo "ci.yml's $cell job gives no python-version" >&2; exit 3; }
done <<<"$CELLS"

log "source ${COMMIT:0:12} (tree $T12)"
[ -d "$SRC" ] || git -C "$REPO" worktree add --quiet --detach "$SRC" "$COMMIT"
# A reused source worktree must still hold this tree, unedited: its wheels
# are labelled with it.
[ "$(git -C "$SRC" rev-parse 'HEAD^{tree}' 2>/dev/null)" = "$TREE" ] \
  || { echo "$SRC is not at tree $T12: check it out at $COMMIT or remove it" >&2; exit 3; }
[ -z "$(git -C "$SRC" status --porcelain --untracked-files=all)" ] \
  || { echo "$SRC has local changes: restore it or remove it" >&2; exit 3; }

log "tools venv (maturin $MATURIN)"
[ -x "$OUT/tools/bin/python" ] || uv venv -q -p 3.12 "$OUT/tools"
uv pip install -q -p "$OUT/tools/bin/python" "maturin==$MATURIN" build

log "client wheel (native_wheel.py, local build)"
CLIENT_DIR=$OUT/client-$T12
if ! ls "$CLIENT_DIR"/structured_data_models_client-*.whl >/dev/null 2>&1; then
  (cd "$SRC" && PATH="$OUT/tools/bin:$PATH" "$OUT/tools/bin/python" -I \
    scripts/rust_core/native_wheel.py --out "$CLIENT_DIR") \
    >"$OUT/logs/client-$T12.log" 2>&1 \
    || { echo "client build failed; see $OUT/logs/client-$T12.log" >&2; exit 1; }
fi
CLIENT=$(ls "$CLIENT_DIR"/structured_data_models_client-*.whl)

sib() { # sib PACKAGE PYTHON|pure: prints the wheel's path
  local pkg=$1 py=$2 tree out
  tree=$(git -C "$REPO" rev-parse "$COMMIT:packages/structured-data-models-$pkg")
  out=$OUT/sib/$pkg-$tree
  [ "$py" = pure ] || out=$out-py$py
  if ! ls "$out"/*.whl >/dev/null 2>&1; then
    local python=()
    [ "$py" = pure ] || python=(--python "$py")
    uv build -q --wheel ${python[@]+"${python[@]}"} --out-dir "$out" \
      "$SRC/packages/structured-data-models-$pkg" >"$OUT/logs/sib-$pkg-$py.log" 2>&1 \
      || { echo "$pkg build failed; see $OUT/logs/sib-$pkg-$py.log" >&2; exit 1; }
  fi
  ls "$out"/*.whl
}
log "sibling wheels"
API=$(sib api pure)
PQL=$(sib pql pure)
CONNECTORS=$(sib connectors pure)

while IFS='|' read -r cell py pins; do
  V=$OUT/cell-$cell
  stamp="$TREE $py $pins duckdb==${DUCKDB#duckdb==}"
  if [ -x "$V/bin/python" ] && [ "$(cat "$V/.cells-stamp" 2>/dev/null)" = "$stamp" ]; then
    log "cell $cell: up to date"
    continue
  fi
  log "cell $cell (python $py: $pins)"
  ENGINE=$(sib relational-engine "$py")
  [ -x "$V/bin/python" ] || uv venv -q -p "$py" "$V"
  # shellcheck disable=SC2086 # the pins are words
  printf '%s\n' $pins "$DUCKDB" >"$V/cell-constraints.txt"
  # As ci.yml's "Install the cell" step; the first-party wheels are
  # reinstalled, since a new tree keeps their version numbers.
  # shellcheck disable=SC2086
  uv pip install -q -p "$V/bin/python" -c "$V/cell-constraints.txt" \
    --reinstall-package structured-data-models-api \
    --reinstall-package structured-data-models-pql \
    --reinstall-package structured-data-models-connectors \
    --reinstall-package structured-data-models-relational-engine \
    --reinstall-package structured-data-models-client \
    $pins "$DUCKDB" pyyaml jsonschema pytest pytest-xdist pytest-timeout \
    "structured-data-models-api @ file://$API" \
    "structured-data-models-pql @ file://$PQL" \
    "structured-data-models-connectors[duckdb] @ file://$CONNECTORS" \
    "structured-data-models-relational-engine @ file://$ENGINE" \
    "structured-data-models-client[test] @ file://$CLIENT"
  "$V/bin/python" -I - "$SRC/contract/runners" "$cell" <<'PY'
import importlib.metadata as metadata
import platform
import sys

sys.path.insert(0, sys.argv[1])
import run_python

versions = {
    'python': platform.python_version(),
    **{k: metadata.version(k) for k in ('pandas', 'pyarrow', 'numpy', 'duckdb')},
}
got = run_python._cell(versions)
assert got == sys.argv[2], (got, versions)
import sdm_client._native as native

print(
    f'  {got}: python {versions["python"]}, '
    f'native contract {native.CONTRACT_VERSION}, test_ops {native.TEST_OPS}'
)
PY
  printf '%s\n' "$stamp" >"$V/.cells-stamp"
done <<<"$CELLS"

log "done: tree $T12 in $OUT"
printf 'CLIENT_WHEEL=%s\n' "$CLIENT"
while IFS='|' read -r cell _ _; do
  printf 'CELL %s=%s\n' "$cell" "$OUT/cell-$cell/bin/python"
done <<<"$CELLS"
