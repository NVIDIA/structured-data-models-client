#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Builds the local runtime cells the contract gate runs in, the way CI's
# "Install the cell" step does, from wheels rather than source trees:
#
#   cells.sh [--rev REV] [OUT]
#   cells.sh --print-key [--rev REV]
#
# OUT is one lane's environment directory (T3.0), from the argument, else
# $SDM_CELLS_DIR, else $SDM_LANE_ENV; one is required. By convention it is
# $SDM_CELLS_ROOT/<lane> (/mnt/localssd/cells/<lane>), and SDM_LANE_ENV names
# it, so the runners can refuse any other interpreter. REV defaults to HEAD.
#
# A lane holds one environment, keyed by a hash of what it installs (the
# key): uv.lock, the native sources (rust/), the packages' sources (their
# tests left out), native_wheel.py, and the cells, pins and DuckDB pin of
# REV's ci.yml. A commit that changes none of them (docs, cases, tests,
# the checking tools) reuses the environment as it is, in seconds; a new
# key builds a new one and then removes the lane's older one, unless a
# process still runs from it (clean_tmp.sh sweeps it later). --print-key
# prints the key and builds nothing:
#
#   tools/               maturin (native_wheel.py's MATURIN pin) and build
#   src/                 one detached worktree, moved to REV when the key
#                        changes; it must be clean
#   cargo-target/        the lane's Cargo target directory, incremental
#                        state only ($CARGO_TARGET_DIR wins when set): the
#                        shared sccache (~/.cargo/config.toml) holds the
#                        builds, and clean_tmp.sh caps this
#   sib/<pkg>-<tree>/    api, pql and connectors (pure) and the relational
#                        engine (per Python version), keyed by the package's
#                        own tree, so an unchanged package is never rebuilt
#   env-<key12>/         the environment: client/ (the local client wheel,
#                        native_wheel.py without --manylinux, and its
#                        identity) and one venv per cell, cell-<name>/,
#                        named, pinned and on the Python version its ci.yml
#                        job gives (CELL, CELL_PINS, python-version), with
#                        DuckDB at the version the install step pins; every
#                        first-party package is a non-editable wheel install;
#                        and kit/, the reference cell with the test-ops build
#                        first (SDM_KIT_PYTHON, as CI's contract jobs set it)
#   current              a link to the environment in use
#
# A cell is checked after install: run_python.py's _cell() must name it from
# its versions, and sdm_client._native must load. Its stamp (.cells-stamp)
# holds the key. It prints CLIENT_WHEEL=..., KIT_PYTHON=... and one CELL <name>=<python>
# line per cell. It deletes only an older env-<key12> of OUT, and touches
# the network only for uv's and cargo's package downloads.
set -euo pipefail

usage() { sed -n '/^# Builds/,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

REV=HEAD
PRINT_KEY=0
OUT=${SDM_CELLS_DIR:-${SDM_LANE_ENV:-}}
while [ $# -gt 0 ]; do
  case $1 in
    -h | --help) usage; exit 0 ;;
    --rev) REV=${2:?--rev needs a revision}; shift 2 ;;
    --print-key) PRINT_KEY=1; shift ;;
    -*) echo "unknown option $1" >&2; exit 2 ;;
    *) OUT=$1; shift ;;
  esac
done
[ -n "$OUT" ] || [ "$PRINT_KEY" = 1 ] || { echo 'give OUT or set SDM_CELLS_DIR or SDM_LANE_ENV' >&2; exit 2; }

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
COMMIT=$(git -C "$REPO" rev-parse "$REV^{commit}")
TREE=$(git -C "$REPO" rev-parse "$COMMIT^{tree}")
T12=${TREE:0:12}
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
' | awk '!seen[$0]++')
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

# The key: what the environment installs, never the commit as a whole.
KEY=$(
  {
    printf 'uv.lock %s\n' "$(git -C "$REPO" rev-parse "$COMMIT:uv.lock")"
    printf 'rust %s\n' "$(git -C "$REPO" rev-parse "$COMMIT:rust")"
    printf 'native_wheel.py %s\n' "$(git -C "$REPO" rev-parse "$COMMIT:scripts/rust_core/native_wheel.py")"
    git -C "$REPO" ls-tree -r "$COMMIT" -- packages | grep -Ev '/(tests|tests_[a-z_]+)/' || true
    printf '%s\n' "$CELLS" "$DUCKDB" "$MATURIN"
  } | git hash-object --stdin
)
if [ "$PRINT_KEY" = 1 ]; then
  printf '%s\n' "$KEY"
  exit 0
fi
K12=${KEY:0:12}
mkdir -p "$OUT/sib" "$OUT/logs"
OUT=$(cd "$OUT" && pwd)
SRC=$OUT/src
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$OUT/cargo-target}
ENV=$OUT/env-$K12
CLIENT_DIR=$ENV/client
log "source ${COMMIT:0:12} (tree $T12), environment key $K12"

# The cells are current when each holds this key: nothing to build.
current=1
while IFS='|' read -r cell py pins; do
  [ -x "$ENV/cell-$cell/bin/python" ] && [ "$(cat "$ENV/cell-$cell/.cells-stamp" 2>/dev/null)" = "$KEY $py $pins $DUCKDB" ] \
    || current=0
done <<<"$CELLS"
ls "$CLIENT_DIR"/structured_data_models_client-*.whl >/dev/null 2>&1 || current=0
[ "$(cat "$ENV/kit/.cells-stamp" 2>/dev/null)" = "$KEY" ] || current=0

if [ "$current" = 0 ]; then
  if [ ! -d "$SRC" ]; then
    git -C "$REPO" worktree add --quiet --detach "$SRC" "$COMMIT"
  else
    # The lane's one source worktree must be unedited before it moves: its
    # wheels are labelled with the commit it holds.
    [ -z "$(git -C "$SRC" status --porcelain --untracked-files=all)" ] \
      || { echo "$SRC has local changes: restore it or remove it" >&2; exit 3; }
    git -C "$SRC" checkout --quiet --detach "$COMMIT"
  fi
  [ "$(git -C "$SRC" rev-parse 'HEAD^{tree}')" = "$TREE" ] \
    || { echo "$SRC is not at tree $T12 after checking out $COMMIT" >&2; exit 3; }
fi

if [ "$current" = 1 ]; then
  log "environment $K12 is current: nothing to build"
else
log "tools venv (maturin $MATURIN)"
[ -x "$OUT/tools/bin/python" ] || uv venv -q -p 3.12 "$OUT/tools"
uv pip install -q -p "$OUT/tools/bin/python" "maturin==$MATURIN" build

log "client wheel (native_wheel.py, local build)"
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
  V=$ENV/cell-$cell
  stamp="$KEY $py $pins $DUCKDB"
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
    "structured-data-models-connectors[duckdb,sqlite,postgres] @ file://$CONNECTORS" \
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

# The provider kit's interpreter, as CI's contract jobs make it: the
# reference cell with the test-ops build of the client first on its path
# (SDM_KIT_PYTHON for contract/runners/provider_kit.py).
log "kit (the test-ops build over the reference cell)"
KIT_WHEELS=$ENV/client-test-ops
if ! ls "$KIT_WHEELS"/structured_data_models_client-*.whl >/dev/null 2>&1; then
  (cd "$SRC" && PATH="$OUT/tools/bin:$PATH" "$OUT/tools/bin/python" -I \
    scripts/rust_core/native_wheel.py --test-ops --out "$KIT_WHEELS") \
    >"$OUT/logs/client-test-ops-$T12.log" 2>&1 \
    || { echo "test-ops build failed; see $OUT/logs/client-test-ops-$T12.log" >&2; exit 1; }
fi
REF=$(printf '%s\n' "$CELLS" | awk -F'|' '$1 == "ref-pandas3" { print $2 }')
[ -n "$REF" ] || { echo "ci.yml names no ref-pandas3 cell for the kit" >&2; exit 3; }
[ -x "$ENV/kit/bin/python" ] || uv venv -q -p "$REF" "$ENV/kit"
"$ENV/cell-ref-pandas3/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])' \
  >"$("$ENV/kit/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/cells-ref-pandas3.pth"
uv pip install -q -p "$ENV/kit/bin/python" --no-deps --reinstall "$(ls "$KIT_WHEELS"/structured_data_models_client-*.whl)"
"$ENV/kit/bin/python" -I -c 'import sdm_client._native as n; assert n.TEST_OPS'
printf '%s\n' "$KEY" >"$ENV/kit/.cells-stamp"
fi

# The lane holds one environment: this one is current, an older one goes
# unless a process still runs from it.
ln -sfn "env-$K12" "$OUT/current.new" && mv -Tf "$OUT/current.new" "$OUT/current"
for old in "$OUT"/env-*; do
  [ -d "$old" ] && [ "$old" != "$ENV" ] || continue
  case "${old##*/}" in env-[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;; *) continue ;; esac
  if ps -axo args= | grep -Fq -- "$old/" || ls -l /proc/[0-9]*/cwd 2>/dev/null | grep -Fq -- "$old"; then
    log "older environment ${old##*/} is in use: kept for clean_tmp.sh"
    continue
  fi
  log "older environment ${old##*/} removed"
  rm -rf -- "$old"
done

CLIENT=$(ls "$CLIENT_DIR"/structured_data_models_client-*.whl)
log "done: tree $T12, environment $K12 in $OUT"
printf 'CLIENT_WHEEL=%s\n' "$CLIENT"
printf 'KIT_PYTHON=%s\n' "$ENV/kit/bin/python"
while IFS='|' read -r cell _ _; do
  printf 'CELL %s=%s\n' "$cell" "$ENV/cell-$cell/bin/python"
done <<<"$CELLS"
