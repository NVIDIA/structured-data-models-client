#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The held-out run, once: a frozen held-out case set run on python-ref,
# python-core, python-old and rust-host, compared, reported and counted
# (heldout.py; docs/design/rust-core/orchestration/briefs/heldout-author.md).
#
#   heldout_run.sh --bundle DIR --prefix P --bundle-sha HEX --inputs-sha HEX
#       --candidate WORKTREE --reference CHECKOUT --reference-sha SHA
#       (--wheel FILE ... | --python PY) --out DIR --threshold N
#       [--cases N] [--reference-python PY] [--register PATH]
#       [--databases LIST] [--stamp FILE] [--execute]
#
# Each option may come from the environment instead: HELDOUT_BUNDLE,
# HELDOUT_PREFIX, HELDOUT_BUNDLE_SHA, HELDOUT_INPUTS_SHA, HELDOUT_CANDIDATE,
# HELDOUT_REFERENCE, HELDOUT_REFERENCE_SHA, HELDOUT_REFERENCE_PYTHON,
# HELDOUT_WHEELS (colon-separated), HELDOUT_PYTHON, HELDOUT_OUT,
# HELDOUT_THRESHOLD, HELDOUT_CASES, HELDOUT_REGISTER, HELDOUT_DATABASES,
# HELDOUT_STAMP.
#
# --bundle: the frozen set, read only (never written).
# --bundle-sha, --inputs-sha: its two digests as recorded when it was
#   frozen, kept outside the bundle (heldout.py digests).
# --candidate: a clean worktree at the candidate commit; its runners,
#   compare.py and rust-host run the cases.
# --reference, --reference-sha: the reference checkout, clean and at that
#   commit; --reference-python (default CHECKOUT/.venv/bin/python) imports
#   its sdm_client from it.
# --wheel: the candidate's locally built wheels (native_wheel.py), installed
#   into a fresh environment under --out; or --python, an environment that
#   already holds them.
# --out: the run directory, made absolute; --stamp (default OUT/.ran) is
#   made, with noclobber, just before the candidate first sees a case, so a
#   second run stops there.
# --register: the register's path in the candidate (default
#   docs/design/rust-core/register.yaml), snapshotted at the run's start:
#   only entries approved before the run count.
# --databases: the databases each case runs on, comma-separated (default
#   duckdb,sqlite,postgres; T2.16a): python-ref, python-core and rust-host
#   run on each, rust-host on Postgres replaying python-core's fetches
#   there; python-old on DuckDB only. A leg off DuckDB is keyed
#   <id>@<database>, and heldout.py counts each. Postgres needs
#   SDM_POSTGRES_TEST_DSN, the server each case gets a new database on.
#   The environment --wheel makes installs the client wheel with the
#   extras of the databases named (the client's own: duckdb, sqlite,
#   postgres); every Python environment, --python and --reference-python
#   included, must import each named database's driver (duckdb,
#   adbc_driver_sqlite, psycopg), checked in the preconditions and again
#   before the stamp, so a missing driver stops the run before it starts.
#
# The default is a dry run: it checks every precondition, verifies the
# bundle's digests, prints each command, and writes nothing. It never
# loads, runs or prints a case. --execute runs, once. Nothing here deletes.
#
# Exit: 0 a dry run with every precondition holding, or a whole run;
# 1 a precondition or a step before the run failed, or more than one leg
# failed; 2 a usage error; 3 a run with one leg missing (partial).
set -euo pipefail

bundle=${HELDOUT_BUNDLE:-}
prefix=${HELDOUT_PREFIX:-}
bundle_sha=${HELDOUT_BUNDLE_SHA:-}
inputs_sha=${HELDOUT_INPUTS_SHA:-}
candidate=${HELDOUT_CANDIDATE:-}
reference=${HELDOUT_REFERENCE:-}
reference_sha=${HELDOUT_REFERENCE_SHA:-}
py_ref=${HELDOUT_REFERENCE_PYTHON:-}
py_cand=${HELDOUT_PYTHON:-}
out=${HELDOUT_OUT:-}
threshold=${HELDOUT_THRESHOLD:-}
cases=${HELDOUT_CASES:-}
register=${HELDOUT_REGISTER:-docs/design/rust-core/register.yaml}
databases=${HELDOUT_DATABASES:-duckdb,sqlite,postgres}
stamp=${HELDOUT_STAMP:-}
mode=dry
wheels=()
if [ -n "${HELDOUT_WHEELS:-}" ]; then
  IFS=: read -r -a wheels <<<"$HELDOUT_WHEELS"
fi

usage() { sed -n '6,14p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
full_help() { sed -n '6,59p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
while [ $# -gt 0 ]; do
  case "$1" in
    --execute) mode=execute; shift; continue ;;
    -h | --help) full_help ;;
  esac
  [ $# -ge 2 ] || { echo "heldout_run.sh: $1 needs a value" >&2; exit 2; }
  case "$1" in
    --bundle) bundle=$2 ;;
    --prefix) prefix=$2 ;;
    --bundle-sha) bundle_sha=$2 ;;
    --inputs-sha) inputs_sha=$2 ;;
    --candidate) candidate=$2 ;;
    --reference) reference=$2 ;;
    --reference-sha) reference_sha=$2 ;;
    --reference-python) py_ref=$2 ;;
    --wheel) wheels+=("$2") ;;
    --python) py_cand=$2 ;;
    --out) out=$2 ;;
    --threshold) threshold=$2 ;;
    --cases) cases=$2 ;;
    --register) register=$2 ;;
    --databases) databases=$2 ;;
    --stamp) stamp=$2 ;;
    *) echo "heldout_run.sh: unknown option $1" >&2; usage ;;
  esac
  shift 2
done

missing=()
for name in bundle prefix bundle_sha inputs_sha candidate reference \
  reference_sha out threshold; do
  [ -n "${!name}" ] || missing+=("--${name//_/-}")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "heldout_run.sh: missing ${missing[*]}" >&2
  usage
fi
if [ ${#wheels[@]} -gt 0 ] && [ -n "$py_cand" ]; then
  echo 'heldout_run.sh: give --wheel or --python, not both' >&2
  exit 2
fi
if [ ${#wheels[@]} -eq 0 ] && [ -z "$py_cand" ]; then
  echo 'heldout_run.sh: give the candidate wheels (--wheel) or --python' >&2
  exit 2
fi
case "$threshold" in '' | *[!0-9]*)
  echo "heldout_run.sh: --threshold takes a count, not $threshold" >&2
  exit 2 ;;
esac
IFS=, read -r -a dbs <<<"$databases"
for db in "${dbs[@]}"; do
  case "$db" in duckdb | sqlite | postgres) ;; *)
    echo "heldout_run.sh: --databases takes duckdb, sqlite and postgres, not $db" >&2
    exit 2 ;;
  esac
done
case ",$databases," in *,duckdb,*) ;; *)
  # Each expect is python-ref's DuckDB outcome (heldout.py expect).
  echo 'heldout_run.sh: --databases must hold duckdb' >&2
  exit 2 ;;
esac

# Absolute: rust-host runs from inside the candidate, and a relative path
# would fail there, after the stamp.
case "$out" in /*) ;; *) out=$PWD/$out ;; esac
case "$stamp" in '' | /*) ;; *) stamp=$PWD/$stamp ;; esac

here=$(cd "$(dirname "$0")" && pwd)
# Each database's driver, as run_python.py and databases.py import it.
drivers=()
for db in "${dbs[@]}"; do
  case "$db" in
    duckdb) drivers+=(duckdb) ;;
    sqlite) drivers+=(adbc_driver_sqlite.dbapi) ;;
    postgres) drivers+=(psycopg) ;;
  esac
done
extras=$(IFS=,; echo "${dbs[*]}")
drivers_in() { # PYTHON: whether it imports every selected database's driver
  "$1" -I -c 'import importlib, sys
missing = []
for name in sys.argv[1:]:
    try:
        importlib.import_module(name)
    except ImportError:
        missing.append(name)
if missing:
    sys.exit("missing " + ", ".join(missing))' "${drivers[@]}"
}
heldout=("${HELDOUT_TOOL_PYTHON:-python3}" -I "$here/heldout.py")
py_ref=${py_ref:-$reference/.venv/bin/python}
stamp=${stamp:-$out/.ran}
env_dir=$out/candidate-env
[ -n "$py_cand" ] || py_cand=$env_dir/bin/python
runpy=$candidate/contract/runners/run_python.py
problems=0
problem() { echo "!! $*"; problems=$((problems + 1)); }
do_() {
  printf '  '
  printf ' %q' "$@"
  echo
  if [ "$mode" = execute ]; then "$@"; fi
}

echo "== preconditions ($mode run)"
count=()
[ -z "$cases" ] || count=(--cases "$cases")
if [ ! -d "$bundle" ]; then
  problem "no bundle directory $bundle"
elif ! "${heldout[@]}" digests "$bundle" --prefix "$prefix" \
  --bundle-sha "$bundle_sha" --inputs-sha "$inputs_sha" \
  "${count[@]+"${count[@]}"}" | sed 's/^/   /'; then
  problem 'the bundle is not the one frozen: its digests or count differ'
fi
if [ "$(git -C "$reference" rev-parse HEAD 2>/dev/null)" != "$reference_sha" ]
then
  problem "the reference checkout is not at $reference_sha"
elif [ -n "$(git -C "$reference" status --porcelain)" ]; then
  problem 'the reference checkout is not clean'
fi
if ! sha=$(git -C "$candidate" rev-parse HEAD 2>/dev/null); then
  problem "the candidate $candidate is not a git worktree"
  sha=unknown
elif [ -n "$(git -C "$candidate" status --porcelain)" ]; then
  problem 'the candidate worktree is not clean'
elif ! git -C "$candidate" cat-file -e "HEAD:$register" 2>/dev/null; then
  problem "the candidate holds no $register"
fi
[ -f "$runpy" ] || problem "the candidate holds no contract/runners/run_python.py"
if ! "$py_ref" -I -c 'import pathlib, sys, sdm_client
p = pathlib.Path(sdm_client.__file__).resolve()
sys.exit(0 if p.is_relative_to(pathlib.Path(sys.argv[1]).resolve()) else 1)' \
  "$reference" 2>/dev/null; then
  problem "the reference python $py_ref does not import the reference's sdm_client"
fi
for wheel in "${wheels[@]+"${wheels[@]}"}"; do
  case "$wheel" in
    *.whl) [ -f "$wheel" ] || problem "no wheel $wheel" ;;
    *) problem "$wheel is not a wheel" ;;
  esac
done
probe='import pathlib, sys, sdm_client._native as n
p = pathlib.Path(n.__file__).resolve()
if p.is_relative_to(pathlib.Path(sys.argv[1]).resolve()):
    sys.exit(f"sdm_client._native from the candidate tree: {p}")
print("   candidate native contract", n.CONTRACT_VERSION, "test_ops", n.TEST_OPS)'
if [ ${#wheels[@]} -eq 0 ] &&
  ! "$py_cand" -I -c "$probe" "$candidate"; then
  problem "the candidate python $py_cand cannot load an installed sdm_client._native"
fi
command -v cargo >/dev/null || problem 'no cargo, for rust-host'
if ! why=$(drivers_in "$py_ref" 2>&1); then
  problem "the reference python cannot import a database driver (${why##*: })"
fi
if [ ${#wheels[@]} -eq 0 ] && ! why=$(drivers_in "$py_cand" 2>&1); then
  problem "the candidate python cannot import a database driver (${why##*: })"
fi
[ ! -e "$stamp" ] || problem "$stamp exists: the held-out cases have run once already"
case ",$databases," in *,postgres,*)
  # Its value is a credential: only whether it is set is said.
  [ -n "${SDM_POSTGRES_TEST_DSN:-}" ] ||
    problem 'a Postgres leg needs SDM_POSTGRES_TEST_DSN (or leave postgres out of --databases)' ;;
esac
if [ "$mode" = execute ] && [ -e "$out" ] && [ ! -e "$stamp" ] &&
  [ -n "$(ls -A "$out")" ]; then
  problem "$out holds files from an earlier attempt: give a new --out"
fi
echo "   candidate $sha; run dir $out; databases ${dbs[*]}"
if [ "$problems" -gt 0 ] && [ "$mode" = execute ]; then
  echo "== $problems precondition(s) failed: nothing run"
  exit 1
fi

snapshot() { # the register and who ran, at the run's start
  git -C "$candidate" show "HEAD:$register" >"$out/register-at-run-start.yaml"
  printf 'candidate %s\nreference %s\nstarted %s\n' "$sha" "$reference_sha" \
    "$(date -u +%FT%TZ)" >"$out/identity.txt"
}
copy_bundle() { # the original is never written
  cp "$bundle/$prefix"*.json "$out/cases-orig/"
  if [ -f "$bundle/README.md" ]; then
    cp "$bundle/README.md" "$out/bundle-README.md"
  fi
}
make_env() { # the wheels, the client's with the named databases' extras
  local wheel install=()
  for wheel in "$@"; do
    case "$(basename "$wheel")" in
      structured_data_models_client-*) install+=("$wheel[$extras]") ;;
      *) install+=("$wheel") ;;
    esac
  done
  "${HELDOUT_VENV_PYTHON:-python3}" -m venv "$env_dir"
  "$env_dir/bin/python" -m pip --disable-pip-version-check --no-input install \
    --quiet --only-binary=:all: "${install[@]}" jsonschema pyyaml
}
drivers_ready() { # before the stamp: every runner's environment has the drivers
  local python why
  for python in "$py_ref" "$py_cand"; do
    if ! why=$(drivers_in "$python" 2>&1); then
      echo "!! $python cannot import a database driver (${why##*: }): nothing run"
      return 1
    fi
  done
}
load_check() {
  "$py_cand" -I -c 'import pathlib, sys
sys.path.insert(0, sys.argv[1])
import run_python
print("  ", len(run_python.load_cases([pathlib.Path(sys.argv[2])])), "cases load")' \
    "$candidate/contract/runners" "$out/cases-orig"
}
stamp_once() { (set -C; echo once >"$stamp"); }
leg_file() { # RUNNER DATABASE: where a runner's run on a database is written
  if [ "$2" = duckdb ]; then echo "$out/$1.jsonl"; else echo "$out/legs/$1.$2.jsonl"; fi
}
append_legs() { # RUNNER: each other database's lines after DuckDB's, in one file
  local db
  for db in "${dbs[@]}"; do
    # A leg that failed wrote nothing; its failure is recorded already.
    if [ "$db" != duckdb ] && [ -f "$(leg_file "$1" "$db")" ]; then
      cat "$(leg_file "$1" "$db")" >>"$out/$1.jsonl"
    fi
  done
}
leg() { # a runner leg; a failure is recorded, not fatal
  local runner=$1
  shift
  if ! "$@"; then
    echo "!! $runner failed"
    case " ${failed[*]-} " in *" $runner "*) ;; *) failed+=("$runner") ;; esac
  fi
}
rust_host() { # DATABASE: rust-host's run there; on Postgres it replays python-core's fetches
  local where=()
  case "$1" in
    sqlite) where=(--database sqlite) ;;
    postgres) where=(--replay "$out/legs/fetches.jsonl") ;;
  esac
  (cd "$candidate/rust" && cargo run -q -p sdm-testkit --bin sdm-contract-run \
    --locked -- "${where[@]+"${where[@]}"}" --out "$(leg_file rust-host "$1")" \
    "$out/cases")
}
compare() { # each database's, one file; a non-zero exit is a result, not an error
  local legs=() runner db
  for runner in python-core python-old rust-host; do
    [ ! -f "$out/$runner.jsonl" ] || legs+=("$out/$runner.jsonl")
  done
  : >"$out/compare.txt"
  for db in "${dbs[@]}"; do
    echo "== $db" >>"$out/compare.txt"
    "$py_cand" -I "$candidate/scripts/rust_core/compare.py" --cases "$out/cases" \
      --database "$db" --reference "$out/python-ref.jsonl" \
      "${legs[@]+"${legs[@]}"}" >>"$out/compare.txt" 2>&1 || true
  done
}
failed=()

if [ ${#wheels[@]} -gt 0 ]; then
  echo "== 0. the candidate's environment, from its wheels"
  do_ mkdir -p "$out"
  do_ make_env "${wheels[@]}"
  do_ "$py_cand" -I -c "$probe" "$candidate"
fi

echo "== 1. snapshot what counts: the register at the run's start"
do_ mkdir -p "$out/cases-orig" "$out/legs"
do_ snapshot

echo "== 2. copy the bundle; the copy's inputs are the bundle's"
do_ copy_bundle
do_ "${heldout[@]}" digests "$out/cases-orig" --prefix "$prefix" \
  --inputs-sha "$inputs_sha" "${count[@]+"${count[@]}"}"

echo "== 3. load check only (the case schema; no core, no data)"
do_ load_check

echo "== 4. python-ref on the author's files, to learn the reference's outcomes"
for db in "${dbs[@]}"; do
  do_ "$py_ref" -I "$runpy" --mode python-ref --database "$db" \
    --reference "$reference" --out "$(leg_file python-ref.orig "$db")" \
    "$out/cases-orig"
done
do_ append_legs python-ref.orig

echo "== 5. each expect from python-ref (the author's was a prediction)"
do_ "${heldout[@]}" expect "$out/cases-orig" "$out/python-ref.orig.jsonl" \
  "$out/cases" --prefix "$prefix"

echo "== 6. python-ref again on the regenerated files: the same outcomes"
for db in "${dbs[@]}"; do
  do_ "$py_ref" -I "$runpy" --mode python-ref --database "$db" \
    --reference "$reference" --out "$(leg_file python-ref "$db")" "$out/cases"
done
do_ append_legs python-ref
do_ "${heldout[@]}" same "$out/python-ref.orig.jsonl" "$out/python-ref.jsonl"

echo "== 7. THE run, once per runner (a second --execute stops here)"
do_ drivers_ready
do_ stamp_once
for db in "${dbs[@]}"; do
  record=()
  [ "$db" != postgres ] || record=(--record-fetches "$out/legs/fetches.jsonl")
  do_ leg python-core "$py_cand" -I "$runpy" --mode python-core --database "$db" \
    "${record[@]+"${record[@]}"}" --out "$(leg_file python-core "$db")" "$out/cases"
done
do_ append_legs python-core
do_ leg python-old "$py_cand" -I "$runpy" --mode python-old \
  --out "$out/python-old.jsonl" "$out/cases"
for db in "${dbs[@]}"; do
  do_ leg rust-host rust_host "$db"
done
do_ append_legs rust-host

echo "== 8. compare"
do_ compare

echo "== 9. the per-case report and the counts"
if [ ${#failed[@]} -gt 1 ]; then
  echo "!! ${#failed[@]} legs failed (${failed[*]}): no report; the owner rules"
  exit 1
fi
partial=()
[ ${#failed[@]} -eq 0 ] || partial=(--missing "${failed[0]}=the leg failed")
do_ "${heldout[@]}" report "$out" --prefix "$prefix" \
  "${partial[@]+"${partial[@]}"}"
# The counts' own exit (no-go before any approval is listed) is not the
# run's: the executor gives no verdict. The run's exit says only whether it
# ran whole (0) or with one leg missing (3).
do_ "${heldout[@]}" counts "$out" --prefix "$prefix" --threshold "$threshold" \
  "${partial[@]+"${partial[@]}"}" || true
echo '   then: list the cases approved before the run, from'
echo '   register-at-run-start.yaml only, fill report.md'"'"'s last column and'
echo '   rerun heldout.py counts with --approved; a reviewer checks it.'

if [ "$mode" = dry ]; then
  if [ "$problems" -gt 0 ]; then
    echo "== dry run: $problems precondition(s) failed"
    exit 1
  fi
  echo '== dry run: every precondition holds; --execute runs it, once'
  exit 0
fi
if [ ${#failed[@]} -gt 0 ]; then
  echo "== ran, with the ${failed[0]} leg missing: partial; the owner rules"
  exit 3
fi
echo '== ran, once'
