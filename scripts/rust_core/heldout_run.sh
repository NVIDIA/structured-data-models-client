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
#       [--databases LIST] [--stamp FILE] [--java-case-ids FILE]
#       [--wheel-identity FILE] [--execute]
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
# --out: the run directory, made absolute; --stamp (default BUNDLE.started)
#   is made, with noclobber, before any candidate-owned case reader, so a
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
# Warehouse environments may be sourced from HELDOUT_<LEG>_ENV, replacing
# '-' by '_' and using uppercase, such as HELDOUT_DATABRICKS_SQL_ENV.
# Spark requires HELDOUT_SPARK_PAYLOAD and HELDOUT_SPARK_VOLUME, prepared
# with dbx_runs.py and a new capture_job.py workspace notebook.
# Its payload includes candidate_wheel_dir plus the ordinary reference
# wheel_dir, clean staged source/reference trees and live guard settings.
# Java requires SDM_JAVA_JAR, HELDOUT_JAVA_CLASSPATH, HELDOUT_JAVA_DRIVERS
# and HELDOUT_JAVA_IDENTITY (the exact final JAR build's sidecar).
# The default is a dry run: it checks external preconditions, defers bundle
# digest verification until after the stamp, prints commands and writes nothing. It never
# loads, runs or prints a case. --execute runs, once. Nothing here deletes.
#
# Exit: 0 a dry run with every precondition holding, or a whole run;
# 1 a precondition or a step before the run failed, or more than one leg
# failed; 2 a usage error; 3 a run with one leg missing (partial).
set -euo pipefail
umask 077

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
java_ids=${HELDOUT_JAVA_CASE_IDS:-}
wheel_identity=${HELDOUT_WHEEL_IDENTITY:-}
tool_python=${HELDOUT_TOOL_PYTHON:-python3}
mode=dry
wheels=()
if [ -n "${HELDOUT_WHEELS:-}" ]; then
  IFS=: read -r -a wheels <<<"$HELDOUT_WHEELS"
fi

usage() { sed -n '6,14p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
full_help() { sed -n '6,/^set -euo pipefail/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0; }
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
    --java-case-ids) java_ids=$2 ;;
    --wheel-identity) wheel_identity=$2 ;;
    --tool-python) tool_python=$2 ;;
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
  case "$db" in duckdb | sqlite | postgres | databricks-sql | databricks-spark | snowflake | lakebase) ;; *)
    echo "heldout_run.sh: --databases takes supported database or warehouse legs, not $db" >&2
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
    postgres | lakebase) drivers+=(psycopg) ;;
    databricks-sql) drivers+=(databricks.sql databricks.sdk) ;;
    snowflake) drivers+=(snowflake.connector) ;;
  esac
done
extras=$(printf '%s\n' "${dbs[@]}" | sed -e 's/^databricks-.*/databricks/' -e 's/^lakebase$/postgres/' | sort -u | paste -sd, -)
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
heldout=("$tool_python" -I "$here/heldout.py")
py_ref=${py_ref:-$reference/.venv/bin/python}
# All output directories for one canonical bundle share its default stamp.
bundle=$(realpath -m "$bundle")
stamp=${stamp:-$bundle.started}
java_args=()
[ -z "$java_ids" ] || java_args=(--java-case-ids "$java_ids")
strict_counts=0
case ",$databases," in *,databricks-*|*,snowflake,*|*,lakebase,*) strict_counts=1 ;; esac
[ -z "$java_ids" ] || strict_counts=1
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
else
  echo "   BUNDLE_SHA256 $bundle_sha (verified only after the once-only stamp)"
  echo "   INPUTS_SHA256 $inputs_sha (verified only after the once-only stamp)"
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
if [ "$strict_counts" = 1 ] && [ "$stamp" != "$bundle.started" ]; then
  problem 'warehouse/Java draws must use the bundle-global default stamp'
fi
case ",$databases," in *,postgres,*)
  # Its value is a credential: only whether it is set is said.
  [ -n "${SDM_POSTGRES_TEST_DSN:-}" ] ||
    problem 'a Postgres leg needs SDM_POSTGRES_TEST_DSN (or leave postgres out of --databases)' ;;
esac
if [ -n "$java_ids" ]; then
  [ -f "$java_ids" ] || problem 'Java subset metadata is missing'
  java_ids_sha=''
  if [ -f "$java_ids" ]; then java_ids_sha=$(sha256sum "$java_ids" | cut -d' ' -f1); fi
  [ -n "$wheel_identity" ] && [ -f "$wheel_identity" ] || problem 'Java/core execution requires --wheel-identity'
  for name in SDM_JAVA_JAR HELDOUT_JAVA_CLASSPATH HELDOUT_JAVA_DRIVERS HELDOUT_JAVA_IDENTITY; do
    [ -n "${!name:-}" ] && [ -e "${!name}" ] || problem "$name is missing"
  done
fi
case ",$databases," in
  *,databricks-*|*,snowflake,*|*,lakebase,*) [ -n "$wheel_identity" ] && [ -f "$wheel_identity" ] || problem 'warehouse execution requires --wheel-identity' ;;
esac
case ",$databases," in *,databricks-spark,*)
  if [ -z "${HELDOUT_SPARK_PAYLOAD:-}" ] || [ -z "${HELDOUT_SPARK_VOLUME:-}" ]; then
    problem 'Spark requires HELDOUT_SPARK_PAYLOAD and HELDOUT_SPARK_VOLUME'
  elif ! (
    if [ -n "${HELDOUT_DATABRICKS_SPARK_ENV:-}" ]; then set -a; source "$HELDOUT_DATABRICKS_SPARK_ENV" || exit $?; fi
    "$py_cand" -I "$candidate/scripts/live/capture_run.py" heldout-spark --plan \
      --mode python-core --cases "$out/cases" --out "$out/spark.jsonl" \
      --payload "$HELDOUT_SPARK_PAYLOAD" --volume "$HELDOUT_SPARK_VOLUME" \
      --commit "$sha" --reference-commit "$reference_sha" --stamp "$stamp"
  ); then
    problem 'the prepared Spark payload or staging environment is incomplete'
  fi ;;
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
  [ -z "$wheel_identity" ] || cp "$wheel_identity" "$out/native-wheel-identity.json"
  [ -z "${HELDOUT_JAVA_IDENTITY:-}" ] || cp "$HELDOUT_JAVA_IDENTITY" "$out/java-identity.json"
}
artifact_ready() { # exact installed native/JAR bytes, before case exposure
  "$py_cand" -I - "$sha" "$wheel_identity" "${SDM_JAVA_JAR:-}" "${HELDOUT_JAVA_IDENTITY:-}" <<'PYARTIFACT'
import hashlib, json, pathlib, sys
sha, native, jar, jar_identity = sys.argv[1:]
def checked(identity, artifact):
    doc = json.loads(pathlib.Path(identity).read_text())
    assert doc['commit'] == sha and doc['tree_clean'] is True, 'artifact names another source'
    assert hashlib.sha256(pathlib.Path(artifact).read_bytes()).hexdigest() == doc['sha256'], 'artifact digest differs'
    return doc
if native:
    doc = checked(native, native.removesuffix('.json'))
    import sdm_client._native as loaded
    assert hashlib.sha256(pathlib.Path(loaded.__file__).read_bytes()).hexdigest() == doc['native_sha256'], 'installed native bytes differ'
if jar_identity:
    checked(jar_identity, jar)
PYARTIFACT
}
freeze_java() { # copy metadata bytes after the stamp, before interpreting IDs
  [ -n "$java_ids" ] || return 0
  cp "$java_ids" "$out/java-case-ids.txt"
  [ "$(sha256sum "$out/java-case-ids.txt" | cut -d' ' -f1)" = "$java_ids_sha" ] || return 1
  java_ids=$out/java-case-ids.txt
  java_args=(--java-case-ids "$java_ids")
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
stamp_once() { (set -C; printf 'candidate %s\nreference %s\n' "$sha" "$reference_sha" >"$stamp"); }
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
has_cases() { # DATABASE CASE_DIRECTORY: no driver, after the global stamp
  "$py_cand" -I - "$candidate/contract/runners" "$2" "$1" <<'PYCASE'
import pathlib, sys
sys.path.insert(0, sys.argv[1])
import run_python
raise SystemExit(0 if any(sys.argv[3] in run_python.case_legs(c) for c, _ in run_python.load_cases([pathlib.Path(sys.argv[2])])) else 3)
PYCASE
}
with_environment() { # DATABASE COMMAND...
  local db=$1 name env_file
  shift
  name=HELDOUT_${db^^}_ENV
  name=${name//-/_}
  env_file=${!name:-}
  (
    if [ -n "$env_file" ]; then set -a; source "$env_file" || return $?; fi
    "$@"
  )
}
python_leg() { # MODE DATABASE CASE_DIRECTORY OUT
  local run_mode=$1 db=$2 cases_dir=$3 target=$4
  local raw_target=$target
  local selection=0
  has_cases "$db" "$cases_dir" || selection=$?
  case "$selection" in 0) ;; 3) return 0 ;; *) return "$selection" ;; esac
  local py=$py_cand options=()
  [ "$run_mode" != python-ref ] || { py=$py_ref; options+=(--reference "$reference"); }
  case "$db" in
    databricks-spark)
      local remote=("$py_cand" -I "$candidate/scripts/live/capture_run.py" heldout-spark
        --mode "$run_mode" --cases "$cases_dir" --out "$target"
        --payload "$HELDOUT_SPARK_PAYLOAD" --volume "$HELDOUT_SPARK_VOLUME"
        --commit "$sha" --reference-commit "$reference_sha" --stamp "$stamp")
      [ "$run_mode" != python-ref ] || remote+=(--record "$target.record.jsonl")
      [ "$run_mode" != python-core ] || remote+=(--record-fetches "$out/legs/fetches.$db.jsonl")
      with_environment "$db" "${remote[@]}"
      return ;;
    databricks-sql | snowflake | lakebase)
      options+=(--guard-live)
      [ "$run_mode" != python-ref ] || options+=(--record "$target.record.jsonl") ;;
  esac
  case "$db:$run_mode" in
    databricks-sql:python-core | snowflake:python-core | lakebase:python-core) raw_target=$target.raw ;;
  esac
  if [ "$run_mode" = python-core ] && [ "$db" != duckdb ] && [ "$db" != sqlite ]; then
    options+=(--record-fetches "$out/legs/fetches.$db.jsonl")
  fi
  with_environment "$db" "$py" -I "$runpy" --mode "$run_mode" --database "$db" "${options[@]}" --out "$raw_target" "$cases_dir" || return $?
  if [ "$raw_target" != "$target" ]; then
    with_environment "$db" "$py_cand" -I "$runpy" publish --database "$db" --on "$raw_target" --out "$target" || return $?
  fi
  if [ "$run_mode" = python-ref ]; then
    case "$db" in
      databricks-sql | snowflake | lakebase)
        "$py_cand" -I "$candidate/contract/runners/capture_check.py" recording --cases "$cases_dir" --capture "$target.record.jsonl" --leg "$db" --locked ;;
    esac
  fi
}
rust_host() { # DATABASE: existing Rust journey, using actual host fetches
  local selection=0
  has_cases "$1" "$out/cases" || selection=$?
  case "$selection" in 0) ;; 3) return 0 ;; *) return "$selection" ;; esac
  local where=() target raw_target
  target=$(leg_file rust-host "$1")
  raw_target=$target
  case "$1" in
    duckdb) ;;
    sqlite) where=(--database sqlite) ;;
    *) where=(--replay "$out/legs/fetches.$1.jsonl" --replay-leg "$1") ;;
  esac
  case "$1" in databricks-* | snowflake | lakebase) raw_target=$target.raw ;; esac
  (cd "$candidate/rust" && cargo run -q -p sdm-testkit --bin sdm-contract-run \
    --locked -- "${where[@]+"${where[@]}"}" --out "$raw_target" \
    "$out/cases") || return $?
  if [ "$raw_target" != "$target" ]; then
    with_environment "$1" "$py_cand" -I "$runpy" publish --database "$1" --on "$raw_target" --out "$target"
  fi
}
java_host() { # DATABASE: exact authored subset through existing Java CLI
  "$py_cand" -I - "$candidate/contract/runners" "$out/cases" "$java_ids" "$1" "$(leg_file java "$1")" <<'PYJAVA'
import json, os, pathlib, sys
sys.path.insert(0, sys.argv[1])
import run_java
ids = {line.strip() for line in pathlib.Path(sys.argv[3]).read_text().splitlines() if line.strip()}
paths = [p for p in pathlib.Path(sys.argv[2]).glob('*.json') if json.loads(p.read_text())['id'] in ids and run_java.on_java(json.loads(p.read_text()), sys.argv[4])]
rows = run_java.run(paths, jar=pathlib.Path(os.environ['SDM_JAVA_JAR']),
    lib=pathlib.Path(os.environ['HELDOUT_JAVA_CLASSPATH']), drivers=pathlib.Path(os.environ['HELDOUT_JAVA_DRIVERS']), database=sys.argv[4]) if paths else []
pathlib.Path(sys.argv[5]).write_text(''.join(json.dumps(row) + '\n' for row in rows))
PYJAVA
}
compare() {
  "$py_cand" -I "$here/heldout.py" compare "$out" --prefix "$prefix" --databases "$databases" "${java_args[@]+"${java_args[@]}"}" >"$out/compare.txt" 2>&1 || true
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

# Freeze before any candidate-owned digest or reader sees the bundle.
do_ drivers_ready
do_ artifact_ready
do_ stamp_once
do_ freeze_java
do_ "${heldout[@]}" digests "$bundle" --prefix "$prefix" \
  --bundle-sha "$bundle_sha" --inputs-sha "$inputs_sha" "${count[@]+"${count[@]}"}"
echo "== 2. copy the bundle; the copy's inputs are the bundle's"
do_ copy_bundle
do_ "${heldout[@]}" digests "$out/cases-orig" --prefix "$prefix" \
  --inputs-sha "$inputs_sha" "${count[@]+"${count[@]}"}"

echo "== 3. load check only (the case schema; no core, no data)"
do_ load_check

echo "== 4. python-ref on the author's files, to learn the reference's outcomes"
for db in "${dbs[@]}"; do
  do_ python_leg python-ref "$db" "$out/cases-orig" "$(leg_file python-ref.orig "$db")"
done
do_ append_legs python-ref.orig

echo "== 5. each expect from python-ref (the author's was a prediction)"
do_ "${heldout[@]}" expect "$out/cases-orig" "$out/python-ref.orig.jsonl" \
  "$out/cases" --prefix "$prefix"

echo "== 6. python-ref again on the regenerated files: the same outcomes"
for db in "${dbs[@]}"; do
  do_ python_leg python-ref "$db" "$out/cases" "$(leg_file python-ref "$db")"
done
do_ append_legs python-ref
do_ "${heldout[@]}" same "$out/python-ref.orig.jsonl" "$out/python-ref.jsonl"

echo "== 7. THE run, once per runner (a second --execute stops here)"
for db in "${dbs[@]}"; do
  do_ leg python-core python_leg python-core "$db" "$out/cases" "$(leg_file python-core "$db")"
done
do_ append_legs python-core
do_ leg python-old "$py_cand" -I "$runpy" --mode python-old \
  --out "$out/python-old.jsonl" "$out/cases"
for db in "${dbs[@]}"; do
  do_ leg rust-host rust_host "$db"
done
do_ append_legs rust-host
if [ -n "$java_ids" ]; then
  for db in "${dbs[@]}"; do
    case "$db" in
      duckdb | sqlite | postgres | databricks-sql) do_ leg java with_environment "$db" java_host "$db" ;;
    esac
  done
  do_ append_legs java
fi

echo "== 8. compare"
do_ compare

echo "== 9. the per-case report and the counts"
do_ artifact_ready
if [ "$mode" = execute ]; then
  [ "$(git -C "$candidate" rev-parse HEAD)" = "$sha" ] && [ -z "$(git -C "$candidate" status --porcelain)" ] || { echo 'candidate changed during the draw' >&2; exit 1; }
fi
if [ ${#failed[@]} -gt 1 ]; then
  echo "!! ${#failed[@]} legs failed (${failed[*]}): no report; the owner rules"
  exit 1
fi
partial=()
[ ${#failed[@]} -eq 0 ] || partial=(--missing "${failed[0]}=the leg failed")
do_ "${heldout[@]}" report "$out" --prefix "$prefix" --databases "$databases" "${java_args[@]+"${java_args[@]}"}" \
  "${partial[@]+"${partial[@]}"}"
# Legacy offline execution reports counts without choosing a final verdict.
# A warehouse/Java phase-end draw must also fail its step on an H11 no-go.
counts_status=0
do_ "${heldout[@]}" counts "$out" --prefix "$prefix" --threshold "$threshold" \
  --databases "$databases" "${java_args[@]+"${java_args[@]}"}" \
  "${partial[@]+"${partial[@]}"}" || counts_status=$?
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
if [ "$strict_counts" = 1 ] && [ "$counts_status" != 0 ]; then exit "$counts_status"; fi
echo '== ran, once'
