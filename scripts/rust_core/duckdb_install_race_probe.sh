#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# T2.0's probe for exit 7 (the independent test author's): the DuckDB
# extension install race, as a stress run.
#
#   scripts/rust_core/duckdb_install_race_probe.sh [ROUNDS] [PROCESSES]
#
# sdm-testkit's duck::open installs DuckDB's signed json and icu extensions
# the first time, into $HOME/.duckdb/extensions, and checks the installed
# file's digest. Tests run in parallel (threads under cargo test, processes
# under nextest) can read the file while another is still installing it:
# "the json extension at .../json.duckdb_extension.tmp-... is not the one
# recorded" (CI run 38842796035). No deterministic test reproduces that in
# CI (it needs an empty extension directory and DuckDB's repository), so
# this is a documented probe, run by hand before and after the fix.
#
# It builds the duckdb_provider test binary once, then for each of ROUNDS
# (default 10) makes a fresh, empty HOME and starts PROCESSES (default 8)
# copies of one test that opens a connection
# (the_host_s_duckdb_links_the_reference_version_with_icu), all at once,
# as nextest would. The extension directory is shared by the copies of a
# round and fresh for each round, so every round races its first install.
# With PROBE_ICU_FROM=DIR (a directory holding an installed
# icu.duckdb_extension and its .info), icu is copied in first, so only
# json's install races and each round downloads less.
#
# If the fix installs the extensions in a step before the tests (rather
# than making the install itself safe), give that step as PROBE_PREPARE, a
# command run with bash -c in each round's fresh HOME before the copies
# start; leave it unset to probe the install in duck::open itself. Say
# which you probed.
#
# It prints one line per round and a last line
#   race: F of N processes failed over R rounds
# and exits 1 if any failed, 0 if none did. Network: DuckDB's extension
# repository, as CI's first run on an empty cache reaches it. Run it from
# the repository root; CARGO_TARGET_DIR is honoured.

set -u
rounds=${1:-10}
processes=${2:-8}
[[ "$rounds" =~ ^[1-9][0-9]*$ && "$processes" =~ ^[1-9][0-9]*$ ]] || {
  echo 'duckdb_install_race_probe.sh: ROUNDS and PROCESSES are positive numbers' >&2
  exit 2
}
root=$(cd "$(dirname "$0")/../.." && pwd)
test_name=the_host_s_duckdb_links_the_reference_version_with_icu

built=$(cd "$root/rust" && cargo test -p sdm-testkit --test duckdb_provider \
  --no-run --locked --message-format=json 2>/dev/null |
  python3 -c '
import json, sys
for line in sys.stdin:
    try:
        message = json.loads(line)
    except ValueError:
        continue
    if message.get("reason") == "compiler-artifact" and message.get("executable") \
            and message["target"]["name"] == "duckdb_provider":
        print(message["executable"])
') || true
[ -n "$built" ] && [ -x "$built" ] || {
  echo 'duckdb_install_race_probe.sh: the duckdb_provider test binary did not build' >&2
  exit 2
}

scratch=$(mktemp -d "${TMPDIR:-/tmp}/sdm-duckdb-race.XXXXXX")
failed=0
total=0
for round in $(seq 1 "$rounds"); do
  home=$scratch/home-$round
  mkdir -p "$home"
  if [ -n "${PROBE_ICU_FROM:-}" ]; then
    platform_dir=$(cd "$PROBE_ICU_FROM" && pwd)
    rel=${platform_dir#*/.duckdb/extensions/}
    mkdir -p "$home/.duckdb/extensions/$rel"
    cp "$PROBE_ICU_FROM"/icu.duckdb_extension* "$home/.duckdb/extensions/$rel/"
  fi
  if [ -n "${PROBE_PREPARE:-}" ]; then
    (cd "$root" && HOME=$home bash -c "$PROBE_PREPARE") </dev/null \
      >"$scratch/prepare-$round.log" 2>&1 || {
      echo "round $round: PROBE_PREPARE failed (log: $scratch/prepare-$round.log)"
      failed=$((failed + processes)); total=$((total + processes)); continue
    }
  fi
  pids=()
  for i in $(seq 1 "$processes"); do
    (cd "$root/rust/crates/sdm-testkit" &&
      HOME=$home "$built" --exact "$test_name" --test-threads=1 -q) \
      </dev/null >"$scratch/round-$round-$i.log" 2>&1 &
    pids+=("$!")
  done
  bad=0
  for pid in "${pids[@]}"; do wait "$pid" || bad=$((bad + 1)); done
  reasons=$(grep -ho 'is not the one recorded\|installing DuckDB.s [a-z]* extension\|reading [^ ]*\.tmp[^ ]*' \
    "$scratch"/round-"$round"-*.log 2>/dev/null | sort | uniq -c | tr -s ' ' | paste -sd ';' -)
  echo "round $round: $bad of $processes failed${reasons:+ ($reasons)}"
  failed=$((failed + bad)); total=$((total + processes))
done
echo "race: $failed of $total processes failed over $rounds rounds (logs: $scratch)"
[ "$failed" -eq 0 ]
