#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Land a CI3 chain in order, each package on its own handoff stamp (owner decision 28; the
# owner's "2026-10-22 evening: CI off the landing path": no CI run is waited on). Since T3.0 a
# stack of up to 4 packages lands on one stamp instead: auto_land.py's "stack" and "stack_tip"
# entries (land.py --stack and stack-land); prefer those. Refuses at once unless every SHA in
# SHAS has a passing handoff stamp in this clone, then queues one package at a time and runs
# the watchdog checks every 2 minutes. Exits (printing why) on: a SHA with no stamp, a package
# failing / needing confirmation, a blocked agent, a rule violation, or when the whole chain
# has landed (CHAIN DONE).
#
# Each package in ORDER waits in the queue as <pkg>.next (an auto_land.py entry); this
# renames it to <pkg>.json only when every package before it has landed. Before anything else
# it checks that every package in ORDER has a <pkg>.next, <pkg>.json or <pkg>.landed (a
# restarted chain) in the queue, and exits 1 naming the ones that have none.
#
# Environment:
#   ORDER             the packages to land, in order (required), e.g. "t2.1-p1 t2.2-p1"
#   SHAS              the stack's candidate SHAs, each with a handoff stamp (required)
#   SDM_TASKS_DIR     the agents' task-output directory (*.output transcripts; required),
#                     read by stuck_agents.py and rule_watch.py
#   SDM_ORCH_STATE    state directory (default ~/.local/share/sdm-orchestration): the queue,
#                     logs/ and rule_watch.ignore (acknowledged violations, one pattern a line)
#   SDM_LAND_QUEUE    the queue directory (default $SDM_ORCH_STATE/land-queue)
#   SDM_PHASE_BRANCH  the phase branch, e.g. rust-core-phase-3 (required; exits 2 when
#                     unset), passed on to the tools
set -u
case "${1:-}" in -h|--help) sed -n '6,27p' "$0" | sed 's/^# \{0,1\}//'; exit 0;; esac
[ -n "${SDM_PHASE_BRANCH:-}" ] || {
  echo "land_chain.sh: set SDM_PHASE_BRANCH to the phase branch, e.g. rust-core-phase-3 (see --help)" >&2
  exit 2
}
export SDM_PHASE_BRANCH
: "${ORDER:?set ORDER to the packages to land, in order (see --help)}"
: "${SHAS:?set SHAS to the stack candidate SHAs (see --help)}"
T=${SDM_TASKS_DIR:?set SDM_TASKS_DIR to the agents task-output directory (see --help)}
STATE=${SDM_ORCH_STATE:-$HOME/.local/share/sdm-orchestration}
Q=${SDM_LAND_QUEUE:-$STATE/land-queue}
missing=""
for p in $ORDER; do
  [ -f "$Q/$p.next" ] || [ -f "$Q/$p.json" ] || [ -f "$Q/$p.landed" ] || missing="$missing $p"
done
[ -z "$missing" ] || { echo "NOT QUEUED: no $Q/<pkg>.next or .json for:$missing" >&2; exit 1; }
B=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$STATE/logs"
LOG=$STATE/logs/land-chain.log
checks() {
  "$B/cancel_superseded.sh" >> "$STATE/logs/cancel-superseded.log" 2>&1
  out=$(python3 "$B/auto_land.py" 2>&1); [ -n "$out" ] && echo "$(date -u +%H:%M) $out" >> "$LOG"
  # rule_watch.py drops the violations acknowledged in $STATE/rule_watch.ignore;
  # a WATCHDOG-ERROR line (a watchdog that cannot read the transcripts) stops the chain too.
  bad=$( { echo "$out" | grep -E '^(FAILED|CONFLICT|NEEDS-CONFIRM)'; python3 "$B/stuck_agents.py" "$T" 40 | grep -E '^(POSSIBLY-BLOCKED|WATCHDOG-ERROR)'; python3 "$B/rule_watch.py" "$T" 6 | grep -E '^(VIOLATION|WATCHDOG-ERROR)'; } 2>/dev/null)
  [ -n "$bad" ] && { echo "$bad" | cut -c1-300; exit 1; }
  echo "$out" | grep '^LANDED' | cut -c1-200
}
# 1. Every candidate handed off: a passing stamp each (handoff.py check), read in this clone.
unstamped=""
for s in $SHAS; do
  (cd "$B" && python3 ../handoff.py check "$s" >/dev/null 2>&1) || unstamped="$unstamped ${s:0:12}"
done
[ -z "$unstamped" ] || { echo "NOT STAMPED:$unstamped (run handoff.py on each first)"; exit 1; }
echo "$(date -u +%H:%M) every candidate stamped; landing in order" >> "$LOG"
# 2. Land in order, one package queued at a time.
for p in $ORDER; do
  [ -f "$Q/$p.next" ] && mv "$Q/$p.next" "$Q/$p.json"
  until [ -f "$Q/$p.landed" ]; do checks; [ -f "$Q/$p.landed" ] && break; sleep 120; done
  echo "$(date -u +%H:%M) $p landed" >> "$LOG"
done
echo "CHAIN DONE: $ORDER landed"
