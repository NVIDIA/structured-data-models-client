#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The gate (design, "Reference observations and the gate"; decisions H2 and
# H3), minimal as T1.14 builds it: the candidate, this worktree's committed
# HEAD, judged by its checkpoint's harness from the checkpoint's own worktree.
#
#   scripts/rust_core/gate.sh (--base REV | --phase-push BEFORE)
#       [--protected-paths TEXT] [--run RUNNERS] [--observations FILE]...
#       [--out DIR] [--no-l0]
#
# --base is the phase-branch tip recorded at package start (or, in CI on a
# topic branch, the tip at run time); --phase-push BEFORE is CI on a push to
# the phase branch, BEFORE its previous tip. The checkpoint is resolved by
# ratchet.py checkpoint (git merge-base CANDIDATE BASE; a landing commit's
# own Base-SHA; one landing per phase push; self-comparison refused). This
# script then checks the checkpoint out into a worktree of its own and runs
# the checkpoint's gate.sh there, which resolves the checkpoint again with
# its own ratchet.py, refuses to judge from any other commit, and runs the
# checkpoint's harness against the candidate. A checkpoint with no ratchet.py
# is the bootstrap, before T1.14's landing: the candidate's harness runs, and
# no acceptance claim is made. Each run prints the commit whose harness
# judged: from T1.14's landing on, that is the checkpoint, a landing commit.
#
# The checks, each run even after one fails, and the summary at the end:
#
#   observations  required: with neither --run nor --observations the gate
#                 fails, since the tree checks alone judge no behaviour;
#                 --run python-core,python-old,rust-host makes them here:
#                 the Python runners with $PYTHON (default python3, an
#                 environment holding the candidate's client), rust-host with
#                 cargo, each with the checkpoint's run_python.py and test
#                 kit case loader and observation writer overlaid on the
#                 candidate (a scratch worktree when they differ); or
#                 --observations FILE gives them, refused when the overlay
#                 differs, since they were made by the candidate's own;
#   ratchet       the checkpoint's ratchet.py check: protected paths against
#                 the landing's trailers (or --protected-paths before
#                 landing), reference immutability, removed, demoted and
#                 changed cases, the register, the policy's and
#                 availability's relaxations, KNOWN_DIFFERING, and with
#                 observations the baselines and availability;
#   compare       the checkpoint's compare.py on the observations, against
#                 the candidate's cases, bases and reference, which the
#                 ratchet lets differ from the checkpoint's only as approved,
#                 each baselined case's outcome against its effective
#                 expectation (--expect, which the ratchet writes: the
#                 checkpoint's baseline or an approved transition from it);
#   l0            the checkpoint's l0_deps.py (with the bridge freeze, based
#                 at the checkpoint) and l0_scan.py on the candidate's tree;
#                 --no-l0 leaves them to the job that has cargo.
#
# --out keeps the observations made, for ratchet.py accept. Locked checks,
# --only and the evidence file are T1.19's (decision E53(b)).

set -uo pipefail

usage() {
  awk 'NR > 5 && /^#/ { sub(/^# ?/, ""); print; next } NR > 5 { exit }' "$0"
}

where=()
protected=()
runs=''
observations=()
out=''
l0=1
harness=0
candidate=''
for_harness=()
while [ $# -gt 0 ]; do
  case "$1" in
    --base | --phase-push)
      [ $# -ge 2 ] || { echo "gate.sh: $1 needs a value" >&2; exit 2; }
      where=("$1" "$2"); for_harness+=("$1" "$2"); shift 2 ;;
    --protected-paths)
      [ $# -ge 2 ] || { echo "gate.sh: $1 needs a value" >&2; exit 2; }
      protected=(--protected-paths "$2"); for_harness+=("$1" "$2"); shift 2 ;;
    --run)
      [ $# -ge 2 ] || { echo "gate.sh: $1 needs a value" >&2; exit 2; }
      runs=$2; for_harness+=("$1" "$2"); shift 2 ;;
    --observations)
      [ $# -ge 2 ] || { echo "gate.sh: $1 needs a value" >&2; exit 2; }
      observations+=("$(cd "$(dirname "$2")" && pwd)/$(basename "$2")")
      for_harness+=("$1" "${observations[${#observations[@]}-1]}"); shift 2 ;;
    --out)
      [ $# -ge 2 ] || { echo "gate.sh: $1 needs a value" >&2; exit 2; }
      mkdir -p "$2" && out=$(cd "$2" && pwd); for_harness+=("$1" "$out"); shift 2 ;;
    --no-l0) l0=0; for_harness+=("$1"); shift ;;
    # Internal: the checkpoint's copy, judging the candidate at DIR.
    --harness-for) harness=1; candidate=$2; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "gate.sh: unknown argument $1 (try --help)" >&2; exit 2 ;;
  esac
done
[ ${#where[@]} -eq 2 ] || { echo 'gate.sh: give --base or --phase-push' >&2; exit 2; }

PYTHON=${PYTHON:-python3}
here=$(cd "$(dirname "$0")/../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/sdm-gate.XXXXXX")
worktrees=()
cleanup() {
  for tree in "${worktrees[@]+"${worktrees[@]}"}"; do
    git -C "$candidate" worktree remove --force "$tree" >/dev/null 2>&1
  done
  rm -rf "$scratch"
}
trap cleanup EXIT

if [ "$harness" = 0 ]; then
  candidate=$(git rev-parse --show-toplevel) || exit 2
  if [ -n "$(git -C "$candidate" status --porcelain --untracked-files=no)" ]; then
    echo 'gate.sh: the worktree has uncommitted changes; the gate judges a commit' >&2
    exit 2
  fi
  resolved=$("$PYTHON" "$candidate/scripts/rust_core/ratchet.py" checkpoint \
    --repo "$candidate" "${where[@]}") || exit 1
  read -r checkpoint _ kind <<<"$resolved"
  # The bootstrap only where the checkpoint's own tree has no ratchet.py
  # (T1.18): a ratchet that reads it otherwise does not skip the harness.
  if [ "$kind" = bootstrap ] &&
    git -C "$candidate" cat-file -e "$checkpoint:scripts/rust_core/ratchet.py" 2>/dev/null; then
    echo "gate.sh: the checkpoint $checkpoint holds ratchet.py, so its harness judges, not the bootstrap" >&2
    exit 1
  fi
  if [ "$kind" = harness ]; then
    git -C "$candidate" worktree add --quiet --detach "$scratch/checkpoint" "$checkpoint" || exit 1
    worktrees+=("$scratch/checkpoint")
    "$scratch/checkpoint/scripts/rust_core/gate.sh" --harness-for "$candidate" "${for_harness[@]}"
    exit $?
  fi
  here=$candidate
fi

# --- the harness: $here judges $candidate ---------------------------------------

resolved=$("$PYTHON" "$here/scripts/rust_core/ratchet.py" checkpoint \
  --repo "$candidate" "${where[@]}") || exit 1
read -r checkpoint base kind <<<"$resolved"
head=$(git -C "$candidate" rev-parse HEAD)
bootstrap=()
if [ "$kind" = bootstrap ]; then
  bootstrap=(--bootstrap)
  judge="bootstrap: the checkpoint has no harness (before T1.14's landing); the candidate's harness runs, and no acceptance claim is made"
elif [ "$(git -C "$here" rev-parse HEAD)" != "$checkpoint" ]; then
  echo "gate.sh: this harness is not the checkpoint's ($checkpoint)" >&2
  exit 1
else
  judge="the checkpoint's harness, at landing $(git -C "$here" log -1 --format='%h %s' "$checkpoint")"
fi
echo "gate: candidate $head"
echo "gate: base $base, checkpoint $checkpoint"
echo "gate: judged by $judge"

names=()
results=()
failed=0
logdir="$scratch/logs"
mkdir -p "$logdir"
step() {
  local name=$1 log="$logdir/${#names[@]}.log" status
  shift
  printf '%-14s ' "$name"
  "$@" >"$log" 2>&1
  status=$?
  names+=("$name")
  if [ "$status" = 0 ]; then
    results+=(ok)
    echo ok
  else
    results+=(FAIL)
    failed=1
    echo "FAIL (exit $status)"
  fi
  sed 's/^/    | /' "$log"
}

# The observation projection and capture the checkpoint's harness overlays
# on the candidate (H2).
overlay=(
  contract/runners/run_python.py
  rust/crates/sdm-testkit/src/case.rs
  rust/crates/sdm-testkit/src/observation.rs
)
differs=0
if [ "$kind" = harness ]; then
  for path in "${overlay[@]}"; do
    cmp -s "$here/$path" "$candidate/$path" || differs=1
  done
fi

made="${out:-$scratch/observations}"
mkdir -p "$made"
observe() {
  local tree=$candidate runner
  if [ "$differs" = 1 ]; then
    tree="$scratch/judge"
    git -C "$candidate" worktree add --quiet --detach "$tree" "$head" || return 1
    worktrees+=("$tree")
    for path in "${overlay[@]}"; do cp "$here/$path" "$tree/$path" || return 1; done
    echo "the checkpoint's runner, case loader and observation writer overlaid"
  fi
  for runner in ${runs//,/ }; do
    case "$runner" in
      python-core | python-old)
        "$PYTHON" -I "$tree/contract/runners/run_python.py" --mode "$runner" \
          --out "$made/$runner.jsonl" "$tree/contract/cases" || return 1 ;;
      rust-host)
        (cd "$tree/rust" && CARGO_TARGET_DIR="$candidate/rust/target" cargo run -q \
          -p sdm-testkit --bin sdm-contract-run --locked -- \
          --out "$made/rust-host.jsonl" ../contract/cases) || return 1 ;;
      *) echo "no runner $runner"; return 1 ;;
    esac
    echo "$runner: $made/$runner.jsonl"
  done
}
given=("${observations[@]+"${observations[@]}"}")
if [ -n "$runs" ]; then
  step observations observe
  for runner in ${runs//,/ }; do given+=("$made/$runner.jsonl"); done
elif [ ${#given[@]} -gt 0 ] && [ "$differs" = 1 ]; then
  refuse() {
    echo "the given observations were made by the candidate's own runner, and"
    echo "the overlay differs from the checkpoint's: make them with --run"
    return 1
  }
  step observations refuse
fi
if [ ${#given[@]} -eq 0 ]; then
  # The gate judges behaviour: without observations it would check the tree
  # alone and say nothing of the outcomes, baselines or availability.
  no_evidence() {
    echo 'no observations: give --run RUNNERS or --observations FILE; the'
    echo 'tree checks alone are no gate'
    return 1
  }
  step observations no_evidence
fi

with=()
for file in "${given[@]+"${given[@]}"}"; do with+=(--observations "$file"); done
step ratchet "$PYTHON" "$here/scripts/rust_core/ratchet.py" check --repo "$candidate" \
  --checkpoint "$checkpoint" "${bootstrap[@]+"${bootstrap[@]}"}" \
  --effective-out "$scratch/effective.json" \
  "${protected[@]+"${protected[@]}"}" "${with[@]+"${with[@]}"}"

if [ ${#given[@]} -gt 0 ]; then
  step compare "$PYTHON" "$here/scripts/rust_core/compare.py" \
    --cases "$candidate/contract/cases" --bases "$candidate/contract/bases.yaml" \
    --reference "$(ls "$candidate"/contract/reference/*.jsonl)" \
    --expect "$scratch/effective.json" "${given[@]}"
fi

if [ "$l0" = 1 ]; then
  step l0 sh -c '"$1" "$2/scripts/rust_core/l0_deps.py" --root "$3" --base "$4" && "$1" "$2/scripts/rust_core/l0_scan.py" --root "$3"' \
    sh "$PYTHON" "$here" "$candidate" "$checkpoint"
fi

echo
echo "gate summary (judged by $judge):"
for i in "${!names[@]}"; do
  printf '  %-4s %s\n' "${results[$i]}" "${names[$i]}"
done
if [ "$failed" = 1 ]; then
  echo 'gate: FAILED'
  exit 1
fi
echo 'gate: passed'
