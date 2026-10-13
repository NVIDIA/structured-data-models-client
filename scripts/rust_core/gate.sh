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
#       [--only CHECK | --locked 1a|1b [--evidence DIR]
#        [--wheel FILE --test-ops-wheel FILE]]
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
#                 --no-l0 leaves them to the job that has cargo;
#   coverage      the checkpoint's coverage.py --check on the candidate's
#                 commit: the scope manifests' rules, S0's paths each with
#                 an acceptance case, and no path or cell removed against
#                 the checkpoint without the owner (T1.15).
#
# --out keeps the observations made, for ratchet.py accept.
#
# The local modes (T1.15, with T1.19 merged in; decisions E53(b), H26, E77):
#
#   --only CHECK  runs one check, observations, ratchet, compare, l0 or
#                 coverage, and nothing else: the summary lists it alone
#                 and the last line says it was not the gate. The evidence
#                 a check needs is made inside it: under --only compare
#                 the observations (--run or --observations) and the
#                 effective expectations (ratchet.py effective), with no
#                 ratchet check.
#   --locked HALF the locked acceptance checks of HALF, 1a or 1b (design,
#                 "Locked acceptance checks"): every check above, the
#                 observations always, and the locked checks' own steps:
#                   runners      the observations cover python-core,
#                                python-old and rust-host;
#                   l1           cargo test --workspace on the candidate:
#                                the L1 corpus and the goldens;
#                   seeded       (1a) the seeded-fault suite: SEEDED's
#                                files under pytest, then the L0 guards'
#                                self-tests (selftest.py --no-clippy);
#                   wheel-smoke  (1a) the checkpoint's wheel_smoke.py in
#                                its two clean containers, on --wheel and
#                                --test-ops-wheel (CI's native-wheel and
#                                native-test-ops artifacts), failing
#                                without them;
#                   inventory    (1b) the 40 cases of the Phase 1b
#                                inventory and base.adventureworks are
#                                acceptance cases.
#                 Check 1 is runners, observations, ratchet, compare,
#                 coverage and l1; 2 is seeded; 3 is wheel-smoke; 4 is
#                 runners, observations, ratchet, compare, l1 and
#                 inventory; 5 is observations, ratchet, compare and
#                 inventory; 6, in both halves, is l0 and every step run.
#                 It prints the table of the half's checks, writes
#                 DIR/gate/<candidate sha>.json (--evidence DIR, by default
#                 sdm-gate-evidence in the clone's git directory, refused
#                 inside the worktree: the evidence stays outside the tested
#                 tree), adding its half to one the other half wrote, and
#                 exits non-zero when a locked check fails.

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
only=''
locked=''
evidence=''
wheel=''
test_ops_wheel=''
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
    --only | --locked)
      [ $# -ge 2 ] || { echo "gate.sh: $1 needs a value" >&2; exit 2; }
      if [ "$1" = --only ]; then only=$2; else locked=$2; fi
      for_harness+=("$1" "$2"); shift 2 ;;
    --evidence | --wheel | --test-ops-wheel)
      [ $# -ge 2 ] || { echo "gate.sh: $1 needs a value" >&2; exit 2; }
      case "$2" in /*) given_path=$2 ;; *) given_path=$PWD/$2 ;; esac
      case "$1" in
        --evidence) evidence=$given_path ;;
        --wheel) wheel=$given_path ;;
        *) test_ops_wheel=$given_path ;;
      esac
      for_harness+=("$1" "$given_path"); shift 2 ;;
    # Internal: the checkpoint's copy, judging the candidate at DIR.
    --harness-for) harness=1; candidate=$2; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) echo "gate.sh: unknown argument $1 (try --help)" >&2; exit 2 ;;
  esac
done
[ ${#where[@]} -eq 2 ] || { echo 'gate.sh: give --base or --phase-push' >&2; exit 2; }
case "$only" in
  '' | observations | ratchet | compare | coverage) ;;
  l0) [ "$l0" = 1 ] || { echo 'gate.sh: --only l0 with --no-l0 runs nothing' >&2; exit 2; } ;;
  *) echo "gate.sh: --only takes observations, ratchet, compare, l0 or coverage, not $only" >&2; exit 2 ;;
esac
case "$locked" in
  '' | 1a | 1b) ;;
  *) echo "gate.sh: --locked takes 1a or 1b, not $locked" >&2; exit 2 ;;
esac
if [ -n "$only" ] && [ -n "$locked" ]; then
  echo 'gate.sh: --only runs one check and --locked the locked ones; give one' >&2
  exit 2
fi
if [ -z "$locked" ] && [ -n "$evidence$wheel$test_ops_wheel" ]; then
  echo 'gate.sh: --evidence, --wheel and --test-ops-wheel are --locked'"'"'s' >&2
  exit 2
fi

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

# The evidence stays outside the tested tree (design, "CI"): the clone's git
# directory, or a directory outside the worktree, judged where each really
# resolves (every symbolic link in the existing part of a path followed),
# not by its spelling.
if [ -n "$locked" ]; then
  common=$(git -C "$candidate" rev-parse --path-format=absolute --git-common-dir) || exit 2
  [ -n "$evidence" ] || evidence=$common/sdm-gate-evidence
  "$PYTHON" - "$evidence" "$candidate" "$common" <<'EOF' || exit 2
import os, sys
evidence, candidate, common = (os.path.realpath(path) for path in sys.argv[1:])
def under(path, root):
    return os.path.commonpath([path, root]) == root
if under(evidence, candidate) and not under(evidence, common):
    sys.exit(f'gate.sh: the evidence directory {sys.argv[1]} resolves to {evidence}, '
             'inside the worktree it judges')
EOF
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
refuse() {
  echo "the given observations were made by the candidate's own runner, and"
  echo "the overlay differs from the checkpoint's: make them with --run"
  return 1
}
# The gate judges behaviour: without observations it would check the tree
# alone and say nothing of the outcomes, baselines or availability.
no_evidence() {
  echo 'no observations: give --run RUNNERS or --observations FILE; the'
  echo 'tree checks alone are no gate'
  return 1
}
evidence_of() {
  if [ -n "$runs" ]; then
    observe
  elif [ ${#observations[@]} -eq 0 ]; then
    no_evidence
  elif [ "$differs" = 1 ]; then
    refuse
  else
    printf 'given: %s\n' "${observations[@]}"
  fi
}
# Under --only, the evidence a check needs is made inside that check.
prepare() {
  [ -n "$only" ] || return 0
  if [ -n "$runs" ]; then
    observe
  elif [ ${#observations[@]} -gt 0 ] && [ "$differs" = 1 ]; then
    refuse
  fi
}
want() { [ -z "$only" ] || [ "$only" = "$1" ]; }

given=("${observations[@]+"${observations[@]}"}")
for runner in ${runs//,/ }; do given+=("$made/$runner.jsonl"); done
if [ -n "$locked" ] || [ "$only" = observations ] ||
  { [ -z "$only" ] && { [ -n "$runs" ] || [ ${#given[@]} -eq 0 ] || [ "$differs" = 1 ]; }; }; then
  step observations evidence_of
fi

with=()
for file in "${given[@]+"${given[@]}"}"; do with+=(--observations "$file"); done
judge_ratchet() {
  prepare || return 1
  if [ ${#given[@]} -eq 0 ]; then
    echo 'no observations: the ratchet judges the tree alone, not the baselines'
    echo 'or availability (give --run or --observations for those)'
  fi
  "$PYTHON" "$here/scripts/rust_core/ratchet.py" check --repo "$candidate" \
    --checkpoint "$checkpoint" "${bootstrap[@]+"${bootstrap[@]}"}" \
    --effective-out "$scratch/effective.json" \
    "${protected[@]+"${protected[@]}"}" "${with[@]+"${with[@]}"}"
}
want ratchet && step ratchet judge_ratchet

judge_compare() {
  if [ -n "$only" ]; then
    prepare || return 1
    [ ${#given[@]} -gt 0 ] || { no_evidence; return 1; }
    "$PYTHON" "$here/scripts/rust_core/ratchet.py" effective --repo "$candidate" \
      "${where[@]}" --out "$scratch/effective.json" || return 1
  fi
  "$PYTHON" "$here/scripts/rust_core/compare.py" \
    --cases "$candidate/contract/cases" --bases "$candidate/contract/bases.yaml" \
    --reference "$(ls "$candidate"/contract/reference/*.jsonl)" \
    --expect "$scratch/effective.json" "${given[@]}"
}
if want compare && { [ ${#given[@]} -gt 0 ] || [ -n "$only" ]; }; then
  step compare judge_compare
fi

if [ "$l0" = 1 ] && want l0; then
  step l0 sh -c '"$1" "$2/scripts/rust_core/l0_deps.py" --root "$3" --base "$4" && "$1" "$2/scripts/rust_core/l0_scan.py" --root "$3"' \
    sh "$PYTHON" "$here" "$candidate" "$checkpoint"
fi

if want coverage; then
  step coverage "$PYTHON" "$here/scripts/rust_core/coverage.py" --check \
    --repo "$candidate" --rev "$head" --checkpoint "$checkpoint"
fi

# --- the locked checks' own steps (--locked) --------------------------------

# The seeded-fault suite of locked check 2: each check's seeded faults
# (as-built, T1.15's table); no file here runs gate.sh --locked itself
# (test_gate_modes.py and test_gate_evidence_paths.py do, so stay out).
SEEDED=(
  contract/tests/test_gate_controls.py
  contract/tests/test_payload_and_result_rules.py
  contract/tests/test_capture.py
  contract/tests/test_result_vectors.py
  contract/tests/test_materialize_route.py
  contract/tests/test_route_proof_seeds.py
  tests/test_contract_ratchet.py
  tests/test_contract_ratchet_approval.py
  tests/test_contract_register.py
  tests/test_contract_coverage.py
  tests/test_contract_coverage_approval.py
  tests/test_contract_compare_depth.py
)
# The runtime cell the Python runners observed in, written to $scratch/cell
# (nothing when they name none or several): the locked checks' reference
# comparisons gate only in the reference's cell (H1).
runners_cover() {
  "$PYTHON" - "$scratch/cell" "${given[@]+"${given[@]}"}" <<'EOF'
import json, sys
seen, cells = set(), set()
for path in sys.argv[2:]:
    try:
        with open(path, encoding='utf-8') as file:
            for line in file:
                identity = json.loads(line)['identity']
                seen.add(identity['runner'])
                if identity['runner'] != 'rust-host':
                    cells.add(str(identity.get('cell')))
    except (OSError, ValueError, RecursionError, KeyError, TypeError, AttributeError):
        sys.exit(f'{path}: not observations it reads')
with open(sys.argv[1], 'w', encoding='utf-8') as file:
    file.write(cells.pop() if len(cells) == 1 else '')
missing = sorted({'python-core', 'python-old', 'rust-host'} - seen)
if missing:
    sys.exit(f'no observations of {", ".join(missing)}')
print('python-core, python-old and rust-host observed')
EOF
}
seeded_suite() {
  (cd "$candidate" && "$PYTHON" -m pytest -q -p no:cacheprovider "${SEEDED[@]}") &&
    (cd "$candidate" && "$PYTHON" scripts/rust_core/selftest.py --no-clippy)
}
smoke() {
  if [ -z "$wheel" ] || [ -z "$test_ops_wheel" ]; then
    echo "no wheels: give --wheel and --test-ops-wheel (CI's native-wheel and"
    echo 'native-test-ops artifacts)'
    return 1
  fi
  local lane status=0
  for lane in floor newest; do
    "$PYTHON" "$here/scripts/rust_core/wheel_smoke.py" --lane "$lane" \
      --wheel "$wheel" --test-ops-wheel "$test_ops_wheel" \
      --out "$scratch/wheel-smoke/$lane" --container || status=1
  done
  return $status
}
inventory() {
  "$PYTHON" - "$candidate" "$head" <<'EOF'
import json, subprocess, sys
import yaml
repo, head = sys.argv[1:]
def show(path):
    return subprocess.run(['git', '-C', repo, 'show', f'{head}:{path}'],
                          capture_output=True, text=True, check=True).stdout
try:
    listed = yaml.safe_load(show('docs/design/rust-core/phase-0/phase-1b-inventory.yaml'))
    wanted = [str(case['id']) for case in listed['cases']]
    names = subprocess.run(['git', '-C', repo, 'ls-tree', '-r', '-z', '--name-only', head,
                            'contract/cases'], capture_output=True, text=True, check=True).stdout
    status = {}
    for name in names.split('\0'):
        if name.endswith('.json'):
            case = json.loads(show(name))
            status[case['id']] = case['status']
except (subprocess.CalledProcessError, yaml.YAMLError, ValueError, RecursionError, KeyError,
        TypeError):
    sys.exit('the inventory or the cases: not what it reads')
if len(set(wanted)) != 40:
    sys.exit(f'the inventory holds {len(set(wanted))} cases, not the locked 40')
short = [case for case in wanted + ['base.adventureworks'] if status.get(case) != 'acceptance']
if short:
    sys.exit(f'{len(short)} of 41 not acceptance cases: {", ".join(short)}')
print('the 40 inventory cases and base.adventureworks are acceptance cases')
EOF
}
if [ -n "$locked" ]; then
  step runners runners_cover
  step l1 sh -c 'cd "$1/rust" && CARGO_TARGET_DIR="$1/rust/target" cargo test --workspace --locked' \
    sh "$candidate"
  if [ "$locked" = 1a ]; then
    step seeded seeded_suite
    step wheel-smoke smoke
  else
    step inventory inventory
  fi
fi

echo
echo "gate summary (judged by $judge):"
for i in "${!names[@]}"; do
  printf '  %-4s %s\n' "${results[$i]}" "${names[$i]}"
done

# The locked checks' table and evidence (H26).
if [ -n "$locked" ]; then
  result_of() {
    local i
    for i in "${!names[@]}"; do
      [ "${names[$i]}" = "$1" ] && { echo "${results[$i]}"; return; }
    done
    echo missing
  }
  if [ "$locked" = 1a ]; then
    checks=('1:runners observations ratchet compare coverage l1' '2:seeded' '3:wheel-smoke')
  else
    checks=('4:runners observations ratchet compare l1 inventory' '5:observations ratchet compare inventory')
  fi
  every=l0
  for name in "${names[@]}"; do [ "$name" = l0 ] || every+=" $name"; done
  checks+=("6:$every")
  # The reference comparisons of checks 1 and 4 gate only in the reference's
  # cell (H1; compare.py reports them elsewhere): in another, or with no
  # one cell observed, the check is reported, not gated, and fails.
  cell=$(cat "$scratch/cell" 2>/dev/null)
  reference=ref-pandas3
  echo
  echo "locked checks ($locked):"
  table=()
  for entry in "${checks[@]}"; do
    number=${entry%%:*}
    verdict=ok
    note=''
    for name in ${entry#*:}; do
      [ "$(result_of "$name")" = ok ] || verdict=FAIL
    done
    if [ "$verdict" = ok ] && { [ "$number" = 1 ] || [ "$number" = 4 ]; } &&
      [ "$cell" != "$reference" ]; then
      verdict=FAIL
      note=" (reported, not gated: observed in ${cell:-no one cell}, gated in $reference)"
    fi
    [ "$verdict" = ok ] || failed=1
    printf '  %-4s check %s: %s%s\n' "$verdict" "$number" "${entry#*:}" "$note"
    if [ -n "$note" ]; then verdict='reported, not gated'; fi
    table+=("$number=$verdict=${entry#*:}")
  done
  steps=()
  for i in "${!names[@]}"; do steps+=("${names[$i]}=${results[$i]}"); done
  mkdir -p "$evidence/gate" &&
    "$PYTHON" - "$evidence/gate/$head.json" "$head" "$locked" "$checkpoint" "$base" \
      "$cell" "$failed" "${#steps[@]}" "${steps[@]}" "${table[@]}" <<'EOF'
import json, os, sys
path, head, half, checkpoint, base, cell, failed, count = sys.argv[1:9]
steps = dict(item.split('=', 1) for item in sys.argv[9:9 + int(count)])
checks = {}
for item in sys.argv[9 + int(count):]:
    number, verdict, names = item.split('=', 2)
    checks[number] = {'result': verdict, 'steps': names.split()}
found = {'sha': head, 'halves': {}}
if os.path.exists(path):
    try:
        with open(path, encoding='utf-8') as file:
            found = json.load(file)
    except (OSError, ValueError, RecursionError):
        sys.exit(f'{path}: not evidence it reads')
    if not isinstance(found, dict) or found.get('sha') != head or not isinstance(
            found.get('halves'), dict):
        sys.exit(f'{path}: not this candidate\'s evidence')
found['halves'][half] = {'checkpoint': checkpoint, 'base': base, 'cell': cell or None,
                         'passed': failed == '0', 'checks': checks, 'steps': steps}
with open(path + '.tmp', 'w', encoding='utf-8') as file:
    json.dump(found, file, indent=2, sort_keys=True)
    file.write('\n')
os.replace(path + '.tmp', path)
print(f'evidence: {path}')
EOF
  [ $? = 0 ] || { echo 'gate: the evidence was not written'; failed=1; }
fi

if [ "$failed" = 1 ]; then
  echo 'gate: FAILED'
  exit 1
fi
if [ -n "$only" ]; then
  echo "gate: passed (--only $only: one check, not the gate)"
elif [ -n "$locked" ]; then
  echo "gate: passed (--locked $locked)"
else
  echo 'gate: passed'
fi
