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
#       [--only CHECK | --locked HALF [--evidence DIR]
#        [--wheel FILE --test-ops-wheel FILE [--wheel-run ID]]]
#
# --base is the phase-branch tip recorded at package start (or, in CI on a
# topic branch, the tip at run time); --phase-push BEFORE is CI on a push to
# the phase branch, BEFORE its previous tip. The checkpoint is resolved by
# ratchet.py checkpoint (git merge-base CANDIDATE BASE; a landing commit's
# own Base-SHA; one landing per phase push; self-comparison refused). This
# script then checks the checkpoint out into a worktree of its own and runs
# the checkpoint's gate.sh there, which resolves the checkpoint again with
# its own ratchet.py, refuses to judge from any other commit, and runs the
# checkpoint's harness against the candidate. A phase's first checkpoint is
# its fork point from the epic branch, no landing commit (T2.0). Phase 2's
# holds Phase 1's ratchet.py, which refuses it; for that one fork point,
# pinned below with the landing that stands in for it, and only when every
# condition bootstrap_problem reads from git holds, that landing's harness
# judges, handed --base <that landing> (the first landing itself judged as
# its parent's tree under its trailers), and the gate says so. A checkpoint
# with no ratchet.py
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
#   --locked HALF the locked acceptance checks of HALF (design, "Locked
#                 acceptance checks"), as the judging harness's tree names
#                 them (T2.0): each phase's
#                 docs/design/rust-core/phase-N/locked-set.yaml lists its
#                 halves (Phase 1's 1a and 1b), each half its own steps,
#                 its numbered checks and the steps each holds, the checks
#                 whose reference comparisons gate only in the reference's
#                 cell, the check that holds every step, and, for an
#                 inventory step, the inventory file, its case count and
#                 the cases added to it; a half no file names, or a file
#                 not of that format, is a usage error. Every check above,
#                 the observations always, and the locked checks' own
#                 steps:
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
#                                without them; each wheel held to its
#                                identity file, built from the candidate's
#                                commit (--commit), and the checkpoint's
#                                binding tests run on the candidate's
#                                contract inputs (--inputs: its contract
#                                version, which sets the checkpoint's own
#                                message builder's, its error codes, stub
#                                and sibling distributions, as the
#                                commit holds them); the evidence records
#                                each wheel's name, SHA-256 and identity
#                                commit, and the CI run --wheel-run names;
#                   inventory    (1b) the cases of the half's inventory (the
#                                Phase 1b inventory's 40) and the cases
#                                added to it (base.adventureworks) are
#                                acceptance cases.
#                 For Phase 1: check 1 is runners, observations, ratchet,
#                 compare, coverage and l1; 2 is seeded; 3 is wheel-smoke;
#                 4 is runners, observations, ratchet, compare, l1 and
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
wheel_run=''
harness=0
candidate=''
for_harness=()
while [ $# -gt 0 ]; do
  case "$1" in
    --base | --phase-push)
      [ $# -ge 2 ] || { echo "gate.sh: $1 needs a value" >&2; exit 2; }
      where=("$1" "$2"); shift 2 ;;
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
    --wheel-run)
      [ $# -ge 2 ] || { echo "gate.sh: $1 needs a value" >&2; exit 2; }
      wheel_run=$2; for_harness+=("$1" "$2"); shift 2 ;;
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
# A half's name here; whether a locked set names it, once the judging
# harness is known (below).
if [ -n "$locked" ] && ! [[ "$locked" =~ ^[0-9]+[a-z]*$ ]]; then
  echo "gate.sh: --locked takes a half a locked set names (Phase 1's 1a or 1b), not $locked" >&2
  exit 2
fi
if [ -n "$only" ] && [ -n "$locked" ]; then
  echo 'gate.sh: --only runs one check and --locked the locked ones; give one' >&2
  exit 2
fi
if [ -z "$locked" ] && [ -n "$evidence$wheel$test_ops_wheel$wheel_run" ]; then
  echo 'gate.sh: --evidence, --wheel, --test-ops-wheel and --wheel-run are --locked'"'"'s' >&2
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

# The one bootstrap a checkpoint is judged by another commit's harness for
# (T2.0): the phase, its fork point from the epic branch, and the landing
# that stands in for it, pinned here. SDM_GATE_BOOTSTRAP ("PHASE FORK
# STAND") replaces the pin for the gate's own tests only; ci.yml sets it
# nowhere (tests/test_t20_bootstrap.py holds it to that).
read -r bootstrap_phase bootstrap_fork bootstrap_stand <<<"${SDM_GATE_BOOTSTRAP:-rust-core-phase-2 9604074af96b8a05a5532c91249470595b38a4c5 bd44b34270ca59679ddf1fb4104d8aaa0373ddfb}"
# The data the harness reads from the checkpoint's tree: the fork point and
# the stand-in must agree on every one of these paths.
judging=(contract docs/design/rust-core/register.yaml docs/design/rust-core/register.schema.json
  ':(glob)docs/design/rust-core/phase-*/decisions.md' ':(glob)docs/design/rust-core/phase-*/locked-set.yaml'
  docs/design/rust-core/phase-0 rust scripts/rust_core/gate.sh scripts/rust_core/ratchet.py
  scripts/rust_core/compare.py scripts/rust_core/coverage.py scripts/rust_core/register.py
  scripts/rust_core/wheel_smoke.py scripts/rust_core/bridge_freeze.py scripts/rust_core/bridge_freeze.json
  scripts/rust_core/bridge_changes.json scripts/rust_core/l0_deps.py scripts/rust_core/l0_scan.py
  scripts/rust_core/selftest.py scripts/rust_core/emulation-allowlist.txt)
# bootstrap_problem: why the bootstrap does not apply to $candidate's HEAD
# and $checkpoint, or nothing (status 0) when every condition holds:
# - the checkpoint is the pinned fork point, which is the merge base of the
#   remote phase branch and the remote epic branch, read from git here;
# - the candidate holds no landing commit after the fork point, but for
#   itself as the first landing: an empty landing commit whose Base-SHA and
#   Checkpoint-SHA are the fork point;
# - the stand-in is the pinned landing, a landing commit, and the merged
#   phase's last landing the fork point holds: the second parent of the
#   newest merge on its first-parent history whose second parent is one;
# - the fork point only adds files to the stand-in's tree, and none of the
#   judging data above.
bootstrap_problem() {
  local g=(git -C "$candidate") remote fork landings head merged=''
  for remote in "$bootstrap_phase" rust-core; do
    "${g[@]}" rev-parse -q --verify "refs/remotes/origin/$remote^{commit}" >/dev/null ||
      { echo "no origin/$remote to read the fork point from"; return 1; }
  done
  fork=$("${g[@]}" merge-base "refs/remotes/origin/$bootstrap_phase" refs/remotes/origin/rust-core) || fork=''
  [ "$fork" = "$bootstrap_fork" ] ||
    { echo "the fork point of origin/$bootstrap_phase from origin/rust-core is ${fork:-none}, not $bootstrap_fork"; return 1; }
  head=$("${g[@]}" rev-parse HEAD)
  landings=$("${g[@]}" log --format=%H -E --grep='^Land ' "$bootstrap_fork..HEAD")
  if [ -n "$landings" ]; then
    [ "$landings" = "$head" ] ||
      { echo "the candidate holds a landing commit after the fork point: judge it against that landing"; return 1; }
    [ "$("${g[@]}" rev-list --parents -n 1 HEAD | wc -w)" = 2 ] &&
      [ "$("${g[@]}" rev-parse 'HEAD^{tree}')" = "$("${g[@]}" rev-parse 'HEAD^^{tree}')" ] ||
      { echo 'the candidate is a landing commit that is not empty, with one parent'; return 1; }
    for key in Base-SHA Checkpoint-SHA; do
      [ "$("${g[@]}" log -1 --format="%(trailers:key=$key,valueonly)" HEAD | tr -d ' \n')" = "$bootstrap_fork" ] ||
        { echo "the first landing's $key is not the fork point $bootstrap_fork"; return 1; }
    done
  fi
  while read -r _ _ second _; do
    [ -n "${second:-}" ] || continue
    case "$("${g[@]}" log -1 --format=%s "$second")" in
      'Land '*) merged=$second; break ;;
    esac
  done < <("${g[@]}" rev-list --first-parent --parents "$bootstrap_fork")
  [ "$merged" = "$bootstrap_stand" ] ||
    { echo "the merged phase's last landing the fork point holds is ${merged:-none}, not $bootstrap_stand"; return 1; }
  [ -n "$("${g[@]}" log -1 --format='%(trailers:key=Base-SHA,valueonly)' "$bootstrap_stand")" ] ||
    { echo "the stand-in $bootstrap_stand records no Base-SHA: no landing commit"; return 1; }
  if [ -n "$("${g[@]}" diff --name-only --diff-filter=a "$bootstrap_stand" "$bootstrap_fork")" ] ||
    ! "${g[@]}" diff --quiet "$bootstrap_stand" "$bootstrap_fork" -- "${judging[@]}"; then
    echo "the fork point and the stand-in differ in more than added files outside the judging data"
    return 1
  fi
  return 0
}

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
    # The bootstrap (T2.0; owner question H2): Phase 2's fork point holds
    # Phase 1's ratchet.py, which refuses any checkpoint that is no landing
    # commit, itself included, so its harness cannot judge. Then, and only
    # then, Phase 1's last landing stands in, its harness judging. Each
    # condition is read here from git, none from the candidate's own tools
    # (bootstrap_problem), and any other non-landing checkpoint takes the
    # normal path, its own harness resolving it again.
    case "$(git -C "$candidate" log -1 --format=%s "$checkpoint")" in
      'Land '*) ;;
      *)
        if [ "$checkpoint" = "$bootstrap_fork" ]; then
          why=$(bootstrap_problem) || { echo "gate.sh: the bootstrap is refused: $why" >&2; exit 1; }
          echo "gate.sh: the checkpoint $checkpoint is $bootstrap_phase's fork point, whose own ratchet.py refuses it; Phase 1's last landing $bootstrap_stand stands in, its harness judging (the T2.0 bootstrap)" >&2
          git -C "$candidate" worktree add --quiet --detach "$scratch/stand-in" "$bootstrap_stand" || exit 1
          worktrees+=("$scratch/stand-in")
          judged=$candidate
          extra=()
          if [ "$(git -C "$candidate" rev-list --count "$checkpoint..HEAD" --grep='^Land ' -E)" != 0 ]; then
            # The first landing itself (its phase push or a replay of it):
            # its parent's tree, which is its own, judged under its
            # trailers, as a landing is.
            git -C "$candidate" worktree add --quiet --detach "$scratch/landed" HEAD^ || exit 1
            worktrees+=("$scratch/landed")
            judged=$scratch/landed
            extra=(--protected-paths "$(git -C "$candidate" log -1 --format='%(trailers:key=Reviewed-by,valueonly,unfold) %(trailers:key=Protected-paths,valueonly,unfold)' HEAD | tr '\n' ' ')")
          fi
          "$scratch/stand-in/scripts/rust_core/gate.sh" --harness-for "$judged" \
            --base "$bootstrap_stand" "${for_harness[@]+"${for_harness[@]}"}" "${extra[@]+"${extra[@]}"}"
          exit $?
        fi ;;
    esac
    "$scratch/checkpoint/scripts/rust_core/gate.sh" --harness-for "$candidate" \
      "${where[@]}" "${for_harness[@]+"${for_harness[@]}"}"
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

# The locked set (T2.0): HALF as the judging harness's tree names it, in
# one of docs/design/rust-core/phase-*/locked-set.yaml.
locked_steps=''
locked_checks=()
locked_reference=''
locked_every=''
inventory_file=''
inventory_count=''
inventory_also=''
if [ -n "$locked" ]; then
  set_lines=$("$PYTHON" - "$here" "$locked" <<'EOF'
import glob, os, re, sys
import yaml
root, half = sys.argv[1:]
COMMON = ['runners', 'observations', 'ratchet', 'compare', 'coverage', 'l1']
STEPS = set(COMMON) | {'l0'}
OWN = {'seeded', 'wheel-smoke', 'inventory'}
CASE = re.compile(r'[a-z0-9][a-z0-9._-]*')
def fail(why):
    sys.exit(f'gate.sh: {why}')
found = {}
for path in sorted(glob.glob(os.path.join(root, 'docs/design/rust-core/phase-*/locked-set.yaml'))):
    name = os.path.relpath(path, root)
    try:
        with open(path, encoding='utf-8') as file:
            loaded = yaml.safe_load(file)
    except (OSError, UnicodeDecodeError, yaml.YAMLError, RecursionError):
        fail(f'{name}: not a locked set it reads')
    if not isinstance(loaded, dict) or not isinstance(loaded.get('halves'), dict) or \
            not set(loaded) <= {'format', 'halves'} or \
            loaded.get('format', 'sdm-locked-set/1') != 'sdm-locked-set/1':
        fail(f'{name}: not a locked set (halves, format sdm-locked-set/1)')
    for key, value in loaded['halves'].items():
        if not isinstance(key, str) or not re.fullmatch(r'[0-9]+[a-z]*', key):
            fail(f'{name}: the half {key!r} is not a half name')
        if key in found:
            fail(f'{name}: the half {key} is also in {found[key][0]}')
        found[key] = (name, value)
if half not in found:
    named = ', '.join(sorted(found)) or 'none'
    fail(f'--locked takes a half a locked set names ({named}), not {half}')
name, spec = found[half]
where = f'{name}, half {half}'
keys = {'steps', 'checks', 'reference', 'every', 'inventory', 'cases'}
if not isinstance(spec, dict) or not set(spec) <= keys or 'inventory' in spec and 'cases' in spec:
    fail(f'{where}: not a half (cases or inventory, and steps, checks, reference, every)')
if 'cases' in spec:
    cases = spec['cases']
    if not isinstance(cases, list) or not cases or len(set(map(str, cases))) != len(cases) \
            or not all(isinstance(x, str) and CASE.fullmatch(x) for x in cases):
        fail(f'{where}: cases is a list of case IDs')
locks = 'cases' in spec or 'inventory' in spec
# The short form: the cases alone, one check holding every common step
# and the inventory step, l0 and every step the next.
own = spec.get('steps', ['inventory'] if locks else [])
if not isinstance(own, list) or not all(isinstance(x, str) and x in OWN for x in own) \
        or len(set(own)) != len(own):
    fail(f'{where}: steps are some of {", ".join(sorted(OWN))}')
run = STEPS | set(own)
checks = spec.get('checks', {'1': COMMON + own})
if not isinstance(checks, dict) or not checks:
    fail(f'{where}: no checks')
lines = [f'steps {" ".join(own)}']
for number, steps in checks.items():
    if not isinstance(number, str) or not number.isdigit() or not isinstance(steps, list) \
            or not steps or not all(isinstance(x, str) and x in run for x in steps):
        fail(f'{where}: check {number!r} is not a number and steps the half runs')
    lines.append(f'check {number}:{" ".join(steps)}')
if not set(own) <= {step for steps in checks.values() for step in steps}:
    fail(f'{where}: a step of its own no check holds')
every = spec.get('every', str(max(int(n) for n in checks) + 1))
if not isinstance(every, str) or not every.isdigit() or every in checks:
    fail(f'{where}: every is a check number of its own')
lines.append(f'every {every}')
reference = spec.get('reference', [n for n, steps in checks.items() if 'compare' in steps][:1])
if not isinstance(reference, list) or not all(isinstance(x, str) and x in checks for x in reference):
    fail(f'{where}: reference lists checks of the half')
lines.append(f'reference {" ".join(reference)}')
if ('inventory' in own) != locks:
    fail(f'{where}: an inventory step and its cases or inventory go together')
if 'inventory' in spec:
    inventory = spec['inventory']
    if not isinstance(inventory, dict) or set(inventory) != {'file', 'count', 'also'} \
            or not isinstance(inventory['file'], str) \
            or not re.fullmatch(r'docs/design/rust-core/[A-Za-z0-9/._-]+\.yaml', inventory['file']) \
            or '..' in inventory['file'] \
            or not isinstance(inventory['count'], int) or isinstance(inventory['count'], bool) \
            or inventory['count'] < 1 or not isinstance(inventory['also'], list) \
            or not all(isinstance(x, str) and CASE.fullmatch(x) for x in inventory['also']):
        fail(f'{where}: inventory is a file, a count and the cases added')
    lines += [f'inventory-file {inventory["file"]}', f'inventory-count {inventory["count"]}',
              f'inventory-also {" ".join(inventory["also"])}']
elif 'cases' in spec:
    lines += ['inventory-count 0', f'inventory-also {" ".join(spec["cases"])}']
print('\n'.join(lines))
EOF
  ) || exit 2
  while IFS= read -r line; do
    case "$line" in
      'steps') locked_steps='' ;;
      'steps '*) locked_steps=${line#steps } ;;
      'check '*) locked_checks+=("${line#check }") ;;
      'every '*) locked_every=${line#every } ;;
      'reference') locked_reference='' ;;
      'reference '*) locked_reference=${line#reference } ;;
      'inventory-file '*) inventory_file=${line#inventory-file } ;;
      'inventory-count '*) inventory_count=${line#inventory-count } ;;
      'inventory-also') inventory_also='' ;;
      'inventory-also '*) inventory_also=${line#inventory-also } ;;
    esac
  done <<<"$set_lines"
fi
locked_has() { [[ " $locked_steps " == *" $1 "* ]]; }
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
      --inputs "$candidate" --commit "$head" \
      --out "$scratch/wheel-smoke/$lane" --container || status=1
  done
  return $status
}
inventory() {
  # shellcheck disable=SC2086 # the cases added, a word each
  "$PYTHON" - "$candidate" "$head" "$inventory_file" "$inventory_count" $inventory_also <<'EOF'
import json, subprocess, sys
import yaml
repo, head, inventory, count = sys.argv[1:5]
count, also = int(count or 0), sys.argv[5:]
def show(path):
    return subprocess.run(['git', '-C', repo, 'show', f'{head}:{path}'],
                          capture_output=True, text=True, check=True).stdout
try:
    # The half's inventory file, or (the short form) its cases alone.
    listed = yaml.safe_load(show(inventory)) if inventory else {'cases': []}
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
if inventory and len(set(wanted)) != count:
    sys.exit(f'the inventory holds {len(set(wanted))} cases, not the locked {count}')
short = [case for case in wanted + also if status.get(case) != 'acceptance']
if short:
    sys.exit(f'{len(short)} of {count + len(also)} not acceptance cases: {", ".join(short)}')
if inventory:
    print(f'the {count} inventory cases' + (f' and {" and ".join(also)}' if also else '')
          + ' are acceptance cases')
else:
    print(f'the {len(also)} locked cases are acceptance cases')
EOF
}
if [ -n "$locked" ]; then
  step runners runners_cover
  step l1 sh -c 'cd "$1/rust" && CARGO_TARGET_DIR="$1/rust/target" cargo test --workspace --locked' \
    sh "$candidate"
  if locked_has seeded; then step seeded seeded_suite; fi
  if locked_has wheel-smoke; then step wheel-smoke smoke; fi
  if locked_has inventory; then step inventory inventory; fi
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
  checks=("${locked_checks[@]}")
  every=l0
  for name in "${names[@]}"; do [ "$name" = l0 ] || every+=" $name"; done
  checks+=("$locked_every:$every")
  # The reference comparisons of the half's reference checks (Phase 1's 1
  # and 4) gate only in the reference's cell (H1; compare.py reports them
  # elsewhere): in another, or with no one cell observed, the check is
  # reported, not gated, and fails.
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
    if [ "$verdict" = ok ] && [[ " $locked_reference " == *" $number "* ]] &&
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
    SDM_GATE_WHEEL=$wheel SDM_GATE_TEST_OPS_WHEEL=$test_ops_wheel SDM_GATE_WHEEL_RUN=$wheel_run \
    SDM_GATE_EVERY=$locked_every \
    "$PYTHON" - "$evidence/gate/$head.json" "$head" "$locked" "$checkpoint" "$base" \
      "$cell" "$failed" "${#steps[@]}" "${steps[@]}" "${table[@]}" <<'EOF'
import json, os, sys
path, head, half, checkpoint, base, cell, failed, count = sys.argv[1:9]
steps = dict(item.split('=', 1) for item in sys.argv[9:9 + int(count)])
checks = {}
for item in sys.argv[9 + int(count):]:
    number, verdict, names = item.split('=', 2)
    checks[number] = {'result': verdict, 'steps': names.split()}
# The wheel-smoke check's wheels (Phase 1's check 3), as given: each
# file's name and SHA-256, the commit its identity file (a build record,
# not an attestation) names, and the CI run the operator says they came
# from.
smoke = [number for number, check in checks.items()
         if 'wheel-smoke' in check['steps'] and number != os.environ.get('SDM_GATE_EVERY')]
if smoke:
    import hashlib
    def wheel(given):
        if not given or not os.path.isfile(given):
            return None
        with open(given, 'rb') as file:
            digest = hashlib.sha256(file.read()).hexdigest()
        try:
            with open(given + '.json', encoding='utf-8') as file:
                commit = json.load(file).get('commit')
        except (OSError, ValueError, RecursionError, AttributeError):
            commit = None
        return {'name': os.path.basename(given), 'sha256': digest,
                'commit': commit if isinstance(commit, str) else None}
    checks[smoke[0]]['wheels'] = {
        'production': wheel(os.environ.get('SDM_GATE_WHEEL')),
        'test-ops': wheel(os.environ.get('SDM_GATE_TEST_OPS_WHEEL')),
        'run': os.environ.get('SDM_GATE_WHEEL_RUN') or None,
    }
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
