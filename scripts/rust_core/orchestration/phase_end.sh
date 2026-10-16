#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Every end-of-phase run, in order, from one final SHA: the evidence pack and
# the one held-out run (handbook, "Ending a phase", step 3).
#
#   phase_end.sh --phase N --sha SHA [options] [-- HELDOUT_ARGS...]   # dry run
#   phase_end.sh --phase N --sha SHA --execute [options] [-- HELDOUT_ARGS...]
#
# Options:
#   --phase N          the phase (or SDM_PHASE); the branch is rust-core-phase-N
#   --sha SHA          the final SHA on origin/rust-core-phase-N, 40 characters
#                      (or SDM_PHASE_END_SHA)
#   --work DIR         where everything goes (or SDM_PHASE_END_WORK; default
#                      ${TMPDIR:-/tmp}/sdm-phase-N-end-<sha7>)
#   --execute          run; without it nothing is built, run or written
#   --base REV         base for a SHA that is not a landing commit
#   --run ID           CI run whose native-wheel ships (default: resolved)
#   --wheel-commit C   the commit that run is on (default: resolved)
#   --landing L        when SHA is an EMPTY evidence commit on top of landing
#                      L (same tree): the phase-branch commit the CI wheel
#                      and the live check name. Default: SHA.
#   --half H           the locked half whose gate run makes the observations
#                      (default 1b)
#   --until N          stop after step N (--until 2: everything before the
#                      one-shot held-out run)
#   --allow-proposed   go on with register entries still proposed (they do
#                      NOT count as approved before the held-out run)
#   HELDOUT_ARGS       added to scripts/rust_core/heldout_run.sh's command
#                      line after --candidate <evidence worktree> --python
#                      <reference cell python> [--bundle B] [--out D] (from
#                      the variables below), so they can override those;
#                      its other options come from here or its HELDOUT_*
#                      variables. It is called once so (its own dry run),
#                      then once with --execute appended.
#
# Environment (no defaults unless stated; nothing private is built in):
#   SDM_CELLS          required: "NAME=PYTHON ..." for each runtime cell, the
#                      first the reference cell (the gate runs there); the
#                      others get python-core and python-old runs. NAME is
#                      run_python.py's cell name, which the build step checks.
#   SDM_TOOLS_BIN      required: a bin directory with python and the pinned
#                      maturin, for native_wheel.py
#   SDM_HELDOUT_BUNDLE required (or HELDOUT_BUNDLE): the held-out bundle,
#                      for heldout_run.sh --bundle and evidence.py --heldout
#   SDM_HELDOUT_RUN_DIR  (or HELDOUT_OUT) the held-out run's directory,
#                      heldout_run.sh --out; the pack copies its report.md,
#                      compare.txt, identity.txt and the digest of
#                      register-at-run-start.yaml. Unset: copied by hand.
#   SDM_LIVE_CHECK_CMD a command for the optional live install check (step
#                      4), run with bash -c; skipped when unset. It gets
#                      PHASE_END_{PHASE,SHA,LANDING,RUN,WHEEL_COMMIT,CI_WHEELS,
#                      PUBLIC} in its environment, writes anything the pack
#                      may publish into $PHASE_END_PUBLIC, and must exit 0
#                      only on a pass. Its output goes to a private log.
#   SDM_INVENTORY      tree path of the inventory evidence.py reads (default
#                      docs/design/rust-core/phase-0/phase-1b-inventory.yaml)
#   SDM_REF_SHA        the reference recording (default: the one
#                      contract/reference/*.jsonl at SHA)
#   SDM_REPO           the clone (default: this script's git toplevel)
#   SDM_GH_REPO        default NVIDIA/structured-data-models-client
#
# Steps (each logged to $WORK/logs/<n>-<name>.log and stamped when done, so a
# rerun after a fix resumes at the failed step and never repeats a one-shot):
#   0 preflight  local reads, git fetch, gh read-only probes
#   1 build      the local client wheel -> every cell; CI's native wheel
#   2 evidence   gate.sh --locked HALF --out OBS in the reference cell,
#                python-core and python-old in the other cells, cargo tree,
#                tokei; then evidence.py into a scratch dir WITHOUT the
#                held-out bundle, so a refused input shows before the one shot
#   3 heldout    heldout_run.sh dry, then --execute: ONCE
#   4 live       SDM_LIVE_CHECK_CMD, once; skipped (not stamped) when unset
#   5 pack       evidence.py with --heldout, plus held-out, live, boundary,
#                size and identity files, staged in $WORK/pack/evidence
#
# A one-shot step (3, 4) writes a "started" marker before it runs; a marker
# with no done stamp stops every rerun: a person decides, never a rerun.
# $WORK belongs to one SHA: the first --execute records it in $WORK/SHA, a
# different --sha on that $WORK stops, and the build and evidence worktrees
# are checked to be at SHA before each step uses them.
# Never: rm -rf, --force, sudo, git config, push. Network probes run under a
# perl alarm with stdin /dev/null.
#
# Ported from Phase 1's private run_all.sh with its workspace, profile and
# storage names taken out. Phase 1's evidence pack recorded the digest of that
# private original; this sanitized port won't reproduce it.
set -euo pipefail
umask 077

usage() { awk 'NR > 5 && /^#/ { sub(/^# ?/, ""); print; next } NR > 5 { exit }' "$0"; }

PHASE=${SDM_PHASE:-}; SHA=${SDM_PHASE_END_SHA:-}; WORK=${SDM_PHASE_END_WORK:-}
MODE=dry; BASE=''; RUN=''; WHEEL_COMMIT=''; ALLOW_PROPOSED=0; CHECKPOINT=''; DIED=''
LANDING=''; UNTIL=5; HALF=1b; HELDOUT_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --execute) MODE=execute; shift ;;
    --allow-proposed) ALLOW_PROPOSED=1; shift ;;
    --phase | --sha | --work | --base | --run | --wheel-commit | --landing | --until | --half)
      [ $# -ge 2 ] || { echo "phase_end.sh: $1 needs a value" >&2; exit 2; }
      case "$1" in
        --phase) PHASE=$2 ;; --sha) SHA=$2 ;; --work) WORK=$2 ;; --base) BASE=$2 ;;
        --run) RUN=$2 ;; --wheel-commit) WHEEL_COMMIT=$2 ;; --landing) LANDING=$2 ;;
        --until) UNTIL=$2 ;; --half) HALF=$2 ;;
      esac
      shift 2 ;;
    --) shift; HELDOUT_ARGS=("$@"); break ;;
    *) echo "phase_end.sh: unknown argument $1 (try --help)" >&2; exit 2 ;;
  esac
done
[[ "$PHASE" =~ ^[0-9]+$ ]] || { echo "phase_end.sh: give --phase N (try --help)" >&2; exit 2; }
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "phase_end.sh: give --sha with the full 40-character SHA" >&2; exit 2; }
[[ "$UNTIL" =~ ^[0-5]$ ]] || { echo "phase_end.sh: --until takes 0 to 5" >&2; exit 2; }

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${SDM_REPO:-$(git -C "$HERE" rev-parse --show-toplevel)}
GH_REPO=${SDM_GH_REPO:-NVIDIA/structured-data-models-client}
PHASE_BRANCH=rust-core-phase-$PHASE
INVENTORY=${SDM_INVENTORY:-docs/design/rust-core/phase-0/phase-1b-inventory.yaml}
OVERLAY=(contract/runners/run_python.py rust/crates/sdm-testkit/src/case.rs rust/crates/sdm-testkit/src/observation.rs)
TOOLS_BIN=${SDM_TOOLS_BIN:-}
BUNDLE=${SDM_HELDOUT_BUNDLE:-${HELDOUT_BUNDLE:-}}
HRUN=${SDM_HELDOUT_RUN_DIR:-${HELDOUT_OUT:-}}
LIVE_CMD=${SDM_LIVE_CHECK_CMD:-}

# The cells: "NAME=PYTHON ...", the first the reference cell.
CELL_NAMES=(); CELL_PYS=()
for pair in ${SDM_CELLS:-}; do
  case "$pair" in
    ?*=?*) CELL_NAMES+=("${pair%%=*}"); CELL_PYS+=("${pair#*=}") ;;
    *) echo "phase_end.sh: SDM_CELLS entry '$pair' is not NAME=PYTHON" >&2; exit 2 ;;
  esac
done
CELL_REF_PY=${CELL_PYS[0]:-}

ts() { date +%H:%M:%S; }
say() { printf '[%s] %s\n' "$(ts)" "$*"; }
die() { DIED=1; printf '\n[%s] !! STOPPED%s: %s\n' "$(ts)" "${CURRENT:+ at step $CURRENT}" "$*"; [ -n "${CURRENT_LOG:-}" ] && echo "   log: $CURRENT_LOG"; exit 1; }
alarm() { local s=$1; shift; perl -e 'alarm shift; exec @ARGV' "$s" "$@" </dev/null; }
# run DESCRIPTION -- COMMAND...: printed always, executed with --execute.
run() {
  local what=$1; shift; [ "$1" = -- ] && shift
  echo "   - $what"; printf '     $'; printf ' %q' "$@"; echo
  if [ "$MODE" = execute ]; then "$@" || die "$what failed (exit $?)"; fi
}
trap 'rc=$?; if [ $rc -ne 0 ] && [ -z "$DIED" ]; then printf "\n[%s] !! STOPPED%s: a command failed (exit %s)\n" "$(ts)" "${CURRENT:+ at step $CURRENT}" "$rc"; fi; exit $rc' EXIT

S7=${SHA:0:7}
TMPBASE=${TMPDIR:-/tmp}
WORK=${WORK:-${TMPBASE%/}/sdm-phase-$PHASE-end-$S7}
WORK=${WORK%/}
LOGS=$WORK/logs; OBS=$WORK/obs; EVW=$WORK/ev-wt; BLD=$WORK/build-wt; ART=$WORK/artifacts
LIVE=$WORK/live; PACK=$WORK/pack/evidence; GTMP=$WORK/tmp; STAMPS=$WORK/stamps
HELDOUT_SH=$EVW/scripts/rust_core/heldout_run.sh
# The stamps, worktrees and artifacts under $WORK are this SHA's only.
WORK_CLASH=''
if [ -e "$WORK/SHA" ]; then
  [ "$(cat "$WORK/SHA")" = "$SHA" ] || WORK_CLASH="$WORK is the work dir of $(cat "$WORK/SHA"), not $SHA: give a new --work"
elif [ -n "$(ls -A "$STAMPS" 2>/dev/null)" ]; then
  WORK_CLASH="$WORK holds step stamps but no SHA record: give a new --work"
fi
HELDOUT_ARGS=(--candidate "$EVW" --python "${CELL_REF_PY:-<reference cell python>}"
  ${BUNDLE:+--bundle "$BUNDLE"} ${HRUN:+--out "$HRUN"} "${HELDOUT_ARGS[@]+"${HELDOUT_ARGS[@]}"}")
if [ "$MODE" = execute ]; then
  [ -z "$WORK_CLASH" ] || die "$WORK_CLASH"
  mkdir -p "$LOGS" "$OBS" "$ART" "$LIVE/log" "$LIVE/public" "$GTMP" "$STAMPS"
  [ -e "$WORK/SHA" ] || echo "$SHA" >"$WORK/SHA"
  exec > >(tee -a "$WORK/phase_end.log") 2>&1
fi
declare -a TIMES=()

begin() {  # begin N NAME
  CURRENT="$1 $2"; CURRENT_LOG=$LOGS/$1-$2.log; STEP_T0=$(date +%s)
  echo; say "== step $1: $2 ($MODE)"
  if [ -e "$STAMPS/$1-$2.done" ]; then say "   done earlier ($(cat "$STAMPS/$1-$2.done")); skipped"; return 1; fi
  if [ "$MODE" = execute ]; then exec 3>&1 4>&2; exec > >(tee -a "$CURRENT_LOG") 2>&1; fi
  return 0
}
end_step() {  # end_step stamp|nostamp
  local took=$(( $(date +%s) - STEP_T0 ))
  TIMES+=("$CURRENT: $((took / 60))m$((took % 60))s${2:+ $2}")
  if [ "$MODE" = execute ]; then
    [ "$1" = stamp ] && date -u +%FT%TZ >"$STAMPS/${CURRENT/ /-}.done"
    say "   ${2:-ok} ($((took / 60))m$((took % 60))s)"
    exec 1>&3 2>&4 3>&- 4>&-
  fi
  CURRENT=''; CURRENT_LOG=''
}
finish() { end_step stamp; }
# A one-shot's marker: made with noclobber, so two runs can't both start it.
started() { ( set -C; date -u +%FT%TZ >"$STAMPS/$1.started" ) 2>/dev/null || die "$1 started before (marker exists): a person decides, never a rerun"; }
half_started() { [ -e "$STAMPS/$1.started" ] && [ ! -e "$STAMPS/$1.done" ]; }
# A worktree made by an earlier step (or run) must still be at SHA.
at_sha() {
  local head; head=$(git -C "$1" rev-parse HEAD 2>/dev/null) || head='(none)'
  [ "$head" = "$SHA" ] || { echo "     !! $1 is at ${head:0:12}, not SHA ${SHA:0:12}"; return 1; }
  echo "     at ${SHA:0:12}"
}

# ===========================================================================
CURRENT='0 preflight'; STEP_T0=$(date +%s)
echo; say "== step 0: preflight ($MODE) for $SHA on $PHASE_BRANCH"
PROBLEMS=()
problem() { PROBLEMS+=("$*"); echo "   !! $*"; }
okay() { echo "   ok  $*"; }
[ -z "$WORK_CLASH" ] || problem "$WORK_CLASH"

alarm 120 git -C "$REPO" fetch -q origin || problem "git fetch origin failed"
LANDING=${LANDING:-$SHA}
for c in "$SHA" "$LANDING"; do
  git -C "$REPO" cat-file -e "$c^{commit}" 2>/dev/null || problem "$c is not a commit here"
done
REF_SHA=${SDM_REF_SHA:-}
if git -C "$REPO" cat-file -e "$SHA^{commit}" 2>/dev/null && git -C "$REPO" cat-file -e "$LANDING^{commit}" 2>/dev/null; then
  TIP=$(git -C "$REPO" rev-parse -q --verify "origin/$PHASE_BRANCH^{commit}" 2>/dev/null || true)
  tree() { git -C "$REPO" rev-parse "$1^{tree}" 2>/dev/null; }
  is_landing() {  # an empty "Land ..." commit
    [[ "$(git -C "$REPO" log -1 --format=%s "$1")" == Land\ * ]] && [ "$(tree "$1")" = "$(tree "$1^1")" ]
  }
  trailer() { git -C "$REPO" log -1 --format="%(trailers:key=$2,valueonly)" "$1" | head -1 | tr -d ' '; }
  if [ -z "$TIP" ]; then problem "no origin/$PHASE_BRANCH"
  elif git -C "$REPO" merge-base --is-ancestor "$LANDING" "origin/$PHASE_BRANCH"; then
    [ "$LANDING" = "$TIP" ] && okay "${LANDING:0:12} is the phase tip" || echo "   note: ${LANDING:0:12} is on the phase branch; the tip is now ${TIP:0:12}"
  else problem "${LANDING:0:12} is not on origin/$PHASE_BRANCH"; fi
  if [ "$LANDING" != "$SHA" ]; then
    [ "$(tree "$SHA")" = "$(tree "$LANDING")" ] && okay "SHA's tree is ${LANDING:0:12}'s (the evidence commit adds no change)" \
      || problem "SHA's tree differs from --landing ${LANDING:0:12}'s"
  fi
  if is_landing "$SHA"; then
    okay "SHA is an empty landing commit: $(git -C "$REPO" log -1 --format=%s "$SHA")"
    [ -n "$BASE" ] || BASE=$(trailer "$SHA" Base-SHA)
    CHECKPOINT=$(trailer "$SHA" Base-SHA)
  else
    [ -n "$BASE" ] || problem "SHA is not a landing commit: give --base (the phase tip the package started from)"
    CHECKPOINT=${BASE:+$(git -C "$REPO" merge-base "$SHA" "$BASE")}
  fi
  # An empty landing has no CI run of its own; its candidate's (or its
  # Moved-from commit's) run built the shipped wheel from the same tree.
  if is_landing "$LANDING"; then
    MOVED=$(trailer "$LANDING" Moved-from)
    [ -n "$WHEEL_COMMIT" ] || WHEEL_COMMIT=${MOVED:-$(git -C "$REPO" rev-parse "$LANDING^1")}
    if [ -z "$RUN" ]; then
      if [ -n "$MOVED" ]; then RUN=$(trailer "$LANDING" Moved-from-run)
      else RUN=$(git -C "$REPO" log -1 --format=%B "$LANDING" | tr '\n' ' ' | grep -oE 'Candidate run [0-9]+ green at +[0-9a-f]{40}' | head -1 | awk '{print $3}' || true); fi
    fi
  else
    WHEEL_COMMIT=${WHEEL_COMMIT:-$LANDING}
  fi
  okay "base ${BASE:0:12}, checkpoint ${CHECKPOINT:0:12}, landing ${LANDING:0:12}, CI wheel from ${WHEEL_COMMIT:0:12}"
  [ "$(tree "$WHEEL_COMMIT")" = "$(tree "$SHA")" ] || problem "the wheel commit's tree differs from SHA's"
  if [ -z "$RUN" ]; then
    RUN=$(alarm 60 gh run list -R "$GH_REPO" --commit "$WHEEL_COMMIT" --workflow ci.yml --json databaseId,conclusion \
      --jq '[.[] | select(.conclusion == "success")][0].databaseId' 2>/dev/null || true)
  fi
  if [ -n "$RUN" ]; then
    ARTS=$(alarm 60 gh api "repos/$GH_REPO/actions/runs/$RUN/artifacts" --jq '.artifacts[] | select(.expired | not) | .name' 2>/dev/null || true)
    RUN_SHA=$(alarm 60 gh api "repos/$GH_REPO/actions/runs/$RUN" --jq .head_sha 2>/dev/null) || RUN_SHA=''
    [[ "$RUN_SHA" =~ ^[0-9a-f]{40}$ ]] || RUN_SHA='(unreadable)'
    [ "$RUN_SHA" = "$WHEEL_COMMIT" ] || problem "CI run $RUN is on ${RUN_SHA:0:12}, not the wheel commit ${WHEEL_COMMIT:0:12}"
    grep -qx native-wheel <<<"$ARTS" && okay "CI run $RUN has native-wheel" || problem "CI run $RUN has no unexpired native-wheel artifact"
  else problem "no green CI run found for ${WHEEL_COMMIT:0:12}: give --run"; fi

  # The gate's observations are dirty when the checkpoint overlays its runner
  # files (gate.sh 'observe'), and evidence.py refuses dirty observations.
  if [ -n "${CHECKPOINT:-}" ]; then
    if git -C "$REPO" diff --quiet "$CHECKPOINT" "$SHA" -- "${OVERLAY[@]}"; then
      okay "the checkpoint's runner files equal SHA's: the gate's --out observations stay clean"
    else
      problem "SHA changes the gate's overlay files against its checkpoint ${CHECKPOINT:0:12} ($(git -C "$REPO" diff --name-only "$CHECKPOINT" "$SHA" -- "${OVERLAY[@]}" | tr '\n' ' ')): the gate observes from an overlaid (dirty) tree and evidence.py refuses it. Remedy: an EMPTY commit E on SHA, landed, then phase_end.sh --sha E --base $SHA --landing $SHA"
    fi
  fi
  PROPOSED=$(git -C "$REPO" show "$SHA:docs/design/rust-core/register.yaml" | grep -cE '^[[:space:]]*status: proposed' || true)
  if [ "$PROPOSED" -gt 0 ]; then
    if [ "$ALLOW_PROPOSED" = 1 ]; then echo "   note: $PROPOSED register entries still proposed; allowed by --allow-proposed (they will NOT count as approved before the held-out run)"
    else problem "$PROPOSED register entries are still proposed: record approvals first (the held-out run counts only entries approved before it), or pass --allow-proposed"; fi
  else okay "no proposed register entries"; fi
  if [ -z "$REF_SHA" ]; then
    REFS=$(git -C "$REPO" ls-tree --name-only "$SHA" contract/reference/ | grep -E '/[0-9a-f]{40}\.jsonl$' || true)
    [ "$(grep -c . <<<"$REFS")" = 1 ] && REF_SHA=$(basename "$REFS" .jsonl) && okay "reference recording ${REF_SHA:0:12}" \
      || problem "not exactly one contract/reference/<sha>.jsonl at SHA: set SDM_REF_SHA"
  fi
  git -C "$REPO" cat-file -e "$SHA:$INVENTORY" 2>/dev/null && okay "inventory $INVENTORY" || problem "no $INVENTORY at SHA: set SDM_INVENTORY"
  git -C "$REPO" cat-file -e "$SHA:scripts/rust_core/heldout_run.sh" 2>/dev/null && okay "heldout_run.sh in SHA's tree" \
    || problem "no scripts/rust_core/heldout_run.sh at SHA"
fi

[ ${#CELL_PYS[@]} -gt 0 ] || problem "SDM_CELLS is unset: give \"NAME=PYTHON ...\", the reference cell first"
for v in "${CELL_PYS[@]+"${CELL_PYS[@]}"}"; do [ -x "$v" ] || problem "cell python $v is not executable"; done
if [ -z "$TOOLS_BIN" ]; then problem "SDM_TOOLS_BIN is unset (a bin directory with python and the pinned maturin)"
else for v in python maturin; do [ -x "$TOOLS_BIN/$v" ] || problem "no $TOOLS_BIN/$v"; done; fi
if [ -z "$BUNDLE" ]; then problem "SDM_HELDOUT_BUNDLE is unset (the held-out bundle evidence.py digests)"
else test -d "$BUNDLE" && okay "held-out bundle directory present (not opened)" || problem "no held-out bundle directory at the SDM_HELDOUT_BUNDLE path"; fi
[ -z "$HRUN" ] && echo "   note: SDM_HELDOUT_RUN_DIR unset: the pack step leaves copying the held-out report to you"
if half_started 3-heldout; then problem "the held-out run started and did not finish: a person decides, never a rerun"; fi
if half_started 4-live; then problem "the live check started and did not finish: a person decides, never a rerun"; fi
[ -n "$LIVE_CMD" ] && okay "live install check set (SDM_LIVE_CHECK_CMD; not echoed)" || echo "   note: SDM_LIVE_CHECK_CMD unset: step 4 is skipped and recorded as not run"
command -v tokei >/dev/null && okay "tokei $(tokei --version | awk '{print $2}')" || problem "tokei missing"
command -v cargo >/dev/null || problem "cargo missing"
command -v uv >/dev/null || problem "uv missing"
alarm 30 gh auth status >/dev/null 2>&1 && okay "gh auth" || problem "gh auth: expired or missing"
if [ ${#PROBLEMS[@]} -gt 0 ]; then
  echo; echo "   preflight: ${#PROBLEMS[@]} problem(s)"
  [ "$MODE" = execute ] && die "preflight failed: ${PROBLEMS[0]}"
else okay "preflight clean"; fi
TIMES+=("0 preflight: $(( $(date +%s) - STEP_T0 ))s"); CURRENT=''

GATE_JSON="$(git -C "$REPO" rev-parse --path-format=absolute --git-common-dir)/sdm-gate-evidence/gate/$SHA.json"
EVIDENCE_ARGS=(--gate-out "$OBS" --gate-json "$GATE_JSON" --reference "$EVW/contract/reference/$REF_SHA.jsonl"
  --cases "$EVW/contract/cases" --inventory "$EVW/$INVENTORY" --register "$EVW/docs/design/rust-core/register.yaml")

# ===========================================================================
if [ "$UNTIL" -ge 1 ] && begin 1 build; then
  run "a clean build worktree at SHA" -- bash -c "[ -d '$BLD' ] || git -C '$REPO' worktree add --quiet --detach '$BLD' '$SHA'"
  run "the build worktree is at SHA" -- at_sha "$BLD"
  run "the local client wheel (native_wheel.py, not --manylinux)" -- \
    bash -c "ls '$ART'/client/*.whl >/dev/null 2>&1 || { cd '$BLD' && PATH='$TOOLS_BIN':\$PATH '$TOOLS_BIN/python' -I scripts/rust_core/native_wheel.py --out '$ART/client'; }"
  run "the local wheel's identity: commit = SHA, clean tree" -- python3 -I - "$ART/client" "$SHA" <<'PY'
import glob, json, sys
w = glob.glob(sys.argv[1] + '/structured_data_models_client-*.whl'); assert len(w) == 1, w
i = json.load(open(w[0] + '.json')); assert i['commit'] == sys.argv[2] and i['tree_clean'] is True and not i['features'], i
print('     client', w[0].rsplit('/', 1)[1], i['sha256'][:12], 'native', i['native_sha256'][:12])
PY
  run "CI's native wheel from run ${RUN:-RUN}" -- bash -c "ls '$ART'/ci-native/*.whl >/dev/null 2>&1 ||
    perl -e 'alarm 600; exec @ARGV' gh run download '${RUN:-RUN}' -R '$GH_REPO' -n native-wheel -D '$ART/ci-native' </dev/null"
  run "CI's wheel identity: commit = the wheel commit, clean tree" -- python3 -I - "$ART/ci-native" "$WHEEL_COMMIT" <<'PY'
import glob, json, sys
w = glob.glob(sys.argv[1] + '/structured_data_models_client-*.whl'); assert len(w) == 1, w
i = json.load(open(w[0] + '.json')); assert i['commit'] == sys.argv[2] and i['tree_clean'] is True, i
print('     CI wheel', w[0].rsplit('/', 1)[1], i['sha256'][:12])
PY
  for k in "${!CELL_NAMES[@]}"; do
    cell=${CELL_NAMES[$k]}; py=${CELL_PYS[$k]}; sib=$ART/sib-$cell
    run "SHA's other packages built for $cell" -- bash -c '
      set -e; bld=$1 py=$2 sib=$3
      ls "$sib"/*.whl >/dev/null 2>&1 && exit 0
      for d in "$bld"/packages/structured-data-models-*/; do
        case "$d" in */structured-data-models-client/) continue ;; esac
        uv build -q --wheel --python "$py" --out-dir "$sib" "$d"
      done
      ls "$sib"' _ "$BLD" "$py" "$sib"
    run "install SHA's packages into $cell (no deps: the cell's pins stay)" -- \
      bash -c "uv pip install -q -p '$py' --no-deps --reinstall '$sib'/*.whl '$ART'/client/structured_data_models_client-*.whl && uv pip check -q -p '$py'"
    run "check $cell is that cell and holds this build's native module" -- "$py" -I - "$BLD/contract/runners" "$ART/client" "$cell" <<'PY'
import glob, hashlib, importlib.metadata as m, json, pathlib, platform, sys
sys.path.insert(0, sys.argv[1]); import run_python as r
versions = {'python': platform.python_version(), **{k: m.version(k) for k in ('pandas', 'pyarrow', 'numpy', 'duckdb')}}
cell = r._cell(versions); assert cell == sys.argv[3], (cell, sys.argv[3], versions)
import sdm_client._native as n
ident = json.load(open(glob.glob(sys.argv[2] + '/*.whl.json')[0]))
got = hashlib.sha256(pathlib.Path(n.__file__).read_bytes()).hexdigest()
assert got == ident['native_sha256'], ('installed _native is not this build', got[:12])
print('     cell', cell, 'contract', n.CONTRACT_VERSION, 'native', got[:12])
PY
  done
  finish
fi

# ===========================================================================
if [ "$UNTIL" -ge 2 ] && begin 2 evidence; then
  run "the evidence worktree, detached at SHA (stays clean)" -- bash -c "[ -d '$EVW' ] || git -C '$REPO' worktree add --quiet --detach '$EVW' '$SHA'"
  run "the evidence worktree is at SHA" -- at_sha "$EVW"
  run "gate --locked $HALF in ${CELL_NAMES[0]:-the reference cell}, observations kept (the gate's temp dirs under \$WORK/tmp)" -- \
    bash -c "set -o pipefail; cd '$EVW' && TMPDIR='$GTMP' PYTHON='$CELL_REF_PY' scripts/rust_core/gate.sh --base '$BASE' \
      --run python-core,python-old,rust-host --locked '$HALF' --out '$OBS' 2>&1 | tee '$OBS/gate-$HALF.log'"
  run "the gate's evidence file says $HALF passed for SHA" -- python3 -I -c '
import json, sys; j = json.load(open(sys.argv[1])); h = j["halves"][sys.argv[3]]
assert j["sha"] == sys.argv[2], j["sha"]; assert h["passed"] is True, h["checks"]
print("     passed; cell", h["cell"])' "$GATE_JSON" "$SHA" "$HALF"
  run "compare.log (the gate log; evidence.py takes the compare step's block) and ratchet.log" -- bash -c "
    cp '$OBS/gate-$HALF.log' '$OBS/compare.log' &&
    awk '/^ratchet +(ok|FAIL)/ {on=1; next} on && /^    \\| / {sub(/^    \\| /, \"\"); print; next} on {exit}' '$OBS/gate-$HALF.log' >'$OBS/ratchet.log'"
  for k in "${!CELL_NAMES[@]}"; do
    [ "$k" -eq 0 ] && continue
    cell=${CELL_NAMES[$k]}; py=${CELL_PYS[$k]}
    for mode in python-core python-old; do
      run "$mode in $cell" -- "$py" -I "$EVW/contract/runners/run_python.py" --mode "$mode" \
        --out "$OBS/$mode.$cell.jsonl" "$EVW/contract/cases"
    done
  done
  run "every observation of SHA, clean, in its cell" -- python3 -I - "$OBS" "$SHA" <<'PY'
import collections, json, pathlib, sys
seen = collections.Counter()
for f in sorted(pathlib.Path(sys.argv[1]).glob('*.jsonl')):
    for line in f.read_text().splitlines():
        o = json.loads(line); i = o['identity']
        assert i['commit']['sha'] == sys.argv[2] and i['commit']['clean'] is True, (f.name, o['case'], i['commit'])
        seen[(i['runner'], i.get('cell'))] += 1
for k, v in sorted(seen.items(), key=str): print('     ', k, v)
PY
  run "boundary: cargo tree for the pure-Rust host; pyo3 must not appear" -- bash -c "
    cd '$EVW/rust' && cargo tree --locked -e normal -p sdm-testkit >'$OBS/cargo-tree-sdm-testkit.txt' &&
    if cargo tree --locked -e normal -p sdm-testkit -i pyo3 >'$OBS/cargo-tree-pyo3.txt' 2>&1; then echo '     !! pyo3 is in the sdm-testkit tree'; exit 1; fi;
    echo '     pyo3 absent from sdm-testkit (cargo tree -i finds no package)'"
  run "size: tokei per crate and for the Python it replaces" -- bash -c "
    cd '$EVW' && tokei --output json rust/crates >'$OBS/tokei-rust-crates.json' &&
    tokei rust/crates rust/bindings >'$OBS/tokei-rust.txt' &&
    tokei packages/structured-data-models-client/src >'$OBS/tokei-client-python.txt' && tail -3 '$OBS/tokei-rust.txt'"
  run "evidence.py WITHOUT the held-out bundle, into scratch: refused inputs show before the one shot" -- \
    "$CELL_REF_PY" -I "$EVW/scripts/rust_core/evidence.py" "${EVIDENCE_ARGS[@]}" --out "$WORK/pack-check"
  finish
fi

# ===========================================================================
if [ "$UNTIL" -ge 3 ] && begin 3 heldout; then
  run "the evidence worktree is at SHA" -- at_sha "$EVW"
  run "held-out preconditions (heldout_run.sh's own dry run)" -- "$HELDOUT_SH" "${HELDOUT_ARGS[@]}"
  [ "$MODE" = execute ] && started 3-heldout
  run "THE held-out run, once" -- "$HELDOUT_SH" "${HELDOUT_ARGS[@]}" --execute
  [ -n "$HRUN" ] && run "the run left its report" -- test -s "$HRUN/report.md"
  finish
fi

# ===========================================================================
if [ "$UNTIL" -ge 4 ] && begin 4 live; then
  if [ -z "$LIVE_CMD" ]; then
    echo "   SDM_LIVE_CHECK_CMD unset: skipped, not stamped (the pack records it as not run)"
    end_step nostamp skipped
  else
    [ "$MODE" = execute ] && started 4-live
    echo "   - the live install check, once; output kept out of the terminal and the pack, in \$WORK/live/log"
    echo "     \$ bash -c \"\$SDM_LIVE_CHECK_CMD\"   (command not echoed)"
    if [ "$MODE" = execute ]; then
      rc=0
      env PHASE_END_PHASE="$PHASE" PHASE_END_SHA="$SHA" PHASE_END_LANDING="$LANDING" PHASE_END_RUN="$RUN" \
        PHASE_END_WHEEL_COMMIT="$WHEEL_COMMIT" PHASE_END_CI_WHEELS="$ART/ci-native" PHASE_END_PUBLIC="$LIVE/public" \
        bash -c "$LIVE_CMD" </dev/null >"$LIVE/log/live.log" 2>&1 || rc=$?
      echo "     exit $rc; public files: $(ls "$LIVE/public" | tr '\n' ' ')"
      [ "$rc" -eq 0 ] || die "the live check did not pass (exit $rc): a person decides before any second run"
    fi
    finish
  fi
fi

# ===========================================================================
if [ "$UNTIL" -ge 5 ] && begin 5 pack; then
  run "the evidence worktree is at SHA" -- at_sha "$EVW"
  run "evidence.py: cases, seams, comparator, held-out digests" -- \
    "$CELL_REF_PY" -I "$EVW/scripts/rust_core/evidence.py" "${EVIDENCE_ARGS[@]}" --heldout "$BUNDLE" --out "$PACK"
  if [ -n "$HRUN" ]; then
    run "held-out: report, compare, identity, register snapshot digest" -- bash -c "
      mkdir -p '$PACK/heldout' && cp '$HRUN/report.md' '$HRUN/compare.txt' '$HRUN/identity.txt' '$PACK/heldout/' &&
      shasum -a 256 '$HRUN/register-at-run-start.yaml' | awk '{print \$1\"  register-at-run-start.yaml\"}' >'$PACK/heldout/register-at-run-start.sha256'"
  else
    echo "   - held-out: SDM_HELDOUT_RUN_DIR unset; copy the run's report, compare and identity into $PACK/heldout/ by hand"
  fi
  run "live install check: its public files, or a note that it did not run" -- bash -c "
    mkdir -p '$PACK/live' && if [ -e '$STAMPS/4-live.done' ]; then cp -R '$LIVE/public/.' '$PACK/live/';
    else echo 'The live install check did not run for this SHA (SDM_LIVE_CHECK_CMD unset).' >'$PACK/live/NOT-RUN.md'; fi"
  run "boundary and size" -- bash -c "mkdir -p '$PACK/boundary' '$PACK/size' &&
    cp '$OBS/cargo-tree-sdm-testkit.txt' '$OBS/cargo-tree-pyo3.txt' '$PACK/boundary/' &&
    cp '$OBS'/tokei-* '$PACK/size/'"
  run "identity" -- python3 -I - "$PACK/identity.md" "$SHA" "$BASE" "$CHECKPOINT" "$REF_SHA" "$WHEEL_COMMIT" "${RUN:-}" \
      "$ART/client" "$ART/ci-native" "$GATE_JSON" "$OBS" <<'PY'
import glob, hashlib, json, pathlib, sys
out, sha, base, ckpt, ref, wcommit, run, client, ci, gate, obs = sys.argv[1:]
def sha256(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
local = glob.glob(client + '/*.whl')[0]; ciw = glob.glob(ci + '/*.whl')[0]
first = json.loads(next(pathlib.Path(obs).glob('python-core.jsonl')).read_text().splitlines()[0])['identity']
rows = [('phase commit', sha), ('base (Base-SHA)', base), ('checkpoint', ckpt), ('reference commit', ref),
        ('CI run (native-wheel)', f'{run} on {wcommit} (same tree)'),
        ('CI wheel', f'{pathlib.Path(ciw).name} sha256 {sha256(ciw)}'),
        ('local build', f'{pathlib.Path(local).name} sha256 {sha256(local)}'),
        ('gate evidence', f'{pathlib.Path(gate).name} sha256 {sha256(gate)} (outside the tree)'),
        ('ref-cell versions', json.dumps(first.get('versions'), sort_keys=True)),
        ('platform', first.get('platform'))]
pathlib.Path(out).write_text('# Identity\n\n| item | value |\n|---|---|\n' + ''.join(f'| {k} | `{v}` |\n' for k, v in rows))
print('     identity.md written')
PY
  echo "   then by hand (implementer; reviewers check): fill the held-out report's 'approved before the run'"
  echo "   column from register-at-run-start.yaml; write evidence/README.md (every item with its command);"
  echo "   copy $PACK into docs/design/rust-core/phase-$PHASE/evidence/ in the phase-end package's lane."
  finish
fi

echo; say "== summary ($MODE)"
for t in "${TIMES[@]+"${TIMES[@]}"}"; do echo "   $t"; done
[ "$MODE" = execute ] && echo "   work dir: $WORK" || echo "   dry run: nothing built, run or written"
DIED=1
if [ "$MODE" = dry ] && [ ${#PROBLEMS[@]} -gt 0 ]; then
  echo "   preflight problems (fix before --execute):"; for p in "${PROBLEMS[@]}"; do echo "     - $p"; done
  exit 1
fi
