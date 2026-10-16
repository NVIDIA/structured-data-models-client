#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The locked checks on the exact merge SHA, from CI's wheels for that SHA
# (handbook, "Ending a phase"): gate.sh --locked for each HALF named, then one
# check that the single evidence file holds every one of them, passed.
#
#   final_gate.sh WORKTREE PYTHON CI_RUN [--base SHA] [--out DIR] HALF...            # dry run
#   final_gate.sh WORKTREE PYTHON CI_RUN [--base SHA] [--out DIR] --execute HALF...
#
# WORKTREE  a clean worktree detached at the SHA to judge: the final
#           candidate, then again its landing commit (the merge SHA).
# PYTHON    the environment holding THIS SHA's locally built client and its
#           test-ops build (gate.sh's $PYTHON), not a stale lane venv.
# CI_RUN    the GitHub Actions run whose native-wheel and native-wheel-test-ops
#           artifacts are used: a run on the SHA itself, or for an empty
#           landing commit its candidate's run (same tree; CI skips landings).
# HALF      the locked halves to run, for example 1a 1b; each gets the wheels.
# --base    the phase tip recorded at the start, for a SHA that is not a
#           landing commit (a landing's Base-SHA trailer is used otherwise).
# --out     where wheels and observations go (or SDM_FINAL_GATE_OUT; default
#           ${TMPDIR:-/tmp}/sdm-final-gate-<sha7>). Nothing is deleted.
# Environment: SDM_GH_REPO (default NVIDIA/structured-data-models-client).
#
# The evidence file is gate.sh's, <git-common-dir>/sdm-gate-evidence/gate/<sha>.json,
# outside the tree. A half with wheel-smoke needs Docker; registry rate limits
# have failed it before: rerun later, never skip. The dry run only reads.
set -u

usage() { awk 'NR > 5 && /^#/ { sub(/^# ?/, ""); print; next } NR > 5 { exit }' "$0"; }
case "${1:-}" in -h | --help) usage; exit 0 ;; esac
[ $# -ge 3 ] || { echo "final_gate.sh: give WORKTREE PYTHON CI_RUN and at least one HALF (try --help)" >&2; exit 2; }
WT=$1; PY=$2; RUN=$3; shift 3
BASE=''; MODE=dry; OUT=${SDM_FINAL_GATE_OUT:-}; HALVES=()
while [ $# -gt 0 ]; do
  case "$1" in
    -h | --help) usage; exit 0 ;;
    --execute) MODE=execute; shift ;;
    --base | --out)
      [ $# -ge 2 ] || { echo "final_gate.sh: $1 needs a value" >&2; exit 2; }
      if [ "$1" = --base ]; then BASE=$2; else OUT=$2; fi; shift 2 ;;
    -*) echo "final_gate.sh: unknown argument $1 (try --help)" >&2; exit 2 ;;
    *) HALVES+=("$1"); shift ;;
  esac
done
[ ${#HALVES[@]} -gt 0 ] || { echo "final_gate.sh: name at least one HALF, for example 1a 1b" >&2; exit 2; }
[[ "$RUN" =~ ^[0-9]+$ ]] || { echo "final_gate.sh: CI_RUN must be a run id" >&2; exit 2; }
GH_REPO=${SDM_GH_REPO:-NVIDIA/structured-data-models-client}
alarm() { local s=$1; shift; perl -e 'alarm shift; exec @ARGV' "$s" "$@" </dev/null; }
do_() { printf '   $'; printf ' %q' "$@"; echo; [ "$MODE" = execute ] || return 0; "$@"; }
in_wt() { (cd "$WT" && "$@"); }
stop() { echo "!! $*"; exit 1; }

SHA=$(git -C "$WT" rev-parse HEAD) || stop "$WT is not a git worktree"
[ -z "$(git -C "$WT" status --porcelain)" ] || stop "the worktree is not clean"
SUBJECT=$(git -C "$WT" log -1 --format=%s)
if [[ "$SUBJECT" == Land\ * ]]; then
  REC=$(git -C "$WT" log -1 --format='%(trailers:key=Base-SHA,valueonly)' | head -1 | tr -d ' ')
  echo "landing commit: Base-SHA ${REC:0:12} (the ratchet uses it whatever --base says)"; BASE=${BASE:-$REC}
fi
[ -n "$BASE" ] || { echo "!! not a landing commit: give --base"; exit 2; }
TMPBASE=${TMPDIR:-/tmp}
OUT=${OUT:-${TMPBASE%/}/sdm-final-gate-${SHA:0:7}}
OUT=${OUT%/}
EVID="$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir)/sdm-gate-evidence/gate/$SHA.json"

echo "== SHA $SHA ($SUBJECT), halves: ${HALVES[*]} ($MODE)"
echo "== 0. CI run $RUN is on this SHA, or on a commit with its tree (read only)"
RUN_SHA=$(alarm 60 gh api "repos/$GH_REPO/actions/runs/$RUN" --jq .head_sha 2>/dev/null) || RUN_SHA=''
[[ "$RUN_SHA" =~ ^[0-9a-f]{40}$ ]] || RUN_SHA=''
if [ -z "$RUN_SHA" ]; then
  [ "$MODE" = execute ] && stop "could not read CI run $RUN (gh auth, network?)"
  echo "   !! could not read CI run $RUN (gh auth, network?): checked again with --execute"
elif [ "$RUN_SHA" = "$SHA" ]; then echo "   ok  run $RUN is on $SHA"
elif [ "$(git -C "$WT" rev-parse -q --verify "$RUN_SHA^{tree}" 2>/dev/null)" = "$(git -C "$WT" rev-parse "$SHA^{tree}")" ]; then
  echo "   ok  run $RUN is on ${RUN_SHA:0:12}, whose tree is this SHA's"
else
  [ "$MODE" = execute ] && stop "run $RUN is on ${RUN_SHA:0:12}, whose tree is not this SHA's"
  echo "   !! run $RUN is on ${RUN_SHA:0:12}, whose tree is not this SHA's"
fi

echo "== 1. CI's wheels from that run"
do_ mkdir -p "$OUT" || stop "mkdir failed"
for art in native-wheel:native native-wheel-test-ops:test-ops; do
  name=${art%%:*}; dir=$OUT/${art#*:}
  if [ "$MODE" = execute ] && ls "$dir"/*.whl >/dev/null 2>&1; then echo "   kept: $dir (downloaded earlier)"; continue; fi
  do_ alarm 600 gh run download "$RUN" -R "$GH_REPO" -n "$name" -D "$dir" || stop "download of $name failed"
done
W="$OUT/native/<structured_data_models_client-*.whl>"; T="$OUT/test-ops/<structured_data_models_client-*.whl>"
if [ "$MODE" = execute ]; then
  W=$(ls "$OUT"/native/*.whl) && T=$(ls "$OUT"/test-ops/*.whl) || stop "no wheel in the downloads"
  for w in "$W" "$T"; do
    python3 -I - "$w.json" "$(git -C "$WT" rev-parse "$SHA^{tree}")" "$WT" <<'PY' || stop "a wheel is not from this SHA's tree"
import json, subprocess, sys
i = json.load(open(sys.argv[1]))
tree = subprocess.run(['git', '-C', sys.argv[3], 'rev-parse', i['commit'] + '^{tree}'], capture_output=True, text=True).stdout.strip()
assert tree == sys.argv[2] and i['tree_clean'] is True, ('wheel commit', i['commit'][:12], 'tree_clean', i['tree_clean'])
print('   ok ', sys.argv[1].rsplit('/', 1)[1][:-5], 'from', i['commit'][:12], 'sha256', i['sha256'][:12])
PY
  done
fi

n=2; FAILED=()
for half in "${HALVES[@]}"; do
  echo "== $n. gate.sh --locked $half"
  do_ in_wt env PYTHON="$PY" scripts/rust_core/gate.sh --base "$BASE" --run python-core,python-old,rust-host \
    --locked "$half" --wheel "$W" --test-ops-wheel "$T" --wheel-run "$RUN" --out "$OUT/obs-$half" || FAILED+=("$half")
  n=$((n + 1))
done
[ ${#FAILED[@]} -eq 0 ] || echo "   !! gate.sh failed for: ${FAILED[*]}"

echo "== $n. every half passed, in the one evidence file"
do_ python3 -I -c '
import json, sys
j = json.load(open(sys.argv[1])); h = j["halves"]; want = sys.argv[3:]
assert j["sha"] == sys.argv[2], ("evidence is for", j["sha"])
bad = [k for k in want if not h.get(k, {}).get("passed")]
print("   halves", sorted(h), "FAILED: " + " ".join(bad) if bad else "all passed")
sys.exit(1 if bad else 0)' "$EVID" "$SHA" "${HALVES[@]}" || stop "not every half passed (evidence: $EVID)"
echo "   evidence: $EVID (outside the tree; the pack cites it by SHA)"
echo "   also: CI's other runtime cells (the contract-python jobs) green on this SHA, checked by hand"
[ "$MODE" = execute ] || echo "   dry run: nothing downloaded or run"
