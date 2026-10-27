#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Cancel in-progress/queued CI runs on a lane branch whose commit is not the branch head
# and is not a landing run (title "Land ...": T<n> and CI<n> landings alike). Lane/phase CI
# has cancel-in-progress off, so a superseded candidate's run otherwise blocks the new one in
# the same concurrency group.
#
# Lanes are discovered from the remote: every refs/heads/<phase>-lane-<name> whose
# <name> matches lane.py's [a-z0-9]{1,16} (so -tests* branches are skipped).
# Topic branches (2026-10-14): every "branch" named by a land-queue/*.json entry
# (<phase>-<name>, not a lane or archive branch) is scanned the same way.
# Moved candidates (CI3, land.py --moved-from; auto_land.py "moved_from"):
#   - a run whose headSha is any queue entry's moved_from is NEVER cancelled: it is the evidence;
#   - when a branch head is a moved_ok entry's candidate and the newest run on its moved_from
#     is green or not finished, the head's own unfinished runs are cancelled, so they do not
#     hold the branch's concurrency group (auto_land.py reruns it if land.py refuses the move).
# --dry-run (or DRY_RUN=1) prints "would cancel ..." instead of cancelling.
#
# Environment:
#   SDM_PHASE_BRANCH  the phase branch, e.g. rust-core-phase-3 (required; exits 2 when unset)
#   SDM_ORCH_STATE    state directory (default ~/.local/share/sdm-orchestration)
#   SDM_LAND_QUEUE    the queue directory (default $SDM_ORCH_STATE/land-queue)
#   SDM_REPO          the repo whose origin is read (default: the git toplevel of this script)
set -u
case "${1:-}" in -h|--help) sed -n '6,26p' "$0" | sed 's/^# \{0,1\}//'; exit 0;; esac
[ -n "${SDM_PHASE_BRANCH:-}" ] || {
  echo "cancel_superseded.sh: set SDM_PHASE_BRANCH to the phase branch, e.g. rust-core-phase-3 (see --help)" >&2
  exit 2
}
R=NVIDIA/structured-data-models-client
DRY=${DRY_RUN:-0}
[ "${1:-}" = "--dry-run" ] && DRY=1
STATE=${SDM_ORCH_STATE:-$HOME/.local/share/sdm-orchestration}
QDIR=${SDM_LAND_QUEUE:-$STATE/land-queue}
REPO=${SDM_REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel)}
P=$SDM_PHASE_BRANCH

# From the queue: "topic <branch>", "protect <sha>", "moved <branch> <candidate> <moved_from>".
QINFO=$(python3 - "$QDIR" "$P" <<'EOF'
import json, pathlib, re, sys
P = sys.argv[2]
topic = re.compile(re.escape(P) + r'-[A-Za-z0-9](?:[A-Za-z0-9_-]|\.(?!\.))*')
hexsha = re.compile(r'[0-9a-f]{40}')
for f in sorted(pathlib.Path(sys.argv[1]).glob('*.json')):
    try:
        q = json.loads(f.read_text())
    except Exception:
        continue
    if not isinstance(q, dict):
        continue
    b = q.get('branch')
    if b is not None and not (isinstance(b, str) and topic.fullmatch(b)
                              and not b.startswith((P + '-lane-', P + '-archive-'))):
        b = None
    if b is None and isinstance(q.get('lane'), str):
        lb = f'{P}-lane-{q["lane"]}'
        b = lb if topic.fullmatch(lb) else None
    x, c = q.get('moved_from'), q.get('candidate')
    if isinstance(x, str) and hexsha.fullmatch(x):
        print('protect', x)
        if b and q.get('moved_ok') is True and isinstance(c, str) and hexsha.fullmatch(c):
            print('moved', b, c, x)
    if b and q.get('branch') == b:
        print('topic', b)
EOF
)
PROTECT=" $(echo "$QINFO" | awk '$1=="protect"{printf "%s ", $2}')"
TOPICS=$(echo "$QINFO" | awk '$1=="topic"{print "refs/heads/" $2}' | sort -u)

# shellcheck disable=SC2086  # TOPICS is a list of ref patterns
git -C "$REPO" ls-remote origin "refs/heads/$P-lane-*" $TOPICS |
while read -r head ref; do
  B=${ref#refs/heads/}
  case "$B" in
    $P-lane-*)
      lane=${B#$P-lane-}
      [[ "$lane" =~ ^[a-z0-9]{1,16}$ ]] || continue
      label="lane $lane";;
    *)
      echo "$TOPICS" | grep -qxF "$ref" || continue
      label="topic $B";;
  esac
  [ -n "$head" ] || continue
  [ "$DRY" = 1 ] && echo "$label: $B head ${head:0:8}"
  # The head is a moved candidate whose evidence run (on moved_from) is green or unfinished?
  X=$(echo "$QINFO" | awk -v b="$B" -v h="$head" '$1=="moved" && $2==b && $3==h {print $4; exit}')
  head_too=0
  if [ -n "$X" ]; then
    xs=$(gh run list -R $R --branch "$B" --limit 20 --json headSha,status,conclusion \
      -q ".[]|select(.headSha==\"$X\")|\"\(.status) \(.conclusion)\"" </dev/null | head -1)
    case "$xs" in
      "completed success") head_too=1;;
      completed*|"") ;;
      *) head_too=1;;
    esac
  fi
  gh run list -R $R --branch "$B" --limit 10 --json databaseId,headSha,status,displayTitle \
    -q '.[]|select(.status!="completed")|"\(.databaseId) \(.headSha) \(.displayTitle)"' </dev/null |
  while read -r id sha title; do
    case "$PROTECT" in *" $sha "*) continue;; esac  # a moved candidate's evidence
    if [ "$sha" = "$head" ]; then
      [ "$head_too" = 1 ] || continue
      why="${head:0:8} lands on ${X:0:8}'s run"
    else
      case "$title" in Land\ *) continue;; esac  # a landing run (Land T<n>... or Land CI<n>...)
      why="$sha superseded by ${head:0:8}"
    fi
    if [ "$DRY" = 1 ]; then
      echo "would cancel $id on $B ($why)"
    else
      gh run cancel "$id" -R $R >/dev/null 2>&1 && echo "cancelled $id on $B ($why)"
    fi
  done
done
