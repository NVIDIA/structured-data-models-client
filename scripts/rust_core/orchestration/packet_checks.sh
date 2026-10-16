#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Read-only checks for the owner's go/no-go packet at the end of phase N
# (handbook, "Ending a phase"). Safe to run any time; it changes nothing.
#
#   packet_checks.sh PHASE [REPO] [SINCE]
#
# PHASE  the phase number: reads docs/design/rust-core/phase-PHASE/ and the
#        branch rust-core-phase-PHASE.
# REPO   a clone or worktree (default: this script's git toplevel). Files are
#        read at the head below with git show, not from the working tree.
# SINCE  the last check-in's landing, for section E (default: the phase's
#        start, the merge base of origin/rust-core and the head).
#
# Environment:
#   SDM_PACKET_HEAD  the head to check (default origin/rust-core-phase-PHASE)
#   PYTHON_YAML      a python with PyYAML for section D (default python3; a
#                    plain count is printed when it can't import yaml)
#
# Sections:
#   A  first-parent landings with no passing local gate evidence for that
#      exact SHA (<git-common-dir>/sdm-gate-evidence/gate/<sha>.json)
#   B  migration-plan.md's "Phase N decision X, <status>" markers against the
#      status in phase-N/decisions.md (a stale "pending" is flagged)
#   C  decision rows still "owner to confirm"
#   D  register.yaml statuses; proposed entries listed by id
#   E  approved_by: owner entries added or changed since SINCE
set -u

usage() { awk 'NR > 5 && /^#/ { sub(/^# ?/, ""); print; next } NR > 5 { exit }' "$0"; }
case "${1:-}" in -h | --help | '') usage; [ -n "${1:-}" ]; exit $? ;; esac
PHASE=$1
[[ "$PHASE" =~ ^[0-9]+$ ]] || { echo "packet_checks.sh: PHASE must be a number (try --help)" >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
WT=${2:-$(git -C "$HERE" rev-parse --show-toplevel)}
HEAD_REV=${SDM_PACKET_HEAD:-origin/rust-core-phase-$PHASE}
D=docs/design/rust-core
HEAD_SHA=$(git -C "$WT" rev-parse -q --verify "$HEAD_REV^{commit}") || { echo "packet_checks.sh: no $HEAD_REV in $WT" >&2; exit 2; }
SINCE=${3:-$(git -C "$WT" merge-base origin/rust-core "$HEAD_SHA")}
EVD="$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir)/sdm-gate-evidence/gate"
show() { git -C "$WT" show "$HEAD_SHA:$1" 2>/dev/null; }
echo "phase $PHASE, head $HEAD_REV = ${HEAD_SHA:0:12}, since ${SINCE:0:12}"

echo "== A. first-parent landings with no passing local gate evidence for that exact SHA"
total=0; missing=0
while read -r sha subject; do
  total=$((total + 1))
  f="$EVD/$sha.json"
  if [ -f "$f" ] && python3 -I -c "import json,sys; j=json.load(open(sys.argv[1])); sys.exit(0 if j.get('halves') and all(h.get('passed') for h in j['halves'].values()) else 1)" "$f"; then :; else
    missing=$((missing + 1)); echo "   ${sha:0:9} $subject"; fi
done < <(git -C "$WT" log --first-parent --format='%H %s' "origin/rust-core..$HEAD_SHA" | grep -E '^[0-9a-f]{40} Land ')
echo "   $missing of $total landings lack local evidence (CI gated each; the owner decides whether CI runs suffice)"

echo "== B. migration-plan.md markers for Phase $PHASE decisions vs phase-$PHASE/decisions.md"
DECISIONS=$(show "$D/phase-$PHASE/decisions.md")
PLAN=$(show "$D/migration-plan.md")
if [ -z "$DECISIONS" ] || [ -z "$PLAN" ]; then echo "   !! no $D/phase-$PHASE/decisions.md or migration-plan.md at the head"
else
  python3 -I - "$PHASE" <(printf '%s\n' "$DECISIONS") <(printf '%s\n' "$PLAN") <<'PY'
import re, sys
phase, decisions, plan = sys.argv[1], open(sys.argv[2]).read(), open(sys.argv[3]).read()
ROW = r'[A-Z]+(?:-[A-Z]+)?\d+'
rows = {}
for line in decisions.splitlines():
    cells = [c.strip() for c in line.split('|')]
    if len(cells) > 4 and re.fullmatch(ROW, cells[1] or ''):
        rows[cells[1]] = cells[-2]
STATUS = r'(pending owner confirmation|(?:approved|confirmed) by owner [0-9-]+)'
marks = {}
for line in plan.splitlines():
    # Only the text after "Phase N decision(s)", up to the next "Phase " citation.
    for seg in re.split(r'(?=Phase \d+ decisions? )', line):
        if not re.match(rf'Phase {phase} decisions? ', seg):
            continue
        for row, status in re.findall(rf'\b({ROW})\b[^,;()\n]*?, {STATUS}', seg):
            marks.setdefault(row, set()).add(status)
if not marks:
    print('   no markers cite this phase\'s decisions')
def settled(s): return re.search(r'(approved|confirmed) by owner', s) is not None
for row in sorted(marks):
    status = rows.get(row, '?')
    pend = any(m.startswith('pending') for m in marks[row])
    if row not in rows: flag = 'NOT in decisions.md'
    elif pend and settled(status): flag = 'STALE marker'
    elif not pend and not settled(status): flag = 'plan ahead of decisions.md'
    elif pend: flag = 'owner to confirm'
    else: flag = 'ok'
    print(f'   {row:6} decisions: {status[:60]:60} plan: {sorted(marks[row])} -> {flag}')
PY
fi

echo "== C. decision rows still 'owner to confirm' (not yet confirmed or approved by the owner)"
if [ -n "$DECISIONS" ]; then
  python3 -I - <(printf '%s\n' "$DECISIONS") <<'PY'
import re, sys
ids = [c[1].strip() for c in (l.split('|') for l in open(sys.argv[1]))
       if len(c) > 4 and 'owner to confirm' in c[-2] and not re.search(r'(approved|confirmed) by owner', c[-2])
       and re.fullmatch(r'[A-Z]+(-[A-Z]+)?\d+', c[1].strip())]
print(f'   {len(ids)}:', ' '.join(ids))
PY
fi

echo "== D. register: entries by status (none may stay proposed without an approver)"
REGISTER=$(show "$D/register.yaml")
if ! printf '%s\n' "$REGISTER" | "${PYTHON_YAML:-python3}" -I -c '
import sys, yaml
d = yaml.safe_load(sys.stdin); es = d["entries"] if isinstance(d, dict) else d
by = {}
for e in es: by.setdefault(e.get("status"), []).append(e["id"])
for k, v in sorted(by.items(), key=str): print(f"   {k}: {len(v)}" + (f"  {chr(32).join(v)}" if k == "proposed" else ""))
' 2>/dev/null; then
  echo "   (no yaml module: plain count of status lines)"
  printf '%s\n' "$REGISTER" | grep -E '^[[:space:]]*status:' | sed -E 's/^[[:space:]]*status:[[:space:]]*//' | sort | uniq -c | sed 's/^/   /'
fi

echo "== E. approved_by: owner entries added or changed since ${SINCE:0:12}"
E=$(git -C "$WT" diff "$SINCE" "$HEAD_SHA" -- "$D/register.yaml" | grep -E '^\+.*(- id:|approved_by: owner)')
[ -n "$E" ] && printf '%s\n' "$E" | sed 's/^/   /' || echo "   none"
