#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Remove stale build/review scratch left by Rust-core agents (owner-approved 2026-10-12, widened 2026-10-13).
# A directory is stale when NOTHING inside it changed for $AGE minutes (default 180) and it is not any
# running process's cwd. Scope:
#   /tmp top-level dirs matching the agents' naming patterns (never git worktrees: those use worktree prune);
#   the top-level dirs of $SDM_SCRATCH, the agents' scratch directory (skipped when it is not set);
#   Cargo build output (rebuilt on demand): <repo>*/{,rust/}target and ~/.cache/sdm-*target*.
#   Build output is removed when stale, or when over CAP_GIB (default 15) with no write for 30 min and no
#   process using its worktree (owner-approved 2026-10-13, step 3).
# `clean_tmp.sh --after-land <worktree>` removes that worktree's build output after a landing unless a
# process is using it (step 4; auto_land.py calls it with --apply only when SDM_CLEAN_AFTER_LAND=1).
#
# Dry run by default: it prints "would remove <dir>" and removes nothing. --apply removes.
# Never removes: a path git tracks, a git worktree, /, $HOME, ~/code, the repo, the state dir or
# anything inside it, or a directory holding one of those: a .git within 4 levels below it, or
# any worktree `git -C <repo> worktree list` names under it.
#
#   clean_tmp.sh [--apply | --dry-run] [--after-land <worktree>]
#
# Environment:
#   SDM_SCRATCH     the agents' scratch directory to sweep (optional; never ~/code or a git worktree)
#   SDM_REPO        the repo whose build output and worktrees it prunes (default: the git toplevel
#                   of this script)
#   SDM_ORCH_STATE  state directory, never touched (default ~/.local/share/sdm-orchestration)
#   AGE, CAP_GIB    as above
set -u
APPLY=0; AFTER=""
while [ $# -gt 0 ]; do
  case "$1" in
    --apply) APPLY=1;;
    --dry-run) APPLY=0;;
    --after-land) AFTER=${2:?--after-land needs a worktree}; shift;;
    -h|--help) sed -n '6,29p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) echo "clean_tmp.sh: unknown argument $1 (see --help)" >&2; exit 2;;
  esac
  shift
done
STATE=${SDM_ORCH_STATE:-$HOME/.local/share/sdm-orchestration}
REPO=${SDM_REPO:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null)}
real() { (cd "$1" 2>/dev/null && pwd -P) || printf '%s\n' "${1%/}"; }  # the physical path, when it exists
[ -n "$REPO" ] && REPO=$(real "$REPO")
uses() { # $1 = path; true if another process (not this script or its pipeline) names it in its args
  ps -axo args= 2>/dev/null | W="$1" awk 'index($0, ENVIRON["W"]) && !/clean_tmp\.sh/ && !/ENVIRON/ {f=1} END {exit !f}'
}
special() { # $1 = path; true (printing why) if it is, holds or lies inside a protected location
  local p=${1%/} rp x rx
  [ -n "$p" ] || { echo "refuse empty path"; return 0; }
  rp=$(real "$p")
  [ "$rp" = / ] && { echo "refuse $p (the root)"; return 0; }
  for x in "$HOME" "$HOME/code" "$STATE" "$REPO"; do
    [ -n "$x" ] || continue
    rx=$(real "$x")
    case "$rx/" in "$rp"/*) echo "refuse $p (is or holds $x)"; return 0;; esac
  done
  case "$rp/" in "$(real "$STATE")"/*) echo "refuse $p (inside the state dir)"; return 0;; esac
  return 1
}
protected() { # $1 = path; true (printing why) if it must never be removed
  local p=${1%/} rp g w
  special "$p" && return 0
  rp=$(real "$p")
  [ -e "$p/.git" ] && { echo "refuse $p (a git worktree)"; return 0; }
  # A worktree or clone anywhere below it (a reviewer's throwaway worktree in its scratch dir).
  g=$(find "$p" -maxdepth 4 -name .git -print -quit 2>/dev/null)
  [ -n "$g" ] && { echo "refuse $p (holds a git worktree or clone: ${g%/.git})"; return 0; }
  if [ -n "$REPO" ]; then
    while IFS= read -r w; do
      [ -n "$w" ] || continue
      case "$(real "$w")/" in "$rp"/*) echo "refuse $p (holds the worktree $w)"; return 0;; esac
      case "${w%/}/" in "$rp"/*|"${p}"/*) echo "refuse $p (holds the worktree $w)"; return 0;; esac
    done <<WT
$(git -C "$REPO" worktree list --porcelain 2>/dev/null | sed -n 's/^worktree //p')
WT
  fi
  if git -C "$(dirname "$p")" rev-parse --is-inside-work-tree >/dev/null 2>&1 &&
     [ -n "$(git -C "$(dirname "$p")" ls-files -- "$(basename "$p")" 2>/dev/null | head -1)" ]; then
    echo "refuse $p (holds tracked files)"; return 0
  fi
  return 1
}
gone() {
  protected "$1" && return 0
  if [ "$APPLY" = 1 ]; then
    echo "remove $1 ($(du -sh "$1" 2>/dev/null | cut -f1))"; rm -rf -- "$1"
  else
    echo "would remove $1 ($(du -sh "$1" 2>/dev/null | cut -f1))"
  fi
}
if [ -n "$AFTER" ]; then
  w=$AFTER; busy=$(lsof -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')
  uses "$w" && { echo "skip busy $w"; exit 0; }
  echo "$busy" | grep -q "^$w" && { echo "skip busy $w"; exit 0; }
  for t in "$w/rust/target" "$w/target"; do
    [ -d "$t" ] && gone "$t"
  done
  exit 0
fi
AGE=${AGE:-180}
busy=$(lsof -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')
stale() { # $1 = path; stale if no file inside newer than AGE minutes and not a cwd prefix
  local p=$1
  echo "$busy" | grep -q "^$p" && { echo "skip busy $p"; return 1; }
  [ -z "$(find "$p" -mmin -"$AGE" -print -quit 2>/dev/null)" ]
}
CAP_GIB=${CAP_GIB:-15}
build_busy() { # $1 = worktree root; busy if any process has its cwd inside it or names it in its args
  local w=$1
  echo "$busy" | grep -q "^$w" && return 0
  uses "$w"
}
gib() { du -sk "$1" 2>/dev/null | awk '{print int($1/1048576)}'; }
# /tmp by its physical path (on macOS a symlink is resolved), the form lsof reports cwds in.
TMPROOT=$(real /tmp)
cd "$TMPROOT" || exit 1
for d in sdm-* rv* rr-* codex-* sub-* t1[0-9]* t12* sdm-review-* tailprep-* *-target* *-venv* *-wheel* *-probe* *-obs* *-diff*; do
  [ -d "$TMPROOT/$d" ] || continue
  case "$d" in claude-*) continue;; esac
  git -C "$TMPROOT/$d" rev-parse --is-inside-work-tree >/dev/null 2>&1 && continue
  stale "$TMPROOT/$d" && gone "$TMPROOT/$d"
done
if [ -z "${SDM_SCRATCH:-}" ]; then
  echo "skip scratch: SDM_SCRATCH is not set"
else
  SCRATCH=$(real "$SDM_SCRATCH")
  if [ ! -d "$SCRATCH" ]; then
    echo "skip scratch: $SDM_SCRATCH is not a directory"
  elif [ "$SCRATCH" = "$TMPROOT" ]; then
    echo "skip scratch: $SDM_SCRATCH is /tmp itself"
  elif why=$(special "$SCRATCH"); then
    echo "skip scratch: $why"
  elif git -C "$SCRATCH" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "skip scratch: $SDM_SCRATCH is inside a git worktree"
  else
    case "$SCRATCH/" in
      "$(real "$HOME/code")"/*) echo "skip scratch: $SDM_SCRATCH is under ~/code";;
      *)
        for d in "$SCRATCH"/*/; do
          [ -d "$d" ] || continue
          d=${d%/}
          git -C "$d" rev-parse --is-inside-work-tree >/dev/null 2>&1 && continue
          stale "$d" && gone "$d"
        done;;
    esac
  fi
fi
if [ -z "$REPO" ]; then
  echo "skip build output: no repo (set SDM_REPO)"
else
  for t in "$REPO"*/target "$REPO"*/rust/target "$HOME"/.cache/sdm-*target*; do
    [ -d "$t" ] || continue
    case "$t" in */rust/target) w=${t%/rust/target};; */.cache/*) w=$t;; *) w=${t%/target};; esac
    if stale "$t"; then gone "$t"; continue; fi
    [ "$(gib "$t")" -ge "$CAP_GIB" ] || continue
    [ -z "$(find "$t" -mmin -30 -print -quit 2>/dev/null)" ] || { echo "skip active $t ($(gib "$t") GiB)"; continue; }
    build_busy "$w" && { echo "skip busy $t ($(gib "$t") GiB)"; continue; }
    gone "$t"
  done
  if [ "$APPLY" = 1 ]; then
    git -C "$REPO" worktree prune
  else
    git -C "$REPO" worktree prune --dry-run --verbose
  fi
fi
df -h / | tail -1
