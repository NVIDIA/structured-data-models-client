#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The checks to run before pushing a Rust core change (decision E43).
#
# The cheap tier, about a minute once the Rust build is warm, runs the guards
# CI runs in lint, hygiene, rust-lint and rust-test that need no CI service,
# with the Python scripts under Python 3.12, CI's interpreter, through
# `uv run --python 3.12` (T1.10c's l0_scan.py failed on CI's 3.12 and passed
# on a newer local Python):
#
#   cargo fmt --check; the three clippy runs CI makes; l0_deps.py and
#   l0_scan.py; contract_lock.py; rust_notices.py --check; the hygiene
#   check; ruff check and ruff format --check; land.py's self-tests.
#
# --full adds the L0 self-tests (selftest.py), the contract schema lock test,
# and replay_jobs.py: a clean virtual environment per CI job the change
# touches, replaying that job's install list and tests, which catches a
# test importing something the job does not install.
#
# --install-hook makes this the pre-push hook of the current worktree only:
# the hook, in the clone's shared hooks directory, runs the cheap tier for
# the worktrees listed beside it and does nothing in any other.
#
# Every check runs even after one fails; the summary lists each, and the exit
# status is non-zero if any failed.
#
#   scripts/rust_core/prepush.sh [--full] [--install-hook]

set -uo pipefail

full=0
install_hook=0
for arg in "$@"; do
  case "$arg" in
    --full) full=1 ;;
    --install-hook) install_hook=1 ;;
    -h | --help)
      awk 'NR > 5 && /^#/ { sub(/^# ?/, ""); print; next } NR > 5 { exit }' "$0"
      exit 0
      ;;
    *)
      echo "prepush.sh: unknown argument $arg (try --help)" >&2
      exit 2
      ;;
  esac
done

root=$(git rev-parse --show-toplevel) || exit 2
cd "$root" || exit 2

if [ "$install_hook" = 1 ]; then
  hooks=$(cd "$(git rev-parse --git-common-dir)" && pwd)/hooks
  hook="$hooks/pre-push"
  list="$hooks/prepush-worktrees"
  marker='# prepush.sh hook (scripts/rust_core/prepush.sh --install-hook)'
  mkdir -p "$hooks"
  if [ -e "$hook" ] && ! grep -qF "$marker" "$hook"; then
    echo "prepush.sh: $hook exists and is not this hook; not replacing it" >&2
    exit 2
  fi
  cat >"$hook" <<EOF
#!/usr/bin/env bash
$marker
# Runs the cheap tier in the worktrees listed in prepush-worktrees beside
# this file; in any other worktree of the clone it does nothing.
top=\$(git rev-parse --show-toplevel) || exit 0
grep -qxF "\$top" "\$(dirname "\$0")/prepush-worktrees" 2>/dev/null || exit 0
if [ ! -x "\$top/scripts/rust_core/prepush.sh" ]; then
  echo "pre-push: \$top has no scripts/rust_core/prepush.sh; not checking" >&2
  exit 0
fi
exec "\$top/scripts/rust_core/prepush.sh"
EOF
  chmod +x "$hook"
  touch "$list"
  grep -qxF "$root" "$list" || echo "$root" >>"$list"
  echo "prepush.sh: the pre-push hook runs the cheap tier in $root"
  echo "  (hook $hook; to stop, remove that line from $list)"
  exit 0
fi

py=(uv run --quiet --no-project --python 3.12 python3)
ruff=(uvx --quiet --from 'ruff==0.16.*' ruff)

names=()
results=()
times=()
failed=0
logdir=$(mktemp -d "${TMPDIR:-/tmp}/sdm-prepush.XXXXXX")

# check NAME DIR COMMAND...: runs COMMAND in DIR, keeping its output for the
# summary when it fails.
check() {
  local name=$1 dir=$2
  shift 2
  local log="$logdir/${#names[@]}.log" start=$SECONDS status
  printf '%-34s ' "$name"
  (cd "$dir" && "$@") >"$log" 2>&1
  status=$?
  local took=$((SECONDS - start))
  names+=("$name")
  times+=("$took")
  if [ "$status" = 0 ]; then
    results+=(ok)
    echo "ok   (${took}s)"
  else
    results+=(FAIL)
    failed=1
    echo "FAIL (${took}s, exit $status)"
    tail -n 30 "$log" | sed 's/^/    | /'
  fi
}

check 'cargo fmt --check' rust cargo fmt --all --check
check 'clippy: workspace' rust \
  cargo clippy --workspace --exclude sdm-testkit --all-targets --locked -- -D warnings
check 'clippy: sdm-py test-ops' rust \
  cargo clippy -p sdm-py --features test-ops --locked -- -D warnings
check 'clippy: sdm-testkit' rust \
  cargo clippy -p sdm-testkit --all-targets --locked -- -D warnings
check 'l0_deps.py' . "${py[@]}" scripts/rust_core/l0_deps.py
check 'l0_scan.py' . "${py[@]}" scripts/rust_core/l0_scan.py
check 'contract_lock.py' . "${py[@]}" scripts/rust_core/contract_lock.py
check 'rust_notices.py --check' . "${py[@]}" scripts/rust_core/rust_notices.py --check
check 'hygiene' . "${py[@]}" scripts/check_repo_hygiene.py all
check 'ruff check' . "${ruff[@]}" check .
check 'ruff format --check' . "${ruff[@]}" format --check .
check 'land.py self-tests' . "${py[@]}" scripts/rust_core/land_selftest.py

if [ "$full" = 1 ]; then
  check 'L0 self-tests' . "${py[@]}" scripts/rust_core/selftest.py
  check 'contract schema lock' rust \
    cargo test -p sdm-contract --locked --test schema_lock
  check 'replay the touched CI jobs' . \
    uv run --quiet --no-project --python 3.12 --with pyyaml \
    python3 scripts/rust_core/replay_jobs.py
fi

echo
echo "prepush summary ($([ "$full" = 1 ] && echo full || echo cheap) tier):"
for i in "${!names[@]}"; do
  printf '  %-4s %-34s %4ss\n' "${results[$i]}" "${names[$i]}" "${times[$i]}"
done
if [ "$failed" = 1 ]; then
  echo "prepush: FAILED; logs in $logdir"
  exit 1
fi
rm -rf "$logdir"
echo 'prepush: every check passed'
