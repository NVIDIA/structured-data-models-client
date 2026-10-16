#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Prepares a one-time run of dbr_install_check.py, the "On Databricks"
# install check, for one commit on one Databricks runtime:
#
#   dbr_install_check_prepare.sh SHA CI_RUN_ID OUT            # dry run
#   dbr_install_check_prepare.sh SHA CI_RUN_ID OUT --build    # local steps
#
# SHA is the commit under test (all 40 characters); CI_RUN_ID the GitHub
# Actions run whose `native-wheel` artifact is the client wheel to install.
# That run is on SHA or, when SHA has no run of its own (an empty landing
# commit), on WHEEL_COMMIT (env), whose tree must equal SHA's: the wheel's
# identity file must name it. OUT is a local directory this script owns:
# OUT/wheels the wheel set, OUT/src a worktree of SHA, OUT/job.json the
# filled run payload. An OUT/wheels that is not empty is refused.
#
# Optional environment:
#   SDM_DBX_TARGET   the runtime, a classic target of dbx_runs.py
#                    (default dbr14.3-x86; the client's native wheel is
#                    x86_64 only, so an -arm target is refused)
#   ANCESTOR_REF     when set, SHA must be an ancestor of it (for instance
#                    origin/<phase branch>)
#   GH_REPO          the repository of CI_RUN_ID (default: this clone's
#                    GitHub origin, through gh)
#   ENGINE_CACHE     a directory of engine-<tree>-<cpXY>/ wheels from an
#                    earlier build of the same engine tree, reused as is; a
#                    fresh build is copied there
#
# --build also fills OUT/job.json, through dbx_runs.py spec, which reads the
# workspace settings from SDM_DBX_WORKSPACE_FOLDER, SDM_DBX_VOLUME_DIR,
# SDM_DBX_POLICY_ID, SDM_DBX_USER and, optionally, SDM_DBX_CLUSTER_EXTRA;
# the file is mode 0600 and never printed. Every workspace value comes from
# those variables: none has a default and none is written here.
#
# This script NEVER contacts the workspace. The upload, the notebook import
# and the one-time submit are printed, with the variables unexpanded, for
# the operator to run once. It deletes nothing and overwrites nothing: the
# upload goes to a new folder per commit, and the notebook is imported as
# a new file, without --overwrite, so a name that exists fails. No existing
# notebook, cluster or job is touched.
set -euo pipefail

usage() { sed -n '/^# Prepares/,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }
case ${1:-} in -h | --help) usage; exit 0 ;; esac

SHA=${1:?full commit SHA (see --help)}
RUN=${2:?CI run id with a native-wheel artifact}
OUT=${3:?output directory}
MODE=${4:-dry}
[ "${#SHA}" -eq 40 ] || { echo 'give the full 40-character SHA' >&2; exit 2; }
case $MODE in dry | --build) ;; *) echo "unknown mode $MODE" >&2; exit 2 ;; esac
S7=${SHA:0:7}
WHEEL_COMMIT=${WHEEL_COMMIT:-$SHA}
[ "${#WHEEL_COMMIT}" -eq 40 ] || { echo 'WHEEL_COMMIT must be a full SHA' >&2; exit 2; }
TARGET=${SDM_DBX_TARGET:-dbr14.3-x86}
# The runtime's Python, for the relational engine's wheel.
case $TARGET in
  dbr14.3-x86) CP=cp310 ;;
  dbr15.4-x86) CP=cp311 ;;
  dbr16.4-x86 | dbr17.3-x86) CP=cp312 ;;
  *) echo "target $TARGET: no x86_64 classic runtime this script knows" >&2; exit 2 ;;
esac
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
GH_REPO=${GH_REPO:-$(cd "$REPO" && gh repo view --json nameWithOwner -q .nameWithOwner)}
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
WHEELS=$OUT/wheels
WT=$OUT/src
FILLED=$OUT/job.json
NOTEBOOK=dbr_install_check_$S7

step() { # step DESCRIPTION -- COMMAND...
  local what=$1
  shift
  [ "$1" = -- ] && shift
  echo "== $what"
  printf '   %q' "$@"
  echo
  if [ "$MODE" = --build ]; then "$@"; fi
}

echo "install check prep for $S7 on $TARGET (mode: ${MODE#--})"

# 0. Preconditions (local reads only).
if [ -n "${ANCESTOR_REF:-}" ]; then
  step "the SHA is on $ANCESTOR_REF" -- git -C "$REPO" merge-base --is-ancestor "$SHA" "$ANCESTOR_REF"
fi
if [ "$WHEEL_COMMIT" != "$SHA" ]; then
  [ "$(git -C "$REPO" rev-parse "$WHEEL_COMMIT^{tree}")" = "$(git -C "$REPO" rev-parse "$SHA^{tree}")" ] \
    || { echo "!! WHEEL_COMMIT ${WHEEL_COMMIT:0:12}'s tree differs from ${SHA:0:12}'s" >&2; exit 3; }
  echo "== the CI wheel is ${WHEEL_COMMIT:0:12}'s, the same tree as ${SHA:0:12}"
fi
ENGINE_TREE=$(git -C "$REPO" rev-parse "$SHA:packages/structured-data-models-relational-engine")
if [ -n "$(ls -A "$WHEELS" 2>/dev/null)" ]; then
  echo "!! $WHEELS is not empty: pick it up or use a new OUT; nothing is deleted" >&2
  exit 3
fi

# 1. A clean tree of exactly SHA.
[ -d "$WT" ] || step "worktree at SHA" -- git -C "$REPO" worktree add --detach "$WT" "$SHA"
step "wheel directory" -- mkdir -p "$WHEELS/native"

# 2. The client: CI's shipped native wheel, held to its identity file
#    (commit, a clean tree, its SHA-256, abi3 and x86_64).
step "download native-wheel of run $RUN" -- gh run download "$RUN" -R "$GH_REPO" -n native-wheel -D "$WHEELS/native"
step "hold the wheel to its identity" -- python3 -I -c '
import hashlib, json, pathlib, sys
d, sha = pathlib.Path(sys.argv[1]), sys.argv[2]
wheels = sorted(d.glob("structured_data_models_client-*.whl"))
assert len(wheels) == 1, wheels
w = wheels[0]
ident = json.loads(pathlib.Path(str(w) + ".json").read_text())
assert ident["commit"] == sha, ("commit", ident["commit"])
assert ident["tree_clean"] is True, "tree not clean"
assert ident["sha256"] == hashlib.sha256(w.read_bytes()).hexdigest(), "sha256"
assert "abi3" in w.name and "x86_64" in w.name, w.name
print("native wheel ok:", w.name, ident["sha256"][:12])
' "$WHEELS/native" "$WHEEL_COMMIT"
step "copy the client wheel beside the others" -- bash -c 'cp "$1"/native/structured_data_models_client-*.whl "$1"/' _ "$WHEELS"

# 3. The three pure-Python siblings, from the clean tree.
for p in api pql connectors; do
  step "build structured-data-models-$p" -- uvx --from build pyproject-build --wheel \
    --outdir "$WHEELS" "$WT/packages/structured-data-models-$p"
done

# 4. The relational engine for the runtime's Python, manylinux x86_64, in
#    docker; the siblings are staged where its test environment finds them.
CACHED=${ENGINE_CACHE:+$ENGINE_CACHE/engine-$ENGINE_TREE-$CP}
if [ -n "$CACHED" ] && ls "$CACHED"/structured_data_models_relational_engine-*-"$CP"-"$CP"-*x86_64*.whl >/dev/null 2>&1; then
  step "reuse the engine built from the same tree ($ENGINE_TREE)" -- bash -c 'cp "$1"/*.whl "$2"/' _ "$CACHED" "$WHEELS"
else
  step "stage the siblings for the engine's test environment" -- bash -c \
    'mkdir -p "$1/_sibling_wheels" && cp "$2"/*.whl "$1/_sibling_wheels/"' _ "$WT" "$WHEELS"
  step "build the engine (docker)" -- bash -c 'cd "$1" && shift && exec "$@"' _ "$WT" \
    env CIBW_BUILD="$CP-*" CIBW_ARCHS_LINUX=x86_64 \
    CIBW_ENVIRONMENT_LINUX='PIP_FIND_LINKS=/project/_sibling_wheels PIP_RETRIES=15 PIP_DEFAULT_TIMEOUT=120' \
    uvx cibuildwheel==4.2.1 --platform linux --output-dir "$WHEELS" packages/structured-data-models-relational-engine
  if [ -n "$CACHED" ]; then
    step "keep the engine for a later build of the same tree" -- bash -c \
      'mkdir -p "$1" && cp "$2"/structured_data_models_relational_engine-*.whl "$1"/' _ "$CACHED" "$WHEELS"
  fi
fi

# 5. The set: five distributions at one version, the client native.
step "check the wheel set" -- python3 -I -c '
import pathlib, re, sys
d, cp = pathlib.Path(sys.argv[1]), sys.argv[2]
names = sorted(p.name for p in d.glob("*.whl"))
versions = {re.match(r"[^-]+-([^-]+)-", n).group(1) for n in names}
assert len(names) == 5 and len(versions) == 1, names
(v,) = versions
want = (f"structured_data_models_api-{v}-", f"structured_data_models_pql-{v}-",
        f"structured_data_models_connectors-{v}-",
        f"structured_data_models_client-{v}-cp310-abi3-",
        f"structured_data_models_relational_engine-{v}-{cp}-{cp}-")
for prefix in want:
    assert sum(n.startswith(prefix) for n in names) == 1, (prefix, names)
print("wheel set ok:", *names, sep="\n  ")
' "$WHEELS" "$CP"

# 6. The one-time run's payload, from the template and SDM_DBX_* settings.
step "fill the one-time run payload (not printed)" -- python3 -I "$HERE/dbx_runs.py" spec \
  --template "$HERE/dbr_install_check.job.template.json" --out "$FILLED" \
  --target "$TARGET" --notebook "$NOTEBOOK" --sha "$SHA"

cat <<EOF

== NOT RUN HERE. For the operator, once, with the SDM_DBX_* variables set:
  # a. the wheels to a NEW volume folder (the ls must fail first):
  databricks --profile "\$SDM_DBX_PROFILE" fs ls "dbfs:\$SDM_DBX_VOLUME_DIR/sdm-client-$S7"
  databricks --profile "\$SDM_DBX_PROFILE" fs mkdir "dbfs:\$SDM_DBX_VOLUME_DIR/sdm-client-$S7"
  for w in $WHEELS/*.whl; do
    databricks --profile "\$SDM_DBX_PROFILE" fs cp "\$w" "dbfs:\$SDM_DBX_VOLUME_DIR/sdm-client-$S7/"
  done
  # b. the notebook as a NEW file (no --overwrite: fails if the name exists):
  databricks --profile "\$SDM_DBX_PROFILE" workspace import "\$SDM_DBX_WORKSPACE_FOLDER/$NOTEBOOK" \\
      --file $HERE/dbr_install_check.py --language PYTHON --format SOURCE
  # c. the one-time run (runs/submit; no saved job), then its result:
  python3 $HERE/dbx_runs.py submit --json $FILLED
  python3 $HERE/dbx_runs.py wait RUN_ID   # the 'SDM_INSTALL_CHECK after' line, or the exit value
EOF
