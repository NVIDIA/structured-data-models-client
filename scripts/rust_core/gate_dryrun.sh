#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# The checking tools' dry run over the design's planned case kinds (T3.0):
#
#   scripts/rust_core/gate_dryrun.sh PLAN.yaml [--base REV] [--python PY]
#       [--rust] [--gate HALF]
#
# gate_dryrun.py beside it says what it does. The tools run with
# $SDM_TOOLS_PYTHON, else python3 when it holds PyYAML and jsonschema, else
# uv's python3 with them.
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -n "${SDM_TOOLS_PYTHON:-}" ]; then
  exec "$SDM_TOOLS_PYTHON" "$HERE/gate_dryrun.py" "$@"
elif python3 -c 'import yaml, jsonschema, referencing' 2>/dev/null; then
  exec python3 "$HERE/gate_dryrun.py" "$@"
else
  exec uv run --quiet --no-project --with pyyaml --with jsonschema python3 "$HERE/gate_dryrun.py" "$@"
fi
