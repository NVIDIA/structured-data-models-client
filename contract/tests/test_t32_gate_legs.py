# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The gate compares each leg its observations hold, each with its own
files (T3.2 fix 5, Claude r1 P1-2): ``gate.sh`` called ``compare.py`` once,
with no ``--database``, so only DuckDB was judged, a warehouse or Java line
was only noted, and the stand-in's files beside python-core's were refused
as "a second file of python-core"."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.tooling

HERE = Path(__file__).resolve().parent
if 'gate_modes_t32' in sys.modules:
    modes = sys.modules['gate_modes_t32']
else:
    spec = importlib.util.spec_from_file_location(
        'gate_modes_t32', HERE / 'test_gate_modes.py'
    )
    assert spec is not None and spec.loader is not None
    modes = importlib.util.module_from_spec(spec)
    sys.modules['gate_modes_t32'] = modes
    spec.loader.exec_module(modes)

origin = modes.origin
observed = modes.observed
repo = modes.repo

#: A stand-in for the gate's ``$PYTHON`` that logs each ``compare.py`` run.
LOGGING_PYTHON = """#!/bin/bash
for given in "$@"; do
  case "$given" in
    */compare.py) printf '%s\\n' "$*" >>"$COMPARE_LOG" ;;
  esac
done
exec {python} "$@"
"""


def test_the_gate_compares_each_leg_with_that_leg_s_files(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    landed = modes.git(repo, 'rev-parse', 'HEAD')
    modes.commit(repo)
    given = modes.files(observed, tmp_path)
    # A warehouse leg's stand-in file: python-core's line of the case, on
    # Snowflake.
    line = next(x for x in observed['python-core'] if x['case'] == modes.CASE)
    standin = tmp_path / 'given' / 'standin-snowflake.jsonl'
    standin.write_text(
        json.dumps({**line, 'case': f'{modes.CASE}@snowflake'}) + '\n'
    )
    python = tmp_path / 'python'
    python.write_text(LOGGING_PYTHON.format(python=sys.executable))
    python.chmod(0o755)
    log = tmp_path / 'compare.log'
    done = modes.subprocess.run(
        [str(repo / 'scripts/rust_core/gate.sh'), '--no-l0', '--base', landed,
         '--only', 'compare', *given, '--observations', str(standin)],
        capture_output=True, text=True, check=False, cwd=repo,
        env={**modes.os.environ, 'PYTHON': str(python), 'COMPARE_LOG': str(log)},
    )  # fmt: skip
    said = done.stdout + done.stderr
    runs = log.read_text().splitlines() if log.exists() else []
    on = {
        run.split('--database ', 1)[1].split()[0]: run
        for run in runs
        if '--database ' in run
    }
    assert sorted(on) == ['duckdb', 'snowflake'], (runs, said[-3000:])
    assert str(standin) in on['snowflake'], on['snowflake']
    assert 'python-core.jsonl' not in on['snowflake'], on['snowflake']
    assert str(standin) not in on['duckdb'], on['duckdb']
    assert 'a second file of python-core' not in said, said[-3000:]
