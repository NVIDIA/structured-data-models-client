# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1a's runner test for the reference side off DuckDB (the independent
test author's), beside ``tests/test_t21a_reference_lines.py``.

It runs ``run_python.py --mode python-ref --database sqlite`` on one case
and holds its outcome to the committed ``@sqlite`` line: off DuckDB the
reference side is python-ref's (the orchestrator's decision, T2.1a fix 4:
until the core's providers land, a SQLite or Postgres job runs python-ref
only, and python-old is refused there by name). The candidate's
``sdm_client`` on its default routes stands in for the reference
checkout's, as in ``contract/tests/test_runners.py``, so the test needs no
reference checkout. It needs the relational engine, which a case's
``from_source`` and ``materialize`` use: here, in the contract jobs that
install it, rather than in the root suite's ``test`` jobs, which do not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
#: A plain S1 case every database loads (int keys, a timestamp, a float).
SMOKE = 'base.regression'
KEY = f'{SMOKE}@sqlite'


def _reference_line() -> dict[str, Any]:
    [path] = sorted((ROOT / 'contract/reference').glob('*.jsonl'))
    lines = map(json.loads, path.read_text().splitlines())
    [line] = [x for x in lines if x['case'] == KEY]
    found: dict[str, Any] = line
    return found


def _statements(line: dict[str, Any]) -> list[Any]:
    capture = (line.get('driver') or {}).get('capture') or {}
    found: list[Any] = capture.get('statements') or []
    return found


def test_python_ref_on_sqlite_meets_the_reference_line(
    tmp_path: Path,
) -> None:
    r"""Exit: the reference side a SQLite job runs: python-ref over the
    loaded SQLite database, its observation keyed ``<case>@sqlite`` and its
    outcome the committed line's; python-old is refused there by name."""
    reference = tmp_path / 'reference'
    (reference / 'scripts').mkdir(parents=True)
    (reference / 'scripts/reference_guard.py').write_text(
        'def require_isolation():\n    pass\n'
    )
    # A clean checkout of one commit, whose SHA the identity records.
    environment = {
        **os.environ,
        'GIT_AUTHOR_NAME': 'T',
        'GIT_AUTHOR_EMAIL': 't@example.invalid',
        'GIT_COMMITTER_NAME': 'T',
        'GIT_COMMITTER_EMAIL': 't@example.invalid',
    }
    for command in (
        ['init', '-q'],
        ['add', '-A'],
        ['commit', '-q', '--no-gpg-sign', '-m', 'the stand-in'],
    ):
        subprocess.run(
            ['git', *command],
            cwd=reference,
            env=environment,
            check=True,
            capture_output=True,
        )
    out = tmp_path / 'python-ref.jsonl'
    # The candidate's sdm_client stands in for the reference checkout's, so
    # the check that it is the checkout's is replaced.
    script = (
        'import importlib.util, sys\n'
        'spec = importlib.util.spec_from_file_location(\n'
        f"    'run_python', {str(RUNNER)!r})\n"
        'runner = importlib.util.module_from_spec(spec)\n'
        'spec.loader.exec_module(runner)\n'
        'runner._require_reference_client = lambda reference: None\n'
        'sys.exit(runner.main([\n'
        f"    '--mode', 'python-ref', '--reference', {str(reference)!r},\n"
        f"    '--database', 'sqlite', '--out', {str(out)!r},\n"
        f'    {str(ROOT / f"contract/cases/s1/{SMOKE}.json")!r},\n'
        ']))\n'
    )
    done = subprocess.run(
        [sys.executable, '-I', '-c', script],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert done.returncode == 0, done.stderr
    [line] = map(json.loads, out.read_text().splitlines())
    assert line['case'] == KEY
    assert line['identity']['runner'] == 'python-ref'
    assert line['outcome'] == _reference_line()['outcome']
    assert _statements(line), 'python-ref read SQLite through the capture'

    refused = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNER),
            '--mode',
            'python-old',
            '--database',
            'sqlite',
            '--out',
            str(tmp_path / 'python-old.jsonl'),
            str(ROOT / f'contract/cases/s1/{SMOKE}.json'),
        ],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert refused.returncode == 1, refused.stdout + refused.stderr
    assert 'python-old runs on DuckDB only' in refused.stderr
    assert not (tmp_path / 'python-old.jsonl').exists()
