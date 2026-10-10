# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""base.regression's schema on the core route against the reference (T1.11b).

Run where the contract tests run (CI's ``engine`` job until T1.18):

    python -m pytest contract/tests -q

On ``python-core``, ``Schema.from_source`` runs on the core: its operation
is ``describe_source``, and the public schema is the reference's, from the
committed reference observation. The driver capture shows the core's SQL
for the schema is the reference's: the same statements, in order, text for
text, each returning the same number of rows (decision H9 gates on the
count and the results; the text is the renderings' own claim). The case is
diagnostic until T1.29, so the comparator reports its differences and
gates none (``test_runners.py`` runs it over every case); this test holds
the schema and the statements.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = ROOT / 'contract/cases/s1'
REFERENCE = next((ROOT / 'contract/reference').glob('*.jsonl'))


def _reference() -> dict[str, Any]:
    for line in REFERENCE.read_text().splitlines():
        observation = json.loads(line)
        if observation['case'] == 'base.regression':
            return observation
    raise AssertionError('no reference observation of base.regression')


def _run(mode: str, out: Path) -> dict[str, Any]:
    run = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNER),
            '--mode',
            mode,
            '--out',
            str(out),
            str(CASES),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    [observation] = [json.loads(line) for line in out.read_text().splitlines()]
    return observation


def test_base_regression_s_schema_on_the_core_is_the_reference_s(
    tmp_path: Path,
) -> None:
    reference = _reference()
    core = _run('python-core', tmp_path / 'python-core.jsonl')
    old = _run('python-old', tmp_path / 'python-old.jsonl')
    # From T1.11c the call's materialize runs on the core too
    # (test_materialize_route.py).
    assert core['identity']['operations'] == ['describe_source', 'materialize']
    assert core['core']['error'] is None
    assert core['core']['described'] is not None
    assert old['identity']['native_loaded'] is False
    # The inputs, step trace and description sdm-testkit's
    # tests/duckdb_provider.rs holds rust-host to: CI gates the two hosts'
    # equality here until T1.18's comparator does.
    pinned = json.loads(
        (ROOT / 'contract/pinned/base.regression.core.json').read_text()
    )
    for field in ('inputs', 'fetches', 'described'):
        assert core['core'][field] == pinned[field], field
    for observed in (core, old):
        assert observed['public']['schema'] == reference['public']['schema']
        assert observed['outcome'] == reference['outcome']


def test_the_core_s_sql_for_the_schema_is_the_reference_s(
    tmp_path: Path,
) -> None:
    reference = _reference()['driver']['capture']['statements']
    core = _run('python-core', tmp_path / 'python-core.jsonl')
    asked = core['core']['fetches']
    # Schema.from_source asks one statement per fetch, the first ten; what
    # follows is materialize's (test_materialize_route.py).
    asked = asked[:10]
    ran = core['driver']['capture']['statements'][: len(asked)]
    assert len(asked) == 10
    assert [s['sql'] for s in ran] == [s['sql'] for s in reference[:10]]
    assert [
        statement['text']
        for fetch in asked
        for statement in fetch['plan']['statements']
    ] == [s['sql'] for s in ran]
    assert [[r['rows'] for r in s['results']] for s in ran] == [
        [r['rows'] for r in s['results']] for s in reference[:10]
    ]
    # Each fixed-shape answer is checked against its shape (decision E36).
    shaped = [
        fetch
        for fetch in asked
        if fetch['plan']['expect']['expect'] == 'schema'
    ]
    assert [f['request']['request'] for f in shaped] == [
        'describe',
        'describe',
        'describe',
        'describe',
        'describe',
        'describe',
        'describe',
        'key_counts',
    ]
