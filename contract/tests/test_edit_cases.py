# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public schema edits use Rust and satisfy their declared outcomes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = sorted((ROOT / 'contract/cases/s1').glob('edit.*.json'))


@pytest.fixture(scope='module')
def observed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    out = tmp_path_factory.mktemp('edits') / 'python-core.jsonl'
    run = subprocess.run(
        [
            sys.executable,
            '-I',
            str(RUNNER),
            '--mode',
            'python-core',
            '--out',
            str(out),
            *map(str, CASES),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    observations = [json.loads(line) for line in out.read_text().splitlines()]
    return {each['case']: each for each in observations}


def test_there_are_refused_and_accepted_edit_cases() -> None:
    expects = [json.loads(path.read_text())['expect'] for path in CASES]
    assert sum(e['outcome'] == 'refusal' for e in expects) == 7
    assert sum(e['outcome'] == 'success' for e in expects) == 2


def test_each_edit_case_has_its_declared_outcome(
    observed: dict[str, Any],
) -> None:
    for path in CASES:
        case = json.loads(path.read_text())
        mine = observed[case['id']]
        expect = case['expect']
        assert mine['outcome']['result'] == expect['outcome'], case['id']
        assert mine['outcome']['stage'] == expect['stage'], case['id']
        assert mine['outcome']['code'] == expect.get('code'), case['id']
        # Public edits reach Rust; schema normalization may add core edits.
        # identity.operations: every core run started, in order.
        ops = mine['identity']['operations']
        made = ops.count('edit_schema')
        if expect['outcome'] == 'refusal':
            assert made >= len(case['input']['edits']), case['id']
            assert ops[-1] == 'edit_schema', case['id']
            assert mine['core']['error']['code'] == expect['code']
        else:
            assert made >= len(case['input']['edits']), case['id']
            assert ops[-1] == 'materialize', case['id']


def test_successful_edits_change_the_materialized_schema(
    observed: dict[str, Any],
) -> None:
    for name, time_column in (
        ('edit.accepted-meaning-key-links', 'ts'),
        ('edit.time-column-text', 'day'),
    ):
        core = observed[name]['core']
        schema = core['inputs'][-1]['schema']
        tables = {table['name']: table for table in schema['tables']}
        assert tables['orders']['time_column'] == time_column
        assert core['materialized']['payload']['context']['instance_table'][
            'rows'
        ]
        if name == 'edit.accepted-meaning-key-links':
            assert tables['users']['primary_key'] == ['user_id']
            columns = {
                column['name']: column for column in tables['users']['columns']
            }
            assert columns['age']['stype'] == 'categorical'
            assert schema['links'] == [
                [
                    {'table': 'orders', 'column': 'user_id'},
                    {'table': 'users', 'column': 'user_id'},
                ]
            ]
