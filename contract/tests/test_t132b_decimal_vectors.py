# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mixed-decimal keys preserve joined rows and result entity identities."""

import copy
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECORDED = json.loads((ROOT / 'contract/decimal-vectors.json').read_text())
BY_ID = {v['id']: v for v in RECORDED['vectors']}

PAIRS = [v['id'] for v in RECORDED['vectors'] if v['id'].startswith('key.pair')]

SAME_TYPE = 'key.both.d9-0'

KEY_COLUMNS = frozenset({'__kumo_entity_ref_0', 'user_id'})


def _key(cell: Any) -> Decimal:
    r"""A key cell's exact value, sent as a number or as its digits."""
    return Decimal(cell['number'] if isinstance(cell, dict) else cell)


def _joined(body: dict[str, Any], offset: int) -> dict[str, Any]:
    r"""Every table of a request, every row and cell, its key cells read
    exactly and less ``offset``: which entity each row is about and which
    foreign rows joined it."""
    out = {}
    for part in ('context', 'predict'):
        tables = {
            'instance_table': body[part]['instance_table'],
            **body[part]['related_tables'],
        }
        for name, table in tables.items():
            columns = table['columns']
            out[f'{part}.{name}'] = (
                columns,
                [
                    [
                        _key(cell) - offset
                        if column in KEY_COLUMNS and cell is not None
                        else cell
                        for column, cell in zip(columns, row, strict=True)
                    ]
                    for row in table['rows']
                ],
            )
    return out


def _offset(vector_id: str) -> int:
    return 10**17 if vector_id.startswith('key.pair-merging.') else 0


def _identities(observation: dict[str, Any], offset: int) -> list[Decimal]:
    r"""Each result row's entity id, exactly, less ``offset``."""
    result = observation['core']['predicted']['result']
    at = [c['name'] for c in result['columns']].index('id')
    return [
        Decimal(next(iter(row[at].values()))) - offset for row in result['rows']
    ]


@pytest.mark.parametrize('vector_id', [pytest.param(i, id=i) for i in PAIRS])
def test_a_decimal_key_pair_joins_exactly_on_the_core(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    r"""Exact decimals (T2.0c): every pair, the merging ones included,
    matches the current ordinary-key twin, every row and cell of every
    table, with keys read exactly and shifted by the pair's base: the same
    entities joined to the same foreign rows. Each result row's id is the
    entity asked for, exactly, in order."""
    observation = core_runs[vector_id]
    assert observation['outcome']['result'] == ('success')
    offset = _offset(vector_id)
    [got] = [p['parsed'] for p in observation['public']['payloads']]
    [same] = [p['parsed'] for p in core_runs[SAME_TYPE]['public']['payloads']]
    assert _joined(got, offset) == _joined(same, 0)
    assert _identities(observation, offset) == [Decimal(0), Decimal(1)]
    assert [
        Decimal(i) - offset for i in BY_ID[vector_id].get('indices', [0, 1])
    ] == [0, 1]


def test_the_exact_join_check_fails_on_a_merged_key(
    core_runs: dict[str, Any],
) -> None:
    r"""Codex's round 2 repro: every non-null ``orders.user_id`` of the
    merging pair replaced by the valid but wrong key 10**17 keeps every row
    and null count, and fails the exact comparison; so does one entity's
    result id merged into the other's."""
    vector_id = 'key.pair-merging.d38-0.d38-2'
    observation = copy.deepcopy(core_runs[vector_id])
    [got] = [p['parsed'] for p in observation['public']['payloads']]
    [same] = [p['parsed'] for p in core_runs[SAME_TYPE]['public']['payloads']]
    offset = _offset(vector_id)
    assert _joined(got, offset) == _joined(same, 0)
    merged = copy.deepcopy(got)
    for part in ('context', 'predict'):
        orders = merged[part]['related_tables']['orders']
        at = orders['columns'].index('user_id')
        for row in orders['rows']:
            if row[at] is not None:
                row[at] = str(10**17)
    assert _joined(merged, offset) != _joined(same, 0)
    result = observation['core']['predicted']['result']
    at = [c['name'] for c in result['columns']].index('id')
    result['rows'][1][at] = dict(result['rows'][0][at])
    assert _identities(observation, offset) != [Decimal(0), Decimal(1)]


@pytest.fixture(scope='module')
def core_runs(tmp_path_factory):
    directory = tmp_path_factory.mktemp('decimal-joins')
    cases = directory / 'cases'
    cases.mkdir()
    template = json.loads(
        (ROOT / 'contract/cases/s1/decimal.feature-38-0-wide.json').read_text()
    )
    for vector_id in [SAME_TYPE, *PAIRS]:
        vector = BY_ID[vector_id]
        case = copy.deepcopy(template)
        case['id'] = vector_id
        case['status'] = 'diagnostic'
        case['routes'] = []
        case['cells'] = []
        case['input']['setup'] = vector['setup']
        case['input']['pql'] = RECORDED['question']
        indices = vector.get('indices', RECORDED['indices'])
        case['input']['materialize']['indices'] = indices
        case['input']['materialize']['anchor_time'] = RECORDED['anchor_time']
        case['core']['materialize']['indices'] = [
            {'t': 'i64', 'v': str(i)} for i in indices
        ]
        case['expect'] = {'outcome': 'success', 'stage': 'predict'}
        case.pop('register', None)
        (cases / f'{vector_id}.json').write_text(json.dumps(case))
    out = directory / 'observations.jsonl'
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/run_python.py'),
            '--mode',
            'python-core',
            '--out',
            str(out),
            str(cases),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    observed = {
        row['case']: row
        for row in map(json.loads, out.read_text().splitlines())
    }
    assert set(observed) == {SAME_TYPE, *PAIRS}
    for row in observed.values():
        assert row['outcome']['result'] == 'success', row['outcome']
        assert row['identity']['native_loaded']
        assert {'materialize', 'predict'} <= set(row['identity']['operations'])
    return observed
