# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.2's SQLite regressions from review rounds 1 and 2 on
``python-core`` (the package's test author's), through the public calls
with SQLite's core routes forced, each held to the reference's answer in
``contract/sqlite-regression-vectors.json`` (recorded by
``contract/tools/sqlite_regressions_record.py``):

- a caller's own temporary table named as the provider's scratch
  relations are or were (``__sdm_seeds__``, ``__sdm_seeds_1__``,
  ``__SDM_SEEDS_1__``) survives ``materialize`` unchanged, and the call
  answers as on the reference;
- an ``INT PRIMARY KEY`` holding 101.5 to 107.5 gets the reference's
  seeded context keys, 102.5, 106.5 and 107.5 (the hash is chosen by the
  column's type, not each value's storage class);
- an ``INT PRIMARY KEY`` holding a NULL materializes as the reference does,
  the null filtered (a SQLite primary key that is not a rowid alias keeps
  its catalog nullability);
- a column declared ``INTEGER`` holding fractions is described as the
  reference describes it (``int64``), and the request is the reference's;
- a composite primary key with one ``INTEGER`` column is described as the
  reference reads it, and refused at ``materialize`` as there;
- a partial unique index on the key is read as unique: the reference's
  schema, statement counts per call and request.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module(
    'sqlite_regressions_record',
    ROOT / 'contract/tools/sqlite_regressions_record.py',
)
RECORDED: dict[str, Any] = json.loads(
    (ROOT / 'contract/sqlite-regression-vectors.json').read_text('utf-8')
)
SEEDS = ('seeds-table', 'seeds-table-numbered', 'seeds-table-upper')


def _on_the_core(monkeypatch: pytest.MonkeyPatch, name: str) -> dict[str, Any]:
    r"""The scenario through the public calls on SQLite's core routes:
    ``from_source`` (tables named or not), ``materialize`` and
    validation."""
    from sdm_client import _routing

    routes = (
        _routing.FROM_SOURCE_SQLITE
        + _routing.FROM_SOURCE_SQLITE_TABLES
        + _routing.MATERIALIZE_SQLITE
        + _routing.S0
    )
    # Not vacuous: the build routes each forced path on the core.
    assert set(routes) <= _routing.CORE_ROUTED
    monkeypatch.setenv(_routing.ENVIRONMENT, ','.join(dict.fromkeys(routes)))
    answer: dict[str, Any] = R.run(RECORDED['scenarios'][name])
    return answer


def _reference(name: str) -> dict[str, Any]:
    found: dict[str, Any] = RECORDED['scenarios'][name]['reference']
    return found


def _context_keys(answer: dict[str, Any]) -> list[Any]:
    [payload] = answer['payloads']
    rows = payload['context']['instance_table']['rows']
    return [row[1] for row in rows]


def test_the_vectors_are_the_reference_s() -> None:
    r"""Control: recorded from the reference this tree names, every
    scenario of the recorder, as the reviews' repros measured it."""
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    assert set(RECORDED['scenarios']) == set(R.SCENARIOS)
    for name, scenario in RECORDED['scenarios'].items():
        assert scenario['setup'] == R.SCENARIOS[name]['setup']
    for name in SEEDS:
        assert _reference(name)['temp_after'] == {
            'columns': ['important'],
            'rows': [['keep me']],
        }
    assert _context_keys(_reference('float-keys')) == [102.5, 106.5, 107.5]
    composite = _reference('composite-key')['outcome']
    assert (composite['result'], composite['code']) == (
        'refusal',
        'INVALID_REQUEST',
    )
    [users] = [
        table
        for table in _reference('integer-fractions')['schema']['tables']
        if table['name'] == 'users'
    ]
    assert {'name': 'score', 'dtype': 'int64', 'stype': 'numerical'} in (
        users['columns']
    )


@pytest.mark.parametrize('name', SEEDS)
def test_a_caller_s_temporary_table_survives_materialize(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    r"""The caller's temporary table, named as a scratch relation is or
    was, holds its one column and ``'keep me'`` after ``materialize``, as
    on the reference, and the call answers as the reference does."""
    wanted = _reference(name)
    got = _on_the_core(monkeypatch, name)
    assert got['temp_after'] == wanted['temp_after']
    assert got['outcome'] == wanted['outcome']
    assert got['payloads'] == wanted['payloads']


def test_float_keys_of_an_int_column_draw_the_reference_s_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""An ``INT PRIMARY KEY`` holding 101.5 to 107.5: the seeded context
    keys are the reference's (102.5, 106.5, 107.5), and the request is the
    reference's."""
    wanted = _reference('float-keys')
    got = _on_the_core(monkeypatch, 'float-keys')
    assert got['outcome'] == wanted['outcome']
    assert _context_keys(got) == _context_keys(wanted)
    assert got['payloads'] == wanted['payloads']


def test_a_null_primary_key_is_filtered_as_the_reference_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""An ``INT PRIMARY KEY`` holding a NULL: ``materialize`` succeeds, the
    null filtered, and the request is the reference's."""
    wanted = _reference('null-key')
    got = _on_the_core(monkeypatch, 'null-key')
    assert got['outcome'] == wanted['outcome']
    assert got['payloads'] == wanted['payloads']


def test_an_integer_column_holding_fractions_is_described_as_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""A column declared ``INTEGER`` holding fractions: the public schema
    is the reference's (``score`` an ``int64``), and so is the request."""
    wanted = _reference('integer-fractions')
    got = _on_the_core(monkeypatch, 'integer-fractions')
    assert got['schema'] == wanted['schema']
    assert got['outcome'] == wanted['outcome']
    assert got['payloads'] == wanted['payloads']


def test_a_composite_key_s_integer_column_is_read_as_the_reference_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""A composite primary key with one ``INTEGER`` column: the public
    schema is the reference's, and ``materialize`` refuses as there."""
    wanted = _reference('composite-key')
    got = _on_the_core(monkeypatch, 'composite-key')
    assert got['schema'] == wanted['schema']
    assert got['outcome'] == wanted['outcome']


def test_a_partial_unique_index_keeps_the_key_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""A partial unique index on the key: the public schema, the
    statements run per call and the request are the reference's."""
    wanted = _reference('partial-unique')
    got = _on_the_core(monkeypatch, 'partial-unique')
    assert got['schema'] == wanted['schema']
    assert got['statements'] == wanted['statements']
    assert got['outcome'] == wanted['outcome']
    assert got['payloads'] == wanted['payloads']
