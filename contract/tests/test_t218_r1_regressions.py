# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.18's round 1 regressions on ``python-core`` (the package's test
author's), through the public calls on the core route, each held to the
reference's answer for the column's ``a_b`` twin in
``contract/column-name-regression-vectors.json`` (recorded by
``contract/tools/column_name_regressions_record.py``, whose ``run`` these
tests call):

- ``materialize(..., task='regression')`` on a query naming a column
  holding a dot, a backtick, a line feed or a carriage return is refused as
  its twin is, and as every relational query is (the review's P2-2): the
  class, code, call and message of the twin's refusal;
- a misspelt ordinary column (``orders.pricee``) over a schema holding
  such a column keeps the reference's diagnostic, the misspelt column
  named (the review's P3-3);
- a forecast of two steps summing such a column, with
  ``sdm_client.decompose.MOST_PARTS`` set to 1, is refused as its twin is,
  the caller's limit honoured (Codex's round 2 P2), with S0 on its
  default route and on the core.

Each runs on DuckDB, SQLite and Postgres, with ``from_source`` and
``materialize`` on the core and the query's own validation (S0) on its
default route, as the ``name.*`` cases route them. The twin's query text
in a message is read as the vector's. Where ``SDM_POSTGRES_TEST_DSN``
names no server and ``SDM_REQUIRE_POSTGRES`` is not set, the Postgres leg
has nothing to hold and returns without asserting; with it set, a missing
server fails (H26).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATABASES = ('duckdb', 'sqlite', 'postgres')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module(
    'column_name_regressions_record',
    ROOT / 'contract/tools/column_name_regressions_record.py',
)
RECORDED: dict[str, Any] = json.loads(
    (ROOT / 'contract/column-name-regression-vectors.json').read_text('utf-8')
)
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}


def reachable(database: str) -> bool:
    if database != 'postgres' or os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


def _on_the_core(
    monkeypatch: pytest.MonkeyPatch, vector: dict[str, Any], s0: bool = False
) -> dict[str, Any]:
    r"""The vector with its database's ``from_source`` and ``materialize``
    forced onto the core, S0 on its default route or, with ``s0``, on the
    core too."""
    from sdm_client import _routing

    database = vector['database']
    routes = (*_routing.FROM_SOURCE[database], *_routing.MATERIALIZE[database])
    if s0:
        routes = (*routes, *_routing.S0)
    # Not vacuous: the build routes each forced path on the core.
    assert set(routes) <= _routing.CORE_ROUTED
    monkeypatch.setenv(_routing.ENVIRONMENT, ','.join(dict.fromkeys(routes)))
    found: dict[str, Any] = R.run(vector)
    return found


def _wanted(vector: dict[str, Any]) -> dict[str, Any]:
    r"""The twin's reference refusal, its query text read as the vector's."""
    twin = BY_ID[vector['twin']]
    wanted = dict(twin['reference'])
    wanted['message'] = wanted['message'].replace(twin['pql'], vector['pql'])
    return wanted


def _vectors(kind: str, database: str) -> list[dict[str, Any]]:
    return [
        v
        for v in RECORDED['vectors']
        if v['kind'] == kind
        and v['database'] == database
        and v['character'] is not None
    ]


def test_the_vectors_are_the_reference_s() -> None:
    r"""Control: recorded from the reference this tree names, every
    vector of the recorder; each twin refused INVALID_REQUEST at
    ``materialize``, ``task`` for a relational query and the typo naming
    ``pricee``, so the twin is the expectation these hold."""
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    expected = R.vectors()
    assert [v['id'] for v in RECORDED['vectors']] == [v['id'] for v in expected]
    for vector, wanted in zip(RECORDED['vectors'], expected, strict=True):
        assert {k: vector[k] for k in wanted} == wanted, vector['id']
        answer = vector['reference']
        assert (answer['result'], answer['stage'], answer['code']) == (
            'refusal',
            'materialize',
            'INVALID_REQUEST',
        ), vector['id']
        if vector['character'] is None:
            assert {
                'task': "task='regression'",
                'typo': "Column 'pricee' does not exist",
                'parts': 'past the 1 this will do',
            }[vector['kind']] in answer['message'], vector['id']


@pytest.mark.parametrize('database', DATABASES)
def test_task_over_any_column_name_is_refused_as_its_twin(
    monkeypatch: pytest.MonkeyPatch, database: str
) -> None:
    r"""P2-2: ``task='regression'`` with a query summing a column named
    with each character is refused as the twin's query is on the
    reference: class, code, call and message."""
    if not reachable(database):
        return
    problems = []
    for vector in _vectors('task', database):
        got = _on_the_core(monkeypatch, vector)
        if got != _wanted(vector):
            problems.append(f'{vector["id"]}: {got}')
    assert not problems, '\n'.join(problems)


@pytest.mark.parametrize('database', DATABASES)
def test_a_misspelt_column_keeps_its_diagnostic_beside_any_column_name(
    monkeypatch: pytest.MonkeyPatch, database: str
) -> None:
    r"""P3-3: ``orders.pricee`` over a schema whose orders hold a column
    named with each character is refused with the reference's diagnostic
    for the twin's schema, naming the misspelt column."""
    if not reachable(database):
        return
    problems = []
    for vector in _vectors('typo', database):
        got = _on_the_core(monkeypatch, vector)
        if got != _wanted(vector):
            problems.append(f'{vector["id"]}: {got}')
    assert not problems, '\n'.join(problems)


@pytest.mark.parametrize('s0', [False, True], ids=['s0-default', 's0-core'])
@pytest.mark.parametrize('database', DATABASES)
def test_a_forecast_over_any_column_name_keeps_the_caller_s_call_limit(
    monkeypatch: pytest.MonkeyPatch, database: str, s0: bool
) -> None:
    r"""Codex's round 2 P2: ``FORECAST 2 TIMEFRAMES`` summing a column named
    with each character, ``sdm_client.decompose.MOST_PARTS`` set to 1, is
    refused as the twin's forecast is on the reference: two steps past the
    caller's limit of one, never two requests."""
    if not reachable(database):
        return
    problems = []
    for vector in _vectors('parts', database):
        got = _on_the_core(monkeypatch, vector, s0=s0)
        if got != _wanted(vector):
            problems.append(f'{vector["id"]}: {got}')
    assert not problems, '\n'.join(problems)
