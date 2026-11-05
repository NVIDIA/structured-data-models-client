# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query diagnostics stay consistent for ordinary and unusual column names.

The same public scenarios run on DuckDB, SQLite and optional PostgreSQL.
Expected refusals are retained regression observations, not computed by the
implementation under test.
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
    'column_name_scenarios',
    ROOT / 'contract/support/column_name_scenarios.py',
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


def _wanted(vector: dict[str, Any]) -> dict[str, Any]:
    """Use the ordinary-name refusal with this scenario's query text."""
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


def test_scenarios_cover_column_names_and_expected_refusals() -> None:
    """Each named scenario keeps a concrete, independent refusal oracle."""
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
    database: str,
) -> None:
    """A relational query rejects the direct-input task override."""
    if not reachable(database):
        pytest.skip('SDM_POSTGRES_TEST_DSN is not configured')
    problems = []
    for vector in _vectors('task', database):
        got = R.run(vector)
        if got != _wanted(vector):
            problems.append(f'{vector["id"]}: {got}')
    assert not problems, '\n'.join(problems)


@pytest.mark.parametrize('database', DATABASES)
def test_a_misspelt_column_keeps_its_diagnostic_beside_any_column_name(
    database: str,
) -> None:
    """A misspelt column is diagnosed independently of other column names."""
    if not reachable(database):
        pytest.skip('SDM_POSTGRES_TEST_DSN is not configured')
    problems = []
    for vector in _vectors('typo', database):
        got = R.run(vector)
        if got != _wanted(vector):
            problems.append(f'{vector["id"]}: {got}')
    assert not problems, '\n'.join(problems)


@pytest.mark.parametrize('database', DATABASES)
def test_a_forecast_over_any_column_name_keeps_the_caller_s_call_limit(
    database: str,
) -> None:
    """A two-step forecast respects a caller limit of one model call."""
    if not reachable(database):
        pytest.skip('SDM_POSTGRES_TEST_DSN is not configured')
    problems = []
    for vector in _vectors('parts', database):
        got = R.run(vector)
        if got != _wanted(vector):
            problems.append(f'{vector["id"]}: {got}')
    assert not problems, '\n'.join(problems)
