# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P3.replay, for the warehouses the corpus can't record offline.

Databricks (through a SQL warehouse connection, a notebook's Spark session
and a metric view) and Snowflake (directly and through a semantic view) have
no recordings yet. Their SQL is pinned instead by ``golden/
warehouse_sql.json``, written by ``_warehouse.py write-golden`` at the commit
phase 3 started from: for each, a prediction over stub drivers, with every
statement's text and parameters in order, the payloads and the warnings.
Phase 5.5 wrote it again: window bounds gained ``+00:00`` on Databricks, and
both warehouses read a table's size before its inference sample
(``DESCRIBE DETAIL``, and ``SHOW TABLES`` earlier and once); every other
statement and every payload stayed as phase 3 pinned them.

The code must send the same statements, byte for byte, in the same order,
and build parsed-equal payloads with the same warnings.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from _seam import strict

pytest.importorskip('sdm_relational_engine.relationallib')
pytest.importorskip('databricks.sql')
pytest.importorskip('snowflake.connector')
pytest.importorskip('yaml')

import _warehouse  # noqa: E402


def _first_difference(left: Any, right: Any, path: str = '') -> str | None:
    if type(left) is not type(right):
        return f'{path or "<root>"}: {left!r:.200} != {right!r:.200}'
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return f'{path}.{key}: present on one side only'
            found = _first_difference(left[key], right[key], f'{path}.{key}')
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return f'{path}: {len(left)} items != {len(right)} items'
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            found = _first_difference(a, b, f'{path}[{i}]')
            if found:
                return found
        return None
    if left != right:
        return f'{path}: {left!r:.300} != {right!r:.300}'
    return None


@pytest.fixture(scope='module')
def golden() -> dict[str, Any]:
    return json.loads(_warehouse.GOLDEN.read_text())


def test_the_golden_covers_every_warehouse(golden: dict[str, Any]) -> None:
    assert sorted(golden) == sorted(_warehouse.WAREHOUSES)
    for name, pinned in golden.items():
        assert len(pinned['statements']) >= 15, name
        for question in pinned['questions'].values():
            assert question['payloads'], name


@pytest.mark.parametrize('name', _warehouse.WAREHOUSES)
def test_the_same_statements_and_payloads(
    golden: dict[str, Any], name: str
) -> None:
    now = json.loads(json.dumps(_warehouse.as_golden(_warehouse.run(name))))
    pinned = golden[name]

    sent = [s['sql'] for s in now['statements']]
    expected = [s['sql'] for s in pinned['statements']]
    for index, (a, b) in enumerate(zip(sent, expected, strict=False)):
        assert a == b, (
            f'{name}: statement {index} differs\n--- now\n{a}\n--- pinned\n{b}'
        )
    assert len(sent) == len(expected), (
        f'{name}: {len(sent)} statements sent, {len(expected)} pinned'
    )
    for index, (a, b) in enumerate(
        zip(now['statements'], pinned['statements'], strict=True)
    ):
        assert strict(a['params']) == strict(b['params']), (
            f'{name}: statement {index} has other parameters\n'
            f'{a["params"]!r:.400}\n!=\n{b["params"]!r:.400}'
        )
    difference = _first_difference(
        {'questions': now['questions'], 'warnings': now['schema_warnings']},
        {
            'questions': pinned['questions'],
            'warnings': pinned['schema_warnings'],
        },
    )
    assert difference is None, f'{name}: {difference}'
