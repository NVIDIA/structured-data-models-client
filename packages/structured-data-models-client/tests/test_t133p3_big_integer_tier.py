# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.33 part 3: a big-integer column keeps the tier it was described in,
on the core's route over real DuckDB (fix round 1, written by the package's
independent test author).

The orchestrator's rule (2026-10-14): a HUGEINT, UHUGEINT or VARINT column
is ``int64`` where every value fits it, else ``decimal128(38, 0)``, else
text, decided for the whole column at describe, never per fetch; a later
fetch wider than the described tier is refused VALUE_OUT_OF_RANGE naming
the column. Codex's review of 08c531b5 exercised these two by hand and asked
for them committed:

- a lossless HUGEINT described as ``int64``, then widened past it before
  materialize, is refused VALUE_OUT_OF_RANGE naming the column;
- on the 1,200-row witness whose wide values come after the 1,000 rows a
  preview reads (the type set's ``lossless.hugeint-wide-after-preview``),
  a ``column_ranges`` the source fails at describe leaves the preview's
  tier standing, and materialize, reaching the wide rows, is refused
  VALUE_OUT_OF_RANGE naming the column, never handed a changed tier.

Both passed at 08c531b5; they hold the behaviour through the fix.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import (
    PredictiveQuery,
    Request,
    Schema,
    StructuredDataError,
    _routing,
)
from sdm_client._rust import schema as _rust_schema

ROOT = Path(__file__).resolve().parents[3]
CORE = ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB)
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
ANCHOR = pd.Timestamp('2024-03-01')
LOSSLESS = 'SET arrow_lossless_conversion = true'
WIDE = '1000000000000000000000000000000'


@pytest.fixture(autouse=True)
def _core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)


def _shop() -> list[str]:
    r"""The ``shop`` base of ``contract/bases.yaml``: the plain list items
    under its name."""
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


def _type_set_witness(row_id: str) -> list[str]:
    loaded = json.loads(
        (ROOT / 'contract/type-set-vectors.json').read_text(encoding='utf-8')
    )
    [row] = [
        r for r in loaded['vectors'] + loaded['reached'] if r['id'] == row_id
    ]
    witness: list[str] = row['witness']
    return witness


def _connection(setup: list[str]) -> Any:
    connection = duckdb.connect()
    for statement in [*_shop(), *setup]:
        connection.execute(statement)
    return connection


def _described(connection: Any) -> Schema:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(DuckDBSource(connection))


def _materialized(schema: Schema) -> Request:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1], anchor_time=ANCHOR
        )


def _refused_out_of_range(schema: Schema, column: str) -> None:
    with pytest.raises(StructuredDataError) as refused:
        _materialized(schema)
    assert refused.value.code == 'VALUE_OUT_OF_RANGE', refused.value
    details = getattr(refused.value, 'details', None) or {}
    assert details.get('column') == column, details


def test_a_hugeint_widened_after_describe_is_refused_by_name() -> None:
    connection = _connection(
        [
            LOSSLESS,
            'CREATE OR REPLACE TABLE users AS SELECT *, '
            'user_id::HUGEINT * 1000 AS f FROM users',
        ]
    )
    schema = _described(connection)
    # Control: as described, every value fits int64, and materialize reads.
    assert _materialized(schema)._payloads
    connection.execute(
        f'UPDATE users SET f = {WIDE}::HUGEINT WHERE user_id IN (0, 1)'
    )
    _refused_out_of_range(schema, 'f')


class _FailingRanges(_rust_schema.DuckDBProvider):
    r"""The DuckDB provider, failing every ``column_ranges`` it is asked."""

    asked: list[dict[str, Any]] = []

    def __call__(self, step: dict[str, Any], keys: Any = None) -> Any:
        if step['request']['request'] == 'column_ranges':
            _FailingRanges.asked.append(step['request'])
            raise RuntimeError('column_ranges failed (injected)')
        return super().__call__(step, keys)


def test_a_failed_column_ranges_leaves_the_preview_s_tier_and_refuses_later(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    witness = _type_set_witness('lossless.hugeint-wide-after-preview')
    # Control: the source answering, the whole column is decimal128(38, 0)
    # and materialize reads the wide rows.
    assert _materialized(_described(_connection(witness)))._payloads
    _FailingRanges.asked = []
    monkeypatch.setattr(_rust_schema, 'DuckDBProvider', _FailingRanges)
    schema = _described(_connection(witness))
    assert any('f' in r.get('columns', []) for r in _FailingRanges.asked), (
        _FailingRanges.asked
    )
    _refused_out_of_range(schema, 'f')
