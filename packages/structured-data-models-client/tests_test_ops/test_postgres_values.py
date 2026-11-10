# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PostgreSQL values remain exact through provider transport and core admission.

Authored SQL inputs define expected JSON text, numeric precision and interval behavior.
Seed bindings preserve subsecond and pre-epoch timestamps.
Run against SDM_POSTGRES_TEST_DSN with the native test-ops build.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import sys
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[3]
PROVIDER_FILE = (
    'packages/structured-data-models-client/src/sdm_client/_rust/providers/'
    'postgres.py'
)
#: The provider's value rules at ledger (row P2-3), each declared
#: moved naming once the rule lives in the core.
RULES = (
    'json_text',
    '_interval',
    '_Numeric.__init__',
    '_Numeric.field',
    'key_set_json',
    '_isoformat',
)
# Independently authored values of the SQL inputs, plus the inserted null row.
DRIVER = {
    'iv': [
        dt.timedelta(days=32, seconds=3),
        dt.timedelta(days=-365),
        dt.timedelta(0),
        None,
    ],
    'js': [True, False, 3, 'x', {'a': 1}, None, None],
    'jn': [True, 2.5, [1, 'b'], None],
    'nw': [Decimal(2**63), Decimal(1), Decimal(-(2**63) - 1), None],
    'nf': [
        Decimal('1.500'),
        Decimal('-2.250'),
        Decimal('12345678901234567.125'),
        None,
    ],
    'nx': [
        Decimal('12345678901234567890.123'),
        Decimal(1),
        Decimal('0.1'),
        None,
    ],
}
COLUMNS = tuple(sorted(DRIVER))
PREFIX = 'pg-values:'
KEYS = (
    *(f'{PREFIX}core:{c}' for c in COLUMNS),
    *(f'{PREFIX}driver-rows:{c}' for c in COLUMNS),
    f'{PREFIX}seed-times',
    *(f'{PREFIX}moved:{rule}' for rule in RULES),
)


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def recorder() -> ModuleType:
    return _module(
        'test_postgres_values_postgres_values_record',
        ROOT / 'contract/support/postgres_value_scenarios.py',
    )


def test_source_cases_cover_each_value_rule() -> None:
    assert set(COLUMNS) == set(recorder().COLUMNS)
    for name in COLUMNS:
        assert len(DRIVER[name]) == len(recorder().COLUMNS[name]['values']) + 1


# --- the database --------------------------------------------------------------


@pytest.fixture(scope='module')
def database() -> Iterator[Any]:
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
        yield None
        return
    made = recorder().Database(dsn)
    try:
        yield made
    finally:
        made.drop()


class _Answers:
    r"""The Postgres provider, keeping each answer it hands the core and
    each parameter bound to a statement."""

    def __init__(self, connection: Any) -> None:
        from sdm_client._rust.providers.postgres import PostgresProvider

        self.bound: list[Any] = []
        self.answers: list[pa.Table] = []
        self._provider = PostgresProvider(_Bound(connection, self.bound))

    def __call__(self, step: dict[str, Any], keys: Any = None) -> Any:
        answer = self._provider(step, keys)
        table = pa.RecordBatchReader._import_from_c_capsule(
            answer.__arrow_c_stream__()
        ).read_all()
        self.answers.append(table)
        return table


class _Bound:
    r"""A psycopg connection whose cursors keep each parameter bound."""

    def __init__(self, connection: Any, bound: list[Any]) -> None:
        self._connection = connection
        self._bound = bound

    def cursor(self, *args: Any, **kwargs: Any) -> Any:
        return _BoundCursor(
            self._connection.cursor(*args, **kwargs), self._bound
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


class _BoundCursor:
    def __init__(self, cursor: Any, bound: list[Any]) -> None:
        self._cursor = cursor
        self._bound = bound

    def __enter__(self) -> _BoundCursor:
        self._cursor.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._cursor.__exit__(*exc)

    def execute(self, query: Any, params: Any = None, **kwargs: Any) -> Any:
        if params is not None:
            self._bound.append(params)
            return self._cursor.execute(query, params, **kwargs)
        return self._cursor.execute(query, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cursor, name)


def _read(database: Any, column: str) -> tuple[Any, _Answers]:
    r"""``column``'s table read through the provider and admitted by the
    core, and what the provider handed over."""
    with database.connect() as connection:
        answers = _Answers(connection)
        return recorder().admitted(answers, column), answers


@pytest.mark.parametrize('column', COLUMNS)
def test_admitted_values_preserve_declared_conversion_rules(
    database: Any, column: str
) -> None:
    if database is None:
        pytest.skip('PostgreSQL value checks require a test DSN')
    admitted, _ = _read(database, column)
    if column == 'iv':
        assert admitted == 'refused UNSUPPORTED_TYPE'
        return
    assert not isinstance(admitted, str)
    expected_type = (
        'double'
        if column == 'nf'
        else 'string arrow.json'
        if column in ('js', 'jn')
        else 'string'
    )
    assert recorder()._arrow_type(admitted.schema.field('v')) == expected_type
    assert admitted.column('id').to_pylist() == list(
        range(1, len(DRIVER[column]) + 1)
    )
    values = admitted.column('v').to_pylist()
    if column in ('js', 'jn'):
        expected = {
            'js': ['True', 'False', '3', 'x', '{"a":1}', None, None],
            'jn': ['True', '2.5', '[1,"b"]', None],
        }[column]
        assert values == expected
        assert all(v is None or type(v) is str for v in values)
    elif column == 'nf':
        assert all(v is None or type(v) is float for v in values)
        assert values == [
            None if v is None else float(v) for v in DRIVER[column]
        ]
    else:
        assert all(v is None or type(v) is str for v in values)
        assert [None if v is None else Decimal(v) for v in values] == DRIVER[
            column
        ]
        if column == 'nx':
            assert [None if v is None else float(v) for v in values] == [
                None if v is None else float(v) for v in DRIVER[column]
            ]


def test_seed_bindings_preserve_exact_key_and_timestamp_text(
    database: Any,
) -> None:
    """Seed timestamp text preserves whole seconds, microseconds and nanoseconds."""
    if database is None:
        pytest.skip('PostgreSQL value checks require a test DSN')
    from sdm_client._rust import codec
    from sdm_client._rust.run import run

    seeds = recorder().SEEDS
    table = pa.table(
        {
            'key': pa.array([key for key, _ in seeds], pa.int64()),
            'batch': pa.array(range(len(seeds)), pa.int64()),
            'end': pa.array([end for _, end in seeds], pa.timestamp('ns')),
        }
    )
    fetch = {
        'request': {
            'request': 'neighbors',
            'table': {'source': ['events']},
            'key': ['uid'],
            'columns': ['v'],
            'seeds': {
                'attachment': 0,
                'columns': [
                    {
                        'name': 'key',
                        'type': {'type': 'int64'},
                        'nullable': False,
                    },
                    {
                        'name': 'batch',
                        'type': {'type': 'int64'},
                        'nullable': False,
                    },
                    {
                        'name': 'end',
                        'type': {
                            'type': 'timestamp',
                            'unit': 'ns',
                            'zone': None,
                        },
                        'nullable': True,
                    },
                ],
            },
            'window': {
                'column': 'at',
                'lower': None,
                'upper': {'seed_column': 'end', 'inclusive': True},
            },
            'per_seed': None,
            'order': None,
        },
        # Under the database's schema, as every Postgres read names one
        # .
        'provider': {
            'provider': 'sql',
            'dialect': 'postgres',
            'namespace': ['public'],
        },
        'echo': True,
    }

    def check() -> None:
        with database.connect() as connection:
            answers = _Answers(connection)
            finished = run(
                {'op': 'test', 'test': 'keyed', 'fetch': 1, 'keys': 2},
                {1: codec.attachment(fetch), 2: table},
                provider=answers,
            )
            finished.take(finished.result['data'])
        texts = [
            json.loads(value)
            for params in answers.bound
            for value in params
            if isinstance(value, str)
        ]
        assert len(texts) == 1, answers.bound
        # Bound JSON retains exact key values and timestamp precision.
        assert [[str(row['id']), row['e']] for row in texts[0]] == (
            [
                ['1', '2024-01-02T03:04:05'],
                ['1', '2024-01-02T03:04:05.500000'],
                ['2', '2024-01-03T00:00:00.000000001'],
                ['2', '1969-12-31T00:00:00.000001'],
            ]
        )

    check()


# --- the provider only moves the driver's rows -------------------------------


def _faithful(field: pa.Field, value: Any, expected: Any) -> bool:
    """Check provider values directly against authored SQL-input values."""
    if value is None or expected is None:
        return value is None and expected is None
    kind = (
        field.type.storage_type
        if isinstance(field.type, pa.BaseExtensionType)
        else field.type
    )
    if recorder()._arrow_type(field).endswith(' arrow.json'):
        decoded = json.loads(value)
        return type(decoded) is type(expected) and decoded == expected
    if pa.types.is_interval(kind):
        return (
            value.months == 0
            and dt.timedelta(
                days=value.days, microseconds=value.nanoseconds // 1000
            )
            == expected
            and value.nanoseconds % 1000 == 0
        )
    if isinstance(expected, Decimal):
        return (
            Decimal(value) if isinstance(value, float) else Decimal(str(value))
        ) == expected
    return type(value) is type(expected) and value == expected


@pytest.mark.parametrize('column', COLUMNS)
def test_the_provider_hands_the_core_the_driver_s_values(
    database: Any, column: str
) -> None:
    """Raw provider Arrow cells preserve the independently authored SQL values."""
    if database is None:
        pytest.skip('PostgreSQL value checks require a test DSN')
    driver = DRIVER[column]

    def check() -> None:
        _, answers = _read(database, column)
        assert answers.answers, 'the provider answered nothing'
        answer = answers.answers[-1]
        field = answer.schema.field('v')
        # Rows put in id order in Python: Table.sort_by on PyArrow 14 (the
        # floor) turns a month_day_nano_interval column's cells into nulls,
        # so the Arrow table holding the interval is never sorted.
        by_id = dict(
            zip(
                answer.column('id').to_pylist(),
                answer.column('v').to_pylist(),
                strict=True,
            )
        )
        values = [by_id[key] for key in sorted(by_id)]
        unfaithful = [
            (value, cell)
            for value, cell in zip(values, driver, strict=True)
            if not _faithful(field, value, cell)
        ]
        assert not unfaithful, f'{field}: {unfaithful}'

    check()
