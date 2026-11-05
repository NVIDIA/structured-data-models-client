# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Public decimal, time-column and seed boundaries on the Rust client."""

from __future__ import annotations

import importlib.util
import json
import math
import operator
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNERS = ROOT / 'contract/runners'
#: The anchor every case's materialize takes unless it says otherwise.
ANCHOR = pd.Timestamp('2024-03-01')
SUM = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
COUNT = 'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'
AVG = 'PREDICT AVG(orders.price, 0, 7, days) FOR EACH users.user_id'
STATIC = 'PREDICT users.age FOR EACH users.user_id'
#: DuckDB's largest seed, the ``max_seed`` its dialect declares (RS-7).
MAX_SEED = 2**31 - 1


def _module(name: str) -> ModuleType:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, RUNNERS / f'{name}.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


run_python = _module('run_python')

Edit = Callable[[Any], None]


def _call(
    setup: tuple[str, ...],
    query: str,
    *,
    edit: Edit | None = None,
    time_columns: dict[str, str] | None = None,
    anchor: pd.Timestamp | None = ANCHOR,
    **options: Any,
) -> dict[str, Any]:
    r"""The public outcome of one call on the route the environment
    chooses: the payload on a success; otherwise the call that raised, the
    error's class, code and details; and the missing-index warnings shown
    from the edit on."""
    from sdm_client import PredictiveQuery, Schema
    from sdm_connectors import DuckDBSource

    connection = duckdb.connect()
    for statement in [*run_python.load_bases()['shop'], *setup]:
        connection.execute(statement)
    stage = 'from_source'
    outcome: dict[str, Any]
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('ignore')
        try:
            arguments = (
                {} if time_columns is None else {'time_columns': time_columns}
            )
            schema = Schema.from_source(DuckDBSource(connection), **arguments)
            warnings.simplefilter('always')
            stage = 'edit'
            if edit is not None:
                edit(schema)
            stage = 'query'
            query_ = PredictiveQuery(query, schema)
            stage = 'materialize'
            given: dict[str, Any] = {'indices': [0, 1], **options}
            if anchor is not None:
                given['anchor_time'] = anchor
            request = query_.materialize(**given)
            outcome = {
                'stage': 'done',
                'payload': json.loads(
                    json.dumps(request._payloads[0], default=str)
                ),
            }
        except Exception as error:  # noqa: BLE001 - the outcome is the test's
            details = getattr(error, 'details', None)
            outcome = {
                'stage': stage,
                'class': type(error).__name__,
                'code': getattr(error, 'code', None),
                'details': dict(details) if isinstance(details, dict) else {},
                'message': str(error),
            }
    outcome['missing_index_warnings'] = sum(
        'Missing' in str(w.message) for w in shown
    )
    return outcome


def _current(*setup: str, query: str, **kwargs: Any) -> dict[str, Any]:
    return _call(setup, query, **kwargs)


def _refused(outcome: dict[str, Any], code: str) -> None:
    assert outcome['stage'] != 'done', outcome
    assert outcome['class'] == 'StructuredDataError', outcome
    assert outcome['code'] == code, outcome
    assert 'rule' not in outcome['details'], outcome


def _succeeded(outcome: dict[str, Any]) -> None:
    assert outcome['stage'] == 'done', outcome
    for part in ('context', 'predict'):
        assert outcome['payload'][part]['instance_table']['rows'], outcome


def _added(
    table: str, column: str, kind: str, expression: str
) -> tuple[str, str]:
    return (
        f'ALTER TABLE {table} ADD COLUMN {column} {kind}',
        f'UPDATE {table} SET {column} = {expression}',
    )


def _decimal(precision: int, scale: int) -> tuple[str, str]:
    return _added('users', 'd', f'DECIMAL({precision}, {scale})', 'user_id % 7')


# --- condition_value: a constant that overflows to an infinity -----------------


def _infinite_constants() -> list[Any]:
    cases = []
    for precision, scale in [
        (38, 0),
        (18, 2),
        (10, 4),
        (38, 10),
        (4, 0),
        (38, 37),
    ]:
        for condition in ['> 1e999', '<= -1e999']:
            cases.append((precision, scale, condition))
    for condition in ['= 1e999', '!= 1e999', '< 1e999', '>= 1e999', '< -1e999']:
        cases.append((38, 0, condition))
    for constant in ['1E999', '1.0e400', '-1.0e400']:
        cases.append((38, 0, f'> {constant}'))
    return [
        pytest.param(p, s, c, id=f'decimal-{p}-{s} {c}') for p, s, c in cases
    ]


#: Each comparison a condition writes, as Python reads it.
COMPARE = {
    '>': operator.gt,
    '>=': operator.ge,
    '<': operator.lt,
    '<=': operator.le,
    '=': operator.eq,
    '!=': operator.ne,
}


@pytest.mark.parametrize(
    ('precision', 'scale', 'condition'), _infinite_constants()
)
def test_an_infinite_constant_against_a_decimal_is_answered_by_the_reference(
    monkeypatch: pytest.MonkeyPatch, precision: int, scale: int, condition: str
) -> None:
    r"""Since main's eb6c68eb (the reference at 6298913c) the reference reads
    the decimal as float64 and materializes the condition: every label is
    the finite value compared with the infinity, as Python compares them.
    With exact decimals (and ) the core answers it too:
    a constant past the column's digits orders by its sign, so its payload,
    the labels included, is the reference's."""
    observed = _current(
        *_decimal(precision, scale),
        query=f'PREDICT users.d {condition} FOR EACH users.user_id',
    )
    op, constant = condition.split()
    infinity = float(constant)
    assert infinity in (math.inf, -math.inf), constant
    assert observed['stage'] == 'done', observed
    context = observed['payload']['context']['instance_table']
    labels = [
        row[context['columns'].index('TARGET')] for row in context['rows']
    ]
    assert labels, observed
    assert labels == [COMPARE[op](0.0, infinity)] * len(labels), labels
    _succeeded(observed)


@pytest.mark.parametrize(
    'query',
    [
        'PREDICT SUM(orders.price, 0, 30, days) > 1e999 FOR EACH users.user_id',
        'PREDICT SUM(orders.price, 0, 30, days) < -1e999 FOR EACH users.user_id',
        'PREDICT users.age > 1e999 FOR EACH users.user_id',
        'PREDICT users.d > 1.5 FOR EACH users.user_id',
    ],
)
def test_an_infinite_constant_elsewhere_materializes_alike(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    observed = _current(*_decimal(38, 0), query=query)
    _succeeded(observed)
    context = observed['payload']['context']['instance_table']
    target = context['columns'].index('TARGET')
    entity = context['columns'].index('__kumo_entity_ref_0')
    for row in context['rows']:
        expected = (
            int(row[entity]) % 7 > 1.5 if 'users.d > 1.5' in query else False
        )
        assert row[target] == expected, row


@pytest.mark.parametrize(
    'constant', ['nan', 'NaN', 'inf', '-inf', "'nan'", '1e999 - 1e999']
)
def test_a_constant_no_decimal_reads_is_refused_alike(
    monkeypatch: pytest.MonkeyPatch, constant: str
) -> None:
    # The PQL parser or validator refuses each, so no NaN reaches the
    # decimal's reading.
    observed = _current(
        *_decimal(38, 0),
        query=f'PREDICT users.d > {constant} FOR EACH users.user_id',
    )
    assert observed['stage'] == 'materialize', observed
    _refused(observed, 'INVALID_REQUEST')


def test_a_finite_constant_past_the_decimal_is_answered_by_the_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""1e308 is finite and past DECIMAL(38, 0). b413e9ad refused it
    INVALID_REQUEST, as the core does; since main's eb6c68eb (6298913c) the
    reference reads the decimal as float64 and materializes it, every label
    false (no value of the column is past 1e308). With exact decimals
     the core answers it too, the constant past the column's
    digits ordering by its sign: the reference's payload."""
    observed = _current(
        *_decimal(38, 0),
        query='PREDICT users.d > 1e308 FOR EACH users.user_id',
    )
    assert observed['stage'] == 'done', observed
    context = observed['payload']['context']['instance_table']
    labels = [
        row[context['columns'].index('TARGET')] for row in context['rows']
    ]
    assert labels and set(labels) == {False}, labels
    _succeeded(observed)


#: Past int64, under i128; past i128.
BIG = '99999999999999999999'
HUGE = '1' + '0' * 42


@pytest.mark.parametrize(
    'query',
    [
        f'PREDICT COUNT(orders.*, 0, 30, days) > {BIG} FOR EACH users.user_id',
        f'PREDICT COUNT(orders.*, 0, 30, days) < -{BIG} FOR EACH users.user_id',
        f'PREDICT COUNT(orders.*, 0, 30, days) > {HUGE} FOR EACH users.user_id',
        f'PREDICT SUM(orders.price, 0, 30, days) > {BIG} FOR EACH users.user_id',
        f'PREDICT users.age > {BIG} FOR EACH users.user_id',
        f'PREDICT users.d > {BIG} FOR EACH users.user_id',
        f'PREDICT users.d > {HUGE} FOR EACH users.user_id',
    ],
)
def test_an_integer_past_int64_is_the_pql_validator_s_refusal_on_both(
    monkeypatch: pytest.MonkeyPatch, query: str
) -> None:
    observed = _current(*_decimal(38, 0), query=query)
    assert observed['stage'] == 'materialize', observed
    _refused(observed, 'INVALID_REQUEST')
    assert 'outside of the range of supported integers' in observed['message']


# --- time_type: a time of day as the entity table's time column ----------------

#: A TIME column on users and on orders.
USERS_TIME = _added(
    'users', 't', 'TIME', "TIME '10:00:00' + to_minutes(user_id::INTEGER)"
)
ORDERS_TIME = _added(
    'orders', 't', 'TIME', "TIME '10:00:00' + to_minutes(order_id::INTEGER)"
)


def _time_column(table: str, column: str, stype: str | None = None) -> Edit:
    def edit(schema: Any) -> None:
        if stype is not None:
            schema[table][column].stype = stype
        schema[table].time_column = column

    return edit


#: Why the reference refuses, as measured: its message's start and its
#: details. Without enough history, its window check refuses before it
#: samples; otherwise it collects no example.
NO_EXAMPLES = ('[INVALID_REQUEST] Failed to collect any context examples', {})
WINDOW_OUTRAN = (
    '[INVALID_REQUEST] Context anchor timestamp is too early',
    {'window_outran_the_data': True},
)


def _entity_time_cases() -> list[Any]:
    timestamp = _time_column('users', 't', 'timestamp')
    cases: list[tuple[str, str, dict[str, Any], tuple[str, dict[str, Any]]]] = [
        ('edit-stype SUM', SUM, {'edit': timestamp}, NO_EXAMPLES),
        ('edit-stype COUNT', COUNT, {'edit': timestamp}, NO_EXAMPLES),
        ('edit-stype AVG', AVG, {'edit': timestamp}, NO_EXAMPLES),
        (
            'edit-stype SUM no-anchor',
            SUM,
            {'edit': timestamp, 'anchor': None},
            NO_EXAMPLES,
        ),
        (
            # The orders start 2023-01-01: a 30-day window back from
            # 2023-01-10 outruns them.
            'edit-stype SUM early-anchor',
            SUM,
            {'edit': timestamp, 'anchor': pd.Timestamp('2023-01-10')},
            WINDOW_OUTRAN,
        ),
        (
            'edit-time-column SUM',
            SUM,
            {'edit': _time_column('users', 't')},
            NO_EXAMPLES,
        ),
        ('from_source SUM', SUM, {'time_columns': {'users': 't'}}, NO_EXAMPLES),
        (
            'from_source COUNT',
            COUNT,
            {'time_columns': {'users': 't'}},
            NO_EXAMPLES,
        ),
        (
            'from_source SUM no-anchor',
            SUM,
            {'time_columns': {'users': 't'}, 'anchor': None},
            NO_EXAMPLES,
        ),
    ]
    return [
        pytest.param(query, kw, id=name) for name, query, kw, reason in cases
    ]


@pytest.mark.parametrize(('query', 'kwargs'), _entity_time_cases())
def test_a_time_of_day_entity_time_column_is_the_reference_s_invalid_request(
    monkeypatch: pytest.MonkeyPatch,
    query: str,
    kwargs: dict[str, Any],
) -> None:
    observed = _current(*USERS_TIME, query=query, **kwargs)
    assert observed['stage'] == 'materialize', observed
    _refused(observed, 'INVALID_REQUEST')


def _cast_failures() -> list[Any]:
    users = _time_column('users', 't', 'timestamp')
    orders = _time_column('orders', 't', 'timestamp')
    cases = [
        ('users static edit', USERS_TIME, STATIC, {'edit': users}),
        (
            'users static from_source',
            USERS_TIME,
            STATIC,
            {'time_columns': {'users': 't'}},
        ),
        ('orders static edit', ORDERS_TIME, STATIC, {'edit': orders}),
        ('orders SUM edit', ORDERS_TIME, SUM, {'edit': orders}),
        (
            'orders SUM from_source',
            ORDERS_TIME,
            SUM,
            {'time_columns': {'orders': 't'}},
        ),
    ]
    return [pytest.param(s, q, kw, id=name) for name, s, q, kw in cases]


@pytest.mark.parametrize(('setup', 'query', 'kwargs'), _cast_failures())
def test_time_of_day_cast_failure_preserves_the_refusal_shape(
    monkeypatch: pytest.MonkeyPatch,
    setup: tuple[str, ...],
    query: str,
    kwargs: dict[str, Any],
) -> None:
    # DuckDB cannot cast TIME to TIMESTAMP: the reference's engine fails
    # (INTERNAL_ERROR), the core reports the provider's error (DT-12).
    observed = _current(*setup, query=query, **kwargs)
    assert (observed['stage'], observed['code']) == (
        'materialize',
        'PROVIDER_ERROR',
    ), observed
    assert observed['details'].get('request') == 'time_bounds', observed


#: Time columns of other types: (kind, expression over a key, stype set).
OTHER_TIMES = [
    ('BIGINT', '1600000000 + {key} * 86400', 'timestamp'),
    ('DOUBLE', '1600000000.5 + {key} * 86400', 'timestamp'),
    ('DATE', "DATE '2022-01-01' + {key}::INTEGER", None),
    (
        'TIMESTAMPTZ',
        "TIMESTAMPTZ '2022-01-01 00:00:00+02' + to_days({key}::INTEGER)",
        None,
    ),
    ('TIMESTAMP_S', "TIMESTAMP '2022-01-01' + to_days({key}::INTEGER)", None),
    ('TIMESTAMP_NS', "TIMESTAMP '2022-01-01' + to_days({key}::INTEGER)", None),
    (
        'VARCHAR',
        "strftime(TIMESTAMP '2022-01-01' + to_days({key}::INTEGER), '%Y-%m-%d')",
        'timestamp',
    ),
]


@pytest.mark.parametrize('table', ['users', 'orders'])
@pytest.mark.parametrize(
    ('kind', 'expression', 'stype'),
    [pytest.param(*row, id=row[0]) for row in OTHER_TIMES],
)
def test_a_time_column_of_another_type_is_answered_alike(
    monkeypatch: pytest.MonkeyPatch,
    table: str,
    kind: str,
    expression: str,
    stype: str | None,
) -> None:
    key = 'user_id' if table == 'users' else 'order_id'
    observed = _current(
        *_added(table, 'x', kind, expression.format(key=key)),
        query=SUM,
        edit=_time_column(table, 'x', stype),
    )
    if kind in {'BIGINT', 'DOUBLE'}:
        # A number cannot mean a timestamp: refused at the edit.
        assert observed['stage'] == 'edit', observed
        _refused(observed, 'INVALID_REQUEST')
    else:
        _succeeded(observed)


# --- seed_size: a DuckDB seed of 10**38 or more ---------------------------------


@pytest.mark.parametrize(
    'seed',
    [
        pytest.param(seed, id=name)
        for name, seed in [
            ('2**31', 2**31),
            ('2**63', 2**63),
            ('2**64', 2**64),
            ('10**38-1', 10**38 - 1),
            ('10**38', 10**38),
            ('10**40', 10**40),
            ('10**100', 10**100),
        ]
    ],
)
def test_a_seed_duckdb_cannot_take_is_rs7_s_named_invalid_request(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    observed = _current(query=SUM, random_seed=seed)
    assert observed['stage'] == 'materialize', observed
    _refused(observed, 'INVALID_REQUEST')
    assert observed['details'].get('field') == 'seed', observed
    assert observed['details'].get('max_seed') == MAX_SEED, observed
    assert observed['details'].get('request') == 'sample_entities', observed
    assert observed['missing_index_warnings'] == 1


def test_duckdb_s_largest_seed_materializes_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _current(query=SUM, random_seed=MAX_SEED)
    _succeeded(observed)


@pytest.mark.parametrize('seed', [-1, -(2**31), -(10**40)])
def test_a_negative_seed_is_refused_alike(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    observed = _current(query=SUM, random_seed=seed)
    assert observed['stage'] == 'materialize', observed
    _refused(observed, 'INVALID_REQUEST')
    assert observed['details'].get('field') == 'seed', observed
