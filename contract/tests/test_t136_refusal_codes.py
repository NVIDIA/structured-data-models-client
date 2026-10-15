# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.36 fix group 9: three refusals the core gives under a code the
reference does not, held to the reference's code at the public route.

Written by the package's independent test author, before the fix.

    python -m pytest contract/tests/test_t136_refusal_codes.py -q

Each case builds the ``shop`` base of ``contract/bases.yaml`` with a few
statements of its own in DuckDB, runs ``Schema.from_source``, an optional
schema edit, ``PredictiveQuery`` and ``materialize`` on the reference's
route (no ``SDM_CORE_ROUTES``) and then on the core's DuckDB routes, in this
process, and compares the two outcomes: the call that raised (the stage),
the error's class and ``code``, and for a success the whole payload, as
the client's ``test_the_materialize_route.py`` compares it. The reference's
outcome is the one this run measures, never one written here by hand; each
case also states the reference's code, so a reference that moved fails
by name rather than silently changing what the core is held to.

- ``condition_value`` (``sdm-engine``'s ``decimal_constant``): a static
  DECIMAL column compared with a float constant that overflows to an
  infinity (``1e999``, ``-1e999``, ``1E999``, ``1.0e400``), at several
  precisions and scales and with every comparison. The reference refuses
  it with INVALID_REQUEST at materialize (pyarrow's "Cannot convert inf to
  Decimal128"); the core refused it as UNSUPPORTED_REQUEST naming the rule.
  Held: the core's refusal is the reference's, INVALID_REQUEST at
  materialize, with no rule. Controls, alike on both routes before and
  after: a float sum compared with ``1e999`` or ``-1e999`` and a static
  integer compared with ``1e999`` materialize the same payload; a finite
  constant a DECIMAL does not hold (``1e308``) is INVALID_REQUEST; ``nan``,
  ``inf``, ``-inf`` and ``'nan'`` are refused by the PQL parser; an integer
  past int64 or past i128 is refused by the PQL validator with the same
  message on both routes. The parser takes no constant that is NaN, so no
  NaN reaches ``decimal_constant`` from a query.
- ``time_type`` (``sdm-core``'s ``times_ns``): a TIME (time of day) column
  as the entity table's time column, set by ``from_source(time_columns=)``
  or by an edit (with or without setting its stype to ``timestamp``), under
  any temporal query, with or without an anchor. The reference refuses it
  with INVALID_REQUEST at materialize ("Failed to collect any context
  examples"); the core refused it as UNSUPPORTED_REQUEST naming the rule.
  Held: INVALID_REQUEST at materialize, with no rule. Controls that stay as
  they are: a static query over such an entity table, or a TIME time
  column on a table that is not the entity's (the reference INTERNAL_ERROR,
  DuckDB's failed cast; the core PROVIDER_ERROR, register DT-12's shape);
  a BIGINT or DOUBLE column set to ``timestamp`` (INVALID_REQUEST at the
  edit on both); DATE, TIMESTAMPTZ, TIMESTAMP_S, TIMESTAMP_NS and text time
  columns on either table (the same payload on both).
- ``seed_size`` (``sdm-core``'s ``seed_scalar``): a DuckDB ``random_seed`` of
  10**38 or more. The reference fails it as it fails every seed from
  2**31, DuckDB's ParserException reported as INTERNAL_ERROR; the core
  refused 2**31 to 10**38 - 1 with INVALID_REQUEST naming the seed and
  ``max_seed`` (register RS-7, approved by the owner), but 10**38 and more as
  UNSUPPORTED_REQUEST naming ``seed_size``. Held: RS-7's refusal for every
  seed from 2**31 (INVALID_REQUEST at materialize, ``field`` ``seed``,
  ``max_seed`` 2**31 - 1, ``request`` ``sample_entities``, no rule), after
  the index read as the reference's failure is, so the refused call shows
  the reference's missing-index warnings. Controls: 2**31 - 1 materializes
  the same payload on both; a negative seed is INVALID_REQUEST at
  materialize on both.

Failing when written (c49ee4f8): every ``condition_value`` and
``time_type`` target, and the seeds 10**38, 10**40 and 10**100; every
control and the seeds 2**31 to 10**38 - 1 pass.
"""

from __future__ import annotations

import importlib.util
import json
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


def _routes(
    monkeypatch: pytest.MonkeyPatch, *setup: str, query: str, **kwargs: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    r"""The reference's outcome, then the core's, for the same call."""
    from sdm_client import _routing

    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    reference = _call(setup, query, **kwargs)
    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB),
    )
    core = _call(setup, query, **kwargs)
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    return reference, core


def _shape(outcome: dict[str, Any]) -> tuple[Any, ...]:
    return (outcome['stage'], outcome.get('class'), outcome.get('code'))


def _refused_alike(
    reference: dict[str, Any], core: dict[str, Any], code: str
) -> None:
    r"""The reference refused with ``code`` (as measured), and the core
    refused at the same stage, with the same class and code, by no rule."""
    assert reference['stage'] != 'done', reference
    assert reference['code'] == code, reference
    assert _shape(core) == _shape(reference), (
        f'core {_shape(core)} rule {core["details"].get("rule")!r}, '
        f'the reference {_shape(reference)}'
    )
    assert 'rule' not in core['details'], core


def _succeeded_alike(reference: dict[str, Any], core: dict[str, Any]) -> None:
    assert reference['stage'] == 'done', reference
    assert core['stage'] == 'done', core
    assert core['payload'] == reference['payload']


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


@pytest.mark.parametrize(
    ('precision', 'scale', 'condition'), _infinite_constants()
)
def test_an_infinite_constant_against_a_decimal_is_the_reference_s_invalid_request(
    monkeypatch: pytest.MonkeyPatch, precision: int, scale: int, condition: str
) -> None:
    reference, core = _routes(
        monkeypatch,
        *_decimal(precision, scale),
        query=f'PREDICT users.d {condition} FOR EACH users.user_id',
    )
    assert reference['stage'] == 'materialize', reference
    assert 'Cannot convert' in reference['message'], reference
    _refused_alike(reference, core, 'INVALID_REQUEST')


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
    reference, core = _routes(monkeypatch, *_decimal(38, 0), query=query)
    _succeeded_alike(reference, core)


@pytest.mark.parametrize(
    'constant', ['1e308', 'nan', 'NaN', 'inf', '-inf', "'nan'", '1e999 - 1e999']
)
def test_a_constant_no_decimal_reads_is_refused_alike(
    monkeypatch: pytest.MonkeyPatch, constant: str
) -> None:
    # 1e308 is finite and past DECIMAL(38, 0); the others the PQL parser
    # or validator refuses, so no NaN reaches the decimal's reading.
    reference, core = _routes(
        monkeypatch,
        *_decimal(38, 0),
        query=f'PREDICT users.d > {constant} FOR EACH users.user_id',
    )
    _refused_alike(reference, core, 'INVALID_REQUEST')


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
    reference, core = _routes(monkeypatch, *_decimal(38, 0), query=query)
    _refused_alike(reference, core, 'INVALID_REQUEST')
    assert 'outside of the range of supported integers' in reference['message']
    assert core['message'] == reference['message']


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


def _entity_time_cases() -> list[Any]:
    timestamp = _time_column('users', 't', 'timestamp')
    cases: list[tuple[str, str, dict[str, Any]]] = [
        ('edit-stype SUM', SUM, {'edit': timestamp}),
        ('edit-stype COUNT', COUNT, {'edit': timestamp}),
        ('edit-stype AVG', AVG, {'edit': timestamp}),
        ('edit-stype SUM no-anchor', SUM, {'edit': timestamp, 'anchor': None}),
        (
            'edit-stype SUM early-anchor',
            SUM,
            {'edit': timestamp, 'anchor': pd.Timestamp('2023-01-10')},
        ),
        ('edit-time-column SUM', SUM, {'edit': _time_column('users', 't')}),
        ('from_source SUM', SUM, {'time_columns': {'users': 't'}}),
        ('from_source COUNT', COUNT, {'time_columns': {'users': 't'}}),
        (
            'from_source SUM no-anchor',
            SUM,
            {'time_columns': {'users': 't'}, 'anchor': None},
        ),
    ]
    return [pytest.param(query, kw, id=name) for name, query, kw in cases]


@pytest.mark.parametrize(('query', 'kwargs'), _entity_time_cases())
def test_a_time_of_day_entity_time_column_is_the_reference_s_invalid_request(
    monkeypatch: pytest.MonkeyPatch, query: str, kwargs: dict[str, Any]
) -> None:
    reference, core = _routes(monkeypatch, *USERS_TIME, query=query, **kwargs)
    assert reference['stage'] == 'materialize', reference
    assert 'Failed to collect any context examples' in reference['message']
    _refused_alike(reference, core, 'INVALID_REQUEST')


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
def test_a_time_of_day_the_time_bounds_cast_fails_on_stays_dt12_s_shape(
    monkeypatch: pytest.MonkeyPatch,
    setup: tuple[str, ...],
    query: str,
    kwargs: dict[str, Any],
) -> None:
    # DuckDB cannot cast TIME to TIMESTAMP: the reference's engine fails
    # (INTERNAL_ERROR), the core reports the provider's error (DT-12).
    reference, core = _routes(monkeypatch, *setup, query=query, **kwargs)
    assert (reference['stage'], reference['code']) == (
        'materialize',
        'INTERNAL_ERROR',
    ), reference
    assert (core['stage'], core['code']) == ('materialize', 'PROVIDER_ERROR'), (
        core
    )
    assert core['details'].get('request') == 'time_bounds', core


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
    reference, core = _routes(
        monkeypatch,
        *_added(table, 'x', kind, expression.format(key=key)),
        query=SUM,
        edit=_time_column(table, 'x', stype),
    )
    if kind in {'BIGINT', 'DOUBLE'}:
        # A number cannot mean a timestamp: refused at the edit.
        assert reference['stage'] == 'edit', reference
        _refused_alike(reference, core, 'INVALID_REQUEST')
    else:
        _succeeded_alike(reference, core)


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
    reference, core = _routes(monkeypatch, query=SUM, random_seed=seed)
    # The reference: DuckDB's parser refuses the seed, reported as
    # INTERNAL_ERROR at materialize; RS-7 names INVALID_REQUEST instead.
    assert (reference['stage'], reference['code']) == (
        'materialize',
        'INTERNAL_ERROR',
    ), reference
    assert 'ParserException' in reference['message'], reference
    assert _shape(core) == (
        'materialize',
        'StructuredDataError',
        'INVALID_REQUEST',
    ), f'core {_shape(core)} rule {core["details"].get("rule")!r}'
    assert 'rule' not in core['details'], core
    assert core['details'].get('field') == 'seed', core
    assert core['details'].get('max_seed') == MAX_SEED, core
    assert core['details'].get('request') == 'sample_entities', core
    # Refused after the index read, as the reference fails after it.
    assert (
        core['missing_index_warnings']
        == reference['missing_index_warnings']
        == 1
    )


def test_duckdb_s_largest_seed_materializes_alike(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference, core = _routes(monkeypatch, query=SUM, random_seed=MAX_SEED)
    _succeeded_alike(reference, core)


@pytest.mark.parametrize('seed', [-1, -(2**31), -(10**40)])
def test_a_negative_seed_is_refused_alike(
    monkeypatch: pytest.MonkeyPatch, seed: int
) -> None:
    reference, core = _routes(monkeypatch, query=SUM, random_seed=seed)
    assert reference['stage'] == 'materialize', reference
    _refused_alike(reference, core, 'INVALID_REQUEST')
    assert core['details'].get('field') == 'seed', core
