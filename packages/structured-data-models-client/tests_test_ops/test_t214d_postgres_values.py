# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T2.14d's exits (the Postgres value rules, MID2 P2-3, riding along in
T2.14b's landing), by T2.14b's test author, before the implementation.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        packages/structured-data-models-client/tests_test_ops/test_t214d_postgres_values.py

Run on the test-ops build (the test-only fetch), as CI's
``native-test-ops`` job runs ``tests_test_ops`` beside its pinned Postgres.
The design: the Postgres value rules move from the Python provider
(``_rust/providers/postgres.py``) into ``sdm-data``'s Postgres dialect and
``sdm-core``; the provider only moves the driver's rows into Arrow. The
expectations are ``contract/l5-vectors.json``'s ``postgres_values``,
recorded by ``contract/tools/postgres_values_record.py``: the reference's
driver's reading and the reference's own reading of each value (python-ref
at ``contract/reference-env.txt``'s commit), the Arrow type the core admits
each column as, and the text the reference binds each seed's key and time
as.

- What the core takes is the reference's (a guard: the move changes no
  answer): each column's admitted Arrow type is the recorded one, and its
  values are the reference's reading of them; a JSON scalar as Python's
  text of it (``true`` as ``True``), a fractional ``numeric`` past 76
  digits as its double, one with no precision as its exact text (read as
  the reference's double where it is not a key), a whole one past ``int64``
  as its exact digits (where the reference overflows), and an interval
  refused by name as the reference leaves it out.
- Each seed's key and time are bound as the reference writes them (a
  guard): a seed time as pandas' ``Timestamp.isoformat``, nanoseconds and a
  time before 1970 included.
- The provider only moves the driver's rows into Arrow: what it hands the
  core for each column is, value by value, the driver's reading, read as
  the comparator reads a statement's result (a JSON column's text as the
  JSON value it holds, an interval as the ``timedelta`` the driver gave);
  it loses nothing and adds no spelling of its own.
- Each rule of the provider is declared ``moved`` naming T2.14d in the
  bridge freeze's declarations (``scripts/rust_core/bridge_changes.json``).

Gating: ``contract/tests/t214b-pending.json``'s ``pending`` names, as
``pg-values:<test>:<column or rule>``, the keys whose test failed when
written. Each must still fail (strict); once one passes it fails, naming
itself, until it is taken out. Where ``SDM_POSTGRES_TEST_DSN`` names no
server and ``SDM_REQUIRE_POSTGRES`` is not set, the tests that need a
server return without asserting; with it set, a missing server fails (H26).
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
import struct
import sys
from collections.abc import Callable, Iterator
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[3]
PENDING = ROOT / 'contract/tests/t214b-pending.json'
VECTORS = json.loads(
    (ROOT / 'contract/l5-vectors.json').read_text(encoding='utf-8')
)['postgres_values']
PROVIDER_FILE = (
    'packages/structured-data-models-client/src/sdm_client/_rust/providers/'
    'postgres.py'
)
#: The provider's value rules at T2.14a's ledger (row P2-3), each declared
#: moved naming T2.14d once the rule lives in the core.
RULES = (
    'json_text',
    '_interval',
    '_Numeric.__init__',
    '_Numeric.field',
    'key_set_json',
    '_isoformat',
)
#: How the core's admitted value of each column is read against the
#: reference's: as the text it is (a JSON column, the reference's text of
#: the value), as the value it is (a double), as an exact text read as the
#: reference's double (a numeric with no precision, not a key), as exact
#: digits held to the driver's value where the reference overflows.
READ = {
    'js': 'text',
    'jn': 'text',
    'nf': 'value',
    'nx': 'exact text, a double',
    'nw': 'exact digits',
}
COLUMNS = tuple(sorted(VECTORS['columns']))
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
        't214d_postgres_values_record',
        ROOT / 'contract/tools/postgres_values_record.py',
    )


def held(name: str, check: Callable[[], None]) -> None:
    r"""Runs ``check``: it must pass, or, while ``name`` is pending, fail."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    if name not in gating['pending']:
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take it out of '
        'contract/tests/t214b-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: this file's keys in ``pending`` are keys it authored, and
    ``authored`` holds exactly this file's keys under its prefix."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    mine = {k for k in gating['authored'] if k.startswith(PREFIX)}
    assert mine == set(KEYS)
    assert {k for k in gating['pending'] if k.startswith(PREFIX)} <= mine


def test_the_vectors_are_the_reference_s_for_these_inputs() -> None:
    r"""Control: the vectors are the recorder's, for its tables and seeds,
    recorded by the reference at ``contract/reference-env.txt``'s commit,
    every column with the reference's two readings and the core's type."""
    made = recorder()
    assert VECTORS['recorded_by'] == 'contract/tools/postgres_values_record.py'
    assert VECTORS['setup'] == made.SETUP
    assert VECTORS['seed_times']['seeds'] == made.SEEDS
    assert sorted(VECTORS['columns']) == sorted(made.COLUMNS)
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {VECTORS["reference"]}' in env
    for name, column in VECTORS['columns'].items():
        assert {'driver', 'reference', 'arrow'} <= set(column), name
        assert len(column['driver']) == len(made.COLUMNS[name]['values']) + 1
    # The rules each column exercises are there: JSON true read as True,
    # an interval of a month read as 30 days, a whole number past int64.
    assert ['s', 'True'] in VECTORS['columns']['js']['reference']
    assert VECTORS['columns']['iv']['driver'][0] == [
        'dur',
        str((32 * 86_400 + 3) * 10**9),
    ]
    assert VECTORS['columns']['nw']['reference'] == {'raised': 'OverflowError'}
    assert set(READ) | {'iv'} == set(COLUMNS)


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


def _cells(values: list[Any]) -> list[Any]:
    return recorder().cells(values)


def _double(cell: Any) -> float:
    return struct.unpack('>d', bytes.fromhex(cell[1]))[0]


def _decimal(cell: Any) -> Decimal:
    r"""A canonical decimal cell's value."""
    assert cell[0] == 'd', cell
    return Decimal(cell[1]).scaleb(-cell[2])


# --- what the core takes is the reference's ----------------------------------


@pytest.mark.parametrize('column', COLUMNS)
def test_each_value_the_core_takes_is_the_reference_s(
    database: Any, column: str
) -> None:
    r"""T2.14d's exit "every Postgres case's answers unchanged": each
    column reaches the core as the recorded Arrow type, its values the
    reference's reading of them (or refused by name where the reference
    leaves the column out); the rule now in the provider holds wherever it
    moves."""
    if database is None:
        return
    vector = VECTORS['columns'][column]

    def check() -> None:
        admitted, _ = _read(database, column)
        if isinstance(admitted, str):
            assert admitted == vector['arrow'], admitted
            assert vector['reference'] == 'unsupported', column
            return
        field = admitted.schema.field('v')
        assert recorder()._arrow_type(field) == vector['arrow']
        assert admitted.column('id').to_pylist() == list(
            range(1, len(vector['driver']) + 1)
        )
        values = admitted.column('v').to_pylist()
        how = READ[column]
        if how in ('text', 'value'):
            assert _cells(values) == vector['reference'], values
        elif how == 'exact text, a double':
            assert all(v is None or isinstance(v, str) for v in values)
            assert [None if v is None else Decimal(v) for v in values] == [
                None if c is None else _decimal(c) for c in vector['driver']
            ]
            assert (
                _cells([None if v is None else float(v) for v in values])
                == (vector['reference'])
            )
        else:
            assert vector['reference'] == {'raised': 'OverflowError'}
            assert [None if v is None else Decimal(str(v)) for v in values] == [
                None if c is None else _decimal(c) for c in vector['driver']
            ]

    held(f'{PREFIX}core:{column}', check)


def test_each_seed_is_bound_as_the_reference_writes_it(database: Any) -> None:
    r"""T2.14d's key-set encoding and seed times: a keyed read of
    ``events`` within each seed's window binds each seed's key and its end
    time as the text the reference binds them as, a time as pandas'
    ``Timestamp.isoformat`` writes it (whole seconds, microseconds,
    nanoseconds, before 1970)."""
    if database is None:
        return
    from sdm_client._rust import codec
    from sdm_client._rust.run import run

    seeds = VECTORS['seed_times']['seeds']
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
        'provider': {'provider': 'sql', 'dialect': 'postgres'},
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
        # Each seed as the old client writes it (T3.28): its key as a
        # number and its end time as ``e``.
        assert [[str(row['id']), row['e']] for row in texts[0]] == (
            VECTORS['seed_times']['reference']
        )

    held(f'{PREFIX}seed-times', check)


# --- the provider only moves the driver's rows -------------------------------


def _faithful(field: pa.Field, value: Any, driver: Any) -> bool:
    r"""Whether ``value``, a cell of a provider's answer in ``field``, is
    the driver's value whose canonical cell is ``driver``, read as the
    comparator reads a statement's result: a JSON column's text as the
    value it holds, an interval as the ``timedelta`` it is (no months: the
    driver's holds none), a number by its value whatever its Arrow type."""
    if value is None or driver is None:
        return value is None and driver is None
    kind = field.type
    if isinstance(kind, pa.BaseExtensionType):
        kind = kind.storage_type
    json_field = recorder()._arrow_type(field).endswith(' arrow.json')
    if json_field:
        try:
            held_value = json.loads(value)
        except (TypeError, ValueError):
            return False
        return _cells([held_value]) == [driver]
    if pa.types.is_interval(kind):
        if value.months:
            return False
        nanos = value.days * 86_400 * 10**9 + value.nanoseconds
        return ['dur', str(nanos)] == driver
    if isinstance(value, dt.timedelta) or isinstance(driver, bool):
        return _cells([value]) == [driver]
    if driver[0] == 'd':
        if isinstance(value, float):
            return Decimal(value) == _decimal(driver)
        try:
            return Decimal(str(value)) == _decimal(driver)
        except ArithmeticError:
            return False
    return _cells([value]) == [driver]


@pytest.mark.parametrize('column', COLUMNS)
def test_the_provider_hands_the_core_the_driver_s_values(
    database: Any, column: str
) -> None:
    r"""T2.14d's exit "the provider holds no value rule": what the Postgres
    provider hands the core for each column is the driver's reading, value
    by value (the vectors' ``driver``), with no rule of its own applied:
    not Python's text of a JSON scalar, not a double of a numeric the
    driver read exactly."""
    if database is None:
        return
    driver = VECTORS['columns'][column]['driver']

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

    held(f'{PREFIX}driver-rows:{column}', check)


@pytest.mark.parametrize('rule', RULES)
def test_each_provider_rule_is_declared_moved_into_the_core(rule: str) -> None:
    r"""T2.14d's exit "each moved rule declared ``moved`` naming T2.14d"
    (or T2.14b, the landing it rides in):
    the bridge freeze's declarations name each of the provider's value
    rules at T2.14a's ledger moved by T2.14d."""

    def check() -> None:
        changes = json.loads(
            (ROOT / 'scripts/rust_core/bridge_changes.json').read_text(
                encoding='utf-8'
            )
        )['changes']
        declared = changes.get(f'{PROVIDER_FILE}::{rule}')
        assert declared is not None, f'{rule}: not declared'
        # T2.14d lands within T2.14b (the owner's granularity decision), so
        # the declaration may name either.
        moved = str(declared.get('moved', ''))
        assert 'T2.14d' in moved or 'T2.14b' in moved, declared

    held(f'{PREFIX}moved:{rule}', check)
