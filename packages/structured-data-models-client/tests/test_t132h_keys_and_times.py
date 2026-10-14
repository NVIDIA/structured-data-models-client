# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""T1.32h (register entry P1B-12): ``Schema.from_source``'s keys and time
columns of every Python type, on the core route.

Written by the package's independent test author from the design of
2026-10-13 (revised after Codex's review), section 4's table and section
6E's plan, before the implementation. It replaces
``test_the_schema_route.py``'s ``WRONGLY_TYPED``, which pinned the route's
outcome before the fix. ``contract/given-vectors.json`` holds the values
JSON can express for every host (``contract/tests/test_t132h_given_vectors.py``
and ``sdm-testkit``'s ``tests/t132h_given.rs``); this file holds the Python
host to the values only Python has.

- The host asks one question of each value: can it be written as the
  contract's type (a key: none, or an ordered list of names; a time
  column: none, or one name)? It writes ``{"value": ...}`` when it can and
  ``"unreadable"`` when it cannot; every refusal of a value is the core's.
  Each row of ``ROWS`` is a value given as the ``orders`` table's key or
  time column, over ``contract/bases.yaml``'s ``shop``: what the host
  writes (a spy on ``sdm_client._native.Run.start``), and the core route's
  answer, exactly: a success's key and time column, or the refusal's class
  and code and its fields (``kind`` and ``retryable`` aside). A refusal of
  a value comes after the source is read (a spy on the DuckDB provider
  counts the fetches before it); the host's own steps (reading an
  iterable, encoding) come before the run starts, and a value JSON cannot
  carry (a lone surrogate) is the codec's ``INVALID_REQUEST`` with no
  fields, before it.
- The reference (the old route, where the relational engine is
  installed) answers each row as the table says: its class and code.
- A generator is read once, before the source is read; ``np.str_`` and a
  ``str`` subclass are written as plain ``str``; a set is
  ``"unreadable"``.
- An iterable whose reading raises: the exception propagates unchanged on
  both routes, ``KeyboardInterrupt`` included, and on the core route
  nothing is started. As a time column it is never read.
- The property: for every value of a fixed, bounded strategy (the leaves
  and containers of section 6E, two deep, drawn from a seeded generator,
  as a key and as a time column), the core route answers with a success
  or a ``StructuredDataError`` coded ``INVALID_REQUEST`` or
  ``UNSUPPORTED_REQUEST``: never ``INVALID_MESSAGE``, never a raw
  exception. Hypothesis is not one of the client's test dependencies, so
  the strategy is drawn by ``random.Random`` with a fixed seed.
- The ordering promise (section 2): the table first, then the value;
  every key before any time column; the core takes the tables in name
  order, not the caller's.
- The setter and ``Schema.from_source`` refuse the same value alike: the
  same code and ``rule``, the field each its own (``primary_key`` against
  ``primary_keys``).
- Seeds: each of the design's seeds 1 to 6 fails a check here.

Gating (``contract/tests/t132h_gate.py``): a test named in
``t132h-pending.json``'s ``pending["exits"]`` (as ``py:<test id>``) failed
when written; a row of a decision still the owner's is gated in that
decision's group (``D3``, ``D4``). It still runs and must still fail; once
it passes it fails, naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import math
import random
import sys
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import Schema, StructuredDataError, _routing
from sdm_client._rust.providers.duckdb import DuckDBProvider

ROOT = Path(__file__).resolve().parents[3]
CORE_ANY = ','.join(
    dict.fromkeys(
        _routing.FROM_SOURCE_DUCKDB + _routing.FROM_SOURCE_DUCKDB_TABLES
    )
)
ENGINE = importlib.util.find_spec('sdm_relational_engine') is not None


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


G = _module('t132h_gate', ROOT / 'contract/tests/t132h_gate.py')


@pytest.fixture(autouse=True)
def _old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


def _shop() -> list[str]:
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


def _connection() -> Any:
    connection = duckdb.connect()
    for statement in _shop():
        connection.execute(statement)
    return connection


# --- what the host wrote, what it fetched, what it answered ---------------------


class Spy:
    r"""What one call did: each ``describe_source`` start's spec as the host
    wrote it, the fetches the provider answered, and the order of the
    events (``start``, ``fetch``, and ``iter`` from a value that counts its
    readings)."""

    def __init__(self) -> None:
        self.written: list[Any] = []
        self.events: list[str] = []

    @property
    def fetches(self) -> int:
        return self.events.count('fetch')

    @property
    def starts(self) -> int:
        return self.events.count('start')


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Spy:
    from sdm_client._rust.run import native

    made = Spy()
    core = native()
    real = core.Run

    class Run:
        @staticmethod
        def start(control: bytes, attachments: Any) -> Any:
            operation = json.loads(control)['operation']
            made.events.append('start')
            if operation['op'] == 'describe_source':
                made.written.append(
                    json.loads(attachments[operation['source']])
                )
            return real.start(control, attachments)

    monkeypatch.setattr(core, 'Run', Run)
    call = DuckDBProvider.__call__

    def counted(self: Any, request: Any, key_sets: Any) -> Any:
        made.events.append('fetch')
        return call(self, request, key_sets)

    monkeypatch.setattr(DuckDBProvider, '__call__', counted)
    return made


def _orders(schema: Schema) -> dict[str, Any]:
    table = schema.tables['orders']
    key = table.primary_key
    if isinstance(key, str):
        key = [key]
    elif key is not None:
        key = list(key)
    return {'primary_key': key, 'time_column': table.time_column}


def observe(
    monkeypatch: pytest.MonkeyPatch,
    spy: Spy,
    arguments: dict[str, Any],
    *,
    core: bool = True,
) -> dict[str, Any]:
    r"""``Schema.from_source`` over ``shop`` with ``arguments``, on the core
    route or the old one: its answer, what the host wrote and fetched."""
    if core:
        monkeypatch.setenv(_routing.ENVIRONMENT, CORE_ANY)
    else:
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    spy.written.clear()
    spy.events.clear()
    got: dict[str, Any]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(
                DuckDBSource(_connection()), **arguments
            )
    except StructuredDataError as error:
        got = {
            'outcome': 'refusal',
            'class': type(error).__name__,
            'code': error.code,
            'fields': {
                k: v
                for k, v in dict(error.details).items()
                if k not in ('kind', 'retryable')
            },
        }
    except Exception as error:  # noqa: BLE001 - the outcome observed
        got = {'outcome': 'raised', 'class': type(error).__name__}
    else:
        got = {'outcome': 'success', 'orders': _orders(schema)}
    got['written'] = spy.written[-1] if spy.written else None
    got['starts'] = spy.starts
    got['fetches'] = spy.fetches
    return got


# --- the rows ----------------------------------------------------------------------

U = 'unreadable'


def V(value: Any) -> dict[str, Any]:
    return {'value': value}


class Name(str):
    r"""A subclass of ``str``: written as the plain ``str`` it holds."""


class Counted:
    r"""An iterable of ``names`` that counts its readings."""

    def __init__(self, names: list[Any], spy: Spy | None = None) -> None:
        self.names = names
        self.read = 0
        self.spy = spy

    def __iter__(self) -> Iterator[Any]:
        self.read += 1
        if self.spy is not None:
            self.spy.events.append('iter')
        return iter(self.names)


class Raising:
    r"""An iterable whose reading raises ``error``."""

    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.read = 0

    def __iter__(self) -> Iterator[Any]:
        self.read += 1
        raise self.error


def _names(*names: str) -> Iterator[str]:
    yield from names


NAN = float('nan')
DATE = dt.date(2024, 1, 1)
DATETIME = dt.datetime(2024, 1, 1, 2, 3, 4)
STAMP = pd.Timestamp('2024-01-01 02:03:04')

KEY_NOT_NAMES = {
    'field': 'primary_keys',
    'table': 'orders',
    'rule': 'key_not_names',
}
TIME_NOT_A_NAME = {
    'field': 'time_columns',
    'table': 'orders',
    'rule': 'time_column_not_a_name',
}
EMPTY_KEY = {'field': 'primary_keys', 'table': 'orders'}
COMPOSITE = {'operation': 'describe_source', 'rule': 'composite_key'}
INFERRED = {'primary_key': ['order_id'], 'time_column': 'ts'}
SDE = 'StructuredDataError'


def row(
    id: str,
    arguments: Callable[[], dict[str, Any]],
    written: dict[str, Any] | None,
    core: tuple[Any, ...],
    reference: tuple[Any, ...],
    owner: str | None = None,
) -> dict[str, Any]:
    r"""One value: ``written``, the spec's ``primary_keys`` and
    ``time_columns`` as the host writes them (``None``: nothing started);
    ``core``, ``('success', orders)`` or ``('refusal', class, code,
    fields, where)``, ``where`` ``run`` (after a fetch) or ``host``
    (before any start); ``reference``, ``('success',)`` or ``(class,
    code)``."""
    return {
        'id': id,
        'arguments': arguments,
        'written': written,
        'core': core,
        'reference': reference,
        'owner': owner,
    }


def key(
    id: str,
    value: Callable[[], Any],
    written: Any,
    core: tuple[Any, ...],
    reference: tuple[Any, ...],
    owner: str | None = None,
) -> dict[str, Any]:
    return row(
        id,
        lambda: {'primary_keys': {'orders': value()}},
        None
        if written is None
        else {'primary_keys': {'orders': written}, 'time_columns': {}},
        core,
        reference,
        owner,
    )


def time(
    id: str,
    value: Callable[[], Any],
    written: Any,
    core: tuple[Any, ...],
    reference: tuple[Any, ...],
) -> dict[str, Any]:
    return row(
        id,
        lambda: {'time_columns': {'orders': value()}},
        None
        if written is None
        else {'primary_keys': {}, 'time_columns': {'orders': written}},
        core,
        reference,
    )


def refused(
    fields: dict[str, Any],
    code: str = 'INVALID_REQUEST',
    cls: str = SDE,
    where: str = 'run',
) -> tuple[Any, ...]:
    return ('refusal', cls, code, fields, where)


NOT_NAMES = refused(KEY_NOT_NAMES)
NOT_A_NAME = refused(TIME_NOT_A_NAME)
REF_REFUSED = (SDE, 'INVALID_REQUEST')
REF_OK = ('success',)

ROWS = [
    # Keys: section 4's rows 1 to 16.
    key('k.none', lambda: None, V(None), refused(EMPTY_KEY), REF_REFUSED),
    key('k.int', lambda: 3, U, NOT_NAMES, REF_REFUSED),
    key('k.float', lambda: 1.5, U, NOT_NAMES, REF_REFUSED),
    key('k.nan', lambda: NAN, U, NOT_NAMES, REF_REFUSED),
    key('k.true', lambda: True, U, NOT_NAMES, REF_REFUSED),
    key('k.np-bool', lambda: np.bool_(True), U, NOT_NAMES, REF_REFUSED),
    key('k.np-int', lambda: np.int64(3), U, NOT_NAMES, REF_REFUSED),
    key('k.np-float', lambda: np.float32(1.5), U, NOT_NAMES, REF_REFUSED),
    key(
        'k.str',
        lambda: 'order_id',
        V(['order_id']),
        ('success', INFERRED),
        REF_OK,
    ),
    key(
        'k.np-str',
        lambda: np.str_('order_id'),
        V(['order_id']),
        ('success', INFERRED),
        REF_OK,
    ),
    key(
        'k.str-subclass',
        lambda: Name('order_id'),
        V(['order_id']),
        ('success', INFERRED),
        REF_OK,
    ),
    key(
        'k.tuple',
        lambda: ('order_id',),
        V(['order_id']),
        ('success', INFERRED),
        REF_OK,
    ),
    key(
        'k.list',
        lambda: ['order_id'],
        V(['order_id']),
        ('success', INFERRED),
        REF_OK,
    ),
    key('k.nested', lambda: [['order_id']], U, NOT_NAMES, REF_REFUSED),
    key('k.set', lambda: {'order_id'}, U, NOT_NAMES, REF_REFUSED),
    key(
        'k.frozenset',
        lambda: frozenset({'order_id'}),
        U,
        NOT_NAMES,
        REF_REFUSED,
    ),
    key('k.dict', lambda: {'order_id': 1}, U, NOT_NAMES, REF_REFUSED),
    key(
        'k.generator',
        lambda: _names('order_id'),
        V(['order_id']),
        ('success', INFERRED),
        REF_OK,
    ),
    key(
        'k.pd-index',
        lambda: pd.Index(['order_id']),
        V(['order_id']),
        ('success', INFERRED),
        REF_OK,
    ),
    key(
        'k.np-array',
        lambda: np.array(['order_id']),
        V(['order_id']),
        ('success', INFERRED),
        REF_OK,
    ),
    key('k.bytes', lambda: b'order_id', U, NOT_NAMES, REF_REFUSED),
    key(
        'k.bytearray', lambda: bytearray(b'order_id'), U, NOT_NAMES, REF_REFUSED
    ),
    key('k.empty', lambda: [], V([]), refused(EMPTY_KEY), REF_REFUSED),
    # D3 (P1B-73, the owner's): a column named twice, refused by name.
    key(
        'k.twice',
        lambda: ['order_id', 'order_id'],
        V(['order_id', 'order_id']),
        refused(
            {
                'field': 'primary_keys',
                'table': 'orders',
                'column': 'order_id',
                'rule': 'key_column_twice',
            }
        ),
        REF_OK,
        owner='D3',
    ),
    key(
        'k.twice-generator',
        lambda: _names('order_id', 'user_id', 'order_id'),
        V(['order_id', 'user_id', 'order_id']),
        refused(
            {
                'field': 'primary_keys',
                'table': 'orders',
                'column': 'order_id',
                'rule': 'key_column_twice',
            }
        ),
        REF_OK,
        owner='D3',
    ),
    key('k.member-int', lambda: ['order_id', 3], U, NOT_NAMES, REF_REFUSED),
    key('k.member-nan', lambda: [NAN], U, NOT_NAMES, REF_REFUSED),
    key('k.member-none', lambda: ['order_id', None], U, NOT_NAMES, REF_REFUSED),
    key(
        'k.member-np-str',
        lambda: [np.str_('order_id')],
        V(['order_id']),
        ('success', INFERRED),
        REF_OK,
    ),
    key('k.date', lambda: DATE, U, NOT_NAMES, REF_REFUSED),
    key('k.datetime', lambda: DATETIME, U, NOT_NAMES, REF_REFUSED),
    key('k.timestamp', lambda: STAMP, U, NOT_NAMES, REF_REFUSED),
    key(
        'k.missing',
        lambda: 'nope',
        V(['nope']),
        refused({**EMPTY_KEY, 'column': 'nope'}),
        REF_REFUSED,
    ),
    key(
        'k.composite',
        lambda: ['order_id', 'user_id'],
        V(['order_id', 'user_id']),
        refused(COMPOSITE, 'UNSUPPORTED_REQUEST', 'UnsupportedRequestError'),
        REF_OK,
    ),
    # Time columns: rows 18 to 22.
    time(
        't.none',
        lambda: None,
        V(None),
        ('success', {'primary_key': ['order_id'], 'time_column': None}),
        REF_OK,
    ),
    time('t.str', lambda: 'ts', V('ts'), ('success', INFERRED), REF_OK),
    time(
        't.np-str',
        lambda: np.str_('ts'),
        V('ts'),
        ('success', INFERRED),
        REF_OK,
    ),
    time(
        't.str-subclass',
        lambda: Name('ts'),
        V('ts'),
        ('success', INFERRED),
        REF_OK,
    ),
    time(
        't.missing',
        lambda: 'nope',
        V('nope'),
        refused({'field': 'time_columns', 'table': 'orders', 'column': 'nope'}),
        REF_REFUSED,
    ),
    time('t.int', lambda: 3, U, NOT_A_NAME, REF_REFUSED),
    time('t.float', lambda: 1.5, U, NOT_A_NAME, REF_REFUSED),
    time('t.nan', lambda: NAN, U, NOT_A_NAME, REF_REFUSED),
    time('t.true', lambda: True, U, NOT_A_NAME, REF_REFUSED),
    time('t.np-bool', lambda: np.bool_(True), U, NOT_A_NAME, REF_REFUSED),
    time('t.np-int', lambda: np.int64(3), U, NOT_A_NAME, REF_REFUSED),
    time('t.date', lambda: DATE, U, NOT_A_NAME, REF_REFUSED),
    time('t.datetime', lambda: DATETIME, U, NOT_A_NAME, REF_REFUSED),
    time('t.timestamp', lambda: STAMP, U, NOT_A_NAME, REF_REFUSED),
    time('t.tuple', lambda: ('ts',), U, NOT_A_NAME, REF_REFUSED),
    time('t.frozenset', lambda: frozenset({'ts'}), U, NOT_A_NAME, REF_REFUSED),
    time('t.generator', lambda: _names('ts'), U, NOT_A_NAME, REF_REFUSED),
    time('t.bytes', lambda: b'ts', U, NOT_A_NAME, REF_REFUSED),
    time(
        't.raising',
        lambda: Raising(RuntimeError('boom')),
        U,
        NOT_A_NAME,
        REF_REFUSED,
    ),
    time('t.list', lambda: ['ts'], U, NOT_A_NAME, ('TypeError', None)),
    time('t.set', lambda: {'ts'}, U, NOT_A_NAME, ('TypeError', None)),
    time('t.dict', lambda: {'ts': 1}, U, NOT_A_NAME, ('TypeError', None)),
    time(
        't.pd-index',
        lambda: pd.Index(['ts']),
        U,
        NOT_A_NAME,
        ('TypeError', None),
    ),
    time(
        't.np-array',
        lambda: np.array(['ts']),
        U,
        NOT_A_NAME,
        ('TypeError', None),
    ),
    # Both mappings: rows 23 and 24.
    row(
        'm.key-table-int',
        lambda: {'primary_keys': {3: 'order_id'}},
        {'primary_keys': {'3': V(['order_id'])}, 'time_columns': {}},
        refused({'field': 'primary_keys', 'table': '3'}),
        REF_REFUSED,
    ),
    row(
        'm.time-table-int',
        lambda: {'time_columns': {3: 'ts'}},
        {'primary_keys': {}, 'time_columns': {'3': V('ts')}},
        refused({'field': 'time_columns', 'table': '3'}),
        REF_REFUSED,
    ),
    # JSON's own coercion of an object's keys (row 23, without D4): None
    # is the table "null"; a tuple is no key JSON writes, the codec's
    # refusal before the run.
    row(
        'm.key-table-none',
        lambda: {'primary_keys': {None: 'order_id'}},
        {'primary_keys': {'null': V(['order_id'])}, 'time_columns': {}},
        refused({'field': 'primary_keys', 'table': 'null'}),
        REF_REFUSED,
    ),
    row(
        'm.key-table-tuple',
        lambda: {'primary_keys': {('orders',): 'order_id'}},
        None,
        refused({}, where='host'),
        REF_REFUSED,
    ),
    key(
        'm.key-surrogate',
        lambda: '\ud800',
        None,
        refused({}, where='host'),
        REF_REFUSED,
    ),
    key(
        'm.key-member-surrogate',
        lambda: ['\ud800'],
        None,
        refused({}, where='host'),
        REF_REFUSED,
    ),
    time(
        'm.time-surrogate',
        lambda: '\ud800',
        None,
        refused({}, where='host'),
        REF_REFUSED,
    ),
]
BY_ID = {r['id']: r for r in ROWS}


def _row_ids() -> list[Any]:
    return [pytest.param(r['id'], id=r['id']) for r in ROWS]


def core_problems(the_row: dict[str, Any], got: dict[str, Any]) -> list[str]:
    r"""Where the core route's answer to a row is not the table's."""
    problems = []
    written = the_row['written']
    if written is not None:
        written = {'tables': [], **written}
    if got['written'] != written:
        problems.append(f'the host wrote {got["written"]}, wanted {written}')
    want = the_row['core']
    if want[0] == 'success':
        if got['outcome'] != 'success':
            return [*problems, f'{got}']
        if got['orders'] != want[1]:
            problems.append(f'orders {got["orders"]}, wanted {want[1]}')
        return problems
    _, cls, code, fields, where = want
    if got['outcome'] != 'refusal':
        return [*problems, f'{got}']
    if (got['class'], got['code']) != (cls, code):
        problems.append(f'{got["class"]} {got["code"]}, wanted {cls} {code}')
    if got['fields'] != fields:
        problems.append(f'fields {got["fields"]}, wanted {fields}')
    if where == 'host' and got['starts']:
        problems.append('a run was started before the host refused')
    if where == 'run' and not got['fetches']:
        problems.append('refused before the source was read (no fetch)')
    return problems


def reference_problems(
    the_row: dict[str, Any], got: dict[str, Any]
) -> list[str]:
    want = the_row['reference']
    if want == REF_OK:
        return [] if got['outcome'] == 'success' else [f'{got}']
    if (got.get('class'), got.get('code')) != want:
        return [f'{got.get("class")} {got.get("code")}, wanted {want}']
    return []


def _gated(the_row: dict[str, Any], name: str) -> tuple[str, str]:
    if the_row['owner']:
        return the_row['owner'], f'exits:py:{name}'
    return 'exits', f'py:{name}'


@pytest.mark.parametrize('row_id', _row_ids())
def test_the_core_route_writes_and_answers_each_value_as_the_table_says(
    row_id: str,
    monkeypatch: pytest.MonkeyPatch,
    spy: Spy,
    request: pytest.FixtureRequest,
) -> None:
    the_row = BY_ID[row_id]

    def check() -> None:
        got = observe(monkeypatch, spy, the_row['arguments']())
        problems = core_problems(the_row, got)
        assert not problems, problems

    G.held(*_gated(the_row, request.node.name), check)


@pytest.mark.parametrize('row_id', _row_ids())
def test_the_reference_answers_each_value_as_the_table_says(
    row_id: str, monkeypatch: pytest.MonkeyPatch, spy: Spy
) -> None:
    if not ENGINE:
        return
    the_row = BY_ID[row_id]
    got = observe(monkeypatch, spy, the_row['arguments'](), core=False)
    assert reference_problems(the_row, got) == []


# --- reading the caller's iterables ---------------------------------------------------


def test_a_key_s_iterable_is_read_once(
    monkeypatch: pytest.MonkeyPatch, spy: Spy, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        value = Counted(['order_id'])
        got = observe(monkeypatch, spy, {'primary_keys': {'orders': value}})
        assert got['outcome'] == 'success', got
        assert got['written']['primary_keys'] == {'orders': V(['order_id'])}
        assert value.read == 1, f'read {value.read} times'
        generator = _names('order_id')
        got = observe(monkeypatch, spy, {'primary_keys': {'orders': generator}})
        assert got['outcome'] == 'success', got
        assert list(generator) == []

    G.held('exits', f'py:{request.node.name}', check)


def test_the_iterable_is_read_before_the_run_starts(
    monkeypatch: pytest.MonkeyPatch, spy: Spy, request: pytest.FixtureRequest
) -> None:
    # Section 2, step 1: the host's own steps come first, on the core route
    # only (row 8's timing note).
    def check() -> None:
        monkeypatch.setenv(_routing.ENVIRONMENT, CORE_ANY)
        spy.events.clear()
        value = Counted(['order_id'], spy)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Schema.from_source(
                DuckDBSource(_connection()), primary_keys={'orders': value}
            )
        assert spy.events.count('iter') == 1, spy.events
        assert spy.events.index('iter') < spy.events.index('start'), spy.events

    G.held('exits', f'py:{request.node.name}', check)


def propagation_problems(
    error: BaseException, seen: dict[str, Any], core: bool
) -> list[str]:
    r"""Where a call given an iterable whose reading raises ``error`` did
    not let that very exception out, read the iterable once and, on the
    core route, start nothing."""
    problems = []
    if seen['raised'] is not error:
        problems.append(
            f"raised {seen['raised']!r}, not the caller's {error!r}"
        )
    if seen['read'] != 1:
        problems.append(f'read {seen["read"]} times')
    if core and seen['starts']:
        problems.append('a run was started')
    return problems


def call_raising(
    error: BaseException, monkeypatch: pytest.MonkeyPatch, spy: Spy, core: bool
) -> dict[str, Any]:
    if core:
        monkeypatch.setenv(_routing.ENVIRONMENT, CORE_ANY)
    else:
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    spy.events.clear()
    value = Raising(error)
    raised: BaseException | None = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Schema.from_source(
                DuckDBSource(_connection()), primary_keys={'orders': value}
            )
    except BaseException as caught:  # noqa: BLE001 - KeyboardInterrupt too
        raised = caught
    return {'raised': raised, 'read': value.read, 'starts': spy.starts}


@pytest.mark.parametrize(
    'error',
    [
        pytest.param(RuntimeError('boom'), id='runtime-error'),
        pytest.param(KeyboardInterrupt(), id='keyboard-interrupt'),
    ],
)
def test_an_iterable_that_raises_propagates_unchanged_on_both_routes(
    error: BaseException,
    monkeypatch: pytest.MonkeyPatch,
    spy: Spy,
    request: pytest.FixtureRequest,
) -> None:
    # The design's policy (section 2, item 2): the classifier catches
    # nothing; the caller's exception is the call's, as list(key) lets it
    # out on the reference. On the core route, no run is started.
    def check() -> None:
        problems = []
        for core in [True, False] if ENGINE else [True]:
            seen = call_raising(error, monkeypatch, spy, core)
            problems.extend(
                f'{"core" if core else "reference"}: {p}'
                for p in propagation_problems(error, seen, core)
            )
        assert not problems, problems

    G.held('exits', f'py:{request.node.name}', check)


# --- the property ----------------------------------------------------------------------

#: The strategy's leaves (section 6E), each made fresh.
LEAVES: list[Callable[[], Any]] = [
    lambda: None,
    *[(lambda i=i: i) for i in range(-3, 4)],
    lambda: 1.5,
    lambda: NAN,
    lambda: math.inf,
    lambda: -math.inf,
    lambda: True,
    lambda: False,
    lambda: np.bool_(True),
    lambda: np.bool_(False),
    lambda: np.int64(3),
    lambda: np.float32(1.5),
    *[(lambda s=s: s) for s in ('order_id', 'ts', 'nope', '')],
    *[(lambda s=s: np.str_(s)) for s in ('order_id', 'ts', 'nope', '')],
    lambda: b'x',
    lambda: DATE,
    lambda: DATETIME,
    lambda: STAMP,
]
CONTAINERS = (
    'list',
    'tuple',
    'set',
    'frozenset',
    'dict',
    'generator',
    'index',
    'array',
)
#: The draws past the leaves, and the seed: fixed, so the run is the same
#: every time.
DRAWS = 120
SEED = 13212


def _hashable(value: Any) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _objects(members: list[Any]) -> Any:
    made = np.empty(len(members), dtype=object)
    for at, member in enumerate(members):
        made[at] = member
    return made


def build(recipe: Any) -> Any:
    r"""The value a recipe names, made fresh (a generator is read once)."""
    if recipe[0] == 'leaf':
        return LEAVES[recipe[1]]()
    kind, parts = recipe
    members = [build(part) for part in parts]
    texts = bool(members) and all(isinstance(m, str) for m in members)
    if kind == 'list':
        return members
    if kind == 'tuple':
        return tuple(members)
    if kind == 'set':
        return {m for m in members if _hashable(m)}
    if kind == 'frozenset':
        return frozenset(m for m in members if _hashable(m))
    if kind == 'dict':
        return {m: 1 for m in members if _hashable(m)}
    if kind == 'generator':
        return (m for m in members)
    if kind == 'index':
        return (
            pd.Index(members)
            if texts
            else pd.Index(_objects(members), dtype=object)
        )
    return np.array(members) if texts else _objects(members)


def draw(rng: random.Random, depth: int) -> Any:
    if depth == 0 or rng.random() < 0.4:
        return ('leaf', rng.randrange(len(LEAVES)))
    kind = rng.choice(CONTAINERS)
    return (kind, [draw(rng, depth - 1) for _ in range(rng.randint(0, 3))])


def strategy() -> list[Any]:
    r"""Every leaf, then ``DRAWS`` values two deep at most."""
    rng = random.Random(SEED)
    return [('leaf', at) for at in range(len(LEAVES))] + [
        draw(rng, 2) for _ in range(DRAWS)
    ]


ALLOWED = {'INVALID_REQUEST', 'UNSUPPORTED_REQUEST'}


def property_problems(
    recipe: Any, field: str, got: dict[str, Any]
) -> list[str]:
    if got['outcome'] == 'success':
        return []
    if got['outcome'] == 'refusal' and got['code'] in ALLOWED:
        return []
    return [f'{field} {recipe}: {got}']


def test_no_value_of_the_strategy_escapes_as_a_raw_error_or_a_host_bug(
    monkeypatch: pytest.MonkeyPatch, spy: Spy, request: pytest.FixtureRequest
) -> None:
    def check() -> None:
        problems = []
        for recipe in strategy():
            for field in ('primary_keys', 'time_columns'):
                got = observe(
                    monkeypatch, spy, {field: {'orders': build(recipe)}}
                )
                problems.extend(property_problems(recipe, field, got))
        assert not problems, f'{len(problems)}: {problems[:20]}'

    G.held('exits', f'py:{request.node.name}', check)


def test_the_strategy_is_bounded_and_reaches_every_kind() -> None:
    drawn = strategy()
    assert len(drawn) == len(LEAVES) + DRAWS

    def depth(recipe: Any) -> int:
        if recipe[0] == 'leaf':
            return 0
        return 1 + max((depth(p) for p in recipe[1]), default=0)

    assert max(depth(r) for r in drawn) <= 2
    kinds = {r[0] for r in drawn}
    assert set(CONTAINERS) <= kinds, set(CONTAINERS) - kinds
    assert all(len(r[1]) <= 3 for r in drawn if r[0] != 'leaf')
    assert strategy() == drawn


# --- the ordering promise ----------------------------------------------------------------


ORDER = [
    pytest.param(
        {'primary_keys': {'nope': 3}},
        {'field': 'primary_keys', 'table': 'nope'},
        id='the-table-before-the-key',
    ),
    pytest.param(
        {'time_columns': {'nope': ['ts']}},
        {'field': 'time_columns', 'table': 'nope'},
        id='the-table-before-the-time-column',
    ),
    pytest.param(
        {
            'primary_keys': {'orders': {'order_id'}},
            'time_columns': {'orders': ['ts']},
        },
        KEY_NOT_NAMES,
        id='every-key-before-any-time-column',
    ),
    pytest.param(
        {'primary_keys': {'users': np.bool_(True), 'orders': 'nope'}},
        {'field': 'primary_keys', 'table': 'orders', 'column': 'nope'},
        id='the-tables-in-name-order',
    ),
    pytest.param(
        {'primary_keys': {'orders': ['order_id', 'nope']}},
        {'field': 'primary_keys', 'table': 'orders', 'column': 'nope'},
        id='a-missing-column-before-a-composite-key',
    ),
]


@pytest.mark.parametrize(('arguments', 'fields'), ORDER)
def test_the_core_refuses_in_the_promised_order(
    arguments: dict[str, Any],
    fields: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    spy: Spy,
    request: pytest.FixtureRequest,
) -> None:
    def check() -> None:
        got = observe(monkeypatch, spy, arguments)
        assert got['outcome'] == 'refusal', got
        assert (got['code'], got['fields']) == ('INVALID_REQUEST', fields), got
        assert got['fetches'] > 0, got

    G.held('exits', f'py:{request.node.name}', check)


# --- the setter beside Schema.from_source ------------------------------------------------

SETTER = [
    pytest.param('primary_key', {'order_id'}, id='key-set'),
    pytest.param('primary_key', np.bool_(True), id='key-np-bool'),
    pytest.param('primary_key', b'order_id', id='key-bytes'),
    pytest.param('primary_key', DATE, id='key-date'),
    pytest.param('primary_key', [NAN], id='key-member-nan'),
    pytest.param('time_column', ['ts'], id='time-list'),
    pytest.param('time_column', {'ts'}, id='time-set'),
    pytest.param('time_column', np.int64(3), id='time-np-int'),
    pytest.param('time_column', STAMP, id='time-timestamp'),
]
RULES = {
    'primary_key': 'key_not_names',
    'time_column': 'time_column_not_a_name',
}


def pair_problems(attribute: str, edited: Any, described: Any) -> list[str]:
    r"""Where the setter's refusal and ``Schema.from_source``'s of the same
    value differ, but in the field, each operation's own."""
    if not isinstance(edited, StructuredDataError):
        return [f'the setter: {edited!r}']
    if not isinstance(described, StructuredDataError):
        return [f'from_source: {described!r}']
    problems = []
    mine = {
        k: v
        for k, v in dict(edited.details).items()
        if k not in ('kind', 'retryable')
    }
    theirs = {
        k: v
        for k, v in dict(described.details).items()
        if k not in ('kind', 'retryable')
    }
    if edited.code != described.code or edited.code != 'INVALID_REQUEST':
        problems.append(f'{edited.code} beside {described.code}')
    if (
        mine.get('rule') != RULES[attribute]
        or theirs.get('rule') != RULES[attribute]
    ):
        problems.append(f'rule {mine.get("rule")} beside {theirs.get("rule")}')
    if mine != {
        'field': attribute,
        'table': 'orders',
        'rule': RULES[attribute],
    }:
        problems.append(f'the setter: {mine}')
    if theirs != {
        'field': f'{attribute}s',
        'table': 'orders',
        'rule': RULES[attribute],
    }:
        problems.append(f'from_source: {theirs}')
    return problems


@pytest.mark.parametrize(('attribute', 'value'), SETTER)
def test_the_setter_and_from_source_refuse_a_value_alike(
    attribute: str,
    value: Any,
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    def check() -> None:
        monkeypatch.setenv(_routing.ENVIRONMENT, CORE_ANY)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(_connection()))
            edited: Any = None
            try:
                setattr(schema['orders'], attribute, value)
            except Exception as error:  # noqa: BLE001 - compared below
                edited = error
            described: Any = None
            try:
                Schema.from_source(
                    DuckDBSource(_connection()),
                    **{f'{attribute}s': {'orders': value}},
                )
            except Exception as error:  # noqa: BLE001 - compared below
                described = error
        problems = pair_problems(attribute, edited, described)
        assert not problems, problems

    G.held('exits', f'py:{request.node.name}', check)


# --- the seeds ---------------------------------------------------------------------------


def test_each_seeded_fault_fails_a_check() -> None:
    def good(the_row: dict[str, Any]) -> dict[str, Any]:
        want = the_row['core']
        written = the_row['written']
        if written is not None:
            written = {'tables': [], **written}
        base = {'written': written, 'starts': 1, 'fetches': 2}
        if want[0] == 'success':
            return {**base, 'outcome': 'success', 'orders': want[1]}
        _, cls, code, fields, where = want
        if where == 'host':
            base = {**base, 'starts': 0, 'fetches': 0}
        return {
            **base,
            'outcome': 'refusal',
            'class': cls,
            'code': code,
            'fields': fields,
        }

    for the_row in ROWS:
        assert core_problems(the_row, good(the_row)) == [], the_row['id']
    # 1: a set written as a sorted value.
    k_set = BY_ID['k.set']
    seeded = {
        **good(k_set),
        'written': {
            'tables': [],
            'primary_keys': {'orders': V(['order_id'])},
            'time_columns': {},
        },
    }
    assert core_problems(k_set, seeded) != []
    # 2: None written as an empty key.
    k_none = BY_ID['k.none']
    seeded = {
        **good(k_none),
        'written': {
            'tables': [],
            'primary_keys': {'orders': V([])},
            'time_columns': {},
        },
    }
    assert core_problems(k_none, seeded) != []
    # 3: the host refuses unreadable itself, before the run.
    k_int = BY_ID['k.int']
    seeded = {**good(k_int), 'written': None, 'starts': 0, 'fetches': 0}
    assert core_problems(k_int, seeded) != []
    # 3, as the host's own words: no rule, no fetch.
    seeded = {**good(k_int), 'fields': {}, 'fetches': 0}
    assert core_problems(k_int, seeded) != []
    # 5: np.str_ classified unreadable.
    k_np_str = BY_ID['k.np-str']
    seeded = {
        'outcome': 'refusal',
        'class': SDE,
        'code': 'INVALID_REQUEST',
        'fields': KEY_NOT_NAMES,
        'written': {
            'tables': [],
            'primary_keys': {'orders': U},
            'time_columns': {},
        },
        'starts': 1,
        'fetches': 2,
    }
    assert core_problems(k_np_str, seeded) != []
    # A time column of the reference's unnamed TypeError left unnamed.
    t_list = BY_ID['t.list']
    seeded = {
        'outcome': 'raised',
        'class': 'TypeError',
        'written': None,
        'starts': 0,
        'fetches': 0,
    }
    assert core_problems(t_list, seeded) != []
    # The property: a host bug or a raw error fails it.
    assert (
        property_problems(
            ('leaf', 0),
            'primary_keys',
            {'outcome': 'refusal', 'code': 'INVALID_MESSAGE'},
        )
        != []
    )
    assert (
        property_problems(
            ('leaf', 0),
            'primary_keys',
            {'outcome': 'raised', 'class': 'TypeError'},
        )
        != []
    )
    assert (
        property_problems(
            ('leaf', 0),
            'primary_keys',
            {'outcome': 'refusal', 'code': 'UNSUPPORTED_REQUEST'},
        )
        == []
    )

    # The setter's field made plural, or its rule dropped.
    def error(**details: Any) -> StructuredDataError:
        made = StructuredDataError('x', code='INVALID_REQUEST')
        made.details.update(details)
        return made

    describe = error(field='primary_keys', table='orders', rule='key_not_names')
    setter = error(field='primary_key', table='orders', rule='key_not_names')
    assert pair_problems('primary_key', setter, describe) == []
    assert (
        pair_problems(
            'primary_key',
            error(field='primary_keys', table='orders', rule='key_not_names'),
            describe,
        )
        != []
    )
    assert (
        pair_problems(
            'primary_key', error(field='primary_key', table='orders'), describe
        )
        != []
    )


def test_a_generator_read_twice_or_an_iteration_error_caught_fails() -> None:
    # 4: a generator read twice reads nothing the second time: a host that
    # reads it twice writes an empty key, which the rows hold.
    generator = _names('order_id')
    first, second = list(generator), list(generator)
    assert (first, second) == (['order_id'], [])
    k_generator = BY_ID['k.generator']
    seeded = {
        'outcome': 'refusal',
        'class': SDE,
        'code': 'INVALID_REQUEST',
        'fields': EMPTY_KEY,
        'written': {
            'tables': [],
            'primary_keys': {'orders': V([])},
            'time_columns': {},
        },
        'starts': 1,
        'fetches': 2,
    }
    assert core_problems(k_generator, seeded) != []
    # 6: a classifier that catches the iteration error (or any
    # BaseException) and writes unreadable answers with the core's refusal,
    # not the caller's exception.
    for error in (RuntimeError('boom'), KeyboardInterrupt()):
        refusal = StructuredDataError('x', code='INVALID_REQUEST')
        caught = {'raised': refusal, 'read': 1, 'starts': 1}
        assert propagation_problems(error, caught, True) != []
        assert (
            propagation_problems(
                error, {**caught, 'raised': error, 'starts': 0}, True
            )
            == []
        )
        assert (
            propagation_problems(
                error, {'raised': error, 'read': 2, 'starts': 0}, True
            )
            != []
        )
