# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The dialect calls no recorded scenario makes, made directly.

The corpus reaches every statement a scenario sends, but not every branch a
dialect has: no scenario asks ``holds_any``, keys a table by two columns,
reads a table large enough for a scan cap, or stores times with a zone in
every backend. :func:`synthetic` makes those calls, each under a
``synthetic:<label>`` source, through the dialect modules the capture has
wrapped, so they are recorded as the scenarios' are.

Every input here is fixed, so the capture writes the same calls each run.
"""

from __future__ import annotations

import datetime as dt
import importlib
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from _dialect_helpers import replay_helper
from predictive_query.schema.typing import Dtype

ENGINE = 'sdm_relational_engine.rfm'
BACKENDS = ('databricks', 'duckdb', 'postgres', 'snow', 'sqlite')

#: An identifier of each backend's quoting, by backend.
QUOTE = {
    'databricks': '`',
    'duckdb': '"',
    'postgres': '"',
    'snow': '"',
    'sqlite': '"',
}


def _module(backend: str, name: str) -> Any:
    return importlib.import_module(f'{ENGINE}.backend.{backend}.{name}')


def _q(backend: str, name: str) -> str:
    char = QUOTE[backend]
    return char + name.replace(char, char + char) + char


class Calls:
    r"""Makes the calls of one backend, each under its label."""

    def __init__(self, capture: Any, backend: str) -> None:
        self.capture = capture
        self.backend = backend
        self.dialect = _module(backend, 'dialect')
        self.table = _module(backend, 'table')

    def __call__(self, label: str, name: str, *args: Any, **kwargs: Any) -> Any:
        r"""``name`` of the backend's ``dialect.py`` with these arguments."""
        with self.capture.under(f'synthetic:{label}'):
            return getattr(self.dialect, name)(*args, **kwargs)

    def utils(self, label: str, name: str, *args: Any, **kwargs: Any) -> Any:
        r"""The SQL fragment ``name``, as the backend imports it."""
        for module in (self.dialect, self.table):
            if name in vars(module):
                with self.capture.under(f'synthetic:{label}'):
                    return getattr(module, name)(*args, **kwargs)
        raise KeyError(f'{self.backend} does not import {name}')

    def helper(self, label: str, function: str, /, **args: Any) -> Any:
        r"""The moved helper ``function``, on an object built from
        ``args``."""
        from _codec import encode

        with self.capture.under(f'synthetic:{label}'):
            return replay_helper(
                self.backend,
                function,
                {key: encode(value) for key, value in args.items()},
            )

    def guarded(self, label: str, call: Callable[[], Any]) -> None:
        r"""``call``, which may raise as today's Python refuses its input;
        only calls that return are captured."""
        with self.capture.under(f'synthetic:{label}'):
            try:
                call()
            except Exception:  # noqa: BLE001 - a refusal is not captured
                pass


# --- Values ----------------------------------------------------------------------


#: Text JSON escapes: quotes, backslashes, control characters, DEL, Latin-1,
#: other planes and a byte-order mark.
AWKWARD_TEXT = (
    '',
    'plain',
    'quote " and \\ back',
    'tab\tnew\nline\r\x00\x01\x1f\x7f',
    '\b\f/',
    'café ñ',
    '中文   ﻿',
    '\U0001f600 \U0010ffff',
)

#: Floats whose Python ``repr`` and JSON text are easy to get wrong.
AWKWARD_FLOATS = (
    0.0,
    -0.0,
    0.1,
    1 / 3,
    1e16,
    1e15,
    9999999999999998.0,
    1e22,
    1e-5,
    1e-7,
    0.0001,
    5e-324,
    2.2250738585072014e-308,
    1.7976931348623157e308,
    123.0,
    -2.5e-7,
    float('nan'),
    float('inf'),
    float('-inf'),
)

AWKWARD_INTS = (
    0,
    -1,
    2**31 - 1,
    2**53 + 1,
    2**63 - 1,
    -(2**63),
    2**64,
    -(10**40),
)


def awkward_values() -> list[Any]:
    r"""What a dialect's ``dumps`` writes, through its ``default`` too."""
    return [
        *AWKWARD_INTS,
        *AWKWARD_FLOATS,
        *AWKWARD_TEXT,
        True,
        False,
        None,
        [1, [2.5, ['x', None]], []],
        (1, 'two', (3.0,)),
        {'a': 1, 'é': [True], '': None},
        np.int8(-5),
        np.int64(2**62),
        np.uint64(2**64 - 1),
        np.float32(0.1),
        np.float16(1.5),
        np.float64(1e16),
        np.bool_(True),
        pd.Timestamp('2024-03-01T12:30:05.123456789'),
        pd.Timestamp('2024-03-01T12:30:05', tz='UTC'),
        pd.Timestamp('2024-03-01T12:30:05.5', tz='Asia/Kolkata'),
        pd.Timestamp('1677-09-21T00:12:43.145224193'),
        pd.NaT,
        Decimal('12.50'),
        Decimal('-0'),
        Decimal('1E+3'),
        Decimal('NaN'),
        dt.date(5, 1, 31),
        dt.datetime(2024, 3, 1, 12, 30, 5, 7),
        dt.datetime(2024, 3, 1, 12, 30, 5),
    ]


def _ns(text: str) -> int:
    r"""Nanoseconds since the epoch of ``text``, beyond ``int64`` too."""
    stamp = pd.Timestamp(text)
    per = {'s': 10**9, 'ms': 10**6, 'us': 10**3, 'ns': 1}[stamp.unit]
    return int(stamp.asm8.view('i8')) * per


def _timestamps(unit: str, tz: str | None, texts: list[str | None]) -> pa.Array:
    per = {'s': 10**9, 'ms': 10**6, 'us': 10**3, 'ns': 1}[unit]
    storage = pa.array(
        [None if t is None else _ns(t) // per for t in texts], pa.int64()
    )
    return storage.view(pa.timestamp(unit, tz=tz))


TIMES = [
    '2024-03-10T06:59:59.999999999',
    '2024-03-10T07:00:00',
    '2024-11-03T05:30:00.000123',
    None,
    '1970-01-01T00:00:00',
    '1969-12-31T23:59:59.5',
    '2099-12-31T23:59:59.000000001',
]


def awkward_arrays(backend: str) -> list[pa.Array]:
    r"""Key or time columns of every Arrow type a key list carries."""
    arrays: list[pa.Array] = []
    for unit in ('s', 'ms', 'us', 'ns'):
        for tz in (None, 'UTC', 'America/New_York', 'Europe/London', '+05:30'):
            arrays.append(_timestamps(unit, tz, TIMES))
    arrays.append(
        _timestamps('us', None, ['0001-01-01', '9999-12-31T23:59:59'])
    )
    arrays += [
        pa.array([0, -719162, 2932896, None], pa.int32()).view(pa.date32()),
        pa.array([0, -62135596800000, None], pa.int64()).view(pa.date64()),
        pa.array([1.5, None, float('nan'), -0.0, 1e16, 5e-324], pa.float64()),
        pa.array([0.1, None, float('inf')], pa.float32()),
        pa.array([2**63 - 1, -(2**63), None, 0], pa.int64()),
        pa.array([1, -2, None], pa.int8()),
        pa.array([2**64 - 1, 0], pa.uint64()),
        pa.array([7, None], pa.uint16()),
        pa.array(list(AWKWARD_TEXT) + [None], pa.string()),
        pa.array(['large', None], pa.large_string()),
        pa.array([True, False, None]),
        pa.array([], pa.int64()),
        pa.chunked_array([[1, 2], [3]], pa.int64()),
    ]
    if backend == 'duckdb':
        arrays += [
            pa.array(
                [Decimal('1.50'), None, Decimal('-0.01'), Decimal('0.00')],
                pa.decimal128(10, 2),
            ),
            pa.array([Decimal('12345678901234567890')], pa.decimal128(38, 0)),
            pa.array([0, 86399999999, 1, None], pa.int64()).view(
                pa.time64('us')
            ),
            pa.array([b'a\x00b', None], pa.binary()),
        ]
    return arrays


def _columns(*specs: tuple[str, Dtype | None]) -> list[Any]:
    from sdm_relational_engine.rfm.base.source import SourceColumn

    return [
        SourceColumn(
            name=name,
            dtype=dtype,
            is_primary_key=False,
            is_unique_key=False,
            is_nullable=True,
        )
        for name, dtype in specs
    ]


def _parts(backend: str) -> list[Any]:
    from sdm_relational_engine.rfm.base.source import KeyCountPart

    q = lambda name: _q(backend, name)  # noqa: E731
    return [
        KeyCountPart(0, q('users'), (q('user_id'),), True),
        KeyCountPart(1, q('orders'), (q('user_id'), q('day')), False),
        KeyCountPart(2, q('lines'), (q('a'), q('b'), q('c')), True),
    ]


# --- Every backend ----------------------------------------------------------------


def common(c: Calls) -> None:
    b = c.backend
    q = lambda name: _q(b, name)  # noqa: E731
    table = q('shop') if b != 'postgres' else f'{q("s")}.{q("shop")}'
    identity = f'{q("a")} || {q("b")}'
    c('holds-any-none', 'holds_any', table, identity, [])
    c('holds-any-several', 'holds_any', table, identity, ['1|2', '3|4'])
    c('holds-any-awkward', 'holds_any', table, identity, list(AWKWARD_TEXT))
    c('composite-keys', 'key_counts', _parts(b))
    c('composite-keys', 'key_counts', _parts(b)[:1])
    c('composite-keys', 'repeated_key', table, [q('user_id'), q('day')])
    c('composite-keys', 'repeated_key', table, [q('user_id')])
    for columns in ([q('a')], [q('a'), q('b')], [q('a'), q('b'), q('c')]):
        c.utils('composite-keys', 'count_distinct_values', columns)
        c.utils('composite-keys', 'repeated_key_select', table, columns)
        for unique in (True, False):
            c.utils(
                'composite-keys',
                'key_count_select',
                3,
                table,
                columns,
                unique,
                'COUNT(DISTINCT x)',
            )
    projections = [q('id'), f'{q("price")} * {q("qty")} AS {q("total")}']
    if b in ('duckdb', 'sqlite'):
        c('expression-columns', 'first_rows', table, projections, 1000)
    if 'quote_ident' in vars(c.dialect):
        for name in ('a', 'two words', 'x"y', 'x`y', "it's", ''):
            c('quoting', 'quote_ident', name)
            c('quoting', 'quote_ident', name, "'")
            c('quoting', 'quote_ident', name, '`')


def warehouse_sample(c: Calls, source: str, label: str) -> None:
    b = c.backend
    q = lambda name: _q(b, name)  # noqa: E731
    projections = [q('id'), f'{q("price")} * {q("qty")} AS {q("total")}']
    hash_name = {'databricks': 'xxhash64', 'snow': 'HASH'}.get(b, 'md5')
    columns = _columns(
        ('id', Dtype.int),
        ('name', Dtype.string),
        ('at', Dtype.date),
        ('tags', Dtype.stringlist),
        ('blob', Dtype.binary),
        ('odd', Dtype.unsupported),
        ('unknown', None),
        ('price', Dtype.float),
    )
    quote = lambda name: _q(b, name)  # noqa: E731
    terms: list[str] = []
    if 'sample_order_terms' in vars(c.table):
        for primary_key in (None, 'id', 'tags', 'missing'):
            terms = c.utils(
                label,
                'sample_order_terms',
                columns,
                primary_key,
                quote,
                hash_name,
            )
        many = _columns(*[(f'c{i}', Dtype.string) for i in range(5)])
        many += _columns(*[(f'n{i}', Dtype.int) for i in range(6)])
        c.utils(label, 'sample_order_terms', many, None, quote, hash_name)
        c.utils(label, 'sample_order_terms', [], None, quote, hash_name)
        c.utils(
            label, 'sample_order_terms', columns[3:4], None, quote, hash_name
        )
    else:
        terms = [f'md5({q("id")})', q('id')]
    c('expression-columns', 'sample', source, projections, terms, 1000)
    c(label, 'sample', source, [q('id')], [], 5)
    c.utils(label, 'ordered_sample_sql', source, [q('id')], [], 5)
    c.utils(label, 'ordered_sample_sql', source, projections, terms, 10)


def time_ranges(c: Calls) -> None:
    b = c.backend
    q = lambda name: _q(b, name)  # noqa: E731
    c('time-ranges', 'time_ranges', [])
    c('time-ranges', 'time_ranges', [(q('a'), q('t')), (q('b'), q('u'))])


# --- SQLite --------------------------------------------------------------------------


def sqlite(c: Calls) -> None:
    common(c)
    time_ranges(c)
    c('catalog', 'list_tables')
    c('catalog', 'index_info', 'idx "one"')
    for seed in (0, 1, 7, 2**32 - 1, 2**32, 2**64 + 1, 10**40):
        c('seeded-draws', 'seeded_hash', seed, '"id"')
    c('entity-rows', 'entity_rows', '"t"', ['"a"'], [], num_rows=5)
    c(
        'entity-rows',
        'entity_rows',
        '"t"',
        ['"a"'],
        [' "a" > 1'],
        order_by='x',
        num_rows=5,
    )
    c(
        'entity-rows',
        'entity_rows',
        '"t"',
        ['"a"', '"b"'],
        [' "a" > 1', ' "b" IS NOT NULL'],
        order_by='"a"',
        key='"a"',
        ids=[1, 'two', 3.5, None, True],
    )
    c(
        'entity-rows',
        'entity_rows',
        '"t"',
        ['"a"'],
        [],
        order_by='"a"',
        key='"a"',
        ids=[],
    )
    c('time-spelling', 'first_time', '"t"', '"at"')
    for array in awkward_arrays('sqlite'):
        c.guarded('key-lists', lambda a=array: c.dialect._sqlite_type(a.type))
        c.guarded('key-lists', lambda a=array: c.dialect._json_values(a))
        c.guarded(
            'key-lists',
            lambda a=array: c.dialect._keys([('__kumo_id__', a)]),
        )
    ids = pa.array([1, 2, 3], pa.int64())
    text_ids = pa.array(['a', 'b"c', None], pa.string())
    empty = pa.array([], pa.int64())
    for key_ids in (ids, text_ids, empty):
        for unique, tie_break, count in (
            (True, [], False),
            (False, ['"x" DESC', '"y"'], True),
            (False, [], False),
        ):
            c(
                'rows-by-key',
                'rows_by_key',
                '"t"',
                '"k"',
                key_ids,
                ['"a"', '"b"'],
                unique=unique,
                tie_break=tie_break,
                count_matches=count,
            )
    times = pa.array(['2024-01-01 00:00:00', None, '2024-01-02 00:00:00'])
    c(
        'neighbors',
        'neighbors',
        '"t"',
        '"k"',
        ids,
        ['"a"'],
        num_neighbors=3,
        order_by='"a"',
    )
    c(
        'neighbors',
        'neighbors',
        '"t"',
        '"k"',
        ids,
        ['"a"'],
        num_neighbors=3,
        order_by='"a"',
        time='"at"',
        end_times=times,
    )
    for start in (None, times):
        c(
            'window',
            'window',
            '"t"',
            '"k"',
            ids,
            ['"a"'],
            time='"at"',
            end_times=times,
            start_times=start,
            order_by='"at"',
        )
    for found in (
        None,
        [],
        [()],
        [(None,)],
        [(5,)],
        [('2024-01-01 10:00:00',)],
        [('2024-01-01T10:00:00.123Z',)],
        [('2024-01-01T10:00:00.123456789+00:00',)],
        [('2024-01-01 10:00:00+0000',)],
        [('2024-01-01 10:00:00-05:00',)],
        [('2024-01-01 10:00:00.5 UTC\n',)],
        [('2024-01-01 10:00:00\nmore',)],
        [('2024-01-01',)],
        [('٢٠٢٤-01-01 10:00:00.٣',)],
    ):
        c.helper(
            'time-spelling',
            'SQLiteSampler._probe_time_spelling',
            table_name='orders',
            time_column='at',
            found=found,
        )
    c.helper(
        'time-spelling',
        'SQLiteSampler._probe_time_spelling',
        table_name='users',
        time_column=None,
        found=None,
    )
    stamps = pd.Series(
        pd.to_datetime(
            [
                '2024-01-01 00:00:00.123456789',
                None,
                '1999-12-31 23:59:59.5',
                '2262-04-11 23:47:16.854775807',
                '1677-09-21 00:12:43.145224193',
            ]
        ),
        name='ANCHOR_TIMESTAMP',
    )
    micro = pd.Series(
        pd.to_datetime(
            ['2024-01-01 00:00:00.000001', '2024-02-29'], format='ISO8601'
        )
    ).astype('datetime64[us]')
    zoned = pd.Series(pd.to_datetime(['2024-03-10 06:59:59'])).dt.tz_localize(
        'America/New_York', nonexistent='shift_forward'
    )
    for times_series in (stamps, micro, zoned, stamps.iloc[:0]):
        for separator, digits, suffix in (
            (' ', 0, ''),
            ('T', 3, 'Z'),
            (' ', 6, '+00:00'),
            ('T', 9, ''),
            ('T', 1, ''),
            ('T', 12, '+0000'),
        ):
            c.helper(
                'time-spelling',
                '_TimeSpelling.spell',
                separator=separator,
                digits=digits,
                suffix=suffix,
                times=times_series,
            )


# --- DuckDB --------------------------------------------------------------------------


def duckdb(c: Calls) -> None:
    common(c)
    time_ranges(c)
    c('catalog', 'list_tables')
    c('catalog', 'count_rows', '"t"')
    c('catalog', 'key_constraints', "it's")
    c('catalog', 'foreign_keys', "it's")
    for zoned in (False, True):
        c('zoned-bounds', 'time_column', '"at"', zoned)
    c('zoned-bounds', 'time_column', '"at"')
    for seed in (None, 0, 2**40):
        c(
            'random-rows',
            'random_rows',
            '"t"',
            ['"a"'],
            [],
            num_rows=5,
            random_seed=seed,
            order_by='"a"',
        )
    c(
        'random-rows',
        'random_rows',
        '"t"',
        ['"a"'],
        [' "a" > 1', ' "b" IS NOT NULL'],
        num_rows=5,
        random_seed=3,
        order_by='"a"',
    )
    for ids in ([1, 'two', 2.5, None], []):
        c(
            'rows-by-ids',
            'rows_by_ids',
            '"t"',
            ['"a"'],
            [' "a" > 1'],
            key='"k"',
            ids=ids,
            order_by='"a"',
        )
    arrays = awkward_arrays('duckdb')
    for array in arrays:
        c.guarded('key-lists', lambda a=array: c.dialect._duckdb_type(a.type))
        c.guarded('key-lists', lambda a=array: c.dialect._json_values(a))
    for array in arrays:
        c.guarded(
            'key-lists',
            lambda a=array: c.dialect._keys(
                [('__kumo_id__', a), ('__kumo_end__', a)]
            ),
        )
    ids = pa.array([1, 2, 3], pa.int64())
    naive = _timestamps('us', None, TIMES[:3])
    zoned = _timestamps('ns', 'UTC', TIMES[:3])
    for key_ids in (ids, pa.array([], pa.int64())):
        c(
            'rows-by-key',
            'rows_by_key',
            '"t"',
            '"k"',
            key_ids,
            ['"a"'],
            order_by='"a"',
        )
    c(
        'neighbors',
        'neighbors',
        '"t"',
        '"k"',
        ids,
        ['"a"'],
        num_neighbors=4,
        order_by='"a"',
    )
    for ends in (naive, zoned):
        c(
            'neighbors',
            'neighbors',
            '"t"',
            '"k"',
            ids,
            ['"a"'],
            num_neighbors=4,
            order_by='"a"',
            time='"at"',
            end_times=ends,
        )
        for start in (None, ends):
            c(
                'zoned-bounds',
                'window',
                '"t"',
                '"k"',
                ids,
                ['"a"'],
                time='"at"',
                end_times=ends,
                start_times=start,
                order_by='"at"',
            )
    for source_column, is_source, source_type, zoned_expr in (
        (True, True, 'TIMESTAMP WITH TIME ZONE', False),
        (True, True, 'TIMESTAMPTZ', False),
        (True, True, 'TIMESTAMP', True),
        (True, False, 'TIMESTAMPTZ', False),
        (True, False, None, True),
        (False, None, None, True),
        (False, None, None, False),
    ):
        c.helper(
            'zoned-bounds',
            'DuckDBTable._is_zoned_time',
            name='at',
            source_column=source_column,
            is_source=is_source,
            source_type=source_type,
            zoned_expr=zoned_expr,
        )


# --- PostgreSQL ------------------------------------------------------------------------


def postgres(c: Calls) -> None:
    common(c)
    time_ranges(c)
    c('catalog', 'current_schema')
    c('catalog', 'list_tables', 'parity_shop')
    warehouse_sample(c, '"s"."t"', 'scan-caps')
    block = c('scan-caps', 'block_sample', '"s"."t"', '0.78125')
    warehouse_sample(c, block, 'scan-caps')
    for wanted in (
        Fraction(0),
        Fraction(-1),
        Fraction(1, 8),
        Fraction(1, 7),
        Fraction(2048, 16384),
        Fraction(2048, 16385),
        Fraction(2048, 10**9),
        Fraction(1, 2**40),
        Fraction(3, 1),
    ):
        fraction = c.utils('scan-caps', 'prefilter_fraction', wanted)
        if fraction is not None:
            c.utils('scan-caps', 'percent_literal', fraction)
    for primary_key, relation in (
        (None, [('r', 1_000_000)]),
        (None, [('p', 16_384)]),
        (None, [('m', 16_383)]),
        (None, [('r', 16_385)]),
        (None, [('v', 1_000_000)]),
        (None, [('r', None)]),
        (None, []),
        (None, None),
        ('id', [('r', 1_000_000)]),
    ):
        c.helper(
            'scan-caps',
            'PostgresTable._prefiltered_sample_source',
            schema='parity_shop',
            source_name='big "one"',
            primary_key=primary_key,
            relation=relation,
            min_blocks=16_384,
            target_blocks=2_048,
        )
    c('scan-caps', 'relation_size', '"s"."t"')
    c('catalog', 'count_rows', '"s"."t"')
    for dtype in (Dtype.int, Dtype.float, Dtype.string, Dtype.date, Dtype.bool):
        c('key-lists', 'element_type', dtype)
    for value, dtype, postgres_type in (
        (np.int64(5), Dtype.int, 'bigint'),
        ('17', Dtype.int, 'integer'),
        (2.5, Dtype.float, 'numeric(10, 2)'),
        (np.float32(0.1), Dtype.float, 'DECIMAL'),
        (2.5, Dtype.float, 'double precision'),
        (7, Dtype.string, 'text'),
        ('x', Dtype.date, 'text'),
    ):
        c('key-lists', 'coerce_id', value, dtype, postgres_type)
    for value in awkward_values():
        c.guarded('json', lambda v=value: c.dialect.dumps([v]))
    c.guarded('json', lambda: c.dialect.dumps(awkward_values()))
    for seed in (0, 7, -3, 2**63, 10**40):
        c('seeded-draws', 'set_seed', seed)
    for value, utc in (
        (pd.Timestamp('2024-12-29T09:00:00'), True),
        (pd.Timestamp('2024-12-29T09:00:00'), False),
        (pd.Timestamp('2024-12-29T09:00:00.000000001'), True),
        (pd.Timestamp('2024-12-29T09:00:00.5', tz='America/New_York'), True),
        (pd.Timestamp('2024-12-29T09:00:00.25', tz='Asia/Kolkata'), False),
        (pd.NaT, True),
        (None, False),
    ):
        c('zoned-bounds', 'serialize_time', value, utc)
    for ref, is_string in (('"a"', True), ('"a"', False)):
        c('tie-breaks', 'tie_break_term', ref, is_string)
    for fields in ((), ('e',), ('e', 's'), ('s',)):
        for time_type in ('timestamp', 'timestamp with time zone', 'text'):
            c('key-lists', 'json_cte', 'bigint', fields, time_type)
    c('random-rows', 'random_rows', '"s"."t"', ['"a"'], [], 5)
    c(
        'random-rows',
        'random_rows',
        '"s"."t"',
        ['"a"'],
        ['"a" > 1', '"b" > 2'],
        5,
    )
    for ids in ([1, 2, 3], [], ['a"b', 'é']):
        c(
            'rows-by-ids',
            'rows_by_ids',
            '"s"."t"',
            ['"a"'],
            ['"a" > 1'],
            key='"k"',
            postgres_type='bigint',
            ids=ids,
            order_by='"a"',
        )
        c(
            'rows-by-ids',
            'rows_by_ids',
            '"s"."t"',
            ['"a"'],
            [],
            key='"k"',
            postgres_type='text',
            ids=ids,
            order_by='"a"',
        )
        c(
            'empty-key-lists',
            'rows_by_key',
            '"s"."t"',
            ['"a"'],
            key='"k"',
            postgres_type='bigint',
            ids=ids,
            order_by='"a"',
        )
    rows = [
        {
            'id': 1,
            'e': '2024-01-01T00:00:00+00:00',
            's': '2023-12-01T00:00:00+00:00',
        }
    ]
    for fields, time in (((), None), (('e',), '"at"'), (('e', 's'), '"at"')):
        c(
            'neighbors',
            'neighbors',
            '"s"."t"',
            ['"a"'],
            key='"k"',
            postgres_type='bigint',
            rows=rows,
            fields=fields,
            time=time,
            time_type='timestamp with time zone',
            order_by='"a"',
            num_neighbors=3,
        )
    for fields in (('e',), ('e', 's')):
        for time_type in ('timestamp with time zone', 'text'):
            c(
                'zoned-bounds',
                'window',
                '"s"."t"',
                ['"a"'],
                key='"k"',
                postgres_type='bigint',
                rows=rows,
                fields=fields,
                time='"at"',
                time_type=time_type,
                order_by='"a"',
            )
    chunks(c)
    for dtype in (Dtype.date, Dtype.string, None):
        c.helper(
            'zoned-bounds',
            'PostgresSampler._utc_bounds',
            table_name='orders',
            column='at',
            dtype=dtype,
        )


def chunks(c: Calls) -> None:
    r"""Key lists in chunks: none, one, and more than one 900 KB chunk."""
    c('empty-key-lists', 'chunk_rows', [])
    c('key-lists', 'chunk_rows', [1, 2, 3], 10)
    c('key-lists', 'chunk_rows', [1, 'two', [3, 4]], 9)
    c('key-lists', 'chunk_rows', [12345, 6], 1)
    # Each row is 37,501 characters outside the BMP, which JSON writes as
    # twelve bytes each: two rows are over 900 KB.
    wide = '\U0001f600' * 37_501
    c('two-chunks', 'chunk_rows', [wide, wide, 'x'], 900_000)


# --- Databricks ----------------------------------------------------------------------


def databricks(c: Calls) -> None:
    common(c)
    time_ranges(c)
    c('catalog', 'current_namespace')
    c('catalog', 'list_tables', '`main`', 'parity_shop')
    c('catalog', 'describe_detail', '`c`.`s`.`t`')
    c('catalog', 'describe_extended', '`c`.`s`.`t`')
    c('catalog', 'count_rows', '`t`')
    warehouse_sample(c, '`c`.`s`.`t`', 'scan-caps')
    bounded = c('scan-caps', 'bounded', '`c`.`s`.`t`', 100_000)
    warehouse_sample(c, bounded, 'scan-caps')
    for detail_size, described in (
        (2**30, None),
        (2**30 - 1, None),
        (0, None),
        (None, {'Type': 'VIEW'}),
        (None, {'Type': ' view '}),
        (None, {'Type': 'MANAGED', 'Statistics': '1073741824 bytes, 10 rows'}),
        (None, {'Type': 'MANAGED', 'Statistics': '12 bytes'}),
        (None, {'Type': 'MANAGED', 'Statistics': 'about 5 bytes'}),
        (None, {'Type': 'MANAGED'}),
        (None, {}),
        (None, None),
    ):
        c.helper(
            'scan-caps',
            'DatabricksTable._is_large',
            detail_size=detail_size,
            described=described,
            min_bytes=1 << 30,
        )
    for dtype in (Dtype.int, Dtype.float, Dtype.string, Dtype.date):
        c('key-lists', 'element_type', dtype)
    for value, dtype in (
        ('5', Dtype.int),
        (np.int64(3), Dtype.float),
        (7, Dtype.string),
    ):
        c('key-lists', 'coerce_id', value, dtype)
    for value in awkward_values():
        c.guarded('json', lambda v=value: c.dialect.dumps([v]))
    c.guarded('json', lambda: c.dialect.dumps(awkward_values()))
    chunks(c)
    c(
        'entity-rows',
        'entity_rows',
        '`t`',
        ['`a`'],
        [],
        key='`k`',
        order_by='`a`',
        percent=12.5,
        num_rows=10,
    )
    c(
        'entity-rows',
        'entity_rows',
        '`t`',
        ['`a`'],
        [' `a` > 1', ' `b` IS NOT NULL'],
        key='`k`',
        order_by='`a`',
        percent=0.001,
        random_seed=2**63,
        num_rows=10,
    )
    for dtype, ids in (
        (Dtype.int, [1, '2', np.int64(3)]),
        (Dtype.string, ['a', 5]),
        (Dtype.float, []),
    ):
        c(
            'entity-rows',
            'entity_rows',
            '`t`',
            ['`a`'],
            [' `a` > 1'],
            key='`k`',
            order_by='`a`',
            ids=ids,
            dtype=dtype,
        )
    for dtype, ids in (
        (Dtype.int, [1, 2]),
        (Dtype.string, []),
        (Dtype.float, [2.5]),
    ):
        c(
            'empty-key-lists',
            'rows_by_key',
            '`t`',
            ['`a`'],
            key='`k`',
            dtype=dtype,
            ids=ids,
            order_by=['`a`', '`b` DESC'],
        )
    rows = [
        {
            'id': 1,
            'e': '2024-01-01 00:00:00+00:00',
            's': '2023-12-01 00:00:00+00:00',
        }
    ]
    bounds = ['2024-01-01 00:00:00+00:00', '2023-12-01 00:00:00+00:00']
    for windowed in (False, True):
        c(
            'neighbors',
            'neighbors',
            '`t`',
            ['`a`'],
            key='`k`',
            dtype=Dtype.string,
            rows=rows if windowed else [1, 2],
            windowed=windowed,
            time='`at`' if windowed else None,
            bounds=bounds if windowed else [],
            order_by='`a`',
            num_neighbors=4,
        )
    for with_start in (False, True):
        c(
            'zoned-bounds',
            'window',
            '`t`',
            ['`a`'],
            key='`k`',
            dtype=Dtype.int,
            rows=rows,
            with_start=with_start,
            time='`at`',
            bounds=bounds if with_start else bounds[:1],
            order_by='`a`',
        )
    for dtype in (Dtype.date, Dtype.string, None):
        c.helper(
            'zoned-bounds',
            'DatabricksSampler._bound_format',
            table_name='orders',
            column='at',
            dtype=dtype,
        )


# --- Snowflake -----------------------------------------------------------------------


def snow(c: Calls) -> None:
    common(c)
    time_ranges(c)
    c('catalog', 'current_namespace')
    c('catalog', 'list_tables', 'TEST', 'PARITY_SHOP')
    c('catalog', 'list_tables', 'my "db"', 'S')
    c('catalog', 'tables_in_schema', 'TEST', 'my "schema"')
    c('catalog', 'tables_like', 'TEST', 'S', 'orders_')
    c('catalog', 'semantic_view', 'v')
    c('catalog', 'describe_table', '"t"')
    c('catalog', 'imported_keys', '"t"')
    warehouse_sample(c, '"D"."S"."T"', 'scan-caps')
    clause = c('scan-caps', 'block_sample', '0.78125', 7)
    warehouse_sample(c, f'"D"."S"."T" {clause}', 'scan-caps')
    for wanted in (
        Fraction(1, 8),
        Fraction(1, 7),
        Fraction(100_000, 10**9),
        Fraction(0),
    ):
        fraction = c.utils('scan-caps', 'prefilter_fraction', wanted)
        if fraction is not None:
            c.utils('scan-caps', 'percent_literal', fraction)
    standard = {
        'rows': 10**9,
        'bytes': 64 << 30,
        'kind': 'TABLE',
        'is_external': 'N',
        'is_hybrid': 'N',
        'is_iceberg': 'N',
        'is_dynamic': 'N',
    }
    kinds = ['TABLE', 'TEMPORARY', 'TRANSIENT']
    for rows, seed, listing in (
        (100_000, 0, standard),
        (100_000, 2**31 + 5, standard),
        (100_000, 2**70, standard),
        (10**8, 0, standard),
        (100_000, 0, {**standard, 'bytes': 1 << 20}),
        (100_000, 0, {**standard, 'rows': 0}),
        (100_000, 0, {**standard, 'rows': '1000000000', 'bytes': ' 42 '}),
        (100_000, 0, {**standard, 'rows': None}),
        (100_000, 0, {**standard, 'rows': 'many'}),
        (100_000, 0, {**standard, 'rows': 1e9}),
        (100_000, 0, {**standard, 'kind': 'view'}),
        (100_000, 0, {**standard, 'kind': ' transient '}),
        (100_000, 0, {**standard, 'is_hybrid': 'Y'}),
        (100_000, 0, {**standard, 'is_iceberg': ' y'}),
        (100_000, 0, None),
    ):
        c.helper(
            'scan-caps',
            'SnowTable._prefilter',
            rows=rows,
            seed=seed,
            listing=listing,
            sampled_kinds=kinds,
            prefilter_partitions=128,
            micro_partition_bytes=16 << 20,
        )
    fields = list(standard)
    for source_name, listed in (
        ('ORDERS', [{'name': 'ORDERS', **standard}]),
        ('orders', [{'name': 'ORDERS', **standard}]),
        (
            'orders',
            [{'name': 'ORDERS', **standard}, {'name': 'orders', **standard}],
        ),
        (
            'Orders',
            [{'name': 'ORDERS', **standard}, {'name': 'orders', **standard}],
        ),
        ('ORDERS', [{'name': 'ORDERSX', **standard}]),
        ('ORDERS', [{'name': 'ORDERS', 'rows': 1}]),
        ('ORDERS', [{'name': None, **standard}]),
        ('ORDERS', []),
        ('ORDERS', None),
    ):
        c.helper(
            'scan-caps',
            'SnowTable._listing',
            source_name=source_name,
            listed=listed,
            fields=fields,
        )
    for dtype, is_date in (
        (Dtype.string, False),
        (Dtype.date, True),
        (Dtype.date, False),
        (Dtype.time, False),
        (Dtype.binary, False),
        (Dtype.int, False),
        (None, False),
    ):
        c('seeded-draws', 'seeded_key_hash', '"k"', dtype, is_date, 2**65)
    seeded = c('seeded-draws', 'seeded_key_hash', '"k"', Dtype.int, False, 7)
    c(
        'entity-rows',
        'entity_rows',
        '"t"',
        ['"a"'],
        [],
        order_by='"a"',
        num_rows=5,
    )
    c(
        'entity-rows',
        'entity_rows',
        '"t"',
        ['"a"'],
        [' "a" > 1', ' "b" IS NOT NULL'],
        order_by='"a"',
        num_rows=5,
        seeded=seeded,
        prefilter=clause,
    )
    c(
        'entity-rows',
        'entity_rows',
        '"t"',
        ['"a"'],
        [],
        order_by='"a"',
        num_rows=5,
        seeded=seeded,
    )
    c(
        'entity-rows',
        'entity_rows',
        '"t"',
        ['"a"'],
        [],
        order_by='"a"',
        num_rows=5,
        prefilter=clause,
    )
    for ids in ([1, 'two', 2.5, None], []):
        c(
            'entity-rows',
            'entity_rows',
            '"t"',
            ['"a"'],
            [' "a" > 1'],
            order_by='"a"',
            num_rows=5,
            key='"k"',
            ids=ids,
        )
    for dtype in (Dtype.int, Dtype.float, Dtype.string, Dtype.date):
        c('key-lists', '_cast', dtype)
    for ids in ([1, 2], [], ['é', 'a"b', [1, (2, 3)], {'k': None}]):
        c(
            'empty-key-lists',
            'rows_by_key',
            '"t"',
            ['"a"'],
            key='"k"',
            dtype=Dtype.string,
            ids=ids,
            order_by=['"a"'],
        )
    for spelled in (
        'TIMESTAMP_LTZ(9)',
        ' timestamp_tz ',
        'TIMESTAMP_NTZ(9)',
        'DATE',
        '',
        None,
    ):
        c('zoned-bounds', 'zone_of', spelled)
    rows = [(1, '2024-01-01 00:00:00', '2023-12-01 00:00:00')]
    bounds = ['2024-01-01 00:00:00', '2023-12-01 00:00:00']
    for zone in (None, 'LTZ', 'TZ'):
        c('zoned-bounds', '_bound', 'x', zone)
        c('zoned-bounds', '_outer_bound', zone)
        for index in (1, 2):
            c('zoned-bounds', '_key_list_time', index, zone)
        for windowed in (False, True):
            c(
                'zoned-bounds',
                'neighbors',
                '"t"',
                ['"a"'],
                key='"k"',
                dtype=Dtype.int,
                rows=rows,
                windowed=windowed,
                time='"at"' if windowed else None,
                bounds=bounds if windowed else [],
                order_by='"a"',
                num_neighbors=4,
                zone=zone,
            )
        for with_start in (False, True):
            c(
                'zoned-bounds',
                'window',
                '"t"',
                ['"a"'],
                key='"k"',
                dtype=Dtype.string,
                rows=rows,
                with_start=with_start,
                time='"at"',
                bounds=bounds if with_start else bounds[:1],
                order_by='"a"',
                zone=zone,
            )
    for source_column, zoned_expr, source_type in (
        (True, False, 'TIMESTAMP_LTZ(9)'),
        (True, False, 'TIMESTAMP_TZ(9)'),
        (True, False, 'TIMESTAMP_NTZ(9)'),
        (True, True, None),
        (False, True, None),
        (False, False, None),
    ):
        for name in ('SnowSampler._zone', 'SnowSampler._bound_format'):
            c.helper(
                'zoned-bounds',
                name,
                table_name='orders',
                column='at',
                source_column=source_column,
                zoned_expr=zoned_expr,
                source_type=source_type,
            )


# --- View expressions -------------------------------------------------------------------

#: An expression for each answer the checks give, and a few they accept.
VIEW_EXPRESSIONS = (
    'price * qty',
    'CASE WHEN x > 1 THEN upper(name) ELSE NULL END',
    'CAST(x AS DECIMAL(10, 2))',
    'x::VARCHAR(10)',
    "x NOT LIKE ('a%')",
    "x LIKE y ESCAPE ('!')",
    'x IS DISTINCT FROM (y)',
    'NOT exists(arr, v -> v > 1)',
    'a; DROP TABLE t',
    'x -- note',
    '/* c */ x',
    'FROM t |> SELECT x',
    'x ->> y',
    '$$select 1$$',
    "'it''s' || name",
    "'a\\'b'",
    '"na\\me"',
    "'unterminated",
    '"unterminated',
    '`unterminated',
    '$$unterminated',
    '(x',
    'x)',
    'select(1)',
    'ſelect 1',
    'ORDERS.MERGE + 1',
    'insert(x, 1, 2, y)',
    'insert into t',
    'truncate(x, 2)',
    '(FROM t)',
    '(TABLE t)',
    'secret(x)',
    'SYSTEM$WAIT(1)',
    'my_udf(x)',
    'a.upper(x)',
    '"q"(x)',
    '`q`(x)',
    'IDENTIFIER(x)',
    '(x)(y)',
    'm[0](x)',
    '1 + (2)',
    'current_user',
    'CURRENT_DATE',
    'current_date()',
    'localtimestamp + 1',
    'escape(x)',
    'NOT escape(secret())',
    'x DIV escape(y)',
    'iff(x, 1, 2)',
    'div0(x, y)',
    'named_struct(1)',
    'xxhash64(x)',
    'İnsert into t',
    'x # (y)',
)


def view_expressions(capture: Any) -> None:
    from sdm_relational_engine.rfm import graph

    with capture.under('synthetic:view-expressions'):
        for expr in VIEW_EXPRESSIONS:
            for dialect in ('databricks', 'snowflake'):
                if graph._unsafe_expr_reason(expr, dialect) is None:
                    graph._untrusted_call_reason(expr, dialect, columns=())


def synthetic(capture: Any) -> None:
    r"""Make every synthetic call, in a fixed order."""
    makers = {
        'databricks': databricks,
        'duckdb': duckdb,
        'postgres': postgres,
        'snow': snow,
        'sqlite': sqlite,
    }
    for backend in BACKENDS:
        makers[backend](Calls(capture, backend))
    view_expressions(capture)
