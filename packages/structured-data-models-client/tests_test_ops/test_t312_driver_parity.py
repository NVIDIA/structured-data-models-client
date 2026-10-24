# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Java's JDBC providers against Python's, value by value (T3.12 fix 1).

The rule: for the same database value, Java's provider hands the core
exactly what Python's provider hands it, or refuses where Python's driver
(psycopg, ADBC SQLite, DuckDB's) refuses. Each case below is one statement
over literal values: every type the conversion table names, nulls, edge
values, and the values the reference's drivers refuse. Each runs through
the Python provider and through the Java program's provider (its ``fetch``
call) on the same database, several times on one connection (so a driver
that changes how it reads from a later call, as pgjdbc does from its fifth
prepared run, is caught), plain and binding a key set; each answer must be
the Python provider's, its schema's metadata included, or both must fail.

Postgres runs in a database of its own, with the session's zone of both
connections the JVM's (pgjdbc sets the JVM's zone; the Python connection is
given the same, so a value spelled in the session's zone is compared on the
same zone).

It needs the JAR under test (``SDM_JAVA_JAR``, ``SDM_JAVA_NATIVE``) and, for
Postgres, ``SDM_POSTGRES_TEST_DSN``; without them it returns without
asserting, unless ``SDM_REQUIRE_JAVA`` or ``SDM_REQUIRE_POSTGRES`` is set.
"""

from __future__ import annotations

import base64
import importlib.util
import math
import os
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa
import pytest

ROOT = Path(__file__).resolve().parents[3]


def _module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RUNNER = _module('t312_parity_run_java', ROOT / 'contract/runners/run_java.py')

#: The JVM's zone, and so the zone of both Postgres sessions.
ZONE = 'Asia/Kolkata'
#: Runs of each statement on one connection: past pgjdbc's prepare
#: threshold (5), after which it reads some types in binary.
RUNS = 7

#: A JSON value nested this deep: past Java's old limit (900), within the
#: depth Python's reader and the core read.
DEEP = 920

POSTGRES_SETUP = [
    "CREATE TYPE mood AS ENUM ('sad', 'happy')",
    'CREATE TYPE pair AS (a int4, b text)',
    'CREATE DOMAIN positive AS int4 CHECK (VALUE > 0)',
]


def _deep(depth: int) -> str:
    r"""SQL text of a JSON array nested ``depth`` deep around ``0``, spaced
    as written, so the server's text is not the compact text."""
    return f"repeat('[ ', {depth}) || '0' || repeat(' ]', {depth})"


def _rows(kind: str, *values: str) -> str:
    r"""A statement of one column ``v`` of type ``kind``, a row per value
    (SQL literal text, or NULL), in order."""
    rows = ', '.join(f'({n}, ({v})::{kind})' for n, v in enumerate(values))
    return f'SELECT v FROM (VALUES {rows}) AS t(n, v) ORDER BY n'


#: Each Postgres case: a name, and its statements (the last one's rows read).
POSTGRES: list[tuple[str, list[str]]] = [
    ('bool', [_rows('bool', 'true', 'false', 'NULL')]),
    ('int2', [_rows('int2', '-32768', '32767', 'NULL')]),
    ('int4', [_rows('int4', '-2147483648', '2147483647', 'NULL')]),
    (
        'int8',
        [_rows('int8', '-9223372036854775808', '9223372036854775807', 'NULL')],
    ),
    ('oid', [_rows('oid', '0', '4294967295', 'NULL')]),
    (
        'float4',
        [
            _rows(
                'float4',
                "'0.1'",
                "'1.1'",
                "'3.4028235e38'",
                "'1e-45'",
                "'1.17549435e-38'",
                "'-0'",
                "'16777217'",
                # An exact tie of the shortest digits, which Postgres
                # writes with nine (Claude T3.12 r2 P1-1).
                "'103528336'",
                "'NaN'",
                "'Infinity'",
                "'-Infinity'",
                'NULL',
            )
        ],
    ),
    (
        'float8',
        [
            _rows(
                'float8',
                "'0.1'",
                "'1e300'",
                "'5e-324'",
                "'-0'",
                "'9007199254740993'",
                "'NaN'",
                "'Infinity'",
                'NULL',
            )
        ],
    ),
    ('bytea', [_rows('bytea', "'\\x00ff'", "''", 'NULL')]),
    (
        'date',
        [_rows('date', "'2024-02-29'", "'0001-01-01'", "'9999-12-31'", 'NULL')],
    ),
    ('date infinity', [_rows('date', "'infinity'")]),
    ('date -infinity', [_rows('date', "'-infinity'")]),
    ('date BC', [_rows('date', "'0044-03-15 BC'")]),
    ('date past 9999', [_rows('date', "'10000-01-01'")]),
    (
        'numeric(10,2)',
        [_rows('numeric(10,2)', "'1.5'", "'-12345678.99'", "'NaN'", 'NULL')],
    ),
    (
        'numeric(38,0)',
        [
            _rows(
                'numeric(38,0)',
                "'99999999999999999999999999999999999999'",
                'NULL',
            )
        ],
    ),
    ('numeric(50,10)', [_rows('numeric(50,10)', "'1.25'", 'NULL')]),
    (
        'numeric',
        [
            _rows(
                'numeric',
                "'1.50'",
                "'0.0000001'",
                "'1e30'",
                "'NaN'",
                "'Infinity'",
                "'-Infinity'",
                'NULL',
            )
        ],
    ),
    ('numeric(5,-2)', [_rows('numeric(5,-2)', "'12345'", 'NULL')]),
    ('numeric(80,10)', [_rows('numeric(80,10)', "'1.5'", "'NaN'", 'NULL')]),
    (
        'timestamp',
        [
            _rows(
                'timestamp',
                "'2024-01-01 12:00:00.5'",
                "'0001-01-01'",
                "'9999-12-31 23:59:59.999999'",
                'NULL',
            )
        ],
    ),
    ('timestamp infinity', [_rows('timestamp', "'infinity'")]),
    ('timestamp BC', [_rows('timestamp', "'0044-03-15 00:00 BC'")]),
    (
        'timestamptz',
        [
            _rows(
                'timestamptz',
                "'2024-01-01 12:00:00.5+00'",
                "'1900-01-01 00:00+00'",
                'NULL',
            )
        ],
    ),
    ('timestamptz infinity', [_rows('timestamptz', "'-infinity'")]),
    ('time', [_rows('time', "'12:34:56.789'", "'00:00'", 'NULL')]),
    ('time 24:00', [_rows('time', "'24:00:00'")]),
    ('timetz', [_rows('timetz', "'12:34:56.5+05:30'", "'00:00-08'", 'NULL')]),
    ('timetz 24:00', [_rows('timetz', "'24:00:00+00'")]),
    (
        'interval',
        [
            _rows(
                'interval',
                "'1 year 2 mons -3 days 04:05:06.789'",
                "'-1 days -00:00:01'",
                "'00:00:00.000001'",
                'NULL',
            )
        ],
    ),
    # psycopg's C loader sums the days in 32 bits, so this one wraps.
    (
        'interval past a timedelta, wrapped',
        [_rows('interval', "'-178000000 years'")],
    ),
    ('interval past a timedelta', [_rows('interval', "'-5000000 years'")]),
    (
        'interval, iso_8601',
        ["SET IntervalStyle = 'iso_8601'", _rows('interval', "'1 day'")],
    ),
    (
        'interval, sql_standard',
        ["SET IntervalStyle = 'sql_standard'", _rows('interval', "'1 day'")],
    ),
    (
        'interval, postgres again',
        ["SET IntervalStyle = 'postgres'", _rows('interval', "'1 day'")],
    ),
    (
        'json',
        [
            _rows(
                'json',
                """'{"a": 1, "b": [1.0, 1E2, -0, "\\u00e9\\ud800"], "a": 2}'""",
                "'null'",
                '\'"text"\'',
                "'12345678901234567890'",
                "'1.5e300'",
                "'true'",
                _deep(DEEP),
                'NULL',
            )
        ],
    ),
    (
        'jsonb',
        [
            _rows(
                'jsonb',
                """'{"b": [1.0, 1E2], "a": {"x": null}}'""",
                "'null'",
                _deep(DEEP),
                'NULL',
            )
        ],
    ),
    # Deeper than Python's recursion limit, which Python's provider reads
    # (its C scanner) and the core reads to 2,000 levels.
    ('json 1500 deep', [_rows('json', _deep(1500))]),
    # Past the core's 2,000 levels (register JV-3).
    ("json past the core's reader", [_rows('json', _deep(2100))]),
    ('text', [_rows('text', "'x'", "''", "'é\\n'", 'NULL')]),
    ('varchar', [_rows('varchar(3)', "'abc'", 'NULL')]),
    ('bpchar', [_rows('char(3)', "'a'", 'NULL')]),
    ('"char"', [_rows('"char"', "'a'", 'NULL')]),
    ('name', [_rows('name', "'n'", 'NULL')]),
    ('uuid', [_rows('uuid', "'A0EEBC99-9C0B-4EF8-BB6D-6BB9BD380A11'", 'NULL')]),
    ('xml', [_rows('xml', "'<a>1</a>'", 'NULL')]),
    ('inet', [_rows('inet', "'192.168.0.1'", "'10.0.0.1/8'", "'::1'", 'NULL')]),
    ('cidr', [_rows('cidr', "'10.0.0.0/8'", "'2001:db8::/32'", 'NULL')]),
    ('macaddr', [_rows('macaddr', "'08:00:2b:01:02:03'", 'NULL')]),
    ('macaddr8', [_rows('macaddr8', "'08:00:2b:01:02:03:04:05'", 'NULL')]),
    ('money', [_rows('money', "'12.5'", 'NULL')]),
    ('bit', [_rows('bit(3)', "'101'", 'NULL')]),
    ('varbit', [_rows('varbit', "'10'", 'NULL')]),
    ('point', [_rows('point', "'(1,2)'", 'NULL')]),
    ('tsvector', [_rows('tsvector', "'a b'", 'NULL')]),
    ('regclass', [_rows('regclass', "'pg_class'", 'NULL')]),
    ('pg_lsn', [_rows('pg_lsn', "'0/16B3748'", 'NULL')]),
    ('enum', [_rows('mood', "'happy'", 'NULL')]),
    ('enum[]', [_rows('mood[]', "'{happy,sad}'", "'{}'", 'NULL')]),
    ('composite', [_rows('pair', "'(1,x)'", 'NULL')]),
    ('composite[]', [_rows('pair[]', "ARRAY['(1,x)'::pair]", 'NULL')]),
    ('domain', [_rows('positive', '1', 'NULL')]),
    ('record', ["SELECT ROW(1, 'x') AS v"]),
    ('int4range', [_rows('int4range', "'[1,5)'", "'empty'", "'(,3]'", 'NULL')]),
    ('int8range', [_rows('int8range', "'[1,)'", 'NULL')]),
    (
        'numrange',
        [
            _rows(
                'numrange',
                "'[1.50,2.5]'",
                "'[1,Infinity]'",
                "'(-Infinity,1)'",
                "'[1,NaN]'",
                'NULL',
            )
        ],
    ),
    (
        'nummultirange',
        [_rows('nummultirange', "'{[1,Infinity], [2,3)}'", 'NULL')],
    ),
    (
        'numrange[]',
        [_rows('numrange[]', "ARRAY['[1,Infinity]'::numrange]", 'NULL')],
    ),
    ('daterange', [_rows('daterange', "'[2024-01-01,2024-02-01)'", 'NULL')]),
    (
        'tsrange',
        [
            _rows(
                'tsrange', "'[2024-01-01 00:00,2024-01-02 12:00:00.5)'", 'NULL'
            )
        ],
    ),
    ('tstzrange', [_rows('tstzrange', "'[2024-01-01 00:00+00,)'", 'NULL')]),
    # A session zone Python's ZoneInfo does not name: psycopg reads the
    # instant in UTC (Codex T3.12 r2 P0-1).
    (
        'tstzrange in GMT+3',
        [
            "SET TimeZone = 'GMT+3'",
            _rows('tstzrange', "'[2024-01-01 00:00+00,2024-01-02 00:00+00)'"),
        ],
    ),
    (
        'tstzrange in +03:00',
        [
            "SET TimeZone = '+03:00'",
            _rows('tstzrange', "'[2024-01-01 00:00+00,)'"),
        ],
    ),
    (
        'timestamptz[] in Europe/Berlin',
        [
            "SET TimeZone = 'Europe/Berlin'",
            _rows('timestamptz[]', '\'{"2024-07-01 00:00+00"}\''),
        ],
    ),
    # Moving the instant to the session's zone leaves years 1 to 9999:
    # psycopg keeps the server's own wall time and offset (Claude T3.12 r4
    # P1-1).
    (
        'timestamptz[] at year 9999 in GMT+3',
        [
            "SET TimeZone = 'GMT+3'",
            _rows('timestamptz[]', '\'{"9999-12-31 23:00:00-03"}\''),
        ],
    ),
    (
        'tstzrange ending at year 9999 in GMT+3',
        [
            "SET TimeZone = 'GMT+3'",
            _rows('tstzrange', "'[9999-12-31 22:00-03,9999-12-31 23:00-03)'"),
        ],
    ),
    (
        'timestamptz[] at year 1 in <+14>-14',
        [
            "SET TimeZone = '<+14>-14'",
            _rows('timestamptz[]', '\'{"0001-01-01 05:00+14"}\''),
        ],
    ),
    # The end of time ("no end"), past the last instant the zone data
    # holds, in zones Python names (Claude T3.12 r5 P0-1).
    (
        'the end of time in America/Los_Angeles',
        [
            "SET TimeZone = 'America/Los_Angeles'",
            'SELECT '
            "'[2020-01-01 00:00+00,9999-12-31 23:59:59+00)'::tstzrange AS r, "
            "'{[2020-01-01 00:00+00,9999-12-31 23:59:59+00)}'::tstzmultirange AS m, "
            '\'{"9999-12-30 22:00:00+00","9999-12-30 22:00:01+00","9999-12-31 23:59:59+00"}\'::timestamptz[] AS a',
        ],
    ),
    (
        'the end of time in America/New_York',
        [
            "SET TimeZone = 'America/New_York'",
            'SELECT '
            "'[2020-01-01 00:00+00,9999-12-31 23:59:59+00)'::tstzrange AS r, "
            "'{[2020-01-01 00:00+00,9999-12-31 23:59:59+00)}'::tstzmultirange AS m, "
            '\'{"9999-12-30 22:00:00+00","9999-12-30 22:00:01+00","9999-12-31 23:59:59+00"}\'::timestamptz[] AS a',
        ],
    ),
    (
        'the end of time in Pacific/Honolulu',
        [
            "SET TimeZone = 'Pacific/Honolulu'",
            'SELECT '
            "'[2020-01-01 00:00+00,9999-12-31 23:59:59+00)'::tstzrange AS r, "
            "'{[2020-01-01 00:00+00,9999-12-31 23:59:59+00)}'::tstzmultirange AS m, "
            '\'{"9999-12-30 22:00:00+00","9999-12-30 22:00:01+00","9999-12-31 23:59:59+00"}\'::timestamptz[] AS a',
        ],
    ),
    (
        'the end of time in Asia/Kolkata',
        [
            "SET TimeZone = 'Asia/Kolkata'",
            'SELECT '
            "'[2020-01-01 00:00+00,9999-12-31 23:59:59+00)'::tstzrange AS r, "
            "'{[2020-01-01 00:00+00,9999-12-31 23:59:59+00)}'::tstzmultirange AS m, "
            '\'{"9999-12-30 22:00:00+00","9999-12-30 22:00:01+00","9999-12-31 23:59:59+00"}\'::timestamptz[] AS a',
        ],
    ),
    (
        'tstzrange back in the JVM zone',
        [
            f"SET TimeZone = '{ZONE}'",
            _rows('tstzrange', "'[2024-01-01 00:00+00,)'"),
        ],
    ),
    (
        'int4multirange',
        [_rows('int4multirange', "'{[1,3),[5,7)}'", "'{}'", 'NULL')],
    ),
    ('bool[]', [_rows('bool[]', "'{t,f,NULL}'", 'NULL')]),
    ('int2[]', [_rows('int2[]', "'{1,-2}'", 'NULL')]),
    ('int4[]', [_rows('int4[]', "'{{1,2},{3,4}}'", "'{}'", 'NULL')]),
    ('int8[]', [_rows('int8[]', "'{9223372036854775807,NULL}'", 'NULL')]),
    ('oid[]', [_rows('oid[]', "'{1}'", 'NULL')]),
    (
        'float4[]',
        [
            _rows(
                'float4[]', "'{0.1,1.1,NaN,-Infinity,1e-45,103528336}'", 'NULL'
            )
        ],
    ),
    ('float8[]', [_rows('float8[]', "'{0.1,1e300,NaN}'", 'NULL')]),
    ('numeric[]', [_rows('numeric[]', "'{1.50,NaN,1e30}'", 'NULL')]),
    ('bytea[]', [_rows('bytea[]', "ARRAY['\\x00ff'::bytea, NULL]", 'NULL')]),
    ('date[]', [_rows('date[]', "'{2024-01-01,NULL}'", 'NULL')]),
    ('date[] infinity', [_rows('date[]', "'{infinity}'")]),
    (
        'timestamp[]',
        [_rows('timestamp[]', '\'{"2024-01-01 12:00:00.5"}\'', 'NULL')],
    ),
    (
        'timestamptz[]',
        [
            _rows(
                'timestamptz[]',
                '\'{"2024-01-01 12:00:00.5+00","1900-01-01 00:00+00"}\'',
                'NULL',
            )
        ],
    ),
    ('time[]', [_rows('time[]', "'{12:00:00.25}'", 'NULL')]),
    ('timetz[]', [_rows('timetz[]', "'{12:00:00.25+02,00:00-08}'", 'NULL')]),
    (
        'interval[]',
        [
            _rows(
                'interval[]',
                '\'{"1 day 02:00:00","-1 days","1 mon","00:00:00.5"}\'',
                'NULL',
            )
        ],
    ),
    (
        'json[]',
        [
            _rows(
                'json[]',
                """ARRAY['{"a": 1}'::json, 'null'::json, NULL]""",
                'NULL',
            )
        ],
    ),
    (
        'jsonb[]',
        [
            _rows(
                'jsonb[]', """ARRAY['[1, 2.0]'::jsonb, 'null'::jsonb]""", 'NULL'
            )
        ],
    ),
    ('text[]', [_rows('text[]', '\'{a,"b c",NULL}\'', 'NULL')]),
    (
        'uuid[]',
        [_rows('uuid[]', "'{a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11}'", 'NULL')],
    ),
    ('inet[]', [_rows('inet[]', "'{192.168.0.1,10.0.0.1/8}'", 'NULL')]),
    ('int4range[]', [_rows('int4range[]', '\'{"[1,5)",empty}\'', 'NULL')]),
    ('pair as the second column', ["SELECT 1 AS n, '(1,x)'::pair AS v"]),
]

#: Each SQLite case.
SQLITE: list[tuple[str, list[str]]] = [
    (
        'integers',
        [
            'SELECT 1 AS v UNION ALL SELECT NULL UNION ALL SELECT 9223372036854775807'
        ],
    ),
    (
        'reals',
        [
            'SELECT 0.1 AS v UNION ALL SELECT 1e300 UNION ALL SELECT -0.0 UNION ALL SELECT 9e999 UNION ALL SELECT NULL'
        ],
    ),
    ('integers and reals', ['SELECT 1 AS v UNION ALL SELECT 2.5']),
    (
        'reals as text, rounded half to even',
        [
            "SELECT 'x' AS v UNION ALL SELECT 1234562.5 UNION ALL SELECT 1234563.5 "
            'UNION ALL SELECT 0.1 UNION ALL SELECT 1e300 UNION ALL SELECT 2.5e-7 '
            'UNION ALL SELECT 9e999 UNION ALL SELECT -9e999 UNION ALL SELECT 1.0000005'
        ],
    ),
    ('text', ["SELECT 'é' AS v UNION ALL SELECT '' UNION ALL SELECT NULL"]),
    (
        'blob',
        [
            "SELECT x'00ff' AS v UNION ALL SELECT 'a' UNION ALL SELECT 2.5 UNION ALL SELECT 3"
        ],
    ),
    (
        'a later row past the first batch',
        [
            'WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 1100) '
            "SELECT CASE WHEN i < 1050 THEN 'x' ELSE i * 0.5 END AS v FROM n"
        ],
    ),
    (
        'a later text in an integer column',
        [
            'WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 1100) '
            "SELECT CASE WHEN i < 1050 THEN i ELSE 'x' END AS v FROM n"
        ],
    ),
    ('no row', ['SELECT 1 AS v WHERE 0']),
]

#: Each DuckDB case.
DUCKDB: list[tuple[str, list[str]]] = [
    (
        'scalars',
        [
            'SELECT true AS b, 1::TINYINT AS i8, 1::UTINYINT AS u8, 1::HUGEINT AS h, '
            '1.5::FLOAT AS f, 0.1::DOUBLE AS d, 1.25::DECIMAL(10,2) AS dec, '
            "1.25::DECIMAL(38,10) AS dec38, 'x'::VARCHAR AS s, 'ab'::BLOB AS bl, "
            "DATE '2024-01-01' AS dt, TIMESTAMP '2024-01-01 12:00:00.5' AS ts, "
            "TIMESTAMPTZ '2024-01-01 12:00:00+00' AS tz, TIME '12:00:00' AS t, "
            "INTERVAL '1 month 2 days' AS iv, 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'::UUID AS u"
        ],
    ),
    (
        'nested',
        [
            "SELECT [1, NULL, 3] AS l, {'a': 1, 'b': 'x'} AS st, MAP {'k': 1} AS m, "
            '\'{"a": 1}\'::JSON AS j'
        ],
    ),
    (
        'nulls',
        [
            'SELECT NULL::INTEGER AS i, NULL::VARCHAR AS s UNION ALL SELECT 1, NULL'
        ],
    ),
]

CASES = {'duckdb': DUCKDB, 'sqlite': SQLITE, 'postgres': POSTGRES}

#: The differences the register states, by case: which side answers and
#: which refuses (``python`` answers and Java refuses, or the reverse), and
#: the entry. Every other case is held to the rule.
KNOWN = {
    # An anonymous record, which no table's column holds: psycopg reads it
    # as a tuple of texts and the provider writes Python's repr of it; Java
    # refuses it.
    'record': ('python answers', 'JV-2'),
    # Past the core's 2,000 levels: Python's provider writes the value
    # back compact, Java's hands the server's text; the core refuses either.
    "json past the core's reader": ('both answer, differently', 'JV-3'),
}

#: The cases both providers refuse, by the driver's error: the SQLSTATE it
#: names, the same on both (none for a refusal of the client's own).
REFUSED = {
    'date infinity',
    'date -infinity',
    'date BC',
    'date past 9999',
    'timestamp infinity',
    'timestamp BC',
    'timestamptz infinity',
    'time 24:00',
    'timetz 24:00',
    'interval past a timedelta',
    'interval, iso_8601',
    'interval, sql_standard',
    'date[] infinity',
    'a later text in an integer column',
}

#: The errors a driver's own refusal of a value is, by host: psycopg's
#: loaders' and ADBC's, and the Java provider's (its conversion table's).
CLIENT_REFUSALS = {
    'python': {
        'DataError',
        'NotImplementedError',
        'ProgrammingError',
        'OSError',
    },
    'java': {'java.sql.SQLDataException'},
}

#: The cases whose answer has no row.
EMPTY = {'no row'}


def _java() -> tuple[Path, Path | None] | None:
    jar = os.environ.get('SDM_JAVA_JAR')
    if not jar:
        assert not os.environ.get('SDM_REQUIRE_JAVA'), (
            'SDM_REQUIRE_JAVA is set but SDM_JAVA_JAR names no JAR'
        )
        return None
    native = os.environ.get('SDM_JAVA_NATIVE')
    return Path(jar), Path(native) if native else None


class Databases:
    r"""The databases: DuckDB in memory (each side its own), a SQLite file,
    and a Postgres database of the test's own."""

    def __init__(self, tmp: Path, dsn: str | None) -> None:
        self.sqlite = tmp / 'parity.sqlite'
        self.sqlite.touch()
        self.dsn = dsn
        self.database: str | None = None
        self.scoped: str | None = None
        if dsn is not None:
            import psycopg
            from psycopg.conninfo import make_conninfo

            self.database = f'sdm_parity_{uuid.uuid4().hex[:16]}'
            with psycopg.connect(dsn, autocommit=True) as admin:
                admin.execute(f'CREATE DATABASE {self.database}')
            self.scoped = make_conninfo(dsn, dbname=self.database)
            with psycopg.connect(self.scoped, autocommit=True) as connection:
                for statement in POSTGRES_SETUP:
                    connection.execute(statement)

    def python(self, database: str) -> Any:
        r"""``database``'s Python provider, on a connection of its own."""
        if database == 'duckdb':
            import duckdb

            from sdm_client._rust.providers.duckdb import DuckDBProvider

            connection = duckdb.connect()
            connection.execute(f"SET GLOBAL TimeZone = '{ZONE}'")
            return DuckDBProvider(connection)
        if database == 'sqlite':
            import adbc_driver_sqlite.dbapi

            from sdm_client._rust.providers.sqlite import SqliteProvider

            return SqliteProvider(
                adbc_driver_sqlite.dbapi.connect(str(self.sqlite))
            )
        import psycopg

        from sdm_client._rust.providers.postgres import PostgresProvider

        assert self.scoped is not None
        return PostgresProvider(
            psycopg.connect(
                self.scoped, autocommit=True, options=f'-c TimeZone={ZONE}'
            )
        )

    def connect(self, database: str) -> dict[str, Any]:
        r"""How the Java program opens ``database``."""
        if database == 'duckdb':
            return {}
        if database == 'sqlite':
            return {'path': str(self.sqlite)}
        from psycopg.conninfo import conninfo_to_dict

        assert self.scoped is not None
        parts = conninfo_to_dict(self.scoped)
        return {
            'url': f'jdbc:postgresql://{parts.get("host", "localhost")}:'
            f'{parts.get("port", "5432")}/{parts["dbname"]}',
            'user': parts.get('user'),
            'password': parts.get('password'),
        }

    def drop(self) -> None:
        if self.dsn is not None and self.database is not None:
            import psycopg

            with psycopg.connect(self.dsn, autocommit=True) as admin:
                admin.execute(
                    f'DROP DATABASE IF EXISTS {self.database} WITH (FORCE)'
                )


@pytest.fixture(scope='module')
def databases(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Databases]:
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
    made = Databases(tmp_path_factory.mktemp('parity'), dsn)
    try:
        yield made
    finally:
        made.drop()


#: The key set a bound statement binds: one column, one row.
KEYS = pa.table({'k': pa.array([1], pa.int64())})


def _steps(database: str, statements: list[str], bound: bool) -> dict[str, Any]:
    r"""The fetch step running ``statements``, the last one's rows read
    through a key set's parameter where ``bound``."""
    planned = [
        {'dialect': database, 'text': text, 'params': []} for text in statements
    ]
    if bound:
        last = planned[-1]
        relation = '__sdm_keys_parity'
        if database == 'postgres':
            text = f'SELECT * FROM ({last["text"]}) AS q WHERE $1::jsonb IS NOT NULL'
        else:
            text = f'SELECT * FROM ({last["text"]}) AS q WHERE EXISTS (SELECT 1 FROM {relation})'
        planned[-1] = {
            'dialect': database,
            'text': text,
            'params': [
                {
                    'param': 'key_set',
                    'relation': relation,
                    'keys': {'attachment': 1},
                }
            ],
        }
    return {'plan': {'plan': 'sql', 'statements': planned}}


class Refused(NamedTuple):
    r"""A provider's refusal: the SQLSTATE its driver named (none for a
    refusal of the client's own), and what it said, for the report."""

    sqlstate: str | None
    said: str


def _python_answer(provider: Any, step: dict[str, Any]) -> pa.Table | Refused:
    try:
        answer = provider(step, {1: KEYS})
    except Exception as error:  # noqa: BLE001 - judged by its SQLSTATE
        return Refused(
            getattr(error, 'sqlstate', None), f'{type(error).__name__}: {error}'
        )
    return pa.RecordBatchReader._import_from_c_capsule(
        answer.__arrow_c_stream__()
    ).read_all()


def _java_answers(
    java: tuple[Path, Path | None],
    databases: Databases,
    database: str,
    steps: list[dict[str, Any]],
) -> list[pa.Table | Refused]:
    jar, native = java
    program = RUNNER.Program(
        java=RUNNER.java_binary(None),
        path=RUNNER.classpath(jar, None),
        native=native,
        options=(f'-Duser.timezone={ZONE}',),
    )
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, KEYS.schema) as writer:
        writer.write_table(KEYS)
    try:
        answer = program.ask(
            {
                'op': 'fetch',
                'case': f'parity@{database}',
                'database': database,
                'connect': databases.connect(database),
                'setup': [f"SET GLOBAL TimeZone = '{ZONE}'"]
                if database == 'duckdb'
                else [],
                'steps': steps,
                'seed': None,
                'keys': {
                    '1': base64.b64encode(sink.getvalue().to_pybytes()).decode()
                },
            }
        )
    finally:
        program.close()
    return [
        pa.ipc.open_stream(base64.b64decode(each['arrow'])).read_all()
        if 'arrow' in each
        else Refused(each.get('sqlstate'), each['error'])
        for each in answer['answers']
    ]


def _plain(value: Any) -> Any:
    r"""A value compared exactly: a float by its bits (a NaN equal to a
    NaN, ``-0.0`` not to ``0.0``), containers element by element."""
    if isinstance(value, float):
        return ('float', value.hex() if not math.isnan(value) else 'nan')
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _values(table: pa.Table) -> Any:
    r"""A table's values, compared exactly; a value Python holds no form of
    (a date past its range) by Arrow's text of its column."""
    try:
        return _plain(table.to_pylist())
    except (OverflowError, ValueError):
        return [column.to_string() for column in table.columns]


def _judged(case: str, mine: Any, theirs: Any) -> str | None:
    r"""What is wrong with the two answers to ``case``, or None: a
    registered difference as its entry states it; a refusal both refuse, by
    the same SQLSTATE; any other case both answer, with a row unless it
    has none, the same Arrow."""
    known = KNOWN.get(case)
    if known is not None:
        if known[0] == 'python answers':
            stated = isinstance(mine, pa.Table) and isinstance(theirs, Refused)
        else:
            stated = (
                isinstance(mine, pa.Table)
                and isinstance(theirs, pa.Table)
                and _values(mine) != _values(theirs)
            )
        if stated:
            return None
        return (
            f'not the difference {known[1]} states: Python {_shown(mine)}; '
            f'Java {_shown(theirs)}'
        )
    if case in REFUSED:
        if not (isinstance(mine, Refused) and isinstance(theirs, Refused)):
            return f'both refuse: Python {_shown(mine)}; Java {_shown(theirs)}'
        if mine.sqlstate != theirs.sqlstate:
            return f'refused by SQLSTATE {theirs.sqlstate}, not {mine.sqlstate}'
        if mine.sqlstate is None and (
            mine.said.split(':')[0] not in CLIENT_REFUSALS['python']
            or theirs.said not in CLIENT_REFUSALS['java']
        ):
            # A refusal of the driver's own, no SQLSTATE named: the error a
            # driver's reading of a value raises (Claude T3.12 r3 P3).
            return f'not a refusal of a value: Python {mine.said}; Java {theirs.said}'
        return None
    if isinstance(mine, Refused) or isinstance(theirs, Refused):
        return f'both answer: Python {_shown(mine)}; Java {_shown(theirs)}'
    if case not in EMPTY and mine.num_rows == 0:
        return 'answered no row'
    return _difference(mine, theirs)


def _difference(mine: pa.Table, theirs: pa.Table) -> str | None:
    if not mine.schema.equals(theirs.schema, check_metadata=True):
        return f'schema: Java {theirs.schema!r}; Python {mine.schema!r}'
    if _values(mine) != _values(theirs):
        return f'values: Java {_values(theirs)}; Python {_values(mine)}'
    return None


def _shown(answer: pa.Table | Refused) -> str:
    if isinstance(answer, Refused):
        return f'refused ({answer.sqlstate}: {answer.said})'
    return f'answered {_values(answer)}'


@pytest.mark.parametrize('database', ['duckdb', 'sqlite', 'postgres'])
def test_java_hands_the_core_what_python_s_provider_hands_it(
    databases: Databases, database: str
) -> None:
    register = (ROOT / 'docs/design/rust-core/register.yaml').read_text(
        encoding='utf-8'
    )
    for _, entry in KNOWN.values():
        assert f'\n- id: {entry}\n' in register, entry
    # Every case named a difference, a refusal or empty is one that runs.
    named = {name for cases in CASES.values() for name, _ in cases}
    assert set(KNOWN) | REFUSED | EMPTY <= named
    java = _java()
    if java is None or (database == 'postgres' and databases.scoped is None):
        return
    steps: list[dict[str, Any]] = []
    names: list[str] = []
    for name, statements in CASES[database]:
        for bound in (False, True):
            for run in range(RUNS):
                steps.append(_steps(database, statements, bound))
                names.append(
                    f'{name} ({"bound" if bound else "plain"}, run {run + 1})'
                )
    provider = databases.python(database)
    mine = [_python_answer(provider, step) for step in steps]
    theirs = _java_answers(java, databases, database, steps)
    # Each case's difference once, naming the runs it showed in.
    problems: dict[str, list[str]] = {}
    for name, a, b in zip(names, mine, theirs, strict=True):
        case, run = name.rsplit(' (', 1)
        found = _judged(case, a, b)
        if found is not None:
            problems.setdefault(f'{case}: {found}', []).append(run.rstrip(')'))
    assert not problems, '\n'.join(
        f'{found} [{", ".join(runs)}]' for found, runs in problems.items()
    )


#: Instants each zone is read at: before and at its first rules, historic
#: offsets, and both sides of DST changes in Europe and North America.
INSTANTS = [
    '1850-01-01 00:00+00',
    '1900-01-01 00:00+00',
    '1916-05-01 00:00+00',
    '1937-07-01 00:00+00',
    '1945-08-01 00:00+00',
    '1970-01-01 00:00+00',
    '1996-10-27 00:59:59+00',
    '1996-10-27 01:00:00+00',
    '2000-03-26 00:59:59+00',
    '2000-03-26 01:00:00+00',
    '2024-03-10 06:59:59+00',
    '2024-03-10 07:00:00+00',
    '2024-11-03 05:59:59+00',
    '2024-11-03 06:00:00+00',
    '2038-06-01 12:00+00',
    '2100-07-01 12:00+00',
]


def test_zoned_text_is_python_s_in_every_zone_postgres_names(
    databases: Databases,
) -> None:
    # Codex T3.12 r3 P0-1: a timestamptz written as text is the instant in
    # the zone psycopg resolves the session's to, by the zone data Python
    # reads (its ZoneInfo), in every zone Postgres names and the names Python
    # reads that no listing gives, at historic instants and DST edges alike.
    java = _java()
    if java is None or databases.scoped is None:
        return
    import psycopg

    with psycopg.connect(databases.scoped, autocommit=True) as connection:
        names = sorted(
            {
                row[0]
                for row in connection.execute(
                    'SELECT name FROM pg_timezone_names'
                )
            }
            | {
                row[0]
                for row in connection.execute(
                    'SELECT abbrev FROM pg_timezone_abbrevs'
                )
            }
            | {
                'posixrules',
                'localtime',
                'Factory',
                'UTC',
                'GMT+3',
                'asia/kolkata',
            }
        )
    values = ','.join(f'"{at}"' for at in INSTANTS)
    read = _rows('timestamptz[]', f"'{{{values}}}'")
    steps = [
        {
            'plan': {
                'plan': 'sql',
                'statements': [
                    {'dialect': 'postgres', 'text': text, 'params': []}
                    for text in (f"SET TimeZone = '{name}'", read)
                ],
            }
        }
        for name in names
    ]
    provider = databases.python('postgres')
    mine = [_python_answer(provider, step) for step in steps]
    theirs = _java_answers(java, databases, 'postgres', steps)
    differ = []
    for name, a, b in zip(names, mine, theirs, strict=True):
        if isinstance(a, Refused) and isinstance(b, Refused):
            continue  # A name Postgres does not take as a session's zone.
        found = (
            _difference(a, b)
            if not isinstance(a, Refused) and not isinstance(b, Refused)
            else f'Python {_shown(a)}; Java {_shown(b)}'
        )
        if found is not None:
            differ.append(f'{name}: {found}')
    assert not differ, (
        f'{len(differ)} of {len(names)} zones differ:\n'
        + '\n'.join(differ[:40])
    )
