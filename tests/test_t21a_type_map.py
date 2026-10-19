# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1a's acceptance tests for the type map's decimal vectors
(``contract/type-map-vectors.json``, recorded by
``contract/tools/type_map_record.py``) and the contract loader that loads
SQLite and Postgres from the DuckDB-built tables through Arrow (the
independent test author's).

The design (choice 2): a SQLite decimal is stored in a column declared
``TEXT``, as its canonical text, since a column declared ``DECIMAL`` takes
numeric affinity and rounds what it stores; two decimal keys equal to 15
significant digits stay distinct after loading on SQLite (KO2-35); two equal
keys of different scales, ``DECIMAL(10,2)`` and ``DECIMAL(12,4)``, still join
(lesson 19); the vector is the one definition of a decimal's canonical text,
which the Python loader and the Rust key binders each implement and both run.

Postgres: the tests reading a server take it from ``SDM_POSTGRES_TEST_DSN``;
without one they stand down, unless ``SDM_REQUIRE_POSTGRES`` is set, where a
missing or unreachable server fails (H26). A local run needs a server, for
example the pinned image CI uses (``POSTGRES_IMAGE`` in
``.github/workflows/ci.yml``), started with ``POSTGRES_PASSWORD`` set and its
DSN exported.

The Rust key binders' half of the canonical text is the implementer's to
add beside the Rust binders (a test reading this same vector), since the
binders' API does not exist yet and a test against it would not build.

Gating: :data:`PENDING`, a strict xfail, as in ``test_t21a_case_format.py``;
the implementer deletes its one definition. Unmarked tests are controls on
the vector file and pass today.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
import uuid
from collections.abc import Iterator
from decimal import Decimal, localcontext
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
VECTORS = json.loads((ROOT / 'contract/type-map-vectors.json').read_text())
LOADER = ROOT / 'contract/runners/databases.py'


# --- the interface assumed (the implementer matches it, or asks) ---------------
#
#: ``contract/runners/databases.py``, the one contract loader:
#: ``canonical_decimal_text(value: Decimal) -> str``, the vector's text;
#: ``load(duck, database, target)``: every table of the DuckDB connection
#: ``duck``'s main schema, copied through Arrow into ``database``
#: (``'sqlite'``: ``target`` a path to a new SQLite file; ``'postgres'``: a
#: libpq DSN, the tables created in its current schema), each column stored
#: as the type map says.
# ---------------------------------------------------------------------------------


def _loader() -> ModuleType:
    name = 't21a_databases'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LOADER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# --- the vector file (controls) --------------------------------------------------


def test_the_vector_file_holds_what_the_design_names() -> None:
    r"""The vector is not vacuous: equal values of different scales share
    a text (``100.00`` and ``100``, ``-0.00`` and ``0``); some canonical
    texts differ from both DuckDB's and Python's (so neither is the rule);
    the SQLite pair agrees past 15 significant digits and a column declared
    DECIMAL keeps one value of it, a TEXT column two; the join pairs are as
    many as DuckDB joins."""
    assert VECTORS['format'] == 'sdm-type-map-vectors/1'
    groups = {g['canonical']: set(g['literals']) for g in VECTORS['one_text']}
    assert {'100', '100.00'} <= groups['100']
    assert {'-0.00', '0'} <= groups['0']
    texts = VECTORS['canonical_text']
    assert any(
        v['canonical'] not in (v['duckdb_text'], v['python_str']) for v in texts
    )
    assert any('E' in v['python_str'] for v in texts)
    pair = VECTORS['sqlite_distinct_keys']
    assert pair['common_significant_digits'] >= 15
    assert len(set(pair['canonical'])) == 2
    declared, text = pair['sqlite_probe']
    assert declared['distinct'] == 1 and text['distinct'] == 2
    join = VECTORS['join_across_scales']
    assert join['left']['duckdb_type'] == 'DECIMAL(10,2)'
    assert join['right']['duckdb_type'] == 'DECIMAL(12,4)'
    assert (
        join['duckdb_joined']
        == len(join['left']['keys'])
        == len(join['right']['keys'])
    )
    assert VECTORS['map']['decimal']['sqlite'] == 'TEXT'


# --- the canonical text ------------------------------------------------------------


@pytest.mark.parametrize(
    'vector',
    VECTORS['canonical_text'],
    ids=[
        f'{v["duckdb_type"]} {v["literal"]}' for v in VECTORS['canonical_text']
    ],
)
def test_the_loader_writes_each_decimal_s_canonical_text(
    vector: dict[str, Any],
) -> None:
    r"""Exit: the Python loader implements the vector's canonical text, for
    the value as Arrow hands it over (at the column's scale)."""
    scale = int(vector['duckdb_type'].rstrip(')').split(',')[1])
    with localcontext() as context:
        context.prec = 100
        value = Decimal(vector['literal']).quantize(Decimal(1).scaleb(-scale))
    assert _loader().canonical_decimal_text(value) == vector['canonical']


# --- loading SQLite ----------------------------------------------------------------


def _duck(*statements: str) -> Any:
    import duckdb

    duck = duckdb.connect()
    for statement in statements:
        duck.execute(statement)
    return duck


def _values(keys: list[str]) -> str:
    return ', '.join(f"('{k}', {i})" for i, k in enumerate(keys))


def _distinct_base() -> Any:
    pair = VECTORS['sqlite_distinct_keys']
    return _duck(
        f'CREATE TABLE users (k {pair["duckdb_type"]} PRIMARY KEY, n INTEGER)',
        f'INSERT INTO users VALUES {_values(pair["keys"])}',
    )


def _join_base() -> Any:
    join = VECTORS['join_across_scales']
    left, right = join['left'], join['right']
    return _duck(
        f'CREATE TABLE a (k {left["duckdb_type"]}, n INTEGER)',
        f'CREATE TABLE b (k {right["duckdb_type"]}, n INTEGER)',
        f'INSERT INTO a VALUES {_values(left["keys"])}',
        f'INSERT INTO b VALUES {_values(right["keys"])}',
    )


def test_sqlite_keeps_two_keys_equal_to_15_digits_distinct(
    tmp_path: Path,
) -> None:
    r"""Exit (KO2-35): loaded on SQLite, the two keys stay two, each stored
    as TEXT in a column declared TEXT, as its canonical text."""
    target = tmp_path / 'base.sqlite'
    _loader().load(_distinct_base(), 'sqlite', target)
    connection = sqlite3.connect(target)
    try:
        declared = {
            row[1]: row[2].upper()
            for row in connection.execute('PRAGMA table_info(users)')
        }
        rows = connection.execute(
            'SELECT k, typeof(k) FROM users ORDER BY n'
        ).fetchall()
    finally:
        connection.close()
    assert declared['k'] == 'TEXT'
    assert rows == [
        (text, 'text') for text in VECTORS['sqlite_distinct_keys']['canonical']
    ]


def test_sqlite_joins_equal_keys_of_different_scales(tmp_path: Path) -> None:
    r"""Exit (lesson 19): ``DECIMAL(10,2)`` and ``DECIMAL(12,4)`` keys of
    equal value join on SQLite as on DuckDB, ``-0.00`` with ``0.0000``
    included."""
    target = tmp_path / 'base.sqlite'
    _loader().load(_join_base(), 'sqlite', target)
    connection = sqlite3.connect(target)
    try:
        joined = connection.execute(
            'SELECT a.n, b.n FROM a JOIN b ON a.k = b.k ORDER BY a.n'
        ).fetchall()
    finally:
        connection.close()
    count = VECTORS['join_across_scales']['duckdb_joined']
    assert joined == [(i, i) for i in range(count)]


# --- loading Postgres ----------------------------------------------------------------


@pytest.fixture
def postgres_dsn() -> Iterator[str]:
    r"""A fresh schema on the configured server, and a DSN whose current
    schema it is; dropped after. Required where ``SDM_REQUIRE_POSTGRES`` is
    set: there a missing DSN or an unreachable server fails."""
    dsn = os.getenv('SDM_POSTGRES_TEST_DSN')
    required = bool(os.getenv('SDM_REQUIRE_POSTGRES'))
    if not dsn:
        if required:
            pytest.fail(
                'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
            )
        pytest.skip('SDM_POSTGRES_TEST_DSN is not set')
    import psycopg
    from psycopg.conninfo import make_conninfo

    schema = f't21a_{uuid.uuid4().hex[:12]}'
    with psycopg.connect(
        dsn, connect_timeout=10, autocommit=True
    ) as connection:
        connection.execute(f'CREATE SCHEMA {schema}')
    try:
        yield make_conninfo(dsn, options=f'-c search_path={schema}')
    finally:
        with psycopg.connect(
            dsn, connect_timeout=10, autocommit=True
        ) as connection:
            connection.execute(f'DROP SCHEMA {schema} CASCADE')


def _postgres(dsn: str, query: str) -> list[tuple[Any, ...]]:
    import psycopg

    with psycopg.connect(dsn, connect_timeout=10) as connection:
        return connection.execute(query).fetchall()


def test_postgres_keeps_the_keys_as_numeric(postgres_dsn: str) -> None:
    r"""Exit: on Postgres a decimal is the engine's own ``numeric`` with its
    precision and scale, and the two keys stay two, exact."""
    _loader().load(_distinct_base(), 'postgres', postgres_dsn)
    [(kind, precision, scale)] = _postgres(
        postgres_dsn,
        'SELECT data_type, numeric_precision, numeric_scale FROM '
        'information_schema.columns WHERE table_schema = current_schema() '
        "AND table_name = 'users' AND column_name = 'k'",
    )
    assert (kind, precision, scale) == ('numeric', 18, 8)
    rows = _postgres(postgres_dsn, 'SELECT k FROM users ORDER BY n')
    assert [str(k) for (k,) in rows] == VECTORS['sqlite_distinct_keys']['keys']


def test_postgres_joins_equal_keys_of_different_scales(
    postgres_dsn: str,
) -> None:
    r"""Exit (lesson 19): the two scales join on Postgres too."""
    _loader().load(_join_base(), 'postgres', postgres_dsn)
    joined = _postgres(
        postgres_dsn, 'SELECT a.n, b.n FROM a JOIN b ON a.k = b.k ORDER BY a.n'
    )
    count = VECTORS['join_across_scales']['duckdb_joined']
    assert joined == [(i, i) for i in range(count)]
