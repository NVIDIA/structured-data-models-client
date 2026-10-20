# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Schema.from_source`` on the core route (T1.11b; tables named, T1.24).

For a DuckDB source, with tables named or none, the core's ``describe_source``
reads the source through the Python host's DuckDB provider and the schema
bridge builds today's ``Schema``: on ``base.regression``'s base it is the
reference's. The relational engine is not imported on the route, which a
test-only import hook in a process of its own shows, and a core route
running leaves another thread's old-route call as it is alone. The providers import no pandas, which an AST scan
of ``_rust/providers`` and a run with pandas unimportable show. What the
core does not describe yet is refused by name (register entry P1B-2); the
call's arguments all pass through to it, the bridge refusing none of its
own (decision E61).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any

import duckdb
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import Relationship, Schema, StructuredDataError, _routing
from sdm_client._rust import providers
from sdm_client._rust.providers.duckdb import DuckDBProvider

ROOT = Path(__file__).resolve().parents[3]
CORE = ','.join(_routing.FROM_SOURCE_DUCKDB)
#: Both of Schema.from_source's DuckDB routes: every table, or tables named.
CORE_ANY = ','.join(
    dict.fromkeys(
        _routing.FROM_SOURCE_DUCKDB + _routing.FROM_SOURCE_DUCKDB_TABLES
    )
)
PROVIDERS = Path(providers.__file__).resolve().parent


@pytest.fixture(autouse=True)
def _old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


def _shop() -> list[str]:
    r"""The ``shop`` base of ``contract/bases.yaml``, read without a YAML
    library: its statements are the plain list items under its name."""
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


def _connection(*setup: str) -> Any:
    connection = duckdb.connect()
    for statement in [*_shop(), *setup]:
        connection.execute(statement)
    return connection


def _projection(schema: Schema) -> dict[str, Any]:
    r"""The public schema as the contract runner projects it."""
    return {
        'tables': [
            {
                'name': name,
                'columns': [
                    [column, c.dtype, c.stype]
                    for column, c in table.columns.items()
                ],
                'primary_key': table.primary_key,
                'time_column': table.time_column,
                'duplicate_keys': table.duplicate_keys,
            }
            for name, table in schema.tables.items()
        ],
        'links': [
            [
                link.source_table,
                link.source_column,
                link.target_table,
                link.target_column,
            ]
            for link in schema.relationships
        ],
    }


#: base.regression's schema, as the reference describes it (its reference
#: observation, contract/reference).
REFERENCE = {
    'tables': [
        {
            'name': 'orders',
            'columns': [
                ['order_id', 'int64', 'ID'],
                ['user_id', 'int64', 'ID'],
                ['price', 'float64', 'numerical'],
                ['ts', 'timestamp[us]', 'timestamp'],
            ],
            'primary_key': 'order_id',
            'time_column': 'ts',
            'duplicate_keys': 0,
        },
        {
            'name': 'users',
            'columns': [
                ['user_id', 'int64', 'ID'],
                ['age', 'int64', 'numerical'],
                ['country', 'string', 'categorical'],
            ],
            'primary_key': 'user_id',
            'time_column': None,
            'duplicate_keys': 0,
        },
    ],
    'links': [['orders', 'user_id', 'users', 'user_id']],
}


def test_the_route_takes_every_table_and_the_catalog_s_keys_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert set(_routing.FROM_SOURCE_DUCKDB) <= _routing.CORE_ROUTED
    assert set(_routing.FROM_SOURCE_DUCKDB_TABLES) <= _routing.CORE_ROUTED
    monkeypatch.setenv(_routing.ENVIRONMENT, _routing.FROM_SOURCE_DUCKDB[0])
    with pytest.raises(_routing.RouteError, match='one route'):
        Schema.from_source(DuckDBSource(_connection()))
    # Tables named take the keys path with their own (T1.24): forcing one
    # without the other fails the call.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with pytest.raises(_routing.RouteError, match='one route'):
        Schema.from_source(DuckDBSource(_connection()), {'users': 'users'})
    monkeypatch.setenv(
        _routing.ENVIRONMENT, _routing.FROM_SOURCE_DUCKDB_TABLES[0]
    )
    with pytest.raises(_routing.RouteError, match='one route'):
        Schema.from_source(DuckDBSource(_connection()), {'users': 'users'})
    monkeypatch.setenv(
        _routing.ENVIRONMENT, ','.join(_routing.FROM_SOURCE_DUCKDB_TABLES)
    )
    schema = Schema.from_source(
        DuckDBSource(_connection()), {'people': 'users'}
    )
    assert list(schema.tables) == ['people']
    assert schema['people'].source_name == 'users'
    assert schema._core_description is not None


def test_base_regression_s_schema_on_the_core_is_the_reference_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    connection = _connection()
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        schema = Schema.from_source(DuckDBSource(connection))
    assert _projection(schema) == REFERENCE
    assert [type(w.message).__name__ for w in shown] == []
    assert schema.source is not None
    # Keys unchecked: nothing counted, and no count asked for.
    unchecked = Schema.from_source(DuckDBSource(connection), check_keys=False)
    assert [t.duplicate_keys for t in unchecked.tables.values()] == [None, None]


def _subprocess(script: str, **env: str) -> subprocess.CompletedProcess[str]:
    import os

    environment = {
        key: value
        for key, value in os.environ.items()
        if key != _routing.ENVIRONMENT
    }
    return subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**environment, **env},
    )


def test_the_route_imports_nothing_of_the_engine(tmp_path: Path) -> None:
    # A stand-in engine any import would find first, installed or not, so
    # an import on the route would succeed if the hook let it.
    fake = tmp_path / 'sdm_relational_engine'
    fake.mkdir()
    (fake / '__init__.py').write_text('IMPORTED = True\n', encoding='utf-8')
    # The hook is this test's own, in a process of its own: production
    # installs none, since a hook on sys.meta_path is the whole process's.
    script = f"""
    import sys
    from contextlib import contextmanager
    sys.path.insert(0, {str(tmp_path)!r})
    import duckdb
    from sdm_client import Schema
    from sdm_client._routing import RouteError
    from sdm_client._rust.providers import duckdb as provider
    from sdm_connectors import DuckDBSource

    ENGINE = 'sdm_relational_engine'

    class NoEngine:
        def find_spec(self, name, path=None, target=None):
            if name == ENGINE or name.startswith(ENGINE + '.'):
                raise RouteError(f'{{name}} was imported after route entry')

    @contextmanager
    def route():
        before = ENGINE in sys.modules
        hook = NoEngine()
        sys.meta_path.insert(0, hook)
        try:
            yield
        finally:
            sys.meta_path.remove(hook)
        assert before or ENGINE not in sys.modules

    connection = duckdb.connect()
    for statement in {_shop()!r}:
        connection.execute(statement)
    before = ENGINE in sys.modules
    with route():
        Schema.from_source(DuckDBSource(connection))
    print(before, ENGINE in sys.modules)

    # The hook sees an import on the route, and the route fails with it.
    answer = provider.DuckDBProvider.__call__
    def importing(self, step, keys=None):
        import sdm_relational_engine
        return answer(self, step, keys)
    provider.DuckDBProvider.__call__ = importing
    try:
        with route():
            Schema.from_source(DuckDBSource(connection))
    except RouteError as refused:
        print('refused', ENGINE in str(refused))
    print(ENGINE in sys.modules)
    """
    result = _subprocess(script, **{_routing.ENVIRONMENT: CORE})
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == [
        'False',
        'False',
        'refused',
        'True',
        'False',
    ]


def _sqlite(path: Path) -> Path:
    r"""A SQLite database of one keyed table, at ``path``."""
    import sqlite3

    database = path / 'old.sqlite'
    with sqlite3.connect(database) as setup:
        setup.execute(
            'CREATE TABLE users (user_id INTEGER PRIMARY KEY, age INTEGER)'
        )
        setup.executemany(
            'INSERT INTO users VALUES (?, ?)', [(i, 20 + i) for i in range(8)]
        )
    return database


def test_a_core_route_running_leaves_another_thread_s_old_call_alone(
    tmp_path: Path,
) -> None:
    # In a process of its own, so the old route's first import of the
    # engine happens while the DuckDB core route runs: the DuckDB call is
    # paused in its provider while a SQLite call, whose route is the old
    # one, runs on another thread; then the same SQLite call runs alone.
    database = _sqlite(tmp_path)
    script = f"""
    import importlib.util, threading
    import adbc_driver_sqlite.dbapi
    import duckdb
    from sdm_client import Schema
    from sdm_client._rust.providers.duckdb import DuckDBProvider
    from sdm_connectors import DuckDBSource, SQLiteSource

    def old():
        connection = adbc_driver_sqlite.dbapi.connect({str(database)!r})
        try:
            schema = Schema.from_source(SQLiteSource(connection))
        except Exception as error:
            return [type(error).__name__, getattr(error, 'code', None)]
        return ['success', sorted(schema.tables)]

    paused, resume = threading.Event(), threading.Event()
    answer = DuckDBProvider.__call__
    def pausing(self, step, keys=None):
        if not paused.is_set():
            paused.set()
            assert resume.wait(60)
        return answer(self, step, keys)
    DuckDBProvider.__call__ = pausing

    core = {{}}
    def described():
        connection = duckdb.connect()
        for statement in {_shop()!r}:
            connection.execute(statement)
        core['tables'] = sorted(Schema.from_source(DuckDBSource(connection)).tables)
    thread = threading.Thread(target=described)
    thread.start()
    assert paused.wait(60)
    during = old()
    resume.set()
    thread.join(60)
    alone = old()
    engine = importlib.util.find_spec('sdm_relational_engine') is not None
    import json
    print(json.dumps([during, alone, core['tables'], engine]))
    """
    result = _subprocess(script, **{_routing.ENVIRONMENT: CORE})
    assert result.returncode == 0, result.stderr
    during, alone, tables, engine = json.loads(result.stdout)
    assert during == alone
    assert tables == ['orders', 'users']
    if engine:
        # With the engine installed, the old route describes the source.
        assert alone == ['success', ['users']]


def _imports(text: str) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            found.update(alias.name.split('.')[0] for alias in node.names)
        elif (
            isinstance(node, ast.ImportFrom) and node.module and not node.level
        ):
            found.add(node.module.split('.')[0])
    return found


def _attributes(text: str) -> set[str]:
    return {
        node.attr
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Attribute)
    }


#: What reads or makes pandas: the module, and the driver methods that give
#: a frame.
PANDAS_IMPORTS = {'pandas'}
PANDAS_METHODS = {
    'df',
    'fetchdf',
    'fetch_df',
    'fetch_df_chunk',
    'to_pandas',
    'from_pandas',
}


def test_the_providers_import_no_pandas_by_their_source() -> None:
    files = sorted(PROVIDERS.rglob('*.py'))
    assert {path.name for path in files} >= {'__init__.py', 'duckdb.py'}
    for path in files:
        text = path.read_text(encoding='utf-8')
        assert not _imports(text) & PANDAS_IMPORTS, path.name
        assert not _attributes(text) & PANDAS_METHODS, path.name
    # The scan sees an import and a frame method where there is one.
    for seeded in (
        'import pandas as pd',
        'from pandas.api import types',
        'import os, pandas',
        'cursor.fetchdf()',
        'table.to_pandas()',
    ):
        assert (_imports(seeded) & PANDAS_IMPORTS) or (
            _attributes(seeded) & PANDAS_METHODS
        ), seeded


def test_the_provider_answers_the_core_s_plans_with_pandas_unimportable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The plans the core makes for base.regression's schema, and the
    # provider's answers, here.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    connection = _connection()
    answer = DuckDBProvider.__call__
    steps: list[dict[str, Any]] = []
    answers: list[Any] = []

    def recording(self: Any, step: dict[str, Any], keys: Any = None) -> Any:
        steps.append(step)
        table = answer(self, step, keys)
        answers.append(table)
        return table

    monkeypatch.setattr(DuckDBProvider, '__call__', recording)
    Schema.from_source(DuckDBSource(connection))
    assert len(steps) == 10
    expected = [table.to_pydict() for table in answers]
    # The same plans, answered by the provider loaded on its own in a
    # process where pandas cannot be imported.
    script = f"""
    import importlib.util, json, sys
    sys.modules['pandas'] = None
    import duckdb
    spec = importlib.util.spec_from_file_location(
        'provider', {str(PROVIDERS / 'duckdb.py')!r})
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    connection = duckdb.connect()
    for statement in {_shop()!r}:
        connection.execute(statement)
    provider = module.DuckDBProvider(connection)
    answers = [provider(step).to_pydict() for step in json.loads(sys.stdin.read())]
    assert sys.modules['pandas'] is None
    print(json.dumps(answers, default=str))
    """
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        input=json.dumps(steps),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == json.loads(
        json.dumps(expected, default=str)
    )


def test_what_the_core_does_not_describe_yet_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    refusals = [
        # An empty table, which the reference refuses unnamed (P1B-11):
        # INVALID_REQUEST since the owner's switch of 2026-10-13 (T1.32g).
        (
            lambda: _connection('CREATE TABLE z (z_id BIGINT)'),
            {},
            'empty_table',
            'INVALID_REQUEST',
        ),
        # A struct column, which the reference fails unnamed (P1B-10).
        (
            lambda: _connection(
                "CREATE OR REPLACE TABLE users AS SELECT *, {'a': 1} AS s "
                'FROM users'
            ),
            {},
            'column_type',
            'UNSUPPORTED_REQUEST',
        ),
    ]
    for build, arguments, rule, code in refusals:
        with pytest.raises(StructuredDataError) as refused:
            Schema.from_source(DuckDBSource(build()), **arguments)
        assert refused.value.code == code
        assert refused.value.details['rule'] == rule
    # A key of two columns, refused by name (composite_key) until T2.12,
    # is described as the reference describes it: the two columns, in
    # order.
    schema = Schema.from_source(
        DuckDBSource(_connection()),
        primary_keys={'orders': ['order_id', 'user_id']},
    )
    assert tuple(schema['orders'].primary_key) == ('order_id', 'user_id')


def test_the_bridge_refuses_none_of_the_call_s_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Until T1.24 the schema bridge refused relationships given and a time
    # column cleared before the core ran (`relationships`,
    # `time_column_cleared`); both pass through now (decision E61), and
    # the core takes them: no relationship at all, and orders with no time
    # column, whose column still means a timestamp. A relationship that is
    # not a Relationship, which the contract cannot carry, is refused as
    # the reference refuses it.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    schema = Schema.from_source(
        DuckDBSource(_connection()),
        relationships=[],
        time_columns={'orders': None},
    )
    assert schema.relationships == []
    assert schema['orders'].time_column is None
    assert schema['orders']['ts'].stype == 'timestamp'
    with pytest.raises(StructuredDataError) as refused:
        Schema.from_source(
            DuckDBSource(_connection()),
            relationships=[('orders', 'user_id', 'users', 'user_id')],
        )
    assert refused.value.code == 'INVALID_REQUEST'
    assert 'must be a Relationship' in str(refused.value)


#: Sources whose schema takes the rules the core holds from T1.21d
#: (decision E55), each beside the shop base, with the codes of the
#: warnings the core gives on it: a lone edit stamp in a table nothing
#: links to (taken) and in one something links to (refused), a stamp of a
#: load and a date of birth passed over; a key whose type cannot join, a
#: foreign key to a column that is not its table's key, two ways from one
#: table to another, two tables naming each other; and a keyless table
#: keyed by the one column a foreign key refers to.
SCHEMA_RULES = {
    'edit stamps': (
        {'TIME_COLUMNS_GUESSED'},
        (
            'CREATE TABLE logs (log_id BIGINT, user_id BIGINT, '
            'updated_at TIMESTAMP)',
            "INSERT INTO logs SELECT i, i % 3, TIMESTAMP '2024-01-01' + "
            'to_days(CAST(i AS INTEGER)) FROM range(9) t(i)',
            'CREATE TABLE plans (plan_id BIGINT, modified_at TIMESTAMP)',
            "INSERT INTO plans SELECT i, TIMESTAMP '2024-01-01' + "
            'to_days(CAST(i AS INTEGER)) FROM range(4) t(i)',
            'CREATE TABLE subscriptions (sub_id BIGINT, plan_id BIGINT, '
            '_loaded_at TIMESTAMP, birth_date DATE, started TIMESTAMP)',
            'INSERT INTO subscriptions SELECT i, i % 4, '
            "TIMESTAMP '2024-02-01' + to_days(CAST(i AS INTEGER)), "
            "DATE '1990-01-01' + CAST(i AS INTEGER), "
            "TIMESTAMP '2023-01-01' + to_days(CAST(i AS INTEGER)) "
            'FROM range(8) t(i)',
        ),
    ),
    'links': (
        {'LINK_CANNOT_JOIN', 'TWO_WAYS_TO_A_TABLE', 'TABLES_NAME_EACH_OTHER'},
        (
            'CREATE TABLE accounts (account_id BIGINT PRIMARY KEY, '
            'refund_id BIGINT)',
            'INSERT INTO accounts SELECT i, i FROM range(4) t(i)',
            'CREATE TABLE refunds (refund_id BIGINT PRIMARY KEY, '
            'account_id BIGINT, payer BIGINT REFERENCES accounts '
            '(account_id))',
            'INSERT INTO refunds SELECT i, i % 4, (i + 1) % 4 '
            'FROM range(6) t(i)',
            'CREATE TABLE notes (note_id BIGINT, account_id VARCHAR)',
            "INSERT INTO notes SELECT i, 'a' || chr(97 + CAST(i AS INTEGER)) "
            'FROM range(5) t(i)',
            'CREATE TABLE items (item_id BIGINT PRIMARY KEY, sku VARCHAR '
            'UNIQUE)',
            "INSERT INTO items SELECT i, 's' || chr(97 + CAST(i AS INTEGER)) "
            'FROM range(5) t(i)',
            'CREATE TABLE lines (line_id BIGINT, sku VARCHAR REFERENCES '
            'items (sku))',
            "INSERT INTO lines SELECT i, 's' || chr(97 + CAST(i % 5 AS "
            'INTEGER)) FROM range(9) t(i)',
        ),
    ),
    # Names a remedy quotes as Python does (Codex's P2 at T1.21d's review):
    # tables holding a quote, a backslash and a tab, each with a lone edit
    # stamp taken; and a table something links to whose two edit stamps are
    # refused, their names holding double quotes and both kinds. A table
    # named with a double quote is refused by both routes' catalog read.
    'names a remedy quotes': (
        {'TIME_COLUMNS_GUESSED'},
        (
            *(
                statement
                for name in ("odd'name", 'back\\slash', 'tab\there')
                for statement in (
                    f'CREATE TABLE "{name}" (id BIGINT, updated_at TIMESTAMP)',
                    f'INSERT INTO "{name}" SELECT i, TIMESTAMP \'2024-01-01\' '
                    '+ to_days(CAST(i AS INTEGER)) FROM range(5) t(i)',
                )
            ),
            'CREATE TABLE plans (plan_id BIGINT, "modified ""x""" TIMESTAMP, '
            '"changed \'a\' ""b""" TIMESTAMP)',
            "INSERT INTO plans SELECT i, TIMESTAMP '2024-01-01' + "
            "to_days(CAST(i AS INTEGER)), TIMESTAMP '2024-02-01' + "
            'to_days(CAST(i AS INTEGER)) FROM range(4) t(i)',
            'CREATE TABLE subs (sub_id BIGINT, plan_id BIGINT)',
            'INSERT INTO subs SELECT i, i % 4 FROM range(8) t(i)',
        ),
    ),
    # T1.23: a timestamp passed over for its values (one value in every
    # row, mostly empty, empty), several left (the earliest taken, a
    # creation stamp preferred), a time of day read as no time, and a list
    # column left out.
    'time values': (
        {'TIME_COLUMNS_GUESSED'},
        (
            'CREATE TABLE plans (plan_id BIGINT, stamped TIMESTAMP, '
            'ended DATE, gone TIMESTAMP)',
            "INSERT INTO plans SELECT i, TIMESTAMP '2024-01-01 10:30:00.25', "
            "CASE WHEN i % 4 = 0 THEN DATE '2024-03-01' + CAST(i AS INTEGER) "
            'END, NULL FROM range(9) t(i)',
            'CREATE TABLE subs (sub_id BIGINT, plan_id BIGINT)',
            'INSERT INTO subs SELECT i, i % 9 FROM range(12) t(i)',
        ),
    ),
    'several timestamps': (
        {'TIME_COLUMNS_GUESSED'},
        (
            'CREATE TABLE shipments (shipment_id BIGINT, user_id BIGINT, '
            'shipped TIMESTAMP, ordered DATE)',
            "INSERT INTO shipments SELECT i, i % 8, TIMESTAMP '2023-02-01' + "
            "to_days(CAST(i AS INTEGER)), DATE '2023-01-01' + CAST(i AS "
            'INTEGER) FROM range(9) t(i)',
            'CREATE TABLE visits (visit_id BIGINT, user_id BIGINT, '
            'seen TIMESTAMP, created_at TIMESTAMP)',
            "INSERT INTO visits SELECT i, i % 8, TIMESTAMP '2020-01-01' + "
            "to_days(CAST(i AS INTEGER)), TIMESTAMP '2024-01-01' + "
            'to_days(CAST(i AS INTEGER)) FROM range(9) t(i)',
        ),
    ),
    'a time of day': (
        {'TIME_VALUES_UNREAD', 'TIME_COLUMNS_GUESSED'},
        (
            'CREATE OR REPLACE TABLE users AS SELECT *, '
            'make_time(user_id::BIGINT, 30, 0) AS f FROM users',
        ),
    ),
    'a list left out': (
        {'COLUMNS_LEFT_OUT'},
        (
            'CREATE OR REPLACE TABLE users AS SELECT *, [user_id, 2]::INTEGER[] '
            "AS f, ['a', country] AS g FROM users",
        ),
    ),
    # T1.23 fix 1 (subagent P2-1): the reasons and variant no source above
    # reached: a date of birth, one value where values are held, and a
    # declared foreign key to a table with no key.
    'more reasons': (
        {'TIME_COLUMNS_GUESSED', 'LINK_CANNOT_JOIN'},
        (
            'CREATE TABLE people (person_id BIGINT, birth_date TIMESTAMP, '
            'stamped TIMESTAMP)',
            "INSERT INTO people SELECT i, TIMESTAMP '1980-01-01' + "
            'to_days(CAST(i AS INTEGER)), CASE WHEN i % 3 = 0 THEN NULL ELSE '
            "TIMESTAMP '2024-01-01 08:00:00' END FROM range(9) t(i)",
            'CREATE TABLE visits (visit_id BIGINT, person_id BIGINT)',
            'INSERT INTO visits SELECT i, i % 9 FROM range(12) t(i)',
            'CREATE TABLE codes (u1 BIGINT UNIQUE, u2 BIGINT UNIQUE)',
            'INSERT INTO codes SELECT i, i + 10 FROM range(5) t(i)',
            'CREATE TABLE coded (coded_id BIGINT, c1 BIGINT REFERENCES codes '
            '(u1), c2 BIGINT REFERENCES codes (u2))',
            'INSERT INTO coded SELECT i, i % 5, i % 5 + 10 FROM range(7) t(i)',
        ),
    ),
    'a key by reference': (
        set(),
        (
            'CREATE TABLE codes (tag VARCHAR UNIQUE, label VARCHAR UNIQUE)',
            "INSERT INTO codes SELECT 't' || chr(97 + CAST(i AS INTEGER)), "
            "'l' || chr(97 + CAST(i AS INTEGER)) FROM range(5) t(i)",
            'CREATE TABLE coded (coded_id BIGINT, tag VARCHAR REFERENCES '
            'codes (tag))',
            "INSERT INTO coded SELECT i, 't' || chr(97 + CAST(i % 5 AS "
            'INTEGER)) FROM range(9) t(i)',
        ),
    ),
}


def _said(shown: list[Any]) -> list[Any]:
    r"""The warnings a call said, without a ``ResourceWarning`` the
    collector raised meanwhile for another test's unclosed connection,
    which ``simplefilter('always')`` lets in at any point (T1.23 fix 1)."""
    return [w for w in shown if not issubclass(w.category, ResourceWarning)]


def _runtime_directories() -> tuple[str, ...]:
    r"""Where the runtime's pandas and pyarrow are installed."""
    found = []
    for name in ('pandas', 'pyarrow'):
        spec = importlib.util.find_spec(name)
        if spec is not None and spec.submodule_search_locations:
            found.extend(
                str(Path(location).resolve()) + os.sep
                for location in spec.submodule_search_locations
            )
    return tuple(found)


def _old_said(shown: list[Any]) -> list[Any]:
    r"""What the old route's call said, as :func:`_said`, without the
    deprecation warnings its pandas or pyarrow raised (register entry
    HOST-5, owner-approved): an older pyarrow (14 and 15, on Databricks
    Runtime 15.4 and 16.4) converts its samples with pandas internals newer
    pandas deprecates (pandas 3 as a ``Pandas4Warning``), so there the old
    route raises them and the core, which converts none, raises none. Only
    the old route's are left out, so one the core raises still fails the
    comparison."""
    runtime = _runtime_directories()
    return [
        w
        for w in _said(shown)
        if not (
            issubclass(w.category, DeprecationWarning)
            and str(Path(w.filename).resolve()).startswith(runtime)
        )
    ]


def _described(
    monkeypatch: pytest.MonkeyPatch, setup: tuple[str, ...], routes: str
) -> tuple[dict[str, Any], list[tuple[str, str]], Schema]:
    if routes:
        monkeypatch.setenv(_routing.ENVIRONMENT, routes)
    else:
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        schema = Schema.from_source(DuckDBSource(_connection(*setup)))
    told = _said(shown) if routes else _old_said(shown)
    said = [(type(w.message).__name__, str(w.message)) for w in told]
    return _projection(schema), said, schema


def _before_key(schema: Schema) -> dict[tuple[str, str], str]:
    return {
        (table, column): c._stype_before_key
        for table, shaped in schema.tables.items()
        for column, c in shaped.columns.items()
        if c._stype_before_key is not None
    }


def _declared(schema: Schema) -> dict[tuple[str, str], bool]:
    r"""Each column's ``stype_declared``: a time column, or one inference
    took as it before the links refused it, reads declared on both routes
    (subagent P3-2, T1.21d's review)."""
    return {
        (table, column): c.stype_declared
        for table, shaped in schema.tables.items()
        for column, c in shaped.columns.items()
    }


@pytest.mark.parametrize('name', sorted(SCHEMA_RULES))
def test_the_core_assembles_the_schema_as_the_old_route_does(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    codes, setup = SCHEMA_RULES[name]
    core, said, schema = _described(monkeypatch, setup, CORE)
    # The core's warnings carry their code before the reference's text.
    given = [text.split(']', 1)[0].lstrip('[') for _, text in said]
    assert set(given) == codes, given
    if name == 'edit stamps':
        # The lone edit stamp a link refused: no time column, and declared
        # as the old route's schema reads it.
        assert schema['plans'].time_column is None
        assert schema['plans']['modified_at'].stype_declared is True
    if name == 'names a remedy quotes':
        # Each remedy the warning suggests is valid Python naming the table
        # it means, a name holding quotes, a backslash or a tab included.
        [(_, text)] = said
        remedies = re.findall(r'time_columns=(\{[^\n]*?\})\.(?:\n|$)', text)
        assert len(remedies) == 5, text
        named = {
            (table, column)
            for remedy in remedies
            for table, column in ast.literal_eval(remedy).items()
        }
        assert named == {
            ("odd'name", None),
            ('back\\slash', None),
            ('tab\there', None),
            ('plans', 'modified "x"'),
            ('plans', 'changed \'a\' "b"'),
        }, text
    if name == 'a key by reference':
        assert schema['codes'].primary_key == 'tag'
        assert _before_key(schema) == {
            ('codes', 'tag'): 'categorical',
            ('coded', 'tag'): 'categorical',
        }
        # Given back when the schema stops joining on it.
        edited = schema.snapshot()
        edited.relationships = []
        edited['codes'].primary_key = 'label'
        assert edited['codes']['tag'].stype == 'categorical'
        assert edited['coded']['tag'].stype == 'categorical'
    if importlib.util.find_spec('sdm_relational_engine') is None:
        return
    old, old_said, old_schema = _described(monkeypatch, setup, '')
    assert core == old
    assert [
        (cls, text.split('] ', 1)[1] if text.startswith('[') else text)
        for cls, text in said
    ] == old_said
    assert _before_key(schema) == _before_key(old_schema)
    assert _declared(schema) == _declared(old_schema)


def _described_with(
    monkeypatch: pytest.MonkeyPatch, routes: str, name: str
) -> Any:
    r"""``Schema.from_source`` over a DuckDB source of one table, ``name``,
    on ``routes`` (the old route when empty)."""
    quoted = '"' + name.replace('"', '""') + '"'
    setup = (
        f'CREATE TABLE {quoted} (id INTEGER PRIMARY KEY, label VARCHAR)',
        f"INSERT INTO {quoted} VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    )
    if routes:
        monkeypatch.setenv(_routing.ENVIRONMENT, routes)
    else:
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(DuckDBSource(_connection(*setup)))


def test_a_table_named_with_a_double_quote_is_described_on_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Register entry P1B-5 (proposed, T1.22): DuckDB's PRAGMA table_info
    # reads a name holding a double quote as another name, so the reference
    # refuses the table (INVALID_REQUEST, its TableNotFoundError); the core
    # reads its columns by its own name and describes it. Pinned here,
    # naming both outcomes, so the divergence fails this test the day
    # either changes.
    name = 'double"quote'
    schema = _described_with(monkeypatch, CORE, name)
    assert name in list(schema.tables)
    assert schema[name].primary_key == 'id'
    assert list(schema[name].columns) == ['id', 'label']
    if importlib.util.find_spec('sdm_relational_engine') is None:
        return
    with pytest.raises(StructuredDataError) as refused:
        _described_with(monkeypatch, '', name)
    assert refused.value.code == 'INVALID_REQUEST'


def test_a_table_named_with_a_dot_is_refused_on_both_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The pragma reads `a.b` as table b of schema a too, and the reference
    # refuses it; from T1.23 the core refuses a table the schema would name
    # so, by the reference's rule on names (`_check_name`), with the same
    # code, so the routes agree.
    with pytest.raises(StructuredDataError) as core:
        _described_with(monkeypatch, CORE, 'a.b')
    assert core.value.code == 'INVALID_REQUEST'
    assert core.value.details['field'] == 'tables'
    if importlib.util.find_spec('sdm_relational_engine') is None:
        return
    with pytest.raises(StructuredDataError) as old:
        _described_with(monkeypatch, '', 'a.b')
    assert old.value.code == 'INVALID_REQUEST'


#: Keys and time columns named in Schema.from_source (T1.23), each beside
#: the shop base, with what is set up first: a key moved off the column
#: links name, a time column named among several, text named as a time
#: column, and both on one table.
NAMED = {
    'a key moved': ((), {'primary_keys': {'users': 'age'}}),
    'a time column among several': (
        (
            'CREATE OR REPLACE TABLE orders AS SELECT *, ts + INTERVAL 1 DAY '
            'AS shipped FROM orders',
        ),
        {'time_columns': {'orders': 'shipped'}},
    ),
    'text as a time column': (
        (
            "CREATE OR REPLACE TABLE users AS SELECT *, 'n' || "
            'chr(97 + user_id::INTEGER) AS note FROM users',
        ),
        {'time_columns': {'users': 'note'}},
    ),
    'both': (
        (),
        {
            'primary_keys': {'orders': 'order_id', 'users': 'user_id'},
            'time_columns': {'orders': 'ts'},
        },
    ),
    # T1.24: the arguments the schema bridge refused until then, and tables
    # named, each now on the core (decision E61): a time column cleared,
    # beside a key and among several; the relationships given, none or
    # one; tables named otherwise than the source names them, with keys,
    # time columns and relationships by the schema's names.
    'a time column cleared': ((), {'time_columns': {'orders': None}}),
    'a time column cleared among several': (
        (
            'CREATE OR REPLACE TABLE orders AS SELECT *, ts + INTERVAL 1 DAY '
            'AS shipped FROM orders',
        ),
        {'time_columns': {'orders': None}, 'primary_keys': {'users': 'age'}},
    ),
    'no relationships': ((), {'relationships': []}),
    'a relationship given': (
        (),
        {
            'relationships': [
                Relationship('orders', 'user_id', 'users', 'user_id')
            ]
        },
    ),
    'tables named': (
        (),
        {'tables': {'buyers': 'users', 'purchases': 'orders'}},
    ),
    'tables named, with everything': (
        (
            'CREATE OR REPLACE TABLE orders AS SELECT *, ts + INTERVAL 1 DAY '
            'AS shipped FROM orders',
        ),
        {
            'tables': {'purchases': 'orders', 'buyers': 'users'},
            'primary_keys': {'buyers': 'user_id'},
            'time_columns': {'purchases': 'shipped', 'buyers': None},
            'relationships': [
                Relationship('purchases', 'user_id', 'buyers', 'user_id')
            ],
        },
    ),
    'tables named, a subset': ((), {'tables': {'users': 'users'}}),
}


def _from_source(
    monkeypatch: pytest.MonkeyPatch,
    setup: tuple[str, ...],
    routes: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, str]], Schema]:
    if routes:
        monkeypatch.setenv(_routing.ENVIRONMENT, routes)
    else:
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        schema = Schema.from_source(
            DuckDBSource(_connection(*setup)), **arguments
        )
    said = [
        (
            type(w.message).__name__,
            str(w.message).split('] ', 1)[1]
            if str(w.message).startswith('[')
            else str(w.message),
        )
        for w in (_said(shown) if routes else _old_said(shown))
    ]
    return _projection(schema), said, schema


@pytest.mark.parametrize('name', sorted(NAMED))
def test_keys_and_time_columns_named_are_applied_as_the_old_route_does(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    setup, arguments = NAMED[name]
    core, said, schema = _from_source(monkeypatch, setup, CORE_ANY, arguments)
    assert schema._core_description is not None
    for table, key in arguments.get('primary_keys', {}).items():
        assert schema[table].primary_key == (
            key if isinstance(key, str) else tuple(key)
        )
    for table, column in arguments.get('time_columns', {}).items():
        assert schema[table].time_column == column
        if column is not None:
            assert schema[table][column].stype == 'timestamp'
    if importlib.util.find_spec('sdm_relational_engine') is None:
        return
    old, old_said, old_schema = _from_source(monkeypatch, setup, '', arguments)
    assert core == old
    assert said == old_said
    assert _before_key(schema) == _before_key(old_schema)
    assert _declared(schema) == _declared(old_schema)


@pytest.mark.parametrize(
    'arguments',
    [
        {'primary_keys': {'nowhere': 'id'}},
        {'primary_keys': {'users': 'nothing'}},
        {'primary_keys': {'orders': 'ts'}},
        {'time_columns': {'nowhere': 'ts'}},
        {'time_columns': {'orders': 'nothing'}},
        {'time_columns': {'orders': 'order_id'}},
        {'time_columns': {'orders': 'price'}},
        # T1.24: a time column cleared on a table not the schema's, and the
        # relationships given wrongly, which the bridge refused unread.
        {'time_columns': {'nowhere': None}},
        {'relationships': [Relationship('orders', 'gone', 'users', 'user_id')]},
        {'relationships': [Relationship('orders', 'user_id', 'users', 'age')]},
        {'relationships': [Relationship('orders', 'ts', 'users', 'user_id')]},
        {
            'tables': {'buyers': 'users'},
            'relationships': [
                Relationship('orders', 'user_id', 'users', 'user_id')
            ],
        },
    ],
)
def test_a_key_or_time_column_named_wrongly_is_refused_as_the_old_route_does(
    monkeypatch: pytest.MonkeyPatch, arguments: dict[str, Any]
) -> None:
    with pytest.raises(StructuredDataError) as core:
        _from_source(monkeypatch, (), CORE_ANY, arguments)
    assert core.value.code == 'INVALID_REQUEST'
    if importlib.util.find_spec('sdm_relational_engine') is None:
        return
    with pytest.raises(StructuredDataError) as old:
        _from_source(monkeypatch, (), '', arguments)
    assert old.value.code == 'INVALID_REQUEST'


def _failing(provider: Any, table: str | None) -> Any:
    r"""``provider``, but failing, as a driver fails, the reading of
    ``table``'s columns: a provider's error the core did not cause. The
    request names the table as a relation of one part (contract v26)."""

    def answer(step: dict[str, Any], keys: Any) -> Any:
        request = step['request']
        if (
            table is not None
            and request.get('part') == 'columns'
            and request.get('tables') == [{'source': [table]}]
        ):
            raise duckdb.IOException('the source went away')
        return provider(step, keys)

    return answer


def _named_on_the_core(
    connection: Any, tables: dict[str, str], fail_columns_of: str | None = None
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    r"""``Schema.from_source`` over ``tables`` named, on the core route
    (T1.24), its provider failing the reading of ``fail_columns_of``'s
    columns when named."""
    from unittest import mock

    from sdm_client._rust import schema as bridge

    provider = bridge.DuckDBProvider
    with (
        mock.patch.dict(
            'os.environ',
            {
                _routing.ENVIRONMENT: ','.join(
                    _routing.FROM_SOURCE_DUCKDB_TABLES
                )
            },
        ),
        mock.patch.object(
            bridge,
            'DuckDBProvider',
            lambda held: _failing(provider(held), fail_columns_of),
        ),
        warnings.catch_warnings(record=True) as shown,
    ):
        warnings.simplefilter('always')
        schema = Schema.from_source(DuckDBSource(connection), tables)
        assert schema._core_description is not None
    said = [
        (type(w.message).__name__, str(w.message).split('] ', 1)[1])
        for w in _said(shown)
    ]
    return _projection(schema), said


def _named_on_the_old_route(
    monkeypatch: pytest.MonkeyPatch, connection: Any, tables: dict[str, str]
) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        schema = Schema.from_source(DuckDBSource(connection), tables)
    return _projection(schema), [
        (type(w.message).__name__, str(w.message)) for w in _old_said(shown)
    ]


#: Sources one physical table of which is named twice (T1.23 fix 1, Codex
#: P2): a declared foreign key to it joins the last name it was given, as
#: the reference maps source names to the schema's (`_declared_links`),
#: found by its exact name or, where the catalog folds it, by the one name
#: equal in any case.
REPEATED = {
    'exact': (
        (
            'CREATE TABLE parent (id BIGINT PRIMARY KEY)',
            'INSERT INTO parent SELECT i FROM range(4) t(i)',
            'CREATE TABLE child (child_id BIGINT PRIMARY KEY, parent_id BIGINT '
            'REFERENCES parent (id))',
            'INSERT INTO child SELECT i, i % 4 FROM range(9) t(i)',
        ),
        {'first': 'parent', 'second': 'parent', 'children': 'child'},
    ),
    'folded': (
        (
            'CREATE TABLE "Parent" (id BIGINT PRIMARY KEY)',
            'INSERT INTO "Parent" SELECT i FROM range(4) t(i)',
            'CREATE TABLE child (child_id BIGINT PRIMARY KEY, parent_id BIGINT '
            'REFERENCES parent (id))',
            'INSERT INTO child SELECT i, i % 4 FROM range(9) t(i)',
        ),
        {'p1': 'Parent', 'kids': 'child', 'p2': 'Parent'},
    ),
}


@pytest.mark.parametrize('name', sorted(REPEATED))
def test_a_source_table_named_twice_is_joined_as_the_old_route_joins_it(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    setup, tables = REPEATED[name]
    connection = duckdb.connect()
    for statement in setup:
        connection.execute(statement)
    core, said = _named_on_the_core(connection, tables)
    last = [n for n, s in tables.items() if s.lower() == 'parent'][-1]
    assert [last, 'id'] in [link[2:] for link in core['links']], core['links']
    if importlib.util.find_spec('sdm_relational_engine') is None:
        return
    old, old_said = _named_on_the_old_route(monkeypatch, connection, tables)
    assert core == old
    assert said == old_said


@pytest.mark.parametrize('alias', ['a.b', 'a`b', 'a\nb', 'a\rb'])
def test_a_name_the_schema_may_not_give_a_table_is_refused_as_the_old_route_does(
    monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    # T1.23 fix 1 (Codex P2): the schema's names are held to the
    # reference's rule (`_check_name`), the source's are not.
    with pytest.raises(StructuredDataError) as core:
        _named_on_the_core(_connection(), {alias: 'users'})
    assert core.value.code == 'INVALID_REQUEST'
    assert core.value.details['field'] == 'tables'
    if importlib.util.find_spec('sdm_relational_engine') is None:
        return
    with pytest.raises(StructuredDataError) as old:
        _named_on_the_old_route(monkeypatch, _connection(), {alias: 'users'})
    assert old.value.code == 'INVALID_REQUEST'


def test_a_name_over_a_quoted_source_table_is_taken_as_the_old_route_takes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The source's name may hold what the schema's may not; a plain name
    # over it is the schema's.
    connection = _connection(
        'CREATE TABLE "odd`name" (thing_id BIGINT PRIMARY KEY, label VARCHAR)',
        "INSERT INTO \"odd`name\" VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    )
    core, said = _named_on_the_core(connection, {'things': 'odd`name'})
    assert [t['name'] for t in core['tables']] == ['things']
    if importlib.util.find_spec('sdm_relational_engine') is None:
        return
    old, old_said = _named_on_the_old_route(
        monkeypatch, connection, {'things': 'odd`name'}
    )
    assert core == old
    assert said == old_said


def test_a_source_table_that_is_not_there_is_refused_as_the_old_route_does(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # T1.23 fix 1 (subagent P2-2): DuckDB fails reading a missing table's
    # columns; the core asks the source's table list then, and refuses the
    # caller's naming with the reference's code, not the provider's.
    with pytest.raises(StructuredDataError) as core:
        _named_on_the_core(_connection(), {'x': 'nope'})
    assert core.value.code == 'INVALID_REQUEST'
    assert core.value.details['field'] == 'tables'
    # A table that is there, in another case, is read.
    core_schema, _ = _named_on_the_core(_connection(), {'x': 'USERS'})
    assert [t['name'] for t in core_schema['tables']] == ['x']
    if importlib.util.find_spec('sdm_relational_engine') is None:
        return
    with pytest.raises(StructuredDataError) as old:
        _named_on_the_old_route(monkeypatch, _connection(), {'x': 'nope'})
    assert old.value.code == 'INVALID_REQUEST'


#: A source holding a table and a view of it, and a table named with a
#: capital that is not ASCII, which DuckDB does not fold.
HELD = (
    'CREATE TABLE parent (id BIGINT PRIMARY KEY, label VARCHAR)',
    "INSERT INTO parent VALUES (1, 'a'), (2, 'b'), (3, 'c')",
    'CREATE VIEW parent_view AS SELECT * FROM parent',
    'CREATE TABLE "Äpfel" (apfel_id BIGINT PRIMARY KEY)',
    'INSERT INTO "Äpfel" VALUES (1), (2)',
    # Relations no unqualified name reads (T1.23 fix 3): another schema's
    # table and view, an attached database's table.
    'CREATE SCHEMA other',
    'CREATE TABLE other.elsewhere (id BIGINT PRIMARY KEY)',
    'INSERT INTO other.elsewhere VALUES (1), (2)',
    'CREATE VIEW other.elsewhere_view AS SELECT * FROM other.elsewhere',
    "ATTACH ':memory:' AS att",
    'CREATE TABLE att.attached (id BIGINT PRIMARY KEY)',
    'INSERT INTO att.attached VALUES (1), (2)',
)


def test_a_relation_the_source_holds_keeps_its_provider_s_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # T1.23 fix 2 (Codex P2, subagent P3-5): the source says whether it
    # holds a relation of the name, a table or a view, by its own matching
    # of names (sdm-data's rendering); the core matches none, and calls
    # the caller's naming wrong only where the source holds nothing of it.
    def held() -> Any:
        connection = duckdb.connect()
        for statement in HELD:
            connection.execute(statement)
        return connection

    # A view is described, as on the old route.
    core, _ = _named_on_the_core(held(), {'valid': 'parent_view'})
    assert [t['name'] for t in core['tables']] == ['valid']
    # A failure reading a view's or a table's columns stays the provider's.
    for source in ('parent_view', 'parent', 'PARENT'):
        with pytest.raises(StructuredDataError) as failed:
            _named_on_the_core(held(), {'x': source}, fail_columns_of=source)
        assert failed.value.code == 'PROVIDER_ERROR', source
    # A table the source does not hold, or holds under a name DuckDB does
    # not fold to the one given, is the caller's naming, as on the old
    # route.
    for source in ('nope', 'äpfel', 'elsewhere', 'elsewhere_view', 'attached'):
        with pytest.raises(StructuredDataError) as missing:
            _named_on_the_core(held(), {'x': source})
        assert missing.value.code == 'INVALID_REQUEST', source
        assert missing.value.details['field'] == 'tables'
        if importlib.util.find_spec('sdm_relational_engine') is None:
            continue
        with pytest.raises(StructuredDataError) as old:
            _named_on_the_old_route(monkeypatch, held(), {'x': source})
        assert old.value.code == 'INVALID_REQUEST', source
    if importlib.util.find_spec('sdm_relational_engine') is None:
        return
    old_schema, _ = _named_on_the_old_route(
        monkeypatch, held(), {'valid': 'parent_view'}
    )
    assert core == old_schema


def test_where_the_core_and_the_old_route_differ_on_names_both_are_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Register P1B-5 (tables named, T1.23 fix 2) and P1B-11 (two faults of
    # one input), proposed: each outcome on both routes, so a change to
    # either fails here.
    dotted = (
        'CREATE TABLE "a.b" (id BIGINT PRIMARY KEY)',
        'INSERT INTO "a.b" VALUES (1), (2)',
    )
    # A plain name over a source table named with a dot: described on the
    # core, refused on the old route, whose pragma reads it as another name.
    core, _ = _named_on_the_core(_connection(*dotted), {'ok': 'a.b'})
    assert [t['name'] for t in core['tables']] == ['ok']
    # A qualified or quoted source name: the core reads it by name, finds
    # nothing and refuses the naming; the old route fails in its engine,
    # unnamed (GraphConstructionError).
    for source in ('main.users', '"users"'):
        with pytest.raises(StructuredDataError) as named:
            _named_on_the_core(_connection(), {'x': source})
        assert named.value.code == 'INVALID_REQUEST', source
    # An empty table under a name the schema may not give: the core checks
    # the names first (INVALID_REQUEST), the old route reads the table
    # first (an unnamed RuntimeError).
    empty = ('CREATE TABLE "a`b" (id BIGINT)',)
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with pytest.raises(StructuredDataError) as both:
        Schema.from_source(DuckDBSource(_connection(*empty)))
    assert both.value.code == 'INVALID_REQUEST'
    if importlib.util.find_spec('sdm_relational_engine') is None:
        return
    monkeypatch.delenv(_routing.ENVIRONMENT)
    with pytest.raises(StructuredDataError) as old_dotted:
        _named_on_the_old_route(
            monkeypatch, _connection(*dotted), {'ok': 'a.b'}
        )
    assert old_dotted.value.code == 'INVALID_REQUEST'
    for source in ('main.users', '"users"'):
        with pytest.raises(Exception) as old_named:
            _named_on_the_old_route(monkeypatch, _connection(), {'x': source})
        assert type(old_named.value).__name__ == 'GraphConstructionError'
        assert not isinstance(old_named.value, StructuredDataError)
    with pytest.raises(RuntimeError) as old_empty:
        Schema.from_source(DuckDBSource(_connection(*empty)))
    assert not isinstance(old_empty.value, StructuredDataError)


def test_a_listed_table_is_held_to_the_schema_s_rule_on_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # T1.23 fix 1, the listed path: a table the source lists under a name
    # the schema may not give is refused by the core, as the old route
    # refuses it.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with pytest.raises(StructuredDataError) as core:
        Schema.from_source(
            DuckDBSource(
                _connection(
                    'CREATE TABLE "a`b" (id BIGINT PRIMARY KEY)',
                    'INSERT INTO "a`b" VALUES (1), (2)',
                )
            )
        )
    assert core.value.code == 'INVALID_REQUEST'
    assert core.value.details['field'] == 'tables'


#: System views named as tables, over a source of one table and over the
#: shop base: the core's rule and the reference's outcome, as measured at
#: T1.24 (None: described). Each view's rows are the database's catalog,
#: so which rule the core meets first depends on what the source holds.
SYSTEM_VIEWS = [
    # Register entry P1B-10: a view holding a MAP column, which the
    # reference fails unnamed, reading the map as text.
    ('one table', 'duckdb_tables', 'column_type', 'RuntimeError'),
    ('shop', 'duckdb_tables', 'column_type', 'RuntimeError'),
    # Register entry P1B-11: a view whose preview holds no row, refused
    # INVALID_REQUEST since the owner's switch of 2026-10-13 (T1.32g).
    ('one table', 'duckdb_views', 'empty_table', 'RuntimeError'),
    # Views the reference describes, whose text the core refused as text
    # that could hold dates (P1B-2) until its text was read by the list
    # (T1.25b): described on both routes.
    ('one table', 'duckdb_columns', None, None),
    ('one table', 'pg_class', None, None),
    ('shop', 'pg_class', None, None),
    ('shop', 'duckdb_columns', None, None),
]


def _system_view_source(base: str) -> Any:
    if base == 'shop':
        return _connection()
    connection = duckdb.connect()
    connection.execute('CREATE TABLE t (a INTEGER)')
    connection.execute('INSERT INTO t VALUES (1)')
    return connection


def _named_view(
    monkeypatch: pytest.MonkeyPatch, routes: str, base: str, view: str
) -> Schema:
    if routes:
        monkeypatch.setenv(_routing.ENVIRONMENT, routes)
    else:
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(
            DuckDBSource(_system_view_source(base)), {'x': view}
        )


@pytest.mark.parametrize(('base', 'view', 'core_rule', 'old'), SYSTEM_VIEWS)
def test_a_system_view_named_as_a_table_is_pinned_on_both_routes(
    monkeypatch: pytest.MonkeyPatch,
    base: str,
    view: str,
    core_rule: str | None,
    old: str | None,
) -> None:
    # Reached through the public call from T1.24, whose route takes tables
    # named; T1.23's review carried it here. Pinned, naming both outcomes,
    # so a divergence fails this test the day either changes.
    tables = ','.join(_routing.FROM_SOURCE_DUCKDB_TABLES)
    if core_rule is None:
        assert list(_named_view(monkeypatch, tables, base, view).tables) == [
            'x'
        ]
    else:
        with pytest.raises(StructuredDataError) as refused:
            _named_view(monkeypatch, tables, base, view)
        assert type(refused.value).__name__ == (
            'StructuredDataError'
            if core_rule == 'empty_table'
            else 'UnsupportedRequestError'
        )
        assert refused.value.details['rule'] == core_rule
    if importlib.util.find_spec('sdm_relational_engine') is None:
        return
    if old is None:
        assert list(_named_view(monkeypatch, '', base, view).tables) == ['x']
        return
    with pytest.raises(Exception) as failed:  # noqa: PT011 - the reference's own
        _named_view(monkeypatch, '', base, view)
    assert type(failed.value).__name__ == old
    assert not isinstance(failed.value, StructuredDataError)


def _assert_text(schema: Any, table: str, column: str) -> None:
    r"""``column`` described as text: no refusal, not the time column, no
    timestamp (T1.33 part 3, the owner's item 24)."""
    assert schema[table].time_column != column
    described = schema[table].columns[column]
    assert str(described.dtype) == 'string'
    assert str(described.stype) != 'timestamp'


def _gate() -> Any:
    path = ROOT / 'contract/tests/t133p3_gate.py'
    spec = importlib.util.spec_from_file_location('t133p3_gate', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


#: T1.33 part 3's pending list (``contract/tests/t133p3-pending.json``):
#: a check named there must still fail; once it passes it fails, naming
#: itself, until it is taken out.
_T133P3 = _gate()


def _offset_events(zone: str) -> Any:
    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE events (id BIGINT PRIMARY KEY, event_date VARCHAR)'
    )
    connection.execute(
        'INSERT INTO events VALUES '
        + ', '.join(
            f"({i}, '2024-01-0{i}T01:02:03{zone}')" for i in range(1, 5)
        )
    )
    return connection


@pytest.mark.parametrize(
    ('zone', 'read', 'core_reads'),
    [
        # An offset of a day or more, either way: no time to the reference
        # (T1.25 fix round 1, Codex P2); outside the core's list from
        # T1.25b, refused by name (register DT-15).
        ('+24:00', False, False),
        ('+99:00', False, False),
        ('+2360', False, False),
        ('-25:00', False, False),
        # Minutes past 59, which the reference counts on: outside the list.
        ('+01:99', True, False),
        ('+0160', True, False),
        # Offsets the list reads, and zero either sign.
        ('+23:59', True, True),
        ('-23:59', True, True),
        ('+00', True, True),
        ('-00', True, True),
        ('Z', True, True),
    ],
)
def test_text_with_an_offset_is_read_by_the_list_or_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, zone: str, read: bool, core_reads: bool
) -> None:
    # The old route describes each with its engine's pandas reading. The
    # core reads an offset in the list as a timestamp as the reference
    # does, and refuses text outside the list (text_time) where it could
    # make the column one (T1.25b).
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    if importlib.util.find_spec('sdm_relational_engine') is not None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            old = Schema.from_source(DuckDBSource(_offset_events(zone)))
        assert (old['events'].time_column == 'event_date') is read
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if core_reads:
            core = Schema.from_source(DuckDBSource(_offset_events(zone)))
            assert core['events'].time_column == 'event_date'
            assert (
                str(core['events'].columns['event_date'].stype) == 'timestamp'
            )
        else:
            # T1.33 part 3 (the owner's item 24): values the core does not
            # read as times are text, under any name, not refused.
            def check() -> None:
                core = Schema.from_source(DuckDBSource(_offset_events(zone)))
                _assert_text(core, 'events', 'event_date')

            _T133P3.held(
                'exits',
                'py:test_the_schema_route::'
                f'test_text_with_an_offset_is_read_by_the_list_or_refused_by_name[{zone}-{read}-{core_reads}]',
                check,
            )


def _text_events(values: list[str]) -> Any:
    connection = duckdb.connect()
    connection.execute(
        'CREATE TABLE events (id BIGINT PRIMARY KEY, event_time VARCHAR)'
    )
    connection.executemany(
        'INSERT INTO events VALUES (?, ?)', list(enumerate(values))
    )
    return connection


@pytest.mark.parametrize(
    'values',
    [
        # A year the list does not hold, which an offset would carry into
        # one it does (T1.25b fix 1): outside the list, so no times, and
        # from T1.33 part 3 (the owner's item 24) text under any name, not
        # refused (DT-15's text_time reversed), the year 0 and its mirror
        # past 9999.
        ['0000-12-31T23:00-02:00'] * 3,
        ['9999-12-31T23:00-02:00'] * 3,
    ],
    ids=['year-0', 'past-9999'],
)
def test_text_the_list_does_not_read_is_text(
    monkeypatch: pytest.MonkeyPatch, values: list[str], request: Any
) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)

    def check() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            core = Schema.from_source(DuckDBSource(_text_events(values)))
        _assert_text(core, 'events', 'event_time')

    _T133P3.held(
        'exits',
        f'py:test_the_schema_route::{request.node.name}',
        check,
    )


@pytest.mark.parametrize(
    ('values', 'rule'),
    [
        # Values in the list whose day and month the column leaves open
        # (register DT-16): refused under the order's own rule.
        (['03/15/99', '15/11/00'], 'text_time_order'),
    ],
)
def test_text_the_list_does_not_decide_is_refused_under_its_rule(
    monkeypatch: pytest.MonkeyPatch, values: list[str], rule: str
) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(StructuredDataError) as refused:
            Schema.from_source(DuckDBSource(_text_events(values)))
    assert refused.value.details['rule'] == rule


def test_a_year_the_list_holds_after_an_offset_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The neighbours of the year 0's case, inside the list: the year written
    # is 1 or 9999 and the instant stays in the years 1 to 9999.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    values = ['0001-01-01T00:30-01:00', '9999-12-31T22:00-01:00'] * 2
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        core = Schema.from_source(DuckDBSource(_text_events(values)))
    assert str(core['events'].columns['event_time'].stype) == 'timestamp'
