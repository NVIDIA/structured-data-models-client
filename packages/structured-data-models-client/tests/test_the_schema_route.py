# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Schema.from_source`` on the core route (T1.11b).

For a DuckDB source with no tables named, the core's ``describe_source``
reads the source through the Python host's DuckDB provider and the schema
bridge builds today's ``Schema``: on ``base.regression``'s base it is the
reference's. The relational engine is not imported on the route, which a
test-only import hook in a process of its own shows, and a core route
running leaves another thread's old-route call as it is alone. The providers import no pandas, which an AST scan
of ``_rust/providers`` and a run with pandas unimportable show. What the
core does not describe yet is refused by name (register entry P1B-2).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any

import duckdb
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import Schema, StructuredDataError, _routing
from sdm_client._rust import providers
from sdm_client._rust.providers.duckdb import DuckDBProvider

ROOT = Path(__file__).resolve().parents[3]
CORE = ','.join(_routing.FROM_SOURCE_DUCKDB)
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
    assert not set(_routing.FROM_SOURCE_DUCKDB_TABLES) <= _routing.CORE_ROUTED
    monkeypatch.setenv(_routing.ENVIRONMENT, _routing.FROM_SOURCE_DUCKDB[0])
    with pytest.raises(_routing.RouteError, match='one route'):
        Schema.from_source(DuckDBSource(_connection()))
    # Tables named have no core route yet: forcing the keys path fails them.
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with pytest.raises(_routing.RouteError, match='one route'):
        Schema.from_source(DuckDBSource(_connection()), {'users': 'users'})


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
        (
            lambda: _connection('CREATE TABLE z (z_id BIGINT)'),
            {},
            'empty_table',
        ),
        (_connection, {'primary_keys': {'users': 'user_id'}}, 'primary_keys'),
        (_connection, {'time_columns': {'orders': 'ts'}}, 'time_columns'),
        # A time of day, which the reference reads as no time and refuses
        # as a time column, with a warning.
        (
            lambda: _connection(
                'CREATE TABLE things (thing_id BIGINT, tod TIME)',
                "INSERT INTO things SELECT i, TIME '01:00:00' + INTERVAL (i) "
                'MINUTE FROM range(30) t(i)',
            ),
            {},
            'time_of_day',
        ),
    ]
    for build, arguments, rule in refusals:
        with pytest.raises(StructuredDataError) as refused:
            Schema.from_source(DuckDBSource(build()), **arguments)
        assert refused.value.code == 'UNSUPPORTED_REQUEST'
        assert refused.value.details['rule'] == rule
    for arguments, rule in (
        ({'relationships': []}, 'relationships'),
        ({'time_columns': {'orders': None}}, 'time_column_cleared'),
    ):
        with pytest.raises(StructuredDataError) as refused:
            Schema.from_source(DuckDBSource(_connection()), **arguments)
        assert refused.value.code == 'UNSUPPORTED_REQUEST'
        assert refused.value.details['rule'] == rule
