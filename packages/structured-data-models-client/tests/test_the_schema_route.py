# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``Schema.from_source`` on the core route (T1.11b).

For a DuckDB source with no tables named, the core's ``describe_source``
reads the source through the Python host's DuckDB provider and the schema
bridge builds today's ``Schema``: on ``base.regression``'s base it is the
reference's. The relational engine is not imported on the route, and an
import of it there fails. The providers import no pandas, which an AST scan
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
    script = f"""
    import sys
    sys.path.insert(0, {str(tmp_path)!r})
    import duckdb
    from sdm_client import Schema
    from sdm_client._routing import RouteError
    from sdm_client._rust import providers, schema
    from sdm_client._rust.providers import duckdb as provider
    from sdm_connectors import DuckDBSource

    statements = {_shop()!r}
    connection = duckdb.connect()
    for statement in statements:
        connection.execute(statement)
    before = 'sdm_relational_engine' in sys.modules
    Schema.from_source(DuckDBSource(connection))
    print(before, 'sdm_relational_engine' in sys.modules)

    # An import of the engine on the route fails the route.
    answer = provider.DuckDBProvider.__call__
    def importing(self, step):
        import sdm_relational_engine
        return answer(self, step)
    provider.DuckDBProvider.__call__ = importing
    try:
        Schema.from_source(DuckDBSource(connection))
    except RouteError as refused:
        print('refused', 'sdm_relational_engine' in str(refused))
    print('sdm_relational_engine' in sys.modules)
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

    def recording(self: Any, step: dict[str, Any]) -> Any:
        steps.append(step)
        table = answer(self, step)
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
    ]
    for build, arguments, rule in refusals:
        with pytest.raises(StructuredDataError) as refused:
            Schema.from_source(DuckDBSource(build()), **arguments)
        assert refused.value.code == 'UNSUPPORTED_REQUEST'
        assert refused.value.details['rule'] == rule
    for arguments in (
        {'relationships': []},
        {'time_columns': {'orders': None}},
    ):
        with pytest.raises(StructuredDataError) as refused:
            Schema.from_source(DuckDBSource(_connection()), **arguments)
        assert refused.value.code == 'UNSUPPORTED_REQUEST'
