# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQL providers return Arrow data without depending on pandas conversion."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import duckdb
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import Schema
from sdm_client._rust import providers
from sdm_client._rust.providers.duckdb import DuckDBProvider

PROVIDERS = Path(providers.__file__).resolve().parent


def _shop() -> list[str]:
    return json.loads((Path(__file__).parent / 'data/shop.json').read_text())[
        'statements'
    ]


def _connection() -> Any:
    connection = duckdb.connect()
    for statement in _shop():
        connection.execute(statement)
    return connection


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
