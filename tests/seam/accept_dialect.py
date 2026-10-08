# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P3.dialect: every SQL text the engine runs is written by a dialect
function.

Writing SQL is separate from running it: each database's SQL comes from plain
functions in ``rfm/backend/<db>/dialect.py`` that return ``(sql, params)``
and touch no connection, and the engine hands what they return to
``Source._run``.

``_dialect_probe.py`` patches every function of every dialect module in
place, so that the SQL text it returns ends in a marker comment, and then
runs:

* every recorded SQLite, DuckDB and PostgreSQL corpus scenario, recorded
  afresh against its real database;
* the Databricks and Snowflake predictions of ``_warehouse.py``;
* a SQLite, DuckDB and PostgreSQL prediction from a schema read without
  naming its tables.

Every statement any driver receives must end with the marker, and apart from
it be the statement recorded, or pinned, without the probe. A statement the
engine built itself, or rebuilt from a dialect function's text, lacks it.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import pytest
from _seam import (
    DIALECT_BACKENDS,
    DRIVER_CALLS,
    ENGINE_SRC,
    REPO,
    SEAM,
    TRANSCRIPTS,
    load_transcript,
    output,
    run_python,
    scenario_dirs,
)

PROBE = SEAM / '_dialect_probe.py'
MARK = '\n/* sdm-seam-dialect */'
DRIVERS = (
    'duckdb',
    '_duckdb',
    'adbc_driver_manager',
    'adbc_driver_sqlite',
    'psycopg',
    'psycopg2',
    'sqlite3',
    'snowflake',
    'databricks',
    'pyspark',
    'sdm_connectors.source',
    'sdm_connectors.backends',
)
CONNECTION_PARAMETERS = {'connection', 'conn', 'cursor', 'session'}


def _dialect_path(backend: str) -> Path:
    return (
        ENGINE_SRC
        / 'sdm_relational_engine'
        / 'rfm'
        / 'backend'
        / backend
        / 'dialect.py'
    )


@pytest.mark.parametrize('backend', DIALECT_BACKENDS)
def test_the_dialect_module_exists(backend: str) -> None:
    path = _dialect_path(backend)
    assert path.is_file(), f'{path.relative_to(REPO)} does not exist'
    tree = ast.parse(path.read_text())
    functions = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
    ]
    assert functions, f'{path.relative_to(REPO)} defines no function'


@pytest.mark.parametrize('backend', DIALECT_BACKENDS)
def test_the_dialect_touches_no_connection(backend: str) -> None:
    path = _dialect_path(backend)
    if not path.is_file():
        pytest.fail(f'{path.relative_to(REPO)} does not exist')
    rel = path.relative_to(REPO).as_posix()
    tree = ast.parse(path.read_text())
    found: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, 'lineno', 0)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in DRIVER_CALLS | {'run', '_run'}:
                found.append(f'{rel}:{line}: calls .{node.func.attr}()')
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(DRIVERS):
                    found.append(f'{rel}:{line}: imports {alias.name}')
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module] + [
                f'{node.module}.{a.name}' for a in node.names
            ]
            for name in names:
                if name.startswith(DRIVERS):
                    found.append(f'{rel}:{line}: imports {name}')
                    break
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = node.args
            for arg in [
                *args.posonlyargs,
                *args.args,
                *args.kwonlyargs,
            ]:
                if arg.arg in CONNECTION_PARAMETERS:
                    found.append(
                        f'{rel}:{line}: {node.name}() takes {arg.arg!r}'
                    )
    assert not found, 'a dialect module reaches a connection:\n  ' + (
        '\n  '.join(found)
    )


def _probe(out: Path, *args: Any, timeout: float = 1800) -> dict[str, Any]:
    proc = run_python(PROBE, out, *args, timeout=timeout)
    assert proc.returncode == 0, output(proc)
    report = json.loads(out.read_text())
    assert 'error' not in report, report['error']
    return report


def _unmarked(statements: list[str], where: str) -> list[str]:
    return [
        f'{where} statement {i}: {s[-200:]!r}'
        for i, s in enumerate(statements)
        if not s.endswith(MARK)
    ]


@pytest.fixture(scope='module')
def corpus_probe(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    names = [
        name
        for name, directory in scenario_dirs(TRANSCRIPTS).items()
        if load_transcript(directory / 'transcript.json')['backend']
        in ('sqlite', 'duckdb', 'postgres')
    ]
    assert len(names) >= 18, f'only {len(names)} SQL scenarios recorded'
    root = tmp_path_factory.mktemp('dialect-corpus')
    fresh = root / 'corpus'
    fresh.mkdir()
    if not os.getenv('SDM_POSTGRES_TEST_DSN'):
        pytest.fail('the PostgreSQL scenarios need SDM_POSTGRES_TEST_DSN')
    report = _probe(
        root / 'report.json',
        'corpus',
        'record',
        '--corpus',
        fresh,
        '--only',
        ','.join(n.replace('--', '#') for n in names),
    )
    assert report.get('exit') == 0, f'marked recording failed: {report}'
    return {'report': report, 'names': names, 'fresh': fresh}


def test_every_corpus_statement_comes_from_a_dialect(
    corpus_probe: dict[str, Any],
) -> None:
    fresh: Path = corpus_probe['fresh']
    problems: list[str] = []
    for name in corpus_probe['names']:
        marked_path = fresh / name / 'transcript.json'
        if not marked_path.is_file():
            problems.append(f'{name}: no marked recording')
            continue
        marked = load_transcript(marked_path)['steps']
        stored = load_transcript(TRANSCRIPTS / name / 'transcript.json')[
            'steps'
        ]
        sql = [s['sql'] for s in marked if s['op'] == 'execute']
        if not sql:
            problems.append(f'{name}: no SQL ran')
        problems += _unmarked(sql, name)
        if [s['op'] for s in marked] != [s['op'] for s in stored]:
            problems.append(f'{name}: the marked run took other steps')
            continue
        for index, (m, s) in enumerate(zip(marked, stored, strict=True)):
            if m['op'] != 'execute':
                continue
            if m['sql'] != s['sql'] + MARK:
                problems.append(
                    f'{name} step {index}: the stored SQL is not what the '
                    f'dialect returned'
                )
    assert not problems, '\n'.join(problems[:60])


@pytest.mark.parametrize(
    'name',
    [
        'databricks',
        'databricks-spark',
        'databricks-metric-view',
        'snowflake',
        'snowflake-semantic-view',
    ],
)
def test_every_warehouse_statement_comes_from_a_dialect(
    name: str, tmp_path: Path
) -> None:
    report = _probe(tmp_path / 'report.json', 'warehouse', name)
    statements = report['statements']
    assert statements, f'{name}: no statement ran'
    unmarked = _unmarked(statements, name)
    assert not unmarked, '\n'.join(unmarked[:40])
    golden = json.loads((SEAM / 'golden' / 'warehouse_sql.json').read_text())
    pinned = [s['sql'] for s in golden[name]['statements']]
    assert [s[: -len(MARK)] for s in statements] == pinned, (
        f'{name}: the marked run sent other statements than are pinned'
    )


@pytest.mark.parametrize('backend', ['sqlite', 'duckdb', 'postgres'])
def test_every_local_statement_comes_from_a_dialect(
    backend: str, tmp_path: Path
) -> None:
    if backend == 'postgres' and not os.getenv('SDM_POSTGRES_TEST_DSN'):
        pytest.fail('the PostgreSQL check needs SDM_POSTGRES_TEST_DSN')
    report = _probe(
        tmp_path / 'report.json', 'local', backend, tmp_path / 'work'
    )
    statements = report['statements']
    assert len(statements) >= 10, f'{backend}: only {statements} ran'
    unmarked = _unmarked(statements, backend)
    assert not unmarked, '\n'.join(unmarked[:40])


def test_every_dialect_module_writes_sql(
    corpus_probe: dict[str, Any], tmp_path: Path
) -> None:
    returned = dict(corpus_probe['report']['returned'])
    for name in ('databricks', 'snowflake'):
        report = _probe(tmp_path / f'{name}.json', 'warehouse', name)
        for key, count in report['returned'].items():
            returned[key] = returned.get(key, 0) + count
    for backend in DIALECT_BACKENDS:
        module = f'sdm_relational_engine.rfm.backend.{backend}.dialect'
        used = {k: v for k, v in returned.items() if k.startswith(module + '.')}
        assert used, f'no function of {module} returned SQL'


def test_the_probe_marks_a_function_however_it_is_reached() -> None:
    import sys
    import types
    from typing import NamedTuple

    import _dialect_probe

    module = types.ModuleType('seam_probe_selftest_dialect')
    sys.modules[module.__name__] = module
    source = (
        'from typing import NamedTuple\n'
        'class Statement(NamedTuple):\n'
        '    sql: str\n'
        '    params: list | None\n'
        'def select(table, *, limit=10):\n'
        "    return f'SELECT * FROM {table} LIMIT {limit}', None\n"
        'def lookup(key):\n'
        "    return Statement('SELECT ? AS k', [key])\n"
        'def chunks(keys):\n'
        "    return [('SELECT ?', [k]) for k in keys]\n"
        'def quote(name):\n'
        "    return '\"' + name + '\"'\n"
        'class Dialect:\n'
        '    @staticmethod\n'
        '    def count(table):\n'
        "        return (f'SELECT COUNT(*) FROM {table}', [])\n"
    )
    module.__file__ = f'<{module.__name__}>'
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    held = module.select  # as `from dialect import select` would hold it
    functions = _dialect_probe._functions(module)
    assert sorted(q for q, _ in functions) == [
        'Dialect.count',
        'chunks',
        'lookup',
        'quote',
        'select',
    ]
    for qualname, func in functions:
        _dialect_probe._patch(module, qualname, func)
    try:
        assert held('t', limit=3) == ('SELECT * FROM t LIMIT 3' + MARK, None)
        statement = module.lookup(7)
        assert type(statement).__name__ == 'Statement'
        assert statement == ('SELECT ? AS k' + MARK, [7])
        assert module.chunks([1, 2]) == [
            ('SELECT ?' + MARK, [1]),
            ('SELECT ?' + MARK, [2]),
        ]
        assert module.quote('a') == '"a"'
        assert module.Dialect.count('t') == (
            'SELECT COUNT(*) FROM t' + MARK,
            [],
        )
        assert _dialect_probe.MARK == MARK
        assert NamedTuple
    finally:
        del sys.modules[module.__name__]
