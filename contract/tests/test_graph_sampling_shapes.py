# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current graph sampling preserves declared shapes and local links."""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]

RUNNER = ROOT / 'contract/runners/run_python.py'

CASES = ROOT / 'contract/cases/s1'

DATABASES = ('duckdb', 'sqlite', 'postgres')

SHAPES = {
    'diamond': [
        ('orders', 'cid', 'customers'),
        ('customers', 'aid', 'addresses'),
        ('orders', 'aid', 'addresses'),
    ],
    'same-table-twice': [
        ('orders', 'cid', 'customers'),
        ('orders', 'cid', 'buyers'),
    ],
    'chain': [
        ('lines', 'oid', 'orders'),
        ('orders', 'cid', 'customers'),
        ('customers', 'aid', 'addresses'),
    ],
    'star': [
        ('orders', 'cid', 'customers'),
        ('orders', 'pid', 'products'),
        ('orders', 'sid', 'stores'),
    ],
}

KEYS = {
    'addresses': 'aid',
    'buyers': 'cid',
    'customers': 'cid',
    'lines': 'lid',
    'orders': 'oid',
    'products': 'pid',
    'stores': 'sid',
}

FROM_SOURCE = 'shape.diamond-from-source'

ONE_COLUMN = 'shape.one-column-two-tables'


def _tables(shape: str) -> list[str]:
    return sorted({t for edge in SHAPES[shape] for t in (edge[0], edge[2])})


SAMPLED = {
    f'shape.{shape}.{entity}': (shape, entity)
    for shape in SHAPES
    for entity in _tables(shape)
    if entity not in {edge[0] for edge in SHAPES[shape]}
}

TWO_PATHS = {
    'shape.two-paths.keyless': {'customers': 'cid', 'headers': 'hid'},
    'shape.two-paths.keyed': {
        'customers': 'cid',
        'headers': 'hid',
        'details': 'did',
    },
}

TWO_PATH_LINKS = [
    ('headers', 'cid', 'customers', 'cid'),
    ('details', 'hid', 'headers', 'hid'),
    ('details', 'cid', 'customers', 'cid'),
]

CASE_IDS = (*SAMPLED, FROM_SOURCE, ONE_COLUMN, *TWO_PATHS)


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('test_graph_sampling_shapes_run_python', RUNNER)


def key(case_id: str, database: str) -> str:
    return runner().case_key(case_id, database)


KEYS_ALL = tuple(key(c, d) for d in DATABASES for c in CASE_IDS)


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""The cases by id, each with its digest, as the runner loads
    them."""
    loaded = runner().load_cases([CASES / f'{c}.json' for c in CASE_IDS])
    return {case['id']: (case, digest) for case, digest in loaded}


def _links(case: dict[str, Any]) -> list[tuple[str, str, str, str]]:
    return [
        (
            link['source_table'],
            link['source_column'],
            link['target_table'],
            link['target_column'],
        )
        for link in case['input']['from_source']['relationships']
    ]


def _entity(case: dict[str, Any]) -> tuple[str, str]:
    table, column = case['input']['pql'].rsplit(' FOR EACH ', 1)[1].split('.')
    return table, column


@pytest.mark.parametrize('case_id', list(SAMPLED))
def test_each_shape_is_main_s_sampled_from_main_s_tables(case_id: str) -> None:
    r"""Exit "``diamond``, ``same-table-twice``, ``chain``, ``star``": the
    case gives ``Schema.from_source`` the shape's tables, keys and links as
    ``main`` writes them, samples it from one of its tables, and runs
    through predict on all three databases."""
    shape, entity = SAMPLED[case_id]
    case, _ = cases()[case_id]
    given = case['input']['from_source']
    assert case['status'] == 'acceptance' and case['slice'] == 'S1'
    assert case['until'] == 'predict'
    assert runner().case_databases(case) == DATABASES
    assert sorted(given['tables']) == _tables(shape)
    assert given['primary_keys'] == {t: KEYS[t] for t in _tables(shape)}
    assert _links(case) == [(s, c, d, KEYS[d]) for s, c, d in SHAPES[shape]]
    assert _entity(case) == (entity, KEYS[entity])
    assert case['input']['materialize']['indices'] == [0, 1, 2]


def _built(case: dict[str, Any]) -> Any:
    r"""The case's DuckDB database, its base and setup run, as the runner
    builds it."""
    import duckdb

    connection = duckdb.connect()
    for statement in [
        *runner().load_bases()[case['input']['base']],
        *case['input'].get('setup', []),
    ]:
        connection.execute(statement)
    return connection


def test_the_from_source_diamond_declares_the_diamond_in_its_catalog() -> None:
    r"""Exit "a ``from_source`` diamond compared at the payload": the case
    calls ``Schema.from_source(source)`` with nothing given, over a catalog
    declaring the diamond's keys and links and no other table, sampled
    from addresses, the table both paths reach, through predict."""
    case, _ = cases()[FROM_SOURCE]
    assert 'from_source' not in case['input']
    assert case['until'] == 'predict'
    assert runner().case_databases(case) == DATABASES
    connection = _built(case)
    try:
        tables = sorted(
            row[0]
            for row in connection.execute(
                "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
            ).fetchall()
        )
        rows = connection.execute(
            'SELECT table_name, constraint_type, constraint_column_names, '
            'referenced_table, referenced_column_names '
            "FROM duckdb_constraints() WHERE schema_name = 'main' "
            "AND constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')"
        ).fetchall()
    finally:
        connection.close()
    assert tables == _tables('diamond')
    keys = {t: c[0] for t, kind, c, _, _ in rows if kind == 'PRIMARY KEY'}
    links = sorted(
        (t, c[0], d, r[0]) for t, kind, c, d, r in rows if kind == 'FOREIGN KEY'
    )
    assert keys == {t: KEYS[t] for t in _tables('diamond')}
    assert links == sorted((s, c, d, KEYS[d]) for s, c, d in SHAPES['diamond'])
    assert _entity(case) == ('addresses', 'aid')


@pytest.mark.parametrize('case_id', list(TWO_PATHS))
def test_the_events_are_reached_by_two_paths(case_id: str) -> None:
    r"""Exit "a table reached by two paths in one hop": the events link to
    the customers directly and through the orders, as ``main``'s
    ``test_two_paths.py`` links them, and are counted for each customer;
    keyed, each line's customer is not its order's, so the orders reach
    events the customers do not."""
    case, _ = cases()[case_id]
    given = case['input']['from_source']
    assert case['until'] == 'predict'
    assert runner().case_databases(case) == DATABASES
    assert given['primary_keys'] == TWO_PATHS[case_id]
    assert _links(case) == TWO_PATH_LINKS
    assert _entity(case) == ('customers', 'cid')
    assert 'COUNT(details.*' in case['input']['pql']
    connection = _built(case)
    try:
        own = connection.execute(
            'SELECT count(*) FROM details d JOIN headers h USING (hid) '
            'WHERE d.cid = h.cid'
        ).fetchone()[0]
        total = connection.execute('SELECT count(*) FROM details').fetchone()[0]
    finally:
        connection.close()
    assert own == (total if case_id.endswith('keyless') else 0)


def test_one_column_links_two_tables() -> None:
    r"""Exit "one column linking two tables": one column of a keyless event
    table refers to two tables, each by its key."""
    case, _ = cases()[ONE_COLUMN]
    assert case['until'] == 'predict'
    assert runner().case_databases(case) == DATABASES
    links = _links(case)
    assert len(links) == 2
    assert links[0][:2] == links[1][:2] and links[0][2] != links[1][2]
    source = links[0][0]
    assert source not in case['input']['from_source']['primary_keys']
    assert case['input']['from_source']['time_columns'] == {source: 'ts'}
    assert _entity(case)[0] in {links[0][2], links[1][2]}


def _server() -> bool:
    r"""Whether a Postgres server is named; where none is, one must not be
    required."""
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


@functools.cache
def core_run(database: str) -> dict[str, Any]:
    r"""python-core over the cases on ``database``, as that database's
    job runs them: each observation by its key, or under ``''`` why the run
    failed whole."""
    with tempfile.TemporaryDirectory(prefix='graph_sampling_shapes-') as tmp:
        given = Path(tmp) / 'cases'
        given.mkdir()
        for case_id in CASE_IDS:
            shutil.copy(CASES / f'{case_id}.json', given)
        out = Path(tmp) / 'python-core.jsonl'
        done = subprocess.run(
            [
                sys.executable,
                '-I',
                str(RUNNER),
                '--mode',
                'python-core',
                '--database',
                database,
                '--out',
                str(out),
                str(given),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            check=False,
            timeout=1800,
        )
        if done.returncode != 0:
            return {'': done.stdout + done.stderr}
        return {
            json.loads(line)['case']: json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        }


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_current_graph_shapes_preserve_links(case_key):
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        pytest.skip('SDM_POSTGRES_TEST_DSN is not configured')
    observed = core_run(database)
    assert '' not in observed, observed.get('')
    line = observed[case_key]
    assert line['outcome'] == {
        'result': 'success',
        'stage': 'predict',
        'error': None,
        'code': None,
    }
    assert line['identity']['native_loaded']
    assert {'describe_source', 'materialize', 'predict'} <= set(
        line['identity']['operations']
    )
    assert line['public']['result']['rows'] > 0
    assert line['public']['payloads']
    links = _module(
        'sampled_links', Path(__file__).with_name('_sampled_links.py')
    )
    for payload in line['public']['payloads']:
        body = payload['parsed']
        case_id = case_key.split('@')[0]
        if case_id == FROM_SOURCE:
            expected = set(SHAPES['diamond'])
        else:
            case, _ = cases()[case_id]
            expected = {
                (source, column, target)
                for source, column, target, _ in _links(case)
            }
        actual = {
            (
                relation['source_table'],
                relation['source_columns'][0],
                relation['target_table'],
            )
            for relation in body['schema']['relationships']
            if relation.get('source_table') is not None
        }
        sampled = set(body['schema']['related_tables'])
        # Fixed fixture expectations, independent of the candidate's output.
        if case_id in SAMPLED:
            shape, _ = SAMPLED[case_id]
            required = set(_tables(shape))
            if shape == 'chain':
                required.remove(
                    'lines'
                )  # Three edges away; default is two hops.
        elif case_id == FROM_SOURCE:
            required = set(_tables('diamond'))
        elif case_id == ONE_COLUMN:
            required = {'customers', 'buyers', 'details'}
        else:
            required = {'customers', 'headers', 'details'}
        assert sampled == required
        for table in required:
            assert body['predict']['related_tables'][table]['rows'], table
        # Hop limits can leave deeper tables (for example chain.lines) unsampled.
        expected = {
            edge
            for edge in expected
            if edge[0] in sampled and edge[2] in sampled
        }
        assert actual == expected
        links.assert_local_links(body)
        if case_key.split('@')[0] == FROM_SOURCE:
            links.assert_diamond_links_retained(body)
