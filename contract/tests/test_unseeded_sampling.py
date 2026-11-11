# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unseeded native sampling: fresh draws, valid source rows and labels."""

from __future__ import annotations

import datetime
import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = ROOT / 'contract/tests/unseeded'
DATABASES = ('duckdb', 'sqlite', 'postgres')
CASE_IDS = tuple(sorted(path.stem for path in CASES.glob('*.json')))


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('test_unseeded_sampling_run_python', RUNNER)


KEYS_ALL = tuple(runner().case_key(c, d) for d in DATABASES for c in CASE_IDS)


FEATURES = {
    'users': ('user_id', 'age', 'country'),
    'orders': ('price', 'ts'),
    'lines': ('qty',),
}


def written(value: Any) -> Any:
    r"""A value as a request writes it, read back: a number as a float, a
    timestamp as its ISO text with microseconds and a Z, text as itself."""
    if isinstance(value, dict) and set(value) == {'number'}:
        return float(value['number'])
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, datetime.datetime):
        return value.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    return value


@functools.cache
def source(case_id: str) -> dict[str, Any]:
    """Read authored source data, independently of client sampling."""
    import duckdb

    case = cases()[case_id]
    connection = duckdb.connect()
    try:
        for statement in [
            *runner().load_bases()[case['input']['base']],
            *case['input'].get('setup', []),
        ]:
            connection.execute(statement)
        tables = ['users', 'orders'] + (
            ['lines'] if case_id.endswith('.chain') else []
        )
        return {
            'entities': {
                row[0]
                for row in connection.execute(
                    'SELECT user_id FROM users'
                ).fetchall()
            },
            'orders': connection.execute(
                'SELECT user_id, ts, price FROM orders'
            ).fetchall(),
            'source': {
                table: {
                    tuple(written(value) for value in row)
                    for row in connection.execute(
                        f'SELECT {", ".join(FEATURES[table])} FROM {table}'
                    ).fetchall()
                }
                for table in tables
            },
        }
    finally:
        connection.close()


def label(case_id: str, entity: Any, anchor: str) -> float:
    """Independent SUM over the open-lower, closed-upper 30-day interval."""
    start = datetime.datetime.fromisoformat(anchor[:26])
    end = start + datetime.timedelta(days=30)
    return sum(
        (
            price
            for user, time, price in source(case_id)['orders']
            if user == entity and start < time <= end
        ),
        0.0,
    )


def sent(line: dict[str, Any]) -> dict[str, Any]:
    r"""The one request a line's endpoint received, parsed."""
    payloads = line['public']['payloads']
    assert len(payloads) == 1, len(payloads)
    return payloads[0]['parsed']


def examples(line: dict[str, Any]) -> list[tuple[Any, str, Any]]:
    r"""Each context example of a line's request: its entity, anchor time
    and label."""
    table = sent(line)['context']['instance_table']
    at = {name: i for i, name in enumerate(table['columns'])}
    return [
        (
            written(row[at['__kumo_entity_ref_0']]),
            row[at['__kumo_anchor_time']],
            written(row[at['TARGET']]),
        )
        for row in table['rows']
    ]


@functools.cache
def cases() -> dict[str, dict[str, Any]]:
    return {
        case['id']: case
        for case, _ in runner().load_cases(
            [CASES / f'{c}.json' for c in CASE_IDS]
        )
    }


@pytest.mark.parametrize('case_id', CASE_IDS)
def test_each_case_asks_for_an_unseeded_sample(case_id: str) -> None:
    r"""Exit "``unseeded``": the case passes ``random_seed=None`` through
    predict on all three databases; fixed fixture expectations describe
    successful distinct draws. Its seeded twin is ``sample.seed-7``."""
    case = cases()[case_id]
    assert case['input']['materialize']['random_seed'] is None
    assert case['until'] == 'predict'
    assert runner().case_databases(case) == DATABASES
    assert case['core']['materialize']['seed'] is None
    assert (ROOT / 'contract/cases/s1/sample.seed-7.json').exists()


def _server() -> bool:
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


@functools.cache
def core_run(database: str, run: int = 0) -> dict[str, Any]:
    r"""python-core over the unseeded cases on ``database``: each
    observation by its key, or under ``''`` why the run failed whole."""
    with tempfile.TemporaryDirectory(prefix='unseeded_sampling-') as tmp:
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
                str(CASES),
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


def assert_structure(case_id: str, line: dict[str, Any]) -> None:
    assert line['outcome'] == {
        'result': 'success',
        'stage': 'predict',
        'error': None,
        'code': None,
    }
    request = sent(line)
    chain = case_id.endswith('.chain')
    tables = ['lines', 'orders', 'users'] if chain else ['orders', 'users']
    schema = request['schema']

    def column(dtype, stype=None, required=False):
        result = {'dtype': dtype}
        if stype is not None:
            result['stype'] = stype
        if required:
            result['nullable'] = False
        return result

    required_id = column('int64', 'ID', True)
    assert schema['instance_table'] == {
        'primary_key': 'instance_id',
        'columns': {
            'instance_id': required_id,
            '__kumo_entity_ref_0': column('int64', 'ID'),
            '__kumo_anchor_time': column('timestamp[us]', 'timestamp', True),
            'TARGET': column('float32', 'numerical'),
        },
    }
    expected_columns = {
        'users': {
            'instance_id': required_id,
            'user_id': column('int64', required=True),
            'age': column('int64', 'numerical'),
            'country': column('string', 'categorical'),
        },
        'orders': {
            'instance_id': required_id,
            '__node_id': required_id,
            'user_id': column('int64', 'ID'),
            'price': column('float64', 'numerical'),
            'ts': column('timestamp[us]', 'timestamp'),
        },
        'lines': {
            'instance_id': required_id,
            '__node_id': required_id,
            'order_id': column('int64', 'ID'),
            'qty': column('int64', 'categorical'),
        },
    }
    assert sorted(schema['related_tables']) == tables
    for table in tables:
        assert schema['related_tables'][table] == {
            'columns': expected_columns[table],
            'primary_key': [
                'instance_id',
                'user_id' if table == 'users' else '__node_id',
            ],
        }
    relationships = [
        {
            'source_columns': ['__kumo_entity_ref_0'],
            'target_table': 'users',
            'target_columns': ['user_id'],
        },
        {
            'source_table': 'orders',
            'source_columns': ['user_id'],
            'target_table': 'users',
            'target_columns': ['user_id'],
        },
    ]
    if chain:
        relationships.append(
            {
                'source_table': 'lines',
                'source_columns': ['order_id'],
                'target_table': 'orders',
                'target_columns': ['__node_id'],
            }
        )
    assert schema['relationships'] == relationships
    context = request['context']['instance_table']
    assert len(context['rows']) == 8 * 14
    for section in ['context', 'predict']:
        assert sorted(request[section]['related_tables']) == tables
    predict = request['predict']['instance_table']
    assert predict['format'] == 'arrays'
    assert predict['columns'] == [
        'instance_id',
        '__kumo_entity_ref_0',
        '__kumo_anchor_time',
    ]
    assert [[written(v) for v in row] for row in predict['rows']] == [
        [112 + i, entity, '2024-03-01T00:00:00.000000Z']
        for i, entity in enumerate(
            cases()[case_id]['input']['materialize']['indices']
        )
    ]
    counts = {'users': 2, 'orders': 8 if chain else 16}
    if chain:
        counts['lines'] = 2
    assert {
        t: len(rows['rows'])
        for t, rows in request['predict']['related_tables'].items()
    } == counts
    result = line['public']['result']
    assert result['rows'] == 2
    assert result['columns'] == [
        {'name': name, 'dtype': dtype}
        for name, dtype in [
            ('row_index', 'int64'),
            ('id', 'int64'),
            ('prediction', 'float64'),
        ]
    ]
    expected_rows = [
        [str(i), str(entity), '1.0']
        for i, entity in enumerate(
            cases()[case_id]['input']['materialize']['indices']
        )
    ]
    assert (
        line['public']['result_text']
        == hashlib.sha256(json.dumps(expected_rows).encode()).hexdigest()
    )
    for section in ['context', 'predict']:
        held = request[section]
        instance = held['instance_table']
        instance_rows = [
            dict(zip(instance['columns'], map(written, row), strict=True))
            for row in instance['rows']
        ]
        assert len({row['instance_id'] for row in instance_rows}) == len(
            instance_rows
        )
        related = {
            name: [
                dict(zip(table['columns'], map(written, row), strict=True))
                for row in table['rows']
            ]
            for name, table in held['related_tables'].items()
        }
        for name, rows in related.items():
            key = 'user_id' if name == 'users' else '__node_id'
            assert len({(row['instance_id'], row[key]) for row in rows}) == len(
                rows
            )
        anchors = {
            row['instance_id']: row['__kumo_anchor_time']
            for row in instance_rows
        }
        assert all(
            row['ts'] <= anchors[row['instance_id']]
            for row in related['orders']
        )
        users = {
            (row['instance_id'], row['user_id']) for row in related['users']
        }
        assert {
            (row['instance_id'], row['__kumo_entity_ref_0'])
            for row in instance_rows
        } == users
        assert all(
            (row['instance_id'], row['user_id']) in users
            for row in related['orders']
        )
        if chain:
            orders = {
                (row['instance_id'], row['__node_id'])
                for row in related['orders']
            }
            assert all(
                (row['instance_id'], row['order_id']) in orders
                for row in related['lines']
            )
    expected_pairs = {
        (
            entity,
            (
                datetime.datetime(2023, 1, 1)
                + datetime.timedelta(days=395 - 30 * step)
            ).isoformat(timespec='microseconds')
            + 'Z',
        )
        for entity in range(8)
        for step in range(14)
    }
    observed = examples(line)
    assert len(observed) == len(expected_pairs)
    assert {
        (entity, anchor) for entity, anchor, _ in observed
    } == expected_pairs


def problems(case_id: str, line: dict[str, Any]) -> list[str]:
    """Source membership and independently computed labels for this draw."""
    assert_structure(case_id, line)
    found = []
    given = source(case_id)
    entities = set(given['entities'])
    for entity, anchor, target in examples(line):
        if entity not in entities:
            found.append(f'a context example of no user: {entity}')
        elif target != label(case_id, entity, anchor):
            found.append(f'user {entity} at {anchor} labelled {target}')
    request = sent(line)
    for section in ('context', 'predict'):
        for table, sampled in request[section]['related_tables'].items():
            columns = FEATURES.get(table)
            if columns is None:
                found.append(f'{section} samples {table}, which it cannot')
                continue
            rows = given['source'][table]
            at = [sampled['columns'].index(c) for c in columns]
            for row in sampled['rows']:
                values = [written(row[i]) for i in at]
                if tuple(values) not in rows:
                    found.append(f'{section} {table} row {values} is not in it')
    return found[:8]


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_native_unseeded_draws_preserve_source_rows_and_labels(
    case_key: str,
) -> None:
    r"""Exit "``unseeded``", judged by properties: python-core
    executes on the native route, drawing the expected number of rows,
    each from the source."""
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test server not configured')

    def check() -> None:
        run = core_run(database)
        assert '' not in run, run.get('')
        line = run.get(case_key)
        assert line is not None, f'{case_key}: not observed'
        assert line['identity']['native_loaded'] and line['core'] is not None
        assert 'materialize' in line['identity']['operations']
        found = problems(case_key.split('@')[0], line)
        # A new sample each call: a second run draws other context
        # example ordering. This is a nondeterminism smoke check, not RNG proof.
        second = core_run(database, 1)
        assert '' not in second, second.get('')
        again = second[case_key]
        assert again['outcome']['result'] == 'success', again['outcome']
        found += problems(case_key.split('@')[0], again)
        if not found:
            if examples(line) == examples(again):
                found.append('two runs drew the same context')
        assert not found, f'{case_key}: ' + '; '.join(found)

    check()
