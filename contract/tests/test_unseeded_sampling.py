# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unseeded native sampling: fresh draws, valid source rows and labels."""

from __future__ import annotations

import datetime
import functools
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
VECTORS = ROOT / 'contract/unseeded-vectors.json'
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


def label(case: dict[str, Any], entity: Any, anchor: str) -> float:
    r"""The label of ``entity`` at ``anchor`` read from the source: the
    query's SUM(orders.price, 0, 30, days), the orders after the anchor up
    to 30 days later."""
    import duckdb

    assert case['input']['pql'].startswith('PREDICT SUM(orders.price, 0, 30')
    connection = duckdb.connect()
    try:
        for statement in [
            *runner().load_bases()[case['input']['base']],
            *case['input'].get('setup', []),
        ]:
            connection.execute(statement)
        found = connection.execute(
            'SELECT COALESCE(SUM(price), 0) FROM orders WHERE user_id = ? '
            'AND ts > CAST(? AS TIMESTAMP) '
            "AND ts <= CAST(? AS TIMESTAMP) + INTERVAL '30 days'",
            [int(entity), anchor[:26], anchor[:26]],
        ).fetchone()[0]
    finally:
        connection.close()
    return float(found)


def sent(line: dict[str, Any]) -> dict[str, Any]:
    r"""The one request a line's endpoint received, parsed."""
    payloads = line['public']['payloads']
    assert len(payloads) == 1, len(payloads)
    return payloads[0]['parsed']


def properties(line: dict[str, Any]) -> dict[str, Any]:
    r"""What an unseeded line holds that does not change from draw to
    draw."""
    outcome = line['outcome']
    found: dict[str, Any] = {
        'outcome': {
            'result': outcome['result'],
            'stage': outcome['stage'],
            'error': outcome.get('error'),
            'code': outcome.get('code'),
        }
    }
    if outcome['result'] != 'success':
        return found
    request = sent(line)
    predict = request['predict']
    found.update(
        {
            'schema': request['schema'],
            'predict': {
                'instance_table': predict['instance_table'],
                'rows': {
                    table: len(held['rows'])
                    for table, held in sorted(predict['related_tables'].items())
                },
            },
            'context_rows': len(request['context']['instance_table']['rows']),
            'context_tables': sorted(request['context']['related_tables']),
            'result': line['public']['result'],
        }
    )
    return found


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
def vectors() -> dict[str, Any]:
    return json.loads(VECTORS.read_text(encoding='utf-8'))


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
    for database in DATABASES:
        line = vectors()['keys'][runner().case_key(case_id, database)]
        assert line['outcome']['result'] == 'success'
        assert line['draws_differ'] is True
        assert line['context_rows'] > 0
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


def problems(
    case_id: str, line: dict[str, Any], want: dict[str, Any]
) -> list[str]:
    r"""Each property of an unseeded draw ``line`` breaks."""
    found = []
    got = properties(line)
    if got['outcome'] != want['outcome']:
        return [f'{got["outcome"]} where the fixture expects {want["outcome"]}']
    for name in (
        'schema',
        'predict',
        'context_rows',
        'context_tables',
        'result',
    ):
        if got.get(name) != want.get(name):
            found.append(f'its {name} differs from the fixed sampling contract')
    case = cases()[case_id]
    given = vectors()['cases'][case_id]
    entities = set(given['entities'])
    for entity, anchor, target in examples(line):
        if entity not in entities:
            found.append(f'a context example of no user: {entity}')
        elif target != label(case, entity, anchor):
            found.append(f'user {entity} at {anchor} labelled {target}')
    request = sent(line)
    for section in ('context', 'predict'):
        for table, sampled in request[section]['related_tables'].items():
            columns = FEATURES.get(table)
            if columns is None:
                found.append(f'{section} samples {table}, which it cannot')
                continue
            rows = {json.dumps(r) for r in given['source'][table]}
            at = [sampled['columns'].index(c) for c in columns]
            for row in sampled['rows']:
                values = [written(row[i]) for i in at]
                if json.dumps(values) not in rows:
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
        found = problems(
            case_key.split('@')[0], line, vectors()['keys'][case_key]
        )
        # A new sample each call: a second run draws other context
        # examples; the source has many more candidates than the sample.
        second = core_run(database, 1)
        assert '' not in second, second.get('')
        again = second[case_key]
        assert again['outcome']['result'] == 'success', again['outcome']
        if not found:
            if examples(line) == examples(again):
                found.append('two runs drew the same context')
        assert not found, f'{case_key}: ' + '; '.join(found)

    check()
