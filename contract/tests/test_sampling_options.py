# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Current native sampling options, endpoint payloads and result fields."""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
CASES = ROOT / 'contract/cases/s1'
DATABASES = ('duckdb', 'sqlite', 'postgres')
CASE_IDS = tuple(sorted(path.stem for path in CASES.glob('sample.*.json')))


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t211_run_python', RUNNER)


def key(case_id: str, database: str) -> str:
    return runner().case_key(case_id, database)


KEYS_ALL = tuple(key(c, d) for d in DATABASES for c in CASE_IDS)
# --- the cases -------------------------------------------------------------------


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""Sampling cases by id, each with its digest, as the runner loads
    them."""
    loaded = runner().load_cases([CASES / f'{c}.json' for c in CASE_IDS])
    return {case['id']: (case, digest) for case, digest in loaded}


#: Each option the design's row names, with the cases that give it.
OPTIONS = {
    'num_hops': 'sample.num-hops',
    'num_neighbors': 'sample.num-neighbors',
    'max_pq_iterations': 'sample.max-pq-iterations',
    'use_prediction_time': 'sample.use-prediction-time',
    'return_embeddings': 'sample.return-embeddings',
    'random_seed': 'sample.seed',
    'exclude_cols_dict': 'sample.exclude-cols',
    'context_anchor_time': 'sample.context-anchor-time',
}


def test_every_option_has_a_value_taken_and_a_value_refused() -> None:
    r"""Exit "cases per option": each option of the design's row is given
    by its cases, with both valid and invalid input examples; ``anchor_entity``
    and ``parent_table`` likewise."""
    by_option: dict[str, set[str]] = {}
    for case, _ in cases().values():
        call = case['input']['materialize']
        for option in OPTIONS:
            if option in call:
                by_option.setdefault(option, set()).add(
                    case['expect']['outcome']
                )
        if call.get('anchor_time') == 'entity':
            by_option.setdefault('anchor_entity', set()).add(
                case['expect']['outcome']
            )
    assert set(by_option) == {*OPTIONS, 'anchor_entity'}
    for option, outcomes in by_option.items():
        if option in {'use_prediction_time', 'return_embeddings'}:
            # These cases exercise accepted boolean options.
            assert 'success' in outcomes, option
        else:
            assert outcomes == {'success', 'refusal'}, option
    parents = [c for c in CASE_IDS if c.startswith('sample.parent-table')]
    assert len(parents) == 5
    # The RelBench questions that reach parent_table (contract/tools/
    # relbench.py), in their shape: the entity table links to a parent.
    for case_id in (
        'sample.parent-table.relbench-stack',
        'sample.parent-table.relbench-avito',
    ):
        case, _ = cases()[case_id]
        entity = case['input']['pql'].rsplit(' FOR EACH ', 1)[1].split('.')[0]
        links = case['input']['from_source']['relationships']
        assert any(
            link['source_table'] == entity and link['target_table'] != entity
            for link in links
        ), case_id


@pytest.mark.parametrize('case_id', CASE_IDS)
def test_each_case_is_small_and_runs_on_all_three_databases(
    case_id: str,
) -> None:
    r"""Exit "cases per option", written for all three databases: each is
    an S1 acceptance case through predict on the shop base, on DuckDB,
    SQLite and Postgres, asking about a few entities."""
    case, _ = cases()[case_id]
    assert case['until'] == 'predict'
    assert case['input']['base'] == 'shop'
    assert runner().case_databases(case) == DATABASES
    assert len(case['input']['materialize']['indices']) <= 3


def _sent(case_key: str) -> dict[str, Any]:
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test server not configured')
    run = core_run(database)
    assert '' not in run, run.get('')
    line = run[case_key]
    assert line['outcome']['result'] == 'success', line['outcome']
    assert line['outcome']['stage'] == 'predict'
    assert len(line['public']['payloads']) == 1
    payload = line['public']['payloads'][0]['parsed']
    assert isinstance(payload, dict)
    return payload


def _number(value: Any) -> Any:
    return Decimal(value['number']) if isinstance(value, dict) else value


def _rows(section: dict[str, Any], table: str) -> list[dict[str, Any]]:
    held = section['related_tables'].get(table)
    if held is None:
        return []
    return [
        dict(zip(held['columns'], row, strict=True)) for row in held['rows']
    ]


@pytest.mark.parametrize('database', DATABASES)
def test_native_requests_show_each_option_effect(
    database: str,
) -> None:
    r"""The cases exercise what they name (no vacuous seed): in
    the native request on each database, num_hops=1 leaves lines out and
    3 reaches shipments, three hops out; num_neighbors=[4] holds each
    scored user to 4 of its 10
    orders and [2, 2] each to 2; [0] leaves orders out; use_prediction_time
    is sent; exclude_cols_dict drops the column it names; a context anchor
    two months early labels no context example after it; an entity anchor
    is each user's signup time; and each account reaches its own owner
    over decimal keys one float64 holds."""
    on = '' if database == 'duckdb' else f'@{database}'
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test server not configured')

    def sent(case_id: str) -> dict[str, Any]:
        return _sent(case_id + on)

    assert 'lines' not in sent('sample.num-hops-1')['predict']['related_tables']
    assert 'shipments' in sent('sample.num-hops-3')['predict']['related_tables']
    for case_id, most in (
        ('sample.num-neighbors-4', 4),
        ('sample.num-neighbors-2-2', 2),
    ):
        orders = _rows(sent(case_id)['predict'], 'orders')
        per_user: dict[Any, int] = {}
        for row in orders:
            per_user[row['instance_id']['number']] = (
                per_user.get(row['instance_id']['number'], 0) + 1
            )
        assert per_user and max(per_user.values()) == most, case_id
    assert (
        'orders'
        not in sent('sample.num-neighbors.zero')['predict']['related_tables']
    )
    # num_neighbors=[-1]: no neighbour on DuckDB, every neighbour on SQLite
    # (its LIMIT -1), a provider failure on Postgres.
    minus_one = 'sample.num-neighbors.minus-one'
    if database != 'postgres':
        reached = 'orders' in sent(minus_one)['predict']['related_tables']
        assert reached == (database == 'sqlite')
    assert sent('sample.use-prediction-time')['inference'][
        'use_prediction_time'
    ]
    assert not sent('sample.use-prediction-time.false')['inference'][
        'use_prediction_time'
    ]
    for case_id, table, column in (
        ('sample.exclude-cols.users-age', 'users', 'age'),
        ('sample.exclude-cols.orders-price', 'orders', 'price'),
    ):
        payload = sent(case_id)
        assert (
            column not in payload['predict']['related_tables'][table]['columns']
        )
        assert (
            column
            in sent('sample.exclude-cols.null')['predict']['related_tables'][
                table
            ]['columns']
        )
    context = sent('sample.context-anchor-time')['context']['instance_table']
    at = context['columns'].index('__kumo_anchor_time')
    assert max(row[at] for row in context['rows']) <= (
        '2024-01-01T00:00:00.000000Z'
    )
    for case_id in ('sample.anchor-entity', 'sample.anchor-entity.regression'):
        predict = sent(case_id)['predict']
        table = predict['instance_table']
        at = table['columns'].index('__kumo_anchor_time')
        users = {
            row['user_id']['number']: row['signup']
            for row in _rows(predict, 'users')
        }
        ref = table['columns'].index('__kumo_entity_ref_0')
        assert [row[at] for row in table['rows']] == [
            users[row[ref]['number']] for row in table['rows']
        ], case_id
    # An entity anchor where user 0 has no signup time: its anchor is sent
    # null and none of its orders pulled; user 1 is anchored at its signup.
    for case_id in (
        'sample.anchor-entity.null-time',
        'sample.anchor-entity.null-time.prediction-time',
    ):
        predict = sent(case_id)['predict']
        table = predict['instance_table']
        at = table['columns'].index('__kumo_anchor_time')
        ref = table['columns'].index('__kumo_entity_ref_0')
        anchors = {row[ref]['number']: row[at] for row in table['rows']}
        assert anchors['0'] is None and anchors['1'] is not None, case_id
        instance = {
            row[ref]['number']: row[0]['number'] for row in table['rows']
        }
        pulled = {
            row['instance_id']['number'] for row in _rows(predict, 'orders')
        }
        assert instance['0'] not in pulled and instance['1'] in pulled, case_id
    assert sent('sample.anchor-entity.null-time.prediction-time')['inference'][
        'use_prediction_time'
    ]
    predict = sent('sample.parent-table.decimal-keys')['predict']
    owners = {
        (_number(row['instance_id']), _number(row['__node_id'])): row['tier']
        for row in _rows(predict, 'owners')
    }
    witnessed = set()
    # Check every instance separately; another occurrence must not hide a bad link.
    for row in _rows(predict, 'accounts'):
        if row['owner_id'] is None:
            continue
        account = row['account_id']['number']
        tier = owners[(_number(row['instance_id']), _number(row['owner_id']))]
        assert tier == ('gold' if int(account) % 2 == 0 else 'basic'), row
        witnessed.add(account)
    assert {'0', '1'} <= witnessed


@pytest.mark.parametrize('database', DATABASES)
def test_native_results_include_requested_embeddings(database: str) -> None:
    r"""Exit "``response_field`` for embeddings": asked for embeddings, and
    answered them, the result carries an ``embeddings`` column;
    ``return_embeddings`` alone asks the endpoint for none."""
    on = '' if database == 'duckdb' else f'@{database}'
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test server not configured')
    for case_id in ('sample.embeddings-answered', 'sample.embeddings-only'):
        _sent(case_id + on)
        line = core_run(database)[case_id + on]
        names = [c['name'] for c in line['public']['result']['columns']]
        assert 'embeddings' in names, case_id
        assert 'embeddings' in _sent(case_id + on)['output']['fields']
    _sent('sample.return-embeddings' + on)
    line = core_run(database)['sample.return-embeddings' + on]
    assert 'embeddings' not in [
        c['name'] for c in line['public']['result']['columns']
    ]


# --- current installed client -----------------------------------------------------


def _server() -> bool:
    r"""Whether a Postgres server is named; where none is, one must not be
    required (H26)."""
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


@functools.cache
def core_run(database: str) -> dict[str, Any]:
    r"""python-core over the T2.11 cases on ``database``, as that database's
    job runs them: each observation by its key, or under ``''`` why the run
    failed whole."""
    with tempfile.TemporaryDirectory(prefix='t211-') as tmp:
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
def test_sampling_option_outcomes_and_native_operations(case_key: str) -> None:
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        pytest.skip('Postgres test server not configured')
    run = core_run(database)
    assert '' not in run, run.get('')
    line = run[case_key]
    expected = json.loads(
        (ROOT / f'contract/expected/{database}-python.json').read_text()
    )['python-core'][case_key]
    assert line['outcome'] == expected
    assert line['identity']['native_loaded'] and line['core'] is not None
    case, _ = cases()[case_key.split('@')[0]]
    operations = set(case['core'])
    if (
        case['id'] == 'sample.num-neighbors.minus-one'
        and database == 'postgres'
    ):
        operations.discard('predict')
    assert operations <= set(line['identity']['operations'])
    if expected['result'] == 'success':
        _sent(case_key)
        assert line['public']['result']['rows']
