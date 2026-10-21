# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.11's exits on ``python-core`` (the package's test author's): the
sampling options.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t211_sampling.py -q

Written from the design's T2.11 row (each option an exit) and the
reference (python-ref at ``contract/reference-env.txt``'s commit), before
the implementation. ``sdm-testkit``'s ``tests/t211_sampling.rs`` holds
``rust-host`` to the same exits, and ``test_t211_unseeded.py`` judges the
unseeded sample by its properties (KO2-76).

The cases, ``contract/cases/s1/sample.*.json``, each a few dozen rows of
the shop base, run on DuckDB, SQLite and Postgres through ``predict``, each
option with values the reference takes and values it refuses:

- ``num_hops`` (1 over users, orders and lines, 3 over shipments too;
  0, 7, 1.5, True and
  '2' refused), ``num_neighbors`` ([4], [2, 2], [0], [-1], with num_hops; [], 5
  and seven hops refused), ``max_pq_iterations`` (1 over a sparse label;
  0 and 'x' refused), ``use_prediction_time`` (True, False),
  ``return_embeddings`` and ``response_field`` for embeddings (an endpoint
  answering embeddings, ``input.embeddings``), ``random_seed`` (7; -1
  refused), ``exclude_cols_dict`` (a feature of each table, None; a list,
  text, an unknown table, an unknown column and a key refused),
  ``context_anchor_time`` (before and after the anchor; before the data
  and 'entity' refused), ``anchor_time='entity'`` (``anchor_entity``: a
  static question, the context anchored likewise; a context timestamp, a
  windowed question and no time column refused);
- ``parent_table``: an entity table that links to a parent (orders, a
  windowed and a static question; RelBench's rel-stack votes per post and
  rel-avito clicks per ad, the questions of ``contract/tools/relbench.py``
  that reach it, in their shape and a few dozen rows), and over two
  DECIMAL(20,0) keys one integer apart that one float64 holds (2**53 and 2**53 + 1), each account
  reaching its own owner (KO2-35).

The core refuses every one of these options today, by name
(``UNSUPPORTED_REQUEST``: ``num_hops``, ``num_neighbors``,
``context_anchor_time``, ``anchor_entity``, ``parent_table`` and the
rest), where the reference answers or refuses it otherwise.

Each test's one line says the exit it proves. Gating: ``t211-pending.json``'s
``pending`` names, as ``py:<case key>``, the keys whose test failed when
written. Each must still fail (strict); once one passes it fails, naming
itself, until it is taken out. ``pending`` stays within ``authored``. Where
``SDM_POSTGRES_TEST_DSN`` names no server and ``SDM_REQUIRE_POSTGRES`` is
not set, the Postgres keys have nothing to hold and return without
asserting; with it set, a missing server fails (H26).
"""

from __future__ import annotations

import copy
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
PENDING = Path(__file__).with_name('t211-pending.json')
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
#: The cases whose answer the orchestrator decided apart from the
#: reference's, by decision row, with the reference's recorded outcome on
#: each database: num_neighbors=[-1] reaches no neighbour on DuckDB, every
#: neighbour on SQLite (its LIMIT -1) and fails INTERNAL_ERROR on
#: Postgres, so the core refuses a negative neighbour count by name,
#: INVALID_REQUEST naming num_neighbors, on all three (KO2-82; register
#: entry P2I-4, the implementer's). The case's expect stays the
#: reference's DuckDB outcome, as every case's does; the core is held to
#: the register's answer: by case, its result, stage and code.
DECIDED = {
    'sample.num-neighbors.minus-one': (
        'P2I-4',
        ('refusal', 'materialize', 'INVALID_REQUEST'),
    ),
}
DECIDED_REFERENCE = {
    'sample.num-neighbors.minus-one': ('success', None, None),
    'sample.num-neighbors.minus-one@sqlite': ('success', None, None),
    'sample.num-neighbors.minus-one@postgres': (
        'refusal',
        'StructuredDataError',
        'INTERNAL_ERROR',
    ),
}


def held(name: str, check: Any) -> None:
    r"""Runs ``check``: it must pass, or, while ``py:<name>`` is pending,
    fail."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    if f'py:{name}' not in gating['pending']:
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take py:{name} out of '
        'contract/tests/t211-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only keys the test author wrote, and
    ``authored`` is every case's key on each database on python-core, and
    on DuckDB and SQLite on rust-host (its Postgres pass replays
    python-core's fetches, which ``cross-host (postgres)`` holds), and the
    unseeded cases' ``unseeded:`` keys."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    authored = {k for k in gating['authored'] if not k.startswith('unseeded')}
    assert authored == {f'py:{k}' for k in KEYS_ALL} | {
        f'rust:{k}' for k in KEYS_ALL if not k.endswith('@postgres')
    }


# --- the cases -------------------------------------------------------------------


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""The T2.11 cases by id, each with its digest, as the runner loads
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
    by its cases, with at least one value the reference takes and, for
    every option the reference checks, one it refuses; ``anchor_entity``
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
            # The reference takes any value of these two.
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
    # The one decided case is diagnostic (DECIDED): its register entry's
    # refusal is checked here, not by the gate.
    assert case['status'] == (
        'diagnostic' if case_id in DECIDED else 'acceptance'
    )
    assert case['slice'] == 'S1'
    if case_id in DECIDED:
        assert case['register'] == DECIDED[case_id][0]
    assert case['until'] == 'predict'
    assert case['input']['base'] == 'shop'
    assert runner().case_databases(case) == DATABASES
    assert len(case['input']['materialize']['indices']) <= 3


# --- the reference ---------------------------------------------------------------


@functools.cache
def reference_lines() -> dict[str, dict[str, Any]]:
    r"""python-ref's committed lines, by key."""
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    commit = next(
        line.split(': ', 1)[1].strip()
        for line in env.splitlines()
        if line.startswith('reference: ')
    )
    path = ROOT / f'contract/reference/{commit}.jsonl'
    found = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        observed = json.loads(line)
        if observed['case'].split('@')[0] in CASE_IDS:
            found[observed['case']] = observed
    return found


def _outcome(line: dict[str, Any]) -> tuple[Any, ...]:
    outcome = line['outcome']
    return (
        outcome['result'],
        outcome['stage'],
        outcome.get('error'),
        outcome.get('code'),
    )


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_python_ref_s_line_is_at_its_case_file_and_ends_as_expected(
    case_key: str,
) -> None:
    r"""Exit "cases per option" on each database: python-ref's line for
    the case, recorded at the case file, ends as the case expects (a
    success through predict with one request, or the refusal at
    materialize with its code), so no case is vacuous."""
    case, digest = cases()[case_key.split('@')[0]]
    line = reference_lines().get(case_key)
    assert line is not None, f'{case_key}: no python-ref line'
    assert line['identity']['runner'] == 'python-ref'
    assert line['identity']['inputs']['case'] == digest
    expect = case['expect']
    if case_key in DECIDED_REFERENCE:
        # The reference's answer on each database, recorded as it is; the
        # case expects its DuckDB one, and the core is held to the
        # register's (DECIDED).
        result, error, code = DECIDED_REFERENCE[case_key]
        outcome = _outcome(line)
        assert (outcome[0], outcome[2], outcome[3]) == (result, error, code)
        if '@' in case_key:
            return
    assert _outcome(line) == (
        expect['outcome'],
        expect['stage'],
        expect.get('error'),
        expect.get('code'),
    )
    if expect['outcome'] == 'success':
        assert len(line['public']['payloads']) == 1


def _sent(case_key: str) -> dict[str, Any]:
    payload = reference_lines()[case_key]['public']['payloads'][0]['parsed']
    assert isinstance(payload, dict)
    return payload


def _number(value: Any) -> Any:
    return float(value['number']) if isinstance(value, dict) else value


def _rows(section: dict[str, Any], table: str) -> list[dict[str, Any]]:
    held = section['related_tables'].get(table)
    if held is None:
        return []
    return [
        dict(zip(held['columns'], row, strict=True)) for row in held['rows']
    ]


@pytest.mark.parametrize('database', DATABASES)
def test_the_reference_s_requests_show_each_option_s_effect(
    database: str,
) -> None:
    r"""The cases exercise what they name (no vacuous seed): in
    python-ref's request on each database, num_hops=1 leaves lines out and
    3 reaches shipments, three hops out; num_neighbors=[4] holds each
    scored user to 4 of its 10
    orders and [2, 2] each to 2; [0] leaves orders out; use_prediction_time
    is sent; exclude_cols_dict drops the column it names; a context anchor
    two months early labels no context example after it; an entity anchor
    is each user's signup time; and each account reaches its own owner
    over decimal keys one float64 holds."""
    on = '' if database == 'duckdb' else f'@{database}'
    if database == 'postgres' and f'sample.num-hops-1{on}' not in (
        reference_lines()
    ):
        pytest.fail('no python-ref line on Postgres')

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
    # (its LIMIT -1), a failure on Postgres (DECIDED_REFERENCE).
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
        _number(row['__node_id']): row['tier']
        for row in _rows(predict, 'owners')
    }
    reached = {
        row['account_id']['number']: owners[_number(row['owner_id'])]
        for row in _rows(predict, 'accounts')
        if row['owner_id'] is not None
    }
    assert reached == {'0': 'gold', '1': 'basic'}


@pytest.mark.parametrize('database', DATABASES)
def test_the_reference_answers_embeddings_where_asked(database: str) -> None:
    r"""Exit "``response_field`` for embeddings": asked for embeddings, and
    answered them, the reference's result carries an ``embeddings`` column;
    ``return_embeddings`` alone asks the endpoint for none."""
    on = '' if database == 'duckdb' else f'@{database}'
    if database == 'postgres' and f'sample.embeddings-only{on}' not in (
        reference_lines()
    ):
        pytest.fail('no python-ref line on Postgres')
    for case_id in ('sample.embeddings-answered', 'sample.embeddings-only'):
        line = reference_lines()[case_id + on]
        names = [c['name'] for c in line['public']['result']['columns']]
        assert 'embeddings' in names, case_id
        assert 'embeddings' in _sent(case_id + on)['output']['fields']
    line = reference_lines()['sample.return-embeddings' + on]
    assert 'embeddings' not in [
        c['name'] for c in line['public']['result']['columns']
    ]


# --- python-core against the reference ---------------------------------------------


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


def payloads(line: dict[str, Any]) -> list[Any]:
    r"""Each request the scripted endpoint received, parsed, as the
    ``payload`` rule compares it: its request id, drawn at random for every
    request, left out (``compare-policy.yaml``)."""
    out = []
    for payload in line['public']['payloads']:
        parsed = copy.deepcopy(payload.get('parsed'))
        if isinstance(parsed, dict) and isinstance(
            parsed.get('metadata'), dict
        ):
            parsed['metadata'].pop('request_id', None)
        out.append(parsed)
    return out


def differences(want: Any, got: Any, at: str = '') -> list[str]:
    r"""Where ``got`` differs from ``want``, a few paths."""
    if isinstance(want, dict) and isinstance(got, dict):
        found = []
        for name in sorted(set(want) | set(got)):
            found += differences(want.get(name), got.get(name), f'{at}/{name}')
        return found[:8]
    if (
        isinstance(want, list)
        and isinstance(got, list)
        and len(want) == len(got)
    ):
        found = []
        for i, (w, g) in enumerate(zip(want, got, strict=True)):
            found += differences(w, g, f'{at}[{i}]')
        return found[:8]
    if want != got:
        if isinstance(want, list) and isinstance(got, list):
            return [
                f'{at}: {len(got)} items where the reference has {len(want)}'
            ]
        return [f'{at}: {got!r:.80} where the reference has {want!r:.80}']
    return []


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_python_core_answers_each_case_as_the_reference_does(
    case_key: str,
) -> None:
    r"""Exits "cases per option", "``parent_table`` on two decimal keys one
    float64 apart" and "``response_field`` for embeddings": python-core
    runs the case on the database on the core route (the native module
    loaded, each of the case's core operations run), and its outcome (a
    refusal's class and code), each request the scripted endpoint received
    and the result (its columns and rows) are python-ref's."""
    database = case_key.partition('@')[2] or 'duckdb'
    if database == 'postgres' and not _server():
        return

    def check() -> None:
        run = core_run(database)
        assert '' not in run, run.get('')
        line = run.get(case_key)
        assert line is not None, f'{case_key}: not observed'
        case, _ = cases()[case_key.split('@')[0]]
        identity = line['identity']
        assert identity['native_loaded'] and line['core'] is not None
        # A decided case is refused at materialize (its register entry's
        # answer), so predict does not run.
        named = set(case['core']) - (
            {'predict'} if case['id'] in DECIDED else set()
        )
        missing = sorted(named - set(identity['operations']))
        assert not missing, f'{case_key}: core operations not run: {missing}'
        if case['id'] in DECIDED:
            # The register's answer, not the reference's: the named
            # refusal (its result, stage and code), no request sent.
            entry, wanted = DECIDED[case['id']]
            outcome = _outcome(line)
            assert (outcome[0], outcome[1], outcome[3]) == wanted, (
                f'{case_key}: {outcome} where {entry} has {wanted}'
            )
            assert line['public']['payloads'] == [], case_key
            return
        want = reference_lines()[case_key]
        assert _outcome(line) == _outcome(want), (
            f'{case_key}: {_outcome(line)} where python-ref has {_outcome(want)}'
        )
        found = differences(payloads(want), payloads(line))
        assert not found, f'{case_key}: the payload differs: ' + '; '.join(
            found
        )
        assert line['public']['result'] == want['public']['result'], (
            f'{case_key}: the result differs'
        )

    held(case_key, check)
