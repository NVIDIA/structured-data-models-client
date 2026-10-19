# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.9's exits on ``python-core`` (the package's test author's):
forecasting's options.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t29_forecasting.py -q

Written from the design's T2.9 row (``num_forecasts``, ``lag_timesteps``,
quantiles; ``response_field`` for quantiles; "forecast cases on rust-host
and python-core") and the reference, before the implementation.
``sdm-testkit``'s ``tests/t29_forecasting.rs`` holds ``rust-host`` to the
same cases. Every case is an S1 case over the ``shop`` base, run on DuckDB,
SQLite and Postgres, its expectation and python-ref's committed lines the
reference's at 6298913c. Each test's one line says what it proves:

- control: the cases are the row's, each on all three databases, with
  python-ref's line on each, whose outcome is the case's ``expect``;
- control: the lines are not vacuous: a forecast of N steps is N requests,
  each about its own window, and N rows per entity; a refused value sends
  nothing; ``lag_timesteps`` of n adds n lagged targets to every request
  (0 adds none, its requests ``forecast.steps-2``'s), and a static target
  none; each request asks for the outputs its case names, an answered
  quantile is a ``quantile_<q>`` column and an unanswered one is said;
- ``num_forecasts``: python-core answers each ``num_forecasts`` case as the
  reference does on each database, through the core (outcome, requests,
  result and warnings), the values the reference takes and the ones it
  refuses;
- ``lag_timesteps``: the same for each ``lag_timesteps`` case, the values
  the reference takes (lagged targets in every step's request) and the
  ones it refuses (``INVALID_REQUEST``, never the core's not-yet
  refusal);
- quantiles: the same for each case naming ``quantiles`` in ``outputs``:
  every request asks for what the case names, and an answer without them
  is said as the reference says it;
- ``response_field`` for quantiles: each case whose answer carries
  quantiles answers through ``predict`` with them laid out as the
  reference lays them out (``to_pandas()``'s columns and rows), each
  step's on its own rows.

Gating: ``t29-pending.json``'s ``pending`` names, as ``py:<test>[<database>]``,
the tests that failed when written. Each must still fail (strict); once one
passes it fails, naming itself, until it is taken out. ``pending`` stays
within ``authored``. Where ``SDM_POSTGRES_TEST_DSN`` names no server and
``SDM_REQUIRE_POSTGRES`` is not set, a Postgres leg has nothing to hold and
returns before it is judged, as T2.3's Postgres tests do; with it set, a
missing server fails (H26).
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'contract/runners/run_python.py'
PENDING = Path(__file__).with_name('t29-pending.json')
CASES = ROOT / 'contract/cases/s1'
DATABASES = ('duckdb', 'sqlite', 'postgres')

#: ``num_forecasts``: the values the reference takes (1, 2, 3, and over
#: months) and refuses (0, -1, 1.5, past the grammar's 10,000, of a static
#: column and of a condition).
NUM_FORECASTS = (
    'forecast.steps-1',
    'forecast.steps-2',
    'forecast.steps-3',
    'forecast.steps-zero',
    'forecast.steps-negative',
    'forecast.steps-fraction',
    'forecast.steps-past-grammar',
    'forecast.static-target',
    'forecast.binary-target',
    'forecast.count-months',
)
#: ``lag_timesteps``: on a two-step forecast, the values the reference
#: takes (0, 1, 2) and refuses (-1, 1.5, 2.0, true, "2"), and beside a
#: forecast (a regression, a binary classification, a static target).
LAG_TIMESTEPS = (
    'forecast.lag-0',
    'forecast.lag-1',
    'forecast.lag-2',
    'forecast.lag-negative',
    'forecast.lag-fraction',
    'forecast.lag-whole-float',
    'forecast.lag-bool',
    'forecast.lag-text',
    'lag.regression-2',
    'lag.binary-2',
    'lag.static-2',
    'lag.static-negative',
)
#: Quantiles asked for (``outputs``), answered or not.
QUANTILES = (
    'forecast.quantiles',
    'forecast.quantiles-and-prediction',
    'forecast.quantiles-unanswered',
    'quantiles.regression',
)
#: The cases whose answer carries quantiles (``input.quantiles``).
RESPONSE_FIELD = (
    'forecast.quantiles',
    'forecast.quantiles-and-prediction',
    'quantiles.regression',
)
FAMILY = (*NUM_FORECASTS, *LAG_TIMESTEPS, *QUANTILES)
#: The queries the reference refuses before anything is read.
REFUSED_QUERIES = (
    'forecast.steps-zero',
    'forecast.steps-negative',
    'forecast.steps-fraction',
    'forecast.steps-past-grammar',
    'forecast.static-target',
    'forecast.binary-target',
)
#: The lagged targets each lag case adds, by its value.
LAGS = {
    'forecast.lag-0': 0,
    'forecast.lag-1': 1,
    'forecast.lag-2': 2,
    'lag.regression-2': 2,
    'lag.binary-2': 2,
    'lag.static-2': 0,
}
#: How many steps each forecast the reference answers is asked as.
STEPS = {
    'forecast.steps-1': 1,
    'forecast.steps-2': 2,
    'forecast.steps-3': 3,
    'forecast.count-months': 2,
    'forecast.lag-0': 2,
    'forecast.lag-1': 2,
    'forecast.lag-2': 2,
    'forecast.quantiles': 2,
    'forecast.quantiles-and-prediction': 2,
    'forecast.quantiles-unanswered': 2,
}
QUANTILE_COLUMNS = ['quantile_0.1', 'quantile_0.5', 'quantile_0.9']


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def held(name: str, check: Callable[[], None]) -> None:
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
        'contract/tests/t29-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only tests the test author wrote, and
    ``authored`` names each exit test on each database."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    assert {name for name in gating['authored'] if name.startswith('py:')} == {
        f'py:{test}[{database}]' for test in EXITS for database in DATABASES
    }


def reachable(database: str) -> bool:
    r"""Whether ``database`` can be run here: DuckDB and SQLite always;
    Postgres where ``SDM_POSTGRES_TEST_DSN`` names a server, and a failure
    where ``SDM_REQUIRE_POSTGRES`` asks for one that is not named (H26)."""
    if database != 'postgres' or os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


# --- the cases and the reference's lines ---------------------------------------


def _runner() -> ModuleType:
    return _module('t29_run_python', RUNNER)


def _cases() -> dict[str, dict[str, Any]]:
    return {
        case['id']: case
        for case, _ in _runner().load_cases(
            [CASES / f'{case_id}.json' for case_id in FAMILY]
        )
    }


def reference_lines() -> dict[str, dict[str, Any]]:
    r"""python-ref's committed lines, by key."""
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    commit = next(
        line.split(': ', 1)[1].strip()
        for line in env.splitlines()
        if line.startswith('reference: ')
    )
    path = ROOT / f'contract/reference/{commit}.jsonl'
    return {
        json.loads(line)['case']: json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
    }


def requests(line: dict[str, Any]) -> list[Any]:
    r"""The requests a line's endpoint received, in order, as the
    ``payload`` rule compares them: each parsed body but its request id."""
    out = []
    for payload in line['public']['payloads']:
        body = json.loads(json.dumps(payload['parsed']))
        if isinstance(body, dict):
            (body.get('metadata') or {}).pop('request_id', None)
        out.append((payload['method'], payload['path'], body))
    return out


def _outcome(line: dict[str, Any]) -> dict[str, Any]:
    outcome = line['outcome']
    return {
        key: outcome.get(key) for key in ('result', 'stage', 'error', 'code')
    }


def _columns(line: dict[str, Any]) -> list[str]:
    result = line['public']['result'] or {}
    return [column['name'] for column in result.get('columns', [])]


def _lagged(body: dict[str, Any]) -> tuple[list[str], list[str]]:
    r"""The lagged targets a request's context and rows to score carry."""
    return tuple(  # type: ignore[return-value]
        [
            column
            for column in body[half]['instance_table']['columns']
            if column.startswith('__kumo_arl')
        ]
        for half in ('context', 'predict')
    )


def test_the_cases_are_the_design_row_s() -> None:
    r"""Control: the T2.9 cases are each of the row's options over the shop
    base, through the public calls, each on all three databases, with
    python-ref's committed line on each, whose outcome is the case's
    ``expect``; a case names its option as written (``lag_timesteps`` in
    materialize's arguments, ``outputs`` for predict, and the endpoint told
    to answer quantiles where the answer carries them)."""
    cases = _cases()
    runner = _runner()
    family = {
        path.stem
        for prefix in ('forecast.', 'lag.', 'quantiles.')
        for path in CASES.glob(f'{prefix}*.json')
    }
    # T2.8's forecast cases (top_k and MOST_PARTS) are ranking's.
    assert family - {'forecast.top-k', 'forecast.most-parts'} == set(FAMILY)
    reference = reference_lines()
    for case_id, case in cases.items():
        given = case['input']
        assert case['status'] == 'acceptance', case_id
        assert given['base'] == 'shop', case_id
        assert runner.case_databases(case) == DATABASES, case_id
        assert given['materialize']['indices'] == [0, 1], case_id
        assert ('lag_timesteps' in given['materialize']) == (
            case_id in LAG_TIMESTEPS
        ), case_id
        assert ('outputs' in given) == (case_id in QUANTILES), case_id
        if case_id in QUANTILES:
            assert 'quantiles' in given['outputs'], case_id
        assert given.get('quantiles', False) == (case_id in RESPONSE_FIELD)
        assert ('FORECAST' in given['pql']) == (
            case_id.startswith('forecast.')
        ), case_id
        # A query the reference refuses before reading anything is held on
        # the core's validate_query, as refuse.unknown-column is; any other
        # case carries its call's operations.
        if case_id in REFUSED_QUERIES:
            wanted = ['describe_source', 'validate_query']
        else:
            wanted = ['describe_source', 'materialize']
            if case['until'] == 'predict':
                wanted.append('predict')
        assert list(case['core']) == wanted, case_id
        for database in DATABASES:
            line = reference[runner.case_key(case_id, database)]
            expect = case['expect']
            assert _outcome(line) == {
                'result': expect['outcome'],
                'stage': expect['stage'],
                'error': expect.get('error'),
                'code': expect.get('code'),
            }, (case_id, database)


def test_the_reference_lines_are_not_vacuous() -> None:
    r"""Control: on each database, python-ref's lines show each case doing
    what it names. A forecast of N steps sends N requests, no two alike,
    and answers N rows per entity with a ``forecast_step``; a refusal sends
    nothing; ``lag_timesteps`` of n puts n lagged targets in the context
    and the rows of every request (0 none, its requests those of the same
    forecast without the option; a static target none); each request asks
    for the outputs its case names; an answered quantile is a
    ``quantile_<q>`` column, and quantiles asked for and not answered are
    said, once a step and once for the whole. Each answered quantile is
    the one the scripted endpoint's rule gave the row's step's call
    (step k answered by call k - 1, row i by the rule's row i): the rows
    built from the rule have the digest of the reference's result, which
    is what ``t29_forecasting.rs`` holds rust-host's result to."""
    runner = _runner()
    cases = _cases()
    reference = reference_lines()
    for database in DATABASES:

        def line(case_id: str) -> dict[str, Any]:
            return reference[runner.case_key(case_id, database)]  # noqa: B023

        for case_id, case in cases.items():
            sent = requests(line(case_id))
            if case['expect']['outcome'] == 'refusal':
                assert sent == [], (case_id, database)
                continue
            columns = _columns(line(case_id))
            if case_id in STEPS:
                assert len(sent) == STEPS[case_id], (case_id, database)
                assert len({json.dumps(r, sort_keys=True) for r in sent}) == (
                    len(sent)
                ), (case_id, database)
                assert columns[:3] == ['row_index', 'id', 'forecast_step']
                assert line(case_id)['public']['result']['rows'] == 2 * len(
                    sent
                ), (case_id, database)
            else:
                assert len(sent) == 1, (case_id, database)
                assert 'forecast_step' not in columns, (case_id, database)
            for _, _, body in sent:
                wanted = case['input'].get('outputs')
                if wanted is not None:
                    assert body['output']['fields'] == wanted, case_id
                if case_id in LAGS:
                    n = LAGS[case_id]
                    names = [f'__kumo_arl{i}__' for i in range(n)]
                    assert _lagged(body) == (names, names), (case_id, database)
                else:
                    assert _lagged(body) == ([], []), (case_id, database)
            if case_id in RESPONSE_FIELD:
                assert columns[-3:] == QUANTILE_COLUMNS, (case_id, database)
            else:
                assert not set(columns) & set(QUANTILE_COLUMNS), case_id
        assert requests(line('forecast.lag-0')) == requests(
            line('forecast.steps-2')
        ), database
        assert requests(line('forecast.lag-1')) != requests(
            line('forecast.lag-2')
        ), database
        said = line('forecast.quantiles-unanswered')['host']['warnings']
        assert said.count('SDMWarning') == 3, database
        for case_id in RESPONSE_FIELD:
            assert 'SDMWarning' not in line(case_id)['host']['warnings']
            result = line(case_id)['public']['result']
            assert result['digest'] == _the_rule_s_rows(case_id), (
                case_id,
                database,
            )


def _the_rule_s_rows(case_id: str) -> str:
    r"""The digest, as the runner takes a result's, of the rows
    ``to_pandas()`` gives for ``case_id`` where each row holds the
    prediction and quantiles the scripted endpoint's rule gives it: the
    entities 0 and 1 in order, each step after the last, step k answered by
    call k - 1."""
    endpoint = _module(
        't29_scripted_endpoint', ROOT / 'contract/runners/scripted_endpoint.py'
    )
    steps = STEPS.get(case_id)
    rows = []
    for index in (0, 1):
        for call in range(steps or 1):
            spread = endpoint._spread(call, index)
            rows.append(
                [
                    index,
                    index,
                    *([call + 1] if steps else []),
                    1.0,
                    *(spread[q] for q in ('0.1', '0.5', '0.9')),
                ]
            )
    return _runner()._rows_digest(rows)


# --- python-core over the cases ----------------------------------------------------


@functools.cache
def _cases_on(database: str) -> dict[str, dict[str, Any]]:
    r"""python-core over the T2.9 cases on ``database``, as that database's
    job runs it: each observation by its key."""
    paths = [CASES / f'{case_id}.json' for case_id in FAMILY]
    with tempfile.TemporaryDirectory(prefix='t29-cases-') as tmp:
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
                *map(str, paths),
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            check=False,
        )
        assert done.returncode == 0, done.stdout + done.stderr
        return {
            json.loads(line)['case']: json.loads(line)
            for line in out.read_text(encoding='utf-8').splitlines()
        }


def as_the_reference(
    database: str, case_ids: tuple[str, ...], result: bool = True
) -> None:
    r"""Each case of ``case_ids`` on ``database``: python-core's outcome
    (result, stage, class and code), every request it sent (the
    ``payload`` rule: each but its request id, in order), its warnings'
    classes and, where ``result``, its result through ``to_pandas()``
    (columns, dtypes and rows) are python-ref's; and it ran on the core,
    every core operation its case names started by the core."""
    runner = _runner()
    cases = _cases()
    observed = _cases_on(database)
    reference = reference_lines()
    problems = []
    for case_id in case_ids:
        key = runner.case_key(case_id, database)
        mine, theirs = observed[key], reference[key]
        if _outcome(mine) != _outcome(theirs):
            problems.append(
                f'{key}: {_outcome(mine)} where python-ref is '
                f'{_outcome(theirs)}'
            )
            continue
        identity = mine['identity']
        if not identity['native_loaded'] or not set(
            cases[case_id]['core']
        ) <= set(identity['operations']):
            problems.append(f'{key}: not on the core route')
        if requests(mine) != requests(theirs):
            problems.append(
                f'{key}: {len(requests(mine))} requests, not the '
                f'{len(requests(theirs))} python-ref sent'
                if len(requests(mine)) != len(requests(theirs))
                else f'{key}: a request is not the one python-ref sent'
            )
        if mine['host']['warnings'] != theirs['host']['warnings']:
            problems.append(
                f'{key}: warnings {mine["host"]["warnings"]} where python-ref '
                f'gives {theirs["host"]["warnings"]}'
            )
        if result and mine['public']['result'] != theirs['public']['result']:
            problems.append(
                f'{key}: the result is not the one python-ref gives'
            )
    assert not problems, '\n'.join(problems)


@pytest.mark.parametrize('database', DATABASES)
def test_num_forecasts_as_the_reference(database: str) -> None:
    r""" "``num_forecasts``": python-core answers each ``num_forecasts`` case
    on ``database`` as the reference does, through the core: 1, 2 and 3
    steps and a COUNT over months (one request per step, each step's rows),
    and 0, -1, 1.5, 10,001, a static target and a condition refused."""
    if not reachable(database):
        return
    held(
        f'test_num_forecasts_as_the_reference[{database}]',
        lambda: as_the_reference(database, NUM_FORECASTS),
    )


@pytest.mark.parametrize('database', DATABASES)
def test_lag_timesteps_as_the_reference(database: str) -> None:
    r""" "``lag_timesteps``": python-core answers each ``lag_timesteps``
    case on ``database`` as the reference does, through the core: 0, 1 and
    2 on a forecast (every step's request carrying its lagged targets), 2 on
    a temporal regression, a binary classification and a static target, and
    -1, 1.5, 2.0, true and "2" refused ``INVALID_REQUEST``."""
    if not reachable(database):
        return
    held(
        f'test_lag_timesteps_as_the_reference[{database}]',
        lambda: as_the_reference(database, LAG_TIMESTEPS),
    )


@pytest.mark.parametrize('database', DATABASES)
def test_quantiles_asked_as_the_reference(database: str) -> None:
    r""" "Quantiles": python-core asks for quantiles as the reference does
    on ``database``: each request of each case naming them in ``outputs``
    asks for the fields the case names (alone or beside the prediction, on
    a forecast's every step and on a regression), and an answer without
    them is given back with each absence said, as python-ref says it."""
    if not reachable(database):
        return
    held(
        f'test_quantiles_asked_as_the_reference[{database}]',
        lambda: as_the_reference(database, QUANTILES, result=False),
    )


@pytest.mark.parametrize('database', DATABASES)
def test_response_field_for_quantiles_as_the_reference(database: str) -> None:
    r""" "``response_field`` for quantiles": each case whose answer carries
    quantiles answers through ``predict`` on ``database`` with them laid
    out as the reference lays them out: ``to_pandas()``'s columns (one
    ``quantile_<q>`` each), dtypes and rows, each step's quantiles on its
    own rows, never refused ``response_field``."""
    if not reachable(database):
        return
    held(
        f'test_response_field_for_quantiles_as_the_reference[{database}]',
        lambda: as_the_reference(database, RESPONSE_FIELD),
    )


#: The exit tests, each run on each database.
EXITS = (
    'test_num_forecasts_as_the_reference',
    'test_lag_timesteps_as_the_reference',
    'test_quantiles_asked_as_the_reference',
    'test_response_field_for_quantiles_as_the_reference',
)
