# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forecast options preserve step counts, quantiles, lag values and row alignment.

Current-client checks use retained expected values and scripted endpoint responses.
No archived client runtime is executed.
"""

from __future__ import annotations

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
#: forecast (a regression, a binary classification, a static target, and
#: negated condition over a filtered count).
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
    'lag.not-filtered-2',
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
#: A lagged average whose windows are sometimes empty: the reference's lags
#: misaligned, the core's each on its own row.
LAG_ALIGNED = 'lag.average-empty-windows'
#: A lagged condition asked one entity a call: each call's lag columns
#: declared bool, as the reference declares them.
LAG_DTYPE = 'lag.condition-per-call'
FAMILY = (*NUM_FORECASTS, *LAG_TIMESTEPS, *QUANTILES, LAG_ALIGNED, LAG_DTYPE)
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
    'lag.not-filtered-2': 2,
    LAG_ALIGNED: 2,
    LAG_DTYPE: 2,
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


def reachable(database: str) -> bool:
    r"""Whether ``database`` can be run here: DuckDB and SQLite always;
    Postgres where ``SDM_POSTGRES_TEST_DSN`` names a server, and a failure
    where ``SDM_REQUIRE_POSTGRES`` asks for one that is not named."""
    if database != 'postgres' or os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


# --- the cases and the reference's lines ---------------------------------------


def _runner() -> ModuleType:
    return _module('test_forecast_options_run_python', RUNNER)


def _cases() -> dict[str, dict[str, Any]]:
    return {
        case['id']: case
        for case, _ in _runner().load_cases(
            [CASES / f'{case_id}.json' for case_id in FAMILY]
        )
    }


def observed_cases() -> dict[str, dict[str, Any]]:
    """Current-client observations on configured local connectors."""
    return {
        key: value
        for database in DATABASES
        if reachable(database)
        for key, value in _cases_on(database).items()
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
    """Validate requested behavior using current-client observations and explicit expectations."""
    cases = _cases()
    runner = _runner()
    family = {
        path.stem
        for prefix in ('forecast.', 'lag.', 'quantiles.')
        for path in CASES.glob(f'{prefix}*.json')
    }
    # forecast cases (top_k and MOST_PARTS) are ranking's.
    assert family - {'forecast.top-k', 'forecast.most-parts'} == set(FAMILY)
    for case_id, case in cases.items():
        given = case['input']
        assert case['status'] == 'acceptance', case_id
        assert given['base'] == 'shop', case_id
        assert runner.case_databases(case) == DATABASES, case_id
        assert given['materialize']['indices'] == [0, 1], case_id
        assert ('lag_timesteps' in given['materialize']) == (
            case_id in (*LAG_TIMESTEPS, LAG_ALIGNED, LAG_DTYPE)
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


def test_requests_and_results_follow_forecast_options() -> None:
    """Validate requested behavior using current-client observations and explicit expectations."""
    runner = _runner()
    cases = _cases()
    observed = observed_cases()
    for database in (d for d in DATABASES if reachable(d)):

        def line(case_id: str) -> dict[str, Any]:
            return observed[runner.case_key(case_id, database)]  # noqa: B023

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
                # One call, but lag.condition-per-call's two (rows_per_call=1).
                calls = 2 if case_id == LAG_DTYPE else 1
                assert len(sent) == calls, (case_id, database)
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
    ``to_pandas`` gives for ``case_id`` where each row holds the
    prediction and quantiles the scripted endpoint's rule gives it: the
    entities 0 and 1 in order, each step after the last, step k answered by
    call k - 1."""
    endpoint = _module(
        'forecast_scripted_endpoint',
        ROOT / 'tests/support/scripted_endpoint.py',
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
    r"""python-core over the cases on ``database``, as that database's
    job runs it: each observation by its key."""
    paths = [CASES / f'{case_id}.json' for case_id in FAMILY]
    with tempfile.TemporaryDirectory(prefix='forecast_options-') as tmp:
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


def assert_case_outcomes(
    database: str, case_ids: tuple[str, ...], result: bool = True
) -> None:
    cases = _cases()
    observed = _cases_on(database)
    for case_id in case_ids:
        line = observed[_runner().case_key(case_id, database)]
        expected = cases[case_id]['expect']
        assert _outcome(line) == {
            'result': expected['outcome'],
            'stage': expected['stage'],
            'error': expected.get('error'),
            'code': expected.get('code'),
        }, case_id
        assert line['identity']['native_loaded']
        assert set(cases[case_id]['core']) <= set(
            line['identity']['operations']
        ), case_id
        if (
            result
            and expected['outcome'] == 'success'
            and cases[case_id]['until'] == 'predict'
        ):
            assert line['public']['result'] is not None, case_id


@pytest.mark.parametrize('database', DATABASES)
def test_num_forecasts_as_the_reference(database: str) -> None:
    """Validate requested behavior using current-client observations and explicit expectations."""
    if not reachable(database):
        pytest.skip('Postgres test DSN is not configured')
    (lambda: assert_case_outcomes(database, NUM_FORECASTS))()


@pytest.mark.parametrize('database', DATABASES)
def test_lag_timesteps_as_the_reference(database: str) -> None:
    """Validate requested behavior using current-client observations and explicit expectations."""
    if not reachable(database):
        pytest.skip('Postgres test DSN is not configured')
    (lambda: assert_case_outcomes(database, LAG_TIMESTEPS))()


@pytest.mark.parametrize('database', DATABASES)
def test_quantiles_asked_as_the_reference(database: str) -> None:
    """Validate requested behavior using current-client observations and explicit expectations."""
    if not reachable(database):
        pytest.skip('Postgres test DSN is not configured')
    (lambda: assert_case_outcomes(database, QUANTILES, result=False))()


@pytest.mark.parametrize('database', DATABASES)
def test_response_field_for_quantiles_as_the_reference(database: str) -> None:
    """Validate requested behavior using current-client observations and explicit expectations."""
    if not reachable(database):
        pytest.skip('Postgres test DSN is not configured')
    (lambda: assert_case_outcomes(database, RESPONSE_FIELD))()


# --- a lagged average with empty windows ---------------------------------

#: The window of LAG_ALIGNED's query, in days, and its lags.
WINDOW_DAYS = 28
LAG_STEPS = 2


@functools.cache
def shop_orders() -> tuple[tuple[int, int, float], ...]:
    r"""The shop base's orders, as ``contract/bases.yaml`` builds them:
    order i is user ``i % 8``'s, at day ``7 i`` from 2023-01-01, priced
    ``10 + 1.5 i``; each as (user, day, price)."""
    bases = _runner().load_bases()
    assert any(
        '(i % 8)::BIGINT AS user_id, (10 + i * 1.5)::DOUBLE AS price, '
        "TIMESTAMP '2023-01-01' + to_days((i * 7)::INTEGER) AS ts FROM "
        'range(80)' in statement
        for statement in bases['shop']
    ), 'the shop base is not the one this computation reads'
    return tuple((i % 8, 7 * i, 10 + 1.5 * i) for i in range(80))


def _day(text: str) -> int:
    r"""A payload's anchor time as days from 2023-01-01 (each a midnight)."""
    import datetime

    at = datetime.datetime.fromisoformat(text.replace('Z', '+00:00'))
    assert at.time() == datetime.time(0), text
    return (at.date() - datetime.date(2023, 1, 1)).days


def window_average(user: int, after: int, until: int) -> float | None:
    r"""The average price of ``user``'s orders after day ``after`` and up to
    day ``until``, or None where the window holds none."""
    prices = [
        p for u, day, p in shop_orders() if u == user and after < day <= until
    ]
    return sum(prices) / len(prices) if prices else None


def by_hand(body: dict[str, Any], step: int) -> dict[str, list[Any]]:
    r"""For the request of forecast step ``step`` (from 0, a regression's one
    request step 0; its window days ``step * 28`` to ``(step + 1) * 28``), each instance row's target and
    lags, by column, context rows then rows to score, as the shop base's
    orders give them: a row's entity is the user ``__kumo_entity_ref_0``
    names, the target the average over (anchor + start, anchor + end], and
    lag j (``__kumo_arl<j>__``) the average over the window j + 1 steps of
    the query's end back: (anchor - (j + 1) end + start, anchor - j end]."""
    start, end = step * WINDOW_DAYS, (step + 1) * WINDOW_DAYS
    out: dict[str, list[Any]] = {}
    for half in ('context', 'predict'):
        table = body[half]['instance_table']
        at = {name: i for i, name in enumerate(table['columns'])}
        for row in table['rows']:
            user = int(row[at['__kumo_entity_ref_0']]['number'])
            day = _day(row[at['__kumo_anchor_time']])
            if half == 'context':
                out.setdefault('TARGET', []).append(
                    window_average(user, day + start, day + end)
                )
            for j in range(LAG_STEPS):
                out.setdefault(f'__kumo_arl{j}__', []).append(
                    window_average(
                        user, day - (j + 1) * end + start, day - j * end
                    )
                )
    return out


def _cells(body: dict[str, Any], column: str) -> list[Any]:
    r"""A column's cells over the context rows then the rows to score, each
    number as a float, an empty cell None."""
    out = []
    for half in ('context', 'predict'):
        table = body[half]['instance_table']
        if column not in table['columns']:
            continue
        at = table['columns'].index(column)
        out.extend(
            None if row[at] is None else float(row[at]['number'])
            for row in table['rows']
        )
    return out


@pytest.mark.parametrize('database', DATABASES)
def test_lagged_average_on_its_own_rows(database: str) -> None:
    """Validate requested behavior using current-client observations and explicit expectations."""
    if not reachable(database):
        pytest.skip('Postgres test DSN is not configured')
    assert_case_outcomes(database, (LAG_ALIGNED,))
    line = _cases_on(database)[_runner().case_key(LAG_ALIGNED, database)]
    sent = requests(line)
    assert len(sent) == 1
    for step, (_, _, body) in enumerate(sent):
        hand = by_hand(body, step)
        assert _cells(body, 'TARGET') == hand['TARGET']
        for j in range(LAG_STEPS):
            name = f'__kumo_arl{j}__'
            assert _cells(body, name) == hand[name], (database, step, name)
        assert hand['__kumo_arl0__'][-2:] == [34.0, 35.5]


# --- a lagged condition asked one entity a call ------------------


def lag_dtypes(sent: list[Any]) -> list[dict[str, str]]:
    r"""Each call's lag columns as its request's schema declares them: their
    dtype, by name."""
    return [
        {
            name: column['dtype']
            for name, column in body['schema']['instance_table'][
                'columns'
            ].items()
            if name.startswith('__kumo_arl')
        }
        for _, _, body in sent
    ]


def test_lag_requests_preserve_boolean_type_and_empty_windows() -> None:
    """Validate requested behavior using current-client observations and explicit expectations."""
    runner = _runner()
    observed = observed_cases()
    both = {'__kumo_arl0__': 'bool', '__kumo_arl1__': 'bool'}
    for database in (d for d in DATABASES if reachable(d)):
        sent = requests(observed[runner.case_key(LAG_DTYPE, database)])
        assert len(sent) == 2, database
        assert lag_dtypes(sent) == [both, both], database
        for _, _, body in sent:
            assert len(body['predict']['instance_table']['rows']) == 1, database
        own = by_hand(sent[1][2], 0)['__kumo_arl0__']
        assert own == [None] * len(own), (database, own)
        assert any(
            v is not None for v in by_hand(sent[0][2], 0)['__kumo_arl0__']
        )


@pytest.mark.parametrize('database', DATABASES)
def test_lag_dtype_per_call_is_boolean_even_for_empty_windows(
    database: str,
) -> None:
    if not reachable(database):
        pytest.skip('Postgres test DSN is not configured')
    assert_case_outcomes(database, (LAG_DTYPE,))
    line = _cases_on(database)[_runner().case_key(LAG_DTYPE, database)]
    assert (
        lag_dtypes(requests(line))
        == [{'__kumo_arl0__': 'bool', '__kumo_arl1__': 'bool'}] * 2
    )


#: The exit tests, each run on each database.
EXITS = (
    'test_num_forecasts_as_the_reference',
    'test_lag_timesteps_as_the_reference',
    'test_quantiles_asked_as_the_reference',
    'test_response_field_for_quantiles_as_the_reference',
    'test_lagged_average_on_its_own_rows',
    'test_lag_dtype_per_call_is_boolean_even_for_empty_windows',
)
