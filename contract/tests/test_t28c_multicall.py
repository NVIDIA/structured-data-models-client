# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.8c's exits on ``python-core`` (the package's test author's):
multi-call ``predict``, ``main``'s 1fc5a686.

    SDM_POSTGRES_TEST_DSN=... python -m pytest \
        contract/tests/test_t28c_multicall.py -q

Written from the design's T2.8c row (each item an exit) and the reference
(python-ref at ``contract/reference-env.txt``'s commit, which holds
1fc5a686), before the implementation. ``sdm-testkit``'s
``tests/t28c_multicall.rs`` holds ``rust-host`` to the same exits.

The cases, ``contract/cases/s1/multicall.*.json``, each run on DuckDB,
SQLite and Postgres (the case format's ``input.materialize.rows_per_call``
and ``input.fail_call``, the call the scripted endpoint fails):

- ``rows-per-call-3``: ``rows_per_call=3`` over the shop's 8 users, three
  calls of 3, 3 and 2 rows, the rows numbered 0 to 7 across them;
- ``fail-call-2`` and ``fail-call-0``: the same with the third, or the
  first, call failed by the endpoint: the calls answered before it carried
  on the error, or none;
- ``rows-per-call.<value>``: invalid values (0, -1, 2.5, 3.0, true, "3"),
  refused at materialize, and two that only lower the cap (10,001 and
  2**64), one call;
- the ceilings' boundaries, on generated users with one key and one
  target: ``entities-per-call.10001`` (calls of 10,000 and 1),
  ``entities.100000`` and ``.100001`` (the most one materialize gathers,
  and one past it), ``calls.100`` and ``.101`` (``rows_per_call=1``: the
  most calls, and one past; ``calls.100`` also sends about 109 context
  rows in each of its 100 calls, more than 10,000 summed, so the pre-send
  context check is held to each call);
- ``body-30mib.one-row`` and ``.several-rows``: a call past the 30 MiB
  body (1,000 context rows of a 40,000-character category), refused at
  materialize naming what to change;
- ``forecast.one-call``, ``forecast.batched`` and ``forecast.fail-call-1``:
  a two-step forecast whose entities fit one call, one whose entities do
  not (the batched-forecast refusal), and one whose second step's call
  fails (the step already answered carried on the error, which register
  P2B-9 says the core route drops until T2.8c).

Each test's one line says the exit it proves. Two kinds of evidence, both
from the reference: its committed lines (``run_python.py``: the outcome,
each request the endpoint received, the result's digest) and
``contract/t28c-multicall-vectors.json`` (``multicall_record.py``: the rows
each call scores, the result's rows, a refusal's details, the answers
carried on it, its first sentence and the 30 MiB body's remedy), which
``python-core`` is observed against by the same code.

Gating: ``t28c-pending.json``'s ``pending`` names, as ``py:<case key>``
(the line) and ``py-carried:<case key>`` (the vectors), the keys whose
test failed when written. Each must still fail (strict); once one passes
it fails, naming itself, until it is taken out. ``pending`` stays within
``authored``. Where ``SDM_POSTGRES_TEST_DSN`` names no server and
``SDM_REQUIRE_POSTGRES`` is not set, the Postgres keys have nothing to hold
and return without asserting; with it set, a missing server fails (H26).
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
RECORDER = ROOT / 'contract/tools/multicall_record.py'
CASES = ROOT / 'contract/cases/s1'
PENDING = Path(__file__).with_name('t28c-pending.json')
DATABASES = ('duckdb', 'sqlite', 'postgres')

EIGHT = list(range(8))
#: The invalid ``rows_per_call`` values and their cases.
INVALID = {
    'multicall.rows-per-call.zero': 0,
    'multicall.rows-per-call.negative': -1,
    'multicall.rows-per-call.fraction': 2.5,
    'multicall.rows-per-call.whole-float': 3.0,
    'multicall.rows-per-call.bool': True,
    'multicall.rows-per-call.text': '3',
}
#: Values that only lower the cap, so 8 entities go in one call.
LOWERED = {
    'multicall.rows-per-call.above-the-cap': 10_001,
    'multicall.rows-per-call.past-int64': 2**64,
}
#: How many rows each call the endpoint received scores, by case, on the
#: reference: the batches each case is about (a refusal at materialize
#: sends none).
CALLS = {
    'multicall.rows-per-call-3': [3, 3, 2],
    'multicall.fail-call-2': [3, 3, 2],
    'multicall.fail-call-0': [3],
    'multicall.rows-per-call.above-the-cap': [8],
    'multicall.rows-per-call.past-int64': [8],
    'multicall.entities-per-call.10001': [10_000, 1],
    'multicall.calls.100': [1] * 100,
    'multicall.forecast.one-call': [3, 3],
    'multicall.forecast.fail-call-1': [3, 3],
}
#: The refusals at materialize, by case: the code, and the details the
#: reference gives (the ceilings' counts, the 30 MiB body's mark).
REFUSED = {
    **{case: ('INVALID_REQUEST', {}) for case in INVALID},
    'multicall.entities.100001': (
        'UNSUPPORTED_REQUEST',
        {'entities': 100_001, 'gathered_times': 1, 'total': 100_001},
    ),
    'multicall.calls.101': (
        'UNSUPPORTED_REQUEST',
        {'entities': 101, 'calls': 101, 'rows_per_call': 1},
    ),
    'multicall.forecast.batched': (
        'UNSUPPORTED_REQUEST',
        {'entities': 8, 'rows_per_call': 3},
    ),
    'multicall.body-30mib.one-row': (
        'INVALID_REQUEST',
        {'batch_too_large_to_send': True},
    ),
    'multicall.body-30mib.several-rows': (
        'INVALID_REQUEST',
        {'batch_too_large_to_send': True},
    ),
}
#: The failed calls, by case: the call failed, and what the error carries
#: of the calls before it (``calls_answered`` and ``rows_answered``, or a
#: forecast's ``parts_answered``), none where the first call failed.
FAILED = {
    'multicall.fail-call-2': (2, {'calls_answered': 2, 'rows_answered': 6}),
    'multicall.fail-call-0': (0, {}),
    'multicall.forecast.fail-call-1': (1, {'parts_answered': 1}),
}
#: The 30 MiB body cases: the remedy the client names, by case.
BODY = {
    'multicall.body-30mib.one-row': 'A call already scores one row',
    'multicall.body-30mib.several-rows': (
        'What comes down is one of three. For an ordinary question it is '
        'the call: materialize(rows_per_call=N) below the 3 this one holds'
    ),
}
CASE_IDS = (
    'multicall.rows-per-call-3',
    'multicall.fail-call-2',
    'multicall.fail-call-0',
    *INVALID,
    *LOWERED,
    'multicall.entities-per-call.10001',
    'multicall.entities.100000',
    'multicall.entities.100001',
    'multicall.calls.100',
    'multicall.calls.101',
    *BODY,
    'multicall.forecast.one-call',
    'multicall.forecast.batched',
    'multicall.forecast.fail-call-1',
)


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t28c_run_python', RUNNER)


def recorder() -> ModuleType:
    return _module('multicall_record', RECORDER)


def key(case_id: str, database: str) -> str:
    return runner().case_key(case_id, database)


KEYS_ALL = tuple(key(c, d) for d in DATABASES for c in CASE_IDS)
#: The case keys observed with a call raising an error that is not the
#: client's own (``multicall_record.RAISED``): what each call answered
#: before it, kept on it, by case.
RAISED = {
    'multicall.rows-per-call-3': {'calls_answered': 2, 'rows_answered': 6},
    'multicall.forecast.one-call': {'parts_answered': 1},
}
RAISED_KEYS = tuple(key(c, d) for d in DATABASES for c in RAISED)


def held(name: str, check: Any) -> None:
    r"""Runs ``check``: it must pass, or, while ``name`` is pending, fail."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    if name not in gating['pending']:
        check()
        return
    try:
        check()
    except Exception:  # noqa: BLE001 - pending: failing, as when written
        return
    raise AssertionError(
        f'{name} passes now: take it out of '
        'contract/tests/t28c-pending.json pending'
    )


def test_the_pending_list_only_shrinks() -> None:
    r"""Control: ``pending`` names only keys the test author wrote, and
    ``authored`` is every case's key on each database on python-core, line
    and vectors, and on DuckDB and SQLite on rust-host (its Postgres pass
    replays python-core's fetches, which ``cross-host (postgres)``
    holds), and each raising case's key on python-core
    (``py-raised:``)."""
    gating = json.loads(PENDING.read_text(encoding='utf-8'))
    assert set(gating['pending']) <= set(gating['authored'])
    assert set(gating['authored']) == (
        {f'py:{k}' for k in KEYS_ALL}
        | {f'py-carried:{k}' for k in KEYS_ALL}
        | {f'rust:{k}' for k in KEYS_ALL if not k.endswith('@postgres')}
        | {f'py-raised:{k}' for k in RAISED_KEYS}
    )


def test_every_multi_call_case_is_named_here() -> None:
    r"""Control: the cases here are every ``multicall.*`` case, once."""
    assert len(set(CASE_IDS)) == len(CASE_IDS)
    assert set(CASE_IDS) == set(recorder().case_ids())


# --- the cases -------------------------------------------------------------------


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""The T2.8c cases by id, each with its digest, as the runner loads
    them."""
    loaded = runner().load_cases([CASES / f'{c}.json' for c in CASE_IDS])
    return {case['id']: (case, digest) for case, digest in loaded}


def _case(case_id: str) -> dict[str, Any]:
    case, _ = cases()[case_id]
    assert case['status'] == 'acceptance' and case['slice'] == 'S1'
    assert runner().case_databases(case) == DATABASES
    return case


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


def test_rows_per_call_3_over_8_entities_and_a_failure_on_call_2() -> None:
    r"""Exits "``rows_per_call=3`` over 8 entities" and "a failure on call
    2": the shop's 8 users asked in calls of 3 through predict, and the same
    with the call numbered 2 (the third, the scripted endpoint numbering
    from zero) or the first failed."""
    for case_id, fail_call in (
        ('multicall.rows-per-call-3', None),
        ('multicall.fail-call-2', 2),
        ('multicall.fail-call-0', 0),
    ):
        case = _case(case_id)
        given = case['input']
        assert given['base'] == 'shop' and 'setup' not in given
        assert given['materialize']['indices'] == EIGHT
        assert given['materialize']['rows_per_call'] == 3
        assert given.get('fail_call') == fail_call
        assert case['until'] == 'predict'
        assert 'FORECAST' not in given['pql']


@pytest.mark.parametrize('case_id', [*INVALID, *LOWERED])
def test_each_rows_per_call_value_is_the_one_named(case_id: str) -> None:
    r"""Exit "invalid ``rows_per_call`` values": each case passes its value
    as written, of its own type (a float with nothing after the point is
    no whole number to the reference), over the shop's 8 users; the two
    that only lower the cap are whole numbers past it."""
    case = _case(case_id)
    wanted = {**INVALID, **LOWERED}[case_id]
    given = case['input']['materialize']['rows_per_call']
    assert type(given) is type(wanted) and given == wanted
    assert case['input']['materialize']['indices'] == EIGHT
    assert 'fail_call' not in case['input']


def test_each_ceiling_s_boundary() -> None:
    r"""Exit "each ceiling's boundary": 10,001 entities (one past a call's
    10,000), 100,000 and 100,001 entities (the most one materialize
    gathers, and one past), 100 and 101 calls of one entity (the most
    calls, and one past); each on generated users holding more rows than
    it asks about, so context is left."""
    for case_id, count, rows_per_call, until in (
        ('multicall.entities-per-call.10001', 10_001, None, 'predict'),
        ('multicall.entities.100000', 100_000, None, 'materialize'),
        ('multicall.entities.100001', 100_001, None, 'materialize'),
        ('multicall.calls.100', 100, 1, 'predict'),
        ('multicall.calls.101', 101, 1, 'predict'),
    ):
        case = _case(case_id)
        given = case['input']['materialize']
        assert given['indices'] == list(range(count)), case_id
        assert given.get('rows_per_call') == rows_per_call, case_id
        assert case['until'] == until, case_id
        connection = _built(case)
        try:
            users = connection.execute('SELECT count(*) FROM users').fetchone()
            tables = connection.execute(
                "SELECT count(*) FROM duckdb_tables() WHERE schema_name = 'main'"
            ).fetchone()
        finally:
            connection.close()
        assert users[0] > count and tables[0] == 1, case_id


@pytest.mark.parametrize('case_id', list(BODY))
def test_a_call_past_the_30_mib_body(case_id: str) -> None:
    r"""Exit "the 30 MiB refusal with one row and with several": one
    entity, or three, over users whose context rows carry more than 30 MiB
    of one text column."""
    case = _case(case_id)
    assert case['input']['materialize']['indices'] == (
        [0] if case_id.endswith('one-row') else [0, 1, 2]
    )
    assert 'rows_per_call' not in case['input']['materialize']
    connection = _built(case)
    try:
        rows, size = connection.execute(
            'SELECT count(*), sum(strlen(cat)) FROM users'
        ).fetchone()
    finally:
        connection.close()
    assert rows > 1_000 and size > 30 * 1024 * 1024


def test_the_forecasts() -> None:
    r"""Exits "a forecast over one call" and "the batched-forecast
    refusal": a two-step forecast over as many entities as one call scores
    (``rows_per_call=3`` over 3), over more (3 over 8), and over one call a
    step with the second step's call failed."""
    for case_id, indices, fail_call in (
        ('multicall.forecast.one-call', [0, 1, 2], None),
        ('multicall.forecast.batched', EIGHT, None),
        ('multicall.forecast.fail-call-1', [0, 1, 2], 1),
    ):
        case = _case(case_id)
        given = case['input']
        assert ' FORECAST 2 TIMEFRAMES ' in given['pql'], case_id
        assert given['materialize']['indices'] == indices, case_id
        assert given['materialize']['rows_per_call'] == 3, case_id
        assert given.get('fail_call') == fail_call, case_id


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
    return {
        json.loads(line)['case']: json.loads(line)
        for line in path.read_text(encoding='utf-8').splitlines()
    }


@functools.cache
def vectors() -> dict[str, Any]:
    return recorder().load()


def _expected(case: dict[str, Any]) -> tuple[Any, ...]:
    expect = case['expect']
    return (
        expect['outcome'],
        expect['stage'],
        expect.get('error'),
        expect.get('code'),
    )


def _outcome(line: dict[str, Any]) -> tuple[Any, ...]:
    outcome = line['outcome']
    return (
        outcome['result'],
        outcome['stage'],
        outcome.get('error'),
        outcome.get('code'),
    )


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_python_ref_s_line_is_at_its_case_file(case_key: str) -> None:
    r"""Exit "cases on DuckDB, SQLite and Postgres": python-ref's line for
    the case on the database, recorded at the case file, ends as the case
    expects, with a request for each call."""
    case_id = case_key.split('@')[0]
    case, digest = cases()[case_id]
    line = reference_lines().get(case_key)
    assert line is not None, f'{case_key}: no python-ref line'
    assert line['identity']['runner'] == 'python-ref'
    assert line['identity']['inputs']['case'] == digest
    assert _outcome(line) == _expected(case)
    assert len(line['public']['payloads']) == len(CALLS.get(case_id, []))


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_the_reference_answers_each_case_as_it_names(case_key: str) -> None:
    r"""The cases exercise what they name, on the reference (so none is
    vacuous): the rows each call scores; a success's rows numbered across
    the calls in the order asked (a forecast's each entity's steps); a
    refusal's details; the answers carried on a failed call, the rows of
    the calls before it; the 30 MiB body's remedy."""
    case_id = case_key.split('@')[0]
    case, digest = cases()[case_id]
    document = vectors()
    assert document['recorded_by'] == 'contract/tools/multicall_record.py'
    assert (
        document['reference']
        == reference_lines()[case_key]['identity']['reference']['commit']['sha']
    )
    vector = document['vectors'].get(case_key)
    assert vector is not None, f'{case_key}: no vector'
    assert vector['case'] == digest
    assert _outcome({'outcome': vector['outcome']}) == _expected(case)
    assert vector['calls'] == CALLS.get(case_id, [])
    indices = case['input']['materialize']['indices']
    if vector['outcome']['result'] == 'success' and case['until'] == 'predict':
        result = vector['result']
        forecast = 'forecast_step' in result['columns']
        numbered = (
            [(n, step) for n in range(len(indices)) for step in (1, 2)]
            if forecast
            else [(n,) for n in range(len(indices))]
        )
        assert result['count'] == len(numbered)
        if 'rows' in result:
            at = [result['columns'].index('row_index')]
            if forecast:
                at.append(result['columns'].index('forecast_step'))
            ids = [row[result['columns'].index('id')] for row in result['rows']]
            assert [tuple(row[i] for i in at) for row in result['rows']] == (
                numbered
            )
            assert ids == [indices[n[0]] for n in numbered]
        else:
            assert [row[:2] for row in result['first']] == [
                [n, indices[n]] for n in range(3)
            ]
            assert [row[:2] for row in result['last']] == [
                [n, indices[n]] for n in range(len(indices) - 3, len(indices))
            ]
    if case_id in REFUSED:
        code, details = REFUSED[case_id]
        assert vector['outcome']['code'] == code
        assert vector['details'] == details
    if case_id in FAILED:
        call, carried = FAILED[case_id]
        assert vector['details'] == {
            'call': call,
            'error': 'scripted_failure',
            **carried,
        }
        answered = vector['answered']
        if not carried:
            assert answered is None
        else:
            # The rows of the calls before the failed one, numbered across
            # them (a forecast's: its first step's, one call).
            assert answered['count'] == sum(vector['calls'][:call])
            assert [row[0] for row in answered['rows']] == list(
                range(answered['count'])
            )
            if 'parts_answered' in carried:
                step = answered['columns'].index('forecast_step')
                assert {row[step] for row in answered['rows']} == {1}
    if case_id in BODY:
        assert vector['remedy'].startswith(BODY[case_id])
        assert vector['opening'] == (
            '[INVALID_REQUEST] Context size exceeds the 30MB limit.'
        )
    else:
        assert vector['remedy'] is None


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
    r"""python-core over the T2.8c cases on ``database``, as that
    database's job runs them: each observation by its key, or under ``''``
    why the run failed whole."""
    with tempfile.TemporaryDirectory(prefix='t28c-') as tmp:
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


def same_outcome(want: tuple[Any, ...], got: tuple[Any, ...]) -> bool:
    r"""Two outcomes the same, but that the core route raises a refusal
    coded ``UNSUPPORTED_REQUEST`` as ``UnsupportedRequestError``, a
    subclass of the reference's ``StructuredDataError`` (register P1B-20,
    approved by the owner, E78)."""
    if want == got:
        return True
    return (
        want[0] == got[0] == 'refusal'
        and want[1:2] == got[1:2]
        and want[3] == got[3] == 'UNSUPPORTED_REQUEST'
        and (want[2], got[2])
        == ('StructuredDataError', 'UnsupportedRequestError')
    )


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_python_core_answers_each_case_as_the_reference_does(
    case_key: str,
) -> None:
    r"""Exits "larger requests split with a per-batch walk and context",
    "``rows_per_call``", "rows numbered across calls", "the 100,000-entity
    and 100-call ceilings", "the pre-send context check per call", "the
    per-batch 30 MiB body refusal" and "the batched-forecast refusal":
    python-core runs the case on the database on the core route (the
    native module loaded, each of the case's core operations run), and its
    outcome, each request the scripted endpoint received (each call's
    walk and context) and its result through ``to_pandas()`` (columns, row
    count and the rows' digest, so their numbering and order) are
    python-ref's."""
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
        missing = sorted(set(case['core']) - set(identity['operations']))
        assert not missing, f'{case_key}: core operations not run: {missing}'
        want = reference_lines()[case_key]
        assert same_outcome(_outcome(want), _outcome(line)), (
            f'{case_key}: {_outcome(line)} where python-ref has {_outcome(want)}'
        )
        found = differences(payloads(want), payloads(line))
        assert not found, f'{case_key}: the payload differs: ' + '; '.join(
            found
        )
        found = differences(want['public']['result'], line['public']['result'])
        assert not found, f'{case_key}: the result differs: ' + '; '.join(found)

    held(f'py:{case_key}', check)


@functools.cache
def core_carried(case_key: str) -> dict[str, Any]:
    r"""python-core's observation of the case, by the recorder's code."""
    case_id, _, database = case_key.partition('@')
    case, _ = cases()[case_id]
    return recorder().observe(case, database or 'duckdb', core=True)


@pytest.mark.parametrize('case_key', KEYS_ALL)
def test_python_core_carries_what_the_reference_carries(case_key: str) -> None:
    r"""Exits "answered parts on a failed call", "its remedy texts",
    "rows numbered across calls" and the ceilings' and invalid values'
    refusals by name: python-core, on the core route, sends calls of the
    reference's sizes; answers with the reference's rows, row by row; and
    refuses with the reference's details (each detail the reference gives,
    the same; the core's own fields beside them), the reference's answers
    carried on a failed call (``details['answered']``, row by row), the
    reference's first sentence and remedy for a call past the 30 MiB body,
    and a message naming ``rows_per_call`` for an invalid one."""
    case_id, _, database = case_key.partition('@')
    if database == 'postgres' and not _server():
        return

    def check() -> None:
        want = vectors()['vectors'][case_key]
        got = core_carried(case_key)
        outcome = (
            _outcome({'outcome': want['outcome']}),
            _outcome({'outcome': got['outcome']}),
        )
        assert same_outcome(*outcome), (
            f'{case_key}: {outcome[1]} where python-ref has {outcome[0]}'
        )
        assert got['calls'] == want['calls'], (
            f'{case_key}: calls of {got["calls"][:5]} rows where python-ref '
            f'sends {want["calls"][:5]}'
        )
        found = differences(want['result'], got['result'], '/result')
        found += differences(want['answered'], got['answered'], '/answered')
        assert not found, f'{case_key}: ' + '; '.join(found)
        details = got['details'] or {}
        for name, value in (want['details'] or {}).items():
            assert details.get(name) == value, (
                f'{case_key}: details[{name!r}] is {details.get(name)!r} '
                f'where python-ref has {value!r}'
            )
        assert got['remedy'] == want['remedy'], (
            f'{case_key}: the remedy is {got["remedy"]!r:.200} where '
            f'python-ref names {want["remedy"]!r:.200}'
        )
        if case_id in BODY:
            assert got['opening'] == want['opening'], (
                f'{case_key}: {got["opening"]!r:.200} where python-ref '
                f'opens {want["opening"]!r}'
            )
        if case_id in INVALID:
            assert 'rows_per_call' in (got['opening'] or ''), (
                f'{case_key}: the refusal does not name rows_per_call: '
                f'{got["opening"]!r:.200}'
            )

    held(f'py-carried:{case_key}', check)


# --- a call raising an error that is not the client's own ---------------------------


@pytest.mark.parametrize('case_key', RAISED_KEYS)
def test_the_reference_keeps_the_answers_when_a_call_raises(
    case_key: str,
) -> None:
    r"""Exit "answered parts on a failed call", for a failure that is not
    the client's own (T2.8c round 1): with the client's transport raising
    ``ValueError`` on the third call of ``rows_per_call=3`` over 8, or on
    the second step's call of the two-step forecast, as the Databricks SDK
    raises one on a failed OAuth refresh, the reference raises that
    ``ValueError`` as it was, the calls before it answered and kept on it:
    6 rows over 2 calls, or the first step's 3 rows."""
    case_id = case_key.split('@')[0]
    case, digest = cases()[case_id]
    vector = vectors()['raised'].get(case_key)
    assert vector is not None, f'{case_key}: no vector'
    call = recorder().RAISED[case_id]
    assert vector['case'] == digest and vector['call'] == call
    assert 'fail_call' not in case['input']
    assert _outcome({'outcome': vector['outcome']}) == (
        'refusal',
        'predict',
        'ValueError',
        None,
    )
    assert vector['calls'] == CALLS[case_id][:call]
    assert vector['details'] == RAISED[case_id]
    answered = vector['answered']
    assert answered['count'] == sum(vector['calls'])
    assert [row[0] for row in answered['rows']] == list(
        range(answered['count'])
    )
    if 'forecast_step' in answered['columns']:
        step = answered['columns'].index('forecast_step')
        assert {row[step] for row in answered['rows']} == {1}


@pytest.mark.parametrize('case_key', RAISED_KEYS)
def test_python_core_keeps_the_answers_when_a_call_raises(
    case_key: str,
) -> None:
    r"""Exit "answered parts on a failed call", for a failure that is not
    the client's own (T2.8c round 1): python-core, on the core route, its
    transport raising the same ``ValueError`` on the same call, raises it
    as the reference does, with the reference's answers kept on it
    (``details['answered']`` row by row, and ``calls_answered`` and
    ``rows_answered``, or a forecast's ``parts_answered``)."""
    case_id, _, database = case_key.partition('@')
    if database == 'postgres' and not _server():
        return

    def check() -> None:
        want = vectors()['raised'][case_key]
        case, _ = cases()[case_id]
        got = recorder().observe(
            case,
            database or 'duckdb',
            core=True,
            raise_on_call=recorder().RAISED[case_id],
        )
        outcome = (
            _outcome({'outcome': want['outcome']}),
            _outcome({'outcome': got['outcome']}),
        )
        assert outcome[0] == outcome[1], (
            f'{case_key}: {outcome[1]} where python-ref has {outcome[0]}'
        )
        assert got['calls'] == want['calls'], (
            f'{case_key}: calls of {got["calls"]} rows where python-ref '
            f'sends {want["calls"]}'
        )
        details = got['details'] or {}
        for name, value in want['details'].items():
            assert details.get(name) == value, (
                f'{case_key}: details[{name!r}] is {details.get(name)!r} '
                f'where python-ref has {value!r}'
            )
        found = differences(want['answered'], got['answered'], '/answered')
        assert not found, f'{case_key}: ' + '; '.join(found)

    held(f'py-raised:{case_key}', check)
