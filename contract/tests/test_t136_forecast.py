# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.36 fix 21: forecast decomposition moves into the core (owner item 50),
held from outside the implementation (the independent test author).

Today ``sdm_client.decompose`` splits a forecast of N steps into N
regressions before the core sees it, and the core refuses a forecast
(``UNSUPPORTED_REQUEST``, rule ``problem_type``, ``sdm-core``'s
``materialize.rs``). After the fix the core decomposes it and the Python
client hands it over whole. These tests hold the Python side to
``contract/t136-forecast-vectors.json`` (``contract/tools/forecast_record.py``):

- ``test_python_core_sends_what_it_sent_before_the_move``: ``python-core``
  (forced onto the core's DuckDB routes) gives, for every vector, what it
  gave at fc2c0e2b: the outcome (code, stage, class and message), each
  request's method, path and tables, count and order included, the result
  through ``to_pandas()`` (dtypes, rows, values) and the warning classes.
  Each request's raw bytes (SHA-256, ``metadata.request_id`` masked) were
  held to fc2c0e2b's too until main's 26ecdd28 (T2.0c) took the inference
  config the caller did not set out of every request on both routes; they
  now differ from fc2c0e2b's in every request, and each request is the
  reference's under the ``payload`` rule.
- ``test_python_core_answers_as_the_reference``: where the reference
  answers and the core ports what is asked, ``python-core``'s requests
  (``payload``-rule digests) and result are the reference's.
- ``test_python_old_answers_as_the_reference_and_as_before``: the default
  route answers as the reference (its requests and result) and as at
  fc2c0e2b, held as ``python-core`` is above (its own messages included).
- The refusals around a forecast stay what they are, on both routes:
  ``top_k``; more steps than ``decompose.MOST_PARTS`` (and raising it lets
  them through); zero steps (``s0.refuse.task-forecast-zero``); pinning a
  session (``_refuse_to_pin_a_rewritten_question``); and a step that fails
  said of the step (``about_the_part``).

From the review of fix 21:

- ``test_a_forecast_of_many_steps_is_refused_at_once``: 10**9, 10,001 (past
  the grammar's 10,000), 10,000 and 201 (past the default ``MOST_PARTS``)
  steps are refused ``INVALID_REQUEST`` within two seconds and 64 MiB of
  peak memory, on the old route, with validation alone on the core, and
  with every DuckDB route on the core.
- ``test_raised_most_parts_lets_steps_through_with_validation_on_the_core``:
  ``MOST_PARTS`` raised to 250 lets 201 steps through, validation included.
- ``test_each_step_of_a_forecast_is_sent_with_its_own_request_id``: N calls,
  N distinct request ids, each ``sdm-`` and twelve hex digits, as the
  reference draws them.

From T2.8, which lifts ``target_filter``: ``count-filtered-3`` (a target
with ``WHERE``) is answered as the reference answers it on both core hosts;
its ``python_core`` is recorded again under T2.8's candidate (the
recorder's ``recandidate``, ``python_core_candidate``), and it is no longer
among the refusals worded as before. From T2.13b, which lifts
``entity_filter``: ``sum-entity-filter-3`` (an entity filter) likewise,
its ``python_core`` recorded again under T2.13b's candidate.

``sdm-testkit``'s ``tests/t136_forecast.rs`` holds ``rust-host`` to the same
vectors. The implementer changes nothing of this file, the vectors or the
recorder.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / 'contract/t136-forecast-vectors.json'


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module('forecast_record', ROOT / 'contract/tools/forecast_record.py')
RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))
BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}
IDS = [v['id'] for v in RECORDED['vectors']]
#: Held to the reference on python-core: it answers, and the core ports
#: every part of the question (the filters are not ported).
AS_THE_REFERENCE = [
    v['id']
    for v in RECORDED['vectors']
    if v['rust_host'] == 'reference'
    and v['reference']['outcome']['result'] == 'success'
]


@pytest.fixture(scope='module')
def core_runs() -> dict[str, Any]:
    return {v['id']: R.drive(v, core=True) for v in R.VECTORS}


@pytest.fixture(scope='module')
def old_runs() -> dict[str, Any]:
    return {v['id']: R.drive(v) for v in R.VECTORS}


def test_the_vector_file_is_the_recorder_s() -> None:
    assert RECORDED['format'] == R.FORMAT
    assert RECORDED['recorded_by'] == 'contract/tools/forecast_record.py'
    assert RECORDED['candidate'].startswith('fc2c0e2b')
    env = (ROOT / 'contract/reference-env.txt').read_text(encoding='utf-8')
    assert f'reference: {RECORDED["reference"]}' in env
    assert RECORDED['reference_environment']['pandas'] == '3.0.6'
    assert RECORDED['reference_environment']['pyarrow'] == '25.0.1'
    assert RECORDED['candidate_environment']['pandas'] == '3.0.6'
    assert RECORDED['payload_exclusions'] == [['metadata', 'request_id']]
    assert IDS == [v['id'] for v in R.VECTORS]
    for vector, written in zip(RECORDED['vectors'], R.VECTORS, strict=True):
        for key in ('pql', 'anchor_time', 'options'):
            assert vector[key] == written[key], vector['id']
        assert vector['case'] == R.case_for(written), vector['id']
        assert vector['rust_host'] == R.RUST_HOST.get(
            written['id'], 'reference'
        )
    # Recorded again under a later candidate: only what that package
    # changed by design.
    assert {
        v['id'] for v in RECORDED['vectors'] if 'python_core_candidate' in v
    } == RECANDIDATED


def test_the_vectors_are_not_vacuous() -> None:
    steps = {}
    for vector in RECORDED['vectors']:
        ref = vector['reference']
        if ref['outcome']['result'] == 'success':
            steps[vector['id']] = len(ref['payloads'])
            assert ref['result']['columns'][2]['name'] == 'forecast_step'
            assert ref['result']['rows'] == 2 * len(ref['payloads'])
            # Each step asks about its own window: no two requests alike.
            digests = [p['digest'] for p in ref['payloads']]
            assert len(set(digests)) == len(digests), vector['id']
    assert steps['sum-days-3'] == 3 and steps['sum-days-1'] == 1
    assert steps['count-days-4'] == 4
    assert len(AS_THE_REFERENCE) >= 8
    # python-core today: a forecast reaches the endpoint as its parts.
    assert len(BY_ID['sum-days-3']['python_core']['payloads']) == 3


def _requests_but_their_bytes(answer: dict[str, Any]) -> list[dict[str, Any]]:
    r"""Each request as recorded, its bytes' digests left out: the method,
    path and each table's rows and nulls, in order."""
    return [
        {k: v for k, v in p.items() if k not in ('masked_sha256', 'digest')}
        for p in answer['payloads']
    ]


#: The vectors whose python-core is recorded again under a later candidate
#: (``python_core_candidate``): T2.8 lifts ``target_filter``, T2.13b
#: ``entity_filter``.
RECANDIDATED = {'count-filtered-3', 'sum-entity-filter-3'}


def _held_before_and_now(
    got: dict[str, Any],
    vector: dict[str, Any],
    before: dict[str, Any],
    again: bool = False,
) -> None:
    r"""What a route sent at fc2c0e2b, held but for the bytes main's
    26ecdd28 moved on both routes (no inference config the caller did not
    set): the outcome, result and warnings exactly; each request's method,
    path and tables, count and order included; and each request's bytes,
    under the payload rule, are now the reference's."""
    assert got['outcome'] == before['outcome']
    assert got['result'] == before['result']
    assert got['warnings'] == before['warnings']
    assert _requests_but_their_bytes(got) == _requests_but_their_bytes(before)
    reference = vector['reference']
    if reference['outcome']['result'] == 'success' and got['payloads']:
        assert [p['digest'] for p in got['payloads']] == [
            p['digest'] for p in reference['payloads']
        ], "each request is the reference's under the payload rule"
    if again:
        # Recorded after 26ecdd28, under the later candidate: its bytes too.
        assert [p['masked_sha256'] for p in got['payloads']] == [
            p['masked_sha256'] for p in before['payloads']
        ]
        return
    # Not vacuous: where it sent requests, fc2c0e2b's bytes were the
    # Python decomposition's with the engine's inference config, which no
    # route sends now.
    for now, then in zip(got['payloads'], before['payloads'], strict=True):
        assert now['masked_sha256'] != then['masked_sha256']


@pytest.mark.parametrize('vector_id', IDS)
def test_python_core_sends_what_it_sent_before_the_move(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    vector = BY_ID[vector_id]
    _held_before_and_now(
        core_runs[vector_id],
        vector,
        vector['python_core'],
        again='python_core_candidate' in vector,
    )


@pytest.mark.parametrize('vector_id', AS_THE_REFERENCE)
def test_python_core_answers_as_the_reference(
    vector_id: str, core_runs: dict[str, Any]
) -> None:
    assert R.public(core_runs[vector_id]) == BY_ID[vector_id]['reference']


@pytest.mark.parametrize('vector_id', IDS)
def test_python_old_answers_as_the_reference_and_as_before(
    vector_id: str, old_runs: dict[str, Any]
) -> None:
    got = old_runs[vector_id]
    assert R.public(got) == BY_ID[vector_id]['reference']
    vector = BY_ID[vector_id]
    _held_before_and_now(got, vector, vector['python_old'])


# --- the refusals around a forecast -------------------------------------------

#: Today's wording of each refusal, on both routes (measured at fc2c0e2b).
TOP_K = (
    '[INVALID_REQUEST] top_k is how many candidates a ranking keeps, and a '
    'forecast ranks nothing; it gives back one value per step'
)
PAST_MOST_PARTS = (
    '[INVALID_REQUEST] 201 forecast steps means 201 calls, one for each, '
    'because the model has no task that answers them together. That is past '
    'the 200 this will do without being told to; ask about fewer, or raise '
    'sdm_client.decompose.MOST_PARTS if the wait is what you want.'
)
ZERO = (
    "[INVALID_REQUEST] 'PREDICT SUM(orders.price, 0, 7, days) FORECAST 0 "
    "TIMEFRAMES FOR EACH users.user_id' is not a query a relational model can "
    'answer: FORECAST value must be at least 1, got 0.'
)
#: A step's window reaching before the data: said of the step that failed.
OUTRAN_CORE = (
    '[INVALID_REQUEST] the request is not one the core accepts. This is '
    'forecast step 2 of 3.'
)
OUTRAN_OLD = (
    '[INVALID_REQUEST] Context anchor timestamp is too early or aggregation '
    'time range is too large. To make this prediction, we would need data '
    "back to '2022-12-27 00:00:00', however, your data only contains data "
    "back to '2023-01-01 00:00:00'. This is forecast step 2 of 3. The window "
    "named above is that step's rather than the one written; ask for fewer "
    'than 2 steps if the data does not reach that far.'
)


@pytest.mark.parametrize(
    ('vector_id', 'core', 'message'),
    [
        ('refuse.top-k', True, TOP_K),
        ('refuse.top-k', False, TOP_K),
        ('refuse.past-most-parts', True, PAST_MOST_PARTS),
        ('refuse.past-most-parts', False, PAST_MOST_PARTS),
        ('refuse.zero-steps', True, ZERO),
        ('refuse.zero-steps', False, ZERO),
        ('refuse.step-outruns-data', True, OUTRAN_CORE),
        ('refuse.step-outruns-data', False, OUTRAN_OLD),
    ],
)
def test_a_forecast_refusal_is_worded_as_before(
    vector_id: str,
    core: bool,
    message: str,
    core_runs: dict[str, Any],
    old_runs: dict[str, Any],
) -> None:
    got = (core_runs if core else old_runs)[vector_id]['outcome']
    assert got['result'] == 'refusal'
    assert got['stage'] == 'materialize'
    assert got['message'] == message
    assert (core_runs if core else old_runs)[vector_id]['payloads'] == []


@pytest.mark.parametrize(
    'core', [True, False], ids=['python-core', 'python-old']
)
def test_most_parts_is_the_caller_s_to_raise(
    core: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""The refusal tells the caller to raise ``decompose.MOST_PARTS``; doing
    so is honoured on both routes, and lowering it is too."""
    from sdm_client import decompose

    vector = {
        'id': 'most-parts',
        'pql': 'PREDICT SUM(orders.price, 0, 7, days) FORECAST 3 TIMEFRAMES '
        'FOR EACH users.user_id',
        'anchor_time': R.ANCHOR,
        'options': {},
    }
    monkeypatch.setattr(decompose, 'MOST_PARTS', 2)
    lowered = R.drive(vector, core=core)['outcome']
    assert lowered['code'] == 'INVALID_REQUEST'
    assert lowered['message'].startswith(
        '[INVALID_REQUEST] 3 forecast steps means 3 calls, one for each'
    )
    assert 'past the 2 this will do' in lowered['message']
    monkeypatch.setattr(decompose, 'MOST_PARTS', 201)
    raised = R.drive(
        {
            **vector,
            'pql': vector['pql']
            .replace('3 TIME', '201 TIME')
            .replace('0, 7, days', '0, 1, days'),
        },
        core=core,
    )
    assert raised['outcome']['result'] == 'success'
    assert len(raised['payloads']) == 201


@pytest.mark.parametrize(
    'core', [True, False], ids=['python-core', 'python-old']
)
def test_a_forecast_cannot_pin_a_session(
    core: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    import duckdb
    import pandas as pd
    from sdm_client import (
        KumoRelational,
        PredictiveQuery,
        Schema,
        SDMClient,
        StructuredDataError,
    )
    from sdm_connectors import DuckDBSource

    serve = R._module(
        'scripted_endpoint', ROOT / 'contract/runners/scripted_endpoint.py'
    ).serve
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)
    if core:
        monkeypatch.setenv('SDM_CORE_ROUTES', ','.join(R.EN.core_routes({})))
    connection = duckdb.connect()
    try:
        for statement in R.EN.shop():
            connection.execute(statement)
        schema = Schema.from_source(DuckDBSource(connection))
        request = PredictiveQuery(
            'PREDICT SUM(orders.price, 0, 7, days) FORECAST 3 TIMEFRAMES '
            'FOR EACH users.user_id',
            schema,
        ).materialize(indices=[0, 1], anchor_time=pd.Timestamp(R.ANCHOR))
        with serve() as (url, received), SDMClient(url) as client:
            with pytest.raises(StructuredDataError) as raised:
                client.fit(request, model=KumoRelational())
            assert received.requests == []
    finally:
        connection.close()
    assert raised.value.code == 'INVALID_REQUEST'
    assert str(raised.value) == (
        '[INVALID_REQUEST] a forecast is asked as 3 questions, each needing '
        'the context its own labels come from, and a session pins one '
        'context. Ask it with predict(), which sends each and puts the '
        'answers back together.'
    )


# --- many steps, refused at once (T1.36 fix 21 review) --------------------------

#: Measures one materialize of a forecast of ``n`` steps in a fresh
#: interpreter: the outcome, the seconds it took and the growth of the
#: process's peak resident memory (bytes on macOS, KiB on Linux) across it.
_MANY_STEPS = r"""
import json, os, resource, sys, time, warnings
sys.path.insert(0, sys.argv[1])
import duckdb, pandas as pd
warnings.simplefilter('ignore')
from sdm_client import PredictiveQuery, Schema
from sdm_connectors import DuckDBSource
import empty_null_record as EN
routes, n = json.loads(sys.argv[2]), sys.argv[3]
if routes:
    os.environ['SDM_CORE_ROUTES'] = ','.join(routes)
connection = duckdb.connect()
for statement in EN.shop():
    connection.execute(statement)
schema = Schema.from_source(DuckDBSource(connection))
text = ('PREDICT SUM(orders.price, 0, 1, days) FORECAST ' + n
        + ' TIMEFRAMES FOR EACH users.user_id')
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
start = time.monotonic()
try:
    PredictiveQuery(text, schema).materialize(
        indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01'))
    code = None
except Exception as error:
    code = getattr(error, 'code', type(error).__name__)
took = time.monotonic() - start
grew = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before
scale = 1 if sys.platform == 'darwin' else 1024
print(json.dumps({'code': code, 'seconds': took, 'grew': grew * scale}))
"""


def _routes(which: str) -> list[str]:
    from sdm_client import _routing

    return {
        'python-old': [],
        'python-core-s0': list(_routing.S0),
        'python-core': list(_routing.S0) + R.EN.core_routes({}),
    }[which]


@pytest.mark.parametrize('n', ['1000000000', '10001', '10000', '201'])
@pytest.mark.parametrize(
    'route', ['python-old', 'python-core-s0', 'python-core']
)
def test_a_forecast_of_many_steps_is_refused_at_once(
    route: str, n: str
) -> None:
    r"""Past the grammar's most (10,000) or the default ``MOST_PARTS`` (200),
    a forecast is refused ``INVALID_REQUEST`` within two seconds, its peak
    memory growing by less than 64 MiB: no step is built for a count that
    is refused. Run in a fresh interpreter, bounded by a timeout."""
    import subprocess

    done = subprocess.run(
        [
            sys.executable,
            '-c',
            _MANY_STEPS,
            str(ROOT / 'contract/tools'),
            json.dumps(_routes(route)),
            n,
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        stdin=subprocess.DEVNULL,
        cwd=ROOT,
    )
    assert done.returncode == 0, done.stderr
    measured = json.loads(done.stdout.strip().splitlines()[-1])
    assert measured['code'] == 'INVALID_REQUEST'
    assert measured['seconds'] < 2.0, measured
    assert measured['grew'] < 64 * 2**20, measured


def test_raised_most_parts_lets_steps_through_with_validation_on_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""``MOST_PARTS`` is the caller's: raised above a forecast's steps, with
    validation (S0) and every DuckDB route on the core, the forecast is
    validated and answered, one request per step; the default is no
    validation-time refusal of its own."""
    from sdm_client import decompose

    monkeypatch.setattr(decompose, 'MOST_PARTS', 250)
    vector = {
        'id': 'raised',
        'pql': 'PREDICT SUM(orders.price, 0, 1, days) FORECAST 201 TIMEFRAMES '
        'FOR EACH users.user_id',
        'anchor_time': R.ANCHOR,
        'options': {},
    }
    got = R.drive(vector, routes=_routes('python-core'))
    assert got['outcome']['result'] == 'success', got['outcome']
    assert len(got['payloads']) == 201
    assert got['result']['rows'] == 402


# --- one request id per call (T1.36 fix 21 review) ------------------------------

#: The reference's request id: ``sdm-`` and twelve hex digits, drawn fresh
#: for every call (``adapters/relational.py``, ``uuid.uuid4().hex[:12]``).
REQUEST_ID = r'sdm-[0-9a-f]{12}'


def _request_ids(routes: list[str], steps: int) -> list[Any]:
    r"""The ``metadata.request_id`` of each call a forecast of ``steps``
    steps sends to the scripted endpoint over ``routes``, in order."""
    import os

    import duckdb
    import pandas as pd
    from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
    from sdm_connectors import DuckDBSource

    serve = R._module(
        'scripted_endpoint', ROOT / 'contract/runners/scripted_endpoint.py'
    ).serve
    previous = os.environ.pop('SDM_CORE_ROUTES', None)
    if routes:
        os.environ['SDM_CORE_ROUTES'] = ','.join(routes)
    connection = duckdb.connect()
    try:
        for statement in R.EN.shop():
            connection.execute(statement)
        schema = Schema.from_source(DuckDBSource(connection))
        request = PredictiveQuery(
            f'PREDICT SUM(orders.price, 0, 7, days) FORECAST {steps} '
            'TIMEFRAMES FOR EACH users.user_id',
            schema,
        ).materialize(indices=[0, 1], anchor_time=pd.Timestamp(R.ANCHOR))
        with serve() as (url, received), SDMClient(url) as client:
            client.predict(request, model=KumoRelational())
        return [
            json.loads(body).get('metadata', {}).get('request_id')
            for _, _, body in received.requests
        ]
    finally:
        connection.close()
        os.environ.pop('SDM_CORE_ROUTES', None)
        if previous is not None:
            os.environ['SDM_CORE_ROUTES'] = previous


@pytest.mark.parametrize(
    'route', ['python-old', 'python-core-s0', 'python-core']
)
def test_each_step_of_a_forecast_is_sent_with_its_own_request_id(
    route: str,
) -> None:
    r"""A forecast of N steps is N calls, each with its own request id, well
    formed as the reference's are: the reference and python-core before the
    move draw one per call. ``python-old`` is the control."""
    import re

    ids = _request_ids(_routes(route), 3)
    assert len(ids) == 3
    assert all(
        isinstance(i, str) and re.fullmatch(REQUEST_ID, i) for i in ids
    ), ids
    assert len(set(ids)) == 3, f'one request id for several calls: {ids}'
