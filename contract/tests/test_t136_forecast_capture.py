# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T1.36 fix 24's request ids through the runner's capture, when a forecast
step sends no call (Codex round 2, P2), held from outside the
implementation (the independent test author).

A forecast step whose instance table has no rows sends no model call
(``sdm-core``'s ``predict.rs``), so the calls a forecast makes are its
steps with rows, each carrying that step's own request id. The capture of
``contract/runners/run_python.py`` (``_Boundary``, as ``--mode
python-core`` installs it) must follow that. Over the three-step SUM
vector of ``contract/t136-forecast-vectors.json`` (``sum-days-3``), with
validation and every DuckDB route on the core, against the scripted
endpoint:

- ``test_a_step_with_no_rows_is_captured_as_it_is_answered[first|middle|
  first-and-last]``: one step's rows emptied, or two. Under the capture,
  ``predict`` answers. The calls the endpoint received are the steps with
  rows, in order, each carrying the id its step was started with (read
  from ``predict``'s start). The capture keeps the last call's body. The
  answer is the one given without the capture, and without the capture
  it is the steps with rows: two rows each. Fails at 8295049c on all
  three: ``RunnerFailure`` (the model call does not carry the request id
  predict was started with), because the capture expects one id per step
  in order and is handed a later step's call.
- ``test_the_capture_still_refuses_a_call_with_another_step_s_id``: the
  negative control. The core is started with step 3's id on step 2, or a
  stale id on step 3, behind the capture's back; the capture refuses the
  call by name, with all steps asked and with the first step empty; the
  same tampered start without the capture is answered. Passes at 8295049c
  (where the first-step-empty variants are refused for the bug as well).
- ``test_the_capture_refuses_an_id_that_is_not_the_next_step_with_rows``
  (Codex round 3, P2): the calls must carry exactly the ids of the steps
  with rows, in order. Step 3 empty with its id on step 2's call, and
  steps 1 and 3 empty with step 3's id on step 2's one call, fail at
  b02af721 (accepted); steps with rows swapped, and an empty later step's
  id on step 1, are refused there already.

The implementer changes nothing of this file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
STEPS = 3
QUESTION = (
    'PREDICT SUM(orders.price, 0, 7, days) FORECAST 3 TIMEFRAMES '
    'FOR EACH users.user_id'
)


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


RUNNER = _module('run_python', ROOT / 'contract/runners/run_python.py')
EN = _module('empty_null_record', ROOT / 'contract/tools/empty_null_record.py')
SERVE = _module(
    'scripted_endpoint', ROOT / 'contract/runners/scripted_endpoint.py'
).serve


def _vector() -> dict[str, Any]:
    vectors = json.loads(
        (ROOT / 'contract/t136-forecast-vectors.json').read_text('utf-8')
    )['vectors']
    return next(v for v in vectors if v['id'] == 'sum-days-3')


def test_the_question_is_the_vector_s() -> None:
    vector = _vector()
    assert vector['pql'] == QUESTION
    assert len(vector['reference']['payloads']) == STEPS


class _Tampered:
    r"""``sdm_client._native`` with ``predict`` started on other ids than
    the host gave: ``rewrite`` changes the start's ids after the capture
    has read them."""

    def __init__(self, native: Any, rewrite: Callable[[list[str]], None]):
        self._native = native
        self.Run = self
        self._rewrite = rewrite

    def start(self, message: bytes, attachments: Any) -> Any:
        started = json.loads(message)
        operation = started['operation']
        if operation['op'] == 'predict':
            steps = operation['state'].get('forecast') or []
            ids = [operation['request_id']] + [s['request_id'] for s in steps]
            self._rewrite(ids)
            operation['request_id'] = ids[0]
            for step, given in zip(steps, ids[1:], strict=True):
                step['request_id'] = given
            message = json.dumps(started).encode()
        return self._native.Run.start(message, attachments)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)


def _unwrapped(native: Callable[[], Any]) -> Callable[[], Any]:
    r"""``run.native`` without the capture an earlier ``_Boundary`` wrapped
    it in (each wraps the function it found, kept in its closure)."""
    while '_Boundary' in getattr(native, '__qualname__', ''):
        cells = dict(
            zip(
                native.__code__.co_freevars,
                native.__closure__ or (),
                strict=True,
            )
        )
        native = cells['imported'].cell_contents
    return native


def _ask(
    monkeypatch: pytest.MonkeyPatch,
    empty: tuple[int, ...],
    *,
    capture: bool,
    rewrite: Callable[[list[str]], None] | None = None,
) -> dict[str, Any]:
    r"""The forecast with the steps ``empty`` (from 1) given no rows, asked
    on the core with or without the runner's capture: the answer or the
    error, the ids ``predict`` was started with, the ids the endpoint
    received and the capture's record."""
    import duckdb
    import pandas as pd
    from sdm_client import (
        KumoRelational,
        PredictiveQuery,
        Schema,
        SDMClient,
        _routing,
    )
    from sdm_client._rust import run
    from sdm_connectors import DuckDBSource

    monkeypatch.setenv(
        'SDM_CORE_ROUTES', ','.join(list(_routing.S0) + EN.core_routes({}))
    )
    # Each run starts from the binding itself; whatever a capture installs
    # is undone after the test.
    monkeypatch.setattr(run, 'native', _unwrapped(run.native))
    if rewrite is not None:
        inner = run.native
        monkeypatch.setattr(run, 'native', lambda: _Tampered(inner(), rewrite))
    started: list[list[str]] = []
    recorded = None
    if capture:
        boundary = RUNNER._Boundary()
        recorded = boundary.reset()
        seen = recorded.started

        def started_and_seen(message: bytes, attachments: Any) -> None:
            operation = json.loads(message)['operation']
            if operation['op'] == 'predict':
                started.append(
                    [operation['request_id']]
                    + [
                        s['request_id']
                        for s in operation['state'].get('forecast') or ()
                    ]
                )
            seen(message, attachments)

        recorded.started = started_and_seen  # type: ignore[method-assign]
    out: dict[str, Any] = {'started': started, 'capture': recorded}
    connection = duckdb.connect()
    try:
        for statement in EN.shop():
            connection.execute(statement)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(connection))
            request = PredictiveQuery(QUESTION, schema).materialize(
                indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
            )
            for step in empty:
                part = request.parts[step - 1]
                part.payloads[0]['predict']['instance_table']['rows'] = []
            with SERVE() as (url, received), SDMClient(url) as client:
                try:
                    out['frame'] = client.predict(
                        request, model=KumoRelational()
                    ).to_pandas()
                    out['error'] = None
                except Exception as error:  # noqa: BLE001 - the outcome
                    out['frame'] = None
                    out['error'] = error
                out['received'] = [
                    json.loads(body)['metadata']['request_id']
                    for _, _, body in received.requests
                ]
    finally:
        connection.close()
    return out


@pytest.mark.parametrize(
    'empty',
    [(1,), (2,), (1, 3)],
    ids=['first', 'middle', 'first-and-last'],
)
def test_a_step_with_no_rows_is_captured_as_it_is_answered(
    monkeypatch: pytest.MonkeyPatch, empty: tuple[int, ...]
) -> None:
    plain = _ask(monkeypatch, empty, capture=False)
    assert plain['error'] is None, plain['error']
    with_rows = [s for s in range(1, STEPS + 1) if s not in empty]
    assert len(plain['received']) == len(with_rows)
    assert len(plain['frame']) == 2 * len(with_rows)
    assert sorted(set(plain['frame']['forecast_step'])) == with_rows

    captured = _ask(monkeypatch, empty, capture=True)
    assert captured['error'] is None, (
        f'{type(captured["error"]).__name__}: {captured["error"]}'
    )
    # The answer is the one given without the capture.
    assert captured['frame'].equals(plain['frame'])
    # Each call carries the id its own step was started with.
    [ids] = captured['started']
    assert len(ids) == STEPS and len(set(ids)) == STEPS
    assert captured['received'] == [ids[s - 1] for s in with_rows]
    # The capture keeps the last call's body.
    body = captured['capture'].predicted['body']
    assert body['metadata']['request_id'] == ids[with_rows[-1] - 1]


def _step_3_s_id_on_step_2(ids: list[str]) -> None:
    ids[1] = ids[2]


def _a_stale_id_on_step_3(ids: list[str]) -> None:
    ids[2] = 'sdm-000000000000'


@pytest.mark.parametrize(
    'rewrite',
    [_step_3_s_id_on_step_2, _a_stale_id_on_step_3],
    ids=['step-3-s-id-on-step-2', 'stale-id-on-step-3'],
)
@pytest.mark.parametrize('empty', [(), (1,)], ids=['all-steps', 'first-empty'])
def test_the_capture_still_refuses_a_call_with_another_step_s_id(
    monkeypatch: pytest.MonkeyPatch,
    rewrite: Callable[[list[str]], None],
    empty: tuple[int, ...],
) -> None:
    got = _ask(monkeypatch, empty, capture=True, rewrite=rewrite)
    assert isinstance(got['error'], RUNNER.RunnerFailure), got['error']
    assert 'does not carry the request id' in str(got['error'])
    # Without the capture the same tampered start is answered: the refusal
    # is the capture's.
    plain = _ask(monkeypatch, empty, capture=False, rewrite=rewrite)
    assert plain['error'] is None, plain['error']


def _steps_1_and_2_swapped(ids: list[str]) -> None:
    ids[0], ids[1] = ids[1], ids[0]


def _step_3_s_id_on_step_1(ids: list[str]) -> None:
    ids[0] = ids[2]


@pytest.mark.parametrize(
    ('empty', 'rewrite'),
    [
        # Codex round 3 (a): calls [step 1, step 3's id] while step 3 has
        # no rows, so the steps with rows are 1 and 2.
        ((3,), _step_3_s_id_on_step_2),
        # Codex round 3 (b): the one call, step 2's, carries the id of
        # step 3, which has no rows.
        ((1, 3), _step_3_s_id_on_step_2),
        # A step with rows given another such step's id, out of order.
        ((), _steps_1_and_2_swapped),
        ((3,), _steps_1_and_2_swapped),
        # A step with rows given the id of an empty step after it.
        ((2,), _step_3_s_id_on_step_1),
    ],
    ids=[
        'step-3-empty-its-id-on-step-2',
        'steps-1-and-3-empty-step-3-s-id-on-step-2',
        'steps-1-and-2-swapped',
        'step-3-empty-steps-1-and-2-swapped',
        'step-2-empty-step-3-s-id-on-step-1',
    ],
)
def test_the_capture_refuses_an_id_that_is_not_the_next_step_with_rows(
    monkeypatch: pytest.MonkeyPatch,
    empty: tuple[int, ...],
    rewrite: Callable[[list[str]], None],
) -> None:
    r"""The calls must carry exactly the ids of the steps with rows (the
    steps whose instance table has rows, as ``sdm-core``'s ``predict.rs``
    decides which steps it calls for), in step order. An id of an empty
    step, or of a step with rows out of its turn, is refused by name; the
    same tampered start without the capture is answered (Codex round 3,
    P2: at b02af721 the capture accepts any later step's id)."""
    got = _ask(monkeypatch, empty, capture=True, rewrite=rewrite)
    assert isinstance(got['error'], RUNNER.RunnerFailure), (
        f'accepted: calls carried {got["received"]}, the steps were '
        f'started with {got["started"]}, steps {empty} empty'
    )
    assert 'does not carry the request id' in str(got['error'])
    plain = _ask(monkeypatch, empty, capture=False, rewrite=rewrite)
    assert plain['error'] is None, plain['error']
