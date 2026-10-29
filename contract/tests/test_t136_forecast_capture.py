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

The independent test author maintains the producer observer and tamper setup.
The current gathered producer starts one typed ``call_relational`` at a time;
empty children have no generated request IDs or actual starts.
Separate raw legacy controls retain the original three-ID producer contract.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

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
        try:
            spec.loader.exec_module(module)
        except BaseException:
            # Optional imports can skip during module setup. A later consumer
            # must repeat that outcome, not receive an incomplete cached owner.
            if sys.modules.get(name) is module:
                del sys.modules[name]
            raise
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


PLANNED_IDS = [f'sdm-{step:012x}' for step in range(1, STEPS + 1)]


class _TamperedHandle:
    def __init__(
        self,
        handle: Any,
        rewrite_scope: str | None,
        reached: list,
        *,
        rewrite_lineage: bool = False,
    ):
        self._handle = handle
        self._rewrite_scope = rewrite_scope
        self._reached = reached
        self._rewrite_lineage = rewrite_lineage

    def take_attachment(self, attachment: int) -> Any:
        value = self._handle.take_attachment(attachment)
        if self._rewrite_scope is not None and isinstance(value, bytes):
            decoded = json.loads(value)
            if isinstance(decoded, dict) and 'scope' in decoded:
                scope = decoded['scope']
                assert scope['call_request_id'] == self._rewrite_scope
                scope['call_request_id'] = 'sdm-stale-response-scope'
                self._reached.append(('scope', self._rewrite_scope))
                value = json.dumps(decoded).encode()
        if self._rewrite_lineage and isinstance(value, bytes):
            decoded = json.loads(value)
            if isinstance(decoded, dict) and 'predicted' in decoded:
                assert decoded['lineage']
                decoded['lineage'][0]['call_request_id'] = (
                    'sdm-stale-final-lineage'
                )
                self._reached.append(('lineage', attachment))
                value = json.dumps(decoded).encode()
        return value

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


class _Tampered:
    r"""Observe actual typed starts, then tamper below the capture hook.

    Planned IDs for empty children exist only in this test's oracle.
    The actual producer allocates IDs only for children that send calls.
    """

    def __init__(
        self,
        native: Any,
        started: list[dict[str, Any]],
        rewritten: list[str],
        reached: list,
        mode: str,
    ):
        self._native = native
        self.Run = self
        self._started = started
        self._rewritten = rewritten
        self._reached = reached
        self._mode = mode

    def start(self, message: bytes, attachments: Any) -> Any:
        started = json.loads(message)
        operation = started['operation']
        rewrite_scope = None
        rewrite_lineage = False
        if operation['op'] == 'prepare_request':
            value = json.loads(attachments[operation['input']])
            rewrite_lineage = (
                self._mode == 'lineage'
                and value['stage'] == 'finish_relational_response'
                and value['input']['aggregation']['kind'] == 'steps'
            )
            if value['stage'] == 'call_relational':
                given = value['input']['request_id']
                self._started.append(copy.deepcopy(value))
                at = PLANNED_IDS.index(given)
                changed = self._rewritten[at]
                if changed != given:
                    attachments = dict(attachments)
                    if self._mode in ('input', 'both'):
                        value['input']['request_id'] = changed
                        attachments[operation['input']] = json.dumps(
                            value
                        ).encode()
                        self._reached.append(('input', given, changed))
                    if self._mode in ('body', 'both'):
                        body_id = value['input']['body']
                        body = json.loads(attachments[body_id])
                        assert body['metadata']['request_id'] == given
                        body['metadata']['request_id'] = changed
                        attachments[body_id] = json.dumps(body).encode()
                        self._reached.append(('body', given, changed))
                    if self._mode == 'scope':
                        rewrite_scope = given
        handle = self._native.Run.start(message, attachments)
        return _TamperedHandle(
            handle,
            rewrite_scope,
            self._reached,
            rewrite_lineage=rewrite_lineage,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._native, name)


def _ask(
    monkeypatch: pytest.MonkeyPatch,
    empty: tuple[int, ...],
    *,
    capture: bool,
    rewrite: Callable[[list[str]], None] | None = None,
    mode: str = 'both',
    model: Any = None,
    outputs: Any = None,
    fail_call: int | None = None,
) -> dict[str, Any]:
    r"""Ask the real forecast and observe only actual typed call starts.

    Empty-step planned IDs are test-only reserved values, never captured
    as starts. The below-capture tamper records each actual changed boundary.
    """
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
    started: list[dict[str, Any]] = []
    reached: list = []
    rewritten = PLANNED_IDS.copy()
    if rewrite is not None:
        rewrite(rewritten)
    inner = run.native
    original_carried = run._carried
    handovers: list[tuple[Any, Any]] = []

    def carried(error, raised):
        result = original_carried(error, raised)
        handovers.append((error, raised))
        return result

    monkeypatch.setattr(run, '_carried', carried)
    monkeypatch.setattr(
        run,
        'native',
        lambda: _Tampered(inner(), started, rewritten, reached, mode),
    )
    recorded = None
    boundary = None
    if capture:
        boundary = RUNNER._Boundary()
        recorded = boundary.reset()
    out: dict[str, Any] = {
        'started': started,
        'capture': recorded,
        'tampered': reached,
        'planned': PLANNED_IDS.copy(),
        'handovers': handovers,
    }
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
            with (
                SERVE(fail_call=fail_call) as (url, received),
                SDMClient(url) as client,
            ):
                ids = [
                    value
                    for step, value in enumerate(PLANNED_IDS, 1)
                    if step not in empty
                ]
                try:
                    with patch(
                        'sdm_client.adapters.relational._request_id',
                        side_effect=ids,
                    ):
                        answer = client.predict(
                            request,
                            model=KumoRelational() if model is None else model,
                            outputs=outputs,
                        )
                    out['frame'] = answer.to_pandas()
                    out['metadata'] = copy.deepcopy(answer.metadata)
                    out['items'] = copy.deepcopy(answer.items)
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
        if boundary is not None:
            boundary.close()
        monkeypatch.setattr(run, 'native', inner)
        monkeypatch.setattr(run, '_carried', original_carried)
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
    ids = [value['input']['request_id'] for value in captured['started']]
    assert ids == [PLANNED_IDS[s - 1] for s in with_rows]
    assert len(set(ids)) == len(with_rows)
    assert not set(ids).intersection(PLANNED_IDS[s - 1] for s in empty)
    assert captured['received'] == ids
    assert captured['metadata'] == plain['metadata']
    assert captured['items'] == plain['items']
    assert 'predict' not in captured['capture'].operations
    # The capture keeps the last call's body.
    body = captured['capture'].predicted['body']
    assert body['metadata']['request_id'] == ids[-1]
    if len(ids) > 1:
        assert [
            call['metadata']['request_id']
            for call in captured['capture'].predicted['calls']
        ] == ids
    inputs = [
        item['input']['input']['request_id']
        for item in captured['capture'].inputs
        if item['op'] == 'prepare_request'
        and item['input']['stage'] == 'call_relational'
    ]
    assert inputs == ids


def _step_3_s_id_on_step_2(ids: list[str]) -> None:
    # Keep native call IDs unique so only the capture's step binding refuses.
    ids[1], ids[2] = ids[2], ids[1]


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
    control = _ask(monkeypatch, empty, capture=True)
    assert control['error'] is None, control['error']
    got = _ask(monkeypatch, empty, capture=True, rewrite=rewrite)
    assert got['tampered'], 'the rewrite never reached an actual call start'
    assert isinstance(got['error'], RUNNER.RunnerFailure), got['error']
    assert 'does not carry the request id' in str(got['error'])
    # Without the capture the same tampered start is answered: the refusal
    # is the capture's.
    plain = _ask(monkeypatch, empty, capture=False, rewrite=rewrite)
    assert plain['error'] is None, plain['error']


def _steps_1_and_2_swapped(ids: list[str]) -> None:
    ids[0], ids[1] = ids[1], ids[0]


def _step_3_s_id_on_step_1(ids: list[str]) -> None:
    ids[0], ids[2] = ids[2], ids[0]


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
    control = _ask(monkeypatch, empty, capture=True)
    assert control['error'] is None, control['error']
    got = _ask(monkeypatch, empty, capture=True, rewrite=rewrite)
    assert got['tampered'], 'the rewrite never reached an actual call start'
    assert isinstance(got['error'], RUNNER.RunnerFailure), (
        f'accepted: calls carried {got["received"]}, the steps were '
        f'started with {got["started"]}, steps {empty} empty'
    )
    assert 'does not carry the request id' in str(got['error'])
    plain = _ask(monkeypatch, empty, capture=False, rewrite=rewrite)
    assert plain['error'] is None, plain['error']


@pytest.mark.parametrize('mode', ['input', 'body', 'scope'])
def test_typed_input_body_and_scope_ids_are_independent_obligations(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    control = _ask(monkeypatch, (), capture=True)
    assert control['error'] is None, control['error']
    got = _ask(
        monkeypatch,
        (),
        capture=True,
        rewrite=_a_stale_id_on_step_3,
        mode=mode,
    )
    assert got['tampered']
    assert {change[0] for change in got['tampered']} == {mode}
    assert isinstance(got['error'], RUNNER.RunnerFailure), got['error']
    assert 'request id' in str(got['error']).lower()
