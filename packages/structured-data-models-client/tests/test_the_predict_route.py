# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``SDMClient.predict`` on the core, of a request materialized on the core
over a DuckDB source (T1.11d).

Forced onto the core (``SDM_CORE_ROUTES``, with the schema's and the
materialize route), the call's model request is written and its answer
read on the core's ``predict``, sent through the client's own transport:
the result is the one the old route gives for the same request in the same
process, from the same answer; the request state comes back at its own
version, and another version is refused by name; the endpoint's refusal is
the transport's own error, as on the old route, and an endpoint that cannot
be reached is ``TRANSPORT_ERROR`` raised with no cause; an answer that is
not this request's is ``INVALID_RESPONSE``, as on the old route. What the
core does not take yet is refused by name. The reference comparison is
``contract/tests``'s.
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Request,
    Schema,
    SDMClient,
    StructuredDataError,
    _routing,
)
from sdm_client.errors import NimRequestError

ROOT = Path(__file__).resolve().parents[3]
MATERIALIZE = ','.join(
    _routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB
)
CORE = ','.join(
    _routing.FROM_SOURCE_DUCKDB
    + _routing.MATERIALIZE_DUCKDB
    + _routing.PREDICT_DUCKDB
)
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'


def _endpoint() -> Any:
    name = 'scripted_endpoint'
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, ROOT / 'contract/runners/scripted_endpoint.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


def _shop() -> list[str]:
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


def _materialized(
    monkeypatch: pytest.MonkeyPatch, indices: list[int] | None = None
) -> Request:
    r"""base.regression's request, materialized on the core, for
    ``indices`` (its own, ``[0, 1]``, by default)."""
    monkeypatch.setenv(_routing.ENVIRONMENT, MATERIALIZE)
    connection = duckdb.connect()
    for statement in _shop():
        connection.execute(statement)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1] if indices is None else indices,
            anchor_time=pd.Timestamp('2024-03-01'),
        )
    monkeypatch.delenv(_routing.ENVIRONMENT)
    return request


@contextmanager
def _answering(status: int, body: bytes) -> Iterator[tuple[str, list[bytes]]]:
    r"""An endpoint answering every request with ``status`` and ``body``:
    its URL, and the bodies it received."""
    received: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            length = int(self.headers.get('Content-Length') or 0)
            received.append(self.rfile.read(length))
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}', received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _predict(url: str, request: Request, **arguments: Any) -> Any:
    with SDMClient(url, max_retries=0) as client:
        return client.predict(request, model=KumoRelational(), **arguments)


def _both(
    monkeypatch: pytest.MonkeyPatch,
    request: Request,
    url: str,
    **arguments: Any,
) -> tuple[Any, Any]:
    r"""The call's outcome on the core and on the old route: the result,
    or the exception raised."""
    outcomes = []
    for routes in (CORE, None):
        if routes is None:
            monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
        else:
            monkeypatch.setenv(_routing.ENVIRONMENT, routes)
        try:
            outcomes.append(_predict(url, request, **arguments))
        except StructuredDataError as error:
            outcomes.append(error)
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    return outcomes[0], outcomes[1]


def test_the_route_takes_a_duckdb_request_s_predict_paths_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert set(_routing.PREDICT_DUCKDB) <= _routing.CORE_ROUTED
    assert len(_routing.PREDICT_DUCKDB) == 6
    request = _materialized(monkeypatch)
    monkeypatch.setenv(
        _routing.ENVIRONMENT, MATERIALIZE + ',' + _routing.PREDICT_DUCKDB[2]
    )
    with pytest.raises(_routing.RouteError):
        _predict('http://127.0.0.1:9', request)


def test_the_result_on_the_core_is_the_old_route_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _materialized(monkeypatch)
    assert request._core_state == {'version': 1}
    with _endpoint().serve() as (url, received):
        core, old = _both(monkeypatch, request, url)
    assert not isinstance(core, Exception), core
    pd.testing.assert_frame_equal(core.to_pandas(), old.to_pandas())
    assert list(core.to_pandas().columns) == ['row_index', 'id', 'prediction']
    assert len(core) == 2
    assert [item.id for item in core.items] == [0, 1]
    assert (core.model, core.request_id, core.metadata, core.outputs) == (
        old.model,
        old.request_id,
        old.metadata,
        old.outputs,
    )
    assert core.items == old.items
    # The same request reached the endpoint from both routes, but for the
    # id each call draws.
    sent = [json.loads(body) for _, _, body in received.requests]
    assert len(sent) == 2
    ids = [body['metadata'].pop('request_id') for body in sent]
    assert all(isinstance(i, str) and i.startswith('sdm-') for i in ids)
    assert sent[0] == sent[1]


def test_another_request_state_version_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _materialized(monkeypatch)
    request._core_state = {'version': 2}
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with _endpoint().serve() as (url, received):
        with pytest.raises(StructuredDataError) as refused:
            _predict(url, request)
    assert refused.value.code == 'INVALID_REQUEST'
    assert refused.value.details['field'] == 'state.version'
    assert refused.value.details['version'] == 2
    assert received.requests == []


def test_a_request_the_core_did_not_materialize_has_no_core_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _materialized(monkeypatch)
    del request._core_state
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with pytest.raises(_routing.RouteError):
        _predict('http://127.0.0.1:9', request)


def test_outputs_other_than_a_regression_s_default_are_refused_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _materialized(monkeypatch)
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with _endpoint().serve() as (url, received):
        # The default, named, is taken.
        assert len(_predict(url, request, outputs=['prediction'])) == 2
        with pytest.raises(StructuredDataError) as refused:
            _predict(url, request, outputs=['prediction', 'quantiles'])
    assert refused.value.code == 'UNSUPPORTED_REQUEST'
    assert refused.value.details == {'operation': 'predict', 'rule': 'outputs'}
    assert len(received.requests) == 1


def test_an_endpoint_s_refusal_is_shown_as_on_the_old_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The endpoint's own text, its code, detail, title and other fields and
    # the client's diagnosis of it, as the old route shows them (decision
    # E47, owner decision 2026-10-10); nothing chained to it.
    request = _materialized(monkeypatch)
    for status, answer in (
        (422, {'code': 'UNSUPPORTED_FEATURE', 'detail': 'column x is 5'}),
        (
            500,
            {
                'code': 'INTERNAL_ERROR',
                'detail': 'failed to collect any context examples for 7',
                'extra': 'value 7',
            },
        ),
        (400, {'title': 'bad request for 7', 'endpoint_code': 'E9'}),
    ):
        with _answering(status, json.dumps(answer).encode()) as (url, _):
            core, old = _both(monkeypatch, request, url)
        assert isinstance(core, NimRequestError)
        assert (type(core), core.status_code, core.code, str(core)) == (
            type(old),
            old.status_code,
            old.code,
            str(old),
        )
        assert core.details == old.details
        assert core.__cause__ is None and core.__context__ is None


def test_an_endpoint_that_cannot_be_reached_is_a_transport_error_with_no_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _materialized(monkeypatch)
    with socket.socket() as closed:
        closed.bind(('127.0.0.1', 0))
        port = closed.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    core, old = _both(monkeypatch, request, url)
    assert (type(core), core.code) == (type(old), old.code)
    assert core.code == 'TRANSPORT_ERROR'
    assert core.details['kind'] == 'model_endpoint'
    assert core.__cause__ is None and core.__suppress_context__


def test_an_answer_that_is_not_this_request_s_is_an_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _materialized(monkeypatch)
    for answered in (
        # A row that was not asked about.
        {
            'id': 'r1',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {'id': '112', 'row_index': 0, 'prediction': 1.0},
                {'id': '113', 'row_index': 5, 'prediction': 1.0},
            ],
        },
        # An answer under another row's instance id.
        {
            'id': 'r1',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {'id': '113', 'row_index': 0, 'prediction': 1.0},
                {'id': '112', 'row_index': 1, 'prediction': 1.0},
            ],
        },
        # Not the contract's: no metadata.
        {'id': 'r1', 'model': 'kumo-relational', 'predictions': []},
    ):
        with _answering(200, json.dumps(answered).encode()) as (url, _):
            core, old = _both(monkeypatch, request, url)
        assert (type(core), core.code) == (type(old), old.code), answered
        assert core.code == 'INVALID_RESPONSE'


def test_distinct_answers_out_of_order_land_on_their_own_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Entities that are not their rows' positions, answered in reverse
    # order with a distinct prediction each: a swapped column or a misplaced
    # prediction differs from the old route (T1.11d review round 1, F4).
    request = _materialized(monkeypatch, indices=[5, 2, 7])
    [rows] = [request.payloads[0]['predict']['instance_table']['rows']]
    instances = [row[0] for row in rows]
    answer = {
        'id': 'r7',
        'model': 'kumo-relational',
        'metadata': {'n': 12345678901234567890, 'tiny': 1e-05},
        'predictions': [
            {
                'id': str(instance),
                'row_index': index,
                'prediction': 10.5 * (index + 1),
            }
            for index, instance in reversed(list(enumerate(instances)))
        ],
    }
    with _answering(200, json.dumps(answer).encode()) as (url, _):
        core, old = _both(monkeypatch, request, url)
    assert not isinstance(core, Exception), core
    pd.testing.assert_frame_equal(core.to_pandas(), old.to_pandas())
    assert core.items == old.items
    assert (core.request_id, core.metadata) == (old.request_id, old.metadata)
    frame = core.to_pandas()
    assert frame['id'].tolist() == [5, 2, 7]
    assert frame['prediction'].tolist() == [10.5, 21.0, 31.5]
    assert [item.row_index for item in core.items] == [2, 1, 0]


def test_an_overflowing_prediction_is_refused_on_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _materialized(monkeypatch)
    for token in ('1e400', '-1e400'):
        answer = (
            '{"id": "r1", "model": "kumo-relational", "metadata": {}, '
            '"predictions": [{"id": "112", "row_index": 0, "prediction": '
            + token
            + '}, {"id": "113", "row_index": 1, "prediction": 1.0}]}'
        )
        with _answering(200, answer.encode()) as (url, _):
            core, old = _both(monkeypatch, request, url)
        # The core refuses it as the Rust host does (sdm-core's
        # tests/predict.rs), where the old route gives an infinity
        # (register P1B-4).
        assert isinstance(core, StructuredDataError)
        assert core.code == 'INVALID_RESPONSE'
        assert core.details['field'] == 'response'
        assert core.details['reason'] == 'not_finite'
        assert old.to_pandas()['prediction'].tolist()[0] in (
            float('inf'),
            float('-inf'),
        )


def test_a_failed_transport_is_named_for_what_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A 200 whose body is not JSON: the reference's TRANSPORT_ERROR, said
    # as a transport failure, not as an endpoint never reached.
    request = _materialized(monkeypatch)
    with _answering(200, b'<html>' + b'x' * 10 + b'</html>') as (url, _):
        core, old = _both(monkeypatch, request, url)
    assert (core.code, old.code) == ('TRANSPORT_ERROR', 'TRANSPORT_ERROR')
    assert 'its answer could not be read' in str(core)


def test_a_non_finite_number_anywhere_in_the_answer_is_refused_on_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One policy over the whole answer, as the Rust host's (sdm-core's
    # tests/predict.rs, the same tokens in the metadata): the old route
    # passes the value on (register P1B-4).
    request = _materialized(monkeypatch)
    for token in ('1e400', '-1e400', 'Infinity', '-Infinity', 'NaN'):
        answer = (
            '{"id": "r1", "model": "kumo-relational", "metadata": {"x": [1, '
            '{"y": ' + token + '}]}, "predictions": [{"id": "112", '
            '"row_index": 0, "prediction": 1.0}, {"id": "113", '
            '"row_index": 1, "prediction": 1.0}]}'
        )
        with _answering(200, answer.encode()) as (url, _):
            core, old = _both(monkeypatch, request, url)
        assert isinstance(core, StructuredDataError), token
        assert core.code == 'INVALID_RESPONSE', token
        assert (core.details['field'], core.details['reason']) == (
            'response',
            'not_finite',
        ), token
        assert not isinstance(old, Exception), token


def test_an_answer_nested_past_the_core_s_depth_is_refused_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Python's reader takes nesting the core refuses: the host writes it
    # without recursion, and the core refuses it as too deep. 126 arrays in
    # the metadata nest 128 deep in all, the deepest the core reads. Past
    # what Python's own reader takes (about a thousand levels on 3.10 and
    # 3.11), the transport refuses the answer itself, TRANSPORT_ERROR, as
    # on the old route; either way nothing raises a bare RecursionError.
    request = _materialized(monkeypatch)
    for depth in (126, 300, 960, 5000):
        answer = (
            '{"id": "r1", "model": "kumo-relational", "metadata": {"x": '
            + '[' * depth
            + '0'
            + ']' * depth
            + '}, "predictions": [{"id": "112", "row_index": 0, '
            '"prediction": 1.0}, {"id": "113", "row_index": 1, '
            '"prediction": 1.0}]}'
        )
        with _answering(200, answer.encode()) as (url, _):
            core, old = _both(monkeypatch, request, url)
        if depth == 126:
            assert len(core) == 2
            continue
        assert isinstance(core, StructuredDataError), depth
        if isinstance(old, StructuredDataError):
            # The transport could not read it, on either route.
            assert (core.code, old.code) == (
                'TRANSPORT_ERROR',
                'TRANSPORT_ERROR',
            )
        else:
            assert core.code == 'INVALID_RESPONSE', depth
            assert core.details['reason'] == 'too_deep', depth
        assert depth != 300 or core.code == 'INVALID_RESPONSE'


def test_a_payload_json_cannot_hold_is_refused_as_the_transport_refuses_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _materialized(monkeypatch)
    request.payloads[0]['context']['instance_table']['rows'][0][-1] = float(
        'nan'
    )
    with _endpoint().serve() as (url, received):
        core, old = _both(monkeypatch, request, url)
    assert (type(core), core.code) == (type(old), old.code)
    assert core.code == 'INVALID_REQUEST'
    assert core.__cause__ is None and core.__context__ is None
    assert received.requests == []
