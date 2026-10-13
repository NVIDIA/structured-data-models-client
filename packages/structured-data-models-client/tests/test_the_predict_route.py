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
import re
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


def test_the_outputs_are_checked_on_the_core_as_on_the_old_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # From T1.28b the core checks the outputs as the client's _prepare does
    # and notes one named and left out as _report_absent does: the class,
    # code and notices are the old route's.
    request = _materialized(monkeypatch)
    with _endpoint().serve() as (url, _):
        for outputs in (
            ['prediction'],
            ['prediction', 'quantiles'],
            ['prediction', 'prediction'],
            [],
            ['bogus'],
        ):
            seen = []
            for routes in (CORE, None):
                if routes is None:
                    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
                else:
                    monkeypatch.setenv(_routing.ENVIRONMENT, routes)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    try:
                        outcome: Any = _predict(url, request, outputs=outputs)
                    except StructuredDataError as error:
                        outcome = error
                notices = sorted(
                    re.sub(r'^\[[A-Z_]+\] ', '', str(w.message))
                    for w in caught
                    if type(w.message).__name__ == 'SDMWarning'
                )
                seen.append((outcome, notices))
            (core, core_notices), (old, old_notices) = seen
            assert core_notices == old_notices, outputs
            if isinstance(old, StructuredDataError):
                assert isinstance(core, type(old)), outputs
                assert core.code == old.code, outputs
            else:
                pd.testing.assert_frame_equal(core.to_pandas(), old.to_pandas())
                assert core.outputs == old.outputs, outputs
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


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
    # The items keep the endpoint's order, as the old route's (the core's
    # rows are ordered by row_index, with each answer's place, T1.28b).
    assert core.items == old.items
    assert (core.request_id, core.metadata) == (old.request_id, old.metadata)
    frame = core.to_pandas()
    assert frame['id'].tolist() == [5, 2, 7]
    assert frame['prediction'].tolist() == [10.5, 21.0, 31.5]
    assert [item.row_index for item in core.items] == [2, 1, 0]


def test_an_overflowing_prediction_is_the_old_route_s_on_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # From T1.28b, by the owner's decision of 2026-10-12, the core reads a
    # token past a double's range as the infinity the old route reads
    # (register P1B-4).
    request = _materialized(monkeypatch)
    for token in ('1e400', '-1e400', 'NaN', 'Infinity', '-Infinity'):
        answer = (
            '{"id": "r1", "model": "kumo-relational", "metadata": {}, '
            '"predictions": [{"id": "112", "row_index": 0, "prediction": '
            + token
            + '}, {"id": "113", "row_index": 1, "prediction": 1.0}]}'
        )
        with _answering(200, answer.encode()) as (url, _):
            core, old = _both(monkeypatch, request, url)
        assert not isinstance(core, Exception), (token, core)
        pd.testing.assert_frame_equal(core.to_pandas(), old.to_pandas())


def test_a_lone_surrogate_in_an_answer_is_refused_on_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Diagnostic, register P1B-24 (proposed): Python's reader takes a
    # lone surrogate escape in text, and the old route passes it on; the
    # core's reader, which holds text as UTF-8, refuses the answer.
    request = _materialized(monkeypatch)
    answer = (
        '{"id": "r1", "model": "kumo-relational", "metadata": {"note": '
        '"\\ud800"}, "predictions": [{"id": "112", "row_index": 0, '
        '"prediction": 1.0}, {"id": "113", "row_index": 1, '
        '"prediction": 1.0}]}'
    )
    with _answering(200, answer.encode()) as (url, _):
        core, old = _both(monkeypatch, request, url)
    assert not isinstance(old, Exception), old
    assert old.metadata['note'] == '\ud800'
    assert isinstance(core, StructuredDataError)
    assert core.code == 'INVALID_RESPONSE'
    assert core.details['field'] == 'response'


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


def test_a_non_finite_number_anywhere_in_the_answer_is_the_old_route_s(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One policy over the whole answer, as the Rust host's (sdm-core's
    # tests/predict.rs, the same tokens in the metadata): from T1.28b the
    # value is passed on, as on the old route (register P1B-4).
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
        assert not isinstance(core, Exception), token
        assert not isinstance(old, Exception), token
        assert json.dumps(core.metadata) == json.dumps(old.metadata), token


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


def _same(left: Any, right: Any) -> bool:
    r"""Whether two decoded JSON values are equal, compared without
    recursion, at any depth."""
    stack = [(left, right)]
    while stack:
        a, b = stack.pop()
        if isinstance(a, dict) and isinstance(b, dict):
            if list(a) != list(b):
                return False
            stack += [(a[k], b[k]) for k in a]
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            stack += list(zip(a, b, strict=True))
        elif type(a) is not type(b) or a != b:
            return False
    return True


def test_a_refusal_with_deeply_nested_details_is_shown_as_on_the_old_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Details the transport read at a depth a recursive copy cannot take
    # (T1.11d review round 3): the old route's class, status, code and
    # details, and nothing chained.
    request = _materialized(monkeypatch)
    for depth in (600, 900):
        body = (
            '{"code": "BAD_REQUEST", "detail": "not served", "extra": '
            + '[' * depth
            + '0'
            + ']' * depth
            + '}'
        )
        with _answering(422, body.encode()) as (url, _):
            core, old = _both(monkeypatch, request, url)
        assert isinstance(core, NimRequestError), depth
        assert (type(core), core.status_code, core.code, str(core)) == (
            type(old),
            old.status_code,
            old.code,
            str(old),
        ), depth
        assert _same(core.details, old.details), depth
        assert core.__cause__ is None and core.__context__ is None


@contextmanager
def _echoing_our_credential() -> Iterator[str]:
    r"""An endpoint refusing with 401 whose detail and other fields echo
    the Authorization header it was sent and the URL's credential, beside
    a sentinel of its own; its URL carries the credential."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args: object) -> None:
            pass

        def do_POST(self) -> None:  # noqa: N802 - http.server's name
            self.rfile.read(int(self.headers.get('Content-Length') or 0))
            sent = self.headers.get('Authorization', '')
            body = json.dumps(
                {
                    'code': {'h': sent},
                    'detail': f'credential {sent} refused (ENDPOINT-ONLY-7c2)',
                    'seen': {'user': 'sdm:SECRETPW-77', 'header': [sent]},
                }
            ).encode()
            self.send_response(401)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://sdm:SECRETPW-77@127.0.0.1:{server.server_address[1]}'
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_an_endpoint_echoing_our_credential_is_shown_as_on_the_old_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # E47: the refusal exactly as the old route shows it, an echo of our
    # credential in the endpoint's own text included (a scrub of such
    # echoes was tried in fix rounds 3 and 4 and removed); nothing of the
    # transport's, and so nothing of the URL, is chained to it.
    request = _materialized(monkeypatch)
    with _echoing_our_credential() as url:
        core, old = _both(monkeypatch, request, url)
    assert isinstance(core, NimRequestError) and core.status_code == 401
    assert (type(core), core.code, str(core), repr(core)) == (
        type(old),
        old.code,
        str(old),
        repr(old),
    )
    assert core.details == old.details
    assert 'ENDPOINT-ONLY-7c2' in str(core)
    assert core.__cause__ is None and core.__context__ is None


def _emoji_request(
    monkeypatch: pytest.MonkeyPatch, emoji: int, ascii: int = 0
) -> Request:
    r"""base.regression's request on the core, one user holding a note of
    ``emoji`` four-byte characters and one order a note of ``ascii``
    one-byte ones."""
    monkeypatch.setenv(_routing.ENVIRONMENT, MATERIALIZE)
    connection = duckdb.connect()
    for statement in [
        *_shop(),
        'CREATE OR REPLACE TABLE users AS SELECT *, CASE user_id WHEN 5 '
        f"THEN repeat('\U0001f600', {emoji}) ELSE '' END AS note FROM users",
        # Order 57 (user 1's, 2024-02-03) is read by the asked entity alone,
        # after every context's anchor: its note is sent once.
        'CREATE OR REPLACE TABLE orders AS SELECT *, CASE order_id WHEN 57 '
        f"THEN repeat('a', {ascii}) ELSE '' END AS note FROM orders",
    ]:
        connection.execute(statement)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
        )
    monkeypatch.delenv(_routing.ENVIRONMENT)
    return request


def _sent(
    monkeypatch: pytest.MonkeyPatch, request: Request
) -> tuple[list[bytes], list[bytes], Any]:
    r"""The calls the core wrote, the bodies the endpoint received, and the
    refusal raised (the endpoint refuses every call)."""
    written: list[bytes] = []
    send = SDMClient._send_body

    def recording(self: Any, adapter: Any, body: bytes) -> Any:
        written.append(bytes(body))
        return send(self, adapter, body)

    monkeypatch.setattr(SDMClient, '_send_body', recording)
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    with _answering(400, b'{"code": "RECORDED"}') as (url, received):
        with pytest.raises(StructuredDataError) as refused:
            _predict(url, request)
    monkeypatch.delenv(_routing.ENVIRONMENT)
    return written, received, refused.value


def test_the_bytes_sent_are_the_core_s_and_held_to_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Register PD-2 (fix round 1, Codex P2-1): the HTTP transport sends the
    # core's call as the core wrote it, so the bytes the endpoint receives
    # are the bytes the core counted against the limit, characters past
    # ASCII as their own bytes, never json.dumps' escapes. At the limit's
    # edge the call is sent, at most the limit; one character past it, the
    # payload still within the limit (materialize takes it), the call is
    # refused before anything is sent.
    limit = 30 * 1024 * 1024
    lengths = []
    for emoji, ascii in ((0, 0), (1, 0), (0, 1)):
        written, received, _ = _sent(
            monkeypatch, _emoji_request(monkeypatch, emoji, ascii)
        )
        assert written == received
        lengths.append(len(received[0]))
    base, wide, narrow = (
        lengths[0],
        lengths[1] - lengths[0],
        lengths[2] - lengths[0],
    )
    assert wide % 4 == 0 and narrow == 1
    emoji = (limit - base) // wide
    ascii = (limit - base - emoji * wide) // narrow
    written, received, refused = _sent(
        monkeypatch, _emoji_request(monkeypatch, emoji, ascii)
    )
    assert refused.code == 'RECORDED'
    assert written == received
    assert len(received[0]) == base + emoji * wide + ascii * narrow <= limit
    assert b'\\ud83d' not in received[0]
    assert '\U0001f600'.encode() in received[0]
    written, received, past = _sent(
        monkeypatch, _emoji_request(monkeypatch, emoji, ascii + 1)
    )
    assert past.code == 'INVALID_REQUEST'
    assert written == [] and received == []
