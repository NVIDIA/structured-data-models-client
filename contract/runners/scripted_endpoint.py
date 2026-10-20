# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The scripted model endpoint, in Python (design, "Runners").

One written rule, served over localhost HTTP, which the Python runners reach
through the client's real transport. The Rust endpoint of T1.11d implements
the same rule; ``contract/scripted-endpoint.json`` holds both to it.

The rule, for ``POST /v1/predictions`` with a JSON object body:

- the answer is ``200`` with ``{"object": "prediction.response", "id":
  "r1", "model": "kumo-relational", "metadata": {}, "predictions": [...]}``,
  one prediction per row of ``predict.instance_table`` in row order:
  ``{"id": <instance id>, "row_index": <the row's position>, "prediction":
  <p>}``;
- the instance id is the row's ``instance_id`` column: an integer is given
  as its decimal digits exactly as the request spelled them, text as itself;
- ``p`` is ``1.0`` when ``task.kind`` is ``regression``, and the first of
  ``task.target.classes`` (text) for any other kind;
- a body that is not JSON, ``NaN``, ``Infinity`` and ``-Infinity``
  included (JSON has none of them; Python's reader takes them and Rust's
  refuses them, so both endpoints refuse them), is answered ``400`` with
  ``{"error": "not_json"}``;
- a request the rule cannot read otherwise (not a JSON object, an
  ``instance_table`` not in the ``arrays`` format or without an
  ``instance_id`` column, an instance id that is neither an integer nor
  text, a classification without classes or with a first class that is not
  text) is answered ``400`` with ``{"error": "<reason>"}``, the reason one
  of :data:`REASONS`;
- any other method or path is answered ``404`` with ``{"error":
  "not_found"}``.

Each request a served endpoint receives is a call, numbered from zero in
the order received (T2.1a; T2.8c's multi-call cases). An endpoint told to
fail a call answers that one ``500`` with ``{"error": "scripted_failure",
"call": <its number>}``, a status the client's transport does not retry,
and every other call by the rule; told to fail none, it answers every call
by the rule.

An endpoint told to answer probabilities (T2.8's ranking cases, whose
model answers how sure it is, as a real one does) answers a
classification call whose ``output.fields`` names ``probabilities`` with
them: for row ``i`` of call ``k``, ``d = (3k + 5i) mod 9 + 1``, the item
carries ``probabilities`` with the first class ``(10 - d) / 10``, the
last ``d / 10`` and any other ``0.0``, and its prediction is the last
class where ``d > 5``, else the first. Every other call, and every call of
an endpoint not told to, is answered by the rule above.

An endpoint told to answer quantiles (T2.9's forecast cases, as a model
answers the spread of a regression) answers a regression call whose
``output.fields`` names ``quantiles`` with them: for row ``i`` of call
``k``, with ``d`` as above, the item carries ``quantiles`` ``{"0.1": d - 1,
"0.5": d, "0.9": d + 1}``, each a float, beside the rule's prediction.
Every other call, and every call of an endpoint not told to, is answered
by the rules above.

An endpoint told to answer explanations (T2.13's cases, as a model
explains the one entity it is asked to) answers a call whose
``output.fields`` names ``explanation`` with one on each item: for row
``i`` of call ``k``, with ``d`` as above, ``{"format": "scripted",
"summary": "row <i> of call <k>", "score": d / 10, "rank": d, "path": [k,
i], "detail": {"call": k, "row_index": i, "note": null}}``, beside what
the rules above answer.

An endpoint told to answer item metadata (T2.13's cases, as a model
reports something of each answer) gives each item of every call it
answers by the rules above ``"metadata": {"call": k, "row_index": i,
"source": "scripted"}``.

An endpoint told to answer embeddings (T2.11's cases, as a model answers
the vector it holds of each entity) answers a call whose ``output.fields``
names ``embeddings`` with one on each item: for row ``i`` of call ``k``,
with ``d`` as above, ``[d / 10, (10 - d) / 10, k, i]``, each a float,
beside what the rules above answer.

Numbers are read as the tokens the request wrote, never through a float,
so an id past 2**53 is answered as it was asked.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: The path the client posts a prediction to.
PREDICTIONS = '/v1/predictions'
#: The status of the call an endpoint is told to fail.
FAILED = 500
#: Why a request the rule cannot read is refused, as the 400 body names it.
REASONS = (
    'not_json',
    'not_an_object',
    'no_instance_table',
    'not_arrays',
    'no_instance_id',
    'bad_instance_id',
    'no_classes',
)


class _Number(str):
    r"""A JSON number, kept as the token the request wrote."""


class _Refused(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _not_json(_constant: str) -> Any:
    raise _Refused('not_json')


def _read(body: bytes) -> Any:
    try:
        return json.loads(
            body,
            parse_int=_Number,
            parse_float=_Number,
            parse_constant=_not_json,
        )
    except ValueError:
        raise _Refused('not_json') from None


def _object(value: Any, reason: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _Refused(reason)
    return value


def _prediction(task: dict[str, Any]) -> Any:
    if task.get('kind') == 'regression':
        return 1.0
    target = task.get('target')
    classes = target.get('classes') if isinstance(target, dict) else None
    if (
        not isinstance(classes, list)
        or not classes
        or isinstance(classes[0], _Number)
        or not isinstance(classes[0], str)
    ):
        raise _Refused('no_classes')
    return str(classes[0])


def _instance_id(value: Any) -> str:
    if isinstance(value, _Number):
        if not value.lstrip('-').isdigit():
            raise _Refused('bad_instance_id')
        return str(value)
    if isinstance(value, str):
        return value
    raise _Refused('bad_instance_id')


def _asks(payload: dict[str, Any], field: str) -> bool:
    output = payload.get('output')
    fields = output.get('fields') if isinstance(output, dict) else None
    return isinstance(fields, list) and field in fields


def _spread(call: int, index: int) -> dict[str, float]:
    r"""Row ``index`` of call ``call``'s quantiles, as an endpoint told to
    answer quantiles gives them."""
    d = (3 * call + 5 * index) % 9 + 1
    return {'0.1': float(d - 1), '0.5': float(d), '0.9': float(d + 1)}


def _explained(call: int, index: int) -> dict[str, Any]:
    r"""Row ``index`` of call ``call``'s explanation, as an endpoint told
    to answer explanations gives it."""
    d = (3 * call + 5 * index) % 9 + 1
    return {
        'format': 'scripted',
        'summary': f'row {index} of call {call}',
        'score': d / 10,
        'rank': d,
        'path': [call, index],
        'detail': {'call': call, 'row_index': index, 'note': None},
    }


def _embedded(call: int, index: int) -> list[float]:
    r"""Row ``index`` of call ``call``'s embedding, as an endpoint told to
    answer embeddings gives it."""
    d = (3 * call + 5 * index) % 9 + 1
    return [d / 10, (10 - d) / 10, float(call), float(index)]


def _sure(task: dict[str, Any], call: int, index: int) -> tuple[Any, dict]:
    r"""Row ``index`` of call ``call``'s prediction and probabilities, as
    an endpoint told to answer probabilities gives them."""
    classes = [str(c) for c in task['target']['classes']]
    d = (3 * call + 5 * index) % 9 + 1
    given = {c: 0.0 for c in classes}
    given[classes[0]] = (10 - d) / 10
    given[classes[-1]] = d / 10
    return (classes[-1] if d > 5 else classes[0]), given


def _predictions(
    payload: Any,
    call: int = 0,
    probabilities: bool = False,
    quantiles: bool = False,
    explanation: bool = False,
    item_metadata: bool = False,
    embeddings: bool = False,
) -> list[dict[str, Any]]:
    payload = _object(payload, 'not_an_object')
    predict = payload.get('predict')
    table = predict.get('instance_table') if isinstance(predict, dict) else None
    if not isinstance(table, dict):
        raise _Refused('no_instance_table')
    columns, rows = table.get('columns'), table.get('rows')
    if (
        table.get('format') != 'arrays'
        or not isinstance(columns, list)
        or not isinstance(rows, list)
    ):
        raise _Refused('not_arrays')
    if 'instance_id' not in columns:
        raise _Refused('no_instance_id')
    at = columns.index('instance_id')
    task = payload.get('task')
    task = task if isinstance(task, dict) else {}
    p = _prediction(task)
    sure = (
        probabilities
        and task.get('kind') != 'regression'
        and _asks(payload, 'probabilities')
    )
    spread = (
        quantiles
        and task.get('kind') == 'regression'
        and _asks(payload, 'quantiles')
    )
    explains = explanation and _asks(payload, 'explanation')
    embeds = embeddings and _asks(payload, 'embeddings')
    answered = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) <= at:
            raise _Refused('not_arrays')
        item: dict[str, Any] = {
            'id': _instance_id(row[at]),
            'row_index': index,
            'prediction': p,
        }
        if sure:
            item['prediction'], item['probabilities'] = _sure(task, call, index)
        if spread:
            item['quantiles'] = _spread(call, index)
        if explains:
            item['explanation'] = _explained(call, index)
        if embeds:
            item['embeddings'] = _embedded(call, index)
        if item_metadata:
            item['metadata'] = {
                'call': call,
                'row_index': index,
                'source': 'scripted',
            }
        answered.append(item)
    return answered


def _failed(call: int) -> tuple[int, dict[str, Any]]:
    return FAILED, {'error': 'scripted_failure', 'call': call}


def answer(
    method: str,
    path: str,
    body: bytes,
    *,
    call: int = 0,
    fail_call: int | None = None,
    probabilities: bool = False,
    quantiles: bool = False,
    explanation: bool = False,
    item_metadata: bool = False,
    embeddings: bool = False,
) -> tuple[int, dict[str, Any]]:
    r"""The rule: the status and JSON body answering the request that is
    call number ``call``, the call numbered ``fail_call`` failed, with a
    classification's probabilities where ``probabilities`` says so, a
    regression's quantiles where ``quantiles`` does, an explanation where
    ``explanation`` does, each item's metadata where ``item_metadata``
    does and each item's embedding where ``embeddings`` does."""
    if call == fail_call:
        return _failed(call)
    if method != 'POST' or path != PREDICTIONS:
        return 404, {'error': 'not_found'}
    try:
        predictions = _predictions(
            _read(body),
            call,
            probabilities,
            quantiles,
            explanation,
            item_metadata,
            embeddings,
        )
    except _Refused as refused:
        return 400, {'error': refused.reason}
    return 200, {
        'object': 'prediction.response',
        'id': 'r1',
        'model': 'kumo-relational',
        'metadata': {},
        'predictions': predictions,
    }


@dataclass
class Received:
    r"""What the endpoint was sent, in order: method, path and body bytes;
    a request's position is its call's number."""

    requests: list[tuple[str, str, bytes]] = field(default_factory=list)
    fail_call: int | None = None
    probabilities: bool = False
    quantiles: bool = False
    explanation: bool = False
    item_metadata: bool = False
    embeddings: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def take(self, method: str, path: str, body: bytes) -> int:
        r"""Records a request; its call's number."""
        with self.lock:
            self.requests.append((method, path, body))
            return len(self.requests) - 1


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    # The reply's headers and body go out as two writes on a kept-alive
    # connection; with Nagle's algorithm the body waits for the client's
    # delayed acknowledgement, about 40 ms a call (T2.1l).
    disable_nagle_algorithm = True

    def log_message(self, *args: object) -> None:
        pass

    def _serve(self) -> None:
        length = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(length) if length else b''
        received: Received = self.server.received  # type: ignore[attr-defined]
        call = received.take(self.command, self.path, body)
        # The chosen call fails here, and every other is the rule's, called
        # as one request is with the call's number and whether to answer
        # probabilities and quantiles as keywords (a test may stand a rule
        # of its own in for it, taking the same arguments). Whether to
        # answer explanations, item metadata and embeddings is passed only
        # when the endpoint is told to, so a stand-in written to take the
        # keywords above still stands in for an endpoint told none of them.
        told = {
            name: True
            for name in ('explanation', 'item_metadata', 'embeddings')
            if getattr(received, name)
        }
        status, reply = (
            _failed(call)
            if call == received.fail_call
            else answer(
                self.command,
                self.path,
                body,
                call=call,
                probabilities=received.probabilities,
                quantiles=received.quantiles,
                **told,
            )
        )
        out = json.dumps(reply).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    do_GET = _serve
    do_POST = _serve
    do_PUT = _serve
    do_DELETE = _serve


@contextmanager
def serve(
    fail_call: int | None = None,
    probabilities: bool = False,
    quantiles: bool = False,
    explanation: bool = False,
    item_metadata: bool = False,
    embeddings: bool = False,
) -> Iterator[tuple[str, Received]]:
    r"""The endpoint on a free localhost port: its base URL, and what it
    receives; the call numbered ``fail_call`` is failed, a classification
    answered with its probabilities where ``probabilities`` says so, a
    regression with its quantiles where ``quantiles`` does, a call asking
    for an explanation with one where ``explanation`` does, each item with
    its metadata where ``item_metadata`` does, and a call asking for
    embeddings with one on each item where ``embeddings`` does."""
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    received = Received(
        fail_call=fail_call,
        probabilities=probabilities,
        quantiles=quantiles,
        explanation=explanation,
        item_metadata=item_metadata,
        embeddings=embeddings,
    )
    server.received = received  # type: ignore[attr-defined]
    thread = threading.Thread(
        target=lambda: server.serve_forever(poll_interval=0.01), daemon=True
    )
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_address[1]}', received
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
