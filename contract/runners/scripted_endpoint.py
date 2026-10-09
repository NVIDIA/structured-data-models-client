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
- a request the rule cannot read (not a JSON object, an
  ``instance_table`` not in the ``arrays`` format or without an
  ``instance_id`` column, an instance id that is neither an integer nor
  text, a classification without classes or with a first class that is not
  text) is answered ``400`` with ``{"error": "<reason>"}``, the reason one
  of :data:`REASONS`;
- any other method or path is answered ``404`` with ``{"error":
  "not_found"}``.

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


def _read(body: bytes) -> Any:
    try:
        return json.loads(body, parse_int=_Number, parse_float=_Number)
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


def _predictions(payload: Any) -> list[dict[str, Any]]:
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
    p = _prediction(task if isinstance(task, dict) else {})
    answered = []
    for index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) <= at:
            raise _Refused('not_arrays')
        answered.append(
            {
                'id': _instance_id(row[at]),
                'row_index': index,
                'prediction': p,
            }
        )
    return answered


def answer(method: str, path: str, body: bytes) -> tuple[int, dict[str, Any]]:
    r"""The rule: the status and JSON body answering one request."""
    if method != 'POST' or path != PREDICTIONS:
        return 404, {'error': 'not_found'}
    try:
        predictions = _predictions(_read(body))
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
    r"""What the endpoint was sent, in order: method, path and body bytes."""

    requests: list[tuple[str, str, bytes]] = field(default_factory=list)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args: object) -> None:
        pass

    def _serve(self) -> None:
        length = int(self.headers.get('Content-Length') or 0)
        body = self.rfile.read(length) if length else b''
        received: Received = self.server.received  # type: ignore[attr-defined]
        received.requests.append((self.command, self.path, body))
        status, reply = answer(self.command, self.path, body)
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
def serve() -> Iterator[tuple[str, Received]]:
    r"""The endpoint on a free localhost port: its base URL, and what it
    receives."""
    server = ThreadingHTTPServer(('127.0.0.1', 0), _Handler)
    received = Received()
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
