# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A stand-in NIM that answers every payload with made-up predictions.

No recorded NIM responses exist, and none are needed to cover how an answer
is decoded: what matters is that each payload gets a well-formed answer that
depends only on the payload. Each prediction is drawn from a hash of the
question the payload asks (its ``metadata.source_query``) and the row's id,
so the same payload always gets the same answer, and the candidates of a
ranking and the steps of a forecast, each asked as its own question, get
different ones.

Each answer is stored in the transcript (``responses``), so a replay answers
from what was stored and no other language has to compute it.

It stands in for the HTTP session inside the client's real transport, so the
payload is captured as the bytes the transport would have sent, and the
answer is read by the transport's own decoding.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from typing import Any

MODEL_NAMES = ('kumo-tabular', 'kumo-relational')

PREDICTIONS = '/v1/predictions'
SESSIONS = '/v1/sessions'
#: What follows a session's id on a scoring call.
SCORING = '/predictions'


def unit(*parts: Any) -> float:
    r"""A number in [0, 1) from a SHA-256 of the parts, joined by ``|``."""
    text = '|'.join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode('utf-8')).digest()
    return int.from_bytes(digest[:8], 'big') / 2.0**64


def _label(value: Any) -> str:
    r"""A class label as an endpoint spells it: ``True``, ``3``, ``books``."""
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _classes(payload: dict[str, Any]) -> list[str]:
    r"""The target's classes, as the context's labelled rows hold them."""
    target = payload['task']['target']['column_name']
    table = payload['context']['instance_table']
    at = table['columns'].index(target)
    seen = {_label(row[at]) for row in table['rows'] if row[at] is not None}
    return sorted(seen)


def _targets(payload: dict[str, Any]) -> list[str]:
    r"""The target entities a link prediction can rank: those the context's
    labels name, as an endpoint spells an id."""
    target = payload['task']['target']['column_name']
    table = payload['context']['instance_table']
    at = table['columns'].index(target)
    seen = {
        _label(value)
        for row in table['rows']
        if isinstance(row[at], list)
        for value in row[at]
        if value is not None
    }
    return sorted(seen)


def _rankings(
    payload: dict[str, Any], seed: tuple[Any, ...]
) -> list[dict[str, Any]]:
    r"""A link prediction's ranking of the target entities the context
    names, best first, cut at the task's ``top_k`` when it gives one."""
    scored = [
        {'id': target, 'score': round(unit(*seed, 'rank', target), 6)}
        for target in _targets(payload)
    ]
    scored.sort(key=lambda entry: (-entry['score'], entry['id']))
    top_k = payload['task'].get('top_k')
    return scored if top_k is None else scored[:top_k]


def _row_ids(payload: dict[str, Any]) -> list[str | None]:
    r"""The id each predicted row is answered under.

    The relational endpoint answers under ``instance_id``; the tabular one
    under the stringified list of the primary key's values.
    """
    table = payload['predict']['instance_table']
    columns, rows = table['columns'], table['rows']
    if 'instance_id' in columns:
        at = columns.index('instance_id')
        return [str(row[at]) for row in rows]
    key = payload.get('schema', {}).get('instance_table', {}).get('primary_key')
    if key and all(name in columns for name in key):
        return [str([row[columns.index(name)] for name in key]) for row in rows]
    return [None] * len(rows)


def _question(payload: Mapping[str, Any], quirks: Mapping[str, Any]) -> tuple:
    r"""What every draw for this payload is seeded with besides the row.

    The question the payload asks, when it says (a relational payload's
    ``metadata.source_query``), so each candidate of a ranking and each step
    of a forecast is answered differently. The quirk ``seed: row`` leaves it
    out, as version 1 of this stand-in did, so every part answers alike.
    """
    if quirks.get('seed') == 'row':
        return ()
    asked = (payload.get('metadata') or {}).get('source_query')
    return (asked,) if asked else ()


def answer(
    payload: dict[str, Any],
    quirks: Mapping[str, Any] | None = None,
    omitted: tuple[str, ...] = (),
) -> dict[str, Any]:
    r"""The response to one prediction payload.

    ``quirks`` makes the answer misbehave the way a scenario needs: ``omit``
    names fields left out of every item, and ``seed: row`` seeds by the row
    alone (see :func:`_question`). ``omitted`` names more fields to leave
    out of this answer alone.
    """
    quirks = quirks or {}
    kind = payload['task']['kind']
    fields = payload.get('output', {}).get('fields') or ['prediction']
    omitted = set(quirks.get('omit') or ()) | set(omitted)
    salt = _question(payload, quirks)
    classes = _classes(payload) if 'classification' in kind else []
    items = []
    for index, row_id in enumerate(_row_ids(payload)):
        seed = (*salt, row_id if row_id is not None else index)
        item: dict[str, Any] = {'row_index': index}
        if row_id is not None:
            item['id'] = row_id
        if classes:
            weights = [1.0 + unit(*seed, 'class', label) for label in classes]
            total = sum(weights)
            probabilities = {
                label: round(weight / total, 6)
                for label, weight in zip(classes, weights, strict=True)
            }
            best = max(classes, key=lambda label: probabilities[label])
            if 'prediction' in fields:
                item['prediction'] = best
            if 'probabilities' in fields:
                item['probabilities'] = probabilities
        elif kind == 'regression':
            if 'prediction' in fields:
                item['prediction'] = round(100.0 * unit(*seed, 'value'), 6)
        elif kind == 'temporal_link_prediction':
            if 'rankings' in fields:
                item['rankings'] = _rankings(payload, seed)
        else:
            raise ValueError(f'the synthetic NIM does not answer {kind!r}')
        if 'scores' in fields:
            item['scores'] = [round(unit(*seed, 'score'), 6)]
        if 'embeddings' in fields:
            item['embeddings'] = [
                round(unit(*seed, 'embedding', at), 6) for at in range(3)
            ]
        if 'explanation' in fields:
            item['explanation'] = {
                'synthetic': True,
                'weight': round(unit(*seed, 'explanation'), 6),
            }
        for name in omitted:
            item.pop(name, None)
        items.append(item)
    return {
        'object': 'prediction.response',
        'id': f'synthetic-{len(items)}',
        'model': payload.get('model', ''),
        'metadata': {'synthetic': True},
        'predictions': items,
    }


class _Response:
    def __init__(self, status: int, body: Any) -> None:
        self.status_code = status
        self.headers = {'Content-Type': 'application/json'}
        self._body = b'' if body is None else json.dumps(body).encode('utf-8')

    def iter_content(self, chunk_size: int = 1) -> Iterator[bytes]:
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class Divergence(BaseException):
    r"""The run sent the model a request the recording did not answer.

    A ``BaseException``, like the database tape's, so client code catching
    ``Exception`` around a call cannot swallow it.
    """


def _path(url: str) -> str:
    r"""The path of a transport URL: what follows the stand-in's host."""
    at = url.find('/v1/')
    return url[at:] if at >= 0 else url


#: What the one normalized field of a request body is set to.
REQUEST_ID = '<request-id>'


def normalized(body: dict[str, Any]) -> dict[str, Any]:
    r"""A request body with its ``metadata.request_id`` set to
    :data:`REQUEST_ID`, the one field that differs from run to run."""
    metadata = body.get('metadata')
    if isinstance(metadata, dict) and 'request_id' in metadata:
        metadata['request_id'] = REQUEST_ID
    return body


def _entry(
    method: str, path: str, status: int, body: Any, request: Any
) -> dict[str, Any]:
    r"""One stored response; a session call's body is kept with it."""
    entry: dict[str, Any] = {'method': method, 'path': path}
    if request is not None and path != PREDICTIONS:
        entry['request'] = normalized(json.loads(json.dumps(request)))
    entry['status'] = status
    entry['body'] = json.loads(json.dumps(body))
    return entry


class SyntheticSession:
    r"""What the client's transport posts through, answering from :func:`answer`.

    ``payloads`` are the prediction request bodies (``POST /v1/predictions``),
    parsed, in the order they were sent. ``responses`` are the answers, one
    per request of any kind, in order, each ``{"method", "path", "status",
    "body"}``, with the body a session call sent under ``request``.

    It serves predictions and the session routes: ``POST /v1/sessions`` pins
    the payload under a new id, a scoring call is answered as the pinned
    payload with the call's own sections over it, and ``DELETE`` releases it.
    ``quirks`` are what :func:`answer` takes, ``omit_at``: a list of
    ``{"at": N, "fields": [...]}``, leaving those fields out of the answer to
    request ``N`` alone (0-based, counting every request), ``items_at``: a
    list of ``{"at": N, "items": [...]}`` rewriting the items of the answer
    to request ``N`` (see :meth:`_items`), and ``fail``: a list of
    ``{"at": N, "status": S, "body": B}``, answering request ``N`` with ``S``
    and ``B`` instead.
    """

    def __init__(self, quirks: Mapping[str, Any] | None = None) -> None:
        self.quirks = dict(quirks or {})
        self.payloads: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self._pinned: dict[str, dict[str, Any]] = {}

    def _failure(self) -> dict[str, Any] | None:
        for failure in self.quirks.get('fail') or ():
            if failure.get('at') == len(self.responses):
                return failure
        return None

    def _omitted(self) -> tuple[str, ...]:
        r"""The fields ``omit_at`` leaves out of the next answer."""
        return tuple(
            name
            for entry in self.quirks.get('omit_at') or ()
            if entry.get('at') == len(self.responses)
            for name in entry.get('fields') or ()
        )

    def _items(self, body: dict[str, Any]) -> dict[str, Any]:
        r"""``body`` with its items rewritten as ``items_at`` says for the
        next answer: each ``{"from": i, "set": {...}, "drop": [...]}`` is item
        ``i`` of the answer with those fields set and dropped."""
        for entry in self.quirks.get('items_at') or ():
            if entry.get('at') != len(self.responses):
                continue
            made = []
            for spec in entry.get('items') or ():
                item = dict(body['predictions'][spec['from']])
                item.update(spec.get('set') or {})
                for name in spec.get('drop') or ():
                    item.pop(name, None)
                made.append(item)
            body = {**body, 'predictions': made}
        return body

    def _answer(self, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
        if path == PREDICTIONS:
            return 200, self._items(
                answer(payload, self.quirks, self._omitted())
            )
        if path == SESSIONS:
            session_id = f'synthetic-session-{len(self._pinned) + 1}'
            self._pinned[session_id] = payload
            return 200, {'session_id': session_id}
        if path.startswith(SESSIONS + '/') and path.endswith(SCORING):
            session_id = path[len(SESSIONS) + 1 : -len(SCORING)]
            pinned = self._pinned.get(session_id)
            if pinned is None:
                return 404, {'title': 'Not Found', 'status': 404}
            return 200, self._items(
                answer({**pinned, **payload}, self.quirks, self._omitted())
            )
        raise ValueError(f'the synthetic NIM does not serve {path}')

    def _serve(
        self, method: str, url: str, payload: dict[str, Any] | None
    ) -> _Response:
        path = _path(url)
        failure = self._failure()
        if failure is not None:
            status, body = int(failure['status']), failure.get('body')
        elif payload is not None:
            status, body = self._answer(path, payload)
        else:
            self._pinned.pop(path[len(SESSIONS) + 1 :], None)
            status, body = 204, None
        self.responses.append(_entry(method, path, status, body, payload))
        return _Response(status, body)

    def post(self, url: str, data: bytes, **kwargs: Any) -> _Response:
        payload = json.loads(data)
        if _path(url) == PREDICTIONS:
            self.payloads.append(payload)
        return self._serve('POST', url, payload)

    def delete(self, url: str, **kwargs: Any) -> _Response:
        return self._serve('DELETE', url, None)

    def close(self) -> None:
        return None


class ReplayingSession:
    r"""Answers each request with the next stored response.

    The request must be the one the response was recorded for: the same
    method and path, and for a session call the same body, but for its
    ``metadata.request_id``. Anything else is a :class:`Divergence`, noted on
    ``divergence``. ``payloads`` are the prediction bodies sent, as on a
    recording, and compared with the stored ones afterwards.
    """

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._stored = list(responses)
        self.payloads: list[dict[str, Any]] = []
        self.responses: list[dict[str, Any]] = []
        self.divergence: Divergence | None = None

    def _diverge(self, message: str) -> Divergence:
        self.divergence = Divergence(
            f'response {len(self.responses)}: {message}'
        )
        return self.divergence

    def _serve(
        self, method: str, url: str, payload: dict[str, Any] | None = None
    ) -> _Response:
        path = _path(url)
        if len(self.responses) >= len(self._stored):
            raise self._diverge(
                f'the run sent {method} {path}, and the recording holds only '
                f'{len(self._stored)} responses'
            )
        stored = self._stored[len(self.responses)]
        if (stored.get('method'), stored.get('path')) != (method, path):
            raise self._diverge(
                f'the run sent {method} {path}, and the recording answered '
                f'{stored.get("method")} {stored.get("path")}'
            )
        if 'request' in stored and stored['request'] != normalized(
            json.loads(json.dumps(payload))
        ):
            raise self._diverge(
                f'the run sent {method} {path} with another body than the '
                f'recorded one'
            )
        self.responses.append(stored)
        return _Response(int(stored['status']), stored.get('body'))

    def leftover(self) -> str | None:
        r"""Why the run is short of the recording, or ``None``."""
        unsent = len(self._stored) - len(self.responses)
        if unsent:
            return (
                f'the run sent {len(self.responses)} requests to the model, '
                f'and the recording answered {len(self._stored)}'
            )
        return None

    def post(self, url: str, data: bytes, **kwargs: Any) -> _Response:
        payload = json.loads(data)
        if _path(url) == PREDICTIONS:
            self.payloads.append(payload)
        return self._serve('POST', url, payload)

    def delete(self, url: str, **kwargs: Any) -> _Response:
        return self._serve('DELETE', url)

    def close(self) -> None:
        return None
