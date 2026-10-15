# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Bridge code: the transport and the result's objects stay the host's
# (design, "Host-held semantics": the model call's errors mapped, the answer
# re-encoded, the request id drawn); the rule rows T1.28b moved into the
# core are gone, and the bridge is deleted in Phase 5 with the old route.
# Frozen meanwhile (E50).

r"""``predict`` on the core: the model call and its answer (T1.11d).

``SDMClient.predict`` of a request materialized on the core over a DuckDB
source comes here on the core route, after the client's own checks of its
arguments' types (the request a ``Request``, the outputs a sequence of
names). Everything else the old route runs before and after the call is
the core's from T1.28b (the design's "Host-held semantics" rows it moved):
the model and output checks and the default outputs (``_prepare``), the
answer to a request with no rows, the adapter's checks of the context
(``_preflight``), the parts merged into one result and the notice of an
output the answer left out, each notice a code and fields this client
words (:mod:`.errors`).

The core's ``predict`` takes the request state ``materialize`` answered,
which the request held since (its version, and its payload and query as
the request holds them now), the model's id, the outputs the caller named
(or that it named none), and the id this call carries, drawn as the
adapter draws one; it makes the call the adapter makes, which this host
sends through the client's own transport, reads the answer as the client's
bindings read it, and lays it out: one row per answer, ordered by
``row_index``, under the entity it is about, with its prediction (a class
as the target's own type) and its probabilities. The endpoint's answer
reaches the core as its JSON, every non-finite number Python's reader made
of it written as a token the core reads as the same number (an infinity as
the overflowing ``1e400``, a NaN as ``NaN``), so both hosts read it alike.

What the endpoint itself wrote in a refusal (HTTP 4xx or 5xx) is shown as
on the old route (decision E47, owner decision 2026-10-10, an exception to
H17 for the endpoint's own text only): the refusal is raised as the
client's transport raises it and its adapter explains it, a
:class:`~sdm_client.errors.NimRequestError` with the endpoint's status,
``code``, message (its ``detail`` or ``title``) and details
(``endpoint_code``, ``diagnosis`` and its other fields, as the transport
reads them), the same class, message and details as the old route's. That
text is the endpoint's, so a request value it echoes back is shown, which
the owner accepted; to it the client adds only its own explanations, as on
the old route: the adapter's naming of the tables and columns of our schema
a refusal is about, and the Databricks transport's naming of the serving
endpoint (both the user's own configuration), never a value, a credential
or a URL. The transport's exception, and with it the endpoint's URL and any
credential the URL holds, is not chained, nor any other context. As on the
old route, an endpoint that echoes a credential of ours in its own text has
it shown: a scrub of such echoes was tried and removed (E47). A failed transport
(the endpoint could not be reached, or its answer could not be read) is
told to the core, which ends the run with ``TRANSPORT_ERROR``; any other
error of the transport keeps its code and gets a fixed sentence.

Nothing here imports the relational engine.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from sdm_client._routing import RouteError
from sdm_client._rust import codec
from sdm_client._rust.run import TransportFailed, run
from sdm_client.errors import NimRequestError, StructuredDataError

if TYPE_CHECKING:
    from sdm_client.client import SDMClient
    from sdm_client.request import Request
    from sdm_client.result import PredictionResult


def _rows(stream: Any) -> list[dict[str, Any]]:
    r"""An Arrow attachment the core gave, as rows."""
    import pyarrow as pa

    reader = pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    )
    return reader.read_all().to_pylist()


def _payload(payload: dict[str, Any]) -> bytes:
    r"""A payload as the core reads it; one holding a value JSON has no form
    for is refused as the client's transport refuses it, with no context."""
    try:
        return json.dumps(payload, allow_nan=False).encode()
    except (TypeError, ValueError):
        pass
    raise StructuredDataError(
        'this request carries a value that cannot be sent as JSON',
        code='INVALID_REQUEST',
    )


#: How deep :func:`_encoded` writes an answer: past the core's reader's
#: depth (``sdm_wire::json::MAX_DEPTH``, 128), so the core, not Python's
#: recursion, refuses one nested deeper, as ``too_deep``.
_ENCODED_DEPTH = 129


def _encoded(value: Any) -> bytes:
    r"""An answer the transport decoded, as JSON text for the core: what
    ``json.dumps`` writes, but each non-finite float Python's reader made of
    the endpoint's text (an overflowing ``1e400``, or the ``NaN`` and
    ``Infinity`` JSON does not hold), wherever it is, as a token the core
    reads as the same number, what it means being the core's: an infinity as
    ``1e400`` or ``-1e400``, a NaN as ``NaN``. It writes without recursion,
    and an answer nested past the core's depth is cut at one level past it,
    which the core refuses as too deep."""
    out: list[str] = []
    # Each entry: text to write, or an item to write at a depth.
    stack: list[tuple[str | None, Any, int]] = [(None, value, 0)]
    while stack:
        text, item, depth = stack.pop()
        if text is not None:
            out.append(text)
            continue
        if isinstance(item, float) and not math.isfinite(item):
            out.append(
                'NaN' if item != item else ('1e400' if item > 0 else '-1e400')
            )
        elif isinstance(item, (dict, list, tuple)):
            if depth >= _ENCODED_DEPTH:
                # Deep enough for the core to refuse; what is inside is
                # never read.
                out.append('[]')
                continue
            members = (
                [(json.dumps(str(k)) + ': ', v) for k, v in item.items()]
                if isinstance(item, dict)
                else [('', v) for v in item]
            )
            close = '}' if isinstance(item, dict) else ']'
            out.append('{' if isinstance(item, dict) else '[')
            stack.append((close, None, depth))
            for index in range(len(members) - 1, -1, -1):
                prefix, member = members[index]
                stack.append((None, member, depth + 1))
                stack.append(((', ' if index else '') + prefix, None, depth))
        else:
            out.append(json.dumps(item))
    return ''.join(out).encode()


def _copied(value: Any) -> Any:
    r"""``value``, JSON the transport decoded, copied without recursion at any
    depth, unchanged."""
    if not isinstance(value, (dict, list, tuple)):
        return value
    root: Any = {} if isinstance(value, dict) else []
    stack: list[tuple[Any, Any]] = [(value, root)]
    while stack:
        source, target = stack.pop()
        pairs = (
            source.items() if isinstance(source, dict) else enumerate(source)
        )
        for key, item in pairs:
            copied: Any = item
            if isinstance(item, (dict, list, tuple)):
                copied = {} if isinstance(item, dict) else []
                stack.append((item, copied))
            if isinstance(target, dict):
                target[key] = copied
            else:
                target.append(copied)
    return root


def _refusal(refused: NimRequestError) -> NimRequestError:
    r"""The endpoint's refusal as this route raises it: exactly the old
    route's class, status, code, message and details, the endpoint's own
    text and the client's explanation of it (decision E47), but a new
    exception, so nothing of the transport's (its URL, a credential) is
    chained to it. Its details are copied without recursion, at any depth
    the transport read."""
    details = refused.details if isinstance(refused.details, dict) else {}
    return type(refused)(
        refused.status_code,
        code=refused.code,
        message=refused.message,
        details=_copied(details),
    )


def _as_answered(row: dict[str, Any]) -> Any:
    r"""A row's prediction as the reference's item holds it: where the
    core's column changed an answer (a regression's integer widened among
    doubles, a class written as a non-finite number), the core gives each
    as the JSON text of that value (``prediction_as_answered``), read as the
    client's transport reads JSON; elsewhere the column's value, a missing
    prediction ``None`` either way."""
    if 'prediction_as_answered' in row:
        text = row['prediction_as_answered']
        return None if text is None else json.loads(text)
    return row.get('prediction')


def _id(row: dict[str, Any]) -> Any:
    r"""A row's id as the reference's item holds it: where the core's ids
    are not all of one type, or one is of another than an integer or text
    (an entity read as written), the core gives each as the JSON text of
    that value (``id_as_written``), read as the client's transport reads
    JSON; elsewhere the column's value, a missing id ``None`` either
    way."""
    if 'id_as_written' in row:
        text = row['id_as_written']
        return None if text is None else json.loads(text)
    return row.get('id')


def _adapter(client: SDMClient, model_id: str) -> Any:
    r"""The client's adapter of the model the core checked, which explains
    the endpoint's refusals as on the old route (E47)."""
    from sdm_client.client import _adapter_for
    from sdm_client.models import spec_for

    spec = spec_for(model_id)
    adapter = client._adapters.get(spec.model_id)
    if adapter is None:
        adapter = _adapter_for(spec)
        client._adapters[spec.model_id] = adapter
    return adapter


def predict(
    client: SDMClient,
    request: Request,
    model: Any,
    outputs: Sequence[str] | None,
) -> PredictionResult:
    r"""The answer to ``request``, predicted on the core.

    Raises:
        RouteError: The request was not materialized on the core, so it
            holds no request state the core wrote, or holds other payloads
            than the one it wrote.
        StructuredDataError: The core's refusal, as :mod:`.errors` maps it
            (the model and output checks, the context's, and what it does
            not predict yet, ``UNSUPPORTED_REQUEST`` naming the rule); a
            model given as neither a model handle nor an id; and the
            endpoint's refusal, as the client's transport reports it.
    """
    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client.adapters.relational import _request_id
    from sdm_client.models import spec_for
    from sdm_client.result import PredictionResult

    state = getattr(request, '_core_state', None)
    if state is None:
        raise RouteError(
            'predict on the core needs a request materialize built on the core'
        )
    # A forecast the core gathered is held as one request per step (T1.36
    # fix 21), which the core predicts in one run, step by step.
    steps = request.parts if request.parts is not None else (request,)
    held = []
    for step in steps:
        payloads = step.payloads
        if payloads is None or len(payloads) != 1:
            raise RouteError(
                'predict on the core takes the one payload materialize wrote'
            )
        held.append((payloads[0], step.query))
    [(payload, query), *later] = held
    # A model handle is the host's object; its id, or the id given, is the
    # core's to check.
    model_id = model if isinstance(model, str) else spec_for(model).model_id

    calls = 0

    def model_call(body: bytes) -> bytes:
        nonlocal calls
        # A forecast's steps are called in order, one call each, so the
        # call is the step a refusal of the endpoint is said of.
        calls += 1
        # What failed is noted inside the handlers and raised after them, so
        # the transport's exception, whose text the endpoint wrote, is not
        # even the context of what is raised.
        failure: StructuredDataError | None = None
        transport_failed = False
        too_deep = False
        refused: NimRequestError | None = None
        try:
            # The core's call, its bytes as written (register PD-2, T1.28a),
            # through the adapter that explains the endpoint's refusals.
            answer = client._send_body(_adapter(client, model_id), body)
        except NimRequestError as refusal:
            refused = refusal
        except StructuredDataError as failed:
            # The flag is set only where the transport's json.loads raised
            # RecursionError; one raised under a lowered recursion limit is
            # read as too deep too, which is acceptable.
            if failed.code == 'TRANSPORT_ERROR' and failed._response_too_deep:
                # An answer nested deeper than Python's reader takes on this
                # interpreter (before 3.12, past about 1,000 levels): the
                # core is told so with an answer cut one level past its own
                # depth, which it refuses as too deep, as on every
                # interpreter (register P1B-22).
                too_deep = True
            elif failed.code == 'TRANSPORT_ERROR':
                transport_failed = True
            else:
                failure = StructuredDataError(
                    'the transport to the model endpoint refused the call',
                    code=failed.code,
                )
        if too_deep:
            return b'[' * _ENCODED_DEPTH + b']' * _ENCODED_DEPTH
        if transport_failed:
            raise TransportFailed
        if refused is not None:
            # Built outside the handler, so nothing failing in the copy is
            # chained to the transport's exception.
            raised = _refusal(refused)
            raised._forecast_step = calls  # type: ignore[attr-defined]
            raise raised
        if failure is not None:
            failure._forecast_step = calls  # type: ignore[attr-defined]
            raise failure
        return _encoded(answer)

    operation: dict[str, Any] = {
        'op': 'predict',
        'state': {'version': state['version'], 'payload': 1, 'query': 2},
        'model': model_id,
        'request_id': _request_id(),
        # The outputs the caller named, or that it named none (the core then
        # asks the task's own, T1.28b).
        'outputs': [] if outputs is None else list(outputs),
        # The inference options the caller set (a mapping, on this
        # route), which the core refuses where the model does not read
        # them (T1.28a).
        'inference_asked': sorted(
            request.requested_inference.get('inference_config', {})
        ),
    }
    if outputs is None:
        operation['default_outputs'] = True
    attachments = {
        1: _payload(payload),
        2: codec.attachment(query),
    }
    if request.parts is not None:
        operation['state']['forecast'] = []
        for number, (more, asked) in enumerate(later, start=2):
            at = 2 * number - 1
            attachments[at] = _payload(more)
            attachments[at + 1] = codec.attachment(asked)
            operation['state']['forecast'].append(
                {'step': number, 'payload': at, 'query': at + 1}
            )
    finished = run(operation, attachments, model=model_call)
    answered = finished.result
    # The core's rows are ordered by row_index, as the result's frame is;
    # its items keep the endpoint's order, as the old route's do.
    rows = sorted(
        _rows(finished.take(answered['result'])),
        key=lambda row: row['answer_index'],
    )
    metadata = json.loads(finished.take(answered['metadata']))
    result = PredictionResult(
        model=answered['model'],
        request_id=answered['request_id'],
        items=[
            PredictionItem(
                row_index=row['row_index'],
                id=_id(row),
                forecast_step=row.get('forecast_step'),
                prediction=_as_answered(row),
                probabilities=None
                if row.get('probabilities') is None
                else dict(row['probabilities']),
            )
            for row in rows
        ],
        metadata=metadata,
        outputs=tuple(answered['outputs']),
    )
    # A forecast's answer names its step, rows or none (`_by_step`).
    result._names_its_step = request.parts is not None
    return result
