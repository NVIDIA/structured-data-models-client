# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``predict`` on the core: the model call and its answer (T1.11d).

``SDMClient.predict`` of a request materialized on the core over a DuckDB
source comes here on the core route, in place of the relational adapter's
call and reading of the answer (``prediction_plan``, ``normalize``).
Everything around them stays the client's own, as on the old route: its
checks of the call before (the request, the outputs, the rows scored from
their own answers, the adapter's checks of the context) and, after, the
merge of the parts and the notice of an output the answer left out.

The core's ``predict`` takes the request state ``materialize`` answered,
which the request held since (its version, and its payload and query as
the request holds them now), and the id this call carries, drawn as the
adapter draws one; it makes the call the adapter makes, which this host
sends through the client's own transport, reads the answer as the client's
bindings read it, and lays it out: one row per answer, under the entity it
is about. The endpoint's answer reaches the core as its JSON, every
non-finite number Python's reader made of it written as a token the core
reads as one (an overflowing ``1e400`` for an infinity), so both hosts
refuse it alike.

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
credential the URL holds, is not chained, nor any other context; and every
spelling of an exact credential of ours the endpoint echoes, in any field of
its refusal, keys included, is replaced by ``<redacted>`` before it is shown
(:mod:`.scrub`; E47, orchestrator reading, owner to confirm). The old route, the
reference's in this phase, shows such an echo unchanged. A failed transport
(the endpoint could not be reached, or its answer could not be read) is
told to the core, which ends the run with ``TRANSPORT_ERROR``; any other
error of the transport keeps its code and gets a fixed sentence.

As built at T1.11d the core predicts a regression asked for its default
output; the rest is refused with ``UNSUPPORTED_REQUEST`` naming the rule
(register entry P1B-4), here for the outputs and in the core for the rest.
Nothing here imports the relational engine.
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING, Any

from sdm_client._routing import RouteError
from sdm_client._rust import codec, scrub
from sdm_client._rust.run import TransportFailed, run
from sdm_client.errors import NimRequestError, StructuredDataError

if TYPE_CHECKING:
    from sdm_client.client import SDMClient
    from sdm_client.request import Request
    from sdm_client.result import PredictionResult


def _not_taken(rule: str) -> StructuredDataError:
    return StructuredDataError(
        'predict on the core does not take this yet',
        code='UNSUPPORTED_REQUEST',
        details={'operation': 'predict', 'rule': rule},
    )


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
    reads as non-finite and refuses alike on every host: an infinity as
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


def _refusal(refused: NimRequestError, found: Any) -> NimRequestError:
    r"""The endpoint's refusal as this route raises it: the old route's
    class, status, code, message and details, the endpoint's own text and
    the client's explanation of it (decision E47), but a new exception, so
    nothing of the transport's (its URL, a credential) is chained to it.
    Every field of it, of any JSON shape and depth, keys included, goes
    through one scrub of each exact credential of ours (:mod:`.scrub`,
    E47's orchestrator reading (a)), without recursion."""
    details = refused.details if isinstance(refused.details, dict) else {}
    record = scrub.scrubbed(
        {'message': refused.message, 'code': refused.code, 'details': details},
        found,
    )
    message = record['message']
    return type(refused)(
        refused.status_code,
        code=record['code'],
        message=message if isinstance(message, str) else str(message),
        details=record['details'],
    )


def predict(
    client: SDMClient,
    adapter: Any,
    request: Request,
    fields: tuple[str, ...],
) -> list[PredictionResult]:
    r"""The answer to each payload of ``request``, predicted on the core.

    Raises:
        RouteError: The request was not materialized on the core, so it
            holds no request state the core wrote.
        StructuredDataError: The core's refusal, as :mod:`.errors` maps it,
            ``UNSUPPORTED_REQUEST`` naming the rule for what it does not
            predict yet; the client's own checks of the call; and the
            endpoint's refusal, as the client's transport reports it.
    """
    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client.adapters.relational import _request_id
    from sdm_client.result import PredictionResult

    state = getattr(request, '_core_state', None)
    if state is None:
        raise RouteError(
            'predict on the core needs a request materialize built on the core'
        )
    if tuple(fields) != ('prediction',):
        raise _not_taken('outputs')
    # The adapter's own checks of what the call sends, which its
    # prediction_plan makes before building any call.
    payloads = adapter._payloads(request)
    adapter._preflight(payloads, fields, request.query)

    def model(body: bytes) -> bytes:
        # What failed is noted inside the handlers and raised after them, so
        # the transport's exception, whose text the endpoint wrote, is not
        # even the context of what is raised.
        failure: StructuredDataError | None = None
        transport_failed = False
        refused: NimRequestError | None = None
        try:
            answer = client._send(adapter, json.loads(body))
        except NimRequestError as refusal:
            refused = refusal
        except StructuredDataError as failed:
            if failed.code == 'TRANSPORT_ERROR':
                transport_failed = True
            else:
                failure = StructuredDataError(
                    'the transport to the model endpoint refused the call',
                    code=failed.code,
                )
        if transport_failed:
            raise TransportFailed
        if refused is not None:
            # Built outside the handler, so nothing failing in the copy is
            # chained to the transport's exception.
            found = scrub.pattern(
                scrub.forms(scrub.credentials(client._transport))
            )
            raise _refusal(refused, found)
        if failure is not None:
            raise failure
        return _encoded(answer)

    parts = []
    for payload in payloads:
        operation = {
            'op': 'predict',
            'state': {'version': state['version'], 'payload': 1, 'query': 2},
            'model': adapter.spec.model_id,
            'request_id': _request_id(),
        }
        attachments = {1: _payload(payload), 2: codec.attachment(request.query)}
        finished = run(operation, attachments, model=model)
        answered = finished.result
        rows = _rows(finished.take(answered['result']))
        metadata = json.loads(finished.take(answered['metadata']))
        parts.append(
            PredictionResult(
                model=answered['model'],
                request_id=answered['request_id'],
                items=[
                    PredictionItem(
                        row_index=row['row_index'],
                        id=row['id'],
                        prediction=row['prediction'],
                    )
                    for row in rows
                ],
                metadata=metadata,
                outputs=fields,
            )
        )
    return parts
