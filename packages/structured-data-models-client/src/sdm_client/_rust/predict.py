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

No text the endpoint wrote, and no value of the request, reaches an
exception on this route (decision H17; E45(c) as revised in T1.11d's first
review round): an endpoint's refusal (HTTP 4xx or 5xx) is raised as the
transport raises it, a :class:`~sdm_client.errors.NimRequestError` with the
endpoint's status and code and the client's own diagnosis, but with a fixed
sentence, none of the endpoint's detail, title or other fields, and no
cause; a failed transport (the endpoint could not be reached, or its
answer could not be read) is told to the core, which ends the run with
``TRANSPORT_ERROR``; any other error of the transport keeps its code and
gets a fixed sentence. Whether a user needs the endpoint's own words on a
refusal is a question for the owner (E45).

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
from sdm_client._rust import codec
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
    for is refused as the client's transport refuses it."""
    try:
        return json.dumps(payload, allow_nan=False).encode()
    except (TypeError, ValueError):
        raise StructuredDataError(
            'this request carries a value that cannot be sent as JSON',
            code='INVALID_REQUEST',
        ) from None


def _encoded(value: Any) -> bytes:
    r"""An answer the transport decoded, as JSON text for the core: what
    ``json.dumps`` writes, but a non-finite float, which Python's reader
    made of the endpoint's text (an overflowing ``1e400``, or a ``NaN`` or
    ``Infinity`` JSON does not hold), as a token the core reads as one: an
    infinity as ``1e400`` or ``-1e400``, a NaN as ``NaN``, which the core
    refuses as not JSON."""
    out: list[str] = []

    def write(item: Any) -> None:
        if isinstance(item, float) and not math.isfinite(item):
            out.append(
                'NaN' if item != item else ('1e400' if item > 0 else '-1e400')
            )
        elif isinstance(item, dict):
            out.append('{')
            for index, (key, member) in enumerate(item.items()):
                if index:
                    out.append(', ')
                out.append(json.dumps(str(key)))
                out.append(': ')
                write(member)
            out.append('}')
        elif isinstance(item, (list, tuple)):
            out.append('[')
            for index, member in enumerate(item):
                if index:
                    out.append(', ')
                write(member)
            out.append(']')
        else:
            out.append(json.dumps(item))

    write(value)
    return ''.join(out).encode()


def _refusal(refused: NimRequestError) -> NimRequestError:
    r"""The endpoint's refusal as this route raises it: its class, status
    and code, and the client's own diagnosis of it, with a fixed sentence
    and none of the endpoint's text (decision H17)."""
    details = refused.details if isinstance(refused.details, dict) else {}
    kept = {
        name: details[name]
        for name in ('diagnosis', 'endpoint_code')
        if isinstance(details.get(name), str)
    }
    return NimRequestError(
        refused.status_code,
        code=refused.code if isinstance(refused.code, str) else None,
        message='the model endpoint refused the request; its own words are '
        'not shown on this route',
        details=kept,
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
        try:
            answer = client._send(adapter, json.loads(body))
        except NimRequestError as refused:
            failure = _refusal(refused)
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
