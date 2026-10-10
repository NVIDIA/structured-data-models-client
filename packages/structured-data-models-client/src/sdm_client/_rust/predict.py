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
is about. An endpoint's refusal is the transport's error, as on the old
route; an endpoint that could not be reached is told to the core, which
ends the run with ``TRANSPORT_ERROR``.

As built at T1.11d the core predicts a regression asked for its default
output; the rest is refused with ``UNSUPPORTED_REQUEST`` naming the rule
(register entry P1B-4), here for the outputs and in the core for the rest.
Nothing here imports the relational engine.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from sdm_client._routing import RouteError
from sdm_client._rust import codec
from sdm_client._rust.run import Unreached, run
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
            'predict on the core needs a request materialize built on the '
            'core'
        )
    if tuple(fields) != ('prediction',):
        raise _not_taken('outputs')
    # The adapter's own checks of what the call sends, which its
    # prediction_plan makes before building any call.
    payloads = adapter._payloads(request)
    adapter._preflight(payloads, fields, request.query)

    def model(body: bytes) -> bytes:
        try:
            answer = client._send(adapter, json.loads(body))
        except NimRequestError:
            # The endpoint answered, and refused: its own error, explained
            # by the adapter, as on the old route.
            raise
        except StructuredDataError as failed:
            if failed.code == 'TRANSPORT_ERROR':
                raise Unreached from None
            raise
        return json.dumps(answer).encode()

    parts = []
    for payload in payloads:
        operation = {
            'op': 'predict',
            'state': {'version': state['version'], 'payload': 1, 'query': 2},
            'model': adapter.spec.model_id,
            'request_id': _request_id(),
        }
        attachments = {
            1: json.dumps(payload, allow_nan=False).encode(),
            2: codec.attachment(request.query),
        }
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
