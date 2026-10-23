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
as the target's own type), its probabilities, from T2.13 its explanation
and metadata, beside the result's explanation, and from T2.9 a
regression's quantiles. The endpoint's answer
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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from sdm_client._rust import codec, results
from sdm_client._rust.run import Refused, TransportFailed, run
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


def _table(stream: Any) -> Any:
    r"""An Arrow attachment the core gave, as a table, which keeps its
    schema's metadata (T2.13: the result's explanation)."""
    import pyarrow as pa

    reader = pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    )
    return reader.read_all()


def _explanation(text: str | None, by_step: bool) -> Any:
    r"""The result's explanation the core gives (T2.13), ``None`` where there
    is none: a mapping as :func:`_mapping` reads one, or, for a forecast,
    each step's under ``by_forecast_step``, keyed by the step's number,
    which the core writes in decimal digits."""
    if text is None:
        return None
    if not by_step:
        return _mapping(text)
    steps = json.loads(text)['by_forecast_step']
    return {
        'by_forecast_step': {
            int(step): dict(explained) for step, explained in steps.items()
        }
    }


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


def _mapping(text: str) -> dict[Any, Any]:
    r"""An item's rankings member as the reference's item holds it, from
    the JSON text the core gives: an object, or, where a key is not text,
    the list of its pairs."""
    read = json.loads(text)
    return dict(read)


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
        StructuredDataError: ``INVALID_REQUEST`` naming the rule
            ``materialize_again`` for a request not materialized on the core,
            which holds no request state the core wrote, or one holding other
            payloads than the core wrote (T3.16); the core's refusal, as :mod:`.errors` maps it
            (the model and output checks, the context's, and what it does
            not predict yet, ``UNSUPPORTED_REQUEST`` naming the rule); a
            model given as neither a model handle nor an id; and the
            endpoint's refusal, as the client's transport reports it.
    """

    from sdm_client.adapters.relational import _request_id
    from sdm_client.models import spec_for

    state = request._core
    if state is None:
        # A request the core did not materialize (one the old route built,
        # or pickled before it could hold the core's state) holds nothing
        # the core can predict from: refused by name, the caller told what
        # to do (the structure design's open question 3, T3.16).
        raise StructuredDataError(
            'this request was not materialized on the core, so it holds '
            'nothing the core can predict from; materialize it again',
            code='INVALID_REQUEST',
            details={'rule': 'materialize_again'},
        )
    # A forecast or a ranking the core gathered is held as one request per
    # part (T1.36 fix 21, T2.8), which the core predicts in one run, part
    # by part; a request of more rows than one call scores, as each call's
    # payload (T2.8c), which the core predicts in one run, call by call.
    steps = request.parts if request.parts is not None else (request,)
    batched = state.batches
    held = []
    for step in steps:
        payloads = step.payloads
        if payloads is None or len(payloads) != 1 + len(batched or ()):
            # Payloads other than the ones the core wrote with its state.
            raise StructuredDataError(
                'this request holds other payloads than materialize wrote '
                'on the core; materialize it again',
                code='INVALID_REQUEST',
                details={'rule': 'materialize_again'},
            )
        held.append((payloads[0], step.query))
    [(payload, query), *later] = held
    ranking = state.ranking
    # A ranking's candidates, by their position among those given, as the
    # caller gave them: the core names each by its position.
    given: dict[int, Any] = (
        {}
        if ranking is None or request._decomposition is None
        else dict(
            zip(
                ranking.positions,
                (about for _, about in request._decomposition.parts),
                strict=True,
            )
        )
    )
    # A model handle is the host's object; its id, or the id given, is the
    # core's to check.
    model_id = model if isinstance(model, str) else spec_for(model).model_id

    def model_call(body: bytes) -> bytes:
        # What failed is noted inside the handlers and raised after them, so
        # the transport's exception, whose text the endpoint wrote, is not
        # even the context of what is raised. The part a refusal is said of,
        # a forecast's step or a ranking's candidate, and how many were
        # answered before it, are the core's, which it names on its end of
        # the run (``run``'s ``_carried``).
        failure: StructuredDataError | None = None
        unexpected: Exception | None = None
        transport_failed = False
        refused: NimRequestError | None = None
        try:
            # The core's call, its bytes as written (register PD-2, T1.28a),
            # through the adapter that explains the endpoint's refusals; the
            # answer, the bytes the endpoint sent, is the core's to read
            # (T2.14b).
            return client._send_body(_adapter(client, model_id), body)
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
        except Exception as other:  # noqa: BLE001 - the reference's own
            # Any other exception the transport raised (the Databricks
            # SDK's ValueError for a failed OAuth refresh): the call failed,
            # and the calls before it were paid for, so the core is told,
            # carries their answers, and this raises it with them, as the
            # reference's client does (T2.8c fix 1). An interrupt is no
            # Exception, and cancels the run as before.
            unexpected = other
        if transport_failed:
            raise TransportFailed
        raised: StructuredDataError | None = failure
        if refused is not None:
            # Built outside the handler, so nothing failing in the copy is
            # chained to the transport's exception. The core is told the
            # endpoint's status, so it ends the run carrying the answers of
            # the calls before this one (T2.8c), which this raises on.
            raised = _refusal(refused)
        if raised is not None:
            raise Refused(
                raised, None if refused is None else refused.status_code
            )
        assert unexpected is not None
        raise Refused(unexpected)

    operation: dict[str, Any] = {
        'op': 'predict',
        'state': {'version': state.version, 'payload': 1, 'query': 2},
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
    if batched is not None:
        # Each later call's payload, the rows the core gathered it to score,
        # and its own id, drawn as the adapter draws one for every call, in
        # call order (T2.8c).
        operation['state']['batches'] = []
        for at, (more, rows) in enumerate(
            zip((request.payloads or ())[1:], batched, strict=True), start=3
        ):
            attachments[at] = _payload(more)
            operation['state']['batches'].append(
                {'payload': at, 'rows': rows, 'request_id': _request_id()}
            )
    if request.parts is not None and ranking is not None:
        operation['state']['ranking'] = {
            'candidate': ranking.positions[0],
            'value': ranking.values[0],
            'parts': [],
        }
        if ranking.top_k is not None:
            operation['state']['ranking']['top_k'] = ranking.top_k
        for number, ((more, asked), position, named) in enumerate(
            zip(
                later,
                ranking.positions[1:],
                ranking.values[1:],
                strict=True,
            ),
            start=2,
        ):
            at = 2 * number - 1
            attachments[at] = _payload(more)
            attachments[at + 1] = codec.attachment(asked)
            # Each question's call its own id, drawn as the adapter draws
            # one for every call, in order.
            operation['state']['ranking']['parts'].append(
                {
                    'candidate': position,
                    'value': named,
                    'payload': at,
                    'query': at + 1,
                    'request_id': _request_id(),
                }
            )
    elif request.parts is not None:
        operation['state']['forecast'] = []
        for number, (more, asked) in enumerate(later, start=2):
            at = 2 * number - 1
            attachments[at] = _payload(more)
            attachments[at + 1] = codec.attachment(asked)
            # Each step's call its own id, drawn as the adapter draws one
            # for every call, in step order (T1.36 fix 24).
            operation['state']['forecast'].append(
                {
                    'step': number,
                    'payload': at,
                    'query': at + 1,
                    'request_id': _request_id(),
                }
            )

    def unscored(fields: dict[str, Any]) -> str:
        # The core's notice of the candidates it left out, by position, said
        # naming each as the caller gave it, as the client says it.
        return _unscored([given[at] for at in fields['candidates']])

    try:
        finished = run(
            operation,
            attachments,
            model=model_call,
            words={'CANDIDATES_UNSCORED': unscored},
        )
    except Exception as error:
        # Whatever the call raised, the answers before it are put on it.
        _carry_the_answers(error, request.parts is not None)
        raise
    answered = results.expect(finished.result, results.Predicted)
    metadata = json.loads(finished.take(answered.metadata))
    if ranking is not None:
        # The core's rows are ordered by row_index; a ranking's items keep
        # the order the answers named them in.
        ranked = sorted(
            _rows(finished.take(answered.result)),
            key=lambda row: row['answer_index'],
        )
        return _ranked(answered, ranked, metadata, given)
    table = _table(finished.take(answered.result))
    # The result's explanation, on the result's schema (T2.13).
    explained = (table.schema.metadata or {}).get(b'explanation')
    return _result(
        table.to_pylist(),
        model=answered.model,
        request_id=answered.request_id,
        metadata=metadata,
        outputs=answered.outputs,
        by_step=request.parts is not None,
        explanation=None if explained is None else explained.decode(),
    )


def _result(
    rows: list[dict[str, Any]],
    *,
    model: str,
    request_id: str,
    metadata: dict[str, Any],
    outputs: tuple[str, ...],
    by_step: bool,
    explanation: str | None = None,
) -> PredictionResult:
    r"""The core's result rows as this client's result: each item's
    explanation and metadata, and the result's explanation, as the core
    gives them (T2.13)."""
    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client.result import PredictionResult

    # The core's rows are ordered by row_index, as the result's frame is;
    # its items keep the endpoint's order, as the old route's do.
    rows = sorted(rows, key=lambda row: row['answer_index'])
    result = PredictionResult(
        model=model,
        request_id=request_id,
        items=[
            PredictionItem(
                row_index=row['row_index'],
                id=_id(row),
                forecast_step=row.get('forecast_step'),
                prediction=_as_answered(row),
                probabilities=None
                if row.get('probabilities') is None
                else dict(row['probabilities']),
                # An item's scores and rankings (T2.8), each mapping given
                # as the JSON text of the reference's item value: an object,
                # or the list of its pairs where a key is not text.
                scores=None
                if row.get('scores') is None
                else tuple(row['scores']),
                rankings=None
                if row.get('rankings') is None
                else tuple(_mapping(text) for text in row['rankings']),
                # Its explanation and own metadata (T2.13), each the JSON
                # text of a mapping, read by the same reader.
                explanation=None
                if row.get('explanation') is None
                else _mapping(row['explanation']),
                metadata=None
                if row.get('metadata') is None
                else _mapping(row['metadata']),
                # A regression's quantiles (T2.9), keyed as the endpoint
                # wrote them, each value a double, read by the same reader;
                # to_pandas spreads them as quantile_<key>.
                quantiles=None
                if row.get('quantiles') is None
                else _mapping(row['quantiles']),
                embeddings=None
                if row.get('embeddings') is None
                else tuple(row['embeddings']),
            )
            for row in rows
        ],
        metadata=metadata,
        outputs=outputs,
        explanation=_explanation(explanation, by_step),
    )
    # A forecast's answer names its step, rows or none (`_by_step`).
    result._names_its_step = by_step
    return result


def _ranked(
    answered: results.Predicted,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    given: dict[int, Any],
) -> PredictionResult:
    r"""A ranking's result as the client gives it (``_ranked``): each row's
    candidates, best first, as the caller gave them (``given``, by the
    position the result names each by), under ``prediction`` and, with each
    one's score, under ``rankings``, each where the outputs name it; the
    candidates left out, unscored, as given (by their positions among those
    given)."""
    from sdm_api.generated.tfm_api import PredictionItem

    from sdm_client.result import PredictionResult

    wanted = answered.outputs
    items = []
    for row in rows:
        chosen = [given[at] for at in row['candidates']]
        items.append(
            PredictionItem(
                row_index=row['row_index'],
                id=_id(row),
                prediction=chosen if 'prediction' in wanted else None,
                rankings=(
                    tuple(
                        {'candidate': candidate, 'score': score}
                        for candidate, score in zip(
                            chosen, row['scores'], strict=True
                        )
                    )
                    if 'rankings' in wanted
                    else None
                ),
            )
        )
    return PredictionResult(
        model=answered.model,
        request_id=answered.request_id,
        items=items,
        metadata=metadata,
        outputs=wanted,
        unscored_candidates=[given[at] for at in answered.unscored_candidates],
    )


def _unscored(cold: list[Any]) -> str:
    r"""The words of the notice of a ranking's candidates left out,
    unscored, naming each as the caller gave it, as the client's
    ``_candidates_no_context_teaches`` says them."""
    from sdm_client.client import _SCORE_THE_UNSCORED

    many = len(cold) > 1
    named = ', '.join(repr(candidate) for candidate in cold[:-1])
    which = (
        f'candidates {named} and {cold[-1]!r} are'
        if many
        else f'candidate {cold[0]!r} is'
    )
    them = 'them' if many else 'it'
    return (
        f'{which} left out of this ranking, unscored: the context sampled '
        f'for {"each" if many else "it"} holds one answer only, nearly '
        f'always because no entity in it linked to the candidate in the '
        f'window, and a classification needs two to learn from. No call was '
        f'sent for {them}, so each row ranks only the rest, and the '
        f"result's unscored_candidates names {them}. {_SCORE_THE_UNSCORED}"
    )


def _carry_the_answers(error: Exception, by_step: bool) -> None:
    r"""The answers received before a call failed, as the core carried them
    on its error (T2.8c, ``main``'s 1fc5a686), put in ``error.details`` as
    the client puts them: ``answered``, the answers received as one result,
    and, for a request asked in several calls, ``rows_answered``, the rows
    those calls score, beside the core's ``calls_answered``; for a
    forecast, its steps' answers, beside the core's ``parts_answered``. The
    error keeps its own type and code."""
    import pyarrow as pa

    carried = getattr(error, '_answered', None)
    if carried is None:
        return
    delattr(error, '_answered')
    stream, rows = carried
    table = pa.RecordBatchReader._import_from_c_capsule(
        stream.__arrow_c_stream__()
    ).read_all()
    # What the result names beside its rows travels on its schema, as text
    # (``sdm_core::predict``'s ``carried``).
    named = {
        key.decode(): value.decode()
        for key, value in (table.schema.metadata or {}).items()
    }
    held = getattr(error, 'details', None)
    details = held if isinstance(held, dict) else {}
    details['answered'] = _result(
        table.to_pylist(),
        model=named.get('model', ''),
        request_id=named.get('request_id', ''),
        metadata=json.loads(named.get('metadata', '{}')),
        outputs=tuple(json.loads(named.get('outputs', '[]'))),
        by_step=by_step,
        explanation=named.get('explanation'),
    )
    if not by_step:
        details['rows_answered'] = rows
    error.details = details  # type: ignore[attr-defined]
