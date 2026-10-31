# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native table ownership and transport for Rust Tabular session operations."""

from __future__ import annotations

import json
from typing import Any

from sdm_client._routing import RouteError
from sdm_client._rust import codec, direct, results
from sdm_client._rust.run import run


def encode_table(frame: Any, columns: dict[str, Any], half: str) -> bytes:
    """Observe only the transmitted frame using existing native conversion."""
    if direct._native_encoding(frame):
        from sdm_client._rust.direct_cells import table

        values = table(
            frame, half, dict(columns['dtypes']), direct._native_calls
        )
        header = {
            'rows': values.num_rows if values.column_names else len(frame),
            'columns': values.column_names,
        }
    else:
        header = direct._header(frame)
        values = direct._table(frame)
    finished = direct._run(
        'encode_tabular_table',
        {'columns': columns, 'header': header},
        {2: values},
        values=2,
    )
    result = results.expect(finished.result, results.DirectValue)
    encoded = json.loads(finished.take(result.value))
    return bytes(finished.take(encoded['table']))


def close(fitted: Any) -> None:
    """Retain the old state if release fails; publish closure only on success."""
    state = fitted._tabular_session

    def send(endpoint: str | dict[str, Any], body: bytes) -> bytes:
        if not isinstance(endpoint, dict) or set(endpoint) != {
            'delete_session'
        }:
            raise RouteError('session close received an unexpected endpoint')
        fitted._client._transport.delete_session(
            endpoint['delete_session']['session_id']
        )
        return b''

    finished = run(
        {'op': 'close_tabular_session', 'input': 1},
        {1: codec.attachment(state)},
        model_endpoints=send,
    )
    result = results.expect(finished.result, results.DirectValue)
    replacement = json.loads(finished.take(result.value))
    fitted._tabular_session = replacement
    fitted._closed = replacement['closed']
    if fitted._closed:
        fitted._client._pinned.discard(fitted)


def question(request: Any) -> dict[str, Any] | None:
    """Observe the public question without retaining native objects in Rust."""
    value = request._question
    if value is None:
        return None
    return {
        'query': value.query,
        'kind': value.kind,
        'target': value.target,
        'dtype': value.dtype,
        'entity_table': value.entity_table,
        'context': None if value.context is None else str(value.context),
    }


def key_part(value: Any) -> dict[str, Any]:
    """Native type/missingness observations; Rust owns key normalization."""
    from decimal import Decimal
    from numbers import Real

    import pandas as pd

    try:
        missing = value is None or bool(pd.isna(value))
    except (ArithmeticError, TypeError, ValueError):
        missing = True
    if missing:
        return {'kind': 'missing'}
    if isinstance(value, bool):
        return {'kind': 'boolean', 'value': value}
    if isinstance(value, (Real, Decimal)):
        return {'kind': 'numeric', 'decimal': str(value)}
    return {'kind': 'spelling', 'value': str(value)}


def _value(finished: Any) -> dict[str, Any]:
    result = results.expect(finished.result, results.DirectValue)
    return json.loads(finished.take(result.value))


def send(
    client: Any,
    endpoint: str | dict[str, Any],
    payload: bytes,
    *,
    inference: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Return native reply and actual sent object without JSON-copying them.

    Rust selects the endpoint and constructs the body. The successful empty
    reply acknowledges I/O only; the retained native reply is interpreted by
    the shared response reader or creation-reply observations afterwards.
    """
    observed: dict[str, Any] = {}

    def transmit(target: str | dict[str, Any], body: bytes) -> bytes:
        native = json.loads(body)
        if 'inference' in native:
            native['inference'] = inference
        observed['payload'] = native
        if target == 'create_session':
            observed['reply'] = client._transport.create_session(native)
        elif isinstance(target, dict) and set(target) == {'session_predict'}:
            observed['reply'] = client._transport.session_predict(
                target['session_predict']['session_id'], native
            )
        else:
            raise RouteError('unexpected session transport endpoint')
        return b'{}'

    run(
        {'op': 'send_tabular_session', 'input': 1, 'payload': 2},
        {1: codec.attachment({'endpoint': endpoint, 'payload': 2}), 2: payload},
        model_endpoints=transmit,
    )
    return observed['reply'], observed['payload']


def fit(
    client: Any, request: Any, model: Any, outputs: Any, holdout: Any
) -> Any:
    """Upload one Rust-prepared context and retain only session metadata."""
    from sdm_client.adapters.tabular import _request_id
    from sdm_client.client import FittedModel, _adapter_for, _report_held_back
    from sdm_client.errors import StructuredDataError

    bound = direct.bind_request(request, model, outputs)
    _require_frames(request)
    table, context, _scoring, resolved, _native_dtypes = direct.resolve_request(
        request, bound
    )
    encoded = encode_table(context, resolved['columns'], 'context')
    key = table.primary_key
    primary_key = (
        [] if not key else [key] if isinstance(key, str) else list(key)
    )
    if isinstance(holdout, (str, bytes)):
        raise StructuredDataError(
            f'holdout names the rows this session may score, and {holdout!r} '
            f'is one string rather than a list of keys; pass [{holdout!r}] if '
            f'that is the key you mean',
            code='INVALID_REQUEST',
        )
    holdout_rows = [
        [
            key_part(part)
            for part in (key if isinstance(key, (tuple, list)) else [key])
        ]
        for key in (() if holdout is None else holdout)
    ]
    finished = direct._run(
        'prepare_tabular_fit',
        {
            'bound': bound,
            'columns': resolved['columns'],
            'primary_key': primary_key,
            'question': question(request),
            'context': 2,
            'holdout': 3,
            'request_id': _request_id(),
            'pinned_over_a_window': request.target_moves_with_time,
        },
        {2: encoded, 3: codec.attachment({'rows': holdout_rows})},
        context=2,
        holdout=3,
    )
    prepared = _value(finished)
    pinned_keys = bytes(finished.take(prepared['pinned_keys']))
    payload = bytes(finished.take(prepared['payload']))
    _report_held_back(prepared['held_back'], prepared['named_holdout_keys'])
    try:
        created, sent = send(client, 'create_session', payload)
    except Exception as error:
        from sdm_client.client import _SESSIONS_UNSUPPORTED

        status = getattr(error, 'status_code', None)
        if status in _SESSIONS_UNSUPPORTED:
            raise StructuredDataError(
                f'this endpoint does not serve sessions (HTTP {status}); '
                f'use predict() instead, which sends the context with every call',
                code='SESSIONS_UNSUPPORTED',
            ) from error
        raise
    primary = created.get('session_id')
    fallback = created.get('id') if not primary else None
    selected = direct._call(
        'check_tabular_session_reply',
        {'session_id_truthy': bool(primary), 'id_truthy': bool(fallback)},
    )
    if selected == 'missing':
        raise StructuredDataError(
            'the endpoint created a session but named no session id',
            code='INVALID_RESPONSE',
            details={'response_keys': sorted(created)},
        )
    session_id = str(primary if selected == 'session_id' else fallback)
    state = {
        'version': 1,
        'session_id': session_id,
        'closed': False,
        'bound': prepared['bound'],
        'pinned_model': prepared['bound']['model'],
        'pinned_kind': prepared['bound']['kind'],
        'columns': prepared['columns'],
        'primary_key': prepared['primary_key'],
        'question': question(request),
        'pinned_keys': 2,
        'held_back': prepared['held_back'],
        'pinned_over_a_window': prepared['pinned_over_a_window'],
    }
    observed = {
        name: sent[name] for name in ('model', 'task', 'schema', 'context')
    }
    accepted = direct._run(
        'accept_tabular_session',
        {'state': state, 'observed': 2},
        {2: codec.attachment(observed)},
        observed=2,
    )
    state = _value(accepted)
    pinned_keys = bytes(accepted.take(state['pinned_keys']))
    state['pinned_keys'] = 2
    spec = client._spec(model)
    adapter = client._adapters.get(spec.model_id)
    if adapter is None:
        adapter = _adapter_for(spec)
        client._adapters[spec.model_id] = adapter
    fitted = FittedModel._open(
        client,
        adapter,
        session_id,
        tuple(bound['outputs']),
        {key: value for key, value in sent.items() if key != 'context'},
        named_outputs=outputs,
        held_back=prepared['held_back'],
        pinned_over_a_window=request.target_moves_with_time,
        question=request._question,
    )
    fitted._tabular_session = state
    fitted._tabular_session_keys = pinned_keys
    client._pinned.add(fitted)
    return fitted


def predict(fitted: Any, request: Any) -> Any:
    """Encode scoring rows against the pinned schema and reuse attribution."""
    from sdm_client._rust import direct_options, direct_response
    from sdm_client.adapters.tabular import (
        _a_row_the_pinned_schema_cannot_hold,
        _request_id,
    )
    from sdm_client.client import _empty_answer
    from sdm_client.errors import StructuredDataError

    state = fitted._tabular_session
    different = direct._call(
        'check_tabular_session_question',
        {'state': state, 'question': question(request)},
    )
    if different is not None:
        labels = {
            'query': 'the query',
            'kind': 'the task kind',
            'target': 'the target column',
            'dtype': "the target column's dtype",
            'entity_table': 'the entity table',
        }
        field = different['field']
        detail = (
            'the context pinned is not the one this request carries, and the '
            'context is what the answer is learnt from'
            if field == 'context'
            else f'{labels[field]} pinned is {different["pinned"]!r} and this asks {different["asked"]!r}'
        )
        raise StructuredDataError(
            f'this session pinned a different question: {detail}. Only the rows '
            f'travel on a scoring call, so the pinned question is the one that '
            f'would be answered. Fit again to ask something else',
            code='INVALID_REQUEST',
        )
    if request.nothing_to_score:
        return _empty_answer(fitted.model, fitted._outputs, request)
    _report_overlap(fitted, request)
    try:
        _require_frames(request)
        request.schema.check_consistent()
        scoring = encode_table(
            request.predict.instance_table, state['columns'], 'predict'
        )
        inference_present = bool(request.inference)
        options = (
            direct_options.observe(request.inference, direct._call)
            if inference_present
            else None
        )
    except Exception as error:
        raise _a_row_the_pinned_schema_cannot_hold(
            request, dict(state['columns']['dtypes']), error
        ) from error
    finished = direct._run(
        'prepare_tabular_score',
        {
            'state': state,
            'predict': 3,
            'request_id': _request_id(),
            'inference_present': inference_present,
        },
        {3: scoring},
        predict=3,
    )
    prepared = _value(finished)
    layout = json.loads(finished.take(prepared['layout']))
    reply, sent = send(
        fitted._client,
        {'session_predict': {'session_id': state['session_id']}},
        bytes(finished.take(prepared['payload'])),
        inference=options,
    )
    # Use the actual mutable object consumed by the transport with the pinned
    # server schema. Ordinary replies share direct prediction's Rust decoder.
    layout.update(sent)
    result = direct_response.decoded_reply(reply, layout, prepared['bound'])
    if result is not None:
        return result
    response = direct_response.Reader(direct._call).response(
        reply, fitted.model
    )
    return direct_response.present(
        response, layout, prepared['bound'], direct._call
    )


def _report_overlap(fitted: Any, request: Any) -> None:
    from sdm_client.client import _report_pinned_overlap

    state = fitted._tabular_session
    names = state['primary_key']
    frame = request.predict.instance_table
    if not names or any(name not in frame.columns for name in names):
        return
    try:
        encoded = encode_table(frame[names], state['columns'], 'predict')
    except Exception:  # noqa: BLE001 - an overlap notice must not cause a conversion failure
        return
    finished = direct._run(
        'count_tabular_session_overlap',
        {'state': state, 'predict': 3},
        {2: fitted._tabular_session_keys, 3: encoded},
        predict=3,
        pinned_keys=2,
    )
    counts = _value(finished)
    _report_pinned_overlap(counts['overlap_rows'], counts['scoring_rows'])


def _require_frames(request: Any) -> None:
    if request.payloads is not None:
        from sdm_client.errors import UnsupportedRequestError

        raise UnsupportedRequestError(
            'this request was gathered for kumo-relational and carries '
            'the sampled payloads that model reads. Gather it with '
            "materialize(..., model='kumo-tabular') to ask this one"
        )
