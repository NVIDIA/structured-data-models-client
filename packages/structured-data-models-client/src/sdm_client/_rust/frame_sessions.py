# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frame-session owners and transport; Rust owns envelopes and compatibility."""

from __future__ import annotations

import json
from typing import Any

from sdm_client._rust import codec, results, sessions
from sdm_client._rust import predict as prediction
from sdm_client._rust.run import Refused, run


def _run(
    client: Any,
    value: dict[str, Any],
    request: Any = None,
    pinned: bytes | None = None,
) -> Any:
    attachments: dict[int, Any] = {1: codec.attachment(value)}
    payloads = []
    if request is not None:
        for payload in request.payloads:
            number = len(attachments) + 1
            # Project the protocol's transmitted half before JSON conversion.
            # The unsent graph can be large and must not be inspected at score.
            names = (
                ('predict', 'output', 'inference', 'metadata')
                if value['action'] == 'predict'
                else (
                    'task',
                    'schema',
                    'context',
                    'output',
                    'inference',
                    'metadata',
                )
            )
            attachments[number] = prediction._payload(
                {name: payload[name] for name in names if name in payload}
            )
            payloads.append(number)
    pinned_id = None
    if pinned is not None:
        pinned_id = len(attachments) + 1
        attachments[pinned_id] = pinned

    def transmit(endpoint: Any, body: bytes) -> bytes:
        payload = json.loads(body)
        if endpoint == 'create_session':
            reply = client._transport.create_session(payload)
        elif isinstance(endpoint, dict) and 'session_predict' in endpoint:
            reply = client._transport.session_predict(
                endpoint['session_predict']['session_id'], payload
            )
        elif isinstance(endpoint, dict) and 'delete_session' in endpoint:
            client._transport.delete_session(
                endpoint['delete_session']['session_id']
            )
            return b''
        else:
            raise RuntimeError('unexpected frame session endpoint')
        return json.dumps(reply).encode()

    def send(endpoint: Any, body: bytes) -> bytes:
        try:
            return transmit(endpoint, body)
        except Exception as error:
            # Let Rust attach earlier paid answers before raising the native error.
            raise Refused(error, getattr(error, 'status_code', None)) from None

    operation = {'op': 'frame_session', 'action': value['action'], 'input': 1}
    if value['action'] != 'close':
        operation['payloads'] = payloads
    if value['action'] == 'predict':
        operation['pinned'] = pinned_id
    return run(operation, attachments, model_endpoints=send)


def _value(finished: Any) -> Any:
    value = results.expect(finished.result, results.DirectValue)
    return json.loads(finished.take(value.value))


def fit(
    client: Any, request: Any, model: Any, outputs: Any, holdout: Any
) -> Any:
    from sdm_client.adapters.relational import _request_id
    from sdm_client.client import (
        _SESSIONS_UNSUPPORTED,
        FittedModel,
        _adapter_for,
        _report_held_back,
    )
    from sdm_client.errors import StructuredDataError

    if isinstance(holdout, (str, bytes)):
        raise StructuredDataError(
            'holdout must be a sequence of entity keys, not one string',
            code='INVALID_REQUEST',
        )
    keys = [
        [
            sessions.key_part(part)
            for part in (key if isinstance(key, (tuple, list)) else [key])
        ]
        for key in (() if holdout is None else holdout)
    ]
    spec = client._spec(model)
    try:
        finished = _run(
            client,
            {
                'action': 'fit',
                'model': spec.model_id,
                'outputs': outputs,
                'question': sessions.question(request),
                'holdout': {'rows': keys},
                'windowed': request.target_moves_with_time,
                'request_id': _request_id(),
                'inference_asked': sorted(
                    request.requested_inference.get('inference_config', {})
                ),
            },
            request,
        )
    except Exception as error:
        if getattr(error, 'status_code', None) in _SESSIONS_UNSUPPORTED:
            raise StructuredDataError(
                'this endpoint does not serve sessions; use predict() instead',
                code='SESSIONS_UNSUPPORTED',
            ) from error
        raise
    accepted = _value(finished)
    state = accepted['state']
    adapter = client._adapters.get(spec.model_id)
    if adapter is None:
        adapter = _adapter_for(spec)
        client._adapters[spec.model_id] = adapter
    fitted = FittedModel._open(
        client,
        adapter,
        state['session_id'],
        tuple(state['outputs']),
        named_outputs=outputs,
        held_back=accepted['held_back'],
        pinned_over_a_window=state['windowed'],
        question=request._question,
    )
    fitted._relational_session = state
    fitted._relational_session_pinned = bytes(finished.take(accepted['pinned']))
    client._pinned.add(fitted)
    _report_held_back(accepted['held_back'], accepted['named_holdout_keys'])
    return fitted


def predict(fitted: Any, request: Any) -> Any:
    from sdm_client.adapters.relational import _request_id

    try:
        finished = _run(
            fitted._client,
            {
                'action': 'predict',
                'state': fitted._relational_session,
                'question': sessions.question(request),
                'request_ids': [_request_id() for _ in request.payloads],
                'inference_asked': sorted(
                    request.requested_inference.get('inference_config', {})
                ),
            },
            request,
            fitted._relational_session_pinned,
        )
    except Exception as error:
        prediction._carry_the_answers(error, False)
        raise
    answered = results.expect(finished.result, results.Predicted)
    table = prediction._table(finished.take(answered.result))
    explained = (table.schema.metadata or {}).get(b'explanation')
    return prediction._result(
        table.to_pylist(),
        model=answered.model,
        request_id=answered.request_id,
        metadata=json.loads(finished.take(answered.metadata)),
        outputs=answered.outputs,
        by_step=False,
        explanation=None if explained is None else explained.decode(),
    )


def close(fitted: Any) -> None:
    finished = _run(
        fitted._client, {'action': 'close', 'state': fitted._relational_session}
    )
    state = _value(finished)
    fitted._relational_session = state
    fitted._closed = state['closed']
    if fitted._closed:
        fitted._client._pinned.discard(fitted)
