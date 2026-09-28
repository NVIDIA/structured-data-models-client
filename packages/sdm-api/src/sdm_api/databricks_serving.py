# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The envelope a Databricks Model Serving endpoint wraps a prediction in.

A served model is invoked with one ``request_json`` row and answers with one
``response_json`` row, each holding the canonical request or response as a
JSON string. That shape is frozen: every endpoint already deployed reads and
writes exactly these two columns.

Two packages speak it -- the client's serving transport and the relational
engine's own -- and they must agree byte for byte, so it lives here, below
both. Nothing here imports the Databricks SDK or opens a connection: it builds
and reads the envelope and names the gateway's failures, and each caller maps
those onto its own errors.

``MAX_REQUEST_BYTES`` is decimal MB rather than MiB because that is the
gateway's own limit; "correcting" it to MiB raises the cap by 5% and the excess
returns as a bare "request failed". It is measured on the serialized envelope,
not on the payload, because the gateway weighs what is sent and every quote in
the payload is re-escaped inside it (~1.15x).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    'MAX_REQUEST_BYTES',
    'REQUEST_COLUMN',
    'RESPONSE_COLUMN',
    'envelope',
    'error_code_of',
    'response_body',
    'status_of',
    'validate_endpoint_name',
]

REQUEST_COLUMN = 'request_json'
RESPONSE_COLUMN = 'response_json'

MAX_REQUEST_BYTES = 16 * 1000 * 1000

_ERROR_CODE_RE = re.compile(r'^[A-Z][A-Z0-9_]{0,63}$')


def validate_endpoint_name(name: object) -> str:
    r"""Reject anything URL-shaped, without echoing it.

    A serving endpoint is named, not addressed, and a pasted workspace URL can
    carry userinfo credentials or a token query parameter. So the URL scan runs
    before any check that echoes the value, and the rejection never quotes it.
    ``@`` is in the reject set because a schemeless ``user:token@host`` carries
    userinfo without any of the other markers.

    Raises:
        ValueError: With a message that does not contain ``name``.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError('endpoint must be a non-empty serving endpoint name')
    for token in ('://', '/', '?', '#', '@'):
        if token in name:
            raise ValueError(
                'endpoint must be a serving endpoint name, not a URL'
            )
    if name != name.strip():
        raise ValueError('endpoint name has surrounding whitespace')
    return name


def envelope(request: Mapping[str, Any]) -> tuple[list[dict[str, str]], int]:
    r"""The ``dataframe_records`` to send, and their size on the wire.

    The size is of the body the gateway weighs, and the records returned are
    the ones measured, so what is measured is what is sent. ``ensure_ascii``
    (the default) makes ``len`` the byte count.

    Raises:
        ValueError: If ``request`` holds a value with no strict JSON form,
            such as ``NaN``.
    """
    payload = json.dumps(request, allow_nan=False, separators=(',', ':'))
    records = [{REQUEST_COLUMN: payload}]
    body = json.dumps({'dataframe_records': records}, separators=(',', ':'))
    return records, len(body)


def response_body(reply: Any) -> str:
    r"""The single canonical response string the contract promises.

    ``reply`` is whatever the SDK returned: an object with ``predictions`` or a
    mapping holding it. Each row may be a mapping or an object.

    Raises:
        ValueError: If there is not exactly one row, or it holds no
            ``response_json`` string.
    """
    predictions = getattr(reply, 'predictions', None)
    if predictions is None and isinstance(reply, Mapping):
        predictions = reply.get('predictions')
    if not isinstance(predictions, list) or len(predictions) != 1:
        raise ValueError(
            'serving endpoint did not return exactly one prediction row'
        )
    row = predictions[0]
    if isinstance(row, Mapping):
        body = row.get(RESPONSE_COLUMN)
    else:
        body = getattr(row, RESPONSE_COLUMN, None)
    if not isinstance(body, str):
        raise ValueError(
            f'serving endpoint response row has no {RESPONSE_COLUMN!r} '
            'string column'
        )
    return body


def status_of(error: BaseException) -> int:
    r"""Map a Databricks SDK error onto an HTTP-equivalent status.

    Matched by class name so the SDK need not be importable to read this
    module, and a version bump that moves the classes cannot break the mapping
    silently. An unclassified failure is 503 rather than 500, because it is
    usually the endpoint being unavailable, and 503 is what callers retry.
    """
    names = {cls.__name__ for cls in type(error).__mro__}
    if names & {'PermissionDenied', 'Unauthenticated'}:
        return 403 if 'PermissionDenied' in names else 401
    if 'NotFound' in names:
        return 404
    if 'ResourceConflict' in names:
        return 409
    if names & {'TooManyRequests', 'RequestLimitExceeded'}:
        return 429
    if names & {
        'DeadlineExceeded',
        'OperationTimeout',
        'TimeoutError',
        'Timeout',
    }:
        return 504
    if 'BadRequest' in names:
        return 400
    status = getattr(error, 'status_code', None)
    return status if isinstance(status, int) else 503


def error_code_of(error: BaseException) -> str | None:
    r"""The provider's error code, when enum-shaped enough to be non-secret."""
    code = getattr(error, 'error_code', None)
    if isinstance(code, str) and _ERROR_CODE_RE.match(code):
        return code
    return None
