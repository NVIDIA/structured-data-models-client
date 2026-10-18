# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Refusing a request the way a NIM does.

An in-process refusal carries the HTTP status and code a NIM answers the same
request with, so code that branches on either behaves alike wherever the model
runs.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from sdm_client.errors import NimRequestError

_STATUS = {
    'INVALID_REQUEST': 400,
    'SESSION_NOT_FOUND': 404,
    'INVALID_SCHEMA': 422,
    'VALIDATION_FAILED': 422,
    'UNSUPPORTED_DTYPE': 422,
    'UNSUPPORTED_TASK': 422,
    'UNSUPPORTED_OUTPUT': 422,
    'UNSUPPORTED_FEATURE': 422,
    'TASK_SCHEMA_MISMATCH': 422,
    'SESSION_CREATE_VALIDATION_FAILED': 422,
    'SESSION_PREDICT_SCHEMA_MISMATCH': 422,
    'INTERNAL_ERROR': 500,
    'MODEL_NOT_READY': 503,
}


#: The refusals a NIM reports as a session's own when they arise in one.
_SESSION_VALIDATION = frozenset(
    {
        'INVALID_SCHEMA',
        'TASK_SCHEMA_MISMATCH',
        'UNSUPPORTED_DTYPE',
        'VALIDATION_FAILED',
    }
)


def refusal(code: str, message: str) -> NimRequestError:
    r"""The error a NIM answers with for ``code``."""
    return NimRequestError(_STATUS[code], code=code, message=message)


@contextlib.contextmanager
def in_session(code: str) -> Iterator[None]:
    r"""Report a request refused inside a session the way a NIM does.

    A NIM files a refusal raised while creating a session, or scoring against
    one, under that session's own code, keeping its message.
    """
    try:
        yield
    except NimRequestError as error:
        if error.code not in _SESSION_VALIDATION:
            raise
        raise refusal(code, error.message) from error
