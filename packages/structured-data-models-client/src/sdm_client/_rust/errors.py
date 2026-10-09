# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The core's errors and warnings as this client's exceptions and warnings.

One reviewed table, keyed on the error's code alone (decisions H6, E22):
each code maps to one :class:`~sdm_client.errors.StructuredDataError` class
and a sentence written for people, and a test iterates every code the core
has. The kind is fixed by the code in ``sdm-contract``, so it decides
nothing here, and the core's message text is never read: the exception's
text is the table's sentence, and what a caller branches on is ``code`` and
``details``. A warning's class is chosen by its code alone; its text is the
core's, which holds no input value (decision H17).

``details`` holds the error's ``fields``, decoded, with its ``kind`` and
``retryable``; a provider error also names the driver's exception class
the host saw (``driver_class``) beside the SQLSTATE and vendor code the
core keeps. The exception is raised ``from None``: no driver exception is
chained, and nothing the driver wrote reaches a traceback (decision H17).
"""

from __future__ import annotations

import warnings
from typing import Any, NoReturn

from sdm_client._rust import codec
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import (
    MissingExtraError,
    StructuredDataError,
    UnknownModelError,
    UnsupportedRequestError,
)

#: Each code the core gives: the class it raises as, and what it says.
ERRORS: dict[str, tuple[type[StructuredDataError], str]] = {
    'INVALID_REQUEST': (
        StructuredDataError,
        'the request is not one the core accepts',
    ),
    'UNSUPPORTED_REQUEST': (
        UnsupportedRequestError,
        'the request is valid, and not one the core supports yet',
    ),
    'INVALID_RESPONSE': (
        StructuredDataError,
        'the model endpoint answered with something the core cannot read',
    ),
    'INVALID_CONFIGURATION': (
        StructuredDataError,
        'the configuration of this call is not one the core accepts',
    ),
    'TRANSPORT_ERROR': (
        StructuredDataError,
        'the model endpoint could not be reached',
    ),
    'INTERNAL_ERROR': (
        StructuredDataError,
        'the core failed; this is a defect, not a problem with the request',
    ),
    'UNKNOWN_MODEL': (UnknownModelError, 'the model named does not exist'),
    'MISSING_EXTRA': (
        MissingExtraError,
        'an optional dependency this call needs is not installed',
    ),
    'INVALID_SCHEMA': (
        StructuredDataError,
        'the schema is not one the core can work with',
    ),
    'SESSIONS_UNSUPPORTED': (
        StructuredDataError,
        'the model endpoint does not support sessions',
    ),
    'UNSUPPORTED_TYPE': (
        StructuredDataError,
        'a column has a type the core does not take',
    ),
    'VALUE_OUT_OF_RANGE': (
        StructuredDataError,
        'a value is outside what its column type can hold in the core',
    ),
    'PROVIDER_SCHEMA_MISMATCH': (
        StructuredDataError,
        'the data source answered with columns other than the ones asked for',
    ),
    'PROVIDER_ERROR': (StructuredDataError, 'the data source failed'),
    'UNKNOWN_REPLY': (
        StructuredDataError,
        'the host answered a request the core did not make',
    ),
    'REPLY_MISSING': (
        StructuredDataError,
        'the host went on before answering every request',
    ),
    'WRONG_REPLY': (
        StructuredDataError,
        'the host answered a request with the wrong kind of reply',
    ),
    'RUN_FINISHED': (StructuredDataError, 'the run had already finished'),
    'RUN_CANCELLED': (StructuredDataError, 'the run was cancelled'),
    'RUN_POISONED': (
        StructuredDataError,
        'the run failed earlier and can no longer be used',
    ),
    'RUN_BUSY': (
        StructuredDataError,
        'the run was already in a call; a run takes one call at a time',
    ),
    'INVALID_MESSAGE': (
        StructuredDataError,
        'the host sent a message the core cannot read',
    ),
}

#: Warning codes the core gives, with the class each is shown as; any other
#: code is shown as :class:`~sdm_client.core.warnings.SDMWarning`. The core
#: promises no warning yet (D4's comes with T1.6b).
WARNINGS: dict[str, type[SDMWarning]] = {}


def _instance(
    cls: type[StructuredDataError],
    message: str,
    code: str,
    details: dict[str, Any],
) -> StructuredDataError:
    r"""An instance of ``cls`` with ``message``, ``code`` and ``details``.

    Some classes take their own arguments to build their message; the
    table's sentence replaces that message, so the base initializer is the
    one that runs.
    """
    error = cls.__new__(cls)
    StructuredDataError.__init__(error, message, code=code, details=details)
    if isinstance(error, UnknownModelError):
        error.model = details.get('model')  # type: ignore[assignment]
    return error


def error_from_core(
    control: bytes,
    attachments: dict[int, Any],
    *,
    driver_class: str | None = None,
) -> StructuredDataError:
    r"""The exception an ``error`` control message raises as.

    ``driver_class`` names the class of the driver exception behind a
    ``PROVIDER_ERROR``, which the run loop saw and the core never does.
    """
    message = codec.read(control, attachments)
    error = message['error']
    code = error['code']
    cls, sentence = ERRORS[code]
    details: dict[str, Any] = {
        name: _detail(value) for name, value in error.get('fields', {}).items()
    }
    details['kind'] = error['kind']
    details['retryable'] = bool(error.get('retryable', False))
    if code == 'PROVIDER_ERROR' and driver_class is not None:
        details['driver_class'] = driver_class
    return _instance(cls, sentence, code, details)


def _detail(value: Any) -> Any:
    if isinstance(value, list):
        return [codec.decode_scalar(item) for item in value]
    return codec.decode_scalar(value)


def raise_from_core(
    control: bytes,
    attachments: dict[int, Any],
    *,
    driver_class: str | None = None,
) -> NoReturn:
    r"""Raises the exception an ``error`` control message is, ``from None``."""
    raise error_from_core(
        control, attachments, driver_class=driver_class
    ) from None


def warn_from_core(control: bytes, attachments: dict[int, Any]) -> None:
    r"""Shows each warning of a ``warnings`` control message, in order."""
    message = codec.read(control, attachments)
    for warning in message['warnings']:
        category = WARNINGS.get(warning['code'], SDMWarning)
        warnings.warn(
            f'[{warning["code"]}] {warning["message"]}',
            category=category,
            stacklevel=caller_stacklevel(),
        )
