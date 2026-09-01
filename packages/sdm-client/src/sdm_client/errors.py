# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The exceptions this client raises.

Every call on :class:`~sdm_client.StructuredDataClient` raises
:class:`StructuredDataError` or one of its subclasses, so ``except
StructuredDataError`` is enough to catch anything the client reports.

Every error carries a ``code``: a short stable string meant to be branched on.
Messages are written for people and will change; codes will not.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

#: Entries rendered from an RFC-9457 ``invalid_params`` list before the rest
#: are summarised as a count. A rejected batch can name thousands of rows.
MAX_INVALID_PARAMS = 5


def format_invalid_params(params: Sequence[Any] | None) -> str:
    r"""Render an endpoint's per-field validation diagnosis into a suffix.

    A validation failure answers with RFC-9457 ``invalid_params``, naming the
    exact table, row and column it rejected, while the top-level ``detail`` is
    often only "Request validation failed." Dropping the list leaves the caller
    with nothing to act on.

    Returns the empty string when there is nothing to render, so callers can
    concatenate unconditionally.
    """
    if not isinstance(params, (list, tuple)):
        return ''
    entries = []
    for param in params[:MAX_INVALID_PARAMS]:
        if not isinstance(param, dict):
            continue
        name, reason = param.get('name'), param.get('reason')
        if name and reason:
            entries.append(f'{name}: {reason}')
        elif name or reason:
            entries.append(str(name or reason))
    if not entries:
        return ''
    omitted = len(params) - len(entries)
    more = f' (and {omitted} more)' if omitted > 0 else ''
    return ' ' + '; '.join(entries) + more


class StructuredDataError(Exception):
    r"""Base class for every error this client raises.

    Attributes:
        message: The human-readable description, without the code prefix.
        code: A short stable identifier, e.g. ``'TRANSPORT_ERROR'`` or
            ``'INVALID_REQUEST'``. Branch on this rather than on the message.
        details: Whatever structured context the failure carried, such as the
            ``invalid_params`` entries from a NIM validation error. Empty when
            there is none.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}

    def __str__(self) -> str:
        if self.code:
            return f'[{self.code}] {self.message}'
        return self.message


class UnknownModelError(StructuredDataError):
    r"""A request named a model this client cannot send.

    The contract's ``model`` field is a closed enum, so an id it does not carry
    cannot be sent at all -- the message names the change that would allow it
    rather than letting the request fail at the endpoint. Code:
    ``UNKNOWN_MODEL``.
    """

    def __init__(self, model: str, known: list[str]) -> None:
        super().__init__(
            f'Unknown model {model!r}; this client can send: {sorted(known)}. '
            f'The Universal TFM contract declares these as a closed set, so a '
            f'new model id has to be added to the contract and the bindings '
            f'regenerated before it can be sent.',
            code='UNKNOWN_MODEL',
            details={'model': model, 'known_models': sorted(known)},
        )
        self.model = model


class UnsupportedRequestError(StructuredDataError):
    r"""The model cannot answer this request, and says so before sending it.

    Raised when a request asks for something the model's profile rejects --
    related tables for a single-table model, an output field it does not
    produce -- or exceeds a limit the endpoint enforces. Code:
    ``UNSUPPORTED_REQUEST``.
    """

    def __init__(
        self, message: str, *, details: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message, code='UNSUPPORTED_REQUEST', details=details)


class MissingExtraError(StructuredDataError):
    r"""An optional dependency this call needs is not installed.

    The relational engine and the connector drivers ship as extras, so the
    client can be installed without them. The message names the ``pip install``
    that fixes it. Code: ``MISSING_EXTRA``.
    """

    def __init__(self, extra: str, package: str) -> None:
        super().__init__(
            f'{package!r} is required for this call; install it with '
            f'`pip install sdm-client[{extra}]`',
            code='MISSING_EXTRA',
            details={'extra': extra, 'package': package},
        )


class NimRequestError(StructuredDataError):
    r"""The NIM answered, and the answer was an error.

    Distinguished from :class:`StructuredDataError` by carrying the HTTP
    :attr:`status_code`, which is what tells a caller whether retrying can
    help: 5xx and 429 are worth retrying, 4xx means the request has to change.

    Attributes:
        status_code: The HTTP status the NIM returned.
    """

    def __init__(
        self,
        status_code: int,
        *,
        code: str | None,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)
        self.status_code = status_code

    def __str__(self) -> str:
        if self.code:
            return f'[{self.status_code} {self.code}] {self.message}'
        return f'[{self.status_code}] {self.message}'
