# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the RFM path requires of a transport, and the reply it reads back.

:class:`~sdm_relational_engine.client.client.Client` speaks HTTP to a Universal TFM NIM
and hands back a ``requests.Response``. A managed-platform transport invokes a
model served by name and hands back a :class:`ServingResponse` -- no HTTP
session, no base URL, no ``requests.Response``. None of them derives from
another.

These are :class:`~typing.Protocol` rather than a union of the concrete classes.
A union has to name every implementation, so each new platform edits this module
and drags its dependencies into anything that imports it. The shared call sites
(``RFMAPI`` and ``raise_on_error``) do not care which transport they hold, only
that it can be asked for a prediction and closed; structural typing states
exactly that, and lets a transport be added without touching this file.

``_request`` is private by convention but is part of this contract: it is the
seam the RFM path calls, so naming it here makes it a stated requirement rather
than one discovered by reading call sites.
"""

from typing import Any, Protocol, runtime_checkable

__all__ = [
    'RFMTransport',
    'ServingResponse',
    'TransportResponse',
]


@runtime_checkable
class TransportResponse(Protocol):
    """The members ``raise_on_error`` and ``RFMAPI.predict`` consume.

    ``requests.Response`` already satisfies this; :class:`ServingResponse` is
    the minimal implementation for a transport that makes no HTTP request.
    """

    status_code: int

    @property
    def ok(self) -> bool: ...

    @property
    def text(self) -> str: ...

    def json(self) -> Any: ...


@runtime_checkable
class RFMTransport(Protocol):
    """A transport the RFM path can be pointed at."""

    def _request(self, endpoint: Any, **kwargs: Any) -> TransportResponse: ...

    def close(self) -> None: ...


class ServingResponse:
    """A managed-platform reply in the shape the RFM path reads.

    Shared by every serving transport: there is no HTTP exchange behind one, so
    there is no ``requests.Response`` to return, and a per-platform stand-in
    would be several classes with one behaviour.
    """

    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        self._body = body

    @property
    def ok(self) -> bool:
        return self.status_code < 400

    @property
    def text(self) -> str:
        return self._body

    def json(self) -> Any:
        import json

        return json.loads(self._body)
