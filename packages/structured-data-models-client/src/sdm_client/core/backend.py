# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What answers a request once the client has prepared it.

Every request is prepared the same way, whichever backend answers it. The
client checks it against the model, and the model's adapter writes it out as
the contract's request: the task, the schema, the context and the rows to
score, the outputs asked for and the inference settings, as Python data. A
backend takes that and gives back a :class:`~sdm_client.result.PredictionResult`.

:class:`NimBackend` sends it to a NIM and reads the NIM's answer back. The
backend that runs a model in this process is in :mod:`sdm_client.in_process`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Protocol

from sdm_client.core.transport import WireTransport
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:  # pragma: no cover - imported for typing only.
    from sdm_client.base import ModelAdapter
    from sdm_client.result import PredictionResult

# A NIM that does not serve the session routes at all, as opposed to one that
# has merely forgotten this session.
_SESSIONS_UNSUPPORTED = frozenset({404, 405, 501})


class Backend(Protocol):
    r"""What a client asks of whatever answers its requests.

    ``adapter`` is the model's own reader. Every backend hands what it
    answered to the adapter, which puts each answer under the key the caller
    asked about, so an answer reads the same wherever it was computed.
    """

    @property
    def url(self) -> str: ...

    #: Whether the backend can pin a context. A client refuses ``fit()`` up
    #: front when it cannot, rather than preparing a session to be told so.
    @property
    def serves_sessions(self) -> bool: ...

    def health_ready(self) -> bool: ...

    def list_models(self) -> list[str]: ...

    def predict(
        self,
        adapter: ModelAdapter,
        payload: dict[str, Any],
        outputs: tuple[str, ...],
    ) -> PredictionResult: ...

    def create_session(
        self, adapter: ModelAdapter, payload: dict[str, Any]
    ) -> str:
        r"""Pin the context in ``payload``, and return the session's id."""
        ...

    def session_predict(
        self,
        adapter: ModelAdapter,
        session_id: str,
        payload: dict[str, Any],
        outputs: tuple[str, ...],
        pinned: Mapping[str, Any],
    ) -> PredictionResult:
        r"""Score the rows in ``payload`` against a pinned context.

        ``pinned`` is what the session was opened with, minus the context:
        reading the answer needs the task and the schema it pinned.
        """
        ...

    def delete_session(self, session_id: str) -> None: ...

    def close(self) -> None: ...


class NimBackend:
    r"""A NIM: each request is sent to it, and its answer read back.

    ``transport`` carries the request, over HTTP to a NIM or through a
    Databricks Model Serving endpoint.
    """

    def __init__(self, transport: WireTransport) -> None:
        self.transport = transport

    @property
    def url(self) -> str:
        return self.transport.url

    @property
    def serves_sessions(self) -> bool:
        return getattr(self.transport, 'serves_sessions', True)

    def health_ready(self) -> bool:
        return self.transport.health_ready()

    def list_models(self) -> list[str]:
        return self.transport.list_models()

    def predict(
        self,
        adapter: ModelAdapter,
        payload: dict[str, Any],
        outputs: tuple[str, ...],
    ) -> PredictionResult:
        with _explained(adapter, payload):
            response = self.transport.predict(payload)
        return adapter.normalize(response, outputs, payload)

    def create_session(
        self, adapter: ModelAdapter, payload: dict[str, Any]
    ) -> str:
        try:
            with _explained(adapter, payload):
                created = self.transport.create_session(payload)
        except Exception as error:
            status = getattr(error, 'status_code', None)
            if status in _SESSIONS_UNSUPPORTED:
                raise StructuredDataError(
                    f'this endpoint does not serve sessions (HTTP {status}); '
                    f'use predict() instead, which sends the context with '
                    f'every call',
                    code='SESSIONS_UNSUPPORTED',
                ) from error
            raise
        session_id = created.get('session_id') or created.get('id')
        if not session_id:
            raise StructuredDataError(
                'the endpoint created a session but named no session id',
                code='INVALID_RESPONSE',
                details={'response_keys': sorted(created)},
            )
        return str(session_id)

    def session_predict(
        self,
        adapter: ModelAdapter,
        session_id: str,
        payload: dict[str, Any],
        outputs: tuple[str, ...],
        pinned: Mapping[str, Any],
    ) -> PredictionResult:
        response = self.transport.session_predict(session_id, payload)
        return adapter.normalize(response, outputs, {**pinned, **payload})

    def delete_session(self, session_id: str) -> None:
        self.transport.delete_session(session_id)

    def close(self) -> None:
        self.transport.close()


@contextlib.contextmanager
def _explained(
    adapter: ModelAdapter, payload: dict[str, Any]
) -> Iterator[None]:
    r"""Let the adapter add what a NIM's refusal left out."""
    try:
        yield
    except StructuredDataError as error:
        explained = adapter.explain_error(error, payload)
        if explained is error:
            # An adapter that had nothing to add returns the same object,
            # and `raise error from error` makes it its own cause, which
            # is a cycle for anything that walks the chain.
            raise
        raise explained from error
