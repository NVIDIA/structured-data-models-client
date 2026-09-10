# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""One connection to one NIM, serving every model it can reach.

The model is an argument:

>>> with SDMClient(url='http://localhost:8000') as client:
...     result = client.predict(request, model=KumoTabular())

Each client owns its own transport, so several can target different endpoints
or tenants at once, including from several threads: a prediction always goes to
the endpoint and credential of the client that started it.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import TracebackType
from typing import Any

from sdm_client.base import (
    ModelAdapter,
    ModelCapabilities,
    ModelSpec,
    check_request_against_model,
    describe,
)
from sdm_client.core.transport import Transport, redact_url
from sdm_client.errors import StructuredDataError, UnknownModelError
from sdm_client.models import MODELS, Model, spec_for
from sdm_client.request import Request
from sdm_client.result import PredictionResult

# A NIM that does not serve the session routes at all, as opposed to one that
# has merely forgotten this session.
_SESSIONS_UNSUPPORTED = frozenset({404, 405, 501})


def _adapter_for(spec: ModelSpec) -> ModelAdapter:
    r"""The adapter that shapes requests for ``spec``.

    Imported here rather than at module scope so that importing the client does
    not pull in an adapter that reaches for an optional driver.
    """
    if spec.model_id == 'kumo-tabular':
        from sdm_client.adapters.tabular import TabularAdapter

        return TabularAdapter()
    if spec.model_id == 'kumo-relational':
        from sdm_client.adapters.relational import RelationalAdapter

        return RelationalAdapter()
    raise UnknownModelError(spec.model_id, list(MODELS))


class FittedModel:
    r"""A context pinned on the endpoint, ready to score against.

    Returned by :meth:`SDMClient.fit`. The contract's
    ``POST /v1/sessions`` uploads a context and runs no inference; scoring
    against it is a second call. That is what ``fit`` and ``predict`` are here.

    Close it when done, or use it as a context manager. A session holds memory
    on the endpoint until it is released or expires.
    """

    def __init__(
        self,
        client: SDMClient,
        adapter: ModelAdapter,
        session_id: str,
        outputs: tuple[str, ...],
        pinned: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._adapter = adapter
        self._session_id = session_id
        self._outputs = outputs
        self._pinned = pinned or {}
        self._closed = False

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def model(self) -> str:
        return self._adapter.spec.model_id

    def predict(self, request: Request) -> PredictionResult:
        r"""Score ``request``'s rows against the pinned context.

        Only the rows travel: the context, schema and task were pinned when
        this handle was created.
        """
        if self._closed:
            raise StructuredDataError(
                'this fitted model is closed; fit again to score more rows',
                code='INVALID_CONFIGURATION',
            )
        # A session split the request in two: model, task, schema and context
        # were pinned at fit time and only the rows travel now. Reading the
        # answer needs both halves -- the key an id stands for is named in the
        # pinned schema -- so the adapter is handed the whole request while the
        # wire gets only the half the contract allows here.
        parts = [
            self._adapter.normalize(
                self._client._transport.session_predict(
                    self._session_id, payload
                ),
                self._outputs,
                {**self._pinned, **payload},
            )
            for payload in self._adapter.session_prediction_plan(
                request, self._outputs
            )
        ]
        if not parts:
            raise StructuredDataError(
                'the adapter produced no scoring call for this request',
                code='INVALID_REQUEST',
            )
        return PredictionResult.merge(parts)

    def close(self) -> None:
        r"""Release the pinned context. Safe to call twice."""
        if self._closed:
            return
        self._closed = True
        self._client._transport.delete_session(self._session_id)

    def __enter__(self) -> FittedModel:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = 'closed' if self._closed else 'open'
        return f'FittedModel({self.model!r}, {state})'


class SDMClient:
    r"""A connection to one Universal TFM NIM.

    Use it as a context manager, or call :meth:`close`.
    """

    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        *,
        verify_ssl: bool = True,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        r"""Open a client against one NIM.

        No request is made here: the endpoint is first contacted by
        :meth:`health_ready`, :meth:`models`, or a prediction.

        Args:
            url: Base URL of the NIM, e.g. ``'http://localhost:8000'``.
            api_key: Sent as ``X-API-Key`` when an authenticating gateway
                fronts the NIM. Refused over plaintext ``http://`` to a
                non-local host.
            verify_ssl: Whether to verify the server's TLS certificate.
            timeout: Seconds to wait for each individual attempt, not for the
                call as a whole.
            max_retries: Transport-level retries of a transient failure (429,
                500, 502, 503, 504, or a dropped connection) with exponential
                backoff. ``POST /v1/sessions`` is excluded: a re-sent create
                would orphan a pinned context on the endpoint.
        """
        self._transport = Transport(
            url,
            api_key,
            verify_ssl=verify_ssl,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._adapters: dict[str, ModelAdapter] = {}

    @property
    def url(self) -> str:
        r"""The NIM base URL this client was opened against."""
        return self._transport.url

    def models(self) -> list[str]:
        r"""The model ids **the endpoint** advertises.

        Read from ``/v1/models``, not from this client's own list. The previous
        generation reported its own registry here, which meant it claimed a
        model the endpoint did not serve.
        """
        return self._transport.list_models()

    def capabilities(self, model: str | Model) -> ModelCapabilities:
        r"""What ``model`` supports, as far as this client knows.

        Describes the client side and is tagged ``source='sdk'``. It needs no
        live connection, and it is not the endpoint's own answer -- use
        :meth:`models` for that.
        """
        return describe(self._spec(model))

    def health_ready(self) -> bool:
        r"""Whether the NIM answers ``GET /v1/health/ready`` with 200."""
        return self._transport.health_ready()

    def predict(
        self,
        request: Request,
        *,
        model: str | Model,
        outputs: Sequence[str] | None = None,
    ) -> PredictionResult:
        r"""Answer ``request`` with ``model``.

        Args:
            request: What to ask. Built directly, or by a predictive query.
            model: A model handle such as ``KumoTabular()``, or a model id.
            outputs: The output fields to ask for. Defaults to what the model
                produces for this task kind.

        Returns:
            A :class:`~sdm_client.result.PredictionResult`, always -- never a
            frame-or-explanation union.

        Raises:
            UnknownModelError: If ``model`` is not one the contract carries.
            UnsupportedRequestError: If the model cannot answer this request.
                Raised before anything is sent.
            NimRequestError: If the endpoint answered with an error.
        """
        adapter, fields = self._prepare(request, model, outputs)
        # Every call's rows are kept. A request past the endpoint's per-call
        # cap is split by the adapter, and each call answers about its own
        # rows, so returning only the last would drop the earlier batches.
        parts = [
            adapter.normalize(self._send(adapter, payload), fields, payload)
            for payload in adapter.prediction_plan(request, fields)
        ]
        if not parts:
            raise StructuredDataError(
                'the adapter produced no call for this request',
                code='INVALID_REQUEST',
            )
        return PredictionResult.merge(parts)

    def _send(self, adapter: Any, payload: dict[str, Any]) -> dict[str, Any]:
        r"""Post one payload, letting the adapter explain a refusal."""
        try:
            return self._transport.predict(payload)
        except StructuredDataError as error:
            explained = adapter.explain_error(error, payload)
            if explained is error:
                # An adapter that had nothing to add returns the same object,
                # and `raise error from error` makes it its own cause, which
                # is a cycle for anything that walks the chain.
                raise
            raise explained from error

    def fit(
        self,
        request: Request,
        *,
        model: str | Model,
        outputs: Sequence[str] | None = None,
    ) -> FittedModel:
        r"""Pin ``request``'s context on the endpoint and return a handle.

        The pinned half is model, task, schema and context; later calls send
        only the rows to score. Use this when asking several questions of the
        same context, which is what makes uploading it once worthwhile.

        Raises:
            UnsupportedRequestError: If the endpoint does not serve sessions.
        """
        adapter, fields = self._prepare(request, model, outputs)
        payload = adapter.session_plan(request, fields)
        try:
            try:
                created = self._transport.create_session(payload)
            except StructuredDataError as error:
                explained = adapter.explain_error(error, payload)
                if explained is error:
                    raise
                raise explained from error
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
        return FittedModel(self, adapter, str(session_id), fields, payload)

    def _prepare(
        self,
        request: Request,
        model: str | Model,
        outputs: Sequence[str] | None,
    ) -> tuple[ModelAdapter, tuple[str, ...]]:
        r"""Resolve the adapter and output fields, and refuse early.

        Every check that can be made without the network is made here, so a
        request the model cannot answer fails with a message naming the model
        rather than as an opaque 422.
        """
        if not isinstance(request, Request):
            raise StructuredDataError(
                f'request must be a Request, got {type(request).__name__}',
                code='INVALID_REQUEST',
            )
        spec = self._spec(model)
        adapter = self._adapters.get(spec.model_id)
        if adapter is None:
            adapter = _adapter_for(spec)
            self._adapters[spec.model_id] = adapter

        fields = (
            tuple(outputs)
            if outputs is not None
            else adapter.default_output(request)
        )
        if not fields:
            raise StructuredDataError(
                'outputs must name at least one field',
                code='INVALID_REQUEST',
            )
        if isinstance(outputs, (str, bytes)):
            raise StructuredDataError(
                f'outputs must be a sequence of field names, not a single '
                f'string; pass [{outputs!r}]',
                code='INVALID_REQUEST',
            )
        check_request_against_model(request, spec, fields)
        return adapter, fields

    @staticmethod
    def _spec(model: str | Model) -> ModelSpec:
        return spec_for(model)

    def close(self) -> None:
        r"""Release pooled connections and retire this client.

        Adapters are asked to release what they opened first, then the
        transport is closed even if one of them fails -- the transport's pool
        is the one that cannot be rebuilt on demand.
        """
        try:
            error: BaseException | None = None
            for adapter in self._adapters.values():
                try:
                    adapter.close()
                except BaseException as adapter_error:  # noqa: BLE001
                    error = error or adapter_error
            if error is not None:
                raise error
        finally:
            self._transport.close()

    def __enter__(self) -> SDMClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f'SDMClient(url={redact_url(self._transport.url)!r})'
