# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import threading
from dataclasses import dataclass, field

import sdm_relational_engine
from sdm_relational_engine.client.transport import RFMTransport

from .base import Table
from .diagnostics import (
    GraphSanitizationReport,
    SanitizationStatus,
    TableSanitizationReport,
    TaskReferenceError,
)
from .graph import Graph, ViewConversionWarning
from .task_table import TaskTable
from .rfm import (
    ExplainConfig,
    Explanation,
    KumoRelational,
    MaterializedPredictionRequest,
)

logger = logging.getLogger('sdm_relational_engine_rfm')

_CLIENT_TOKEN = object()

_DIRECT_USE_MESSAGE = (
    'Direct use of the relational engine is not supported. Run inference '
    'through the client:\n'
    '    from sdm_client import (\n'
    '        PredictiveQuery, SDMClient, KumoRelational,\n'
    '    )\n'
    '    request = PredictiveQuery(query, tables).materialize(indices=[...])\n'
    '    with SDMClient(url=...) as client:\n'
    '        client.predict(request, model=KumoRelational())'
)


_URL_NOT_PROVIDED = '__url_not_provided__'


@dataclass
class RfmGlobalState:
    _url: str = _URL_NOT_PROVIDED

    # Thread-safe init-once. Built per instance: a bare `threading.Lock()`
    # default is evaluated once at class-definition time, so every instance
    # would share one lock and `reset()` would serialize across all of them.
    _initialized: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def client(self) -> RFMTransport:
        if not self._initialized:
            raise RuntimeError(_DIRECT_USE_MESSAGE)
        return sdm_relational_engine.global_state.client

    def reset(self) -> None:  # For testing only.
        with self._lock:
            self._initialized = False
            self._url = _URL_NOT_PROVIDED


global_state = RfmGlobalState()


def _configure(
    url: str | None,
    api_key: str | None,
    verify_ssl: bool,
    log_level: str,
    timeout: float | None,
    max_retries: int = 3,
) -> None:
    resolved_url = url or os.getenv('KUMO_RELATIONAL_API_ENDPOINT')

    sdm_relational_engine.init(
        url=resolved_url,
        api_key=api_key,
        verify_ssl=verify_ssl,
        log_level=log_level,
        timeout=timeout,
        max_retries=max_retries,
    )

    global_state._url = sdm_relational_engine.global_state._url
    global_state._initialized = True


def init(
    url: str | None = None,
    api_key: str | None = None,
    verify_ssl: bool = True,
    log_level: str = 'INFO',
    *,
    timeout: float | None = None,
    max_retries: int = 3,
    _token: object | None = None,
) -> None:
    if _token is not _CLIENT_TOKEN:
        raise RuntimeError(_DIRECT_USE_MESSAGE)
    with global_state._lock:
        _configure(url, api_key, verify_ssl, log_level, timeout, max_retries)


def close_client(_token: object | None = None) -> None:
    r"""Releases this thread's pooled connections to the engine's endpoint.

    The counterpart to :func:`init_client`, for a caller that is done with the
    endpoint it configured. The client is closed and evicted, but the
    configuration is left in place, so the next prediction on this thread
    rebuilds a client and carries on rather than failing. That is what makes
    this safe to call while another caller is still using the same endpoint:
    the cost of releasing early is a reconnect.

    Only this thread's client is released. Another thread's is its own, and
    closing it from here would break a prediction already in flight on it.

    Does nothing if this thread never built a client.
    """
    if _token is not _CLIENT_TOKEN:
        raise RuntimeError(_DIRECT_USE_MESSAGE)
    with global_state._lock:
        thread_local = sdm_relational_engine.global_state.thread_local
        client = getattr(thread_local, '_client', None)
        if client is None:
            return
        del thread_local._client
        if hasattr(thread_local, '_client_config'):
            del thread_local._client_config
        client.close()


def init_client(
    url: str | None = None,
    api_key: str | None = None,
    verify_ssl: bool = True,
    log_level: str = 'INFO',
    *,
    timeout: float | None = None,
    max_retries: int = 3,
    _token: object | None = None,
) -> RFMTransport:
    r"""Configures the engine and returns the client that configuration
    resolves to, as one atomic step.

    The engine's endpoint and credential live in a process-wide singleton, so
    configuring it and then reading back the resulting client as two steps is a
    race: a second caller configuring a different endpoint in between makes the
    first caller read the second's client. Because
    :attr:`~sdm_relational_engine.rfm.KumoRelational._api_client` resolves lazily -- at the first
    HTTP call, after graph materialization and subgraph sampling -- that window
    is wide enough to matter in practice.

    Resolving under the same lock that guards configuration closes it, provided
    the caller binds the returned client to its prediction rather than reading
    the global again later. Pass it to :class:`~sdm_relational_engine.rfm.KumoRelational` as
    ``_client``.
    """
    if _token is not _CLIENT_TOKEN:
        raise RuntimeError(_DIRECT_USE_MESSAGE)
    with global_state._lock:
        _configure(url, api_key, verify_ssl, log_level, timeout, max_retries)
        return sdm_relational_engine.global_state.client


def init_databricks_serving(
    endpoint: str,
    *,
    workspace_client: object | None = None,
    max_request_bytes: int | None = None,
    timeout: float | None = None,
    log_level: str = 'INFO',
    _token: object | None = None,
) -> None:
    r"""Initialize against a Databricks Model Serving endpoint.

    The counterpart to :func:`init`, which resolves a NIM base URL from its
    argument or the environment; there is no URL to resolve here.

    Gated like :func:`init`: the engine is reachable only through
    ``SDMClient``, so both entry points must refuse a direct call rather than
    leaving one of them as a way around the boundary.
    """
    if _token is not _CLIENT_TOKEN:
        raise RuntimeError(_DIRECT_USE_MESSAGE)
    with global_state._lock:
        sdm_relational_engine.init_databricks_serving(
            endpoint,
            workspace_client=workspace_client,
            max_request_bytes=max_request_bytes,
            timeout=timeout,
            log_level=log_level,
        )
        global_state._url = f'databricks-serving:{endpoint}'
        global_state._initialized = True


LocalGraph = Graph  # NOTE Backward compatibility - do not use anymore.


def init_snowflake_serving(
    service: str,
    *,
    session: object | None = None,
    method: str = 'PREDICT',
    max_request_bytes: int | None = None,
    timeout: float | None = None,
    log_level: str = 'INFO',
    _token: object | None = None,
) -> None:
    """Initialize against a model served on Snowpark Container Services.

    The Snowflake counterpart to :func:`init_databricks_serving`. A service is
    invoked as a SQL method over a session, so there is no URL to resolve.

    Gated like :func:`init`: the engine is reachable only through
    ``SDMClient``, so both entry points must refuse a direct call rather than
    leaving one of them as a way around the boundary.
    """
    if _token is not _CLIENT_TOKEN:
        raise RuntimeError(_DIRECT_USE_MESSAGE)
    with global_state._lock:
        sdm_relational_engine.init_snowflake_serving(
            service,
            session=session,
            method=method,
            max_request_bytes=max_request_bytes,
            timeout=timeout,
            log_level=log_level,
        )
        global_state._url = f'snowflake-serving:{service}'
        global_state._initialized = True


LocalGraph = Graph  # NOTE Backward compatibility - do not use anymore.


__all__ = [
    'ExplainConfig',
    'Explanation',
    'Graph',
    'GraphSanitizationReport',
    'KumoRelational',
    'LocalTable',
    'MaterializedPredictionRequest',
    'SanitizationStatus',
    'Table',
    'TableSanitizationReport',
    'TaskReferenceError',
    'TaskTable',
    'ViewConversionWarning',
    'close_client',
    'init',
    'init_client',
    'init_databricks_serving',
    'init_snowflake_serving',
]


def __getattr__(name: str) -> object:
    r"""Resolve ``LocalTable`` only when something asks for it.

    Importing it here would pull in ``backend.local``, which requires the
    compiled neighbour sampler at import time. That made every path through
    this package depend on the extension, including the Databricks and
    Snowflake serving paths, which run inference on the endpoint and never
    sample locally. Deferring the import keeps the requirement where the
    requirement actually is.
    """
    if name == 'LocalTable':
        from .backend.local import LocalTable

        return LocalTable
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
