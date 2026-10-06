# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import logging
import os
import threading
from collections.abc import Callable
from typing import Any, TypeVar, cast

from sdm_relational_engine._version import __version__
from sdm_relational_engine.client.client import NimClient, redact_url
from sdm_relational_engine.client.state import global_state
from sdm_relational_engine.client.transport import RFMTransport
from sdm_relational_engine.utils.logging import _ENV_KUMO_RELATIONAL_LOG

_Configure = TypeVar('_Configure', bound=Callable[..., None])

_configure_lock = threading.RLock()


def _serialized(configure: _Configure) -> _Configure:
    """Let one thread at a time point the process at a deployment."""

    @functools.wraps(configure)
    def guarded(*args: Any, **kwargs: Any) -> None:
        with _configure_lock:
            return configure(*args, **kwargs)

    return cast(_Configure, guarded)


def set_log_level(level: str) -> None:
    r"""Sets the Kumo Relational logging level."""
    logging.getLogger('sdm_relational_engine').setLevel(level)


@_serialized
def init(
    url: str | None = None,
    api_key: str | None = None,
    verify_ssl: bool = True,
    log_level: str = 'INFO',
    timeout: float | None = None,
    max_retries: int = 3,
) -> None:
    r"""Initializes the Kumo Relational client against a Universal TFM NIM.

    NIMs are unauthenticated by contract, so ``api_key`` is optional and only
    needed when a deployment fronts the NIM with an authenticating gateway.
    Re-initializing with different settings reconfigures the client and closes
    the previous session; re-initializing with identical settings is a no-op.

    ``timeout`` bounds each individual request attempt, in seconds; ``None``
    leaves requests unbounded. ``max_retries`` bounds the transport-level
    retries of a transient failure; ``0`` disables them.
    """
    set_log_level(os.getenv(_ENV_KUMO_RELATIONAL_LOG) or log_level)

    api_key = api_key or os.getenv('KUMO_RELATIONAL_API_KEY')
    url = url or os.getenv('KUMO_RELATIONAL_API_ENDPOINT')
    if not url:
        raise ValueError(
            'Kumo Relational initialization failed since no endpoint '
            'URL was provided. Please either set the '
            "'KUMO_RELATIONAL_API_ENDPOINT' environment variable or "
            'explicitly call `sdm_relational_engine.init(url=...)`.'
        )

    if global_state.initialized:
        unchanged = (
            url == global_state._url
            and api_key == global_state._api_key
            and verify_ssl == global_state._verify_ssl
            and timeout == global_state._timeout
            and max_retries == global_state._max_retries
        )
        if unchanged:
            return

    client = NimClient(
        url=url,
        api_key=api_key,
        verify_ssl=verify_ssl,
        timeout=timeout,
        max_retries=max_retries,
    )
    client.authenticate()

    global_state._replace(
        url=client._url,
        api_key=client._api_key,
        verify_ssl=verify_ssl,
        timeout=timeout,
        max_retries=max_retries,
    )

    logging.getLogger('sdm_relational_engine').info(
        "Initialized KumoRelational client v%s against deployment '%s'",
        __version__,
        redact_url(url),
    )


def _init_client(
    url: str | None = None,
    api_key: str | None = None,
    verify_ssl: bool = True,
    log_level: str = 'INFO',
    timeout: float | None = None,
    max_retries: int = 3,
) -> RFMTransport:
    with _configure_lock:
        init(
            url=url,
            api_key=api_key,
            verify_ssl=verify_ssl,
            log_level=log_level,
            timeout=timeout,
            max_retries=max_retries,
        )
        return global_state.client


@_serialized
def init_snowflake_serving(
    service: str,
    *,
    session: Any | None = None,
    method: str = 'PREDICT',
    max_request_bytes: int | None = None,
    timeout: float | None = None,
    log_level: str = 'INFO',
) -> None:
    r"""Initialize against a model served on Snowpark Container Services.

    The counterpart to :func:`init`, which addresses a Universal TFM NIM by
    base URL. A model service is invoked as a SQL method on the service over a
    session, so there is no URL, no ``api_key``, and no readiness probe.

    Re-initializing with identical arguments is a no-op, as for :func:`init`,
    and the unchanged check runs before anything is constructed so that
    repeating the call cannot resolve a session again.

    Args:
        service: The service name, optionally qualified as
            ``DATABASE.SCHEMA.SERVICE``.
        session: An existing Snowpark ``Session`` or ``snowflake.connector``
            connection. When omitted the active Snowpark session is used, which
            is how a Snowflake notebook connects without handling credentials.
        method: The service method to call. ``PREDICT`` is what the model
            registry generates for a ``CustomModel``.
        max_request_bytes: Overrides the client-side payload cap. Omit to use
            the transport's own default, which tracks the platform limit.
        timeout: Overrides the per-statement timeout, in seconds.
        log_level: As for :func:`init`.

    Raises:
        ValueError: if ``service`` is empty or is not a bare, optionally
            qualified service name, or if an override is out of range.
        ImportError: if the ``snowflake-serving`` extra is not installed and no
            ``session`` was supplied.
        HTTPException: if no session was supplied and no active Snowflake
            session exists.
    """
    overrides: dict[str, Any] = {
        name: value
        for name, value in (
            ('max_request_bytes', max_request_bytes),
            ('timeout', timeout),
        )
        if value is not None
    }
    overrides['method'] = method

    if global_state.initialized:
        unchanged = (
            global_state._client_factory is not None
            and global_state._serving_kind == 'snowflake'
            and service == global_state._serving_endpoint
            and session is global_state._serving_platform_client
            and overrides == global_state._serving_overrides
        )
        if unchanged:
            set_log_level(os.getenv(_ENV_KUMO_RELATIONAL_LOG) or log_level)
            return

    from sdm_relational_engine.client.snowflake_serving import (
        SnowflakeServingClient,
    )

    probe = SnowflakeServingClient(service, session, **overrides)

    set_log_level(os.getenv(_ENV_KUMO_RELATIONAL_LOG) or log_level)

    global_state._replace(
        client_factory=lambda: SnowflakeServingClient(
            service, session, **overrides
        ),
        serving_kind='snowflake',
        serving_endpoint=service,
        serving_platform_client=session,
        serving_overrides=overrides,
    )
    global_state.thread_local._client = probe
    global_state.thread_local._client_config = global_state._config

    logging.getLogger('sdm_relational_engine').info(
        'Initialized Kumo Relational Client v%s against Snowflake model service %r',
        __version__,
        service,
    )
