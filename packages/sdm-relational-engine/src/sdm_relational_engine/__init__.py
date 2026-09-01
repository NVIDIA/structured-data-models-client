# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from collections.abc import Callable
from typing import Any

from sdm_relational_engine._logging import (
    _ENV_KUMO_RELATIONAL_LOG,
    initialize_logging,
)
from sdm_relational_engine._singleton import Singleton
from sdm_relational_engine._version import __version__
from predictive_query.schema.typing import Dtype, Stype
from sdm_relational_engine.exceptions import (
    AuthenticationError,
    ClientInitializationError,
    GraphConstructionError,
    HTTPException,
    InvalidResponseError,
    KumoRelationalError,
    NimFailureError,
    NimTimeoutError,
    NimUnreachableError,
    UnknownDatasetError,
)
from sdm_relational_engine.client.client import NimClient, redact_url
from sdm_relational_engine.client.transport import RFMTransport

initialize_logging()


@dataclass
class GlobalState(metaclass=Singleton):
    r"""Global state needed by the RFM client.

    A deployment is addressed either by base URL (``_url``, the raw-NIM mode)
    or by ``_client_factory``, which builds a transport addressed some other
    way -- a named Databricks Model Serving endpoint, say. The two never
    coexist: every mode switch goes through :meth:`clear`, which resets both
    before the new mode installs its own. The ``_serving_*`` fields record what
    :func:`init_databricks_serving` was last called with, so repeating the call
    returns early rather than re-resolving credentials.
    """

    _url: str | None = None
    # Kept out of the repr: `global_state` is exported, and a dataclass repr
    # reaches tracebacks, notebook echoes and crash reporters. The key is
    # readable through the attribute for anyone who actually wants it.
    _api_key: str | None = field(default=None, repr=False)
    _verify_ssl: bool = True
    _timeout: float | None = None
    _max_retries: int = 3
    _client_factory: Callable[[], Any] | None = None
    _serving_kind: str | None = None
    _serving_endpoint: str | None = None
    _serving_platform_client: Any | None = None
    _serving_overrides: dict[str, Any] | None = None

    thread_local: threading.local = field(default_factory=threading.local)

    def clear(self) -> None:
        self._client_factory = None
        self._serving_kind = None
        self._serving_endpoint = None
        self._serving_platform_client = None
        self._serving_overrides = None
        if hasattr(self.thread_local, '_client'):
            try:
                self.thread_local._client.close()
            except Exception:
                pass
            del self.thread_local._client
        if hasattr(self.thread_local, '_client_config'):
            del self.thread_local._client_config
        self._url = None
        self._api_key = None
        self._verify_ssl = True
        self._timeout = None
        self._max_retries = 3

    @property
    def _config(self) -> tuple[Any, ...]:
        return (
            self._url,
            self._api_key,
            self._verify_ssl,
            self._timeout,
            self._max_retries,
            self._client_factory is not None,
            self._serving_endpoint,
            id(self._serving_platform_client),
            tuple(sorted((self._serving_overrides or {}).items())),
        )

    @property
    def initialized(self) -> bool:
        return self._url is not None or self._client_factory is not None

    @property
    def client(self) -> RFMTransport:
        r"""The request client for this thread.

        A :class:`NimClient` for the raw-NIM deployment, or whatever
        ``_client_factory`` builds otherwise.

        Cached per thread and keyed by the configuration it was built from.
        ``init()`` returns early when re-initialized with identical settings
        and only ``clear()`` evicts, and ``clear()`` reaches just the thread
        that called it -- so without the key a thread that had cached a client
        for one endpoint would keep using it after another thread pointed the
        process at a second endpoint, sending that thread's requests, and its
        API key, to the wrong deployment. A superseded client is dropped
        rather than closed: another thread's :class:`~sdm_relational_engine.rfm.KumoRelational` may
        still hold it for an in-flight prediction.

        Auto-init from an ambient ``KUMO_RELATIONAL_API_ENDPOINT`` is gated on
        ``initialized`` rather than on ``_url``, because a serving deployment
        has no URL and would otherwise have the factory it just installed
        replaced. It is skipped under pytest so a test run cannot silently
        connect to a real endpoint.
        """
        if (
            not self.initialized
            and (os.getenv('KUMO_RELATIONAL_API_ENDPOINT'))
            and 'pytest' not in sys.modules
        ):
            init()

        if not self.initialized:
            raise ValueError(
                'Client creation or authentication failed. '
                'Please re-create your client before proceeding.'
            )

        config = self._config
        if (
            hasattr(self.thread_local, '_client')
            and getattr(self.thread_local, '_client_config', None) == config
        ):
            return self.thread_local._client

        if self._client_factory is not None:
            client = self._client_factory()
        else:
            client = NimClient(
                self._url,
                self._api_key,
                verify_ssl=self._verify_ssl,
                timeout=self._timeout,
                max_retries=self._max_retries,
            )
        self.thread_local._client = client
        self.thread_local._client_config = config
        return client


global_state: GlobalState = GlobalState()


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
        global_state.clear()

    client = NimClient(
        url=url,
        api_key=api_key,
        verify_ssl=verify_ssl,
        timeout=timeout,
        max_retries=max_retries,
    )
    client.authenticate()

    global_state._url = client._url
    global_state._api_key = client._api_key
    global_state._verify_ssl = verify_ssl
    global_state._timeout = timeout
    global_state._max_retries = max_retries

    logging.getLogger('sdm_relational_engine').info(
        "Initialized KumoRelational client v%s against deployment '%s'",
        __version__,
        redact_url(url),
    )


def init_databricks_serving(
    endpoint: str,
    *,
    workspace_client: Any | None = None,
    max_request_bytes: int | None = None,
    timeout: float | None = None,
    log_level: str = 'INFO',
) -> None:
    r"""Initialize against a Databricks Model Serving endpoint.

    The counterpart to :func:`init`, which addresses a Universal TFM NIM by
    base URL. A serving endpoint is addressed by **name** through a workspace
    client, so there is no URL, no ``api_key``, and no readiness probe -- the
    endpoint does not advertise one.

    Re-initializing with identical arguments is a no-op, as for :func:`init`.
    That matters more here: when ``workspace_client`` is omitted, building the
    client resolves ambient Databricks credentials, and callers score in
    partitions by calling this once per batch. The unchanged check therefore
    runs before anything is constructed, and the endpoint is validated before
    that, so a rejected name leaves the process exactly as it was, log level
    included. Overrides are forwarded only when given, leaving the transport
    owning its own defaults and their validation.

    Args:
        endpoint: The serving endpoint name.
        workspace_client: An existing ``WorkspaceClient``. When omitted one is
            built from the ambient Databricks configuration, which is how a
            notebook authenticates without handling a token.
        max_request_bytes: Overrides the client-side request size cap. Omit to
            use the transport's own default, which tracks the platform limit.
        timeout: Overrides the request timeout, in seconds. Applies only when
            no ``workspace_client`` is supplied; an injected one keeps its own
            configuration.
        log_level: As for :func:`init`.

    Raises:
        ValueError: if ``endpoint`` is empty, whitespace-padded, or URL-shaped
            (a serving endpoint is named, not addressed), or if an override is
            out of range.
        ImportError: if the ``databricks-serving`` extra is not installed and
            no ``workspace_client`` was supplied.
        HTTPException: if a workspace client has to be built from the ambient
            configuration and that authentication fails.
    """
    overrides: dict[str, Any] = {
        name: value
        for name, value in (
            ('max_request_bytes', max_request_bytes),
            ('timeout', timeout),
        )
        if value is not None
    }

    if global_state.initialized:
        unchanged = (
            global_state._client_factory is not None
            and endpoint == global_state._serving_endpoint
            and workspace_client is global_state._serving_platform_client
            and overrides == global_state._serving_overrides
        )
        if unchanged:
            set_log_level(os.getenv(_ENV_KUMO_RELATIONAL_LOG) or log_level)
            return

    from sdm_relational_engine.client.databricks_serving import (
        DatabricksServingClient,
    )

    probe = DatabricksServingClient(endpoint, workspace_client, **overrides)

    set_log_level(os.getenv(_ENV_KUMO_RELATIONAL_LOG) or log_level)

    if global_state.initialized:
        global_state.clear()

    global_state._client_factory = lambda: DatabricksServingClient(
        endpoint, workspace_client, **overrides
    )
    global_state._serving_kind = 'databricks'
    global_state._serving_endpoint = endpoint
    global_state._serving_platform_client = workspace_client
    global_state._serving_overrides = overrides
    global_state.thread_local._client = probe
    global_state.thread_local._client_config = global_state._config

    logging.getLogger('sdm_relational_engine').info(
        'Initialized KumoRelational client v%s against Databricks Model Serving '
        "endpoint '%s'",
        __version__,
        endpoint,
    )


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

    The Snowflake counterpart to :func:`init_databricks_serving`. A model
    service is invoked as a SQL method on the service over a session, so there
    is no URL, no ``api_key``, and no readiness probe.

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

    if global_state.initialized:
        global_state.clear()

    global_state._client_factory = lambda: SnowflakeServingClient(
        service, session, **overrides
    )
    global_state._serving_kind = 'snowflake'
    global_state._serving_endpoint = service
    global_state._serving_platform_client = session
    global_state._serving_overrides = overrides
    global_state.thread_local._client = probe

    logging.getLogger('sdm_relational_engine').info(
        'Initialized Kumo Relational Client v%s against Snowflake model service %r',
        __version__,
        service,
    )


def set_log_level(level: str) -> None:
    r"""Sets the Kumo Relational logging level."""
    logging.getLogger('sdm_relational_engine').setLevel(level)


@lru_cache
def in_streamlit_notebook() -> bool:
    try:
        import streamlit  # noqa: F401
        from snowflake.snowpark.context import get_active_session

        get_active_session()
        return True
    except Exception:
        return False


@lru_cache
def in_jupyter_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if 'google.colab' in str(shell.__class__):
            return True
        if 'DatabricksShell' in str(shell.__class__):
            return True
        return shell.__class__.__name__ == 'ZMQInteractiveShell'
    except Exception:
        return False


@lru_cache
def in_vnext_notebook() -> bool:
    try:
        from snowflake.snowpark.context import get_active_session

        get_active_session()
        return in_jupyter_notebook()
    except Exception:
        return False


@lru_cache
def in_notebook() -> bool:
    return in_streamlit_notebook() or in_jupyter_notebook()


@lru_cache
def in_tmux() -> bool:
    r"""Whether output should avoid emoji.

    Terminal multiplexers commonly mis-measure the display width of an emoji
    and corrupt the rest of the line, so the table and graph banners fall back
    to plain text. Each name below marks one such environment; add another only
    alongside a terminal that shows the same defect.
    """
    return 'TMUX' in os.environ or 'HERDR_ENV' in os.environ


__all__ = [
    'AuthenticationError',
    'ClientInitializationError',
    'Dtype',
    'GraphConstructionError',
    'HTTPException',
    'InvalidResponseError',
    'KumoRelationalError',
    'NimClient',
    'NimFailureError',
    'NimTimeoutError',
    'NimUnreachableError',
    'Stype',
    'UnknownDatasetError',
    '__version__',
    'global_state',
    'init',
    'init_databricks_serving',
    'set_log_level',
]
