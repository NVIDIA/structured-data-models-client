# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import sys
import threading
from dataclasses import dataclass, field, replace
import functools
from functools import lru_cache
from collections.abc import Callable
from typing import Any, TypeVar, cast

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


@dataclass(frozen=True)
class Deployment:
    r"""Everything that says where requests go and how they authenticate.

    Held as one immutable value and replaced in a single assignment, so a
    reader sees one deployment or another and never a mix of two. Published
    field by field, there is a window in which the URL is the new deployment's
    and the API key is still the last one's, and a request made in that window
    sends that key to a host it does not belong to.
    """

    url: str | None = None
    api_key: str | None = field(default=None, repr=False)
    verify_ssl: bool = True
    timeout: float | None = None
    max_retries: int = 3
    client_factory: Callable[[], Any] | None = None
    serving_kind: str | None = None
    serving_endpoint: str | None = None
    serving_platform_client: Any | None = None
    serving_overrides: dict[str, Any] | None = None


_Configure = TypeVar('_Configure', bound=Callable[..., None])

_configure_lock = threading.RLock()

# Held only for the read-modify-write of the configuration value itself, never
# across the network. Publishing a timeout should not queue behind another
# thread's in-flight authentication.
_publish_lock = threading.RLock()


def _config_of(deployment: Deployment) -> tuple[Any, ...]:
    """What a cached client was built from, so a stale one is recognised."""
    return (
        deployment.url,
        deployment.api_key,
        deployment.verify_ssl,
        deployment.timeout,
        deployment.max_retries,
        deployment.client_factory is not None,
        deployment.serving_kind,
        deployment.serving_endpoint,
        id(deployment.serving_platform_client),
        tuple(sorted((deployment.serving_overrides or {}).items())),
    )


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

    The fields are views onto one :class:`Deployment` value. Reading any of
    them takes the whole of it once, so a reader cannot see half of one
    configuration and half of another however the writers interleave.
    """

    _deployment: Deployment = field(default_factory=Deployment)
    thread_local: threading.local = field(default_factory=threading.local)

    def _publish(self, **changes: Any) -> None:
        r"""Install a configuration change, whole, in one assignment.

        Locked because this reads the current configuration before writing the
        next: two callers changing different fields at once would otherwise
        each start from the same value and the second would drop the first's
        change. The assignment itself is what readers rely on being one step.
        """
        with _publish_lock:
            self._deployment = replace(self._deployment, **changes)

    @property
    def _url(self) -> str | None:
        return self._deployment.url

    @_url.setter
    def _url(self, value: str | None) -> None:
        self._publish(url=value)

    @property
    def _api_key(self) -> str | None:
        return self._deployment.api_key

    @_api_key.setter
    def _api_key(self, value: str | None) -> None:
        self._publish(api_key=value)

    @property
    def _verify_ssl(self) -> bool:
        return self._deployment.verify_ssl

    @_verify_ssl.setter
    def _verify_ssl(self, value: bool) -> None:
        self._publish(verify_ssl=value)

    @property
    def _timeout(self) -> float | None:
        return self._deployment.timeout

    @_timeout.setter
    def _timeout(self, value: float | None) -> None:
        self._publish(timeout=value)

    @property
    def _max_retries(self) -> int:
        return self._deployment.max_retries

    @_max_retries.setter
    def _max_retries(self, value: int) -> None:
        self._publish(max_retries=value)

    @property
    def _client_factory(self) -> Callable[[], Any] | None:
        return self._deployment.client_factory

    @_client_factory.setter
    def _client_factory(self, value: Callable[[], Any] | None) -> None:
        self._publish(client_factory=value)

    @property
    def _serving_kind(self) -> str | None:
        return self._deployment.serving_kind

    @_serving_kind.setter
    def _serving_kind(self, value: str | None) -> None:
        self._publish(serving_kind=value)

    @property
    def _serving_endpoint(self) -> str | None:
        return self._deployment.serving_endpoint

    @_serving_endpoint.setter
    def _serving_endpoint(self, value: str | None) -> None:
        self._publish(serving_endpoint=value)

    @property
    def _serving_platform_client(self) -> Any | None:
        return self._deployment.serving_platform_client

    @_serving_platform_client.setter
    def _serving_platform_client(self, value: Any | None) -> None:
        self._publish(serving_platform_client=value)

    @property
    def _serving_overrides(self) -> dict[str, Any] | None:
        return self._deployment.serving_overrides

    @_serving_overrides.setter
    def _serving_overrides(self, value: dict[str, Any] | None) -> None:
        self._publish(serving_overrides=value)

    def clear(self) -> None:
        self._deployment = Deployment()
        self._drop_cached_client()

    def _drop_cached_client(self) -> None:
        """Close and forget this thread's cached client.

        Separate from `clear` so pointing the process at a new deployment can
        drop the stale client without first emptying the deployment: a reader
        between those two steps would otherwise see an uninitialized state and
        be told the client failed to authenticate.
        """
        if hasattr(self.thread_local, '_client'):
            try:
                self.thread_local._client.close()
            except Exception:
                pass
            del self.thread_local._client
        if hasattr(self.thread_local, '_client_config'):
            del self.thread_local._client_config

    def _replace(self, **fields: Any) -> None:
        """Swap in a complete deployment in one assignment.

        Unlike `_publish`, which edits the current value, this starts from the
        defaults, so a field the caller does not name is reset rather than
        inherited from the deployment being replaced.
        """
        with _publish_lock:
            self._deployment = Deployment(**fields)
        self._drop_cached_client()

    @property
    def _config(self) -> tuple[Any, ...]:
        return _config_of(self._deployment)

    @property
    def initialized(self) -> bool:
        deployment = self._deployment
        return (
            deployment.url is not None or deployment.client_factory is not None
        )

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

        deployment = self._deployment
        config = _config_of(deployment)
        if (
            hasattr(self.thread_local, '_client')
            and getattr(self.thread_local, '_client_config', None) == config
        ):
            return self.thread_local._client

        if deployment.client_factory is not None:
            client = deployment.client_factory()
        else:
            client = NimClient(
                deployment.url,
                deployment.api_key,
                verify_ssl=deployment.verify_ssl,
                timeout=deployment.timeout,
                max_retries=deployment.max_retries,
            )
        self.thread_local._client = client
        self.thread_local._client_config = config
        return client


global_state: GlobalState = GlobalState()


def _serialized(configure: _Configure) -> _Configure:
    """Let one thread at a time point the process at a deployment."""

    @functools.wraps(configure)
    def guarded(*args: Any, **kwargs: Any) -> None:
        with _configure_lock:
            return configure(*args, **kwargs)

    return cast(_Configure, guarded)


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


@_serialized
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

    global_state._replace(
        client_factory=lambda: DatabricksServingClient(
            endpoint, workspace_client, **overrides
        ),
        serving_kind='databricks',
        serving_endpoint=endpoint,
        serving_platform_client=workspace_client,
        serving_overrides=overrides,
    )
    global_state.thread_local._client = probe
    global_state.thread_local._client_config = global_state._config

    logging.getLogger('sdm_relational_engine').info(
        'Initialized KumoRelational client v%s against Databricks Model Serving '
        "endpoint '%s'",
        __version__,
        endpoint,
    )


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
