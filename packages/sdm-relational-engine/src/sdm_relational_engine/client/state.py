# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from sdm_relational_engine.client.client import NimClient
from sdm_relational_engine.client.transport import RFMTransport


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


def _config_of(deployment: Deployment) -> tuple[Any, ...]:
    """What a cached client was built from, so a stale one is recognised."""
    return (
        deployment.url,
        deployment.api_key,
        deployment.verify_ssl,
        deployment.timeout,
        deployment.max_retries,
        id(deployment.client_factory),
        deployment.serving_kind,
        deployment.serving_endpoint,
        id(deployment.serving_platform_client),
        tuple(sorted((deployment.serving_overrides or {}).items())),
    )


# Held only for the read-modify-write of the configuration value itself, never
# across the network. Publishing a timeout should not queue behind another
# thread's in-flight authentication.
_publish_lock = threading.RLock()


@dataclass
class GlobalState:
    r"""Global state needed by the RFM client.

    A deployment is addressed either by base URL (``_url``, the raw-NIM mode)
    or by ``_client_factory``, which builds a transport addressed some other
    way -- a named Snowflake model service, say. The two never coexist: every
    mode switch goes through :meth:`clear`, which resets both before the new
    mode installs its own. The ``_serving_*`` fields record what
    :func:`init_snowflake_serving` was last called with, so repeating the call
    returns early rather than resolving a session again.

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
            and os.getenv('KUMO_RELATIONAL_API_ENDPOINT')
            and 'pytest' not in sys.modules
        ):
            # Local import avoids a state -> initialization -> state cycle.
            from sdm_relational_engine.client.initialization import init

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


global_state = GlobalState()
