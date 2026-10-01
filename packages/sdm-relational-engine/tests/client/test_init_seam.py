# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Initializing against something other than a base URL.

The RFM path reads its client from ``GlobalState.client``. That used to be
hardwired to build a ``NimClient`` from ``_url``, so a deployment addressed by
name rather than by URL was unreachable. These tests cover the seam and, more
importantly, that the raw-NIM path is unchanged by it.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
import sdm_relational_engine
from sdm_relational_engine.client.client import NimClient
from sdm_relational_engine.client.databricks_serving import (
    DatabricksServingClient,
)


class _Endpoints:
    @staticmethod
    def query(**kwargs: Any) -> Any:
        return {'predictions': [{'response_json': '{}'}]}


class _Workspace:
    serving_endpoints = _Endpoints()


@pytest.fixture(autouse=True)
def _clean_state() -> Any:
    sdm_relational_engine.global_state.clear()
    yield
    sdm_relational_engine.global_state.clear()


# -- the new mode ----------------------------------------------------------


def test_initializes_without_a_url() -> None:
    sdm_relational_engine.init_databricks_serving(
        'kumo-relational', workspace_client=_Workspace()
    )
    assert sdm_relational_engine.global_state.initialized
    assert sdm_relational_engine.global_state._url is None
    assert isinstance(
        sdm_relational_engine.global_state.client, DatabricksServingClient
    )


def test_a_bad_endpoint_leaves_state_untouched() -> None:
    """Validate before mutating: a rejected endpoint name must not leave the
    process half-initialized.
    """
    with pytest.raises(ValueError):
        sdm_relational_engine.init_databricks_serving(
            'https://workspace/serving-endpoints/relational-endpoint',
            workspace_client=_Workspace(),
        )
    assert not sdm_relational_engine.global_state.initialized


def test_repeated_init_does_not_rebuild_the_client() -> None:
    """Building the transport resolves Databricks credentials, and the Kumo Relational
    adapter re-initializes on every predict. An unchanged re-init must be a
    no-op rather than a fresh authentication per scored partition.
    """
    workspace = _Workspace()
    sdm_relational_engine.init_databricks_serving(
        'kumo-relational', workspace_client=workspace
    )
    first_client = sdm_relational_engine.global_state.client

    sdm_relational_engine.init_databricks_serving(
        'kumo-relational', workspace_client=workspace
    )

    assert sdm_relational_engine.global_state.client is first_client


def test_changed_arguments_still_rebuild_the_client() -> None:
    """The fast path must key on the arguments, not merely on being
    initialized, or a caller could never switch endpoint or workspace.
    """
    sdm_relational_engine.init_databricks_serving(
        'kumo-relational', workspace_client=_Workspace()
    )
    first_client = sdm_relational_engine.global_state.client

    sdm_relational_engine.init_databricks_serving(
        'relational-endpoint-2', workspace_client=_Workspace()
    )
    assert sdm_relational_engine.global_state.client is not first_client
    assert (
        sdm_relational_engine.global_state.client.endpoint
        == 'relational-endpoint-2'
    )


def test_transport_overrides_reach_the_client() -> None:
    """The size cap and timeout are constructor arguments on the transport;
    without plumbing them through init no caller can reach them.
    """
    sdm_relational_engine.init_databricks_serving(
        'kumo-relational',
        workspace_client=_Workspace(),
        max_request_bytes=1234,
        timeout=12.5,
    )
    client = sdm_relational_engine.global_state.client
    assert client._max_request_bytes == 1234
    assert client._timeout == 12.5


def test_clear_resets_the_factory() -> None:
    sdm_relational_engine.init_databricks_serving(
        'kumo-relational', workspace_client=_Workspace()
    )
    sdm_relational_engine.global_state.clear()
    assert not sdm_relational_engine.global_state.initialized
    assert sdm_relational_engine.global_state._client_factory is None


def test_each_thread_gets_its_own_client() -> None:
    """The cache is per thread, so an existing Kumo Relational keeps the client it was
    built with.
    """
    sdm_relational_engine.init_databricks_serving(
        'kumo-relational', workspace_client=_Workspace()
    )
    main = sdm_relational_engine.global_state.client
    seen: list[Any] = []

    thread = threading.Thread(
        target=lambda: seen.append(sdm_relational_engine.global_state.client)
    )
    thread.start()
    thread.join()

    assert isinstance(seen[0], DatabricksServingClient)
    assert seen[0] is not main
    assert sdm_relational_engine.global_state.client is main


def test_serving_reinit_replaces_cached_clients_in_other_threads() -> None:
    r"""A worker thread must not keep using a stale serving endpoint.

    ``GlobalState.clear()`` can only delete the current thread's cache. Other
    threads rely on the cache key changing when the process is reconfigured.
    """
    sdm_relational_engine.init_databricks_serving(
        'kumo-relational', workspace_client=_Workspace()
    )
    first_ready = threading.Event()
    reconfigured = threading.Event()
    seen: list[Any] = []
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            seen.append(sdm_relational_engine.global_state.client)
            first_ready.set()
            reconfigured.wait(timeout=5)
            seen.append(sdm_relational_engine.global_state.client)
        except BaseException as error:
            errors.append(error)
            first_ready.set()

    thread = threading.Thread(target=_worker)
    thread.start()
    assert first_ready.wait(timeout=5)
    assert not errors

    sdm_relational_engine.init_databricks_serving(
        'relational-endpoint-2', workspace_client=_Workspace()
    )
    reconfigured.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert not errors
    assert len(seen) == 2
    assert seen[0] is not seen[1]
    assert seen[1].endpoint == 'relational-endpoint-2'


# -- the raw-NIM path is unchanged ----------------------------------------


def test_url_mode_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """A URL still builds an authenticated NimClient, and is still required.

    ``authenticate()`` probes the NIM; serving mode has no equivalent, but the
    URL path must not have quietly lost it.
    """
    called: list[bool] = []
    monkeypatch.setattr(
        NimClient, 'authenticate', lambda self: called.append(True)
    )

    sdm_relational_engine.init(url='http://nim.test')

    assert isinstance(sdm_relational_engine.global_state.client, NimClient)
    assert sdm_relational_engine.global_state._url == 'http://nim.test'
    assert sdm_relational_engine.global_state._client_factory is None
    assert called == [True]

    sdm_relational_engine.global_state.clear()
    monkeypatch.delenv('KUMO_RELATIONAL_API_ENDPOINT', raising=False)
    with pytest.raises(ValueError, match='no endpoint'):
        sdm_relational_engine.init()


def test_switching_modes_replaces_the_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-initializing must not leave a stale client of the previous kind."""
    monkeypatch.setattr(NimClient, 'authenticate', lambda self: None)
    sdm_relational_engine.init(url='http://nim.test')
    assert isinstance(sdm_relational_engine.global_state.client, NimClient)

    sdm_relational_engine.init_databricks_serving(
        'kumo-relational', workspace_client=_Workspace()
    )
    assert isinstance(
        sdm_relational_engine.global_state.client, DatabricksServingClient
    )
    assert sdm_relational_engine.global_state._url is None


def test_uninitialized_state_still_raises() -> None:
    with pytest.raises(ValueError, match='Client creation'):
        _ = sdm_relational_engine.global_state.client
