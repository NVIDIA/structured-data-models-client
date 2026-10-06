# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Releasing the engine's connection pool.

``init_client`` configures the engine and caches a client per thread. Nothing
released that client until the process exited, so a caller that closed its
``NimClient`` left the engine's pool open. ``close_client`` is the release,
and these fix the two properties that make it safe to call while another
caller is still pointed at the same endpoint: the configuration survives, and
the next use rebuilds rather than fails.
"""

from __future__ import annotations

from typing import Any

import pytest
import sdm_relational_engine
import sdm_relational_engine.rfm as rfm_engine
from sdm_relational_engine.rfm import close_client, init_client


@pytest.fixture(autouse=True)
def _clean_state() -> Any:
    sdm_relational_engine.global_state.clear()
    rfm_engine.global_state.reset()
    yield
    sdm_relational_engine.global_state.clear()
    rfm_engine.global_state.reset()


@pytest.fixture
def token() -> object:
    return rfm_engine._CLIENT_TOKEN


@pytest.fixture
def nim() -> Any:
    r"""A NIM that answers the two probes ``authenticate`` makes."""
    import requests_mock

    with requests_mock.Mocker() as mock:
        mock.get('http://nim.test/v1/health/ready', json={'status': 'ready'})
        mock.get(
            'http://nim.test/v1/models',
            json={'data': [{'id': 'kumo-relational'}]},
        )
        yield mock


def _configured(token: object) -> Any:
    return init_client(url='http://nim.test', _token=token)


def test_close_client_is_gated_like_the_other_entry_points() -> None:
    r"""The engine is reachable only through ``NimClient``; a release that
    anyone could call would be a way around that boundary.
    """
    with pytest.raises(RuntimeError, match='not supported'):
        close_client()


def test_close_client_without_a_client_does_nothing(token: object) -> None:
    close_client(_token=token)


def test_close_client_closes_the_pool(token: object, nim: Any) -> None:
    client = _configured(token)
    closed: list[bool] = []
    original = client._session.close
    client._session.close = lambda: (closed.append(True), original())[1]

    close_client(_token=token)

    assert closed == [True]


def test_the_configuration_survives_the_release(
    token: object, nim: Any
) -> None:
    _configured(token)

    close_client(_token=token)

    assert sdm_relational_engine.global_state.initialized
    assert sdm_relational_engine.global_state._url == 'http://nim.test'


def test_the_next_use_rebuilds_rather_than_fails(
    token: object, nim: Any
) -> None:
    r"""What makes an early release safe: another client still pointed here
    reconnects instead of erroring.
    """
    first = _configured(token)
    close_client(_token=token)

    second = sdm_relational_engine.global_state.client

    assert second is not first
    assert second._session is not None


def test_a_release_is_idempotent(token: object, nim: Any) -> None:
    _configured(token)
    close_client(_token=token)
    close_client(_token=token)


def test_a_failing_close_still_evicts(
    token: object, nim: Any, monkeypatch
) -> None:
    r"""A client that cannot be closed must not stay cached: it would be
    handed to the next prediction after the caller was told it was gone.
    """
    client = _configured(token)
    monkeypatch.setattr(
        type(client),
        'close',
        lambda self: (_ for _ in ()).throw(RuntimeError('socket')),
    )

    with pytest.raises(RuntimeError, match='socket'):
        close_client(_token=token)

    assert sdm_relational_engine.global_state.client is not client
