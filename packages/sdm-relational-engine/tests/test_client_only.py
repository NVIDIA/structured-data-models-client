# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Kumo Relational engine may only be initialized by sdm_client.RelationalClient.

Direct `rfm.init(...)` + `KumoRelational(graph).predict(...)` against a NIM used to
bypass the supported client surface (issue #22). These tests lock in that the
engine now redirects any non-adapter caller to RelationalClient.
"""

from __future__ import annotations

import pytest

try:
    import sdm_relational_engine
    import sdm_relational_engine.rfm as rfm_engine
except (ImportError, RuntimeError) as error:
    rfm_engine = None
    _UNUSABLE = str(error)
else:
    _UNUSABLE = ''

requires_engine = pytest.mark.skipif(
    rfm_engine is None,
    reason=f'sdm_relational_engine.rfm is not usable in this environment: {_UNUSABLE}',
)


@requires_engine
def test_direct_init_is_blocked():
    with pytest.raises(RuntimeError, match='RelationalClient'):
        rfm_engine.init(url='http://nim.example.com:8000')


@requires_engine
def test_init_without_token_is_blocked_even_with_all_args():
    with pytest.raises(RuntimeError, match='RelationalClient'):
        rfm_engine.init(
            url='http://nim.example.com:8000',
            api_key='k',
            verify_ssl=False,
            log_level='INFO',
        )


@requires_engine
def test_authorized_init_passes_the_guard(monkeypatch):
    calls = {}
    monkeypatch.setattr(
        sdm_relational_engine, 'init', lambda **kwargs: calls.update(kwargs)
    )
    monkeypatch.setattr(
        type(sdm_relational_engine.global_state),
        '_url',
        'http://x',
        raising=False,
    )
    try:
        rfm_engine.init(url='http://x', _token=rfm_engine._CLIENT_TOKEN)
        assert rfm_engine.global_state._initialized is True
        assert calls['url'] == 'http://x'
    finally:
        rfm_engine.global_state.reset()


@requires_engine
def test_client_access_without_init_redirects_to_structured_client():
    rfm_engine.global_state.reset()
    with pytest.raises(RuntimeError, match='RelationalClient'):
        _ = rfm_engine.global_state.client


@requires_engine
def test_client_cache_follows_the_current_endpoint(monkeypatch):
    r"""Regression test for `quality-clients-share-process-global-engine-state`.

    The per-thread client cache is only evicted by `clear()`, which reaches
    just the thread that called it, and `init()` returns early when the
    settings it is given already match the globals. A thread that had resolved
    a client for one endpoint therefore kept using it once another thread had
    pointed the process at a second endpoint -- deterministically, with no
    race, which is exactly what a thread pool serving one client per tenant
    does.
    """
    import threading

    from sdm_relational_engine.client.client import NimClient

    monkeypatch.setattr(NimClient, 'authenticate', lambda self: None)

    a = ('https://tenant-a.example', 'key-A')
    b = ('https://tenant-b.example', 'key-B')

    def resolve(config):
        rfm_engine.init(
            url=config[0],
            api_key=config[1],
            _token=rfm_engine._CLIENT_TOKEN,
        )
        client = sdm_relational_engine.global_state.client
        return (client._url, client._api_key)

    try:
        assert resolve(a) == a

        worker: dict[str, tuple[str, str]] = {}
        thread = threading.Thread(
            target=lambda: worker.update(resolved=resolve(b))
        )
        thread.start()
        thread.join()
        assert worker['resolved'] == b

        assert resolve(b) == b
    finally:
        rfm_engine.global_state.reset()
        sdm_relational_engine.global_state.clear()
