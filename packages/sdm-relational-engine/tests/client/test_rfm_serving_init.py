# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``sdm_relational_engine.rfm.init_databricks_serving``, without the native extension.

``sdm_relational_engine/rfm/backend/local/__init__.py`` does a bare ``import
sdm_relational_engine.relationallib`` at module scope and turns any failure into a RuntimeError, so
``import sdm_relational_engine.rfm`` is impossible wherever the compiled extension is absent.

That guard exists for the *local* sampler. The Databricks Model Serving path
never touches it, inference happens on the endpoint. Satisfying the guard with
a stub is therefore not papering over a real dependency; it demonstrates that
the dependency is not real for this path, which is the argument for making the
import lazy.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


class _Endpoints:
    @staticmethod
    def query(**kwargs: Any) -> Any:
        return {'predictions': [{'response_json': '{}'}]}


class _Workspace:
    serving_endpoints = _Endpoints()


def _init(
    rfm_engine: Any, endpoint: str = 'kumo-relational', **kwargs: Any
) -> None:
    """Initialize the way ``NimClient`` does, token included.

    Every test below stands in for that caller; a bare call is refused, which
    ``test_a_direct_call_is_refused`` covers.
    """
    rfm_engine.init_databricks_serving(
        endpoint,
        workspace_client=_Workspace(),
        _token=rfm_engine._CLIENT_TOKEN,
        **kwargs,
    )


class _RecordingModule(types.ModuleType):
    """A stub that records *every* attribute access, not just missing ones.

    A plain module with ``__getattr__`` records nothing once real attributes
    exist, because PEP 562 only invokes it on lookup failure. Overriding
    ``__getattribute__`` is what actually observes use.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        object.__setattr__(self, '_accessed', [])

    def __getattribute__(self, name: str) -> Any:
        if not name.startswith('__'):
            object.__getattribute__(self, '_accessed').append(name)
        return object.__getattribute__(self, name)


@pytest.fixture()
def stubbed_relationallib(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Only meaningful when the real extension is absent."""
    if 'sdm_relational_engine.relationallib' in sys.modules:
        pytest.skip(
            'the real relationallib is present; stubbing it would mutate the real '
            'module and the recording would not apply'
        )
    stub = _RecordingModule('sdm_relational_engine.relationallib')
    monkeypatch.setitem(
        sys.modules, 'sdm_relational_engine.relationallib', stub
    )
    return stub


@pytest.fixture()
def rfm_engine(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import sdm_relational_engine.rfm with the local-sampler guard satisfied."""
    if 'sdm_relational_engine.relationallib' not in sys.modules:
        monkeypatch.setitem(
            sys.modules,
            'sdm_relational_engine.relationallib',
            types.ModuleType('sdm_relational_engine.relationallib'),
        )
    try:
        import sdm_relational_engine.rfm as engine
    except RuntimeError as error:  # pragma: no cover - environment dependent
        pytest.skip(
            f'sdm_relational_engine.rfm unusable even with the guard stubbed: {error}'
        )
    yield engine
    engine.global_state.reset()
    import sdm_relational_engine

    sdm_relational_engine.global_state.clear()


def test_initializes_against_a_named_endpoint(rfm_engine: Any) -> None:
    _init(rfm_engine)
    assert rfm_engine.global_state._initialized


def test_records_the_endpoint_rather_than_a_stale_url(rfm_engine: Any) -> None:
    """There is no URL in this mode. Leaving the previous deployment's URL in
    place would make the state read as if a NIM were still configured.
    """
    _init(rfm_engine)
    assert rfm_engine.global_state._url == 'databricks-serving:kumo-relational'


def test_the_client_is_the_serving_transport(rfm_engine: Any) -> None:
    from sdm_relational_engine.client.databricks_serving import (
        DatabricksServingClient,
    )

    _init(rfm_engine)
    client = rfm_engine.global_state.client
    assert isinstance(client, DatabricksServingClient)


def test_a_direct_call_is_refused(rfm_engine: Any) -> None:
    """The serving entry point is gated like ``init``.

    Both unlock the engine, so leaving this one open would make it the way
    around the boundary the other enforces.
    """
    with pytest.raises(RuntimeError):
        rfm_engine.init_databricks_serving(
            'kumo-relational', workspace_client=_Workspace()
        )
    assert not rfm_engine.global_state._initialized


def test_a_url_shaped_endpoint_is_rejected(rfm_engine: Any) -> None:
    with pytest.raises(ValueError):
        _init(
            rfm_engine,
            'https://workspace/serving-endpoints/kumo-relational',
        )
    assert not rfm_engine.global_state._initialized


def test_the_serving_path_never_needs_the_native_sampler(
    stubbed_relationallib: Any, rfm_engine: Any
) -> None:
    """Nothing on the serving path calls into relationallib.

    Scope, stated plainly: this observes access *after*
    ``sdm_relational_engine.rfm`` is already imported. Import-time use (the bare
    ``import sdm_relational_engine.relationallib`` in
    ``backend/local/__init__.py``) happens before the recorder can see it and
    is not covered. What is covered is that initializing against a serving
    endpoint and obtaining a client touch no symbol on the extension, which is
    the claim that matters: the one real use site is
    ``relationallib.NeighborSampler`` in ``backend/local/sampler.py``, and the
    serving path does not sample locally.
    """
    before = list(stubbed_relationallib._accessed)

    _init(rfm_engine)
    _ = rfm_engine.global_state.client

    during = [
        n
        for n in stubbed_relationallib._accessed[len(before) :]
        if n != '_accessed'
    ]
    assert during == [], (
        f'the serving path reached into relationallib: {during}'
    )
