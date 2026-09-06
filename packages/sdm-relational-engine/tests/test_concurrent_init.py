# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Pointing the process at a deployment, from more than one thread at once."""

import threading
import time
from typing import Any

import pytest
import sdm_relational_engine as kre
from sdm_relational_engine import _configure_lock, global_state

DEPLOYMENTS = (('https://a.internal', 'key-a'), ('https://b.internal', 'key-b'))


@pytest.fixture(autouse=True)
def _clean() -> Any:
    global_state.clear()
    yield
    global_state.clear()


def _slow_client(authentications: list[int]) -> type:
    class SlowClient:
        def __init__(self, url: str, api_key: str, **kwargs: Any) -> None:
            self._url = url
            self._api_key = api_key

        def authenticate(self) -> None:
            authentications.append(1)
            time.sleep(0.05)

    return SlowClient


def test_two_deployments_configured_at_once_are_never_mixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""One deployment's URL held with another's key sends that key to the wrong place.

    Sampled throughout rather than only at the end: publishing the fields one at
    a time leaves a window in which the pair is a mix of two deployments, and a
    request made in that window is the one that leaks.
    """
    monkeypatch.setattr(kre, 'NimClient', _slow_client([]))
    mixed: list[tuple[str, str]] = []
    samples = 0
    done = threading.Event()

    def watch() -> None:
        nonlocal samples
        while not done.is_set():
            url, key = global_state._url, global_state._api_key
            if url and key:
                samples += 1
                if url[len('https://')] != key[-1]:
                    mixed.append((url, key))

    observer = threading.Thread(target=watch)
    observer.start()
    threads = [
        threading.Thread(target=kre.init, kwargs={'url': url, 'api_key': key})
        for url, key in DEPLOYMENTS
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    done.set()
    observer.join(5)

    assert mixed == []
    # Without this the test passes when the observer never ran during a write,
    # which is a pass that proves nothing.
    assert samples > 0, 'the observer never sampled a configured deployment'
    assert global_state._url is not None


def test_an_identical_deployment_is_authenticated_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Re-initializing with identical settings is documented as a no-op."""
    authentications: list[int] = []
    monkeypatch.setattr(kre, 'NimClient', _slow_client(authentications))

    threads = [
        threading.Thread(
            target=kre.init,
            kwargs={'url': 'https://a.internal', 'api_key': 'key-a'},
        )
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert authentications == [1]


def test_a_thread_already_configuring_may_configure_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""The client property calls init(), and may be reached while one is running."""
    monkeypatch.setattr(kre, 'NimClient', _slow_client([]))

    with _configure_lock:
        kre.init(url='https://a.internal', api_key='key-a')

    assert global_state._url == 'https://a.internal'


def test_a_reader_never_sees_half_of_each_deployment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Serializing the writers narrows the window; it does not close it.

    A reader that took url, then api_key, could still have a reconfiguration
    land between the two. Reading the whole deployment once is what closes it,
    so this samples what the client would actually be built from.
    """
    monkeypatch.setattr(kre, 'NimClient', _slow_client([]))
    mixed: list[Any] = []
    samples = 0
    done = threading.Event()

    def watch() -> None:
        nonlocal samples
        while not done.is_set():
            deployment = global_state._deployment
            if deployment.url and deployment.api_key:
                samples += 1
                if deployment.url[len('https://')] != deployment.api_key[-1]:
                    mixed.append(deployment)

    observer = threading.Thread(target=watch)
    observer.start()
    for _ in range(20):
        for url, key in DEPLOYMENTS:
            kre.init(url=url, api_key=key)
    done.set()
    observer.join(5)

    assert mixed == []
    assert samples > 0, 'the observer never sampled a configured deployment'


def test_a_configuration_is_replaced_whole() -> None:
    r"""One assignment, so there is no moment at which it is half applied."""
    global_state._publish(url='https://a.internal', api_key='key-a')
    before = global_state._deployment

    global_state._publish(url='https://b.internal', api_key='key-b')

    assert before.url == 'https://a.internal'
    assert before.api_key == 'key-a'
    assert global_state._deployment is not before


def test_the_key_is_kept_out_of_what_gets_printed() -> None:
    r"""global_state is exported, and a repr reaches tracebacks and crash reports."""
    global_state._publish(url='https://a.internal', api_key='hunter2')

    assert 'hunter2' not in repr(global_state)
    assert 'hunter2' not in repr(global_state._deployment)
    assert global_state._api_key == 'hunter2'


def test_two_fields_changed_at_once_do_not_lose_each_other(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Publishing reads before it writes, so unlocked it drops one of the two.

    The interleaving is forced rather than raced: replacing a dataclass is fast
    enough that two threads hammering it will usually miss each other, and a
    test that only usually fails is one that will pass over a real regression.
    """
    import sdm_relational_engine as engine

    global_state._publish(url='https://a.internal', api_key='key-a')
    real_replace = engine.replace
    reading = threading.Event()
    may_write = threading.Event()

    def slow_replace(deployment: Any, **changes: Any) -> Any:
        if 'timeout' in changes:
            reading.set()
            may_write.wait(5)
        return real_replace(deployment, **changes)

    monkeypatch.setattr(engine, 'replace', slow_replace)

    first = threading.Thread(
        target=global_state._publish, kwargs={'timeout': 30.0}
    )
    first.start()
    reading.wait(5)

    second = threading.Thread(
        target=global_state._publish, kwargs={'max_retries': 7}
    )
    second.start()
    second.join(0.3)
    may_write.set()
    first.join(5)
    second.join(5)

    assert global_state._timeout == 30.0
    assert global_state._max_retries == 7


def test_every_setting_is_still_writable() -> None:
    r"""These were plain dataclass fields, and ``global_state`` is exported.

    Turning them into read-only views would break any caller that assigns one,
    which is a breaking change to a public object rather than the internal
    refactor this is meant to be.
    """
    settings: list[tuple[str, Any]] = [
        ('_url', 'https://a.internal'),
        ('_api_key', 'key-a'),
        ('_verify_ssl', False),
        ('_timeout', 30.0),
        ('_max_retries', 5),
        ('_client_factory', None),
        ('_serving_kind', 'databricks'),
        ('_serving_endpoint', 'an-endpoint'),
        ('_serving_platform_client', None),
        ('_serving_overrides', {'a': 1}),
    ]

    for name, value in settings:
        setattr(global_state, name, value)
        assert getattr(global_state, name) == value, name


def test_assigning_one_setting_leaves_the_others_alone() -> None:
    r"""Each setter replaces the whole value, so it must carry the rest forward."""
    global_state._publish(
        url='https://a.internal', api_key='key-a', max_retries=4
    )

    global_state._timeout = 30.0

    assert global_state._url == 'https://a.internal'
    assert global_state._api_key == 'key-a'
    assert global_state._max_retries == 4
    assert global_state._timeout == 30.0
