# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Carried row G6: the run loop names a provider's driver class, and
carries the driver's exception, only on the error of the step whose fetch
failed.

``_rust/run.py``'s ``_steps`` resets the class it keeps at every fetch
step: the core may carry on past one failed fetch (a whole-table read it
falls back from), and a later error is no other step's. Driven
here by a scripted core, step by step, with no native module: a failure
the core carries past, then a step answered, then the core's error; and,
as the control, the core's error at the failing step itself.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from sdm_client._rust import run as run_module
from sdm_client.errors import StructuredDataError


class _CoreError(Exception):
    def __init__(self, control: bytes) -> None:
        self.control = control
        self.attachments: dict[int, Any] = {}


class _Core:
    CoreError = _CoreError


def _fetch(step: int, requests: int = 1) -> bytes:
    plan = {
        'plan': 'sql',
        'statements': [{'dialect': 'sqlite', 'text': 'SELECT 1', 'params': []}],
        'expect': {'expect': 'discover'},
    }
    return json.dumps(
        {
            'message': 'step',
            'step': 'fetch',
            'requests': [
                {'id': 10 * step + n, 'request': {}, 'plan': plan}
                for n in range(requests)
            ],
        }
    ).encode()


def _provider_error() -> bytes:
    error = {
        'kind': 'data_source',
        'code': 'PROVIDER_ERROR',
        'sentence': 'provider_failed',
        'message': 'a provider failed',
    }
    return json.dumps({'message': 'error', 'error': error}).encode()


def _unknown_reply() -> bytes:
    error = {
        'kind': 'protocol',
        'code': 'UNKNOWN_REPLY',
        'sentence': 'unknown_reply',
        'message': 'a reply the run does not await',
    }
    return json.dumps({'message': 'error', 'error': error}).encode()


class _Handle:
    r"""A run that asks for one fetch step per step, of ``requests``
    requests, then ends with the core's provider error; ``refusing``, the
    error it refuses the first reply with instead."""

    def __init__(
        self, fetches: int, requests: int = 1, refusing: bytes | None = None
    ) -> None:
        self._fetches = fetches
        self._requests = requests
        self._refusing = refusing
        self._step = 0

    def next(self) -> tuple[bytes, dict[int, Any]]:
        self._step += 1
        if self._step > self._fetches:
            raise _CoreError(_provider_error())
        return _fetch(self._step, self._requests), {}

    def take_warnings(self) -> None:
        return None

    def reply(self, reply: bytes, attached: dict[int, Any]) -> None:
        if self._refusing is not None:
            raise _CoreError(self._refusing)

    def cancel(self) -> None:
        return None


class OperationalError(Exception):
    r"""A driver's failure."""


def _failing_first() -> Any:
    calls = {'n': 0}

    def provider(step: dict[str, Any], keys: dict[int, Any]) -> Any:
        calls['n'] += 1
        if calls['n'] == 1:
            raise OperationalError('the source refused the read')
        return object()

    return provider


def _failing_each() -> Any:
    calls = {'n': 0}

    def provider(step: dict[str, Any], keys: dict[int, Any]) -> Any:
        calls['n'] += 1
        raise OperationalError(f'read {calls["n"]} failed')

    return provider


def _raised(
    fetches: int, handle: _Handle | None = None, provider: Any = None
) -> StructuredDataError:
    with pytest.raises(StructuredDataError) as raised:
        run_module._steps(
            _Core(),
            handle or _Handle(fetches),
            provider or _failing_first(),
            None,
            frozenset(),
            None,
        )
    return raised.value


def test_a_later_step_s_error_names_no_earlier_driver() -> None:
    r"""The fetch of step 1 fails, the core carries on, step 2's fetch is
    answered, and the core's error after it names no driver class."""
    error = _raised(fetches=2)
    assert error.code == 'PROVIDER_ERROR'
    assert 'driver_class' not in error.details
    assert error.source_error is None


def test_the_failing_step_s_error_names_its_driver() -> None:
    r"""Control: the core's error at the step whose fetch failed names the
    driver's class."""
    error = _raised(fetches=1)
    assert error.code == 'PROVIDER_ERROR'
    assert error.details['driver_class'] == (
        f'{OperationalError.__module__}.{OperationalError.__qualname__}'
    )
    assert isinstance(error.source_error, OperationalError)
    assert str(error.source_error) == 'the source refused the read'


def test_a_step_s_first_failure_is_the_one_carried() -> None:
    r"""Both requests of the failing step fail: the error carries the first
    request's failure, as the Java host does."""
    error = _raised(
        fetches=1, handle=_Handle(1, requests=2), provider=_failing_each()
    )
    assert str(error.source_error) == 'read 1 failed'


def test_a_refused_reply_that_is_not_a_provider_error_carries_nothing() -> None:
    r"""The core refuses the failed fetch's reply with its own protocol
    error: no driver class is named and no source error carried."""
    error = _raised(fetches=1, handle=_Handle(1, refusing=_unknown_reply()))
    assert error.code == 'UNKNOWN_REPLY'
    assert 'driver_class' not in error.details
    assert error.source_error is None
