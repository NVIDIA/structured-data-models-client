# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Carried row G6 (T2.2): the run loop names a provider's driver class
only on the error of the step whose fetch failed.

``_rust/run.py``'s ``_steps`` resets the class it keeps at every fetch
step: the core may carry on past one failed fetch (a whole-table read it
falls back from, T1.32e), and a later error is no other step's. Driven
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


def _fetch(step: int) -> bytes:
    plan = {
        'plan': 'sql',
        'statements': [{'dialect': 'sqlite', 'text': 'SELECT 1', 'params': []}],
        'expect': {'expect': 'discover'},
    }
    return json.dumps(
        {
            'message': 'step',
            'step': 'fetch',
            'requests': [{'id': step, 'request': {}, 'plan': plan}],
        }
    ).encode()


def _provider_error() -> bytes:
    error = {
        'kind': 'data_source',
        'code': 'PROVIDER_ERROR',
        'message': 'a provider failed',
    }
    return json.dumps({'message': 'error', 'error': error}).encode()


class _Handle:
    r"""A run that asks for one fetch per step, then ends with the core's
    provider error."""

    def __init__(self, fetches: int) -> None:
        self._fetches = fetches
        self._step = 0

    def next(self) -> tuple[bytes, dict[int, Any]]:
        self._step += 1
        if self._step > self._fetches:
            raise _CoreError(_provider_error())
        return _fetch(self._step), {}

    def take_warnings(self) -> None:
        return None

    def reply(self, reply: bytes, attached: dict[int, Any]) -> None:
        return None

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


def _raised(fetches: int) -> StructuredDataError:
    with pytest.raises(StructuredDataError) as raised:
        run_module._steps(
            _Core(),
            _Handle(fetches),
            _failing_first(),
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


def test_the_failing_step_s_error_names_its_driver() -> None:
    r"""Control: the core's error at the step whose fetch failed names the
    driver's class."""
    error = _raised(fetches=1)
    assert error.code == 'PROVIDER_ERROR'
    assert error.details['driver_class'] == (
        f'{OperationalError.__module__}.{OperationalError.__qualname__}'
    )
