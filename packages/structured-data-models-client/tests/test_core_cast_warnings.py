# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from sdm_client._core import Driver, give

CAST_WARNING = {
    'category': 'builtins.RuntimeWarning',
    'message': 'overflow encountered in cast',
}


def host_cast(mode, warning_error=False):
    with np.errstate(over=mode), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('error' if warning_error else 'always')
        try:
            np.array([np.finfo(np.float64).max]).astype('float32')
        except (FloatingPointError, RuntimeWarning) as error:
            return type(error), []
    return None, [(item.category, str(item.message)) for item in caught]


@pytest.mark.parametrize('mode', ['raise', 'warning-error'])
def test_cast_policy_refuses_before_later_host_io(mode):
    calls = []
    policy = 'raise' if mode == 'raise' else 'warn'
    expected, _ = host_cast(policy, warning_error=True)

    class Run:
        submitted = False

        def next(self):
            if self.submitted:
                return SimpleNamespace(warnings=[], kind='done', value='done')
            return SimpleNamespace(
                warnings=[CAST_WARNING],
                kind='run_queries',
                queries=[SimpleNamespace(id='later')],
            )

        def submit(self, *_arguments):
            self.submitted = True

    driver = Driver()
    driver._ran = lambda _query: calls.append('host-io')
    with np.errstate(over=policy):
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            if expected is None:
                assert driver.drive(Run()) == 'done'
            else:
                with pytest.raises(
                    expected, match='^overflow encountered in cast$'
                ):
                    driver.drive(Run())
    assert calls == ([] if expected is not None else ['host-io'])


def test_public_cast_warning_preserves_order_and_caller_location():
    error, expected = host_cast('warn')
    assert error is None
    with (
        np.errstate(over='warn'),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter('always')
        give(
            [
                {'category': 'builtins.UserWarning', 'message': 'before'},
                CAST_WARNING,
                {'category': 'builtins.UserWarning', 'message': 'after'},
            ]
        )
    assert [(item.category, str(item.message)) for item in caught] == [
        (UserWarning, 'before'),
        *expected,
        (UserWarning, 'after'),
    ]
    assert all(item.filename == __file__ for item in caught)


def test_public_cast_warning_respects_numpy_ignore():
    with (
        np.errstate(over='ignore'),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter('always')
        give([CAST_WARNING])
    assert caught == []


def test_public_cast_warning_uses_numpy_callback_policy():
    expected = []
    calls = []
    previous = np.geterrcall()
    try:
        np.seterrcall(lambda message, flag: expected.append((message, flag)))
        with np.errstate(over='call'):
            np.array([np.finfo(np.float64).max]).astype('float32')
        np.seterrcall(lambda message, flag: calls.append((message, flag)))
        with np.errstate(over='call'):
            give([CAST_WARNING])
    finally:
        np.seterrcall(previous)
    assert calls == expected
