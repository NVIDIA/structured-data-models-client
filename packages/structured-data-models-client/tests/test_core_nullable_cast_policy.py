# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sdm_client._core import Driver, give

CAST_EVENT = {
    'category': 'builtins.RuntimeWarning',
    'message': 'overflow encountered in cast',
    'masked_float32_cast': True,
}


@pytest.mark.parametrize('mode', ['warn', 'ignore', 'raise', 'warning-error'])
def test_public_masked_cast_matches_pandas(mode):
    expected_error = None
    with np.errstate(over='warn' if mode == 'warning-error' else mode):
        with warnings.catch_warnings(record=True) as baseline:
            warnings.simplefilter(
                'error' if mode == 'warning-error' else 'always', RuntimeWarning
            )
            try:
                pd.Series([np.finfo(np.float64).max], dtype='Float64').astype(
                    'float32'
                )
            except Exception as error:
                expected_error = (type(error), str(error))
        actual_error = None
        with warnings.catch_warnings(record=True) as actual:
            warnings.simplefilter(
                'error' if mode == 'warning-error' else 'always', RuntimeWarning
            )
            try:
                give([CAST_EVENT])
            except Exception as error:
                actual_error = (type(error), str(error))
    assert actual_error == expected_error
    assert [(item.category, str(item.message)) for item in actual] == [
        (item.category, str(item.message)) for item in baseline
    ]


def test_masked_raise_stops_before_later_host_io():
    class HostReached(RuntimeError):
        pass

    class Run:
        def next(self):
            return SimpleNamespace(
                warnings=[CAST_EVENT],
                kind='run_queries',
                queries=[SimpleNamespace(id='later')],
            )

        def submit(self, *_arguments):
            raise AssertionError('unexpected host result')

    calls = []
    driver = Driver()

    def host_io(_query):
        calls.append('host-io')
        raise HostReached('host reached')

    driver._ran = host_io
    expected_error = None
    with np.errstate(over='raise'):
        try:
            pd.Series([np.finfo(np.float64).max], dtype='Float64').astype(
                'float32'
            )
        except FloatingPointError as error:
            expected_error = str(error)
        with pytest.raises(
            FloatingPointError if expected_error else HostReached
        ) as actual:
            driver.drive(Run())
    assert str(actual.value) == (expected_error or 'host reached')
    assert calls == ([] if expected_error else ['host-io'])


def test_masked_cast_preserves_other_warning_order_and_location():
    with (
        np.errstate(over='warn'),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter('always')
        give(
            [
                {'category': 'builtins.UserWarning', 'message': 'before'},
                CAST_EVENT,
                {'category': 'builtins.UserWarning', 'message': 'after'},
            ]
        )
    assert [str(item.message) for item in caught] == ['before', 'after']
    assert all(
        item.category is UserWarning and item.filename == __file__
        for item in caught
    )


def test_masked_cast_uses_numpy_callback_policy():
    calls = []
    previous = np.geterrcall()
    try:
        np.seterrcall(lambda message, flag: calls.append((message, flag)))
        with np.errstate(over='call'):
            pd.Series([np.finfo(np.float64).max], dtype='Float64').astype(
                'float32'
            )
            expected = list(calls)
            calls.clear()
            give([CAST_EVENT])
    finally:
        np.seterrcall(previous)
    assert calls == expected
