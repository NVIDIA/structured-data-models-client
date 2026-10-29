# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neutral reached casing observations keep complete public Text behavior."""

from __future__ import annotations

import unicodedata
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
from test_t44_native_profile_observations import product_notices
from test_t44_native_text_profiles import (
    _observe_effective_target_profile_keeps_full_public_bodies,
    expected_record,
    reference,
)
from test_t44_text_frames import core as select_core

from sdm_client._rust.frame_native import NativeColumn, NativeFrameFailure


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize('storage', ['python', 'pyarrow'])
def test_observed_casing_facts_keep_the_frozen_public_text_journey(
    monkeypatch, core, storage
):
    case = f'python-target-{storage}'
    observations = []
    native = NativeColumn.text_facts

    def record(column, identity_input):
        fact = native(column, identity_input)
        dtype = column.values.dtype
        observations.append(
            {
                'fact': fact,
                'is_string': type(dtype) is pd.StringDtype,
                'storage': getattr(dtype, 'storage', None),
                'dtype': str(dtype),
                'values': [
                    None
                    if value is None or value is pd.NA
                    else list(map(ord, value))
                    for value in column.values
                ],
            }
        )
        return fact

    with monkeypatch.context() as route:
        select_core.__wrapped__(SimpleNamespace(param=False), route)
        original, _, _ = (
            _observe_effective_target_profile_keeps_full_public_bodies(
                False, case
            )
        )
    if core:
        with (
            monkeypatch.context() as route,
            patch.object(NativeColumn, 'text_facts', record),
        ):
            select_core.__wrapped__(SimpleNamespace(param=True), route)
            actual, _, _ = (
                _observe_effective_target_profile_keeps_full_public_bodies(
                    True, case
                )
            )
    else:
        actual = original
    assert actual == expected_record(original, core)
    expected = expected_record(reference('profile')[case], core)
    assert actual['requests'] == expected['requests']
    assert actual['sent'] == expected['sent']
    assert product_notices(actual) == expected['warnings']
    if not core:
        assert observations == []
        return
    assert observations
    supported = {'iσⰯᲉ꟎', 'iσⱟᲉ꟎', 'iσⱟᲊ꟎', 'iσⱟᲊ꟏'}
    for observed in observations:
        fact = observed['fact']
        if observed['is_string'] and observed['storage'] != 'python':
            assert set(fact) == {'mapping', 'identity'}
            assert fact['mapping'] == 'simple'
            assert fact['identity'] in supported
        else:
            assert fact == {
                'mapping': 'full_root',
                'unicode_version': unicodedata.unidata_version,
            }


def test_failure_capture_records_message_casing_without_displaying_native_error():
    class Undisplayed(ValueError):
        def __str__(self):
            raise AssertionError(
                'capturing failure facts displayed an exception'
            )

    error = Undisplayed()
    failure = NativeFrameFailure.capture(error)
    assert failure.error is error
    assert failure.facts.as_wire()['message_casing'] == {
        'mapping': 'full_root',
        'unicode_version': unicodedata.unidata_version,
    }
