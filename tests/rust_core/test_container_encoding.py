# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Container encoding preserves supported values and active ancestors."""

from __future__ import annotations

import subprocess
import sys
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from sdm_core._containers import NestingTooDeep
from sdm_core.answers import decode, encode
from sdm_core.describe import _encode


def test_arrays_retain_active_ancestors():
    array = np.empty(1, dtype=object)
    parent = [array]
    array[0] = parent
    assert encode(parent) == [
        {'$ndarray': {'dtype': 'object', 'values': [{'$refers_back': 'list'}]}}
    ]
    assert _encode(parent) == {'$container': 'list'}


@pytest.mark.parametrize('encoder', [encode, _encode])
def test_shared_arrays_and_nested_scalars_are_preserved(encoder):
    array = np.array([1, 2], dtype=np.int64)
    shared = {
        'array': array,
        'values': [None, pd.NA, pd.NaT, float('nan'), Decimal('1.25')],
        'tuple': (True, 'text'),
    }
    expected = {
        'array': {'$ndarray': {'dtype': 'int64', 'values': [1, 2]}},
        'values': [
            None,
            {'$na': None},
            {'$nat': None},
            {'$float': 'nan'},
            {'$decimal': '1.25'},
        ],
        'tuple': {'$tuple': [True, 'text']},
    }
    assert encoder([shared, shared]) == [expected, expected]


def test_deep_values_have_no_plain_form_but_keep_their_column_shape():
    value = 1
    for _ in range(1500):
        value = [value]
    with pytest.raises(NestingTooDeep, match='more than 100 levels'):
        encode(value, max_depth=100)
    assert _encode(value) == {'$container': 'list'}
    actual = encode(value)
    for _ in range(1500):
        assert isinstance(actual, list)
        assert len(actual) == 1
        actual = actual[0]
    assert actual == 1


@pytest.mark.parametrize('encoder', [encode, _encode])
def test_supported_nesting_is_not_truncated(encoder):
    value = 1
    for _ in range(100):
        value = [value]
    assert encoder(value) == value


@pytest.mark.parametrize('encoder', [encode, _encode])
def test_long_double_item_cannot_repeat_forever(encoder):
    value = np.longdouble('1.25')
    assert encoder(value) == 1.25


@pytest.mark.parametrize('encoder', ['answers', 'describe'])
def test_numpy_scalar_families_terminate_with_values_or_supported_fallback(
    encoder,
):
    script = """
import sys
import numpy as np
from sdm_core.answers import decode, encode
from sdm_core.describe import _encode
encoder = encode if sys.argv[1] == 'answers' else _encode
for value in (
    np.bool_(True), np.int8(-3), np.uint64(2**64 - 1),
    np.float16(1.25), np.float32(1.25), np.float64(1.25),
    np.longdouble(1.25), np.str_('scalar'),
):
    assert decode(encoder(value)) == value
for value in (np.bytes_(b'scalar'), np.void(b'scalar')):
    actual = encoder(value)
    assert actual['$np']['type'] == (value.dtype.str if sys.argv[1] == 'answers' else value.dtype.name)
    assert decode(actual['$np']['value']) == b'scalar'
    restored = decode(actual)
    assert restored.dtype == value.dtype
    assert restored.tobytes() == value.tobytes()
for scalar_type in (np.complex64, np.complex128, np.clongdouble):
    value = scalar_type(1 + 2j)
    try:
        actual = encoder(value)
    except TypeError as error:
        assert sys.argv[1] == 'answers'
        assert 'no plain form for' in str(error)
    else:
        assert sys.argv[1] == 'describe'
        assert '1' in str(actual) and '2j' in str(actual)
        assert '$object' in str(actual)
"""
    completed = subprocess.run(
        [sys.executable, '-c', script, encoder],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize('scalar_type', [np.bytes_, np.void])
@pytest.mark.parametrize('payload', [b'scalar', b'a\x00b', b'\x00a\x00'])
def test_flexible_scalars_roundtrip_current_and_legacy_descriptors(
    scalar_type, payload
):
    value = scalar_type(payload)
    plain = encode(value)
    for dtype in (value.dtype.str, value.dtype.name):
        restored = decode({'$np': {**plain['$np'], 'type': dtype}})
        assert restored.dtype == value.dtype
        assert restored.tobytes() == value.tobytes()
