# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless fixture scalars needed by the declared public dtype matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _fixture_ops import decode, encode  # noqa: E402


@pytest.mark.parametrize(
    'kind', ['longdouble', 'clongdouble', 'complex64', 'complex128']
)
@pytest.mark.parametrize(
    'value', ['precision', 'negative-zero', 'infinity', 'nan', 'negative-nan']
)
def test_numpy_numeric_roundtrip(kind, value):
    real = {
        'precision': np.nextafter(np.longdouble(1), np.longdouble(2)),
        'negative-zero': np.longdouble('-0'),
        'infinity': np.longdouble('inf'),
        'nan': np.longdouble('nan'),
        'negative-nan': -np.longdouble('nan'),
    }[value]
    original = np.empty((), dtype=kind)
    original.real = real
    if np.iscomplexobj(original):
        original.imag = -real
    original = original[()]
    actual = decode(json.loads(json.dumps(encode(original))))
    assert type(actual) is type(original)
    assert actual.dtype == original.dtype
    for component in ('real', 'imag'):
        wanted = getattr(original, component)
        observed = getattr(actual, component)
        assert np.isnan(observed) if np.isnan(wanted) else observed == wanted
        assert np.signbit(observed) == np.signbit(wanted)


@pytest.mark.parametrize(
    'value', [complex(1.25, -0.0), complex(float('nan'), float('inf'))]
)
def test_python_complex_roundtrip(value):
    actual = decode(json.loads(json.dumps(encode(value))))
    assert type(actual) is complex
    for component in ('real', 'imag'):
        observed = getattr(actual, component)
        wanted = getattr(value, component)
        assert np.isnan(observed) if np.isnan(wanted) else observed == wanted
        assert np.signbit(observed) == np.signbit(wanted)


@pytest.mark.parametrize('frequency', ['M', '2M', 'D', 'h'])
@pytest.mark.parametrize('ordinal', [-1, 0, 12345])
def test_period_roundtrip(frequency, ordinal):
    original = pd.Period(ordinal=ordinal, freq=frequency)
    actual = decode(json.loads(json.dumps(encode(original))))
    assert type(actual) is pd.Period
    assert actual.ordinal == original.ordinal
    assert actual.freqstr == original.freqstr
    assert actual == original


def test_existing_scalar_encodings_are_unchanged():
    assert encode(np.float64(1.5)) == {'$np': {'type': 'float64', 'value': 1.5}}
    assert encode(pd.Timedelta(1, 'ns')) == {'$timedelta_us': 0}
    assert encode(np.int64(2)) == {'$np': {'type': 'int64', 'value': 2}}
