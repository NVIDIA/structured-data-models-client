# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Actual native first-scalar reads after a core-selected time-facts plan."""

from __future__ import annotations

import traceback
import warnings
from decimal import InvalidOperation

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from test_t42_arrow_decimal_frames import _decimal_error_graph, raw_series
from test_t42_frames import PATHS, projection

from sdm_client import Schema

WIDTHS = [bits for bits in (32, 64, 128, 256) if hasattr(pa, f'decimal{bits}')]


def native_time_values(bits, scale, values):
    if all(value is not None for value in values):
        return raw_series(bits, 3, scale, values)
    make = getattr(pa, f'decimal{bits}')
    # Preserve the exact native validity rather than casting a missing cell
    # to a numeric value for the time read.
    dtype = make(3, scale)
    raw = b''.join(
        (value or 0).to_bytes(bits // 8, 'little', signed=True)
        for value in values
    )
    valid = bytearray((len(values) + 7) // 8)
    for at, value in enumerate(values):
        if value is not None:
            valid[at // 8] |= 1 << (at % 8)
    array = pa.Array.from_buffers(
        dtype, len(values), [pa.py_buffer(valid), pa.py_buffer(raw)]
    )
    return pd.Series(pd.arrays.ArrowExtensionArray(array))


def native_dropna_failure(bits, scale, *, missing):
    # Independent of Schema and either SDK outcome: the original time read
    # filters this exact input before checking lengths or boxing a scalar.
    values = native_time_values(
        bits, scale, [None] * 100 if missing else list(range(100))
    )
    try:
        values.dropna()
    except pa.ArrowNotImplementedError as error:
        return ('error', type(error), str(error)), _decimal_error_graph(error)
    return None


def assert_native_dropna_failure(observed, expected):
    outcome, graph = expected
    assert observed[0] == outcome
    assert observed[3]['error_graph'] == graph
    assert observed[3]['warning_categories'] == [
        'sdm_client.core.warnings.SDMWarning'
    ]
    assert observed[3]['events'] == [
        'warning',
        'mutation-complete',
        'constructor-error',
    ]
    assert len(observed[1]) == 1
    assert 'dropna' in observed[2]
    assert 'pyarrow.lib.ChunkedArray.filter' in observed[2]
    assert not any('Scalar.as_py' in name for name in observed[2])


def replaced(monkeypatch, bits, scale, variant, *, core):
    frame = pd.DataFrame(
        {
            'row_id': np.array([np.nan] + list(range(1, 100))),
            'observed_at': pd.date_range('2026-01-01', periods=100, freq='s'),
            'amount': np.arange(100, dtype='float64'),
        }
    )
    if variant in ('multiple', 'load', 'birth'):
        frame['other_at'] = pd.date_range('2025-01-01', periods=100, freq='s')
    name = (
        'loaded_at'
        if variant == 'load'
        else 'birth_date'
        if variant == 'birth'
        else 'observed_at'
    )
    if name != 'observed_at':
        frame.rename(columns={'observed_at': name}, inplace=True)
    values = (
        [None] * 100
        if variant in ('null', 'null_windowed')
        else [0] * 100
        if variant in ('constant', 'constant_windowed', 'multiple')
        else [0] + [None] * 99
        if variant == 'sparse'
        else list(range(100))
    )
    changed = False
    events = []
    monkeypatch.setenv('SDM_CORE_ROUTES', PATHS if core else '')
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        show = warnings.showwarning

        def mutate(message, *args, **kwargs):
            nonlocal changed
            events.append('warning')
            if not changed:
                assert 'have no value for it' in str(message)
                changed = True
                if variant == 'empty':
                    frame.drop(frame.index, inplace=True)
                    frame[name] = raw_series(bits, 3, scale, [])
                elif variant in ('null', 'null_windowed', 'sparse'):
                    frame[name] = native_time_values(bits, scale, values)
                else:
                    frame[name] = raw_series(bits, 3, scale, values)
                events.append('mutation-complete')
            show(message, *args, **kwargs)

        warnings.showwarning = mutate
        try:
            frames = {'rows': frame}
            if variant.endswith('_windowed'):
                frames['anchor'] = pd.DataFrame({'anchor_id': [1, 2]})
            schema = Schema.from_data(frames)
        except Exception as error:
            outcome = ('error', type(error), str(error))
            error_graph = _decimal_error_graph(error)
            events.append('constructor-error')
            native_trace = [
                item.name for item in traceback.extract_tb(error.__traceback__)
            ]
        else:
            outcome = ('ok', projection(schema))
            error_graph = None
            events.append('constructor-complete')
            native_trace = []
    assert changed
    return (
        outcome,
        [str(item.message) for item in notices],
        native_trace,
        {
            'error_graph': error_graph,
            'warning_categories': [
                f'{item.category.__module__}.{item.category.__qualname__}'
                for item in notices
            ],
            'events': events,
        },
    )


@pytest.mark.parametrize('bits', WIDTHS)
@pytest.mark.parametrize('scale', [0, 2, -129, 129])
def test_frozen_native_first_boxing_happens_after_missing_key_advice(
    monkeypatch, bits, scale
):
    old = replaced(monkeypatch, bits, scale, 'ordinary', core=False)
    new = replaced(monkeypatch, bits, scale, 'ordinary', core=True)
    assert new[:2] == old[:2]
    assert new[3] == old[3]
    native_filter = native_dropna_failure(bits, scale, missing=False)
    if native_filter is not None:
        assert_native_dropna_failure(old, native_filter)
        assert_native_dropna_failure(new, native_filter)
    elif abs(scale) == 129:
        assert new[0] == (
            'error',
            InvalidOperation,
            "[<class 'decimal.ConversionSyntax'>]",
        )
        assert len(new[1]) == 1
        assert any(f'Decimal{bits}Scalar.as_py' in name for name in new[2])
    else:
        assert new[0][0] == 'ok'


@pytest.mark.parametrize('bits', WIDTHS)
@pytest.mark.parametrize('scale', [2, 129])
@pytest.mark.parametrize(
    'variant',
    [
        'empty',
        'null',
        'constant',
        'sparse',
        'multiple',
        'load',
        'birth',
        'constant_windowed',
        'null_windowed',
    ],
)
def test_first_scalar_plan_retains_skips_counts_order_and_native_constant(
    monkeypatch, bits, scale, variant
):
    old = replaced(monkeypatch, bits, scale, variant, core=False)
    new = replaced(monkeypatch, bits, scale, variant, core=True)
    assert new[:2] == old[:2]
    assert new[3] == old[3]
    native_filter = (
        native_dropna_failure(bits, scale, missing=True)
        if variant in ('null', 'null_windowed')
        else None
    )
    if native_filter is not None:
        assert_native_dropna_failure(old, native_filter)
        assert_native_dropna_failure(new, native_filter)
    elif variant in ('empty', 'null', 'null_windowed', 'load', 'birth'):
        assert new[0][0] == 'ok'
