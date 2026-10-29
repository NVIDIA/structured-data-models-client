# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neutral original native declarations preserve the actual dtype owners."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from test_t44_text_frames import REFERENCE, materialize, question, shop
from test_t44_text_frames import core as core

from sdm_client import StructuredDataError
from sdm_client._rust.frame_facts import (
    native_declaration,
    native_source_facts,
    observed_content,
)
from sdm_client._rust.frame_native import NativeColumn, NativeFrameColumns


@pytest.mark.parametrize(
    'physical',
    [
        pa.bool_(),
        pa.int8(),
        pa.int16(),
        pa.int32(),
        pa.int64(),
        pa.uint8(),
        pa.uint16(),
        pa.uint32(),
        pa.uint64(),
        pa.string(),
        pa.large_string(),
    ],
)
def test_fixed_native_declarations_describe_the_original_reconstruction_type(
    physical,
):
    dtype = pd.ArrowDtype(physical)
    declaration = native_declaration(dtype)
    if pa.types.is_large_string(physical):
        assert declaration == {'declaration': 'large_utf8'}
    else:
        tag = 'utf8' if pa.types.is_string(physical) else str(physical)
        assert declaration == {
            'declaration': 'physical',
            'physical': {'type': tag},
        }
    assert native_source_facts(dtype) == {
        'declared': declaration,
        'logical_text': None,
    }
    held = NativeFrameColumns({'rows': {'value': dtype}})
    assert (
        held.dtype({'source': {'source': ['rows']}, 'column': 'value'}) is dtype
    )


@pytest.mark.parametrize(
    'bits,precision',
    [
        (bits, precision)
        for bits, precision in [(32, 9), (64, 18), (128, 38), (256, 76)]
        if hasattr(pa, f'decimal{bits}')
    ],
)
@pytest.mark.parametrize('scale', [-129, 0, 129])
def test_decimal_declaration_does_not_use_widened_scale_zero_export(
    bits, precision, scale
):
    make = getattr(pa, f'decimal{bits}')
    dtype = pd.ArrowDtype(make(precision, scale))
    assert native_declaration(dtype) == {
        'declaration': 'decimal',
        'native': {
            'type': f'decimal{bits}',
            'precision': precision,
            'scale': scale,
        },
    }


def test_equal_exported_strings_keep_distinct_original_backing_and_declarations():
    logical = []
    for storage, backing in [
        ('python', 'element_values'),
        ('pyarrow', 'offset_utf8'),
    ]:
        dtype = pd.StringDtype(storage=storage)
        series = pd.Series(['a', None], dtype=dtype)
        rows, encoding = NativeColumn(series).arrow()
        assert encoding == 'utf8'
        assert rows.column(0).to_pylist() == ['a', None]
        fact = native_source_facts(dtype)
        assert fact == {
            'declared': None,
            'logical_text': {'backing': backing, 'missing': 'null_scalar'},
        }
        logical.append(fact)
    assert logical[0] != logical[1]
    fixed = native_source_facts(pd.ArrowDtype(pa.large_string()))
    assert fixed not in logical
    assert native_source_facts(np.dtype('O')) == {
        'declared': None,
        'logical_text': None,
    }
    assert native_source_facts(pd.CategoricalDtype(['a'])) == {
        'declared': None,
        'logical_text': None,
    }


def test_available_floating_nan_string_missing_fact_differs_from_null_scalar():
    if int(pd.__version__.split('.')[0]) < 3:
        return
    for storage in ('python', 'pyarrow'):
        masked = pd.StringDtype(storage=storage)
        floating = pd.StringDtype(storage=storage, na_value=np.nan)
        first = native_source_facts(masked)
        second = native_source_facts(floating)
        assert (
            first['logical_text']['backing']
            == second['logical_text']['backing']
        )
        assert first['logical_text']['missing'] == 'null_scalar'
        assert second['logical_text']['missing'] == 'ieee_nan'


@pytest.mark.parametrize(
    'native,expected',
    [
        ('string', 'text'),
        ('decimal', 'decimal'),
        ('date', 'temporal'),
        ('datetime', 'temporal'),
        ('datetime64', 'temporal'),
        ('timedelta', 'other'),
        ('empty', 'other'),
        ('mixed', 'other'),
        ('integer', 'other'),
        (None, None),
    ],
)
def test_observed_content_maps_only_the_original_native_labels(
    native, expected
):
    assert observed_content(native) == expected


def test_declaration_adapter_does_not_read_cells_or_convert_decimal_values():
    class Unobserved(Decimal):
        def __str__(self):
            raise AssertionError('native declaration read a value')

    assert native_declaration(np.dtype('O')) is None
    # The retained native owner remains independent of the source facts.
    value = Unobserved('1.2')
    series = pd.Series([value], dtype=object)
    assert native_source_facts(series.dtype) == {
        'declared': None,
        'logical_text': None,
    }
    assert series.iloc[0] is value


def test_fixed_large_utf8_native_message_and_cause_are_presented_at_table_admission(
    core, monkeypatch
):
    from sdm_client._rust import frame_materialize

    reached = []
    original = frame_materialize._convert

    def observe(*args, **kwargs):
        reached.append('export')
        return original(*args, **kwargs)

    monkeypatch.setattr(frame_materialize, '_convert', observe)
    asked = question(shop('largeutf8'))
    with pytest.raises(StructuredDataError) as raised:
        materialize(asked)
    expected = REFERENCE['records']['largeutf8-age-ordinary']['error']
    assert raised.value.code == expected['code']
    assert str(raised.value) == expected['message']
    assert type(raised.value.__cause__) is RuntimeError
    prefix = f'[{expected["code"]}] the relational engine failed while building this request: RuntimeError: '
    assert expected['message'].startswith(prefix)
    assert str(raised.value.__cause__) == expected['message'][len(prefix) :]
    assert reached == []
