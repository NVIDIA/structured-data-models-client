# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Completed native groups keep their actual empty and lossless scalar data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from sdm_client._rust.native_target import export_groups


def _target(scalar, values):
    return pa.array(
        values,
        type=pa.list_(
            pa.field('actual_item', scalar, nullable=False, metadata={b'k': b'v'})
        ),
    )


@pytest.mark.parametrize('dtype', ['Int64', 'string', 'string[pyarrow]'])
@pytest.mark.parametrize('scalar', [pa.int64(), pa.string(), pa.binary()])
def test_empty_native_extension_result_keeps_the_completed_group_schema(
    dtype, scalar
):
    # A native empty extension array is a real result, not a scalar array
    # which Arrow can be asked to cast into a list after the fact.
    actual = pd.array([], dtype=dtype)
    expected = _target(scalar, [])
    encoded, encoding = export_groups(
        actual,
        expected,
        text=pa.types.is_string(scalar) or pa.types.is_binary(scalar),
    )
    assert encoded.to_pylist() == []
    assert encoded.type.value_field.name == 'actual_item'
    assert encoded.type.value_field.nullable is False
    assert encoded.type.value_field.metadata == {b'k': b'v'}
    if pa.types.is_integer(scalar):
        assert encoded.type == expected.type
        assert encoding is None
    else:
        assert encoded.type.value_type == pa.string()
        assert encoding == 'utf8'


@pytest.mark.parametrize('before', [pa.string(), pa.binary()])
@pytest.mark.parametrize(
    ('actual', 'scalar', 'encoding'),
    [
        ([['changed', '\U0001f600'], []], pa.string(), 'utf8'),
        ([['changed\ud800', '\udfff'], []], pa.binary(), 'utf8_surrogatepass'),
    ],
)
def test_actual_changed_native_text_is_authoritative_across_physical_encodings(
    before, actual, scalar, encoding
):
    boxed = np.empty(len(actual), dtype=object)
    for at, group in enumerate(actual):
        boxed[at] = group
    previous = _target(before, [])
    encoded, observed = export_groups(boxed, previous, text=True)
    assert observed == encoding
    assert encoded.type.value_type == scalar
    assert encoded.type.value_field.name == previous.type.value_field.name
    assert (
        encoded.type.value_field.metadata
        == previous.type.value_field.metadata
    )
    expected = (
        actual
        if encoding == 'utf8'
        else [
            [str.encode(value, 'utf-8', 'surrogatepass') for value in group]
            for group in actual
        ]
    )
    assert encoded.to_pylist() == expected


def test_lossless_export_uses_builtin_string_encoding_without_scalar_override():
    class Observed(str):
        def encode(self, *args, **kwargs):
            pytest.fail('the fixed codec must not invoke a scalar override')

    boxed = np.empty(1, dtype=object)
    boxed[0] = [Observed('a\ud800')]
    actual, encoding = export_groups(
        boxed, _target(pa.string(), [['before']]), text=True
    )
    assert encoding == 'utf8_surrogatepass'
    assert actual.to_pylist() == [[b'a\xed\xa0\x80']]


@pytest.mark.parametrize(
    'scalar', [pa.int64(), pa.uint64(), pa.float32(), pa.bool_()]
)
def test_ordinary_completed_groups_preserve_the_actual_fixed_scalar_width(
    scalar,
):
    values = [[True, False]] if pa.types.is_boolean(scalar) else [[0, 1]]
    boxed = np.empty(1, dtype=object)
    boxed[0] = values[0]
    previous = _target(scalar, values)
    actual, encoding = export_groups(boxed, previous, text=False)
    assert actual.equals(previous)
    assert encoding is None
