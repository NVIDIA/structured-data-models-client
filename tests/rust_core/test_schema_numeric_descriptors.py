# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from sdm_core import describe


@pytest.mark.parametrize('values', [[0, 1, 3], [0, np.nan, 3], []])
def test_longdouble_descriptor_preserves_numeric_values(values):
    series = pd.Series(np.asarray(values, dtype=np.longdouble))
    array, tag = describe._column(series)
    assert array.type == pa.float64()
    assert array.to_pylist() == [
        None if np.isnan(value) else float(value) for value in values
    ]
    assert 'cells' not in tag


@pytest.mark.parametrize(
    ('dtype', 'values', 'expected'),
    [
        ('int64', [0, 1, 2**62], pa.int64()),
        ('uint64', [0, 1, 2**64 - 1], pa.uint64()),
        ('float32', [0, np.nan, 3], pa.float32()),
        ('float64', [0, np.nan, 3], pa.float64()),
        ('int64', [], pa.int64()),
    ],
)
def test_sparse_descriptor_preserves_numeric_width_and_nulls(
    dtype, values, expected
):
    series = pd.Series(pd.arrays.SparseArray(values, dtype=dtype))
    array, tag = describe._column(series)
    assert array.type == expected
    assert array.to_pylist() == [
        None if pd.isna(value) else value for value in values
    ]
    assert tag['dtype'] == str(series.dtype)
    assert 'cells' not in tag


@pytest.mark.parametrize('sparse', [False, True])
def test_numeric_descriptor_keeps_inferred_relationship(sparse):
    from sdm_client import Schema

    series = (
        pd.Series(pd.arrays.SparseArray([0, 1, 2], dtype='int64'))
        if sparse
        else pd.Series(np.asarray([0, 1, 2], dtype=np.longdouble))
    )
    schema = Schema.from_data(
        {
            'parent': pd.DataFrame({'parent_id': [0, 1, 2]}),
            'child': pd.DataFrame({'child_id': [0, 1, 2], 'parent_id': series}),
        },
        primary_keys={'parent': 'parent_id', 'child': 'child_id'},
    )
    column = schema['child']['parent_id']
    assert column.dtype == ('int64' if sparse else 'float64')
    assert column.stype == 'ID'
    assert len(schema.relationships) == 1


def test_fixed_binary_refuses_raw_bytes_before_categorical_inference():
    from sdm_client import Schema
    from sdm_client.errors import StructuredDataError

    series = pd.Series(
        pd.arrays.ArrowExtensionArray(
            pa.array([b'one', None], type=pa.binary(3))
        )
    )
    with pytest.raises(StructuredDataError) as caught:
        Schema.from_data({'items': pd.DataFrame({'value': series})})
    assert caught.value.code == 'UNSUPPORTED_REQUEST'
    assert caught.value.message == (
        'items.value holds raw bytes (bytes), and the contract has no cell '
        'that carries them. It is already one value per row, so what it '
        'needs is a reading rather than a flattening: decode the column '
        'to text, or leave it out of the schema'
    )
    assert str(caught.value) == f'[UNSUPPORTED_REQUEST] {caught.value.message}'


@pytest.mark.parametrize('name', ['bool8', 'json', 'uuid'])
def test_arrow_extension_unique_refuses_during_schema_inference(name):
    from sdm_client import Schema

    kind = {'bool8': pa.bool8, 'json': pa.json_, 'uuid': pa.uuid}[name]()
    storage = {
        'bool8': pa.array([1, None, 0], type=pa.int8()),
        'json': pa.array(['{"value":1}', None, '{"value":3}']),
        'uuid': pa.array([b'0' * 16, None, b'1' * 16], type=pa.binary(16)),
    }[name]
    series = pd.Series(
        pd.arrays.ArrowExtensionArray(
            pa.ExtensionArray.from_storage(kind, storage)
        )
    )
    expected_class, expected_message = {
        '2': (NotImplementedError, str(kind)),
        '3': (
            pa.ArrowNotImplementedError,
            f"Function 'unique' has no kernel matching input types ({kind})",
        ),
    }[pd.__version__.split('.')[0]]
    with pytest.raises(expected_class) as caught:
        Schema.from_data({'items': pd.DataFrame({'value': series})})
    assert type(caught.value) is expected_class
    assert str(caught.value) == expected_message


@pytest.mark.parametrize(('width', 'precision'), [(32, 8), (64, 16)])
def test_decimal_unique_refusal_happens_during_schema_key_inference(
    width, precision
):
    from decimal import Decimal

    from sdm_client import Schema

    kind = getattr(pa, f'decimal{width}')(precision, 2)
    series = pd.Series(
        pd.arrays.ArrowExtensionArray(pa.array([Decimal('1.00')], type=kind))
    )
    with pytest.raises(pa.ArrowNotImplementedError) as caught:
        Schema.from_data({'items': pd.DataFrame({'item_id': series})})
    assert str(caught.value) == (
        f"Function 'unique' has no kernel matching input types ({kind})"
    )
    schema = Schema.from_data({'items': pd.DataFrame({'value': series})})
    assert schema['items']['value'].dtype == 'float64'
    empty = Schema.from_data(
        {'items': pd.DataFrame({'item_id': series.iloc[:0]})}
    )
    assert empty['items'].primary_key is None
