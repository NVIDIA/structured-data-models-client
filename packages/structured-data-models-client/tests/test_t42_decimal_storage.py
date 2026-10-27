# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The typed native Decimal export retains buffers without scalar conversion."""

from __future__ import annotations

import gc

import pandas as pd
import pyarrow as pa
import pytest

from sdm_client._rust.frame_decimals import native_decimal_column
from sdm_client import StructuredDataError


def arrays(bits, precision, scale):
    make = pa.decimal128 if bits == 128 else pa.decimal256
    dtype = make(precision, scale)
    raw = b''.join(
        value.to_bytes(bits // 8, byteorder='little', signed=True)
        for value in (0, 1, -1, 2)
    )
    base = pa.Array.from_buffers(
        dtype, 4, [pa.py_buffer(b'\x0b'), pa.py_buffer(raw)]
    )
    return dtype, [base.slice(1, 3), base.slice(0, 1), base.slice(0, 0)]


@pytest.mark.parametrize('bits', [128, 256])
@pytest.mark.parametrize('scale', [-129, -128, -1, 0, 1, 2, 129])
def test_lossless_native_decimal_mantissa_view(bits, scale):
    dtype, chunks = arrays(bits, 1, scale)
    original = pa.chunked_array(chunks, type=dtype)
    series = pd.Series(pd.arrays.ArrowExtensionArray(original))
    storage, descriptor = native_decimal_column(series)
    assert descriptor == {
        'type': f'decimal{bits}',
        'precision': 1,
        'scale': scale,
    }
    make = pa.decimal128 if bits == 128 else pa.decimal256
    assert storage.type == make(1, 0)
    assert storage.num_chunks == len(chunks)
    for source, view in zip(chunks, storage.chunks, strict=True):
        assert len(view) == len(source)
        assert view.offset == source.offset
        assert view.null_count == source.null_count
        addresses = [
            buffer.address if buffer else None for buffer in view.buffers()
        ]
        assert addresses == [
            buffer.address if buffer else None for buffer in source.buffers()
        ]
    # The C stream carries representable typed mantissas, with the original
    # full native descriptor retained separately and no per-cell tree.
    table = pa.table({'value': storage})
    reader = pa.RecordBatchReader.from_stream(table)
    restored = reader.read_all()['value']
    del original, series, chunks, storage, table, reader
    gc.collect()
    values = [
        str(value) if value is not None else None
        for value in restored.to_pylist()
    ]
    assert values == ['1', None, '2', '0']


@pytest.mark.parametrize('dtype', [pa.int64(), pa.binary(), pa.string()])
def test_ordinary_storage_is_not_interpreted_as_decimal(dtype):
    values = (
        [1]
        if pa.types.is_integer(dtype)
        else [b'1']
        if pa.types.is_binary(dtype)
        else ['1']
    )
    series = pd.Series(
        pd.arrays.ArrowExtensionArray(pa.array(values, type=dtype))
    )
    assert native_decimal_column(series) is None


def test_table_result_preserves_logical_type_attachment_with_owned_rows():
    from sdm_client._rust import results

    table = results.expect(
        {'op': 'describe_source', 'description': 1, 'frame_rows': 2, 'frame_decimals': 3},
        results.Described,
    )
    assert table.frame_rows == 2
    assert table.frame_decimals == 3
    sql = results.expect({'op': 'describe_source', 'description': 1}, results.Described)
    assert sql.frame_rows is None
    assert sql.frame_decimals is None


@pytest.mark.parametrize('value', [True, -1, '3', 1.5, [], {}])
def test_table_result_rejects_malformed_logical_type_attachment(value):
    from sdm_client._rust import results

    with pytest.raises(StructuredDataError, match='frame_decimals is not a count'):
        results.expect(
            {'op': 'describe_source', 'description': 1, 'frame_rows': 2, 'frame_decimals': value},
            results.Described,
        )
