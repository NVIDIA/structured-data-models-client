# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless native Decimal buffers and their original logical descriptors.

The native scale is an i32 property; Arrow Rust's Decimal type has an i8
scale. Export typed scale-zero mantissas and retain the original descriptor.
No native scalar is converted and no schema meaning is chosen here.
"""

from __future__ import annotations

import pandas as pd
import pyarrow as pa


def native_decimal_type(dtype: object) -> dict[str, str | int] | None:
    """Original immutable built-in native type facts, without reading cells."""
    if type(dtype) is not pd.ArrowDtype:
        return None
    physical = dtype.pyarrow_dtype
    tag = next(
        (
            f'decimal{bits}'
            for bits in (32, 64, 128, 256)
            if (check := getattr(pa.types, f'is_decimal{bits}', None)) is not None
            and check(physical)
        ),
        None,
    )
    if tag is None:
        return None
    return {
        'type': tag,
        'precision': physical.precision,
        'scale': physical.scale,
    }


def native_decimal_column(
    series: pd.Series,
) -> tuple[pa.ChunkedArray, dict[str, str | int]] | None:
    """Export an exact built-in native Decimal column at its consumed stage.

    The caller binds the returned descriptor to the same field and frame
    attachment as the returned storage, and supplies fresh facts after any
    intervening native effects. Custom array conversion remains a separate
    native-family obligation; this helper never calls a custom Arrow hook.
    """
    if type(series) is not pd.Series:
        return None
    dtype = series.dtype
    native = series.array
    if (
        type(dtype) is not pd.ArrowDtype
        or type(native) is not pd.arrays.ArrowExtensionArray
    ):
        return None
    descriptor = native_decimal_type(dtype)
    if descriptor is None:
        return None
    physical = dtype.pyarrow_dtype
    # Decimal32/64 remain their original typed zero-copy views here.
    # Existing shared physical admission widens them exactly to128.
    make = getattr(pa, f'decimal{physical.bit_width}')
    storage_type = make(physical.precision, 0)
    original = native.__arrow_array__()
    chunks = [
        pa.Array.from_buffers(
            storage_type,
            len(chunk),
            chunk.buffers(),
            null_count=chunk.null_count,
            offset=chunk.offset,
        )
        for chunk in original.chunks
    ]
    storage = pa.chunked_array(chunks, type=storage_type)
    return storage, descriptor
