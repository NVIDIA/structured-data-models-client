# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lossless native Decimal buffers, unlinked pending shared composition.

The native scale is an i32 property; Arrow Rust's Decimal type has an i8
scale. Export typed scale-zero mantissas and retain the original descriptor.
No native scalar is converted and no schema meaning is chosen here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pyarrow as pa


def native_decimal_column(
    series: pd.Series,
) -> tuple[pa.ChunkedArray, dict[str, Any]] | None:
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
    physical = dtype.pyarrow_dtype
    if pa.types.is_decimal128(physical):
        tag = 'decimal128'
        storage_type = pa.decimal128(physical.precision, 0)
    elif pa.types.is_decimal256(physical):
        tag = 'decimal256'
        storage_type = pa.decimal256(physical.precision, 0)
    else:
        return None
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
    return storage, {
        'type': tag,
        'precision': physical.precision,
        'scale': physical.scale,
    }
