# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate source dtype names into scalar wire dtype names."""

from typing import Any

_WIRE_DTYPE = {
    'bool': 'bool',
    'int': 'int64',
    'byte': 'int32',
    'int16': 'int32',
    'int32': 'int32',
    'int64': 'int64',
    'float': 'float64',
    'float32': 'float32',
    'float64': 'float64',
    'string': 'string',
    'date': 'timestamp[us]',
    'time': 'timestamp[us]',
}


def wire_dtype(source_dtype: Any) -> str | None:
    r"""The wire dtype for a source column, or ``None`` if it has none."""
    return _WIRE_DTYPE.get(str(source_dtype))
