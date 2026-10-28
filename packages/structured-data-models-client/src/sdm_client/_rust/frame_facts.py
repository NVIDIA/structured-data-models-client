# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neutral facts from already reached native dtype/content observations."""

from __future__ import annotations

from typing import Literal, TypeAlias, TypedDict

import pandas as pd
import pyarrow as pa

from sdm_client._rust.frame_decimals import (
    _native_arrow_dtype,
    native_decimal_type,
)

ObservedContent: TypeAlias = Literal['text', 'decimal', 'temporal', 'other']


class PhysicalDeclaration(TypedDict):
    declaration: Literal['physical']
    physical: dict[str, str | None]


class LargeTextDeclaration(TypedDict):
    declaration: Literal['large_utf8']


class DecimalDeclaration(TypedDict):
    declaration: Literal['decimal']
    native: dict[str, str | int]


NativeDeclaration: TypeAlias = (
    PhysicalDeclaration | LargeTextDeclaration | DecimalDeclaration
)


class LogicalTextFacts(TypedDict):
    backing: Literal['element_values', 'offset_utf8']
    missing: Literal['null_scalar', 'ieee_nan']


class NativeSourceFacts(TypedDict):
    declared: NativeDeclaration | None
    logical_text: LogicalTextFacts | None


def native_declaration(dtype: object) -> NativeDeclaration | None:
    """Keep the original declared type, never the converted export's type.

    This describes existing native metadata only. The existing constructor
    converter still admits the current families, independently of this fact.
    """
    if not _native_arrow_dtype(dtype):
        return None
    physical = dtype.pyarrow_dtype
    decimal = native_decimal_type(dtype)
    if decimal is not None:
        return {'declaration': 'decimal', 'native': decimal}
    if pa.types.is_large_string(physical):
        return {'declaration': 'large_utf8'}
    if pa.types.is_string(physical):
        return {'declaration': 'physical', 'physical': {'type': 'utf8'}}
    if pa.types.is_boolean(physical):
        return {'declaration': 'physical', 'physical': {'type': 'bool'}}
    if pa.types.is_integer(physical):
        prefix = 'int' if pa.types.is_signed_integer(physical) else 'uint'
        return {
            'declaration': 'physical',
            'physical': {'type': f'{prefix}{physical.bit_width}'},
        }
    if pa.types.is_timestamp(physical):
        return {
            'declaration': 'physical',
            'physical': {
                'type': 'timestamp',
                'unit': physical.unit,
                'zone': physical.tz,
            },
        }
    return None


def native_source_facts(dtype: object) -> NativeSourceFacts:
    """Describe logical text separately from a fixed native declaration."""
    logical: LogicalTextFacts | None = None
    if (
        isinstance(dtype, pd.StringDtype)
        and type(dtype) is pd.StringDtype
        and dtype.storage in ('python', 'pyarrow')
    ):
        logical = {
            'backing': 'element_values'
            if dtype.storage == 'python'
            else 'offset_utf8',
            'missing': 'null_scalar' if dtype.na_value is pd.NA else 'ieee_nan',
        }
    return {'declared': native_declaration(dtype), 'logical_text': logical}


def observed_content(native: str | None) -> ObservedContent | None:
    """Adapt the native result without selecting schema or cast policy."""
    if native is None:
        return None
    if native == 'string':
        return 'text'
    if native == 'decimal':
        return 'decimal'
    if native in ('date', 'datetime', 'datetime64'):
        return 'temporal'
    return 'other'
