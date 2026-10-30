# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed facts from the actual sampled index; no native sorting is performed."""

from __future__ import annotations

import hashlib
import importlib
import struct
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sdm_client._routing import RouteError

# Official Linux wheel binaries, mapped to their source-backed dispatch family.
# Scalar argsort predates the numeric SIMD dispatch; AVX2 alone does not select
# it in NumPy 1.26. Acquisition and source receipts are in the producer audit.
_PRODUCERS = {
    'ceec5ba13ab88318632307fdd2df74f317fa293ce77be559c4298ddd1f4306b0': (
        'x86_v3_v4',
        'numpy25_weak',
    ),
    'b185dfe64b07180bedc7c15fb2735b0d0f4f4e09eb9c15bd8e67f74568aebcbf': (
        'x86_v3_v4',
        'numpy25_weak',
    ),
    'c535a00761735b19153a71e1702f66482838421c78f96ecc3fe2c8d060662fd9': (
        'scalar',
        'numpy123_legacy',
    ),
    '6a650810a8735cbdb988f74729973740a337645dd1585fef696e73e2dbc65beb': (
        'avx512_skx_or_scalar',
        'numpy126_legacy',
    ),
}


@lru_cache(maxsize=4)
def _producer_families(path: str) -> tuple[str, str] | None:
    return _PRODUCERS.get(hashlib.sha256(Path(path).read_bytes()).hexdigest())


def _producer(path: str) -> str | None:
    families = _producer_families(path)
    return None if families is None else families[0]


def _comparison_semantics() -> str:
    namespace = (
        'numpy.core' if np.__version__.startswith('1.') else 'numpy._core'
    )
    native = importlib.import_module(f'{namespace}._multiarray_umath')
    families = _producer_families(native.__file__)
    if families is None:
        raise RouteError(
            'this native scalar comparison producer has not been bound to core semantics'
        )
    semantics = families[1]
    # These families identify audited binaries, not arbitrary version strings.
    if semantics == 'numpy126_legacy':
        if getattr(np, '_get_promotion_state')() != 'legacy':
            raise RouteError(
                'this NumPy scalar promotion mode is not yet represented by the core'
            )
    return semantics


def _profile() -> str:
    namespace = (
        'numpy.core' if np.__version__.startswith('1.') else 'numpy._core'
    )
    native = importlib.import_module(f'{namespace}._multiarray_umath')

    path = native.__file__
    if path is None:
        raise RouteError('this native index sort producer has no binary path')
    family = _producer(path)
    if family is None:
        raise RouteError(
            'this native index sort producer has not been bound to a core profile'
        )
    # These are this producer's actual dispatched groups, not CPU-name guesses.
    # Disabling AVX2 alone does not necessarily disable its X86_V3 dispatcher.
    features = native.__cpu_features__
    if family == 'scalar':
        return 'scalar'
    if family == 'avx512_skx_or_scalar':
        if features.get('AVX512_SKX', False):
            raise RouteError(
                'this NumPy AVX512 argsort producer has not yet been bound '
                'to a core profile'
            )
        return 'scalar'
    if features.get('X86_V4', False):
        return 'avx512_gnu14'
    if features.get('X86_V3', False):
        return 'avx2_gnu14'
    return 'scalar'


def _bits(value: Any, width: int) -> int:
    if width <= 4:
        return int.from_bytes(struct.pack('>f', float(value)), 'big')
    return int.from_bytes(struct.pack('>d', float(value)), 'big')


def _scalar(value: Any) -> dict[str, Any]:
    # Preserve comparison semantics: numeric native scalars inside object arrays
    # may round mixed comparisons differently from Python integer owners.
    if type(value) in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.longlong,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.ulonglong,
    ):
        return {
            'kind': 'numpy_integer',
            'signed': value.dtype.kind == 'i',
            'width': value.dtype.itemsize,
            'value': str(int(value)),
        }
    if type(value) in (np.float16, np.float32, np.float64):
        return {
            'kind': 'numpy_floating',
            'width': value.dtype.itemsize,
            'bits': int.from_bytes(value.tobytes(), sys.byteorder),
        }
    if type(value) in (np.bool_, np.str_):
        value = value.item()
    if type(value) is bool:
        return {'kind': 'boolean', 'value': value}
    if type(value) is int:
        return {'kind': 'integer', 'value': str(value)}
    if type(value) is float:
        return {'kind': 'floating', 'bits': _bits(value, 8)}
    if type(value) is str:
        return {'kind': 'text', 'points': [ord(char) for char in value]}
    if type(value) is complex:
        return {
            'kind': 'complex',
            'real_bits': _bits(value.real, 8),
            'imag_bits': _bits(value.imag, 8),
        }
    if type(value) is Decimal and value.is_finite():
        parts = value.as_tuple()
        return {
            'kind': 'decimal',
            'negative': bool(parts.sign),
            'digits': ''.join(str(digit) for digit in parts.digits),
            'exponent': parts.exponent,
        }
    if type(value) is date:
        return {'kind': 'date', 'ordinal': value.toordinal()}
    if type(value) is datetime and (
        value.tzinfo is None or type(value.tzinfo) is timezone
    ):
        offset = value.utcoffset()
        return {
            'kind': 'date_time',
            'ordinal': value.toordinal(),
            'micros': (
                (value.hour * 3600 + value.minute * 60 + value.second)
                * 1_000_000
                + value.microsecond
            ),
            'offset_micros': (
                None
                if offset is None
                else (offset.days * 86400 + offset.seconds) * 1_000_000
                + offset.microseconds
            ),
        }
    raise RouteError(
        'this object index comparison family is not yet represented by the core'
    )


def _array(values: Any, missing: list[bool], profile: str) -> dict[str, Any]:
    dtype = values.dtype
    result: dict[str, Any]
    if isinstance(dtype, np.dtype) and dtype.kind in 'biu':
        unsigned = dtype.kind in 'bu'
        result = {
            'kind': 'unsigned' if unsigned else 'signed',
            'values': [
                0 if absent else int(value)
                for value, absent in zip(values, missing, strict=True)
            ],
            'width': dtype.itemsize,
        }
    elif (
        isinstance(dtype, np.dtype)
        and dtype.kind == 'f'
        and dtype.itemsize <= 8
    ):
        result = {
            'kind': 'floating',
            'width': dtype.itemsize,
            'bits': [
                0 if absent else _bits(value, dtype.itemsize)
                for value, absent in zip(values, missing, strict=True)
            ],
        }
    elif (
        isinstance(dtype, np.dtype)
        and dtype.kind == 'c'
        and dtype.itemsize in (8, 16)
    ):
        width = dtype.itemsize // 2
        result = {
            'kind': 'complex',
            'width': width,
            'real_bits': [
                0 if absent else _bits(value.real, width)
                for value, absent in zip(values, missing, strict=True)
            ],
            'imag_bits': [
                0 if absent else _bits(value.imag, width)
                for value, absent in zip(values, missing, strict=True)
            ],
        }
    elif isinstance(dtype, np.dtype) and dtype.kind in 'mM':
        result = {'kind': 'temporal', 'values': values.view('i8').tolist()}
    else:
        # Read object values only after their native missingness observation.
        atoms: list[dict[str, Any]] = [
            {'kind': 'integer', 'value': '0'} if absent else _scalar(value)
            for value, absent in zip(values, missing, strict=True)
        ]
        kinds = {atom['kind'] for atom in atoms}
        if 'decimal' in kinds and 'floating' in kinds:
            raise RouteError(
                'mixed Decimal/float index comparison effects are not yet '
                'represented by the core'
            )
        has_numpy_missing = any(
            absent and type(value) in (np.float16, np.float32, np.float64)
            for value, absent in zip(values, missing, strict=True)
        )
        if kinds & {'numpy_integer', 'numpy_floating'} or has_numpy_missing:
            if not kinds <= {
                'integer',
                'floating',
                'boolean',
                'numpy_integer',
                'numpy_floating',
            }:
                raise RouteError(
                    'this mixed NumPy object comparison family is not yet represented by the core'
                )
            for position, (value, absent) in enumerate(
                zip(values, missing, strict=True)
            ):
                if absent:
                    if value is None or value is pd.NA or value is pd.NaT:
                        atoms[position] = {
                            'kind': 'native_missing',
                            'owner': 'none'
                            if value is None
                            else 'ambiguous'
                            if value is pd.NA
                            else 'temporal',
                        }
                    elif type(value) in (
                        float,
                        np.float16,
                        np.float32,
                        np.float64,
                    ):
                        atoms[position] = _scalar(value)
                    else:
                        raise RouteError(
                            'this missing NumPy companion owner is not yet represented by the core'
                        )
            result = {
                'kind': 'numpy_object',
                'values': atoms,
                'semantics': _comparison_semantics(),
            }
        elif all(
            absent or atom['kind'] == 'text'
            for atom, absent in zip(atoms, missing, strict=True)
        ):
            result = {
                'kind': 'unicode',
                'values': [atom.get('points', []) for atom in atoms],
            }
        else:
            result = {'kind': 'object', 'values': atoms}
    return {'values': result, 'missing': missing, 'profile': profile}


def _facts(index: pd.Index, profile: str) -> dict[str, Any]:
    ordinary = (
        pd.Index,
        pd.RangeIndex,
        pd.DatetimeIndex,
        pd.TimedeltaIndex,
        pd.PeriodIndex,
        pd.CategoricalIndex,
        pd.MultiIndex,
    )
    if type(index) not in ordinary:
        raise RouteError(
            'custom index producers require their separately approved sorting route'
        )
    if type(index) is pd.MultiIndex:
        return {
            'values': {
                'kind': 'multi',
                'levels': [_facts(level, profile) for level in index.levels],
                'codes': [code.tolist() for code in index.codes],
            },
            'missing': [False] * len(index),
            'profile': 'stable',
        }
    if type(index) is pd.CategoricalIndex:
        codes = index.codes
        return {
            'values': {
                'kind': 'categorical',
                'codes': codes.tolist(),
                'width': codes.dtype.itemsize,
            },
            'missing': (codes == -1).tolist(),
            'profile': profile,
        }
    values = index._values
    if type(values) is np.ndarray:
        return _array(values, pd.isna(values).tolist(), profile)
    if type(values) in (
        pd.arrays.DatetimeArray,
        pd.arrays.TimedeltaArray,
        pd.arrays.PeriodArray,
    ):
        return _array(values._ndarray, values.isna().tolist(), profile)
    if type(values) in (
        pd.arrays.IntegerArray,
        pd.arrays.FloatingArray,
        pd.arrays.BooleanArray,
    ):
        # Preserve nullable validity separately from any present NaN value.
        missing = values.isna().tolist()
        return _array(values._data, missing, profile)
    if type(values) is pd.arrays.StringArray:
        return _array(values._ndarray, values.isna().tolist(), profile)
    if type(values) in (
        pd.arrays.ArrowExtensionArray,
        pd.arrays.ArrowStringArray,
    ):
        import pyarrow as pa

        array = values._pa_array
        dtype = array.type
        missing = array.is_null().to_pylist()
        if pa.types.is_timestamp(dtype) or pa.types.is_duration(dtype):
            data = array.cast(pa.int64()).fill_null(0).to_numpy()
            return {
                'values': {'kind': 'temporal', 'values': data.tolist()},
                'missing': missing,
                'profile': 'stable',
            }
        if (
            pa.types.is_integer(dtype)
            or pa.types.is_boolean(dtype)
            or pa.types.is_floating(dtype)
        ):
            data = array.fill_null(
                False if pa.types.is_boolean(dtype) else 0
            ).to_numpy()
            return _array(data, missing, 'stable')
        if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
            return {
                'values': {
                    'kind': 'unicode',
                    'values': [
                        [] if absent else [ord(char) for char in value]
                        for value, absent in zip(
                            array.to_pylist(), missing, strict=True
                        )
                    ],
                },
                'missing': missing,
                'profile': 'stable',
            }
        raise RouteError(
            'this Arrow index ordering family is not yet represented by the core'
        )
    raise RouteError(
        'this index array comparison family is not yet represented by the core'
    )


def observe(index: pd.Index) -> dict[str, Any]:
    """Observe a sampled subset after reduction, without sorting in Python."""
    return {'index': _facts(index, 'unbound')}


def bind_profile(observation: dict[str, Any]) -> dict[str, Any]:
    """Bind retained facts only after the core requests a native sort profile."""
    profile: str | None = None

    def bind(facts: dict[str, Any]) -> dict[str, Any]:
        nonlocal profile
        values = facts['values']
        selected = facts['profile']
        if selected == 'unbound':
            if values['kind'] == 'numpy_object':
                # Object argsort uses the generic scalar implementation even
                # when this binary dispatches numeric arrays to SIMD kernels.
                selected = 'scalar'
            else:
                if profile is None:
                    profile = _profile()
                selected = profile
        result = {
            **facts,
            'profile': selected,
        }
        if values['kind'] == 'multi':
            result['values'] = {
                **values,
                'levels': [bind(level) for level in values['levels']],
            }
        return result

    return {'index': bind(observation['index'])}


def has_custom_sorter(frame: pd.DataFrame) -> bool:
    """Classify sorting overrides without invoking them or reading labels."""
    import inspect

    def custom(owner: Any, names: tuple[str, ...]) -> bool:
        for name in names:
            method = inspect.getattr_static(owner, name, None)
            if method is None:
                continue
            module = (
                getattr(method, '__module__', None)
                or getattr(
                    getattr(method, '__objclass__', None), '__module__', ''
                )
                or ''
            )
            if not (
                module == 'numpy' or module.startswith(('pandas.', 'numpy.'))
            ):
                return True
        return False

    def index_custom(index: pd.Index) -> bool:
        if custom(
            index,
            ('argsort', '_sort_levels_monotonic', '_get_codes_for_sorting'),
        ):
            return True
        if isinstance(index, pd.MultiIndex):
            levels = inspect.getattr_static(index, '_levels')
            return any(index_custom(level) for level in levels)
        data = inspect.getattr_static(index, '_data', None)
        return data is not None and custom(
            data, ('argsort', '_values_for_argsort')
        )

    if custom(frame, ('sort_index',)):
        return True
    # Reading the stored manager avoids firing a subclass's index property at
    # route-selection time, before the reference would consume that property.
    manager = inspect.getattr_static(frame, '_mgr')
    return index_custom(manager.axes[1])
