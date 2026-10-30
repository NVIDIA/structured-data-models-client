# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed facts from the actual sampled index; no native sorting is performed."""

from __future__ import annotations

import hashlib
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sdm_client._routing import RouteError

# Official CPython 3.12 Linux wheels with identical sort sources and compiler.
# Acquisition receipts are recorded in the Phase 4 producer audit.
_PRODUCERS = {
    'ceec5ba13ab88318632307fdd2df74f317fa293ce77be559c4298ddd1f4306b0',
    'b185dfe64b07180bedc7c15fb2735b0d0f4f4e09eb9c15bd8e67f74568aebcbf',
}


@lru_cache(maxsize=4)
def _producer(path: str) -> bool:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() in _PRODUCERS


def _profile() -> str:
    from numpy._core import _multiarray_umath as native

    if not _producer(native.__file__):
        raise RouteError('this native index sort producer has not been bound to a core profile')
    # These are this producer's actual dispatched groups, not CPU-name guesses.
    # Disabling AVX2 alone does not necessarily disable its X86_V3 dispatcher.
    features = native.__cpu_features__
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
    raise RouteError('this object index comparison family is not yet represented by the core')


def _array(values: Any, missing: list[bool], profile: str) -> dict[str, Any]:
    dtype = values.dtype
    if isinstance(dtype, np.dtype) and dtype.kind in 'biu':
        unsigned = dtype.kind in 'bu'
        result = {'kind': 'unsigned' if unsigned else 'signed',
                  'values': [0 if absent else int(value) for value, absent in zip(values, missing, strict=True)],
                  'width': dtype.itemsize}
    elif isinstance(dtype, np.dtype) and dtype.kind == 'f' and dtype.itemsize <= 8:
        result = {'kind': 'floating', 'width': dtype.itemsize,
                  'bits': [0 if absent else _bits(value, dtype.itemsize) for value, absent in zip(values, missing, strict=True)]}
    elif isinstance(dtype, np.dtype) and dtype.kind in 'mM':
        result = {'kind': 'temporal', 'values': values.view('i8').tolist()}
    else:
        # Read object values only after their native missingness observation.
        atoms = [{'kind': 'integer', 'value': '0'} if absent else _scalar(value)
                 for value, absent in zip(values, missing, strict=True)]
        if all(absent or atom['kind'] == 'text' for atom, absent in zip(atoms, missing, strict=True)):
            result = {'kind': 'unicode', 'values': [atom.get('points', []) for atom in atoms]}
        else:
            result = {'kind': 'object', 'values': atoms}
    return {'values': result, 'missing': missing, 'profile': profile}


def _facts(index: pd.Index, profile: str) -> dict[str, Any]:
    ordinary = (pd.Index, pd.RangeIndex, pd.DatetimeIndex, pd.TimedeltaIndex,
                pd.PeriodIndex, pd.CategoricalIndex, pd.MultiIndex)
    if type(index) not in ordinary:
        raise RouteError('custom index producers require their separately approved sorting route')
    if type(index) is pd.MultiIndex:
        return {'values': {'kind': 'multi',
                           'levels': [_facts(level, profile) for level in index.levels],
                           'codes': [code.tolist() for code in index.codes]},
                'missing': [False] * len(index), 'profile': 'stable'}
    if type(index) is pd.CategoricalIndex:
        codes = index.codes
        return {'values': {'kind': 'categorical', 'codes': codes.tolist(), 'width': codes.dtype.itemsize},
                'missing': (codes == -1).tolist(), 'profile': profile}
    values = index._values
    if type(values) is np.ndarray:
        return _array(values, pd.isna(values).tolist(), profile)
    if type(values) in (pd.arrays.DatetimeArray, pd.arrays.TimedeltaArray, pd.arrays.PeriodArray):
        return _array(values._ndarray, values.isna().tolist(), profile)
    if type(values) in (pd.arrays.IntegerArray, pd.arrays.FloatingArray, pd.arrays.BooleanArray):
        # Preserve nullable validity separately from any present NaN value.
        missing = values.isna().tolist()
        return _array(values._data, missing, profile)
    if type(values) is pd.arrays.StringArray:
        return _array(values._ndarray, values.isna().tolist(), profile)
    if type(values) in (pd.arrays.ArrowExtensionArray, pd.arrays.ArrowStringArray):
        import pyarrow as pa

        array = values._pa_array
        dtype = array.type
        missing = array.is_null().to_pylist()
        if pa.types.is_timestamp(dtype) or pa.types.is_duration(dtype):
            data = array.cast(pa.int64()).fill_null(0).to_numpy()
            return {'values': {'kind': 'temporal', 'values': data.tolist()},
                    'missing': missing, 'profile': 'stable'}
        if pa.types.is_integer(dtype) or pa.types.is_boolean(dtype) or pa.types.is_floating(dtype):
            data = array.fill_null(False if pa.types.is_boolean(dtype) else 0).to_numpy()
            return _array(data, missing, 'stable')
        if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
            return {'values': {'kind': 'unicode',
                               'values': [[] if absent else [ord(char) for char in value]
                                          for value, absent in zip(array.to_pylist(), missing, strict=True)]},
                    'missing': missing, 'profile': 'stable'}
        raise RouteError('this Arrow index ordering family is not yet represented by the core')
    raise RouteError('this index array comparison family is not yet represented by the core')


def observe(index: pd.Index) -> dict[str, Any]:
    """Observe a sampled subset after reduction, without sorting in Python."""
    return {'index': _facts(index, _profile())}


def has_custom_sorter(frame: pd.DataFrame) -> bool:
    """Classify sorting overrides without invoking them or reading labels."""
    import inspect

    def custom(owner: Any, names: tuple[str, ...]) -> bool:
        for name in names:
            method = inspect.getattr_static(owner, name, None)
            if method is None:
                continue
            module = getattr(method, '__module__', None) or getattr(
                getattr(method, '__objclass__', None), '__module__', ''
            )
            if not (module == 'numpy' or module.startswith(('pandas.', 'numpy.'))):
                return True
        return False

    def index_custom(index: pd.Index) -> bool:
        if custom(index, ('argsort', '_sort_levels_monotonic', '_get_codes_for_sorting')):
            return True
        if isinstance(index, pd.MultiIndex):
            levels = inspect.getattr_static(index, '_levels')
            return any(index_custom(level) for level in levels)
        data = inspect.getattr_static(index, '_data', None)
        return data is not None and custom(data, ('argsort', '_values_for_argsort'))

    if custom(frame, ('sort_index',)):
        return True
    # Reading the stored manager avoids firing a subclass's index property at
    # route-selection time, before the reference would consume that property.
    manager = inspect.getattr_static(frame, '_mgr')
    return index_custom(manager.axes[1])
