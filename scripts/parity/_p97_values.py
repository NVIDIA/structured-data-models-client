# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Preserve scalar types, array shapes and ordered table inputs in recipes."""

from __future__ import annotations

import math


def typed(value):
    if isinstance(value, dict):
        return {
            'mapping': [
                [typed(key), typed(item)] for key, item in value.items()
            ]
        }
    if type(value) in {list, tuple}:
        return {
            'list' if type(value) is list else 'tuple': [
                typed(item) for item in value
            ]
        }
    if value is None or type(value) in {str, bool, int}:
        return {'scalar': value}
    if type(value) is float:
        return {'float': value.hex() if math.isfinite(value) else str(value)}
    import numpy as np
    import pandas as pd

    if isinstance(value, np.generic):
        return {
            'numpy_scalar': {
                'dtype': value.dtype.str,
                'bytes_hex': value.tobytes().hex(),
            }
        }
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        timestamp = isinstance(value, pd.Timestamp)
        return {
            'pandas_scalar': {
                'class': type(value).__name__,
                'value': typed(value.asm8),
                'timezone': str(value.tz)
                if timestamp and value.tz is not None
                else None,
                'fold': value.fold if timestamp else None,
            }
        }
    if isinstance(value, pd.Index):
        multi = isinstance(value, pd.MultiIndex)
        return {
            'index': {
                'class': type(value).__name__,
                'dtype': str(value.dtype),
                'names': typed(list(value.names)),
                'values': typed(value.tolist()),
                'levels': [typed(level) for level in value.levels]
                if multi
                else None,
                'codes': [typed(code) for code in value.codes]
                if multi
                else None,
            }
        }
    if isinstance(value, pd.DataFrame):
        return {
            'frame': {
                'columns': typed(value.columns),
                'index': typed(value.index),
                'attrs': typed(value.attrs),
                'series': [
                    typed(value.iloc[:, ordinal])
                    for ordinal in range(len(value.columns))
                ],
            }
        }
    if isinstance(value, pd.Series):
        categorical = isinstance(value.dtype, pd.CategoricalDtype)
        return {
            'series': {
                'dtype': str(value.dtype),
                'name': typed(value.name),
                'index': typed(value.index),
                'attrs': typed(value.attrs),
                'values': typed(value.tolist()),
                'categories': typed(value.cat.categories)
                if categorical
                else None,
                'ordered': value.cat.ordered if categorical else None,
            }
        }
    if isinstance(value, np.ndarray):
        return {
            'ndarray': {
                'dtype': value.dtype.str,
                'descriptor': typed(value.dtype.descr),
                'shape': list(value.shape),
                'values': typed(value.tolist()),
            }
        }
    import _fixture_ops

    return {'fixture_scalar': _fixture_ops.encode(value)}
