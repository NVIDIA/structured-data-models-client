# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct-operation transport, including batches of passive native values."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from sdm_client._rust import results


class DirectOperations:
    def __init__(self, call: Callable[..., Any], run: Callable[..., Any], table: Callable[..., Any]) -> None:
        self._call = call
        self._run = run
        self._table = table

    def __call__(self, operation: str, value: dict[str, Any]) -> Any:
        return self._call(operation, value)

    @staticmethod
    def passive(series: pd.Series, wire_dtype: str | None = None) -> bool:
        if wire_dtype not in (None, 'bool', 'int32', 'int64', 'float32', 'float64', 'string'):
            return False
        if type(series) is not pd.Series:
            return False
        array = series.array
        if type(array) not in (
            pd.arrays.NumpyExtensionArray, pd.arrays.IntegerArray,
            pd.arrays.FloatingArray, pd.arrays.BooleanArray,
            pd.arrays.StringArray, pd.arrays.ArrowStringArray,
            pd.arrays.ArrowExtensionArray,
        ):
            return False
        dtype = series.dtype
        if type(dtype) is pd.StringDtype:
            return wire_dtype in (None, 'string') and all(
                type(item) is str or item is None or item is pd.NA
                or type(item) is float and math.isnan(item) for item in array
            )
        if isinstance(dtype, np.dtype):
            return dtype.isnative and dtype.kind in 'biuf' and dtype.itemsize <= 8 and not (dtype.kind == 'f' and dtype.itemsize < 4)
        if type(dtype) is pd.ArrowDtype:
            import pyarrow as pa

            physical = dtype.pyarrow_dtype
            return (
                pa.types.is_integer(physical) or pa.types.is_boolean(physical)
                or pa.types.is_decimal(physical) or physical in (pa.float32(), pa.float64())
            )
        return type(dtype) in (
            pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype,
            pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype,
            pd.Float32Dtype, pd.Float64Dtype, pd.BooleanDtype,
        )

    def _column(self, operation: str, series: pd.Series, dtype: str | None = None, *, name: str | None = None) -> dict[str, Any]:
        frame = series.to_frame() if name is None else series.to_frame(name=name)
        header = {'rows': len(frame), 'columns': list(frame.columns)}
        value = header if dtype is None else {'column': header, 'dtype': dtype}
        finished = self._run(operation, value, {2: self._table(frame)}, values=2)
        result = results.expect(finished.result, results.DirectValue)
        return json.loads(finished.take(result.value))

    def infer_column(self, series: pd.Series) -> dict[str, Any] | None:
        if not self.passive(series):
            return None
        return self._column('infer_direct_column', series)

    def time_start(self, series: pd.Series, name: str) -> dict[str, Any]:
        return self._column('infer_direct_time_start', series, name=name)

    def encode_column(self, series: pd.Series, dtype: str) -> list[dict[str, Any]] | None:
        if not self.passive(series, dtype):
            return None
        return self._column('encode_direct_column', series, dtype)['values']
