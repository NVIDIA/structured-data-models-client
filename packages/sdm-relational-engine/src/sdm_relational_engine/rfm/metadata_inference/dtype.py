# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

import numpy as np
import pandas as pd
import pyarrow as pa
from predictive_query.schema.typing import Dtype

from sdm_relational_engine.rfm.base.utils import is_datetime

# What `pandas.api.types.infer_dtype` reports for the contents of an `object`
# column, mapped to the data type those values actually are. `object` is a
# container rather than a type: `pd.read_sql` returns `object` for a
# `NUMERIC`/`DECIMAL` column on Postgres, Oracle, MySQL and most ODBC drivers,
# and calling that a string demotes a quantity to a category (or, once it is
# high-cardinality enough to be inferred as text, to a bag of word tokens split
# on the decimal point).
OBJECT_CONTENT_TO_DTYPE: dict[str, Dtype] = {
    'integer': Dtype.int,
    'floating': Dtype.float,
    'mixed-integer-float': Dtype.float,
    'decimal': Dtype.float,
    'boolean': Dtype.bool,
    'timedelta': Dtype.timedelta,
}

PANDAS_TO_DTYPE: dict[str, Dtype] = {
    'bool': Dtype.bool,
    'boolean': Dtype.bool,
    # Every unsigned width, not just `uint8`: `astype('uint32')` is a routine
    # memory optimisation and what an unsigned Parquet or Arrow column arrives
    # as, and the SQL backends already map `UINTEGER`/`UBIGINT` to `Dtype.int`
    # from the catalog without consulting this table.
    'uint8': Dtype.int,
    'uint16': Dtype.int,
    'uint32': Dtype.int,
    'uint64': Dtype.int,
    'int8': Dtype.int,
    'int16': Dtype.int,
    'int32': Dtype.int,
    'int64': Dtype.int,
    'float': Dtype.float,
    'double': Dtype.float,
    'float16': Dtype.float,
    'float32': Dtype.float,
    'float64': Dtype.float,
    'object': Dtype.string,
    'str': Dtype.string,
    'string': Dtype.string,
    'string[python]': Dtype.string,
    'string[pyarrow]': Dtype.string,
    'binary': Dtype.binary,
    'binary[python]': Dtype.binary,
    'binary[pyarrow]': Dtype.binary,
}


def infer_dtype(ser: pd.Series) -> Dtype:
    r"""Extracts the :class:`Dtype` from a :class:`pandas.Series`.

    Args:
        ser: A :class:`pandas.Series` to analyze.

    Returns:
        The data type.
    """
    if is_datetime(ser):
        return Dtype.date
    if pd.api.types.is_timedelta64_dtype(ser.dtype):
        return Dtype.timedelta
    if isinstance(ser.dtype, pd.CategoricalDtype):
        return Dtype.string

    if pd.api.types.is_object_dtype(ser.dtype) and not isinstance(
        ser.dtype, pd.ArrowDtype
    ):
        index = ser.iloc[:1000].first_valid_index()
        if index is not None and pd.api.types.is_list_like(ser[index]):
            pos = ser.index.get_loc(index)
            assert isinstance(pos, int)
            ser = ser.iloc[pos : pos + 1000].dropna()
            arr = pa.array(ser.tolist())
            ser = pd.Series(arr, dtype=pd.ArrowDtype(arr.type))
        else:
            content = pd.api.types.infer_dtype(ser.iloc[:1000], skipna=True)
            if content in OBJECT_CONTENT_TO_DTYPE:
                return OBJECT_CONTENT_TO_DTYPE[content]

    if isinstance(ser.dtype, pd.ArrowDtype):
        if pa.types.is_list(
            ser.dtype.pyarrow_dtype
        ) or pa.types.is_fixed_size_list(ser.dtype.pyarrow_dtype):
            elem_dtype = ser.dtype.pyarrow_dtype.value_type
            if pa.types.is_integer(elem_dtype):
                return Dtype.intlist
            if pa.types.is_floating(elem_dtype):
                return Dtype.floatlist
            if pa.types.is_decimal(elem_dtype):
                return Dtype.floatlist
            if pa.types.is_string(elem_dtype):
                return Dtype.stringlist
            if pa.types.is_null(elem_dtype):
                return Dtype.floatlist

    if isinstance(ser.dtype, np.dtype):
        dtype_str = str(ser.dtype).lower()
    elif isinstance(ser.dtype, pd.api.extensions.ExtensionDtype):
        dtype_str = ser.dtype.name.lower()
        dtype_str = dtype_str.split('[')[0]  # Remove backend metadata
    elif isinstance(ser.dtype, pa.DataType):
        dtype_str = str(ser.dtype).lower()
    else:
        dtype_str = 'object'

    if dtype_str not in PANDAS_TO_DTYPE:
        raise ValueError(f"Unsupported data type '{ser.dtype}'")

    return PANDAS_TO_DTYPE[dtype_str]
