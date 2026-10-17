# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from predictive_query.schema.typing import Dtype
from sdm_relational_engine.rfm.metadata_inference import infer_dtype


def test_infer_dtype_all_nan() -> None:
    ser = pd.Series([None, None, None, None])
    assert infer_dtype(ser) == Dtype.string


def test_infer_dtype_stringlist() -> None:
    arr = pa.array([[1, 2, 3], [4, 5], None])
    ser = pd.Series(arr)
    assert infer_dtype(ser) == Dtype.intlist
    ser = pd.Series(arr, dtype=pd.ArrowDtype(arr.type))
    assert infer_dtype(ser) == Dtype.intlist

    arr = pa.array([[1.0, 2, 3], [4, 5], None])
    ser = pd.Series(arr)
    assert infer_dtype(ser) == Dtype.floatlist
    ser = pd.Series(arr, dtype=pd.ArrowDtype(arr.type))
    assert infer_dtype(ser) == Dtype.floatlist

    arr = pa.array([['A', 'B', 'C'], ['D', 'E'], None])
    ser = pd.Series(arr)
    assert infer_dtype(ser) == Dtype.stringlist
    ser = pd.Series(arr, dtype=pd.ArrowDtype(arr.type))
    assert infer_dtype(ser) == Dtype.stringlist

    ser = pd.Series([None, [1, 2], [2, 1]])
    assert infer_dtype(ser) == Dtype.intlist

    ser = pd.Series([None, np.array([1, 2]), np.array([2, 1])])
    assert infer_dtype(ser) == Dtype.intlist

    ser = pd.Series([None, [1.0, 2.0], [2, 1]])
    assert infer_dtype(ser) == Dtype.floatlist

    ser = pd.Series([None, np.array([1.0, 2.0]), np.array([2, 1])])
    assert infer_dtype(ser) == Dtype.floatlist

    ser = pd.Series([None, ['A', 'B'], ['B', 'A']])
    assert infer_dtype(ser) == Dtype.stringlist

    ser = pd.Series([None, np.array(['A', 'B']), np.array(['B', 'A'])])
    assert infer_dtype(ser) == Dtype.stringlist

    ser = pd.Series([None, [], []])
    assert infer_dtype(ser) == Dtype.floatlist

    ser = pd.Series([None, np.array([]), np.array([])])
    assert infer_dtype(ser) == Dtype.floatlist

    ser = pd.Series([None, [], [1, 2]])
    assert infer_dtype(ser) == Dtype.intlist

    ser = pd.Series([None, np.array([]), np.array([1, 2])])
    assert infer_dtype(ser) == Dtype.floatlist

    ser = pd.Series([None, [1, None]])
    assert infer_dtype(ser) == Dtype.intlist

    ser = pd.Series([None, np.array([1, None])])
    assert infer_dtype(ser) == Dtype.intlist

    ser = pd.Series([None, [1.0, None]])
    assert infer_dtype(ser) == Dtype.floatlist

    ser = pd.Series([None, np.array([1.0, None])])
    assert infer_dtype(ser) == Dtype.floatlist

    with pytest.raises(pa.lib.ArrowTypeError, match='Expected bytes'):
        ser = pd.Series([None, ['A', 1]])
        infer_dtype(ser)

    ser = pd.Series([None, np.array(['A', 1])])
    assert infer_dtype(ser) == Dtype.stringlist

    with pytest.raises(ValueError, match=r'Unsupported data type'):
        ser = pd.Series([None, [True, False], [False, True]])
        infer_dtype(ser)

    with pytest.raises(ValueError, match=r'Unsupported data type'):
        ser = pd.Series([None, np.array([True, False])])
        infer_dtype(ser)

    with pytest.raises(ValueError, match='cannot mix list and non-list'):
        infer_dtype(pd.Series([None, ['A'], 1]))


@pytest.mark.parametrize('dtype', ['uint8', 'uint16', 'uint32', 'uint64'])
def test_infer_dtype_accepts_every_unsigned_width(dtype: str) -> None:
    """Only ``uint8`` was mapped, so ``astype('uint32')``, a routine memory
    optimisation, and what an unsigned Parquet column arrives as, made
    ``Graph.from_data`` raise. The SQL backends already map ``UINTEGER`` /
    ``UBIGINT`` to ``Dtype.int`` from the catalog, so refusing them here was an
    inconsistency rather than a decision.
    """
    ser = pd.Series(np.arange(8), dtype=dtype)
    assert infer_dtype(ser) == Dtype.int


@pytest.mark.parametrize('dtype', ['UInt8', 'UInt16', 'UInt32', 'UInt64'])
def test_infer_dtype_accepts_the_nullable_unsigned_widths(dtype: str) -> None:
    ser = pd.Series(pd.array([1, 2, None], dtype=dtype))
    assert infer_dtype(ser) == Dtype.int


def test_infer_dtype_still_refuses_a_type_it_cannot_carry() -> None:
    r"""The widening must not become "accept anything numeric-looking"."""
    with pytest.raises(ValueError, match='Unsupported data type'):
        infer_dtype(pd.Series(np.arange(4), dtype='complex128'))


def test_an_arrow_decimal_is_named_as_the_payload_names_it() -> None:
    r"""A scale-0 decimal wider than 64 bits was inferred an integer, and
    then stayed a decimal, since no int64 holds it.
    """
    import decimal

    def _series(values: list, precision: int, scale: int) -> pd.Series:
        arrow = pa.array(
            [None if v is None else decimal.Decimal(v) for v in values],
            pa.decimal128(precision, scale),
        )
        return pd.Series(arrow, dtype=pd.ArrowDtype(arrow.type))

    assert infer_dtype(_series(['7', None], 10, 0)) == Dtype.int
    assert infer_dtype(_series(['123456789012345678'], 38, 0)) == Dtype.int
    assert infer_dtype(_series([str(10**30)], 38, 0)) == Dtype.string
    assert infer_dtype(_series(['1.25'], 10, 2)) == Dtype.float
