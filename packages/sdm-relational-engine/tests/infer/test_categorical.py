# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest
from predictive_query.schema.typing import Dtype
from sdm_relational_engine.rfm.infer import contains_categorical


def test_contains_categorical() -> None:
    assert not contains_categorical(pd.Series([1]), 'test', Dtype.date)
    assert contains_categorical(pd.Series([1]), 'test', Dtype.bool)
    assert contains_categorical(
        ser=pd.Series(['A', 'B', 'C']),
        column_name='test',
        dtype=Dtype.string,
    )
    assert contains_categorical(
        ser=pd.Series(list(range(20)) * 20),
        column_name='test',
        dtype=Dtype.int,
    )
    assert not contains_categorical(
        ser=pd.Series(list(range(20)) * 19),
        column_name='test',
        dtype=Dtype.int,
    )


@pytest.mark.parametrize('column_name', ['flag', 'total', 'price', 'is_high'])
@pytest.mark.parametrize(
    'ser',
    [
        pd.Series([True, False, True]),
        pd.Series([], dtype='boolean'),
        pd.Series([pd.NA, pd.NA], dtype='boolean'),
    ],
)
def test_bool_is_always_categorical(ser: pd.Series, column_name: str) -> None:
    # A boolean column is categorical unconditionally: not via the sampling and
    # `nunique()` path below it, and not subject to the numeric name blocklist.
    assert contains_categorical(ser, column_name, Dtype.bool)
