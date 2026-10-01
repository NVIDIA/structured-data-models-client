# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest
from predictive_query.schema.typing import Dtype
from sdm_relational_engine.rfm.base.mapper import Mapper


@pytest.mark.parametrize('dtype', [Dtype.int, Dtype.float, Dtype.string])
def test_mapper(dtype: Dtype) -> None:
    if dtype == Dtype.int:
        a = 2
        b = 1
        c = 0
    elif dtype == Dtype.float:
        a = 1.5
        b = 3.7
        c = 0.3
    else:
        assert dtype == Dtype.string
        a = 'A'
        b = 'B'
        c = 'C'

    mapper = Mapper(num_examples=2)
    out = mapper.get(
        pkey=pd.Series([a, b, c, a, b, c]),
        batch=np.array([1, 1, 1, 0, 0, 0]),
    )
    assert np.array_equal(out, np.array([-1, -1, -1, -1, -1, -1]))

    mapper.add(
        pkey=pd.Series([a, a, b, b]),
        batch=np.array([0, 0, 0, 1]),
    )
    out = mapper.get(
        pkey=pd.Series([a, b, c, a, b, c]),
        batch=np.array([1, 1, 1, 0, 0, 0]),
    )
    assert np.array_equal(out, np.array([-1, 2, -1, 0, 1, -1]))

    mapper.add(
        pkey=pd.Series([a, b]),
        batch=np.array([1, 1]),
    )
    out = mapper.get(
        pkey=pd.Series([a, b, c, a, b, c]),
        batch=np.array([1, 1, 1, 0, 0, 0]),
    )
    assert np.array_equal(out, np.array([3, 2, -1, 0, 1, -1]))
