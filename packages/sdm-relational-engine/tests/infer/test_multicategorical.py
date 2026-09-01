# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from predictive_query.schema.typing import Dtype
from sdm_relational_engine.rfm.infer import contains_multicategorical


def test_contains_multicategorical() -> None:
    assert not contains_multicategorical(pd.Series([1]), 'test', Dtype.bool)
    assert contains_multicategorical(
        ser=pd.Series([['A', 'B'], ['A']]),
        column_name='test',
        dtype=Dtype.stringlist,
    )
    assert not contains_multicategorical(
        ser=pd.Series(['A', 'B', 'C']),
        column_name='test',
        dtype=Dtype.string,
    )
    assert contains_multicategorical(
        ser=pd.Series(['A|B', 'A', 'B', 'B|A']),
        column_name='test',
        dtype=Dtype.string,
    )
    assert contains_multicategorical(
        ser=pd.Series([[0, 1], [0]]),
        column_name='test',
        dtype=Dtype.intlist,
    )
    assert not contains_multicategorical(
        ser=pd.Series([None, None, True, False]),
        column_name='test',
        dtype=Dtype.string,
    )
