# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import pandas as pd
from predictive_query.schema.typing import Dtype, Stype


def contains_categorical(
    ser: pd.Series,
    column_name: str,
    dtype: Dtype,
) -> bool:

    if not Stype.categorical.supports_dtype(dtype):
        return False

    if dtype == Dtype.bool:
        return True

    if dtype.is_numerical():
        match = re.search(
            (
                r'(^|_)(price|sales|amount|quantity|total|cost|score|rating|'
                'avg|average|recency|age|num|pos|number|position)(_|$)'
            ),
            column_name,
            re.IGNORECASE,
        )
        if match is not None:
            return False

    ser = ser.iloc[:1000]
    ser = ser.dropna()

    num_unique = ser.nunique()

    if num_unique < 20:
        return True

    if dtype.is_string():
        return num_unique / len(ser) <= 0.5

    return num_unique / len(ser) <= 0.05
