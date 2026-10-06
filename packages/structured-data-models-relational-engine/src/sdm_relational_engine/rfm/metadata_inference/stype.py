# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from predictive_query.schema.typing import Dtype, Stype

from sdm_relational_engine.rfm.metadata_inference import (
    contains_categorical,
    contains_id,
    contains_multicategorical,
    contains_timestamp,
)


def infer_stype(ser: pd.Series, column_name: str, dtype: Dtype) -> Stype:
    r"""Infers the :class:`Stype` from a :class:`pandas.Series`.

    Args:
        ser: A :class:`pandas.Series` to analyze.
        column_name: The column name.
        dtype: The data type.

    Returns:
        The semantic type.
    """
    if contains_id(ser, column_name, dtype):
        return Stype.ID

    if contains_timestamp(ser, column_name, dtype):
        return Stype.timestamp

    if contains_multicategorical(ser, column_name, dtype):
        return Stype.multicategorical

    if contains_categorical(ser, column_name, dtype):
        return Stype.categorical

    return dtype.default_stype
