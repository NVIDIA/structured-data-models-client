# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from predictive_query.schema.typing import Dtype
from sdm_relational_engine.rfm.metadata_inference import contains_timestamp


def test_contains_timestamp() -> None:
    invalid_ser = pd.Series(['A', 'B', 'C'])
    valid_ser = pd.Series(['1990', '1991', '1992'])

    assert contains_timestamp(invalid_ser, 'test', dtype=Dtype.date)
    assert contains_timestamp(invalid_ser, 'test', dtype=Dtype.time)
    assert not contains_timestamp(invalid_ser, 'datetime', dtype=Dtype.bool)
    assert not contains_timestamp(invalid_ser, 'datetime', dtype=Dtype.string)
    assert contains_timestamp(valid_ser, 'test', dtype=Dtype.string)
    assert not contains_timestamp(invalid_ser, 'test', dtype=Dtype.string)
