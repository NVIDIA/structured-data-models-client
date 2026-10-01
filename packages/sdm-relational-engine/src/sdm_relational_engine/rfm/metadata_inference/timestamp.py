# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

import re
import warnings

import pandas as pd
from dateutil.parser import UnknownTimezoneWarning
from predictive_query.schema.typing import Dtype, Stype


def contains_timestamp(ser: pd.Series, column_name: str, dtype: Dtype) -> bool:
    if not Stype.timestamp.supports_dtype(dtype):
        return False

    if dtype.is_timestamp():
        return True

    column_name = column_name.lower()

    match = re.search(
        (
            '(^|_)(date|datetime|dt|time|timedate|timestamp|ts|'
            'created|updated)(_|$)'
        ),
        column_name,
        re.IGNORECASE,
    )
    score = 0.3 if match is not None else 0.0

    ser = ser.iloc[:100]
    ser = ser.dropna()
    ser = ser[ser != '']

    if len(ser) == 0:
        return False

    ser = ser.astype(str)  # Avoid parsing numbers as unix timestamps.

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UnknownTimezoneWarning)
        warnings.filterwarnings('ignore', message='Could not infer format')
        mask = pd.to_datetime(ser, errors='coerce').notna()
        score += int(mask.sum()) / len(mask)

    return score >= 1.0
