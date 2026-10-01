# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

import re

import pandas as pd
from predictive_query.schema.typing import Dtype, Stype

# Column names suffixes that end in "id" but should not be given the ID stype.
_IGNORED_ID_SUFFIXES = [
    'bid',
    'acid',
    'grid',
    'maid',
    'paid',
    'raid',
    'void',
    'avoid',
    'braid',
    'covid',
    'fluid',
    'rabid',
    'solid',
    'hybrid',
    'inlaid',
    'liquid',
]


def contains_id(ser: pd.Series, column_name: str, dtype: Dtype) -> bool:
    if not Stype.ID.supports_dtype(dtype):
        return False

    column_name = column_name.lower()

    match = re.search(
        r'(^|_)(id|hash|key|code|uuid)(_|$)',
        column_name,
        re.IGNORECASE,
    )
    if match is not None:
        return True

    if not column_name.endswith('id'):
        return False
    for suffix in _IGNORED_ID_SUFFIXES:
        if column_name.endswith(suffix):
            return False
    return True
