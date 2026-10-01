# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

import re
import warnings

import pandas as pd

from sdm_relational_engine.rfm.base.utils import is_datetime, to_datetime


def infer_time_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    r"""Auto-detect potential time column.

    Args:
        df: The pandas DataFrame to analyze.
        candidates: A list of potential candidates.

    Returns:
        The name of the detected time column, or ``None`` if not found.
    """
    candidates = [  # Exclude all candidates with `*last*` in column names:
        col_name
        for col_name in candidates
        if not re.search(r'(^|_)last(_|$)', col_name, re.IGNORECASE)
    ]

    if len(candidates) == 0:
        return None

    if len(candidates) == 1:
        return candidates[0]

    # If there exists a dedicated `create*` column, use it as time column:
    create_candidates = [
        candidate
        for candidate in candidates
        if candidate.lower().startswith('create')
    ]
    if len(create_candidates) == 1:
        return create_candidates[0]
    if len(create_candidates) > 1:
        candidates = create_candidates

    datetime_candidates = [
        candidate for candidate in candidates if is_datetime(df[candidate])
    ]
    if len(datetime_candidates) == 1:
        return datetime_candidates[0]
    if len(datetime_candidates) > 1:
        candidates = datetime_candidates

    # Find the most optimal time column. Usually, it is the one pointing to
    # the oldest timestamps:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Could not infer format')
        min_timestamp_dict = {
            key: to_datetime(df[key].iloc[:10_000]) for key in candidates
        }
    min_timestamp_dict = {
        key: value.min() for key, value in min_timestamp_dict.items()
    }
    min_timestamp_dict = {
        key: value
        for key, value in min_timestamp_dict.items()
        if not pd.isna(value)
    }

    if len(min_timestamp_dict) == 0:
        return None

    return min(min_timestamp_dict, key=min_timestamp_dict.__getitem__)
