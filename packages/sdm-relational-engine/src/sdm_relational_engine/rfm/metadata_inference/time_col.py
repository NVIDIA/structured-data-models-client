# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

import pandas as pd
from predictive_query.time_columns import (
    FullRange,
    RejectedTimeColumn,
    screen_time_columns,
)

from sdm_relational_engine.rfm.base.utils import is_datetime, to_datetime


def infer_time_column(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    full_range: FullRange | None = None,
) -> str | None:
    r"""Auto-detect potential time column.

    Args:
        df: The pandas DataFrame to analyze.
        candidates: A list of potential candidates.
        full_range: Reads a column's MIN and MAX over the whole table, when
            ``df`` holds only some of its rows.

    Returns:
        The name of the detected time column, or ``None`` if not found.
    """
    return choose_time_column(df, candidates, full_range=full_range)[0]


def choose_time_column(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    full_range: FullRange | None = None,
) -> tuple[str | None, tuple[RejectedTimeColumn, ...]]:
    r"""The inferred time column, and the candidates refused with why.

    See :func:`infer_time_column` for the arguments.
    """
    screen = screen_time_columns(df, candidates, full_range=full_range)
    if len(screen.eligible) <= 1:
        return screen.chosen, screen.rejected
    return _oldest(df, list(screen.eligible)), screen.rejected


def _oldest(df: pd.DataFrame, candidates: list[str]) -> str | None:
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
