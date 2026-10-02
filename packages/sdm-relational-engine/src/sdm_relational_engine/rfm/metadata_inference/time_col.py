# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from predictive_query.time_columns import FullRange, choose_time_column


def infer_time_column(
    df: pd.DataFrame,
    candidates: list[str],
    *,
    full_range: FullRange | None = None,
) -> str | None:
    r"""Auto-detect potential time column.

    The rule is :func:`predictive_query.time_columns.choose_time_column`,
    the one ``Schema.from_data`` applies to a frame.

    Args:
        df: The pandas DataFrame to analyze.
        candidates: A list of potential candidates.
        full_range: Reads a column's MIN, MAX and count of values over the
            whole table, when ``df`` holds only some of its rows.

    Returns:
        The name of the detected time column, or ``None`` if not found.
    """
    return choose_time_column(df, candidates, full_range=full_range).column
