# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

r"""Turning a DataFrame into the wire's ``arrays`` table."""

from __future__ import annotations

from typing import Any

import pandas as pd

from sdm_client.wire.dtypes import serialize_column


def encode_table(frame: pd.DataFrame, dtypes: dict[str, str]) -> dict[str, Any]:
    r"""Encode ``frame`` as an ``arrays`` table.

    ``arrays`` is the only table format the contract defines today: an ordered
    list of column names and a list of row arrays positionally matching it.
    Arrow IPC is a reserved extension point everywhere in this system and is
    not implemented.

    Serialization is per column rather than per cell, because a column has one
    dtype and the rule for writing a value follows from it. The rows are then
    transposed out of the serialized columns, which is what keeps a column's
    values consistently written even where a row-major walk would see mixed
    types.

    A frame with no columns still carries its row count, so an empty-column
    table is a list of empty rows rather than an empty table. The two mean
    different things to the server.

    Args:
        frame: The rows to encode.
        dtypes: Wire dtype per column name. Must cover every column of
            ``frame``.
    """
    columns = list(frame.columns)
    if not columns:
        return {
            'format': 'arrays',
            'columns': [],
            'rows': [[] for _ in range(len(frame))],
        }
    serialized_columns = [
        serialize_column(frame[column], dtypes[column]) for column in columns
    ]
    rows = [list(row) for row in zip(*serialized_columns, strict=True)]
    return {'format': 'arrays', 'columns': columns, 'rows': rows}
