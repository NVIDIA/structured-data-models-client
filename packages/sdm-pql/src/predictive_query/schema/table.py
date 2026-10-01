# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from enum import Enum

from pydantic.dataclasses import dataclass

from predictive_query.schema.typing import (
    Dtype,
    Stype,
    compatible_field_validator,
)


class TimestampUnit(Enum):
    r"""A timestamp unit for a column."""

    #: Specified in seconds
    SECOND = 's'

    #: Specified in milliseconds
    MILLISECOND = 'ms'

    #: Specified in microseconds
    MICROSECOND = 'us'

    #: Specified in nanoseconds
    NANOSECOND = 'ns'


@dataclass(frozen=True)
class Column:
    r"""A column of a table."""

    name: str
    stype: Stype
    dtype: Dtype

    # If the column represents a timestamp, the format that the timestmap
    # is represented in:
    timestamp_format: TimestampUnit | str | None = None


@dataclass
class TableDefinition:
    r"""The columns of a table, and which of them carry its identity."""

    # List of ALL columns selected from source table
    cols: list[Column]

    # Name of the primary key column.
    pkey: str | None = None

    # Name of the time column, required to have stype=SemanticType.timestamp
    time_col: str | None = None

    # Name of the end time column:
    end_time_col: str | None = None

    @compatible_field_validator('time_col', 'end_time_col', 'pkey')
    def empty_str_to_none(cls, v: str | None) -> str | None:
        if v == '':
            return None
        return v
