# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable facts about rows excluded while preparing a relational graph."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TableSanitizationReport:
    """Exclusive row-drop counts for one prepared input table."""

    table: str
    input_rows: int
    output_rows: int
    null_primary_key_rows: int
    duplicate_primary_key_rows: int
    null_time_rows: int

    def __post_init__(self) -> None:
        counts = (
            self.input_rows,
            self.output_rows,
            self.null_primary_key_rows,
            self.duplicate_primary_key_rows,
            self.null_time_rows,
        )
        if not isinstance(self.table, str) or any(
            type(value) is not int or value < 0 for value in counts
        ):
            raise ValueError('invalid sanitization table or count')
        if self.input_rows != sum(counts[1:]):
            raise ValueError('sanitization row counts do not balance')


@dataclass(frozen=True)
class SanitizationReport:
    """Preparation facts; unavailable never means no rows were dropped."""

    status: Literal['available', 'unavailable', 'not_applicable']
    tables: tuple[TableSanitizationReport, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, 'tables', tuple(self.tables))
        if any(
            not isinstance(table, TableSanitizationReport)
            for table in self.tables
        ):
            raise TypeError(
                'tables must contain TableSanitizationReport values'
            )
        if self.status not in ('available', 'unavailable', 'not_applicable'):
            raise ValueError('invalid sanitization report status')
        if self.status != 'available' and self.tables:
            raise ValueError('only available reports carry table counts')
