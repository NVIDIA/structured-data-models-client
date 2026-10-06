# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from sdm_relational_engine.exceptions import KumoRelationalError


class SanitizationStatus(str, Enum):
    r"""Whether a graph kept the counts of the rows it dropped."""

    AVAILABLE = 'available'
    NOT_AVAILABLE = 'not_available'


@dataclass(frozen=True)
class TableSanitizationReport:
    r"""Exclusive row-removal counts for one graph table.

    Reasons use this precedence: null primary key, duplicate primary key,
    then null timestamp. Consequently, ``input_rows - output_rows`` equals
    the sum of the three reason counts.
    """

    table_name: str
    input_rows: int
    output_rows: int
    null_primary_key_rows: int
    duplicate_primary_key_rows: int
    null_time_rows: int

    @property
    def total_dropped_rows(self) -> int:
        return self.input_rows - self.output_rows


@dataclass(frozen=True)
class GraphSanitizationReport:
    r"""Sanitization diagnostics for a graph backend."""

    status: SanitizationStatus
    tables: Mapping[str, TableSanitizationReport]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'tables', MappingProxyType(dict(self.tables)))

    @classmethod
    def not_available(cls) -> 'GraphSanitizationReport':
        return cls(status=SanitizationStatus.NOT_AVAILABLE, tables={})


class TaskReferenceError(KumoRelationalError, ValueError):
    r"""Raised when task rows reference entities absent after sanitization."""

    def __init__(self, table_name: str, unresolved_rows: int) -> None:
        self.table_name = table_name
        self.unresolved_rows = unresolved_rows
        super().__init__(
            f'Task references {unresolved_rows:,} unresolved row(s) in table '
            f"'{table_name}' after graph sanitization"
        )
