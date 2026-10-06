# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""How a semantic model was read where nothing was lost, as data.

Each note renders its own text, so a caller that renames a table renders
the note again rather than rewriting its words.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias


def quoted_list(names: Sequence[str]) -> str:
    r"""``'a'``, ``'a' and 'b'``, or ``'a', 'b' and 'c'``."""
    quoted = [f"'{name}'" for name in names]
    if len(quoted) <= 1:
        return ''.join(quoted)
    return f'{", ".join(quoted[:-1])} and {quoted[-1]}'


def _pql_name(name: str) -> str:
    return name if re.fullmatch(r'[A-Za-z_]\w*', name) else f'`{name}`'


@dataclass(frozen=True)
class MeasureNote:
    r"""A measure of the view and the predictive query aggregation it is.

    Args:
        measure: The measure's name.
        table: The graph table it aggregates, the view's source.
        function: ``'SUM'``, ``'AVG'``, ``'MIN'``, ``'MAX'``, ``'COUNT'`` or
            ``'COUNT_DISTINCT'``.
        column: The column it aggregates, or ``None`` for a count of rows.
    """

    measure: str
    table: str
    function: str
    column: str | None

    def aggregation(
        self, start: int = 0, end: int = 30, unit: str = 'days'
    ) -> str:
        r"""The aggregation over a window, such as
        ``SUM(order_lines.amount, 0, 30, days)``.
        """
        column = '*' if self.column is None else _pql_name(self.column)
        return (
            f'{self.function}({_pql_name(self.table)}.{column}, {start}, '
            f'{end}, {unit})'
        )

    def __str__(self) -> str:
        return f"'{self.measure}' is {self.aggregation()} over the next 30 days"


@dataclass(frozen=True)
class TimeColumnNote:
    r"""The time column taken for a table in place of the dimensions that
    truncate or cast it, or because a window measure orders by it.

    Args:
        table: The graph table.
        column: The time column taken.
        instead_of: The dimensions that truncate or cast it.
        source_column: The source column it is, where a dimension has that
            column's name and it was added under a name of its own.
        added: Whether it was added to the table for the purpose.
        window_order: Whether a window measure of the view orders by it.
    """

    table: str
    column: str
    instead_of: tuple[str, ...] = ()
    source_column: str | None = None
    added: bool = False
    window_order: bool = False

    def __str__(self) -> str:
        reasons: list[str] = []
        if self.window_order:
            reasons.append('a window measure of the view orders by it')
        if self.instead_of:
            verbs = (
                'truncates or casts'
                if len(self.instead_of) == 1
                else 'truncate or cast'
            )
            reasons.append(
                f'{quoted_list(self.instead_of)} {verbs} it, and it says when '
                f'a row happened more precisely'
            )
        added = renamed = ''
        if self.source_column is not None:
            renamed = (
                f". It is the source column '{self.source_column}', added "
                f'under a name of its own because a dimension has its name'
            )
        elif self.added:
            added = ', and added it as a column'
        return (
            f"Took '{self.column}' as the time column of table "
            f"'{self.table}'{added}, since {' and '.join(reasons)}{renamed}"
        )


@dataclass(frozen=True)
class ParameterNote:
    r"""A parameter the view references, read as its default.

    Args:
        parameter: The parameter's name.
        default: Its default, as text.
    """

    parameter: str
    default: str

    def __str__(self) -> str:
        return (
            f"Read the parameter '{self.parameter}' as its default "
            f"'{self.default}', as a query of the view without arguments does"
        )


ConversionNote: TypeAlias = MeasureNote | TimeColumnNote | ParameterNote
