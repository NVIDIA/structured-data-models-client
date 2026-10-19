# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What a Databricks failure says about the table or expression it read.

The SQL connector's errors carry no state of their own, only a message such
as ``[UNRESOLVED_COLUMN.WITH_SUGGESTION] ... SQLSTATE: 42703``, while a
Spark session's errors answer for theirs. Both are read here, so a caller
can leave out what Databricks refused and re-raise everything else.
"""

from __future__ import annotations

import re

from sdm_relational_engine.exceptions import (
    EmptyTableError,
    MetricViewAsTableError,
    TableNotFoundError,
)

_SQL_STATE_RE = re.compile(r'SQLSTATE:\s*([0-9A-Z]{5})')
# Where Databricks appends the statement that failed, which may quote
# anything, a state included.
_SQL_CONTEXT_RE = re.compile(r'\n\s*== SQL\b')
_ERROR_CLASS_RE = re.compile(r'\s*\[([A-Z0-9_]+(?:\.[A-Z0-9_]+)*)\]')

_PERMISSION_CLASSES = {'INSUFFICIENT_PERMISSIONS', 'PERMISSION_DENIED'}


def _answer(error: BaseException, *methods: str) -> str | None:
    r"""What the first of ``methods`` a Spark error has answers, if any."""
    for method in methods:
        answer = getattr(error, method, None)
        value = answer() if callable(answer) else None
        if isinstance(value, str) and value:
            return value
    return None


def sql_state(error: BaseException) -> str | None:
    r"""The SQLSTATE of a Databricks failure, if it says one.

    Databricks writes the state last, after a message that may quote a
    value, or a nested error, holding a state of its own, so the last one
    before the statement it appends is the failure's.
    """
    if state := _answer(error, 'getSqlState'):
        return state
    message = _SQL_CONTEXT_RE.split(str(error), maxsplit=1)[0]
    states = _SQL_STATE_RE.findall(message)
    return states[-1] if states else None


def error_class(error: BaseException) -> str | None:
    r"""The error class of a Databricks failure, such as
    ``UNRESOLVED_COLUMN.WITH_SUGGESTION``, if it says one.
    """
    if name := _answer(error, 'getCondition', 'getErrorClass'):
        return name
    match = _ERROR_CLASS_RE.match(str(error))
    return None if match is None else match.group(1)


def unusable_table_reason(error: BaseException) -> str | None:
    r"""Why a table that failed to be read cannot be used, as a clause, or
    ``None`` if the failure says nothing about the table.

    Unity Catalog answers a table you may not see as one that does not
    exist, so neither is told apart from the other.
    """
    if isinstance(error, MetricViewAsTableError):
        return (
            'its table is a metric view; a metric view built on another '
            "metric view is not supported, point the join's source at the "
            'underlying table'
        )
    if isinstance(error, EmptyTableError):
        return 'its table is empty'
    state = sql_state(error)
    if isinstance(error, TableNotFoundError) or state == '42P01':
        return 'its table does not exist or you may not see it'
    name = error_class(error)
    if state == '42501' or (
        name is not None and name.split('.')[0] in _PERMISSION_CLASSES
    ):
        return 'you may not read its table'
    return None


def refuses_expression(error: BaseException) -> bool:
    r"""Whether Databricks refused an expression for what it says or reads:
    a data error, SQLSTATE class ``22``, or a syntax or analysis error, class
    ``42``.
    """
    state = sql_state(error)
    return state is not None and state[:2] in ('22', '42')
