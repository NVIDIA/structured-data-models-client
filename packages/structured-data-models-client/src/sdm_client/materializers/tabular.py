# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Host value observations and request metadata for tabular materialization."""

from __future__ import annotations

import numbers
from decimal import Decimal
from typing import Any

import pandas as pd

from sdm_client.errors import StructuredDataError
from sdm_client.request import Request
from sdm_client.schema import SourceTable, Table

TABULAR_TASK_KINDS = frozenset({'classification', 'regression'})


def _is_whole(value: object) -> bool:
    r"""Whether ``value`` is a number with nothing after the decimal point."""
    if isinstance(value, bool):
        return True
    if isinstance(value, numbers.Integral):
        return True
    if isinstance(value, Decimal):
        return value.is_finite() and value == value.to_integral_value()
    if isinstance(value, numbers.Real):
        try:
            return float(value).is_integer()
        except (OverflowError, TypeError, ValueError):
            return False
    return False


def _holds_two_whole_numbers(target: pd.Series) -> bool:
    r"""Whether ``target`` is two distinct whole numbers and nothing else.

    The values decide, not their storage: a 0/1 column with a null arrives as
    ``float64`` and is still a label. Anything wider or fractional is left to
    ``task=``.
    """
    values = target.dropna().unique()
    if len(values) != 2:
        return False
    return all(_is_whole(value) for value in values)


def _asking(request: Request, query: Any) -> Request:
    """Record the question a request was built from.

    A session pins one question and only the rows travel on a scoring call, so
    the question is what tells one request from another. The relational route
    records it when it gathers; this route builds the request directly and did
    not, which left a tabular session unable to tell that it had been handed
    something else.
    """
    request.query = query.to_string()
    return request


def _rows(table: Table | SourceTable) -> pd.DataFrame:
    r"""The rows of a table the tabular model reads, which it holds in memory.

    `PredictiveQuery.materialize` sends a schema over tables in their source to
    the relational model, so only a hand-built route reaches here with one.
    """
    if isinstance(table, SourceTable):
        raise StructuredDataError(
            f'{table.name!r} stays in its source, and the tabular model reads '
            f'rows in memory; build the schema with Schema.from_data',
            code='INVALID_REQUEST',
        )
    return table.data
