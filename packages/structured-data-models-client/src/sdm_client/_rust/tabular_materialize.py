# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native frame conversion and take for core-owned tabular context selection."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from sdm_client._routing import RouteError
from sdm_client._rust import errors
from sdm_client.core.warnings import SDMWarning, caller_stacklevel
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from predictive_query import ValidatedPredictiveQuery

    from sdm_client import Request, Schema


_INTEGERS = frozenset(
    {
        int,
        bool,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    }
)


def _integer(value: Any) -> int | None:
    if type(value) not in _INTEGERS:
        raise RouteError(
            'tabular entity conversion beyond primitive integers is unfinished'
        )
    integer = int(value)
    return integer if -(2**63) <= integer < 2**63 else None


def _keys(frame: pd.DataFrame, column: str) -> list[int]:
    series = frame[column]
    if not isinstance(series, pd.Series) or series.dtype != np.dtype('int64'):
        raise RouteError(
            'tabular entity conversion beyond int64 storage is unfinished'
        )
    return series.tolist()


def _count(value: Any) -> int | None:
    if value is None:
        return None
    if type(value) not in _INTEGERS:
        raise RouteError('tabular custom count conversion is unfinished')
    count = int(value)
    if not 0 <= count <= np.iinfo(np.uintp).max:
        raise RouteError(
            'tabular count conversion beyond native row widths is unfinished'
        )
    return count


def materialize(
    schema: Schema,
    query: ValidatedPredictiveQuery,
    *,
    indices: Any = None,
    predict: pd.DataFrame | None = None,
    num_context_samples: int | None = None,
    task: str | None = None,
) -> Request:
    from predictive_query.validator.utils import col_name, table_name

    from sdm_client import Request, _native
    from sdm_client.materializers.tabular import (
        TabularMaterializer,
        _asking,
        _rows,
    )

    name = table_name(query.entity_column)
    entity = col_name(query.entity_column)
    target = col_name(query.target_column)
    table = _rows(schema[name])
    # The original reference is captured before consuming the indices iterable.
    asked = list(indices) if predict is None else None
    if (
        type(table) is not pd.DataFrame
        or type(table.index) is not pd.RangeIndex
        or table.index.step <= 0
    ):
        raise RouteError(
            'tabular subset sorting beyond increasing RangeIndex is unfinished'
        )
    if predict is None:
        assert asked is not None
        scored, missing = _native.tabular_indices(
            _keys(table, entity), [_integer(value) for value in asked]
        )
        if missing:
            # Core selected the missing request ordinals; preserve native repr.
            shown = [asked[position] for position in missing[:10]]
            tail = '' if len(missing) <= 10 else ', ...'
            raise StructuredDataError(
                f'{len(missing)} of the {len(asked)} indices are not in '
                f'{query.entity_column}: {shown}{tail}. Pass predict=<frame> '
                f'to score rows not present in the table',
                code='INVALID_REQUEST',
            )
        predict = table.iloc[scored].drop(columns=[target])
        remedy = 'Pass num_context_samples to choose how many, or select the rows yourself.'
    else:
        scored = []
        if entity in table.columns and entity in predict.columns:
            scored = _native.tabular_matched_rows(
                _keys(table, entity), _keys(predict, entity)
            )
        remedy = (
            'This route takes no num_context_samples, so choose the '
            'context by building the schema from the rows you want.'
        )
    known = (
        table[target].notna().tolist()
        if target in table.columns
        else [True] * len(table)
    )
    try:
        ordinals = _native.tabular_labelled_rows(known, scored)
        # Like the reference's mask, this captures values before cap advice.
        candidates = table.iloc[ordinals]
        context = _native.TabularContext(ordinals, _count(num_context_samples))
        count = context.notice_rows()
        wanted = context.wanted()
        if count is not None:
            warnings.warn(
                f'{name} has {count:,} rows that could be context and '
                f'the endpoint accepts {wanted:,}, so {wanted:,} were '
                f'sampled. {remedy}',
                category=SDMWarning,
                stacklevel=caller_stacklevel(),
            )
        positions = context.positions()
        selected = (
            candidates if positions is None else candidates.iloc[positions]
        )
    except _native.CoreError as error:
        errors.raise_from_core(
            error.control,
            error.attachments,
            internal_context=error,
            presentation='frame_constructor',
        )
    # Task derivation is the next T4.6 extraction; constructor/prediction reuse
    # their existing core routes when selected, rather than a second wire path.
    return _asking(
        Request(
            context=selected,
            predict=predict,
            target=target,
            task=TabularMaterializer._task_kind(query, selected[target], task),
            schema=schema.snapshot(),
            entity_table=name,
        ),
        query,
    )
