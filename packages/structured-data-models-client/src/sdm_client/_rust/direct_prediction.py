# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native observations around direct prediction dtype resolution and keys."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from sdm_client._rust import direct_admission, direct_cells, direct_dtypes, errors, spec


def raw_dtype(
    series: pd.Series, declared: str | None, call: Callable[..., Any]
) -> dict[str, Any]:
    if (
        declared in ('int32', 'int64')
        and pd.api.types.is_float_dtype(series.dtype)
        and series.isna().any()
    ):
        present = series.dropna()
        equal = present.empty or bool(
            (present == present.astype('int64')).all()
        )
        resolved = call(
            'resolve_direct_dtype',
            {
                'kind': 'nullable_integer',
                'declared': declared,
                'equal': equal,
            },
        )
        if resolved['dtype'] is not None:
            return {'dtype': resolved['dtype'], 'untyped': False}
    return direct_dtypes.observe(series, call)


def widen(
    before: str | None, arriving: dict[str, Any], call: Callable[..., Any]
) -> str | None:
    return call(
        'resolve_direct_dtype',
        {
            'kind': 'widen',
            'before': before,
            'arriving': arriving['dtype'],
            'untyped': arriving['untyped'],
        },
    )['dtype']


def inspect(
    request: Any, call: Callable[..., Any], run: Callable[..., Any]
) -> None:
    from sdm_client.schema import _as_key_tuple

    schema, entity = request.schema, request.entity_table
    target = request.task.target.column_name
    if schema is None or entity is None or not target:
        return
    try:
        context = request.context.instance_table
        scoring = request.predict.instance_table
    except Exception:  # noqa: BLE001 - reference warning boundary
        return
    if target not in context.columns:
        return
    key = list(_as_key_tuple(schema[entity].primary_key))
    if not key or any(
        name not in context.columns or name not in scoring.columns
        for name in key
    ):
        return
    answered = context[context[target].notna()]
    if answered.empty:
        return
    table = schema[entity]
    try:
        dtypes = {}
        for name in key:
            column = table.columns[name]
            declared = call(
                'check_direct_preparation',
                {
                    'kind': 'declarations',
                    'columns': [(name, column.dtype, column.untyped)],
                },
            )['dtypes'][0][1]
            dtypes[name] = (
                widen(declared, raw_dtype(scoring[name], declared, call), call)
                or 'string'
            )
    except Exception:  # noqa: BLE001 - reference key dtype warning boundary
        return
    try:
        context_values = direct_cells.table(
            answered[key], 'context', dtypes, call
        )
    except Exception:  # noqa: BLE001 - reference key encoding warning boundary
        return
    if context_values.num_rows == 0:
        return
    try:
        scoring_values = direct_cells.table(
            scoring[key], 'predict', dtypes, call
        )
    except Exception:  # noqa: BLE001 - reference key encoding warning boundary
        return
    # Warning delivery is outside the conversion catches: warnings-as-errors
    # must still stop the request at this boundary.
    run(
        'inspect_observed_keys',
        {
            'context': {
                'rows': context_values.num_rows,
                'columns': context_values.column_names,
            },
            'predict': {
                'rows': scoring_values.num_rows,
                'columns': scoring_values.column_names,
            },
            'target': target,
            'scoring_rows': len(scoring),
        },
        {2: context_values, 3: scoring_values},
        context=2,
        predict=3,
    )


def resolve(
    request: Any,
    bound: dict[str, Any],
    call: Callable[..., Any],
    flags: Callable[..., Any],
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, str]]:
    call(
        'check_direct_preparation',
        {'kind': 'consistent', 'schema': spec.held_schema(request.schema)},
    )
    schema = request.schema
    table = schema[request.entity_table]
    context = request.context.instance_table
    scoring = request.predict.instance_table
    dtypes = {}
    for name, column in table.columns.items():
        declared = None if column.untyped else column.dtype
        dtypes[name] = declared
    for name in scoring.columns:
        if name in dtypes:
            dtypes[name] = widen(
                dtypes[name], raw_dtype(scoring[name], dtypes[name], call), call
            )
    dtypes = {name: dtype or 'string' for name, dtype in dtypes.items()}
    target = request.task.target.column_name
    missing = [name for name in context.columns if name not in dtypes]
    with errors.holding(direct_columns=lambda fields: missing):
        direct_admission.observe(call, 'schema_columns', missing=bool(missing))
    columns = {}
    as_text = []
    for name in context.columns:
        declared = table[name].stype
        if declared == 'text':
            as_text.append(name)
        wire_dtype = dtypes[name]
        different = None if declared == 'text' else bool(name != request.task.target.column_name)
        classification = request.task.is_classification if different is False else None
        selected = call('check_direct_preparation', {
            'kind': 'tabular_meaning', 'inferred': declared,
            'different': different, 'classification': classification,
        })['stype']
        actual_dtype = dtypes[name]
        column = table[name]
        coherent = call('check_direct_preparation', {
            'kind': 'coherent_meaning', 'inferred': selected,
            'dtype': actual_dtype, 'declared': None,
        })['stype']
        if coherent is None:
            coherent = call('check_direct_preparation', {
                'kind': 'coherent_meaning', 'inferred': selected,
                'dtype': actual_dtype, 'declared': bool(column.stype_declared),
            })['stype']
        columns[name] = {'name': name, 'dtype': wire_dtype, 'stype': coherent}
    resolved = call(
        'resolve_observed_tabular',
        {
            'bound': bound, 'target': target,
            'columns': list(columns.values()), 'text_columns': as_text,
            'dtypes': list(dtypes.items()),
        },
    )
    return table, context, scoring, resolved, dtypes
