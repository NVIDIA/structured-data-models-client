# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native observations around direct prediction dtype resolution and keys."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from sdm_client._rust import direct_cells, direct_dtypes, spec


def raw_dtype(series: pd.Series, declared: str | None, call: Callable[..., Any]) -> dict[str, Any]:
    if (
        declared in ('int32', 'int64')
        and pd.api.types.is_float_dtype(series.dtype)
        and series.isna().any()
    ):
        present = series.dropna()
        equal = present.empty or bool((present == present.astype('int64')).all())
        resolved = call('resolve_direct_dtype', {
            'kind': 'nullable_integer', 'declared': declared, 'equal': equal,
        })
        if resolved['dtype'] is not None:
            return {'dtype': resolved['dtype'], 'untyped': False}
    return direct_dtypes.observe(series, call)


def widen(before: str | None, arriving: dict[str, Any], call: Callable[..., Any]) -> str | None:
    return call('resolve_direct_dtype', {
        'kind': 'widen', 'before': before,
        'arriving': arriving['dtype'], 'untyped': arriving['untyped'],
    })['dtype']


def inspect(request: Any, call: Callable[..., Any], run: Callable[..., Any]) -> None:
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
    if not key or any(name not in context.columns or name not in scoring.columns for name in key):
        return
    answered = context[context[target].notna()]
    if answered.empty:
        return
    table = schema[entity]
    try:
        dtypes = {}
        for name in key:
            column = table.columns[name]
            declared = call('check_direct_preparation', {
                'kind': 'declarations', 'columns': [(name, column.dtype, column.untyped)],
            })['dtypes'][0][1]
            dtypes[name] = widen(declared, raw_dtype(scoring[name], declared, call), call) or 'string'
    except Exception:  # noqa: BLE001 - reference key dtype warning boundary
        return
    try:
        context_values = direct_cells.table(answered[key], 'context', dtypes, call)
    except Exception:  # noqa: BLE001 - reference key encoding warning boundary
        return
    if context_values.num_rows == 0:
        return
    try:
        scoring_values = direct_cells.table(scoring[key], 'predict', dtypes, call)
    except Exception:  # noqa: BLE001 - reference key encoding warning boundary
        return
    # Warning delivery is outside the conversion catches: warnings-as-errors
    # must still stop the request at this boundary.
    run('inspect_observed_keys', {
        'context': {'rows': context_values.num_rows, 'columns': context_values.column_names},
        'predict': {'rows': scoring_values.num_rows, 'columns': scoring_values.column_names},
        'target': target, 'scoring_rows': len(scoring),
    }, {2: context_values, 3: scoring_values}, context=2, predict=3)


def resolve(
    request: Any, bound: dict[str, Any], call: Callable[..., Any],
    flags: Callable[..., Any], header: Callable[..., Any],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    call('check_direct_preparation', {'kind': 'consistent', 'schema': spec.held_schema(request.schema)})
    schema = request.schema
    table = schema[request.entity_table]
    context = request.context.instance_table
    scoring = request.predict.instance_table
    declarations = call('check_direct_preparation', {
        'kind': 'declarations',
        'columns': [(name, column.dtype, column.untyped) for name, column in table.columns.items()],
    })
    dtypes = dict(declarations['dtypes'])
    for name in scoring.columns:
        if name in dtypes:
            dtypes[name] = widen(dtypes[name], raw_dtype(scoring[name], dtypes[name], call), call)
    resolved = call('resolve_observed_tabular', {
        'input': {
            'bound': bound, 'schema': spec.held_schema(schema),
            'flags': flags(schema, request.entity_table),
            'entity_table': request.entity_table,
            'target': request.task.target.column_name, 'kind': request.task.kind,
            'context': header(context), 'predict': header(scoring),
        },
        'dtypes': [(name, dtype or 'string') for name, dtype in dtypes.items()],
    })
    return table, context, scoring, resolved
