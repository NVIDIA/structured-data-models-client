# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native schema observations sequenced around Rust-owned schema policy."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from sdm_client._rust import direct_cells, direct_dtypes, errors, spec
from sdm_client._rust.direct_io import DirectOperations


def time_column(
    frame: pd.DataFrame,
    columns: list[dict[str, Any]],
    key: str | None,
    call: DirectOperations,
) -> str | None:
    names = call(
        'observe_direct_schema',
        {
            'kind': 'time_candidates',
            'columns': columns,
            'primary_key': key,
        },
    )['names']
    populations = []
    for name in names:
        present = frame[name].dropna()
        rows, held = len(frame), len(present)
        first = present.iloc[0] if held else None
        same = held > 0 and bool((present == first).all())
        populations.append(
            {'name': name, 'rows': rows, 'held': held, 'same': same}
        )
    # Keep the population observations across parsing: native conversion
    # may mutate the source, but the reference does not reread populations.
    observed = {
        'kind': 'time_choice',
        'columns': columns,
        'primary_key': key,
        'rows': populations[-1]['rows'] if populations else len(frame),
        'populations': populations,
        'starts': None,
    }
    choice = call('observe_direct_schema', observed)
    if choice['earliest']:
        starts = []
        for name in choice['earliest']:
            values = frame[name]
            if isinstance(values.dtype, pd.ArrowDtype):
                values = pd.Series(values.to_numpy(), index=values.index)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    parsed = pd.to_datetime(values, errors='coerce', utc=True)
            except (TypeError, ValueError, OverflowError):
                start = {'nanos': None}
            else:
                start = call.time_start(parsed, name)
            starts.append((name, start))
        choice = call('observe_direct_schema', {**observed, 'starts': starts})
    return choice['column']


def categorical_advice(schema: Any, call: Callable[..., Any]) -> None:
    for name, table in schema.tables.items():
        frame = table.data
        if frame is None or len(frame) == 0:
            continue
        for column, meta in table.columns.items():
            if meta.stype_declared or meta.stype != 'categorical':
                continue
            if meta.dtype != 'string' or column not in frame:
                continue
            distinct = frame[column].nunique(dropna=True)
            present = int(frame[column].count())
            call(
                'observe_direct_schema',
                {
                    'kind': 'categorical_advice',
                    'table': name,
                    'column': column,
                    'distinct': int(distinct),
                    'present': present,
                },
            )


def date_advice(
    frame: pd.DataFrame,
    entity: str,
    columns: list[dict[str, Any]],
    key: str | None,
    selected_time: str | None,
    call: DirectOperations,
) -> None:
    from sdm_client.core import dates

    candidates = call(
        'observe_direct_schema',
        {
            'kind': 'date_advice_candidates',
            'columns': columns,
            'primary_key': key,
            'time_column': selected_time,
            'empty': frame.empty,
        },
    )
    for name in candidates['names']:
        present = frame[name].dropna()
        if not dates.looks_unambiguous(present):
            continue
        # The conversion's order read is separate from the shape read.
        order = dates.day_comes_first(present)
        for utc in candidates['utc_readings']:
            candidate = dates.Conversion(utc=utc, dayfirst=order)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    converted = pd.to_datetime(
                        present, errors='coerce', **candidate.arguments
                    )
            except (ValueError, TypeError, OverflowError):
                continue
            if not pd.api.types.is_datetime64_any_dtype(converted):
                continue
            if converted.isna().any():
                continue
            years = converted.dt.year
            minimum_year = int(years.min())
            minimum = call(
                'observe_direct_schema',
                {
                    'kind': 'date_minimum_year',
                    'year': minimum_year,
                },
            )
            if not minimum['accepted']:
                continue

            # Do not reach max() when the lower bound already refused it.
            def date_arguments(
                fields: dict[str, Any],
                conversion: dates.Conversion = candidate,
            ) -> str:
                return str(conversion)

            with errors.holding(direct_date_arguments=date_arguments):
                maximum = call(
                    'observe_direct_schema',
                    {
                        'kind': 'date_maximum_year',
                        'table': entity,
                        'column': name,
                        'minimum': minimum_year,
                        'year': int(years.max()),
                    },
                )
            if maximum['accepted']:
                return


def repeated_key(
    frame: pd.DataFrame,
    entity: str,
    key: str,
    columns: list[dict[str, Any]],
    call: DirectOperations,
) -> int:
    from sdm_client.schema import _key_quality

    if frame.empty:
        return 0
    quality = _key_quality(frame, (key,))
    if quality.duplicates:
        # Complete the composite duplicate reads before any null reads,
        # matching the two native passes in Table._adopt_key.
        distinguishing = [
            column['name']
            for column in columns
            if column['name'] != key
            and _key_quality(frame, (key, column['name'])).duplicates == 0
        ]
        candidates = call(
            'observe_direct_schema',
            {
                'kind': 'key_companions',
                'columns': columns,
                'distinguishing': distinguishing,
            },
        )['names']
        usable = [
            name
            for name in candidates
            if name not in frame.columns or not bool(frame[name].isna().any())
        ]
        companions = ', '.join(repr(name) for name in usable)
        with errors.holding(
            direct_key_example=lambda fields: quality.example,
            direct_key_companions=lambda fields: companions,
        ):
            call(
                'observe_direct_schema',
                {
                    'kind': 'key_repeated',
                    'table': entity,
                    'column': key,
                    'duplicates': quality.duplicates,
                    'distinguishing': distinguishing,
                    'usable': usable,
                },
            )
    return int(quality.duplicates)


def preparation_requires(frame: pd.DataFrame) -> bool:
    # Retain the landed integer/Boolean constructor batch. Extended families
    # reach additional native schema/time/advice observations before scoring.
    return any(
        not isinstance(dtype, np.dtype)
        or not dtype.isnative
        or dtype.kind not in 'biu'
        for dtype in frame.dtypes
    )


def requires(frame: pd.DataFrame) -> bool:
    import pyarrow as pa

    for at, dtype in enumerate(frame.dtypes):
        if not DirectOperations.passive_owner(frame.iloc[:, at]):
            return True
        if type(dtype) is pd.StringDtype:
            if not DirectOperations.passive(frame.iloc[:, at]):
                return True
            continue
        if isinstance(dtype, np.dtype):
            if dtype.isnative and (
                dtype.kind in 'biuM'
                or dtype.kind == 'f'
                and 4 <= dtype.itemsize <= 8
            ):
                continue
        elif type(dtype) in (
            pd.DatetimeTZDtype,
            pd.Int8Dtype,
            pd.Int16Dtype,
            pd.Int32Dtype,
            pd.Int64Dtype,
            pd.UInt8Dtype,
            pd.UInt16Dtype,
            pd.UInt32Dtype,
            pd.UInt64Dtype,
            pd.Float32Dtype,
            pd.Float64Dtype,
            pd.BooleanDtype,
        ):
            continue
        elif type(dtype) is pd.ArrowDtype and (
            pa.types.is_integer(dtype.pyarrow_dtype)
            or pa.types.is_boolean(dtype.pyarrow_dtype)
            or pa.types.is_decimal(dtype.pyarrow_dtype)
            or pa.types.is_timestamp(dtype.pyarrow_dtype)
            or dtype.pyarrow_dtype in (pa.float32(), pa.float64())
        ):
            continue
        return True
    return False


def prepare(
    context: pd.DataFrame,
    scoring: pd.DataFrame,
    shape: dict[str, Any],
    schema: Any,
    call: DirectOperations,
    build: Callable[..., Any],
    flags: Callable[..., Any],
) -> Any:
    entity = shape['entity_table']
    target = shape['target']
    call(
        'check_direct_preparation',
        {
            'kind': 'population',
            'shape': shape,
            'known': int(context[target].notna().sum()),
        },
    )
    if schema is None:
        call(
            'check_direct_preparation',
            {
                'kind': 'names',
                'table': entity,
                'columns': list(context.columns),
            },
        )
        for name in context.columns:
            direct_cells.admit(context[name], 'context', 'string', call)
        columns = []
        for name in context.columns:
            series = context[name]
            finding = direct_dtypes.observe(series, call)
            observed = call(
                'observe_direct_schema',
                {
                    'kind': 'column',
                    'name': name,
                    'finding': finding,
                    'distinct': int(series.nunique(dropna=True))
                    if finding['dtype'] == 'string'
                    else None,
                },
            )
            columns.append(observed['column'])
        call(
            'observe_direct_schema',
            {'kind': 'table_names', 'table': entity, 'columns': []},
        )
        for name in context.columns:
            call(
                'observe_direct_schema',
                {'kind': 'table_names', 'table': entity, 'columns': [name]},
            )
            direct_cells.admit(context[name], 'context', 'string', call)
        candidates = call(
            'observe_direct_schema',
            {
                'kind': 'key_candidates',
                'columns': columns,
                'rows': len(context),
            },
        )
        key = None
        for name in candidates['names']:
            distinct = int(context[name].nunique(dropna=True))
            counted = call(
                'observe_direct_schema',
                {
                    'kind': 'key_count',
                    'rows': len(context),
                    'distinct': distinct,
                },
            )
            if counted['accepted']:
                key = name
                break
        duplicates = 0
        if key is not None:
            if bool(context[key].isna().any()):
                call(
                    'check_direct_preparation',
                    {
                        'kind': 'missing_key',
                        'table': entity,
                        'column': key,
                        'count': int(context[key].isna().sum()),
                    },
                )
            elif not context.empty:
                from sdm_client.schema import _key_quality

                quality = _key_quality(context, (key,))
                empty = (
                    [key]
                    if quality.nulls
                    and key in context.columns
                    and bool(context[key].isna().any())
                    else []
                )
                with errors.holding(
                    direct_key_example=lambda fields: quality.example,
                ):
                    call(
                        'observe_direct_schema',
                        {
                            'kind': 'key_validity',
                            'table': entity,
                            'column': key,
                            'nulls': quality.nulls,
                            'duplicates': quality.duplicates,
                            'rows': len(context),
                            'empty': empty,
                        },
                    )
            duplicates = repeated_key(context, entity, key, columns, call)
        selected_time = time_column(context, columns, key, call)
        described = call(
            'observe_direct_schema',
            {
                'kind': 'assemble',
                'table': entity,
                'columns': columns,
                'primary_key': key,
                'duplicates': duplicates,
                'time_column': selected_time,
            },
        )
        schema = build(described['schema'], context, described['flags'])
        date_advice(context, entity, columns, key, selected_time, call)
        categorical_advice(schema, call)
    table = schema[entity]
    for name in scoring.columns:
        column = table.columns.get(name)
        if column is None or column.untyped and not column.stype_declared:
            continue
        finding = direct_dtypes.observe(scoring[name], call)
        call(
            'check_direct_preparation',
            {
                'kind': 'scored_column',
                'table': entity,
                'schema': spec.held_schema(schema),
                'flags': flags(schema, entity),
                'name': name,
                'finding': finding,
            },
        )
    current_shape = {
        **shape,
        'context': {'rows': len(context), 'columns': list(context.columns)},
        'predict': {'rows': len(scoring), 'columns': list(scoring.columns)},
    }
    call(
        'check_direct_preparation',
        {
            'kind': 'scored_complete',
            'shape': current_shape,
            'schema': spec.held_schema(schema),
        },
    )
    present = context[target].dropna()
    distinct = int(present.nunique()) if not present.empty else 0
    call(
        'check_direct_preparation',
        {
            'kind': 'constancy',
            'target': target,
            'task': shape['task'],
            'known': len(present),
            'distinct': distinct,
        },
    )
    return schema
