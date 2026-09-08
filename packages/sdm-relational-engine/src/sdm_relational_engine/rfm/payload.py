# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import datetime
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
from predictive_query.schema.task import TaskType
from predictive_query.schema.typing import Stype
from sdm_api.generated.tfm_api import (
    TFM_MODEL_KUMO_RELATIONAL,
    TFM_OUTPUT_FIELD_EMBEDDINGS,
    TFM_OUTPUT_FIELD_EXPLANATION,
    TFM_OUTPUT_FIELD_PREDICTION,
    TFM_OUTPUT_FIELD_PROBABILITIES,
    TFM_OUTPUT_FIELD_QUANTILES,
    TFM_OUTPUT_FIELD_RANKINGS,
)

from sdm_relational_engine.api.rfm import RFMPredictRequest
from sdm_relational_engine.api.rfm.context import REV_REL, Context, EdgeLayout
from sdm_relational_engine.api.rfm.inference import (
    ClassificationInferenceConfig,
    InferenceConfig,
    RegressionInferenceConfig,
)
from sdm_relational_engine.runmode import RunMode

INSTANCE_ID = 'instance_id'
SYNTHETIC_NODE_ID = '__node_id'
ENTITY_REFERENCE_PREFIX = '__kumo_entity_ref'
ANCHOR_TIME_PREFIX = '__kumo_anchor_time'

JSON_SAFE_INT_MAX = 9007199254740991
JSON_SAFE_INT_MIN = -9007199254740991

_NON_FINITE_MSG = (
    'encountered a non-finite value ({value}); NIM requests '
    'must contain finite numbers or nulls'
)

# What `pandas.api.types.infer_dtype` reports for the contents of an `object`
# column, mapped to the wire dtype that represents those values. A content kind
# that is absent here (strings, bytes, mixed types, an empty column) travels as
# `string`, whose cells are stringified to match.
_OBJECT_CONTENT_DTYPES = {
    'integer': 'int64',
    'floating': 'float64',
    'mixed-integer-float': 'float64',
    'decimal': 'float64',
    'boolean': 'bool',
    'date': 'timestamp[us]',
    'datetime': 'timestamp[us]',
    'datetime64': 'timestamp[us]',
    'timedelta': 'int64',
    'timedelta64': 'int64',
}


@dataclass(frozen=True)
class PayloadTables:
    r"""The frames a request is built from, plus the wire dtype of every column.

    ``instance_column_dtypes`` and ``related_table_column_dtypes`` are the
    single authority for how a column travels: the schema declares them and the
    cells are serialized under them, so the two halves of a request cannot
    disagree. They are derived from the *whole* of each logical table, not from
    the context or predict half alone, so that the two splits of one table
    always declare the same dtype.
    """

    context_instance_table: pd.DataFrame
    predict_instance_table: pd.DataFrame
    related_table_frames: dict[str, pd.DataFrame]
    context_related_tables: dict[str, pd.DataFrame]
    predict_related_tables: dict[str, pd.DataFrame]
    related_table_primary_keys: dict[str, str | list[str]]
    related_table_stype_overrides: dict[str, dict[str, Stype]]
    entity_reference_columns: dict[str, str]
    anchor_time_column: str | None
    relationships: list[dict[str, Any]]
    instance_column_dtypes: dict[str, str]
    related_table_column_dtypes: dict[str, dict[str, str]]


def predict_request_to_json(
    request: RFMPredictRequest,
    *,
    explain: bool = False,
) -> dict[str, Any]:
    output_fields = _output_fields(
        request.context.task_type,
        return_embeddings=request.return_embeddings,
        explain=explain,
        quantile_output=(
            isinstance(request.inference_config, RegressionInferenceConfig)
            and request.inference_config.output_type == 'quantiles'
        ),
    )
    payload = _base_payload(
        context=request.context,
        run_mode=request.run_mode,
        use_prediction_time=request.use_prediction_time,
        inference_config=request.inference_config,
        output={'fields': output_fields},
        metadata={
            'source_query': request.query,
        },
    )
    return payload


MAX_TABLE_ROWS = 10_000

_SESSION_CREATE_SECTIONS = ('model', 'task', 'schema', 'context', 'metadata')
_SESSION_PREDICT_SECTIONS = ('predict', 'output', 'inference', 'metadata')


def session_create_payload(payload: dict[str, Any]) -> dict[str, Any]:
    r"""The context-only body for ``POST /v1/sessions``.

    A session pins ``model`` + ``task`` + ``schema`` + ``context`` once; this
    keeps exactly those sections (plus ``metadata``) from a full prediction
    payload and drops the per-call ``predict`` / ``output`` / ``inference``.
    """
    return {
        key: payload[key] for key in _SESSION_CREATE_SECTIONS if key in payload
    }


def session_predict_payload(payload: dict[str, Any]) -> dict[str, Any]:
    r"""The per-call body for ``POST /v1/sessions/{id}/predictions``.

    The session already holds ``model`` / ``task`` / ``schema`` / ``context``,
    so only ``predict`` + ``output`` + ``inference`` (plus ``metadata``) travel
    with each prediction.
    """
    return {
        key: payload[key] for key in _SESSION_PREDICT_SECTIONS if key in payload
    }


def payload_size_bytes(payload: dict[str, Any]) -> int:
    return len(
        json.dumps(
            payload,
            allow_nan=False,
            separators=(',', ':'),
        ).encode('utf-8')
    )


def validate_payload_table_rows(
    payload: dict[str, Any],
    *,
    batch_index: int,
    limit: int = MAX_TABLE_ROWS,
) -> None:
    r"""Validate instance-table row limits in serialized requests.

    The KumoRelational NIM row-caps only context/predict instance tables. Related
    tables are bounded by the overall payload-size limit instead.
    """
    for section_name in ('context', 'predict'):
        table = payload[section_name]['instance_table']
        num_rows = len(table['rows'])
        if num_rows > limit:
            path = f'{section_name}.instance_table'
            raise ValueError(
                f"Request batch {batch_index} table '{path}' contains "
                f'{num_rows:,} rows, exceeding the {limit:,}-row limit'
            )


def _payload_tables_with_names(
    payload: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    r"""Yield ``(table_name, table)`` for every serialized table in a request.

    Related tables are keyed by the name the caller gave them; the entity rows
    travel as ``instance_table``, which is resolved back to the entity table so
    every name reported is one the caller can look up on their graph.
    """
    task = payload.get('task')
    entity_names = (
        task.get('entity_table_names') if isinstance(task, dict) else None
    )
    instance_name = (
        entity_names[0]
        if isinstance(entity_names, list) and entity_names
        else 'instance_table'
    )
    for section_name in ('context', 'predict'):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        instance = section.get('instance_table')
        if isinstance(instance, dict):
            yield str(instance_name), instance
        related = section.get('related_tables')
        if isinstance(related, dict):
            for table_name, table in related.items():
                if isinstance(table, dict):
                    yield table_name, table


def _string_values(cell: Any) -> tuple[str, ...]:
    r"""The string values a serialized cell contributes to a column's
    cardinality.

    A text or multicategorical cell travels as a list of tokens rather than a
    bare string, and the NIM counts those tokens too, so both shapes have to be
    unwrapped to arrive at the count it reports.
    """
    if isinstance(cell, str):
        return (cell,)
    if isinstance(cell, (list, tuple)):
        return tuple(value for value in cell if isinstance(value, str))
    return ()


def high_cardinality_columns(
    payload: dict[str, Any],
    limit: int,
) -> list[tuple[str, str, int]]:
    r"""Return ``(table, column, cardinality)`` for string columns holding more
    than ``limit`` distinct values, most offending first.

    Mirrors the NIM's own cardinality guard, which counts string cells only, so
    a rejection the server reports as a bare number can be traced back to the
    columns that caused it. The same table appears in both the context and the
    predict section, so counts are reduced to the larger of the two.
    """
    counts: dict[tuple[str, str], int] = {}
    for table_name, table in _payload_tables_with_names(payload):
        columns = table.get('columns')
        rows = table.get('rows')
        if not isinstance(columns, list) or not isinstance(rows, list):
            continue
        for index, column in enumerate(columns):
            distinct: set[str] = set()
            for row in rows:
                if not isinstance(row, list) or index >= len(row):
                    continue
                distinct.update(_string_values(row[index]))
            if len(distinct) > limit:
                key = (table_name, str(column))
                counts[key] = max(counts.get(key, 0), len(distinct))
    return sorted(
        ((table, column, count) for (table, column), count in counts.items()),
        key=lambda entry: entry[2],
        reverse=True,
    )


def context_size_stats(context: Context) -> str:
    table_stats = []
    for table_name, table in context.subgraph.table_dict.items():
        size = int(table.df.memory_usage(deep=True).sum())
        table_stats.append((table_name, sum(table.num_sampled_nodes), size))
    table_stats = sorted(table_stats, key=lambda item: item[2], reverse=True)
    num_nodes = sum(num_nodes for _, num_nodes, _ in table_stats)
    num_edges = sum(
        link.num_edges for link in context.subgraph.link_dict.values()
    )
    top = table_stats[:5]
    top_repr = ', '.join(
        f'{name}: {size / (1024 * 1024):.2f}MB' for name, _, size in top
    )
    return (
        f'Current context contains {num_nodes:,} nodes and '
        f'{num_edges:,} edges across {len(table_stats)} tables. '
        f'Top-{len(top)} tables contributing most to the context size: '
        f'{top_repr}'
    )


def _base_payload(
    *,
    context: Context,
    run_mode: RunMode,
    use_prediction_time: bool,
    inference_config: InferenceConfig | None,
    output: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    tables = _payload_tables(context)
    metadata = {
        key: value
        for key, value in metadata.items()
        if value is not None and value != ''
    }
    metadata['num_context_examples'] = context.num_train
    metadata['num_prediction_examples'] = context.num_test

    instance_dtypes = tables.instance_column_dtypes
    return {
        'model': TFM_MODEL_KUMO_RELATIONAL,
        'task': _task_spec(context, tables),
        'schema': _schema_spec(context, tables),
        'context': {
            'instance_table': _dataframe_table(
                _instance_payload_dataframe(
                    tables.context_instance_table,
                ),
                instance_dtypes,
                'instance_table',
            ),
            'related_tables': {
                table_name: _dataframe_table(
                    df,
                    tables.related_table_column_dtypes[table_name],
                    table_name,
                )
                for table_name, df in tables.context_related_tables.items()
            },
        },
        'predict': {
            'instance_table': _dataframe_table(
                _instance_payload_dataframe(
                    tables.predict_instance_table,
                ),
                instance_dtypes,
                'instance_table',
            ),
            'related_tables': {
                table_name: _dataframe_table(
                    df,
                    tables.related_table_column_dtypes[table_name],
                    table_name,
                )
                for table_name, df in tables.predict_related_tables.items()
            },
        },
        'output': output,
        'inference': {
            'run_mode': RunMode(run_mode).value,
            'use_prediction_time': use_prediction_time,
            'inference_config': _inference_config_to_json(inference_config),
        },
        'metadata': metadata,
    }


def _payload_tables(context: Context) -> PayloadTables:
    instance_df, entity_reference_columns, anchor_time_column = (
        _instance_dataframe(context)
    )
    related_table_frames: dict[str, pd.DataFrame] = {}
    related_table_primary_keys: dict[str, str | list[str]] = {}
    related_table_stype_overrides: dict[str, dict[str, Stype]] = {}
    related_table_key_columns: dict[str, str] = {}

    for table_name, table in context.subgraph.table_dict.items():
        df, key_column = _occurrence_dataframe(table)
        stype_overrides = {INSTANCE_ID: Stype.ID, **table.stype_dict}
        if key_column != table.primary_key:
            stype_overrides[key_column] = Stype.ID

        related_table_frames[table_name] = df
        related_table_key_columns[table_name] = key_column
        related_table_primary_keys[table_name] = [INSTANCE_ID, key_column]
        related_table_stype_overrides[table_name] = stype_overrides

    relationships = _populate_relationship_columns(
        context,
        related_table_frames,
        related_table_key_columns,
        related_table_stype_overrides,
        entity_reference_columns,
    )

    context_related_tables: dict[str, pd.DataFrame] = {}
    predict_related_tables: dict[str, pd.DataFrame] = {}
    for table_name, df in related_table_frames.items():
        instance_ids = df[INSTANCE_ID].to_numpy()
        context_mask = instance_ids < context.num_train
        predict_mask = instance_ids >= context.num_train
        context_related_tables[table_name] = df.loc[context_mask].reset_index(
            drop=True
        )
        predict_related_tables[table_name] = df.loc[predict_mask].reset_index(
            drop=True
        )

    instance_column_dtypes = _column_dtypes(instance_df)
    instance_column_dtypes[context.y_train.name or 'TARGET'] = _target_dtype(
        context.task_type,
        context.y_train,
    )

    return PayloadTables(
        context_instance_table=instance_df.iloc[
            : context.num_train
        ].reset_index(drop=True),
        predict_instance_table=instance_df.iloc[context.num_train :]
        .reset_index(drop=True)
        .drop(
            columns=[context.y_train.name or 'TARGET'],
            errors='ignore',
        ),
        related_table_frames=related_table_frames,
        context_related_tables=context_related_tables,
        predict_related_tables=predict_related_tables,
        related_table_primary_keys=related_table_primary_keys,
        related_table_stype_overrides=related_table_stype_overrides,
        entity_reference_columns=entity_reference_columns,
        anchor_time_column=anchor_time_column,
        relationships=relationships,
        instance_column_dtypes=instance_column_dtypes,
        related_table_column_dtypes={
            table_name: _column_dtypes(df)
            for table_name, df in related_table_frames.items()
        },
    )


def _column_dtypes(df: pd.DataFrame) -> dict[str, str]:
    return {
        str(column_name): _dtype_name(df[column_name])
        for column_name in df.columns
    }


def _instance_dataframe(
    context: Context,
) -> tuple[pd.DataFrame, dict[str, str], str | None]:
    # Request-local transport keys are independent of user entity values.
    df = pd.DataFrame({INSTANCE_ID: range(context.subgraph.batch_size)})
    occupied = {INSTANCE_ID, context.y_train.name or 'TARGET'}
    if context.task_table is not None:
        occupied.update(context.task_table.df.columns)

    entity_names = context.entity_table_names
    entity_reference_columns: dict[str, str] = {}
    for i, table_name in enumerate(entity_names):
        values = _entity_values(context, table_name)
        if values is None:
            raise ValueError(
                f'Entity table {table_name!r} does not expose a primary key '
                'for Universal payload materialization.'
            )
        column_name = _unique_internal_column(
            occupied,
            f'{ENTITY_REFERENCE_PREFIX}_{i}',
        )
        occupied.add(column_name)
        entity_reference_columns[table_name] = column_name
        df[column_name] = values

    anchor_values = np.asarray(context.subgraph.anchor_time, dtype=np.int64)
    anchor_time_column: str | None = None
    if len(anchor_values) and not np.all(
        anchor_values == pd.Timestamp.min.value
    ):
        anchor_time_column = _unique_internal_column(
            occupied,
            ANCHOR_TIME_PREFIX,
        )
        occupied.add(anchor_time_column)
        df[anchor_time_column] = pd.to_datetime(anchor_values)

    target_name = context.y_train.name or 'TARGET'
    df[target_name] = None
    train_targets = [
        _target_json_value(context.task_type, value)
        for value in context.y_train.tolist()
    ]
    if train_targets:
        df.iloc[: context.num_train, df.columns.get_loc(target_name)] = (
            _object_array(train_targets)
        )
    if context.y_test is not None:
        test_targets = [
            _target_json_value(context.task_type, value)
            for value in context.y_test.tolist()
        ]
        if test_targets:
            df.iloc[
                context.num_train : context.num_train + len(test_targets),
                df.columns.get_loc(target_name),
            ] = _object_array(test_targets)
    elif context.num_test > 0:
        df.loc[context.num_train :, target_name] = None

    if context.task_table is not None:
        for column_name in context.task_table.df.columns:
            df[column_name] = context.task_table.df[column_name].tolist()

    return df, entity_reference_columns, anchor_time_column


def _object_array(values: list[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    array[:] = values
    return array


def _entity_values(context: Context, table_name: str) -> list[Any] | None:
    table = context.subgraph.table_dict.get(table_name)
    if table is None or table.primary_key is None:
        return None
    if table.primary_key not in table.df:
        return None

    values: list[Any] = []
    batch = np.asarray(table.batch)
    df = table.df.reset_index(drop=True)
    # The column, not the row: `df.iloc[i][key]` collapses a row to one dtype
    # and returns an integer key as a float.
    key_values = df[table.primary_key]
    row = np.asarray(table.row) if table.row is not None else None
    # One pass for every instance's first occurrence, rather than a full scan
    # of `batch` per instance (`np.unique` returns the first index per value
    # whether or not `batch` is sorted).
    first_occurrence: dict[int, int] = {}
    if batch.size > 0:
        unique, first = np.unique(batch, return_index=True)
        first_occurrence = dict(zip(unique.tolist(), first.tolist()))
    for instance_id in range(context.subgraph.batch_size):
        position = first_occurrence.get(instance_id)
        if position is None:
            values.append(None)
            continue
        row_index = position if row is None else int(row[position])
        if row_index >= len(key_values):
            values.append(None)
            continue
        values.append(_json_value(key_values.iloc[row_index]))
    return values


def _occurrence_dataframe(table: Any) -> tuple[pd.DataFrame, str]:
    df = table.df.reset_index(drop=True)
    batch = np.asarray(table.batch)
    if table.row is not None:
        row = np.asarray(table.row)
        if len(row) == len(batch) and len(row) > 0:
            if int(row.max()) >= len(df) or int(row.min()) < 0:
                raise ValueError(
                    'Sampled RFM context references rows outside the table '
                    'frame.'
                )
            df = df.iloc[row].reset_index(drop=True)
        elif len(row) == 0:
            df = df.iloc[0:0].copy()
        elif len(df) != len(batch):
            raise ValueError(
                'Sampled RFM context has incompatible row and batch lengths.'
            )
    elif len(df) != len(batch):
        raise ValueError(
            'Sampled RFM context cannot be converted to table-oriented JSON '
            'because rows are not aligned with batch instances.'
        )
    if len(df) != len(batch):
        raise ValueError(
            'Sampled RFM context has incompatible table and batch lengths.'
        )

    df = df.copy(deep=False)
    df.insert(0, INSTANCE_ID, batch)
    if table.primary_key is None or table.primary_key not in df:
        key_column = _unique_internal_column(
            set(df.columns),
            SYNTHETIC_NODE_ID,
        )
        df.insert(1, key_column, np.arange(len(df), dtype=np.int64))
    else:
        key_column = table.primary_key
    return df, key_column


def _populate_relationship_columns(
    context: Context,
    related_table_frames: dict[str, pd.DataFrame],
    related_table_key_columns: dict[str, str],
    related_table_stype_overrides: dict[str, dict[str, Stype]],
    entity_reference_columns: dict[str, str],
) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []

    for table_name in context.entity_table_names:
        source_column = entity_reference_columns[table_name]
        target_column = related_table_key_columns.get(table_name)
        if target_column is None:
            raise ValueError(
                f'Entity table {table_name!r} is missing from the sampled '
                'subgraph.'
            )
        table = context.subgraph.table_dict[table_name]
        if table.primary_key is None or target_column != table.primary_key:
            raise ValueError(
                f'Entity table {table_name!r} requires a declared scalar '
                'primary key.'
            )
        relationships.append(
            {
                'source_columns': [source_column],
                'target_table': table_name,
                'target_columns': [target_column],
            }
        )

    processed_edges: set[tuple[str, str, str]] = set()
    for edge_type, link in context.subgraph.link_dict.items():
        logical_edge_type = _logical_forward_edge_type(edge_type)
        if logical_edge_type in processed_edges:
            continue
        src_table, fkey, dst_table = logical_edge_type
        if src_table not in related_table_frames:
            continue
        if dst_table not in related_table_frames:
            continue

        src_df = related_table_frames[src_table]
        dst_df = related_table_frames[dst_table]
        dst_key = related_table_key_columns[dst_table]
        edge_pairs = _logical_forward_edge_pairs(context, edge_type, link)
        if not edge_pairs:
            continue

        # Gather from the key column in one take: `dst_df.iloc[i][dst_key]`
        # materializes a whole row Series per edge, which dominates request
        # assembly on any non-trivial subgraph. Later pairs still overwrite
        # earlier ones, as the equivalent loop did.
        fkey_values: list[Any] = [None] * len(src_df)
        pairs = np.asarray(edge_pairs, dtype=np.int64).reshape(-1, 2)
        in_range = (pairs[:, 0] < len(src_df)) & (pairs[:, 1] < len(dst_df))
        src_positions = pairs[in_range, 0].tolist()
        if not src_positions:
            continue
        gathered = dst_df[dst_key].iloc[pairs[in_range, 1]]
        for position, value in zip(src_positions, gathered):
            fkey_values[position] = _json_value(value)
        if all(value is None for value in fkey_values):
            continue

        src_df[fkey] = fkey_values
        related_table_stype_overrides[src_table][fkey] = Stype.ID
        relationships.append(
            {
                'source_table': src_table,
                'source_columns': [fkey],
                'target_table': dst_table,
                'target_columns': [dst_key],
            }
        )
        processed_edges.add(logical_edge_type)

    return relationships


def _logical_forward_edge_type(
    edge_type: tuple[str, str, str],
) -> tuple[str, str, str]:
    src_table, fkey, dst_table = edge_type
    if fkey.startswith(REV_REL):
        return (dst_table, fkey[len(REV_REL) :], src_table)
    return edge_type


def _logical_forward_edge_pairs(
    context: Context,
    edge_type: tuple[str, str, str],
    link: Any,
) -> list[tuple[int, int]]:
    pairs = _edge_pairs(context, edge_type, link)
    if edge_type[1].startswith(REV_REL):
        return [(dst, src) for src, dst in pairs]
    return pairs


def _edge_pairs(
    context: Context,
    edge_type: tuple[str, str, str],
    link: Any,
) -> list[tuple[int, int]]:
    if link.layout == EdgeLayout.REV:
        rev_edge_type = context.subgraph.rev_edge_type(edge_type)
        rev_link = context.subgraph.link_dict.get(rev_edge_type)
        if rev_link is None:
            return []
        return [
            (dst, src)
            for src, dst in _edge_pairs(context, rev_edge_type, rev_link)
        ]

    num_edges = int(link.num_edges)
    if num_edges == 0:
        return []

    if link.layout == EdgeLayout.CSC:
        if link.col is None:
            return []
        target_count = context.subgraph.table_dict[edge_type[2]].num_rows
        colptr = np.asarray(link.col)
        if len(colptr) != target_count + 1:
            return []
        row = (
            np.arange(num_edges, dtype=np.int64)
            if link.row is None
            else np.asarray(link.row)
        )
        pairs: list[tuple[int, int]] = []
        for dst_index in range(target_count):
            start = int(colptr[dst_index])
            end = int(colptr[dst_index + 1])
            pairs.extend(
                (int(src_index), dst_index) for src_index in row[start:end]
            )
        return pairs

    row = (
        np.arange(num_edges, dtype=np.int64)
        if link.row is None
        else np.asarray(link.row)
    )
    col = (
        np.arange(num_edges, dtype=np.int64)
        if link.col is None
        else np.asarray(link.col)
    )
    return [(int(src), int(dst)) for src, dst in zip(row, col)]


def _task_spec(context: Context, tables: PayloadTables) -> dict[str, Any]:
    target_name = context.y_train.name or 'TARGET'
    target_dtype = tables.instance_column_dtypes[target_name]
    spec: dict[str, Any] = {
        'kind': TaskType(context.task_type).value,
        'target': {
            'column_name': target_name,
            'dtype': target_dtype,
        },
        'entity_table_names': list(context.entity_table_names),
    }
    if tables.anchor_time_column is not None:
        spec['anchor_time_column'] = tables.anchor_time_column
    if TaskType(context.task_type) == TaskType.BINARY_CLASSIFICATION:
        spec['target']['classes'] = ['false', 'true']
        spec['target']['positive_class'] = 'true'
    elif TaskType(context.task_type) == TaskType.MULTICLASS_CLASSIFICATION:
        spec['target']['classes'] = [
            _class_label(value, target_dtype)
            for value in pd.unique(context.y_train)
        ]
    if context.top_k is not None:
        spec['top_k'] = context.top_k
    if context.step_size is not None:
        spec['step_size'] = context.step_size
    if context.num_forecasts != 1:
        spec['num_forecasts'] = context.num_forecasts
    return spec


def _schema_spec(
    context: Context,
    tables: PayloadTables,
) -> dict[str, Any]:
    spec = {
        'instance_table': _schema_for_dataframe(
            tables.instance_column_dtypes,
            stype_overrides={
                INSTANCE_ID: Stype.ID,
                **dict.fromkeys(
                    tables.entity_reference_columns.values(), Stype.ID
                ),
                **(
                    {tables.anchor_time_column: Stype.timestamp}
                    if tables.anchor_time_column is not None
                    else {}
                ),
                context.y_train.name or 'TARGET': _target_stype(
                    context.task_type
                ),
            },
            primary_key=INSTANCE_ID,
        ),
        'related_tables': {
            table_name: _schema_for_dataframe(
                dtypes,
                stype_overrides=tables.related_table_stype_overrides[
                    table_name
                ],
                primary_key=tables.related_table_primary_keys[table_name],
            )
            for table_name, dtypes in tables.related_table_column_dtypes.items()
        },
        'relationships': tables.relationships,
    }
    if tables.anchor_time_column is not None:
        anchor_schema = spec['instance_table']['columns'][
            tables.anchor_time_column
        ]
        anchor_schema['nullable'] = False
    return spec


def _schema_for_dataframe(
    dtypes: dict[str, str],
    *,
    stype_overrides: dict[str, Stype],
    primary_key: str | list[str],
) -> dict[str, Any]:
    r"""Declares one table's columns from the wire dtypes its cells are written
    under, so the schema cannot describe a different encoding than the one the
    rows carry.
    """
    primary_keys = (
        {primary_key} if isinstance(primary_key, str) else set(primary_key)
    )
    return {
        'columns': {
            column_name: {
                'dtype': dtype,
                **(
                    {'stype': _stype_name(stype_overrides[column_name])}
                    if column_name in stype_overrides
                    else {}
                ),
                **({'nullable': False} if column_name in primary_keys else {}),
            }
            for column_name, dtype in dtypes.items()
        },
        'primary_key': primary_key,
    }


def _instance_payload_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=False)


def _unique_internal_column(occupied: set[str], prefix: str) -> str:
    if prefix not in occupied:
        return prefix
    suffix = 1
    while f'{prefix}_{suffix}' in occupied:
        suffix += 1
    return f'{prefix}_{suffix}'


def _dataframe_table(
    df: pd.DataFrame,
    dtypes: dict[str, str],
    table_name: str | None = None,
) -> dict[str, Any]:
    r"""Serializes ``df`` under the wire dtypes its schema declares.

    ``dtypes`` is the same mapping :func:`_schema_for_dataframe` declares from,
    which is what keeps a cell from contradicting the dtype written next to it.
    """
    columns = df.columns.tolist()
    cell_dtypes = [dtypes[str(column)] for column in columns]
    rows = []
    for row_index, row in enumerate(df.itertuples(index=False, name=None)):
        cells = []
        for column_index, value in enumerate(row):
            try:
                cells.append(_cell_json_value(value, cell_dtypes[column_index]))
            except ValueError as error:
                column = columns[column_index]
                qualified = (
                    f"'{table_name}.{column}'"
                    if table_name is not None
                    else f"'{column}'"
                )
                raise ValueError(
                    f'Column {qualified} row {row_index}: {error}'
                ) from error
        rows.append(cells)
    return {'format': 'arrays', 'columns': columns, 'rows': rows}


def _output_fields(
    task_type: TaskType,
    *,
    return_embeddings: bool,
    explain: bool,
    quantile_output: bool = False,
) -> list[str]:
    if TaskType(task_type).is_link_pred:
        fields = [TFM_OUTPUT_FIELD_RANKINGS]
    elif quantile_output:
        fields = [TFM_OUTPUT_FIELD_QUANTILES]
    else:
        fields = [TFM_OUTPUT_FIELD_PREDICTION]
    if TaskType(task_type).is_classification:
        fields.append(TFM_OUTPUT_FIELD_PROBABILITIES)
    if return_embeddings:
        fields.append(TFM_OUTPUT_FIELD_EMBEDDINGS)
    if explain:
        fields.append(TFM_OUTPUT_FIELD_EXPLANATION)
    return fields


def _inference_config_to_json(
    inference_config: InferenceConfig | None,
) -> dict[str, Any] | None:
    if inference_config is None:
        return None

    out: dict[str, Any] = {
        'kind': 'base',
        'num_estimators': int(inference_config.num_estimators),
        'column_shuffle': bool(inference_config.column_shuffle),
        'category_shuffle': bool(inference_config.category_shuffle),
        'hop_shuffle': bool(inference_config.hop_shuffle),
    }
    if isinstance(inference_config, ClassificationInferenceConfig):
        out['kind'] = 'classification'
        out['class_shuffle'] = bool(inference_config.class_shuffle)
    elif isinstance(inference_config, RegressionInferenceConfig):
        out['kind'] = 'regression'
        out['target_transforms'] = [
            _json_value(value) for value in inference_config.target_transforms
        ]
        out['output_type'] = inference_config.output_type
    return out


def _target_stype(task_type: TaskType) -> Stype:
    if TaskType(task_type).is_link_pred:
        return Stype.multicategorical
    if TaskType(task_type).is_classification:
        return Stype.categorical
    return Stype.numerical


def _target_dtype(task_type: TaskType, target: pd.Series) -> str:
    r"""The wire dtype the target column travels under.

    A *boolean* multiclass target travels as a string. ``bool`` is the binary
    task's target dtype, where the contract fixes the two classes as ``false``
    and ``true``; a multiclass target's classes are free-form strings, and the
    NIM has no way to map a boolean class label back onto the ones declared.
    Encoding the column as its two string labels is what a caller gets today by
    writing ``column.astype(str)`` themselves.
    """
    task_type = TaskType(task_type)
    if task_type.is_link_pred:
        return 'stringlist'
    if task_type == TaskType.BINARY_CLASSIFICATION:
        return 'bool'
    dtype = _dtype_name(target)
    if task_type == TaskType.MULTICLASS_CLASSIFICATION and dtype == 'bool':
        return 'string'
    return dtype


def _decimal_text(value: Any) -> str | None:
    r"""A finite decimal's own digits, for a value that travels as text.

    `_json_value` rounds a fractional decimal through float, so anything that
    stringifies its result loses digits the declaration promised to keep.
    Returns ``None`` when *value* is not a decimal, so a caller can fall back
    to its ordinary conversion.
    """
    if isinstance(value, Decimal) and value.is_finite():
        return format(value, 'f')
    return None


def _target_json_value(task_type: TaskType, value: Any) -> Any:
    task_type = TaskType(task_type)
    if task_type.is_link_pred:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            raise ValueError(
                'Link prediction target values must be stringlist arrays, '
                f'but got {value!r}.'
            )
        result: list[str] = []
        for item in value:
            if _is_null_like(item):
                raise ValueError(
                    'Link prediction target stringlist values must not contain '
                    f'null items, but got {value!r}.'
                )
            text = _decimal_text(item)
            if text is not None:
                result.append(text)
                continue
            item_value = _json_value(item)
            if item_value is None:
                raise ValueError(
                    'Link prediction target stringlist values must not contain '
                    f'null items, but got {value!r}.'
                )
            result.append(str(item_value))
        return result

    if (
        task_type != TaskType.BINARY_CLASSIFICATION
        and _decimal_text(value) is not None
    ):
        # Left a decimal on purpose. This value is written into the instance
        # table and serialised again under the dtype that table declares, and
        # rounding it here beats that decision to it: a target column declared
        # `string` because one value is too wide for int64 would still lose
        # the digits of every fractional value beside it.
        return value
    value = _json_value(value)
    if task_type != TaskType.BINARY_CLASSIFICATION or value is None:
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, Real) and value in (0, 1):
        return bool(value)
    raise ValueError(
        'Binary classification target values must be booleans or numeric '
        f'0/1 values, but got {value!r}.'
    )


def _dtype_name(data: pd.Series) -> str:
    r"""The wire dtype a column travels under.

    Decided from what the column *holds*, because ``object`` is a container
    rather than a type: a ``DECIMAL`` read through a warehouse driver, a column
    that has been through ``fillna``, and a frame built from records are all
    ``object`` while holding numbers, and declaring those as ``string`` either
    contradicts the cells that are sent or asks the model to treat a quantity
    as a category.

    Every value :func:`_cell_json_value` can produce for the dtype returned
    here is one the contract accepts for it, so the declaration and the cells
    are always consistent.
    """
    list_dtype = _list_dtype_name(data)
    if list_dtype is not None:
        return list_dtype

    decimal_dtype = _decimal_dtype_name(data)
    if decimal_dtype is not None:
        return decimal_dtype

    dtype = data.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return 'bool'
    if pd.api.types.is_integer_dtype(dtype):
        return 'int64'
    if pd.api.types.is_float_dtype(dtype):
        return 'float32' if dtype.itemsize <= 4 else 'float64'
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return 'timestamp[us]'
    if pd.api.types.is_timedelta64_dtype(dtype):
        # A duration is a quantity, and `Dtype.timedelta` carries
        # `Stype.numerical`; nanoseconds are how pandas already holds it.
        return 'int64'
    if pd.api.types.is_object_dtype(dtype):
        content = pd.api.types.infer_dtype(data, skipna=True)
        return _OBJECT_CONTENT_DTYPES.get(content, 'string')
    return 'string'


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_INT64_SAFE_PRECISION = 18


def _fits_int64(value: Decimal) -> bool:
    return _INT64_MIN <= value <= _INT64_MAX


def _decimal_dtype_name(data: pd.Series) -> str | None:
    r"""Name a DECIMAL column by what it holds, not by how pandas stores it.

    Neither an Arrow-backed ``decimal128`` nor an object column of
    ``Decimal`` satisfies ``is_integer_dtype``/``is_float_dtype``, so without
    this a decimal column is announced as ``string`` -- and an ID column
    announced as a string stops being an ID to the model.

    An integral decimal is only named ``int64`` when it fits: ``decimal(38,
    0)`` allows 38 digits and ``int64`` holds 19, so anything wider is named
    ``string``, which keeps every digit, rather than ``int64``, which would
    declare a width the value does not have.
    """
    pyarrow_dtype = getattr(data.dtype, 'pyarrow_dtype', None)
    if pyarrow_dtype is not None:
        if not pa.types.is_decimal(pyarrow_dtype):
            return None
        if pyarrow_dtype.scale != 0:
            return 'float64'
        if pyarrow_dtype.precision <= _INT64_SAFE_PRECISION:
            return 'int64'
        fits = all(_is_null_like(value) or _fits_int64(value) for value in data)
        return 'int64' if fits else 'string'

    if not pd.api.types.is_object_dtype(data.dtype):
        return None

    found = fractional = too_wide = False
    for value in data:
        if _is_null_like(value):
            continue
        if not isinstance(value, Decimal):
            return None  # mixed content; leave it to the caller's fallback
        found = True
        if not value.is_finite():
            continue  # encodes as null, so it constrains nothing
        if value != value.to_integral_value():
            fractional = True
        elif not _fits_int64(value):
            too_wide = True
    if not found:
        return None
    # Decided once the whole column has been read, not on whichever value
    # happened to come first. Returning early made the answer depend on row
    # order: the same two values named the column float64 one way round and
    # string the other, so reordering rows changed whether the request was
    # even legal.
    if too_wide:
        # A width no float64 holds. `string` keeps every digit, and it keeps
        # the fractional values in the same column exactly too.
        return 'string'
    if fractional:
        return 'float64'
    return 'int64'


def _list_dtype_name(data: pd.Series) -> str | None:
    if not pd.api.types.is_object_dtype(data.dtype):
        return None

    found_list = False
    found_scalar = False
    for value in data:
        if _is_null_like(value):
            continue
        if isinstance(value, np.ndarray):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            found_scalar = True
            continue
        found_list = True
        for item in value:
            if _is_null_like(item):
                raise ValueError(
                    'stringlist columns must not contain null items, '
                    f'but got {value!r}.'
                )
            item_value = _json_value(item)
            if item_value is None:
                raise ValueError(
                    'stringlist columns must not contain null items, '
                    f'but got {value!r}.'
                )

    if found_list and found_scalar:
        raise ValueError(
            'Columns with stringlist values must contain only arrays or nulls.'
        )
    if not found_list:
        return None
    return 'stringlist'


def _is_null_like(value: Any) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _stype_name(stype: Stype) -> str:
    if stype == Stype.ID:
        return 'ID'
    return Stype(stype).value


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Decimal):
        if not value.is_finite():
            return None
        if value != value.to_integral_value():
            number = float(value)
            if not math.isfinite(number):
                # Too wide for a float to hold even the integer part, so the
                # `inf` it becomes is not a rounding of the original.
                raise ValueError(
                    f'value {value!r} is too large for float64 to represent; '
                    f'send the column as strings to keep it exact'
                )
            if abs(value) > JSON_SAFE_INT_MAX:
                # Past 2**53 a float cannot hold even the integer part, so
                # what it becomes is not a rounding of the original.
                raise ValueError(
                    f'value {value!r} is too large for float64 to represent; '
                    f'send the column as strings to keep it exact'
                )
            return number
        if _fits_int64(value):
            return int(value)
        # `str` switches to an exponent once it is positive, and this branch
        # exists to keep every digit.
        return format(value, 'f')
    if isinstance(value, pd.Timestamp):
        return _timestamp_json_value(value)
    if isinstance(value, np.datetime64):
        return _timestamp_json_value(pd.Timestamp(value))
    if isinstance(value, (pd.Timedelta, datetime.timedelta)):
        # Nanoseconds: the unit pandas stores durations in, and the one wire
        # dtype `_dtype_name` gives a `timedelta` column.
        return pd.Timedelta(value).value
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode('ascii')
    if isinstance(value, np.ndarray):
        return [_json_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return None
        raise ValueError(_NON_FINITE_MSG.format(value=value))
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _class_label(value: Any, dtype: str) -> str:
    r"""The class label spelling the declared target ``dtype`` gives ``value``.

    A boolean target's classes are the contract's ``'false'`` / ``'true'``, the
    same spelling the binary-classification branch uses; ``str(np.True_)``
    would contradict both the declared dtype and the ``true`` / ``false`` row
    values travelling beside it.
    """
    if dtype == 'bool':
        return 'true' if bool(value) else 'false'
    text = _decimal_text(value)
    if text is not None:
        return text
    return str(_json_value(value))


def _cell_json_value(value: Any, dtype: str) -> Any:
    r"""Serializes a table cell under the wire ``dtype`` its column declares.

    The dtype is the authority: a cell is converted into the JSON type the
    contract defines for that dtype rather than into whatever its Python type
    happens to map to, so a column can never declare one encoding and send
    another. ``int64`` values outside the JavaScript safe-integer range travel
    as base-10 strings, which is how the contract preserves their precision.
    """
    # A decimal column is declared `string` when any of its values is too wide
    # for int64, and the point of that declaration is that every value in it
    # travels exactly. Normalising first defeats it: `_json_value` rounds a
    # fractional Decimal through float, and the string branch below then
    # stringifies the rounded float, so a column declared string to stay exact
    # quietly loses digits.
    if dtype == 'string':
        text = _decimal_text(value)
        if text is not None:
            return text
    if dtype == 'stringlist' and isinstance(value, (list, tuple)):
        # Each item, for the same reason and before the same rounding: the
        # items are stringified, and normalising first stringifies a float.
        return [
            None
            if item is None
            else (_decimal_text(item) or _wire_string(_json_value(item)))
            for item in value
        ]
    # Measured before normalisation. `_json_value` turns a whole Decimal too
    # wide for int64 into a string, and a string carries no exactness the
    # float branch below can check, so the loss happened silently.
    original = _integral_source(value)
    value = _json_value(value)
    if value is None:
        return None
    if dtype == 'stringlist':
        if not isinstance(value, list):
            raise ValueError(
                f"expected an array of values for a 'stringlist' column, but "
                f'got {value!r}'
            )
        return [None if item is None else _wire_string(item) for item in value]
    if dtype == 'string':
        return _wire_string(value)
    if dtype == 'bool':
        return bool(value)
    if dtype in ('int64', 'int32'):
        integer = int(value)
        if not _INT64_MIN <= integer <= _INT64_MAX:
            # `_dtype_name` answers int64 for uint64 too, so without this a
            # value the declared dtype cannot hold travels as though it could.
            raise ValueError(
                f'value {value!r} is outside the range {dtype} represents '
                f'({_INT64_MIN} to {_INT64_MAX}); send the column as strings'
            )
        if integer > JSON_SAFE_INT_MAX or integer < JSON_SAFE_INT_MIN:
            return str(integer)
        return integer
    if dtype in ('float64', 'float32'):
        if _exceeds_float64(value):
            raise ValueError(
                f'value {value!r} cannot be sent as {dtype} without losing '
                f'precision; send the column as strings to keep it exact'
            )
        whole = original
        if whole is not None and not _survives_float(whole):
            # The tabular encoder refuses this; a float column that silently
            # renumbers an integer is the same defect on either path. Asked as
            # a round trip rather than against the safe-integer ceiling,
            # because 2**53 itself is representable exactly and only the
            # values that do not survive may be refused.
            raise ValueError(
                f'value {value!r} cannot be sent as {dtype} without losing '
                f'precision; give the column an integer dtype, or convert it '
                f'to strings'
            )
        try:
            number = float(value)
        except OverflowError:
            # A Python int has no width, so `float()` refuses rather than
            # returning `inf` the way a wide Decimal does.
            raise ValueError(
                f'value {value!r} is too large for {dtype} to represent; '
                f'send the column as strings to keep it exact'
            ) from None
        if not math.isfinite(number) and _is_finite_input(value):
            raise ValueError(
                f'value {value!r} is too large for {dtype} to represent; '
                f'send the column as strings to keep it exact'
            )
        return number
    if dtype == 'timestamp[us]':
        if isinstance(value, str):
            return value
        return _timestamp_json_value(pd.Timestamp(value))
    if not isinstance(value, (str, int, float, bool, list)):
        raise ValueError(
            f'cannot serialize a value of type {type(value).__name__} under '
            f"wire dtype '{dtype}'; convert the column to a supported dtype"
        )
    return value


def _exceeds_float64(value: Any) -> bool:
    r"""Whether ``value`` carries precision a ``float64`` would drop.

    A numeric string is exact text and a `longdouble` is wider than a
    `float64`; `float()` silently rounds both. The tabular encoder refuses
    them, so this one must too.
    """
    if isinstance(value, str):
        try:
            exact = Decimal(value)
        except (ArithmeticError, ValueError):
            return False
        if not exact.is_finite():
            return False
        return Decimal(repr(float(exact))) != exact
    if isinstance(value, np.floating) and value.dtype.itemsize > 8:
        if not np.isfinite(value):
            return False
        return np.longdouble(float(value)) != value
    return False


def _survives_float(whole: int) -> bool:
    r"""Whether ``whole`` is the same integer after a float64 round trip."""
    try:
        return int(float(whole)) == whole
    except OverflowError:
        return False


def _integral_source(value: Any) -> int | None:
    r"""``value`` as an exact integer, if it is one, before any conversion.

    A cell reaches the float branch after `_json_value` has already
    normalised it, and that normalisation is lossy for exactly the values the
    exactness guard exists to catch. Asked here, of the original.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, Decimal):
        if not value.is_finite() or value != value.to_integral_value():
            return None
        return int(value)
    return None


def _is_finite_input(value: Any) -> bool:
    r"""Whether ``value`` was finite before the float conversion.

    A `Decimal` too wide for a float becomes `inf`, which is a different fault
    from a value that was infinite to begin with and must not be reported as
    one.
    """
    try:
        return bool(Decimal(str(value)).is_finite())
    except (ArithmeticError, ValueError, TypeError):
        return False


def _wire_string(value: Any) -> str:
    return value if isinstance(value, str) else str(value)


def _timestamp_json_value(value: pd.Timestamp) -> str | None:
    if pd.isna(value):
        return None
    if value.tzinfo is not None:
        value = value.tz_convert('UTC').tz_localize(None)
    return value.isoformat(timespec='microseconds') + 'Z'
