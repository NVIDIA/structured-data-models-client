# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native observations between fixed core direct-request preparation stages."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from sdm_client._routing import RouteError
from sdm_client._rust import (
    codec,
    errors,
    frames,
    results,
    schema as schemas,
    spec,
)
from sdm_client._rust.run import run
from sdm_client.errors import StructuredDataError

if TYPE_CHECKING:
    from sdm_client import Request, Schema, SDMClient
    from sdm_client.result import PredictionResult


def _text(value: str) -> bool:
    try:
        value.encode('utf-8')
    except UnicodeEncodeError:
        return False
    return True


def _form(value: Any) -> dict[str, Any]:
    if isinstance(value, pd.DataFrame):
        return {'kind': 'frame'}
    if isinstance(value, dict):
        return {'kind': 'dict'}
    if isinstance(value, pd.Series):
        return {'kind': 'series'}
    return {'kind': 'other', 'native_type': type(value).__name__}


def _call(
    value: dict[str, Any], attached: dict[int, Any] | None = None
) -> dict[str, Any]:
    try:
        finished = run(
            {'op': 'prepare_request', 'input': 1},
            {1: codec.attachment(value), **(attached or {})},
        )
    except StructuredDataError as error:
        # The core selected this closed native exception or development
        # boundary. This mapping presents it; it performs no validation.
        kind = {
            'direct_lookup': TypeError,
            'direct_ambiguous': ValueError,
            'direct_predict_dtype': AttributeError,
            'direct_nullable_cast': pd.errors.IntCastingNaNError,
            'direct_schema_missing': KeyError,
            'direct_unfinished': RouteError,
        }.get(error._sentence)
        if kind is not None:
            raise kind(error.message) from None
        raise
    answer = results.expect(finished.result, results.Prepared)
    return json.loads(finished.take(answer.value))


def _lookup(index: pd.Index, value: Any) -> dict[str, Any]:
    # Membership/get_loc is a native Index observation, not target/schema
    # validation. Capture its fault without raising it before an earlier guard.
    try:
        found = value in index
    except TypeError:
        return {'positions': [], 'unhashable': type(value).__name__}
    if not found:
        return {'positions': []}
    at = index.get_loc(value)
    if isinstance(at, slice):
        positions = list(range(*at.indices(len(index))))
    elif isinstance(at, np.ndarray):
        positions = (
            np.flatnonzero(at).tolist() if at.dtype.kind == 'b' else at.tolist()
        )
    else:
        positions = [int(at)]
    return {'positions': positions}


def _name(value: Any) -> dict[str, Any]:
    if type(value) is str:
        return (
            {'kind': 'text', 'value': value}
            if _text(value)
            else {'kind': 'unencodable_text'}
        )
    if value is None or type(value) in (bool, int, float):
        return {'kind': 'non_text'}
    raise RouteError(
        'the partial direct request requires ordinary scalar column labels'
    )


def _header(frame: pd.DataFrame) -> dict[str, Any]:
    if type(frame) is not pd.DataFrame or type(frame.columns) not in (
        pd.Index,
        pd.RangeIndex,
    ):
        raise RouteError(
            'the partial direct request requires built-in DataFrames and ordinary scalar column labels'
        )
    return {
        'rows': len(frame),
        'columns': [
            {
                'ordinal': at,
                'name': _name(name),
                'physical': {'type': physical}
                if (
                    physical := frames._numeric_physical(
                        frame.iloc[:, at].dtype
                    )
                )
                is not None
                else None,
                'float_storage': (
                    'nan_missing'
                    if isinstance(frame.iloc[:, at].dtype, np.dtype)
                    else 'validity_masked'
                )
                if physical in ('float32', 'float64')
                else None,
            }
            for at, name in enumerate(frame.columns)
        ],
        'duplicates': np.flatnonzero(frame.columns.duplicated()).tolist(),
    }


def _buffers(frame: pd.DataFrame, header: dict[str, Any]) -> Any:
    import pyarrow as pa

    arrays = []
    for at, column in enumerate(header['columns']):
        if column['physical'] is None:
            raise RouteError(
                'the partial direct request requires built-in NumPy or pandas masked numeric and Boolean storage'
            )
        arrays.append(frames._primitive_array(frame.iloc[:, at]))
    if not arrays:
        # Keep a row count even for a zero-column scoring half.
        batch = pa.record_batch(
            [pa.nulls(len(frame))], names=['__rows']
        ).select([])
        return pa.Table.from_batches([batch])
    return pa.Table.from_arrays(
        arrays, names=[str(at) for at in range(len(arrays))]
    )


def construct(
    request: Request,
    context: Any,
    predict: Any,
    *,
    target: Any,
    task: Any,
    related_tables: Any,
    schema: Schema | None,
    entity_table: Any,
    inference: Any,
    task_options: Any,
) -> None:
    from sdm_client import Schema
    from sdm_client.request import Split, Target, Task

    if related_tables is not None and type(related_tables) is not dict:
        raise RouteError(
            'the partial direct request requires ordinary related-table mappings'
        )
    related = dict(related_tables or {})
    with ExitStack() as holding:
        holding.enter_context(
            errors.holding(
                direct_task=lambda fields: task,
                direct_frame_name=lambda fields: (
                    fields['name']
                    if 'related_ordinal' not in fields
                    else f'related_tables[{list(related)[fields["related_ordinal"]]!r}]'
                ),
                direct_target=lambda fields: target,
                direct_context_labels=lambda fields: sorted(
                    map(str, context.columns)
                ),
                direct_related=lambda fields: repr(sorted(related)),
                direct_columns=lambda fields: repr(
                    [
                        (
                            context if fields['half'] == 'context' else predict
                        ).columns[at]
                        for at in fields['positions']
                    ]
                ),
                direct_duplicates=lambda fields: repr(
                    sorted(
                        {
                            str(
                                (
                                    context
                                    if fields['half'] == 'context'
                                    else predict
                                ).columns[at]
                            )
                            for at in fields['positions']
                        }
                    )
                ),
            )
        )
        arguments = _call(
            {
                'stage': 'arguments',
                'context': _form(context),
                'predict': _form(predict),
                'related': [
                    (name, _form(frame)) for name, frame in related.items()
                ],
                # Given remains unchanged: no eager repr of invalid arguments.
                'task': codec.given(task)
                if type(task) is str and _text(task)
                else codec.UNREADABLE,
            }
        )
        context_header, predict_header = _header(context), _header(predict)
        headers = _call(
            {
                'stage': 'headers',
                'arguments': arguments,
                'context': context_header,
                'predict': predict_header,
                'target_context': _lookup(context.columns, target),
                'target_predict': _lookup(predict.columns, target),
                'predict_context': [
                    _lookup(context.columns, column)['positions']
                    for column in predict.columns
                ],
            }
        )
        if type(entity_table) is not str or not _text(entity_table):
            raise RouteError(
                'the partial direct request requires an encodable entity-table name'
            )
        native_types = {
            entity_table: {
                name: context.iloc[:, at].dtype
                for at, name in enumerate(context.columns)
                if type(name) is str
            }
        }
        holding.enter_context(frames._presentation(native_types))
        answer = _call(
            {
                'stage': 'schema',
                'headers': headers,
                'context': 2,
                'predict': 3,
                'entity_table': entity_table,
                'schema': None if schema is None else spec.held_schema(schema),
            },
            {
                2: _buffers(context, context_header),
                3: _buffers(predict, predict_header),
            },
        )
        if schema is None:
            schema = schemas.build(
                Schema,
                None,
                answer['description'],
                frames={entity_table: context},
            )
    request.schema = schema
    request.task = Task(
        kind=answer['task'],
        target=Target(
            column_name=answer['target'],
            dtype=schema[entity_table][answer['target']].dtype,
            table_name=entity_table,
        ),
        entity_table=entity_table if related else None,
        options=dict(task_options or {}),
    )
    request._context = Split(instance_table=context, related_tables=related)
    request._predict = Split(instance_table=predict, related_tables=related)
    request.inference = dict(inference or {})
    request.requested_inference = dict(inference or {})
    request.entity_table = entity_table


def _json(value: Any) -> Any:
    # Native conversion only. Core wire encoding decides integer precision.
    if type(value) is not dict or any(
        type(key) is not str or not _text(key) for key in value
    ):
        raise RouteError(
            'the partial direct request requires a flat primitive inference mapping'
        )
    entries = []
    for key, item in value.items():
        if type(item) is int:
            encoded = {'integer': str(item)}
        elif item is None:
            encoded = {'value': {'t': 'null'}}
        elif type(item) is bool:
            encoded = {'value': {'t': 'bool', 'v': item}}
        elif type(item) is float and math.isfinite(item):
            encoded = {'value': {'t': 'f64', 'v': format(item, '.17g')}}
        elif type(item) is str and _text(item):
            encoded = {'value': {'t': 'utf8', 'v': item}}
        else:
            raise RouteError(
                'the partial direct request requires finite built-in scalar inference values'
            )
        entries.append((key, encoded))
    return {'entries': entries}


def _cast_message(scored: pd.DataFrame, fields: Any) -> str:
    """Present the native wording of the cast fault already selected by Rust."""
    try:
        scored.iloc[:, fields['ordinal']].dropna().astype('int64')
    except pd.errors.IntCastingNaNError as error:
        return str(error)
    raise RouteError(
        'the selected native cast fault no longer matches its captured column'
    )


def predict(
    client: SDMClient,
    request: Request,
    model: Any,
    outputs: Sequence[str] | None,
) -> PredictionResult:
    from sdm_client._rust import predict as prediction
    from sdm_client.adapters.tabular import _request_id
    from sdm_client.client import _require_output_names
    from sdm_client.models import KumoRelational, KumoTabular, spec_for

    _require_output_names(outputs)
    if type(model) is str:
        model_id = model
    elif type(model) in (KumoTabular, KumoRelational):
        model_id = model.name
    else:
        # The existing native model-form diagnostic, before a core lookup.
        model_id = spec_for(model).model_id
    call = _call({'stage': 'model', 'model': model_id})
    if outputs is not None and type(outputs) not in (list, tuple):
        raise RouteError(
            'the partial direct request requires built-in output sequences'
        )
    if outputs is None:
        call = _call(
            {'stage': 'outputs', 'call': call, 'kind': request.task.kind}
        )
    else:
        call = _call(
            {'stage': 'outputs', 'call': call, 'outputs': list(tuple(outputs))}
        )
    capability_kind = request.task.kind
    anchored = bool(request.task.options.get('anchor_time_column'))
    linked = bool(request.schema.relationships)
    table_count = len(request.schema)
    call = _call(
        {
            'stage': 'capabilities',
            'call': call,
            'kind': capability_kind,
            'anchored': anchored,
            'relationships': linked,
            'tables': table_count,
        }
    )
    call = _call({'stage': 'task', 'call': call, 'kind': request.task.kind})
    call = _call(
        {
            'stage': 'capacity',
            'call': call,
            'rows': request._context_rows_per_call(),
        }
    )
    empty = request.nothing_to_score
    attached: dict[int, Any] = {}
    if empty:
        # The reference's empty result reads no context values or inference.
        scored = request.predict.instance_table
        value = {
            'stage': 'empty',
            'bound': call,
            'schema': spec.held_schema(request.schema),
            'entity_table': request.entity_table,
            'predict_header': _header(scored),
        }
    else:
        # One-shot overlap advice runs before the adapter captures its refs.
        schema = request.schema
        entity_table = request.entity_table
        target = request.task.target.column_name
        context = request.context.instance_table
        scored = request.predict.instance_table
        context_header, predict_header = _header(context), _header(scored)
        overlap = _call(
            {
                'stage': 'overlap',
                'input': {
                    'bound': call,
                    'context': 2,
                    'predict': 3,
                    'context_header': context_header,
                    'predict_header': predict_header,
                    'schema': spec.held_schema(schema),
                    'entity_table': entity_table,
                    'target': target,
                    'task': request.task.kind,
                },
            },
            {
                2: _buffers(context, context_header),
                3: _buffers(scored, predict_header),
            },
        )
        # A leakage warning may have replaced Request's references. Capture
        # the adapter's references now and retain them across its text advice.
        schema = request.schema
        context = request.context.instance_table
        scored = request.predict.instance_table
        context_header, predict_header = _header(context), _header(scored)
        with errors.holding(
            direct_nullable_cast=lambda fields: _cast_message(scored, fields)
        ):
            envelope = _call(
                {
                    'stage': 'envelope',
                    'input': {
                        'overlap': overlap,
                        'context': 2,
                        'predict': 3,
                        'context_header': context_header,
                        'predict_header': predict_header,
                        'schema': spec.held_schema(schema),
                        'entity_table': request.entity_table,
                        'target': request.task.target.column_name,
                        'task': request.task.kind,
                    },
                },
                {
                    2: _buffers(context, context_header),
                    3: _buffers(scored, predict_header),
                },
            )
        # The same captured frames are read after text advice. In-place edits
        # are visible, while replacement Request references are not recaptured.
        context_header, predict_header = _header(context), _header(scored)
        value = {
            'stage': 'body',
            'envelope': envelope,
            'context': 2,
            'predict': 3,
            'context_header': context_header,
            'predict_header': predict_header,
            'task': request.task.kind,
            'inference': _json(request.inference),
        }
        attached = {
            2: _buffers(context, context_header),
            3: _buffers(scored, predict_header),
        }
    operation = {
        'op': 'predict',
        'direct': 1,
        'model': call['model'],
        'request_id': _request_id(),
        'outputs': call['outputs'],
        'default_outputs': not call['named'],
    }
    endpoint = getattr(client._transport, 'model_endpoint', None)
    if endpoint is not None:
        operation['model_endpoint'] = endpoint
    finished = run(
        operation,
        {1: codec.attachment(value), **attached},
        model=lambda body: prediction._send(client, call['model'], body),
    )
    answered = results.expect(finished.result, results.Predicted)
    table = prediction._table(finished.take(answered.result))
    result = prediction._result(
        table.to_pylist(),
        model=answered.model,
        request_id=answered.request_id,
        metadata=json.loads(finished.take(answered.metadata)),
        outputs=answered.outputs,
        by_step=False,
    )
    names = (table.schema.metadata or {}).get(b'names_its_rows')
    if names is not None:
        result._names_its_rows = names == b'true'
    return result
