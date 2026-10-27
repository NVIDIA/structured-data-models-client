# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native observations between fixed core direct-request preparation stages."""

from __future__ import annotations

import ast
import decimal
import json
import math
import struct
import sys
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
from sdm_client._rust.run import Finished, run
from sdm_client.errors import StructuredDataError
from sdm_client.schema import _as_key_tuple

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
    finished = _prepared_run(value, attached)
    answer = results.expect(finished.result, results.Prepared)
    return json.loads(finished.take(answer.value))


def _prepared_run(
    value: dict[str, Any], attached: dict[int, Any] | None = None
) -> Finished:
    try:
        finished = run(
            {'op': 'prepare_request', 'input': 1},
            {1: codec.attachment(value), **(attached or {})},
        )
    except StructuredDataError as error:
        # The core selected this closed native exception or development
        # boundary. This mapping presents it; it performs no validation.
        if error._sentence == 'direct_unfinished':
            raise RouteError(error.message) from None
        raise
    return finished


def _builtin(value: Any) -> dict[str, Any]:
    """Convert literal-built native values without invoking their display."""
    if value is None:
        return {'kind': 'null'}
    if type(value) is bool:
        return {'kind': 'boolean', 'value': value}
    if type(value) is int:
        return {'kind': 'integer', 'digits': hex(value), 'radix': 'hexadecimal'}
    if type(value) is float:
        return {'kind': 'float', 'bits': list(struct.pack('>d', value))}
    if type(value) is str:
        return {'kind': 'unicode', 'points': [ord(c) for c in value]}
    if value is Ellipsis:
        return {'kind': 'ellipsis'}
    return {
        'kind': {
            bytes: 'bytes',
            complex: 'complex',
            list: 'list',
            tuple: 'tuple',
            dict: 'dictionary',
            set: 'set',
        }[type(value)]
    }


def _failure(error: Exception) -> dict[str, Any]:
    return {
        'ancestry': [
            name
            for kind, name in (
                (SyntaxError, 'syntax_error'),
                (ValueError, 'value_error'),
                (TypeError, 'type_error'),
                (MemoryError, 'memory_error'),
                (RecursionError, 'recursion_error'),
            )
            if isinstance(error, kind)
        ]
    }


def _native_value_call(
    value: dict[str, Any], native_error: Exception | None
) -> dict[str, Any]:
    selected = None
    try:
        return _call(value)
    except StructuredDataError as error:
        if error._sentence != 'direct_native_failure' or native_error is None:
            raise
        selected = native_error
    # The original native exception survives the plain-data decision.
    assert selected is not None  # Core selected the preserved native failure.
    errors.raise_native(selected)


def _native_json(text: str) -> tuple[Any, Exception | None]:
    """Observe actual native JSON conversion without choosing fallback."""
    try:
        return json.loads(text), None
    except Exception as error:
        return None, error


def _native_literal(
    text: str, json_error: Exception
) -> tuple[Any, Exception | None]:
    """Evaluate a selected literal within the same native JSON exception."""
    original_context = json_error.__context__
    original_traceback = json_error.__traceback__
    try:
        raise json_error
    except Exception as selected:
        # Re-entering an existing exception inside a caller handler can
        # replace its scanner context. Restore it before any AST warnings.
        selected.__context__ = original_context
        selected.__traceback__ = original_traceback
        try:
            return ast.literal_eval(text), None
        except Exception as error:
            return None, error


def _parsed_value(native: Any, failure: Exception | None) -> dict[str, Any]:
    if failure is not None:
        return {'outcome': 'failed', 'failure': _failure(failure)}
    return {
        'outcome': 'value',
        'value': _builtin(native),
        'members': [_builtin(v) for v in native]
        if type(native) is list
        else None,
    }


def _attribute_native(checked: dict[str, Any]) -> dict[str, Any]:
    text = checked['id']['value']
    native, json_error = _native_json(text)
    interpreted = _native_value_call(
        {
            'stage': 'interpret_response_json',
            'checked': checked,
            'json': {
                'scope': checked['scope'],
                'ordinal': checked['row']['ordinal'],
                'text': text,
                'value': _parsed_value(native, json_error),
            },
        },
        json_error,
    )
    literal = None
    native_error = None
    if interpreted['checked']['id']['representation'] == 'native_literal_text':
        assert (
            json_error is not None
        )  # Core selected the held failed conversion.
        native, native_error = _native_literal(text, json_error)
        literal = {
            'scope': checked['scope'],
            'ordinal': checked['row']['ordinal'],
            'text': text,
            'value': _parsed_value(native, native_error),
        }
    selected = _native_value_call(
        {
            'stage': 'select_response_member',
            'interpreted': interpreted,
            'literal': literal,
        },
        native_error,
    )
    string = None
    native_error = None
    if selected['member'] is not None:
        string = {
            'scope': checked['scope'],
            'ordinal': checked['row']['ordinal'],
        }
        try:
            displayed = str(native[0])
        except Exception as error:
            native_error = error
            string['value'] = {'outcome': 'failed', 'failure': _failure(error)}
        else:
            string['value'] = {
                'outcome': 'value',
                'points': [ord(c) for c in displayed],
            }
    return _native_value_call(
        {
            'stage': 'attribute_response_row',
            'selected': selected,
            'string': string,
        },
        native_error,
    )


def _response(finished: Finished) -> Finished:
    """Use immutable owned buffers once and bounded slices after native effects."""
    answer = results.expect(finished.result, results.DecodedDirectResponse)
    data = json.loads(finished.take(answer.value))
    buffers = {
        name: finished.take(data[name])
        for name in ('payload', 'response', 'rows')
    }
    rows = json.loads(buffers['rows'])
    accepted = []
    at = 0
    end = len(rows)
    getter = getattr(sys, 'get_int_max_str_digits', None)
    while at < len(rows):
        checked = _call(
            {
                'stage': 'check_response_rows',
                'scope': data['scope'],
                'rows': rows[at:end],
            }
        )
        accepted.extend(checked['rows'])
        at += len(checked['rows'])
        if checked['row'] is not None:
            accepted.append(_attribute_native(checked['row']))
            at += 1
        # Native JSON is observed only for the reached checked row. Bounded
        # later slices never replay a growing response or scoring-key set.
        end = min(len(rows), at + 256)
    attached = {
        n: buffers[name]
        for n, name in enumerate(('payload', 'response', 'rows'), 2)
    }
    data.update(payload=2, response=3, rows=4)
    attached[5] = codec.attachment(accepted)
    return _prepared_run(
        {
            'stage': 'layout_response',
            'data': data,
            'attributed': 5,
            'decimal_limit': getter() if getter else 0,
        },
        attached,
    )


def _lookup(
    index: pd.Index,
    value: Any,
    failures: dict[tuple[str, int | None], Exception] | None = None,
    source: str = '',
    ordinal: int | None = None,
) -> dict[str, Any]:
    # Membership/get_loc is a native Index observation, not target/schema
    # validation. Capture its fault without raising it before an earlier guard.
    try:
        found = value in index
    except TypeError as error:
        if failures is not None:
            failures[source, ordinal] = error
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
    native_lookups: dict[tuple[str, int | None], Exception] = {}

    def lookup_fault(fields: Any) -> None:
        errors.raise_native(
            native_lookups[fields['lookup'], fields.get('ordinal')]
        )

    def ambiguous_fault(fields: Any) -> None:
        values = context[target]
        # Only the native truth conversion of the already selected guard.
        if fields['source'] == 'known_target':
            bool(values.notna().sum() == 0)
        else:
            bool(values.dropna().nunique() >= 2)

    with ExitStack() as holding:
        holding.enter_context(
            errors.holding(
                direct_lookup=lookup_fault,
                direct_ambiguous=ambiguous_fault,
                direct_predict_dtype=lambda fields: (
                    predict[predict.columns[fields['ordinal']]].dtype
                ),
                direct_schema_missing=lambda fields: context[fields['name']],
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
                direct_related_names=lambda fields: sorted(related),
                direct_related=lambda fields: repr(fields['related_tables']),
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
                'related': [_form(frame) for frame in related.values()],
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
                'target_context': _lookup(
                    context.columns, target, native_lookups, 'target_context'
                ),
                'target_predict': _lookup(
                    predict.columns, target, native_lookups, 'target_predict'
                ),
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
        if schema is None:
            early = _call(
                {
                    'stage': 'context',
                    'headers': headers,
                    'context': 2,
                    'predict': 3,
                    'entity_table': entity_table,
                },
                {
                    2: _buffers(context, context_header),
                    3: _buffers(predict, predict_header),
                },
            )
            schema = frames.from_data(
                Schema, {entity_table: context}, relationships=[]
            )
            answer = _call(
                {
                    'stage': 'inferred_schema',
                    'context': early,
                    'schema': spec.held_schema(schema),
                }
            )
        else:
            answer = _call(
                {
                    'stage': 'schema',
                    'headers': headers,
                    'context': 2,
                    'predict': 3,
                    'entity_table': entity_table,
                    'schema': None
                    if schema is None
                    else spec.held_schema(schema),
                },
                {
                    2: _buffers(context, context_header),
                    3: _buffers(predict, predict_header),
                },
            )
        # Schema inference advice has now been presented. These are the same
        # original native frames, observed again only for the remaining guards.
        context_header, predict_header = _header(context), _header(predict)
        answer = _call(
            {
                'stage': 'finish_constructor',
                'schema': answer,
                'context': 2,
                'predict': 3,
                'context_header': context_header,
                'predict_header': predict_header,
                'target_context': _lookup(
                    context.columns, target, native_lookups, 'target_context'
                ),
                'context_predict': [
                    _lookup(
                        predict.columns,
                        column,
                        native_lookups,
                        'context_predict',
                        at,
                    )
                    for at, column in enumerate(context.columns)
                ],
            },
            {
                2: _buffers(context, context_header),
                3: _buffers(predict, predict_header),
            },
        )
        # Task construction reaches only the selected native metadata now.
        # No Schema/Table __getitem__ validation or unrelated column meaning
        # is consumed by this conversion.
        task_tables = schema.tables
        holding.enter_context(
            errors.holding(
                direct_task_tables=lambda fields: sorted(task_tables),
                direct_task_column=lambda fields: {
                    'table': task_tables[entity_table].name,
                    'columns': sorted(task_tables[entity_table].columns),
                },
            )
        )
        table = task_tables.get(entity_table)
        column = None if table is None else table.columns.get(target)
        task_data = _call(
            {
                'stage': 'constructor_task',
                'prepared': answer,
                'entity_table': entity_table,
                'tables': [_name(name) for name in task_tables],
                'table': None
                if table is None
                else {
                    'columns': [_name(name) for name in table.columns],
                    'target_dtype': None if column is None else column.dtype,
                },
            }
        )
    request.schema = schema
    request.task = Task(
        kind=task_data['kind'],
        target=Target(
            column_name=task_data['target'],
            dtype=task_data['dtype'],
            table_name=task_data['table'],
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
        # Native ownership/back-reference copying precedes core key marking.
        # Publish the core's canonical metadata before a consistency refusal.
        schema._own_tables()
        canonical = _call(
            {'stage': 'adopt_schema', 'schema': spec.held_schema(schema)}
        )
        from sdm_client._rust.edits import _store

        _store(schema, canonical, None)
        _call({'stage': 'consistent_schema', 'schema': canonical})
        # Capture the table object as the adapter does. A missing table remains
        # a core-selected schema error in Envelope, before this reference is used.
        table = (
            schema.tables.get(request.entity_table)
            if schema is not None
            else None
        )
        context = request.context.instance_table
        scored = request.predict.instance_table
        context_header, predict_header = _header(context), _header(scored)
        with errors.holding(
            direct_nullable_cast=lambda fields: (
                scored.iloc[:, fields['ordinal']].dropna().astype('int64')
            )
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
        assert table is not None  # Envelope proved the captured entity exists.
        primary_key = list(_as_key_tuple(table.primary_key))
        task = request.task.kind
        context_header, predict_header = _header(context), _header(scored)
        value = {
            'envelope': envelope,
            'context': 2,
            'predict': 3,
            'context_header': context_header,
            'predict_header': predict_header,
            'task': task,
            'primary_key': primary_key,
        }
        attached = {
            2: _buffers(context, context_header),
            3: _buffers(scored, predict_header),
        }
    if not empty:
        encoded = _prepared_run(
            {'stage': 'encode_body', 'input': value}, attached
        )
        body_result = results.expect(encoded.result, results.Prepared)
        body = json.loads(encoded.take(body_result.value))
        attached = {2: encoded.take(body['payload'])}
        body['payload'] = 2
        # Both encoders have finished before native ID/inference observations.
        request_id = _request_id()
        if type(request.inference) not in (
            dict,
            type(None),
            bool,
            int,
            float,
            str,
            bytes,
            list,
            tuple,
            set,
            frozenset,
        ):
            raise RouteError(
                'the partial direct request requires built-in inference values before truthiness'
            )
        value = {
            'stage': 'encoded',
            'body': body,
            'inference': _json(request.inference)
            if request.inference
            else {'entries': []},
        }
    else:
        request_id = ''
    operation = {
        'op': 'predict',
        'direct': 1,
        'model': call['model'],
        'request_id': request_id,
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
    if not empty:
        with errors.holding(
            direct_row_index=lambda fields: str(
                int(decimal.Decimal(fields['index']))
            )
        ):
            finished = _response(finished)
    answered = results.expect(finished.result, results.Predicted)
    table = prediction._table(finished.take(answered.result))
    result = prediction._result(
        table.to_pylist(),
        model=answered.model,
        request_id=answered.request_id,
        metadata=codec.restored_json(finished.take(answered.metadata)),
        outputs=answered.outputs,
        by_step=False,
    )
    names = (table.schema.metadata or {}).get(b'names_its_rows')
    if names is not None:
        result._names_its_rows = names == b'true'
    return result
