# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native ownership/conversion for the ordinary direct tabular consumer."""

from __future__ import annotations

import ast
import json
import math
import struct
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from sdm_client._routing import RouteError
from sdm_client._rust import codec, errors, results, spec
from sdm_client._rust import predict as prediction
from sdm_client._rust.run import Finished, run
from sdm_client._rust.direct_io import DirectOperations
from sdm_client.errors import StructuredDataError


def _run(
    op: str,
    value: dict[str, Any],
    attached: dict[int, Any] | None = None,
    **fields: Any,
) -> Finished:
    return run(
        {'op': op, 'input': 1, **fields},
        {1: codec.attachment(value), **(attached or {})},
    )


def _call(op: str, value: dict[str, Any]) -> dict[str, Any]:
    finished = _run(op, value)
    answer = results.expect(finished.result, results.DirectValue)
    return json.loads(finished.take(answer.value))


def _kind(value: Any) -> dict[str, Any]:
    if isinstance(value, pd.DataFrame):
        return {'kind': 'frame'}
    if isinstance(value, dict):
        return {'kind': 'dict'}
    if isinstance(value, pd.Series):
        return {'kind': 'series'}
    return {'kind': 'other', 'native_type': type(value).__name__}


def _header(frame: Any) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame):
        return {'rows': 0, 'columns': []}
    if type(frame) is not pd.DataFrame or any(
        type(name) is not str for name in frame.columns
    ):
        raise RouteError(
            'the direct pilot requires built-in DataFrames and string column labels'
        )
    return {'rows': len(frame), 'columns': list(frame.columns)}


def _table(frame: pd.DataFrame) -> Any:
    import pyarrow as pa

    arrays = []
    fields = []
    for at in range(len(frame.columns)):
        series = frame.iloc[:, at]
        dtype = series.dtype
        numeric = (
            isinstance(dtype, np.dtype)
            and dtype.isnative
            and dtype.kind in 'biuf'
            and dtype.itemsize <= 8
            and not (dtype.kind == 'f' and dtype.itemsize < 4)
        )
        nullable = type(dtype) in (
            pd.Int8Dtype, pd.Int16Dtype, pd.Int32Dtype, pd.Int64Dtype,
            pd.UInt8Dtype, pd.UInt16Dtype, pd.UInt32Dtype, pd.UInt64Dtype,
            pd.Float32Dtype, pd.Float64Dtype, pd.BooleanDtype,
        )
        arrow = type(dtype) is pd.ArrowDtype and (
            pa.types.is_integer(dtype.pyarrow_dtype)
            or pa.types.is_boolean(dtype.pyarrow_dtype)
            or dtype.pyarrow_dtype in (pa.float32(), pa.float64())
        )
        string = type(dtype) is pd.StringDtype
        timestamp = (
            isinstance(dtype, np.dtype) and dtype.kind == 'M'
        ) or type(dtype) is pd.DatetimeTZDtype or (
            type(dtype) is pd.ArrowDtype and pa.types.is_timestamp(dtype.pyarrow_dtype)
        )
        decimal = type(dtype) is pd.ArrowDtype and pa.types.is_decimal(dtype.pyarrow_dtype)
        categorical = type(dtype) is pd.CategoricalDtype and (
            type(dtype.categories.dtype) is pd.StringDtype
            or (
                isinstance(dtype.categories.dtype, np.dtype)
                and dtype.categories.dtype.kind in 'iuf'
                and dtype.categories.dtype.itemsize <= 8
            )
        )
        if not (numeric or nullable or arrow or string or categorical or timestamp or decimal):
            raise RouteError(
                'this direct column needs native storage observations'
            )
        array = pa.array(
            series,
            type=pa.string() if string else None,
            from_pandas=numeric or string or timestamp,
        )
        if categorical:
            array = array.dictionary_decode()
            if pa.types.is_large_string(array.type):
                array = array.cast(pa.string())
        storage: dict[str, Any]
        if timestamp:
            storage = {
                'family': 'timestamp',
                'unit': {
                    's': 'seconds', 'ms': 'milliseconds',
                    'us': 'microseconds', 'ns': 'nanoseconds',
                }[array.type.unit],
                'zone': array.type.tz,
            }
            array = array.cast(pa.int64())
        elif decimal:
            storage = {
                'family': 'decimal', 'precision': array.type.precision,
                'scale': array.type.scale,
            }
        elif categorical:
            storage = {'family': 'categorical'}
        elif string:
            storage = {'family': 'string'}
        else:
            storage = {
                'family': 'primitive',
                'owner': 'array' if numeric else 'nullable' if nullable else 'arrow',
                'floats': (
                    'nan_missing' if numeric else 'validity_masked'
                ) if pa.types.is_floating(array.type) else None,
            }
        arrays.append(array)
        fields.append(pa.field(
            frame.columns[at], array.type,
            metadata={'sdm.direct.storage.v1': json.dumps(storage)},
        ))
    # The explicit DirectHeader carries zero-column row counts separately.
    # The core reconstructs such a RecordBatch using RecordBatchOptions.
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


_native_calls = DirectOperations(_call, _run, _table)


def _flags(schema: Any, entity: str) -> list[dict[str, Any]]:
    table = schema.tables.get(entity)
    return (
        []
        if table is None
        else [
            {
                'name': name,
                'untyped': column.untyped,
                'stype_declared': column.stype_declared,
            }
            for name, column in table.columns.items()
        ]
    )


def _native_encoding(frame: pd.DataFrame) -> bool:
    """Whether this frame needs reached scalar conversions during encoding."""
    for at, dtype in enumerate(frame.dtypes):
        if type(dtype) is pd.StringDtype:
            if not DirectOperations.passive(frame.iloc[:, at]):
                return True
            continue
        if (
            isinstance(dtype, np.dtype) and dtype.isnative
            and dtype.kind in 'biuf' and dtype.itemsize <= 8
            and not (dtype.kind == 'f' and dtype.itemsize < 4)
        ):
            continue
        return True
    return False


def _schema(held: dict[str, Any], context: pd.DataFrame, flags: Sequence[Mapping[str, Any]] = ()) -> Any:
    from sdm_client.schema import Column, Schema, Table

    made = Schema.__new__(Schema)
    tables = {}
    for definition in held['tables']:
        table = Table.__new__(Table)
        columns = {}
        for definition_column in definition['columns']:
            column = Column.__new__(Column)
            for name, value in (
                ('name', definition_column['name']),
                ('dtype', definition_column['dtype']),
                ('stype', definition_column['stype']),
                ('untyped', any(flag['name'] == definition_column['name'] and flag['untyped'] for flag in flags)),
                ('stype_declared', False),
                ('_stype_before_key', definition_column['stype_before_key']),
                (
                    '_source',
                    None
                    if definition_column.get('source') is None
                    else results.SourceFacts.from_contract(
                        definition_column['source']
                    ),
                ),
                ('_table', table),
            ):
                object.__setattr__(column, name, value)
            columns[column.name] = column
        key = definition['primary_key']
        for name, value in (
            ('name', definition['name']),
            ('data', context),
            ('columns', columns),
            (
                'primary_key',
                None if not key else key[0] if len(key) == 1 else tuple(key),
            ),
            ('time_column', definition['time_column']),
            (
                'duplicate_keys',
                0
                if definition.get('duplicate_keys') is None
                else int(codec.decode_scalar(definition['duplicate_keys'])),
            ),
            ('_schema', made),
        ):
            object.__setattr__(table, name, value)
        tables[table.name] = table
    for name, value in (
        ('_tables', tables),
        ('_source', None),
        ('relationships', []),
        ('_engine_tables', None),
        ('_conversion_notes', ()),
    ):
        object.__setattr__(made, name, value)
    return made


def construct(
    request: Any,
    context: Any,
    predict: Any,
    *,
    target: str,
    task: Any,
    related_tables: Mapping[str, Any] | None,
    schema: Any,
    entity_table: str,
    inference: Mapping[str, Any] | None,
    task_options: Mapping[str, Any] | None,
) -> None:
    from sdm_client.request import Split, Target, Task, _has_container_cells

    related = dict(related_tables or {})
    shape = {
        'target_containers': False,
        'context_kind': _kind(context),
        'predict_kind': _kind(predict),
        'related': [(name, _kind(frame)) for name, frame in related.items()],
        'task': task if type(task) is str else None,
        'target': target,
        'entity_table': entity_table,
        'context': _header(context),
        'predict': _header(predict),
    }
    with errors.holding(
        direct_task=lambda fields: task,
        direct_frame=lambda fields: 'pandas DataFrame',
    ):
        _call('check_direct_preparation', {'kind': 'arguments', 'shape': shape})
        shape['target_containers'] = target in context.columns and _has_container_cells(context[target])
        _call('check_direct_shape', shape)
    from sdm_client._rust import direct_schema

    if direct_schema.preparation_requires(context) or direct_schema.preparation_requires(predict):
        schema = direct_schema.prepare(context, predict, shape, schema, _native_calls, _schema, _flags)
        value = {'task': shape['task'], 'target': target}
    else:
        converted_context = _table(context)
        converted_predict = _table(predict)
        prepared = _run(
            'prepare_direct',
            {
                'shape': shape,
                'schema': None if schema is None else spec.held_schema(schema),
                'flags': [] if schema is None else _flags(schema, entity_table),
            },
            {2: converted_context, 3: converted_predict},
            context=2,
            predict=3,
        )
        description = results.expect(prepared.result, results.DirectValue)
        value = json.loads(prepared.take(description.value))
        if schema is None:
            schema = _schema(value['schema'], context)
    request.schema = schema
    request.task = Task(
        kind=value['task'],
        target=Target(
            column_name=value['target'],
            dtype=schema[entity_table][value['target']].dtype,
            table_name=entity_table,
        ),
        entity_table=None,
        options=dict(task_options or {}),
    )
    request._context = Split(instance_table=context, related_tables=related)
    request._predict = Split(instance_table=predict, related_tables=related)
    request.inference = dict(inference or {})
    request.requested_inference = dict(inference or {})
    request.entity_table = entity_table


def predict(
    client: Any, request: Any, model: Any, outputs: Sequence[str] | None
) -> Any:
    from sdm_client.adapters.tabular import _request_id
    from sdm_client.models import KumoRelational, KumoTabular, spec_for

    if type(model) is str:
        model_id = model
    elif type(model) in (KumoTabular, KumoRelational):
        model_id = model.name
    else:
        model_id = spec_for(model).model_id
    bound = _call(
        'bind_direct',
        {
            'model': model_id,
            'outputs': None if outputs is None else list(outputs),
            'kind': request.task.kind,
            'anchored': bool(request.task.options.get('anchor_time_column')),
            'relationships': bool(request.schema.relationships),
            'tables': len(request.schema),
            'context_rows': request._context_rows_per_call(),
        },
    )
    attached: dict[int, Any] = {}
    request_id = None
    if request.nothing_to_score:
        data = {
            'kind': 'empty',
            'bound': bound,
            'target_dtype': request.task.target.dtype,
        }
        direct = {'input': 1}
    else:
        from sdm_client._rust import direct_prediction, direct_schema

        if direct_schema.requires(request.context.instance_table) or direct_schema.requires(request.predict.instance_table):
            direct_prediction.inspect(request, _native_calls, _run)
        else:
            _run(
                'inspect_direct_pair',
                {
                    'bound': bound,
                    'schema': spec.held_schema(request.schema),
                    'flags': _flags(request.schema, request.entity_table),
                    'entity_table': request.entity_table,
                    'target': request.task.target.column_name,
                    'kind': request.task.kind,
                    'context': _header(request.context.instance_table),
                    'predict': _header(request.predict.instance_table),
                },
                {
                    2: _table(request.context.instance_table),
                    3: _table(request.predict.instance_table),
                },
                context=2,
                predict=3,
            )

        if direct_schema.requires(request.context.instance_table) or direct_schema.requires(request.predict.instance_table):
            table, context, scoring, resolved = direct_prediction.resolve(
                request, bound, _native_calls, _flags, _header,
            )
        else:
            # Capture the actual live owners and their dtypes at the adapter boundary.
            schema = request.schema
            table = schema[request.entity_table]
            context = request.context.instance_table
            scoring = request.predict.instance_table
            resolved_run = _run(
                'resolve_tabular',
                {
                    'bound': bound,
                    'schema': spec.held_schema(schema),
                    'flags': _flags(schema, request.entity_table),
                    'entity_table': request.entity_table,
                    'target': request.task.target.column_name,
                    'kind': request.task.kind,
                    'context': _header(context),
                    'predict': _header(scoring),
                },
                {2: _table(context), 3: _table(scoring)},
                context=2,
                predict=3,
            )
            resolved_value = results.expect(
                resolved_run.result, results.DirectValue
            )
            resolved = json.loads(resolved_run.take(resolved_value.value))
        # Advice has reached Python. Read the held native owners again, while
        # retaining the wire dtype decisions made before that advice.
        key = table.primary_key
        primary_key = [] if not key else [key] if isinstance(key, str) else list(key)
        kind = request.task.kind
        context_header = _header(context)
        native_encoding = _native_encoding(context) or _native_encoding(scoring)
        if native_encoding:
            from sdm_client._rust.direct_cells import table as native_table

            dtypes = dict(resolved['columns']['dtypes'])
            # Finish context before observing scoring values. In particular,
            # a failing context conversion prevents later native side effects.
            context_values = native_table(context, 'context', dtypes, _native_calls)
            scoring_header = _header(scoring)
            scoring_values = native_table(scoring, 'predict', dtypes, _native_calls)
        else:
            context_values = _table(context)
            scoring_header = _header(scoring)
            scoring_values = _table(scoring)
        encoded = _run(
            'encode_tabular',
            {
                'resolved': resolved,
                'primary_key': primary_key,
                'kind': kind,
                'context': context_header,
                'predict': scoring_header,
            },
            {2: context_values, 3: scoring_values},
            context=2,
            predict=3,
        )
        description = results.expect(encoded.result, results.DirectValue)
        body = json.loads(encoded.take(description.value))
        attached[2] = encoded.take(body['payload'])
        body['payload'] = 2
        request_id = _request_id()
        # No native inference conversion occurs until column encoding succeeded.
        from sdm_client._rust import direct_options

        observed_options = direct_options.observe(request.inference, _call) if request.inference else {}
        if not direct_options.can_encode(observed_options):
            raise RouteError('native inference transport is not enabled in this input-family increment')
        data = {'kind': 'encoded', 'body': body, 'inference': direct_options.encode(observed_options)}
        direct = {'input': 1, 'payload': 2}
    operation = {
        'op': 'predict',
        'direct': direct,
        'model': bound['model'],
        'request_id': _request_id() if request_id is None else request_id,
        'outputs': bound['outputs'],
        'default_outputs': not bound['named'],
    }
    endpoint = getattr(client._transport, 'model_endpoint', None)
    if endpoint is not None:
        operation['model_endpoint'] = endpoint
    finished = run(
        operation,
        {1: codec.attachment(data), **attached},
        model=prediction.sender(client, model_id),
    )
    if finished.result['op'] == 'decoded_direct_response':
        finished = _response(finished)
    answer = results.expect(finished.result, results.Predicted)
    return prediction._result(
        prediction._rows(finished.take(answer.result)),
        model=answer.model,
        request_id=answer.request_id,
        metadata=json.loads(finished.take(answer.metadata)),
        outputs=answer.outputs,
        by_step=False,
    )


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
        return _call(
            value['op'],
            {key: item for key, item in value.items() if key != 'op'},
        )
    except StructuredDataError as error:
        if error._sentence != 'direct_native_failure' or native_error is None:
            raise
        selected = native_error
    # The original native exception survives the plain-data decision.
    assert selected is not None  # Core selected the preserved native failure.
    raise selected


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
            'op': 'interpret_direct_json',
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
            'op': 'select_direct_member',
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
            'op': 'attribute_direct_row',
            'selected': selected,
            'string': string,
        },
        native_error,
    )


def _response(finished: Finished) -> Finished:
    answer = results.expect(finished.result, results.DecodedDirectResponse)
    data = json.loads(finished.take(answer.value))
    buffers = {
        name: finished.take(data[name])
        for name in ('payload', 'response', 'rows')
    }
    rows = json.loads(buffers['rows'])
    accepted = []
    at = 0
    while at < len(rows):
        checked = _call(
            'check_direct_rows',
            {'scope': data['scope'], 'rows': rows[at : at + 256]},
        )
        accepted.extend(checked['rows'])
        at += len(checked['rows'])
        if checked['row'] is not None:
            accepted.append(_attribute_native(checked['row']))
            at += 1
    data.update(payload=2, response=3)
    return _run(
        'layout_direct',
        {'data': data, 'attributed': accepted},
        {2: buffers['payload'], 3: buffers['response']},
        payload=2,
        response=3,
    )
