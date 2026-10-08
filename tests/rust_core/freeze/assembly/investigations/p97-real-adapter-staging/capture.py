# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Capture actual public operations, including complete refused observations."""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any

from common import (
    diagnostic,
    helpers,
    observe_public_call,
    outcome,
    require,
    require_path,
    strict_json,
)


def schema_columns(schema: Any) -> list[dict]:
    if schema is None:
        return []
    return [
        {
            'table': table_name,
            'column': name,
            'dtype': str(column.dtype),
            'stype': str(column.stype),
            'is_time': name == table.time_column,
        }
        for table_name, table in schema.tables.items()
        for name, column in table.columns.items()
    ]


def materialized_rows(observation: dict) -> int:
    rows = 0
    for payload in observation['payloads']:
        require(isinstance(payload, dict), 'capture.payload_shape')
        present = False
        for half in ('context', 'predict'):
            if half not in payload:
                continue
            present = True
            split = payload[half]
            require(isinstance(split, dict), 'capture.split_shape')
            table = split.get('instance_table')
            require(isinstance(table, dict), 'capture.table_shape')
            values = table.get('rows')
            require(isinstance(values, list), 'capture.rows_shape')
            rows += len(values)
        require(present, 'capture.no_materialized_half')
    if observation['payloads']:
        return rows
    result = observation['result']
    frames = (
        result.get('materialized_frames') if isinstance(result, dict) else None
    )
    if frames is None:
        return 0
    require(set(frames) == {'context', 'predict'}, 'capture.frame_halves')
    for frame in frames.values():
        require(isinstance(frame['index'], list), 'capture.frame_index')
        count = len(frame['index'])
        require(
            all(
                len(column['values']) == count
                for column in frame['columns'].values()
            ),
            'capture.frame_lengths',
        )
        rows += count
    return rows


def validate_materialized_rows(record: dict) -> None:
    require(
        type(record['materialized_rows']) is int
        and record['materialized_rows']
        == materialized_rows(record['observation']),
        'capture.row_count',
    )


@contextlib.contextmanager
def observe_sql(switch: Any):
    import sdm_connectors.source as source

    classes = {
        value
        for module_name, module in list(sys.modules.items())
        if module_name.startswith('sdm_connectors') and module is not None
        for value in list(vars(module).values())
        if inspect.isclass(value)
        and issubclass(value, source.Source)
        and '_run' in vars(value)
    }
    originals = {value: vars(value)['_run'] for value in classes}
    recorded = switch.SourceRuns()
    recorded.install()
    try:
        yield recorded.calls
    finally:
        for value, original in originals.items():
            value._run = original


def complete_public_capture_v1(repo: Path, recipe: dict) -> dict:
    switch, _driver, _request = helpers(repo)
    import _fixture_ops as operations
    import pandas as pd
    import sdm_core._native as native
    from sdm_client import PredictiveQuery, Schema
    from sdm_client.client import FittedModel, SDMClient

    entrypoints = {
        'PredictiveQuery.materialize': PredictiveQuery.materialize,
        'SDMClient.predict': SDMClient.predict,
        'FittedModel.predict': FittedModel.predict,
    }
    require(
        recipe.get('entrypoint') in entrypoints, 'capture.public_entrypoint'
    )
    require(
        recipe.get('operation') in {'materialize', 'client.run'},
        'capture.operation',
    )
    require(
        recipe['operation'] != 'materialize'
        or recipe['entrypoint'] == 'PredictiveQuery.materialize',
        'capture.entrypoint_operation',
    )
    classifier = switch.PortedCode()
    native_functions = tuple(
        value for value in vars(native).values() if inspect.isbuiltin(value)
    )
    require(bool(native_functions), 'capture.native_exports')
    schema = None
    result = None
    payloads = []
    error = None

    def operation() -> None:
        nonlocal schema, result, payloads, error
        try:
            if recipe['operation'] == 'client.run':
                value = recipe['input']
                frames = {
                    name: operations.build_frame(spec)
                    for name, spec in value['tables'].items()
                }
                session = operations.Answering(value.get('responses', []))
                flow = operations.Flow(frames, session)
                try:
                    flow.run(value['calls'])
                finally:
                    result = operations.encode(flow.output())
                    schema = flow.schema
                    payloads = [
                        operations._without_request_id(body)
                        for body in session.bodies
                    ]
                return
            value = recipe['input']
            if 'tables' in value:
                frames = {
                    name: operations.build_frame(spec)
                    for name, spec in value['tables'].items()
                }
            else:
                from _parity import data as datagen

                frames = datagen.generate(
                    value['data']['generator'], value['data']['seed']
                )
                frames = {name: frames[name] for name in value['table_names']}
            backend = value['backend']
            require(
                backend in {'memory', 'duckdb', 'sqlite'},
                'capture.offline_backend',
            )
            options = {
                'primary_keys': operations._keys(value.get('primary_keys')),
                'time_columns': value.get('time_columns'),
                'relationships': operations._relationships(
                    value.get('relationships')
                ),
            }
            with contextlib.ExitStack() as stack:
                if backend == 'memory':
                    schema = Schema.from_data(frames, **options)
                else:
                    from _parity.runner import _duckdb, _source, _sqlite

                    spec = value.get('database', {})
                    if backend == 'duckdb':
                        connection = _duckdb(frames, spec)
                    else:
                        directory = Path(
                            stack.enter_context(tempfile.TemporaryDirectory())
                        )
                        connection = _sqlite(frames, directory, spec)
                    stack.callback(connection.close)
                    source = _source(
                        {'backend': backend, 'data': spec},
                        connection,
                        connection,
                    )
                    schema = Schema.from_source(
                        source,
                        tables={name: name for name in frames},
                        **options,
                    )
                for edit in value.get('edits', []):
                    setattr(
                        schema[edit['table']][edit['column']],
                        edit['set'],
                        edit['value'],
                    )
                request = PredictiveQuery(value['query'], schema).materialize(
                    **operations.decode(value.get('options', {}))
                )
                payloads = [
                    operations._without_request_id(payload)
                    for payload in request.payloads or ()
                ]
                result = {'payload_count': len(payloads)}
                if not payloads:
                    result['materialized_frames'] = {
                        half: {
                            **operations._frame_out(
                                getattr(request, half).instance_table
                            ),
                            'index': operations.encode(
                                getattr(
                                    request, half
                                ).instance_table.index.tolist()
                            ),
                        }
                        for half in ('context', 'predict')
                    }
        except Exception as raised:
            error = diagnostic(raised, operations.encode)

    with (
        warnings.catch_warnings(record=True) as caught,
        observe_sql(switch) as sql,
    ):
        warnings.simplefilter('always')
        _ignored, counts = observe_public_call(
            operation,
            entrypoints[recipe['entrypoint']].__code__,
            classifier.symbol,
            native_functions,
        )
    selector = os.environ.get(switch.SELECTOR, switch.CORE)
    require_path(counts, selector)
    observed = outcome(
        result,
        schema=schema_columns(schema),
        payloads=payloads,
        sql=[
            {'statement': statement, 'parameters': parameters}
            for statement, parameters in sql
        ],
        error=error,
        warnings=[
            diagnostic(item.message, operations.encode) for item in caught
        ],
    )
    return {
        'observation': observed,
        **counts,
        'selector': selector,
        'pandas_major': pd.__version__.split('.')[0],
        'materialized_rows': materialized_rows(observed),
        'native_sha256': hashlib.sha256(
            Path(native.__file__).read_bytes()
        ).hexdigest(),
    }


if __name__ == '__main__':
    request = strict_json(sys.stdin.read())
    print(
        json.dumps(
            complete_public_capture_v1(Path(sys.argv[1]), request['recipe']),
            allow_nan=False,
        )
    )
