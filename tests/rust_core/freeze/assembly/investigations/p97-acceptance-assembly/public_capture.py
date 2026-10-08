# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance-owned materialize capture using reviewed serializers and observer."""

from __future__ import annotations

import contextlib
import inspect
import os
import sys
import tempfile
import warnings
from pathlib import Path

import schema_boundary
from catalog import VettedCatalog
from dimensions import construct
from inline_old import observe as observe_inline_old
from recipes import open_recipe
from support import INVESTIGATIONS, Rejected, load, require, same_json, sha

sys.path.insert(0, str(INVESTIGATIONS / 'p97-real-adapter-staging'))
from capture import materialized_rows, observe_sql, schema_columns  # noqa: E402
from common import (  # noqa: E402
    diagnostic,
    helpers,
    observe_public_call,
    outcome,
)


def source_frame(repo, trace):
    from generator_oracles import source_codes
    from support import checked_path

    code = trace.tb_frame.f_code
    filename = Path(code.co_filename)
    globals_ = trace.tb_frame.f_globals
    module_file = globals_.get('__file__')
    spec = globals_.get('__spec__')
    origin = getattr(spec, 'origin', None) or module_file
    root = repo.resolve()

    def owned(value):
        return (
            isinstance(value, str)
            and Path(value).is_absolute()
            and Path(value).resolve().is_relative_to(root)
        )

    if not owned(code.co_filename) and not owned(origin):
        return None
    require(
        owned(origin)
        and filename.is_absolute()
        and filename.resolve() == Path(origin).resolve()
        and (module_file is None or module_file == origin),
        'capture.first_party_origin',
    )
    relative = str(filename.resolve().relative_to(root))
    source = checked_path(repo, relative)
    require(source.is_file(), 'capture.first_party_missing')
    raw = source.read_bytes()
    require(
        code in source_codes(raw, code.co_filename),
        'capture.first_party_code_changed',
    )
    return {
        'path': relative,
        'function': code.co_name,
        'line': trace.tb_lineno,
        'source_sha256': sha(raw),
    }


def materialize(
    repo: Path,
    envelope: dict,
    templates: dict | None = None,
    reference_scopes: dict | None = None,
    *,
    postgres_source_factory=None,
    postgres_connection_factory=None,
) -> dict:
    recipe = open_recipe(envelope)
    if recipe['operation'] == 'fixture.reference':
        from references import capture

        return capture(
            repo,
            envelope,
            reference_scopes or {},
            postgres_connection_factory=postgres_connection_factory,
        )
    if recipe['operation'] == 'postgres.carry':
        require(
            callable(postgres_source_factory),
            'postgres_carry.owned_source_required',
        )
    switch, _driver, _request = helpers(repo)
    import _fixture_ops as operations
    import pandas as pd
    import sdm_core._native as native
    import sdm_core.describe as describe
    from sdm_client import (
        Column,
        PredictionResult,
        PredictiveQuery,
        Schema,
        Table,
    )
    from sdm_client.client import FittedModel, SDMClient

    entries = {
        'PredictiveQuery.materialize': PredictiveQuery.materialize,
        'SDMClient.predict': SDMClient.predict,
        'SDMClient.fit': SDMClient.fit,
        'FittedModel.predict': FittedModel.predict,
        'Schema.from_data': Schema.from_data,
        'Schema.from_source': Schema.from_source,
        'Schema.__init__': Schema.__init__,
        'PredictionResult.to_pandas': PredictionResult.to_pandas,
    }
    require(
        recipe['entrypoint'] in entries
        and recipe['operation']
        in {'materialize', 'client.run', 'result.to_pandas', 'postgres.carry'},
        'capture.public_operation',
    )
    require(
        recipe['operation'] != 'materialize'
        or recipe['entrypoint'] == 'PredictiveQuery.materialize',
        'capture.entrypoint_operation',
    )
    require(
        recipe['operation'] != 'result.to_pandas'
        or recipe['entrypoint'] == 'PredictionResult.to_pandas',
        'capture.result_entrypoint',
    )
    entry_code = entries[recipe['entrypoint']].__code__

    selector = os.environ[switch.SELECTOR]
    major = pd.__version__.split('.')[0]
    classifier = switch.PortedCode()
    if major == '2':
        classifier = VettedCatalog(
            classifier,
            repo
            / 'packages/structured-data-models-relational-engine/src/sdm_relational_engine',
            Path(os.environ['P97_LEGACY_COPY_ROOT']),
        )
    native_functions = tuple(
        value for value in vars(native).values() if inspect.isbuiltin(value)
    )
    require(bool(native_functions), 'capture.native_exports')
    descriptor_source = (
        repo / 'rust/bindings/python/python/sdm_core/describe.py'
    )
    require(
        Path(describe.__file__).read_bytes() == descriptor_source.read_bytes(),
        'capture.descriptor_bytes',
    )
    schema, result, error, event = None, None, None, None
    payloads = []
    dimensions = None
    value = recipe['input']
    sampler = __import__(
        'sdm_relational_engine.rfm.base.sampler'
        if selector == 'python'
        else 'sdm_client._request_core',
        fromlist=['_coverage_warned'],
    )
    sampler._coverage_warned = False

    def record_request(request):
        nonlocal payloads, result
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
                        getattr(request, half).instance_table.index.tolist()
                    ),
                }
                for half in ('context', 'predict')
            }

    def operation():
        nonlocal schema, result, error, event, payloads, dimensions
        try:
            with contextlib.ExitStack() as stack:
                if recipe['operation'] == 'postgres.carry':
                    path = (
                        INVESTIGATIONS
                        / 'p97-postgres-carry-staging/postgres_carry.py'
                    )
                    require(
                        sha(path.read_bytes())
                        == 'b277d62744d7e85b3206cf1d240d116455d17840029eeb93059d2c2a030a4839',
                        'capture.postgres_helper_binding',
                    )
                    postgres = load('_p97_postgres_carry', path)
                    require(
                        recipe['input']['carry_case_id'] in postgres.recipes()
                        and same_json(
                            recipe,
                            postgres.recipes()[
                                recipe['input']['carry_case_id']
                            ],
                        ),
                        'capture.postgres_recipe',
                    )
                    schema, request = postgres.operation(
                        value,
                        stack,
                        postgres_source_factory=postgres_source_factory,
                    )
                    if request is not None:
                        record_request(request)
                    else:
                        result = {'schema_constructed': True}
                    return
                if recipe['operation'] == 'result.to_pandas':
                    from sdm_api.generated.tfm_api import PredictionItem

                    arguments = operations.decode(value)
                    arguments['items'] = [
                        PredictionItem(**item) for item in arguments['items']
                    ]
                    answer = PredictionResult(**arguments)
                    frame = answer.to_pandas()

                    def cell(value):
                        import numpy as numpy

                        if isinstance(value, numpy.complexfloating):
                            return {
                                '$numpy_complex': {
                                    'dtype': value.dtype.str,
                                    'bytes': value.tobytes().hex(),
                                }
                            }
                        if isinstance(value, complex):
                            return {
                                '$complex': [
                                    operations.encode(value.real),
                                    operations.encode(value.imag),
                                ]
                            }
                        return operations.encode(value)

                    result = {
                        'frame': {
                            'columns': [
                                operations.encode(name)
                                for name in frame.columns
                            ],
                            'dtypes': [str(dtype) for dtype in frame.dtypes],
                            'index': [
                                operations.encode(item) for item in frame.index
                            ],
                            'values': [
                                [
                                    cell(frame.iloc[row, column])
                                    for column in range(len(frame.columns))
                                ]
                                for row in range(len(frame))
                            ],
                        }
                    }
                    return
                if recipe['operation'] == 'client.run':
                    frames = {
                        name: operations.build_frame(spec)
                        for name, spec in value['tables'].items()
                    }
                    session = operations.Answering(value.get('responses', []))

                    class OwnedFlow(operations.Flow):
                        def _Schema_from_duckdb(self, load=None, **args):
                            from _parity.runner import _duckdb
                            from sdm_connectors import DuckDBSource

                            selected = (
                                self.tables
                                if load is None
                                else {name: self.tables[name] for name in load}
                            )
                            connection = _duckdb(selected, {})
                            stack.callback(connection.close)
                            if args.get('relationships') is not None:
                                args['relationships'] = (
                                    operations._relationships(
                                        args['relationships']
                                    )
                                )
                            self.schema = Schema.from_source(
                                DuckDBSource(connection), **args
                            )

                    flow = OwnedFlow(frames, session)
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
                if 'explicit_dtype' in value:
                    require(
                        value['explicit_dtype']
                        in {'arrow/json', 'arrow/uuid', 'arrow/bool8'}
                        and value['backend'] == 'memory',
                        'capture.explicit_dtype',
                    )
                    probe = load(
                        '_p97_dtype_shapes',
                        INVESTIGATIONS
                        / 'p97-base-contract-staging/probe_dtypes.py',
                    )
                    builders = dict(probe.cases())
                    frame = pd.DataFrame(
                        {
                            'row_id': [0, 1, 2],
                            'target': [1.0, 2.0, 3.0],
                            'feature': builders[value['explicit_dtype']](),
                        }
                    )
                    table = Table(
                        'public_table',
                        frame.assign(feature=pd.Series(['one', None, 'two'])),
                        {
                            'row_id': Column('row_id', 'int64', 'ID'),
                            'target': Column('target', 'float64', 'numerical'),
                            'feature': Column(
                                'feature', 'string', 'categorical'
                            ),
                        },
                        primary_key='row_id',
                    )
                    schema = Schema({'public_table': table})
                    schema['public_table'].data = frame
                else:
                    if 'tables' in value:
                        frames = {
                            name: operations.build_frame(spec)
                            for name, spec in value['tables'].items()
                        }
                    else:
                        from _parity import data as datagen

                        generated = datagen.generate(
                            value['data']['generator'], value['data']['seed']
                        )
                        frames = {
                            name: generated[name]
                            for name in value['table_names']
                        }
                    declared = value.get('constructor') == 'declared_table_data'
                    if 'constructor' in value:
                        require(
                            declared
                            and value['backend'] == 'memory'
                            and 'matrix' in recipe,
                            'capture.declared_constructor',
                        )
                    if not declared:
                        dimensions = construct(
                            recipe, templates or {}, frames, operations
                        )
                    options = {
                        'primary_keys': operations._keys(
                            value.get('primary_keys')
                        ),
                        'time_columns': value.get('time_columns'),
                        'relationships': operations._relationships(
                            value.get('relationships')
                        ),
                    }

                    if declared:
                        schema = Schema.from_data(frames, **options)
                        dimensions = construct(
                            recipe, templates or {}, frames, operations
                        )
                    backend = value['backend']
                    require(
                        backend in {'memory', 'sqlite', 'duckdb'},
                        'capture.offline_backend',
                    )
                    if backend == 'memory':
                        if declared:
                            for name, frame in frames.items():
                                schema[name].data = frame
                        else:
                            schema = Schema.from_data(frames, **options)
                    else:
                        from _parity.runner import _duckdb, _source, _sqlite

                        spec = value.get('database', {})
                        if backend == 'duckdb':
                            connection = _duckdb(frames, spec)
                        else:
                            directory = Path(
                                stack.enter_context(
                                    tempfile.TemporaryDirectory()
                                )
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
                record_request(request)
        except Rejected:
            raise
        except Exception as raised:
            error = diagnostic(raised, operations.encode)
            frames = []
            trace = raised.__traceback__
            inside_public = False
            descriptor_seen = False
            while trace is not None:
                code = trace.tb_frame.f_code
                if code is entry_code:
                    inside_public = True
                if inside_public and code is describe.describe_frames.__code__:
                    descriptor_seen = True
                if inside_public:
                    origin = source_frame(repo, trace)
                    if origin is not None:
                        frames.append(origin)
                trace = trace.tb_next
            if inside_public and descriptor_seen:
                event = {
                    'input_sha256': envelope['input_sha256'],
                    'entrypoint': recipe['entrypoint'],
                    'frames': frames,
                    'exception': error,
                    'pandas_major': major,
                    'native_sha256': sha(Path(native.__file__).read_bytes()),
                    'descriptor_sha256': sha(descriptor_source.read_bytes()),
                    'descriptor_code_observed': True,
                    'native_boundary_calls': [],
                }

    with (
        warnings.catch_warnings(record=True) as caught,
        observe_sql(switch) as sql,
        (
            observe_inline_old(
                repo, entries[recipe['entrypoint']], envelope['input_sha256']
            )
            if recipe['operation'] == 'result.to_pandas'
            else contextlib.nullcontext([])
        ) as legacy_branch_events,
    ):
        warnings.simplefilter('always')
        observer = schema_boundary.observer(
            repo, envelope, Schema.from_source, observe_public_call
        )
        _ignored, counts = observer(
            operation,
            entry_code,
            classifier.symbol,
            native_functions,
        )
    if isinstance(classifier, VettedCatalog):
        classifier.verify()
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
    preparation = counts.pop('_schema_preparation', None)
    if preparation is not None and counts['public_calls'] == 0:
        return schema_boundary.refused_record(
            envelope,
            counts,
            preparation,
            observed,
            selector,
            major,
            sha(Path(native.__file__).read_bytes()),
            getattr(classifier, 'copied_calls', 0),
            dimensions,
        )
    require(counts['public_calls'] > 0, 'capture.no_public_entry')
    require(selector in {'python', 'core'}, 'capture.selector')
    if selector == 'python':
        require(
            (counts['legacy_calls'] > 0 or bool(legacy_branch_events))
            and counts['native_calls'] == 0,
            'capture.old_path',
        )
    else:
        require(
            counts['legacy_calls'] == 0 and not legacy_branch_events,
            'capture.core_legacy',
        )
        if counts['native_calls'] == 0:
            require(
                observed['status'] == 'refused' and event is not None,
                'capture.empty_boundary',
            )
    return {
        'observation': observed,
        **counts,
        'selector': selector,
        'pandas_major': major,
        'materialized_rows': materialized_rows(observed),
        'native_sha256': sha(Path(native.__file__).read_bytes()),
        'copied_legacy_calls': getattr(classifier, 'copied_calls', 0),
        'host_event': event
        if selector == 'core' and counts['native_calls'] == 0
        else None,
        'dimensions': dimensions,
        'legacy_branch_events': legacy_branch_events,
    }
