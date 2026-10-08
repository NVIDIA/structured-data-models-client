# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Input-only semantic source capture for an authenticated fresh core worker."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import time
import warnings
from pathlib import Path


def collect(root, value, asset_sink):
    if os.environ.get('SDM_CLIENT_IMPLEMENTATION') != 'core':
        raise ValueError('semantic input capture requires a fresh core worker')
    import sdm_core._native as native
    from generator_inputs import typed
    from generator_oracles import OriginalCalls, source_codes
    from generator_primitives import module
    from sdm_client import Schema, _core, _impl, _schema_core
    from sdm_core.describe import timestamp_series_units
    from support import INVESTIGATIONS, require

    require(_impl.OLD is False, 'semantic.capture_core_selector')

    sys.path.insert(0, str(INVESTIGATIONS / 'p97-real-adapter-staging'))
    import _fixture_ops as operations
    from common import diagnostic

    original = typed(value)
    original_digest = hashlib.sha256(
        json.dumps(
            original,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    consumer = module(root, 'tests/rust_core/schema/accept_views.py')
    request = module(root, 'tests/rust_core/request/_request.py')
    authenticated = {}
    for function in (
        consumer._python_semantic,
        consumer.python_outcome,
        consumer.source,
        consumer.Warehouse.answer,
        Schema.from_source,
        timestamp_series_units,
        request.Recorder.install,
        request.Recorder._wrap,
        request.Recorder.host_call,
        request.Guard.install,
        request.Guard._event,
        request.Encoder.value,
        request.Encoder.outcome,
        _schema_core.from_source,
    ):
        code = inspect.unwrap(function).__code__
        path = Path(code.co_filename).resolve()
        canonical = (
            root / 'rust/bindings/python/python/sdm_core/describe.py'
            if function is timestamp_series_units
            else path
        )
        require(
            canonical.is_relative_to(root.resolve()),
            'semantic.capture_source_path',
        )
        raw = path.read_bytes()
        require(raw == canonical.read_bytes(), 'semantic.capture_helper_source')
        require(
            code in source_codes(raw, code.co_filename),
            'semantic.capture_source_code',
        )
        authenticated[path] = hashlib.sha256(raw).hexdigest()
        authenticated[canonical] = hashlib.sha256(raw).hexdigest()
    require(
        inspect.isbuiltin(native.schema_source_run)
        and native.schema_source_run.__name__ == 'schema_source_run',
        'semantic.capture_native_function',
    )
    arguments = (
        value['definition'],
        {tuple(item['identity']): item['columns'] for item in value['tables']},
        value['trust_expressions'],
    )
    observer = OriginalCalls(root)

    def observe(invoke):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            actual = invoke(consumer._python_semantic, *arguments)
        return {
            'status': 'success',
            'schema': [],
            'payloads': [],
            'sql': [],
            'result': typed(actual),
            'error': None,
            'warnings': [
                diagnostic(item.message, operations.encode) for item in caught
            ],
        }

    independent = observe(observer.invoke)
    require(
        len(observer.events) == 1
        and observer.events[0]['operation_calls'] > 0
        and observer.events[0]['native_calls'] > 0,
        'semantic.capture_native_entry',
    )
    recorder = request.Recorder(native)
    recorder.install()
    recorder._patch(
        _core, 'describe_frames', recorder._described(_core.describe_frames)
    )
    original_from_source = _schema_core.from_source

    def recorded_source(*args, **kwargs):
        with recorder.host_call('from_source', given_frames=False):
            return original_from_source(*args, **kwargs)

    recorder._patch(_schema_core, 'from_source', recorded_source)
    try:
        observed = observe(lambda function, *args: function(*args))
    finally:
        recorder.uninstall()
    assert observed == independent
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in authenticated.items()
    )
    assert typed(value) == original
    assert os.environ.get('SDM_CLIENT_IMPLEMENTATION') == 'core'
    assert _impl.OLD is False
    assert not recorder.problems
    initial = [
        call
        for call in recorder.calls
        if call['function'] == 'schema_source_run'
    ]
    assert len(initial) == 1 and not initial[0]['kwargs']
    encoder = request.Encoder(Path(asset_sink))
    queries = []
    for step in initial[0].get('steps', []):
        if step['kind'] == 'run_queries':
            for query, answer in zip(
                step['queries'], step['outcomes'], strict=True
            ):
                queries.append(
                    {
                        'sql': query['sql'],
                        'params': query['params'],
                        'outcome': encoder.outcome(answer),
                    }
                )
        else:
            assert step['kind'] in ('done', 'raised')
    now = time.localtime()
    measured_units = timestamp_series_units()
    witness = {
        'host': {
            'year': now.tm_year,
            'today': [now.tm_year, now.tm_mon, now.tm_mday],
            'pads_year': time.strftime('%Y', (1, 1, 1, 0, 0, 0, 0, 1, -1))
            == '0001',
            'tznames': list(time.tzname),
            'timestamp_series_units': (
                ['s', 'ms', 'us', 'ns']
                if measured_units is None
                else measured_units
            ),
        },
        'external': {
            'original_input_sha256': original_digest,
            'initial_schema': {
                'function': initial[0]['function'],
                'args': [encoder.value(item) for item in initial[0]['args']],
            },
            'queries': queries,
        },
    }
    assets = {}

    def dependencies(item):
        if isinstance(item, dict):
            if len(item) == 1:
                tag, identity = next(iter(item.items()))
                if tag in {'$table', '$batch', '$array', '$chunked'}:
                    relative = 'tables/' + identity + '.arrow'
                    raw = (Path(asset_sink) / relative).read_bytes()
                    assert hashlib.sha256(raw).hexdigest() == identity
                    assets[relative] = raw.hex()
            for child in item.values():
                dependencies(child)
        elif isinstance(item, list):
            for child in item:
                dependencies(child)

    dependencies(witness['external'])
    return {
        'host_inputs': witness,
        'observation': observed,
        'source_consumer_events': observer.events,
        'sources': {
            path.relative_to(root.resolve()).as_posix(): digest
            for path, digest in authenticated.items()
            if path.is_relative_to(root.resolve())
        },
        'assets': assets,
    }
