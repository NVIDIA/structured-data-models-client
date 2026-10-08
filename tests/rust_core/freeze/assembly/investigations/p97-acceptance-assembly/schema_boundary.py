# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact earlier public Schema refusal, never a materialization observation."""

from __future__ import annotations

import sys
from pathlib import Path

from recipes import open_recipe
from support import contract, require, same_json, sha

SOURCE = 'packages/structured-data-models-client/src/sdm_client/schema.py'
SOURCE_SHA = '153b6209ffbef017818148f10e4a78b1780856e6677be7aae94147611104d24a'


def eligible(recipe):
    matrix, value = recipe.get('matrix', {}), recipe.get('input', {})
    backend = value.get('backend')
    return (
        recipe.get('operation') == 'materialize'
        and recipe.get('entrypoint') == 'PredictiveQuery.materialize'
        and (backend, matrix.get('dtype'))
        in {('duckdb', 'pandas/Int64'), ('sqlite', 'numpy/int64')}
        and matrix.get('template')
        in {
            backend + '-time_column-' + task
            for task in (
                'classification',
                'regression',
                'ranking',
                'link_prediction',
            )
        }
        and value.get('time_columns') == {'orders': 'ordered_at'}
    )


def observer(root, envelope, schema_function, original):
    if not eligible(open_recipe(envelope)):
        return original
    code = schema_function.__code__
    require(
        Path(code.co_filename).resolve() == (root / SOURCE).resolve()
        and sha((root / SOURCE).read_bytes()) == SOURCE_SHA
        and code.co_name == 'from_source',
        'schema_stage.source',
    )

    def observe(operation, requested_code, legacy_symbol, native_functions):
        require(sys.getprofile() is None, 'coupling.profile_already_active')
        requested = {'public_calls': 0, 'legacy_calls': 0, 'native_calls': 0}
        preparation = dict(requested)
        codes, counters, depths = (
            (requested_code, code),
            (requested, preparation),
            [0, 0],
        )

        def profile(frame, event, argument):
            for index, target in enumerate(codes):
                if event == 'call' and frame.f_code is target:
                    depths[index] += 1
                    counters[index]['public_calls'] += 1
            legacy = (
                event == 'call' and any(depths) and legacy_symbol(frame.f_code)
            )
            native = event == 'c_call' and any(
                argument is function for function in native_functions
            )
            for index, target in enumerate(codes):
                if depths[index]:
                    counters[index]['legacy_calls'] += bool(legacy)
                    counters[index]['native_calls'] += bool(native)
                if event == 'return' and frame.f_code is target:
                    depths[index] -= 1

        sys.setprofile(profile)
        try:
            result = operation()
        finally:
            sys.setprofile(None)
        requested['_schema_preparation'] = {
            'recipe': envelope,
            'entrypoint': 'Schema.from_source',
            'source': SOURCE,
            'source_sha256': SOURCE_SHA,
            'counts': preparation,
        }
        return result, requested

    return observe


def validate(envelope, record):
    recipe = open_recipe(envelope)
    require(eligible(recipe), 'schema_stage.scope')
    stage = record.get('schema_preparation')
    require(
        type(stage) is dict
        and set(stage)
        == {
            'recipe',
            'entrypoint',
            'source',
            'source_sha256',
            'counts',
        }
        and same_json(stage['recipe'], envelope)
        and stage['entrypoint'] == 'Schema.from_source'
        and stage['source'] == SOURCE
        and stage['source_sha256'] == SOURCE_SHA,
        'schema_stage.binding',
    )
    require(
        all(
            type(record[key]) is int and record[key] == 0
            for key in (
                'public_calls',
                'legacy_calls',
                'native_calls',
                'materialized_rows',
            )
        )
        and record['host_event'] is None
        and record.get('legacy_branch_events', []) == [],
        'schema_stage.request_not_entered',
    )
    observed = record['observation']
    contract.observation(observed)
    require(
        observed['status'] == 'refused'
        and observed['schema'] == []
        and observed['payloads'] == []
        and observed['result'] is None
        and observed['warnings'] == []
        and bool(observed['sql'])
        and observed['error']
        == {
            'class': 'sdm_client.errors.StructuredDataError',
            'code': 'INVALID_REQUEST',
            'details': {},
            'message': "column 'ordered_at' holds 'int64', which cannot mean 'timestamp'",
        },
        'schema_stage.exact_refusal',
    )
    counts = stage['counts']
    require(
        type(counts) is dict
        and set(counts) == {'public_calls', 'legacy_calls', 'native_calls'}
        and all(type(value) is int and value >= 0 for value in counts.values())
        and counts['public_calls'] == 1,
        'schema_stage.counts',
    )
    require(record['selector'] in {'python', 'core'}, 'schema_stage.selector')
    if record['selector'] == 'python':
        require(
            counts['legacy_calls'] > 0 and counts['native_calls'] == 0,
            'schema_stage.original',
        )
        if record['pandas_major'] == '2':
            require(
                type(record['copied_legacy_calls']) is int
                and record['copied_legacy_calls'] > 0,
                'schema_stage.copied_original',
            )
    else:
        require(
            counts['legacy_calls'] == 0 and counts['native_calls'] > 0,
            'schema_stage.native',
        )
    dimensions = record['dimensions']
    require(
        type(dimensions) is dict
        and dimensions['dtype'] == recipe['matrix']['dtype']
        and dimensions['template'] == recipe['matrix']['template']
        and dimensions['backend'] == recipe['input']['backend']
        and dimensions['role'] == 'time_column'
        and dimensions['table'] == 'orders'
        and dimensions['column'] == 'ordered_at',
        'schema_stage.dimensions',
    )


def validate_route(binding, record, event):
    stage = record['schema_preparation']
    validate(stage['recipe'], record)
    require(
        binding['kind'] == 'native_semantic'
        and binding['entrypoint'] == 'PredictiveQuery.materialize'
        and binding['input_sha256'] == stage['recipe']['input_sha256']
        and record['selector'] == 'core'
        and event is None,
        'schema_stage.route',
    )


def refused_record(
    envelope,
    counts,
    preparation,
    observed,
    selector,
    major,
    native_sha,
    copied_calls,
    dimensions,
):
    record = {
        'observation': observed,
        **counts,
        'selector': selector,
        'pandas_major': major,
        'materialized_rows': 0,
        'native_sha256': native_sha,
        'copied_legacy_calls': copied_calls,
        'host_event': None,
        'dimensions': dimensions,
        'legacy_branch_events': [],
        'schema_preparation': preparation,
    }
    validate(envelope, record)
    return record
