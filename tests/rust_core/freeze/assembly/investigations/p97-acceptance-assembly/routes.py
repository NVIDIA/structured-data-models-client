# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DV85/DV86 disjoint native computation and genuine empty-boundary checks."""

from __future__ import annotations

import schema_boundary
from support import contract, require, same_json


def validate_host_event(binding: dict, event: dict, error: dict):
    require(
        type(event) is dict
        and set(event)
        == {
            'input_sha256',
            'entrypoint',
            'frames',
            'exception',
            'pandas_major',
            'native_sha256',
            'descriptor_sha256',
            'descriptor_code_observed',
            'native_boundary_calls',
        },
        'route.event_fields',
    )
    host = binding['host_binding']
    require(
        type(host) is dict
        and set(host)
        == {'frames', 'pandas_major', 'native_sha256', 'descriptor_sha256'},
        'route.host_binding',
    )
    require(
        all(same_json(event[key], value) for key, value in host.items())
        and bool(event['frames'])
        and event['descriptor_code_observed'] is True
        and event['input_sha256'] == binding['input_sha256']
        and event['entrypoint'] == binding['entrypoint']
        and event['native_boundary_calls'] == []
        and same_json(event['exception'], error),
        'route.host_event',
    )


def validate_capture_route(binding: dict, record: dict, event: dict | None):
    if record.get('schema_preparation') is not None:
        schema_boundary.validate_route(binding, record, event)
        return
    require(
        type(record['public_calls']) is int and record['public_calls'] > 0,
        'route.public_entry',
    )
    require(
        type(record['native_calls']) is int
        and record['native_calls'] >= 0
        and type(record['legacy_calls']) is int
        and record['legacy_calls'] == 0,
        'route.counts',
    )
    contract.observation(record['observation'])
    if binding['kind'] == 'native_semantic':
        require(
            record['native_calls'] > 0 and event is None,
            'route.native_required',
        )
        return
    require(binding['kind'] == 'public_host_refusal', 'route.kind')
    require(
        record['native_calls'] == 0
        and record['observation']['status'] == 'refused',
        'route.host_refusal',
    )
    validate_host_event(binding, event, record['observation']['error'])


def replay_join(
    expected: dict,
    routes: dict,
    binding: dict,
    rust: dict,
    files: dict[str, bytes],
) -> None:
    require(set(routes) == set(expected), 'replay.locked_routes')
    require(
        set(binding)
        == {
            'observations',
            'native_calls',
            'case_native_calls',
            'host_events',
            'files_verified',
            'oracle',
        },
        'replay.binding_fields',
    )
    require(
        set(binding['observations']) == set(expected)
        and same_json(binding['observations'], expected),
        'replay.binding_expected',
    )
    require(
        set(binding['case_native_calls']) == set(expected), 'replay.case_counts'
    )
    counts = binding['case_native_calls']
    require(
        all(type(count) is int and count >= 0 for count in counts.values())
        and type(binding['native_calls']) is int
        and binding['native_calls'] == sum(counts.values()),
        'replay.true_sum',
    )
    native = {
        key
        for key, value in routes.items()
        if value['kind'] == 'native_semantic'
    }
    host = {
        key
        for key, value in routes.items()
        if value['kind'] == 'public_host_refusal'
    }
    require(
        native.isdisjoint(host) and native | host == set(expected),
        'replay.partition',
    )
    require(
        all(counts[key] > 0 for key in native)
        and all(counts[key] == 0 for key in host),
        'replay.boundary_counts',
    )
    require(set(binding['host_events']) == host, 'replay.host_events')
    require(
        set(rust)
        == {
            'observations',
            'host_dispositions',
            'files_verified',
            'no_python_binding',
        },
        'replay.rust_fields',
    )
    require(
        same_json(rust['observations'], {key: expected[key] for key in native})
        and set(rust['host_dispositions']) == host,
        'replay.rust_partition',
    )
    require(
        type(rust['no_python_binding']) is dict
        and rust['no_python_binding']['basis']
        == 'source-derived, not runtime descendant census'
        and bool(rust['no_python_binding']['source_bindings']),
        'replay.rust_source_binding',
    )
    require(
        binding['oracle']
        == {
            'absent_modules': [
                'predictive_query',
                'relationallib',
                'sdm_relational_engine',
            ],
            'absent_distributions': [
                'sdm-pql',
                'sdm-relational-engine',
                'structured-data-models-pql',
                'structured-data-models-relational-engine',
            ],
        },
        'replay.oracle',
    )
    for record in (binding, rust):
        require(
            type(record['files_verified']) is list
            and len(record['files_verified'])
            == len(set(record['files_verified']))
            and set(record['files_verified']) == set(files),
            'replay.all_bytes',
        )
    for identity in host:
        route = routes[identity]
        event = binding['host_events'][identity]
        require(
            expected[identity]['family'] == 'public-observation-v1'
            and expected[identity]['observation']['status'] == 'refused',
            'replay.host_binding',
        )
        validate_host_event(
            route, event, expected[identity]['observation']['error']
        )
        disposition = rust['host_dispositions'][identity]
        require(
            disposition
            == {
                'case_id': identity,
                'disposition': 'no_native_boundary_reached',
                'input_sha256': route['input_sha256'],
                'host_event_sha256': contract.digest(event),
                'native_boundary_calls': [],
                'files_verified': route['files'],
            },
            'replay.host_disposition',
        )
        require(set(route['files']) <= set(files), 'replay.host_dependencies')
