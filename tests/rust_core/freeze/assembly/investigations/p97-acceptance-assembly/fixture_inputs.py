# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-derived dependency dispositions and exact zero-operation recordings."""

from pathlib import Path, PurePosixPath

from support import require, sha, unique_json


def empty_boundary(name, raw):
    scope = unique_json(
        Path(__file__).with_name('empty-boundaries.json').read_bytes()
    )
    if name not in scope:
        return False
    binding = scope[name]
    require(sha(raw) == binding['sha256'], 'manifest.empty_boundary_changed')
    require(
        unique_json(raw)
        == {'format': 1, 'scenario': binding['scenario'], 'calls': []},
        'manifest.empty_boundary_shape',
    )
    return True


def walk(value):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def dispositions(records, contents):
    edges = {name: set() for name in records}
    for name, raw in contents.items():
        if not name.endswith('.json'):
            continue
        parent = PurePosixPath(name).parent
        document = unique_json(raw)
        public = (
            isinstance(document, dict)
            and document.get('format') == 'p97-public-v2'
        )
        if public:
            require(
                name.startswith('tests/parity/frozen-public/'),
                'manifest.public_owner_path',
            )
            for case in document['cases']:
                used = {
                    str(parent / 'tables' / f'{value[tag]}.arrow')
                    for value in walk(case['replay_input'])
                    if isinstance(value, dict) and len(value) == 1
                    for tag in ('$table', '$batch', '$array', '$chunked')
                    if tag in value
                    and isinstance(value[tag], str)
                    and len(value[tag]) == 64
                }
                require(
                    used <= set(case['depends_on']),
                    'manifest.public_case_dependency',
                )
        values = (
            [case['replay_input'] for case in document['cases']]
            if public
            else [document]
        )
        for value in walk(values):
            dependency = None
            if (
                not public
                and isinstance(value, str)
                and value.endswith(('.arrow', '.pkl'))
            ):
                dependency = str(parent / value)
            if isinstance(value, dict) and len(value) == 1:
                tag, body = next(iter(value.items()))
                if (
                    tag in {'$table', '$batch', '$array', '$chunked'}
                    and isinstance(body, str)
                    and len(body) == 64
                ):
                    dependency = (
                        str(parent / 'tables' / f'{body}.arrow')
                        if public
                        else f'tests/parity/requests/tables/{body}.arrow'
                    )
                    require(
                        dependency in contents
                        and sha(contents[dependency]) == body,
                        'manifest.content_address',
                    )
            if dependency is not None:
                require(
                    dependency in records, 'manifest.actual_dependency_missing'
                )
                edges[name].add(dependency)
        if public:
            declared = {
                dependency
                for case in document['cases']
                for dependency in case['depends_on']
            }
            require(declared == edges[name], 'manifest.public_dependency_union')
    for name, record in records.items():
        expected = set(edges[name])
        if name.endswith('.pkl') and record['artifact_assertion'] is not None:
            expected.add('tests/rust_core/switch/pickles/expected.json')
        require(
            set(record['depends_on']) == expected,
            'manifest.actual_dependency_graph',
        )
    reachable = set()
    pending = [
        name for name, record in records.items() if record['semantic_cases']
    ]
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        reachable.add(name)
        pending.extend(edges[name])
    allowed = set()
    for name, record in records.items():
        if (
            name.endswith('.arrow')
            and not record['semantic_cases']
            and not record['depends_on']
        ):
            require(
                record['kind'] == 'contract'
                and record['artifact_assertion'] is None,
                'manifest.arrow_disposition',
            )
            require(name in reachable, 'manifest.unused_arrow')
            allowed.add(name)
        elif empty_boundary(name, contents[name]):
            require(
                record['kind'] == 'contract'
                and record['semantic_cases'] == []
                and record['depends_on'] == []
                and record['artifact_assertion'] is None,
                'manifest.empty_boundary_disposition',
            )
            allowed.add(name)
    return allowed
