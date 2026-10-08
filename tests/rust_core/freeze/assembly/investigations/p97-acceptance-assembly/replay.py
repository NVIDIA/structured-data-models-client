# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute fixed replay routes once per bounded immutable snapshot."""

from __future__ import annotations

import tempfile
from pathlib import Path

from execution import batch
from native_binding import verify_no_python
from support import HERE, json_bytes, require, same_json, sha, unique_json

MAX_SNAPSHOT = 512 * 1024 * 1024
MAX_RESULT = 128 * 1024 * 1024


def execute_snapshot(
    root: Path,
    contents: dict,
    routes: dict,
    *,
    route: str,
    python: Path,
    binary: Path | None,
    native_sha256: str,
    no_python_binding: dict | None,
    review: dict | None,
) -> dict:
    require(
        route in {'rust', 'binding'} and bool(contents) and bool(routes),
        'replay.route_input',
    )
    before = {name: sha(value) for name, value in contents.items()}
    request = {
        'files': {name: value.hex() for name, value in contents.items()},
        'cases': list(routes),
        'routes': routes,
        'native_sha256': native_sha256,
    }
    raw = json_bytes(request)
    require(len(raw) <= MAX_SNAPSHOT, 'replay.snapshot_bound')
    trusted_no_python = None
    if route == 'rust':
        require(
            binary is not None
            and no_python_binding is not None
            and review is not None,
            'replay.rust_binding_absent',
        )
        trusted_no_python = verify_no_python(
            root, binary, no_python_binding, review
        )
        command = [str(binary), '--format', 'p97-snapshot-v1']
    else:
        require(
            python is not None and python.is_file(),
            'replay.binding_interpreter_absent',
        )
        if any(
            binding['kind'] == 'native_semantic' for binding in routes.values()
        ):
            require(
                (
                    root / 'rust/bindings/python/python/sdm_core/replay.py'
                ).is_file(),
                'replay.binding_target_absent',
            )
        command = [str(python), '-B', str(HERE / 'replay_worker.py'), str(root)]
    with tempfile.TemporaryDirectory(prefix='replay-', dir=HERE) as temporary:
        scratch = Path(temporary)
        for name in ('home', 'tmp', 'cache'):
            (scratch / name).mkdir()
        snapshot = scratch / 'snapshot.json'
        snapshot.write_bytes(raw)
        environment = {
            'HOME': str(scratch / 'home'),
            'TMPDIR': str(scratch / 'tmp'),
            'XDG_CACHE_HOME': str(scratch / 'cache'),
            'PATH': '/usr/bin:/bin',
            'LANG': 'en_US.UTF-8',
            'PYTHONDONTWRITEBYTECODE': '1',
            'PYTHONNOUSERSITE': '1',
            'SDM_CLIENT_IMPLEMENTATION': 'core',
            'P97_SNAPSHOT': str(snapshot),
        }
        pid, output, elapsed = batch.bounded(
            [
                '/bin/sh',
                '-c',
                'exec "$@" < "$P97_SNAPSHOT"',
                'p97-replay',
                *command,
            ],
            b'',
            environment,
            root,
            deadline=1200,
            output_bound=MAX_RESULT,
        )
        require(snapshot.read_bytes() == raw, 'replay.snapshot_file_mutated')
        result = unique_json(output)
        require(type(result) is dict, 'replay.result')
        if route == 'rust':
            require(
                'no_python_binding' not in result
                and 'python_processes' not in result,
                'replay.product_supplied_process_claim',
            )
            result['no_python_binding'] = trusted_no_python
    require(not scratch.exists(), 'replay.cleanup')
    require(
        {name: sha(value) for name, value in contents.items()} == before,
        'replay.input_mutated',
    )
    return {
        'process_id': pid,
        'elapsed_seconds': elapsed,
        'route': route,
        'input_sha256': sha(raw),
        'result': result,
    }


def snapshot_partitions(contents, routes, manifest, native_sha256, limit):
    records = {record['path']: record for record in manifest}
    require(len(records) == len(manifest), 'replay.manifest_duplicate')
    require(set(contents) <= set(records), 'replay.unlisted_file')
    owners = {}
    for name, record in records.items():
        for identity in record['semantic_cases']:
            require(identity not in owners, 'replay.duplicate_owner')
            owners[identity] = name
    require(set(owners) == set(routes), 'replay.partition_case_union')

    def closure(names):
        selected = set()
        pending = list(names)
        while pending:
            name = pending.pop()
            require(name in records, 'replay.unknown_dependency')
            if name not in selected:
                selected.add(name)
                pending.extend(records[name]['depends_on'])
        return selected

    covered = closure(owners.values())
    extra = set(records) - covered
    partitions = []
    selected_routes = {}
    selected_files = closure(extra)

    def size(files, chosen):
        request = {
            'files': {name: '' for name in sorted(files) if name in contents},
            'cases': list(chosen),
            'routes': chosen,
            'native_sha256': native_sha256,
        }
        return len(json_bytes(request)) + sum(
            2 * len(contents[name]) for name in files if name in contents
        )

    for identity, route in routes.items():
        needed = closure([owners[identity], *route.get('files', [])])
        proposed_files = selected_files | needed
        proposed_routes = {**selected_routes, identity: route}
        if selected_routes and (
            len(proposed_routes) > 32
            or size(proposed_files, proposed_routes) > limit
        ):
            partitions.append((selected_files, selected_routes))
            selected_files, selected_routes = set(), {}
            proposed_files, proposed_routes = needed, {identity: route}
        require(
            size(proposed_files, proposed_routes) <= limit,
            'replay.snapshot_bound',
        )
        selected_files, selected_routes = proposed_files, proposed_routes
    require(bool(selected_routes), 'replay.empty_partition')
    partitions.append((selected_files, selected_routes))
    identities = [identity for _, chosen in partitions for identity in chosen]
    require(identities == list(routes), 'replay.partition_order')
    require(
        set().union(*(files for files, _ in partitions)) == set(records),
        'replay.partition_files',
    )
    return [
        (
            {
                name: contents[name]
                for name in sorted(files)
                if name in contents
            },
            chosen,
        )
        for files, chosen in partitions
    ]


def execute_snapshots(root, contents, routes, *, manifest, **options):
    partitions = snapshot_partitions(
        contents, routes, manifest, options['native_sha256'], MAX_SNAPSHOT
    )
    executions = []
    merged = None
    verified = set()
    for files, chosen in partitions:
        execution = execute_snapshot(root, files, chosen, **options)
        result = execution['result']
        native = {
            key
            for key, value in chosen.items()
            if value['kind'] == 'native_semantic'
        }
        host = set(chosen) - native
        require(
            all(
                value['kind'] in {'native_semantic', 'public_host_refusal'}
                for value in chosen.values()
            ),
            'replay.partition_kind',
        )
        require(
            type(result.get('files_verified')) is list
            and len(result['files_verified'])
            == len(set(result['files_verified']))
            and set(result['files_verified']) == set(files),
            'replay.partition_verified',
        )
        if options['route'] == 'binding':
            fields = {
                'observations',
                'native_calls',
                'case_native_calls',
                'host_events',
                'files_verified',
                'oracle',
            }
            mappings = {
                'observations': set(chosen),
                'case_native_calls': set(chosen),
                'host_events': host,
            }
            constant = 'oracle'
            counts = result.get('case_native_calls', {})
            require(
                all(
                    type(value) is int and value >= 0
                    for value in counts.values()
                )
                and type(result.get('native_calls')) is int
                and result['native_calls'] == sum(counts.values()),
                'replay.partition_counts',
            )
            require(
                all(counts.get(key, 0) > 0 for key in native)
                and all(counts.get(key) == 0 for key in host),
                'replay.partition_native',
            )
        else:
            fields = {
                'observations',
                'host_dispositions',
                'files_verified',
                'no_python_binding',
            }
            mappings = {'observations': native, 'host_dispositions': host}
            constant = 'no_python_binding'
        require(set(result) == fields, 'replay.partition_result_fields')
        for field, identities in mappings.items():
            require(
                type(result[field]) is dict
                and set(result[field]) == identities,
                'replay.partition_result_cases',
            )
        if merged is None:
            merged = {field: {} for field in mappings}
            merged[constant] = result[constant]
            if options['route'] == 'binding':
                merged['native_calls'] = 0
        require(
            same_json(merged[constant], result[constant]),
            'replay.partition_authority',
        )
        for field in mappings:
            require(
                not set(merged[field]) & set(result[field]),
                'replay.partition_duplicate_result',
            )
            merged[field].update(result[field])
        if options['route'] == 'binding':
            merged['native_calls'] += result['native_calls']
        verified.update(result['files_verified'])
        executions.append(
            {
                **{
                    key: value
                    for key, value in execution.items()
                    if key != 'result'
                },
                'cases': list(chosen),
                'files': {name: sha(raw) for name, raw in files.items()},
            }
        )
    require(verified == set(contents), 'replay.partition_all_bytes')
    merged['files_verified'] = sorted(verified)
    return {
        'route': options['route'],
        'executions': executions,
        'elapsed_seconds': sum(
            entry['elapsed_seconds'] for entry in executions
        ),
        'result': merged,
    }
