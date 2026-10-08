# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run independently bound core input capture beside unchanged original workers."""

import tempfile
from pathlib import Path

from candidate import context, lane
from execution import batch, source
from generator_semantic import FAMILY, validate
from generator_semantic_presentation import validate as presentation
from support import (
    HERE,
    checked_path,
    json_bytes,
    require,
    same_json,
    sha,
    unique_json,
)


def attach(root, binding, major, stream, sources, original, executions):
    require(stream['name'] == FAMILY, 'semantic.capture_family')
    source(root, binding['candidate'])
    context(root, binding)
    runtime, metadata = lane(root, major, binding)
    provider = binding['lanes'][major]['extension_provider']
    provider = {
        **provider,
        'path': str(Path(provider['path']).expanduser().resolve()),
    }
    require(
        sha(Path(provider['path']).read_bytes()) == provider['sha256'],
        'semantic.parent_extension',
    )
    with tempfile.TemporaryDirectory(
        prefix='p97-semantic-worker-'
    ) as temporary:
        environment = batch.environment(
            root, runtime, Path(temporary), 'core', major
        )
        if major == '2':
            copied = (
                Path(binding['lanes'][major]['legacy_copy_root'])
                .expanduser()
                .resolve()
            )
            environment['PYTHONPATH'] = ':'.join(
                [str(copied.parent), *environment['PYTHONPATH'].split(':')[1:]]
            )
            environment['P97_LEGACY_COPY_ROOT'] = str(copied)
        process_id, raw, elapsed = batch.bounded(
            [
                str(runtime / 'bin/python'),
                '-B',
                str(HERE / 'generator_semantic_worker.py'),
                str(root),
            ],
            json_bytes(
                {
                    'sources': sources,
                    'stream': stream,
                    'extension_provider': provider,
                }
            ),
            environment,
            root,
        )
        result = unique_json(raw)
        require(
            result['process_id'] == process_id
            and result['pandas_version'] == metadata['pandas_version']
            and result['python_version'] == metadata['python_version']
            and result['native_sha256'] == metadata['native_sha256'],
            'semantic.worker_identity',
        )
        require(
            result['extension_provider_sha256'] == provider['sha256'],
            'semantic.worker_extension',
        )
    require(not Path(temporary).exists(), 'semantic.worker_cleanup')
    source(root, binding['candidate'])
    context(root, binding)
    lane(root, major, binding)
    require(
        sha(Path(provider['path']).read_bytes()) == provider['sha256'],
        'semantic.parent_extension_changed',
    )
    require(
        set(result['samples'])
        == set(original)
        == {str(ordinal) for ordinal in stream['ordinals']},
        'semantic.executed_coordinates',
    )
    for ordinal, measured in result['samples'].items():
        sample = original[ordinal]
        require(
            same_json(measured['input'], sample['input']),
            'semantic.same_original_input',
        )
        correction = presentation(
            root,
            major,
            {
                'generator': {
                    'name': stream['name'],
                    'stream_seed': stream['seed'],
                    'ordinal': int(ordinal),
                },
                'generator_input': sample['input'],
            },
            sample['outcome'],
            measured['observation'],
        )
        if correction is not None:
            sample['semantic_public_correction'] = correction
        require(
            bool(measured['sources'])
            and all(
                path in sources
                and sources[path] == expected
                and sha(checked_path(root, path).read_bytes()) == expected
                for path, expected in measured['sources'].items()
            ),
            'semantic.capture_source_closure',
        )
        events = measured['source_consumer_events']
        require(
            len(events) == 1
            and events[0]['path'] == 'tests/rust_core/schema/accept_views.py'
            and events[0]['source_sha256'] == sources[events[0]['path']]
            and events[0]['operation_calls'] > 0
            and events[0]['native_calls'] > 0,
            'semantic.actual_source_entry',
        )
        validate(
            measured['host_inputs'], measured['host_inputs'], sample['input']
        )
        sample['host_inputs_witness'] = measured['host_inputs']
        sample['semantic_assets'] = measured['assets']
    executions.append(
        {
            **result,
            'metadata': metadata,
            'elapsed_seconds': elapsed,
            'purpose': 'independent-semantic-input-witness',
        }
    )
