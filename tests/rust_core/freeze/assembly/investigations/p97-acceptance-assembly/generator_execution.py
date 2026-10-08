# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Parent-owned original consumer workers, separate from public API capture."""

from __future__ import annotations

import tempfile
from pathlib import Path

import generator_semantic_execution
from candidate import context, lane
from execution import batch, source
from generator_inputs import source_bindings
from generator_native import NATIVE_STREAMS
from support import HERE, json_bytes, require, same_json, sha, unique_json


def execute(root, binding, major, request, *, hashseed=None):
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
        'generator.parent_extension',
    )
    with tempfile.TemporaryDirectory(
        prefix='p97-original-consumer-'
    ) as temporary:
        environment = batch.environment(
            root, runtime, Path(temporary), 'python', major
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
        if hashseed is not None:
            environment['PYTHONHASHSEED'] = hashseed
        if request.get('mode') == 'native-consumer':
            environment['SDM_P7_WHEEL_INSTALLED'] = '1'
        process_id, raw, elapsed = batch.bounded(
            [
                str(runtime / 'bin/python'),
                '-B',
                str(HERE / 'generator_worker.py'),
                str(root),
            ],
            encoded_request(request, provider),
            environment,
            root,
        )
        result = unique_json(raw)
        require(
            result['process_id'] == process_id
            and result['extension_provider_sha256'] == provider['sha256']
            and result['pandas_major'] == major
            and result['pandas_version'] == metadata['pandas_version']
            and result['python_version'] == metadata['python_version']
            and result['native_sha256'] == metadata['native_sha256']
            and result['native_variant'] == request.get('native_variant')
            and result['hashseed'] == environment.get('PYTHONHASHSEED'),
            'generator.worker_identity',
        )
    require(
        sha(Path(provider['path']).read_bytes()) == provider['sha256'],
        'generator.parent_extension_changed',
    )
    require(not Path(temporary).exists(), 'generator.worker_cleanup')
    source(root, binding['candidate'])
    context(root, binding)
    lane(root, major, binding)
    return {**result, 'metadata': metadata, 'elapsed_seconds': elapsed}


def variants(root, binding, name):
    if name not in NATIVE_STREAMS:
        return [('python', binding, None)]
    if name != 'sampling.debug_jobs':
        from generator_primitives import module

        original = module(root, 'tests/rust_core/sampling/accept_hashseed.py')
        require(
            tuple(original.SEEDS) == ('1', '4242'),
            'generator.hashseed_source_contract',
        )
        return [(seed, binding, seed) for seed in original.SEEDS]
    builds = binding.get('generator_debug_builds', {})
    require(
        set(builds) == {'debug', 'release'},
        'generator.owned_debug_release_required',
    )
    require(
        len({item['native_build']['native_sha256'] for item in builds.values()})
        == 2,
        'generator.distinct_debug_release',
    )
    for profile, supplied in builds.items():
        require(
            supplied['candidate'] == binding['candidate'],
            'generator.debug_candidate',
        )
        context(root, supplied)
        receipt = unique_json(
            Path(supplied['native_build']['receipt_path'])
            .expanduser()
            .read_bytes()
        )
        accepted = {
            'debug': {'debug', 'dev, unoptimized with debuginfo'},
            'release': {'release', 'release, optimized'},
        }
        require(
            receipt['build']['profile'] in accepted[profile],
            'generator.debug_profile_receipt',
        )
    return [
        (profile, builds[profile], None) for profile in ('debug', 'release')
    ]


def capture(root, binding, requests, *, executions=None):
    sources = source_bindings(root)
    require(
        same_json(requests['sources'], sources),
        'generator.capture_source_binding',
    )
    executions = [] if executions is None else executions
    samples = {}
    for major in ('2', '3'):
        samples[major] = {}
        for stream in requests['streams']:
            key = f'{stream["name"]}/{stream["seed"]}'
            require(
                key not in samples[major], 'generator.capture_duplicate_stream'
            )
            samples[major][key] = {}
            choices = variants(root, binding, stream['name'])
            chunk_size = (
                4
                if stream['name'] == 'labels.question_batches'
                else batch.MAX_CASES
            )
            for start in range(0, len(stream['ordinals']), chunk_size):
                chunk = {
                    **stream,
                    'ordinals': stream['ordinals'][start : start + chunk_size],
                }
                supplied = None
                if stream['name'] in NATIVE_STREAMS:
                    construction = execute(
                        root,
                        binding,
                        major,
                        {
                            'sources': sources,
                            'streams': [chunk],
                            'mode': 'native-input',
                        },
                        hashseed='1',
                    )
                    executions.append(construction)
                    require(
                        construction['failure'] is None
                        and set(construction['streams']) == {key},
                        'generator.native_input_worker',
                    )
                    supplied = construction['streams']
                consumer_chunks = [chunk]
                if supplied is not None:
                    consumer_chunks = native_partitions(
                        chunk,
                        sources,
                        supplied,
                        choices,
                        major,
                        batch.MAX_INPUT,
                    )
                for consumer_chunk in consumer_chunks:
                    observed = []
                    for variant, runtime_binding, hashseed in choices:
                        request = {
                            'sources': sources,
                            'streams': [consumer_chunk],
                        }
                        if stream['name'] in NATIVE_STREAMS:
                            request = consumer_request(
                                sources, consumer_chunk, supplied, variant
                            )
                        execution = execute(
                            root,
                            runtime_binding,
                            major,
                            request,
                            hashseed=hashseed,
                        )
                        executions.append(execution)
                        require(
                            execution['failure'] is None,
                            'generator.original_worker_failure',
                        )
                        require(
                            set(execution['streams']) == {key},
                            'generator.executed_stream',
                        )
                        values = execution['streams'][key]
                        require(
                            set(values)
                            == {
                                str(ordinal)
                                for ordinal in consumer_chunk['ordinals']
                            },
                            'generator.executed_ordinals',
                        )
                        for value in values.values():
                            require(
                                value['public_calls'] == 0
                                and bool(value['source_consumer_events']),
                                'generator.no_public_credit',
                            )
                        observed.append(values)
                    if len(observed) == 2:
                        require(
                            all(
                                same_json(
                                    observed[0][ordinal]['input'],
                                    observed[1][ordinal]['input'],
                                )
                                and same_json(
                                    observed[0][ordinal]['outcome'],
                                    observed[1][ordinal]['outcome'],
                                )
                                for ordinal in observed[0]
                            ),
                            'generator.native_consistency',
                        )
                    if stream['name'] == 'schema.semantic_views':
                        generator_semantic_execution.attach(
                            root,
                            binding,
                            major,
                            consumer_chunk,
                            sources,
                            observed[0],
                            executions,
                        )
                    samples[major][key].update(observed[0])
    require(
        same_json(source_bindings(root), sources),
        'generator.capture_source_changed',
    )
    return {
        'candidate': binding['candidate'],
        'sources': sources,
        'executions': executions,
        'samples': samples,
    }


def request_provider(binding, major):
    provider = binding['lanes'][major]['extension_provider']
    return {
        **provider,
        'path': str(Path(provider['path']).expanduser().resolve()),
    }


def encoded_request(request, provider):
    return json_bytes({**request, 'extension_provider': provider})


def consumer_request(sources, stream, supplied, variant):
    key = f'{stream["name"]}/{stream["seed"]}'
    return {
        'sources': sources,
        'streams': [stream],
        'native_variant': variant,
        'mode': 'native-consumer',
        'native_inputs': {
            key: {
                str(ordinal): supplied[key][str(ordinal)]
                for ordinal in stream['ordinals']
            }
        },
    }


def native_partitions(stream, sources, supplied, choices, major, maximum):
    key = f'{stream["name"]}/{stream["seed"]}'
    ordinals = stream['ordinals']
    require(
        type(ordinals) is list
        and bool(ordinals)
        and all(type(ordinal) is int and ordinal >= 0 for ordinal in ordinals)
        and ordinals == sorted(set(ordinals)),
        'generator.partition_coordinates',
    )
    require(
        set(supplied) == {key}
        and set(supplied[key]) == {str(ordinal) for ordinal in ordinals},
        'generator.partition_input_map',
    )
    require(
        len(choices) == 2
        and len({variant for variant, _binding, _seed in choices}) == 2
        and type(maximum) is int
        and maximum > 0,
        'generator.partition_parameters',
    )

    def fits(selected):
        chunk = {**stream, 'ordinals': selected}
        return all(
            len(
                encoded_request(
                    consumer_request(sources, chunk, supplied, variant),
                    request_provider(binding, major),
                )
            )
            <= maximum
            for variant, binding, _seed in choices
        )

    chunks, pending = [], []
    for ordinal in ordinals:
        proposed = [*pending, ordinal]
        if fits(proposed):
            pending = proposed
            continue
        require(bool(pending), 'generator.partition_single_input_bound')
        chunks.append({**stream, 'ordinals': pending})
        pending = [ordinal]
        require(fits(pending), 'generator.partition_single_input_bound')
    chunks.append({**stream, 'ordinals': pending})
    require(
        [ordinal for chunk in chunks for ordinal in chunk['ordinals']]
        == ordinals,
        'generator.partition_complete_order',
    )
    return chunks
