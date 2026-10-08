# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original graph and hash-seed job inputs, including owned Arrow dependencies."""

from __future__ import annotations

import base64
import random
import tempfile
from pathlib import Path

from generator_primitives import module, select
from support import require


def arrow_bytes(table):
    import pyarrow as pa

    output = pa.BufferOutputStream()
    with pa.ipc.new_stream(output, table.schema) as writer:
        writer.write_table(table)
    return {
        'arrow_ipc_base64': base64.b64encode(
            output.getvalue().to_pybytes()
        ).decode('ascii')
    }


class GraphInput(dict):
    def __init__(self, value, graph):
        super().__init__(value)
        self.original_graph = graph


def graph_input(producer, graph):
    return GraphInput(
        {
            'graph_json': producer.graph_json(graph),
            'tables': {
                name: arrow_bytes(table)
                for name, table in producer.arrow_tables(graph).items()
            },
        },
        graph,
    )


def stores(root, name, seed, ordinals):
    producer = module(root, 'tests/rust_core/sampling/_sampling.py')
    consumer = module(root, 'tests/rust_core/sampling/accept_graph_store.py')
    if name == 'sampling.graph_store':
        require(seed == producer.STORE_SEED, 'generator.original_seed')
        return select(
            consumer._graphs(),
            ordinals,
            lambda graph, _ordinal: graph_input(producer, graph),
        )
    require(
        name == 'sampling.graph_store_calls'
        and seed == producer.STORE_SEED + 1,
        'generator.original_seed',
    )
    rng = random.Random(seed)

    def calls():
        for graph in consumer._graphs():
            original = producer.python_store(graph)
            if 'error' in original:
                yield {
                    'store': graph_input(producer, graph),
                    'store_refused': True,
                    'calls': [],
                }
                continue
            generated = consumer._calls(rng, original)
            yield {
                'store': graph_input(producer, graph),
                'sampler': {
                    **producer.store_sampler_args(original, graph),
                    'seed': 7,
                    'calls': generated,
                },
            }

    return select(calls(), ordinals, lambda value, _ordinal: value)


def hashseed(root, name, seed, ordinals, *, array_count):
    producer = module(root, 'tests/rust_core/sampling/_jobs.py')
    sampling = module(root, 'tests/rust_core/sampling/_sampling.py')
    if name == 'sampling.hashseed_python_jobs':
        require(seed == sampling.RNG_SEED + 11, 'generator.original_seed')
        return select(
            producer.python_jobs(500), ordinals, lambda value, _ordinal: value
        )
    labels = module(root, 'tests/rust_core/labels/_labels.py')
    require(
        name == 'sampling.hashseed_array_jobs'
        and seed in {sampling.STORE_SEED, labels.LABELS_SEED + 9},
        'generator.original_seed',
    )
    require(
        type(array_count) is int and array_count >= 1280,
        'generator.array_population_floor',
    )
    with tempfile.TemporaryDirectory(
        prefix='p97-original-arrow-inputs-'
    ) as temporary:
        scratch = Path(temporary)
        jobs = producer.array_jobs(array_count, scratch)
        selected_kind = 'store' if seed == sampling.STORE_SEED else 'execute'

        def dependencies(value):
            if isinstance(value, dict):
                return {key: dependencies(item) for key, item in value.items()}
            if isinstance(value, list):
                return [dependencies(item) for item in value]
            if isinstance(value, str) and value.startswith(str(scratch) + '/'):
                path = Path(value)
                require(
                    path.is_file()
                    and path.resolve().is_relative_to(scratch.resolve()),
                    'generator.arrow_dependency',
                )
                return {
                    'arrow_ipc_base64': base64.b64encode(
                        path.read_bytes()
                    ).decode('ascii')
                }
            return value

        return select(
            (job for job in jobs if job['kind'] == selected_kind),
            ordinals,
            lambda value, _ordinal: dependencies(value),
        )
