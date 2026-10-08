# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Complete linked inputs from retained request and label producer consumers."""

from __future__ import annotations

import json
import random

from generator_primitives import module, select
from support import require


def result_input(result):
    return {
        **{key: value for key, value in vars(result).items() if key != 'items'},
        'items': [dict(vars(item)) for item in result.items],
    }


def request(root, name, seed, ordinals):
    if name == 'request.ranked_nan':
        producer = module(root, 'tests/rust_core/test_ranked_nan_order.py')
        require(seed == producer.SEED, 'generator.original_seed')
        return select(
            producer._rankings(), ordinals, lambda value, _ordinal: value
        )
    if name == 'request.payload_size':
        producer = module(
            root, 'tests/rust_core/request/accept_request_size.py'
        )
        require(
            seed == producer.SEED and max(ordinals) < producer.COUNT,
            'generator.original_seed_population',
        )
        rng = random.Random(seed)

        def payloads():
            for _ordinal in range(producer.COUNT):
                payload = producer._object(rng, 0)
                text = producer.spell(rng, payload)
                yield {
                    'payload': payload,
                    'text': text,
                    'parsed': json.loads(text),
                }

        return select(payloads(), ordinals, lambda value, _ordinal: value)
    producer = module(
        root, 'tests/rust_core/request/accept_request_generated.py'
    )
    if name == 'request.ranked':
        require(seed == producer.RANKED_SEED, 'generator.original_seed')
        return select(
            producer._ranked_inputs(), ordinals, lambda value, _ordinal: value
        )
    results = producer.results.__wrapped__()
    if name == 'request.results':
        require(seed == producer.RESULTS_SEED, 'generator.original_seed')
        return select(
            results, ordinals, lambda value, _ordinal: result_input(value)
        )
    if name == 'request.absent':
        require(seed == producer.RESULTS_SEED + 1, 'generator.original_seed')
        rng = random.Random(seed)
        names = [*producer.FIELDS, 'unknown_field', 'id', 'prediction']
        return select(
            (
                {
                    'result': result_input(result),
                    'requested': [
                        rng.choice(names) for _ in range(rng.randint(0, 6))
                    ],
                }
                for result in results
            ),
            ordinals,
            lambda value, _ordinal: value,
        )
    require(
        name == 'request.merge' and seed == producer.RESULTS_SEED + 2,
        'generator.original_seed',
    )
    rng = random.Random(seed)
    by_shape = {}
    for result in results:
        by_shape.setdefault((result.model, result.outputs), []).append(result)
    alike = [group for group in by_shape.values() if len(group) > 1]

    def merges():
        for _ordinal in range(producer.MERGES):
            if rng.random() < 0.8 and alike:
                group = rng.choice(alike)
                parts = rng.sample(group, rng.randint(1, min(4, len(group))))
            else:
                parts = rng.sample(results, rng.randint(0, 3))
            yield {'parts': [result_input(part) for part in parts]}

    return select(merges(), ordinals, lambda value, _ordinal: value)


def labels(root, name, seed, ordinals):
    producer = module(root, 'tests/rust_core/labels/_labels.py')
    if name == 'labels.aggregation_type':
        require(seed == producer.LABELS_SEED, 'generator.original_seed')
        return select(
            producer.aggregation_type_inputs(),
            ordinals,
            lambda value, _ordinal: value,
        )
    if name == 'labels.aggregation':
        require(
            seed == producer.LABELS_SEED + 1 and max(ordinals) < 3000,
            'generator.original_seed_population',
        )
        return select(
            producer.aggregation_inputs(max(ordinals) + 1),
            ordinals,
            lambda value, _ordinal: value,
        )
    if name == 'labels.questions':
        require(
            seed == producer.LABELS_SEED + 2 and max(ordinals) < 1500,
            'generator.original_seed_population',
        )
        schema = producer.shop_schema()
        return select(
            producer.generated_questions(max(ordinals) + 1),
            ordinals,
            lambda value, _ordinal: {'query': value, 'schema': schema},
        )
    require(
        name == 'labels.question_batches' and seed == producer.LABELS_SEED + 3,
        'generator.original_seed',
    )
    consumer = module(root, 'tests/rust_core/labels/accept_generated.py')
    from predictive_query.schema import json_serde

    rng = random.Random(seed)

    def batches():
        for query, schema, resolved in consumer._questions():
            yield {
                'query': query,
                'schema': schema,
                'resolved': json_serde.to_json_dict(resolved),
                'batches': [
                    producer.sample_data(rng, schema, resolved.entity_table)
                    for _ in range(2)
                ],
            }

    return select(batches(), ordinals, lambda value, _ordinal: value)


def sequences(root, seed, ordinals):
    producer = module(root, 'tests/rust_core/rfm/test_sequences.py')
    require(seed == producer.SEED, 'generator.original_seed')
    return select(
        producer._rows(),
        ordinals,
        lambda value, _ordinal: {
            'calls': [
                {
                    'series': 'arrow',
                    'values': [value, '[1]'],
                    'dtype': dtype,
                    'stype': stype,
                }
                for dtype, stype in (
                    ('string', 'sequence'),
                    ('intlist', None),
                    ('stringlist', None),
                )
            ],
        },
    )
