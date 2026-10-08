# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Original dialect, schema and sampling constructors with complete call variants."""

from __future__ import annotations

import itertools
import random
import sys

from support import checked_path, load, require


def module(root, relative):
    path = checked_path(root, relative)
    for directory in (
        path.parent,
        root / 'tests/rust_core',
        root / 'tests/parity',
        root / 'scripts/parity',
    ):
        sys.path.insert(0, str(directory))
    return load(
        '_p97_generator_' + relative.replace('/', '_').replace('.', '_'), path
    )


def select(generated, ordinals, project):
    wanted = set(ordinals)
    found = {
        ordinal: project(value, ordinal)
        for ordinal, value in enumerate(
            itertools.islice(generated, max(wanted) + 1)
        )
        if ordinal in wanted
    }
    require(set(found) == wanted, 'generator.original_population')
    return found


def dialects(root, name, seed, ordinals):
    filenames = {
        'dialects.random_state': 'accept_rng.py',
        'dialects.json': 'accept_json.py',
        'dialects.view_expressions': 'accept_view_expr.py',
    }
    producer = module(root, 'tests/rust_core/dialects/' + filenames[name])
    require(seed == producer.SEED, 'generator.original_seed')
    if name == 'dialects.random_state':

        def project(value, ordinal):
            low, high = producer.BOUNDS[ordinal % len(producer.BOUNDS)]
            return {
                'calls': [
                    {
                        'backend': 'sqlite',
                        'function': 'seeded_hash',
                        'arguments': {
                            'random_seed': producer.encode(value),
                            'ref': 'rowid',
                        },
                    },
                    {
                        'backend': producer.BACKENDS[
                            ordinal % len(producer.BACKENDS)
                        ],
                        'function': 'utils.random_state.randint',
                        'arguments': {
                            'random_seed': producer.encode(value),
                            'low': producer.encode(low),
                            'high': producer.encode(high),
                            'size': 1 + ordinal % 12,
                        },
                    },
                ]
            }

        return select(producer.seeds(), ordinals, project)
    if name == 'dialects.json':
        return select(
            producer.cases(),
            ordinals,
            lambda value, _ordinal: {
                'backend': value[0],
                'function': value[1],
                'arguments': value[2],
            },
        )
    return select(
        producer.expressions(),
        ordinals,
        lambda value, _ordinal: {
            'expression': value,
            'dialects': [None, 'databricks', 'snowflake'],
            'functions': ['_unsafe_expr_reason', '_untrusted_call_reason'],
            'untrusted_only_if_safe': True,
        },
    )


def schema(root, name, seed, ordinals):
    count = max(ordinals) + 1
    if name == 'schema.date_columns':
        producer = module(root, 'tests/rust_core/schema/_dates.py')
        require(
            seed == producer.SEED and count <= producer.COUNT,
            'generator.original_seed_population',
        )
        return select(
            producer.columns(count, seed),
            ordinals,
            lambda value, _ordinal: {
                'column': value,
                'operations': list(producer.OPERATIONS),
                'successful_conversion_read': True,
            },
        )
    producer = module(root, 'tests/rust_core/schema/_views.py')
    require(count <= 10000, 'generator.original_population')
    if name == 'schema.metric_views':
        require(seed == producer.METRIC_SEED, 'generator.original_seed')
        return select(
            producer.metric_views(count, seed),
            ordinals,
            lambda value, _ordinal: {
                'definition': value[0],
                'trust_expressions': value[1],
            },
        )
    require(
        name == 'schema.semantic_views' and seed == producer.SEMANTIC_SEED,
        'generator.original_seed',
    )
    return select(
        producer.semantic_views(count, seed),
        ordinals,
        lambda value, _ordinal: {
            'definition': value[0],
            'tables': [
                {'identity': list(key), 'columns': columns}
                for key, columns in value[1].items()
            ],
            'trust_expressions': value[2],
        },
    )


def sampling(root, name, seed, ordinals):
    producer = module(root, 'tests/rust_core/sampling/_sampling.py')
    if name == 'sampling.random_state':
        require(seed == producer.RNG_SEED, 'generator.original_seed')
        draws = random.Random(producer.RNG_SEED + 7)
        return select(
            ((value, producer.rng_ops(draws)) for value in producer.seeds()),
            ordinals,
            lambda value, _ordinal: {
                'seed': value[0],
                'operations': value[1],
                'modes': ['utils.random_state', 'RandomState.integer-or-words'],
                'words': producer.seed_words(value[0]),
            },
        )
    if name == 'sampling.random_draws':
        require(seed == producer.RNG_SEED + 7, 'generator.original_seed')
        rng = random.Random(seed)
        return select(
            (
                {
                    'seed': value,
                    'operations': producer.rng_ops(rng),
                    'modes': [
                        'utils.random_state',
                        'RandomState.integer-or-words',
                    ],
                }
                for value in producer.seeds()
            ),
            ordinals,
            lambda value, _ordinal: value,
        )
    if name == 'sampling.random_word_lists':
        require(seed == producer.RNG_SEED + 3, 'generator.original_seed')
        rng = random.Random(seed)
        words = [
            [
                rng.randrange(2**32)
                for _ in range(rng.choice((1, 2, 623, 624, 625, 1000)))
            ]
            for _ in range(300)
        ]
        words += [[0], [2**32 - 1], [0] * 700, [1, 0, 0]]
        draws = random.Random(producer.RNG_SEED + 7)
        return select(
            (
                {'words': value, 'operations': producer.rng_ops(draws)}
                for value in words
            ),
            ordinals,
            lambda value, _ordinal: value,
        )
    if name == 'sampling.mt19937':
        require(seed == producer.RNG_SEED + 1, 'generator.original_seed')
        return select(
            producer.mt_jobs(producer.seeds()),
            ordinals,
            lambda value, _ordinal: {'seed': value[0], 'operations': value[1]},
        )
    if name == 'sampling.graphs':
        require(
            seed == producer.GRAPH_SEED and max(ordinals) < producer.NUM_GRAPHS,
            'generator.original_seed_population',
        )
        return select(
            producer.graphs(max(ordinals) + 1, seed),
            ordinals,
            lambda value, _ordinal: value,
        )
    if name == 'sampling.offset_pairs':
        require(seed == producer.OFFSET_SEED, 'generator.original_seed')
        return select(
            producer.offset_pairs(),
            ordinals,
            lambda value, _ordinal: {
                **value,
                'seconds_bounds': ['earliest', 'latest'],
                'nanoseconds': True,
            },
        )
    if name == 'sampling.offset_pooling':
        require(
            seed == producer.OFFSET_SEED + 1 and max(ordinals) < 5000,
            'generator.original_seed_population',
        )
        rng = random.Random(seed)

        def groups():
            for _ordinal in range(5000):
                group = [
                    producer.random_offset(rng)
                    for _ in range(rng.randint(1, 4))
                ]
                if rng.random() < 0.3:
                    group.append(dict(group[0]))
                yield {'offsets': group, 'earliest_variants': [True, False]}

        return select(groups(), ordinals, lambda value, _ordinal: value)
    if name == 'sampling.text':
        require(seed == producer.TEXT_SEED, 'generator.original_seed')
        return select(
            producer.texts(),
            ordinals,
            lambda value, _ordinal: {
                'values': [value],
                'max_words_variants': [50, 3, None],
            },
        )
    require(name == 'sampling.debug_jobs', 'generator.sampling_family')
    debug = module(
        root, 'tests/rust_core/sampling/driver/accept_driver_debug.py'
    )
    require(seed == debug.SEED, 'generator.original_seed')
    jobs = debug.jobs()
    prefix = 2 + len(debug.PAST_INT64) * 2
    grouped = [{'calls': [job]} for job in jobs[:prefix]]
    grouped.extend(
        {'calls': jobs[start : start + 4]}
        for start in range(prefix, prefix + debug.NUM_OFFSETS * 4, 4)
    )
    grouped.extend(
        {'calls': [job]} for job in jobs[prefix + debug.NUM_OFFSETS * 4 :]
    )
    return select(grouped, ordinals, lambda value, _ordinal: value)
