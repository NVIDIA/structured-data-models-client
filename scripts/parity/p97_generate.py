# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate input-only recipes from the original seeded parity producers."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from _p97_values import typed

ROOT = Path(__file__).resolve().parents[2]
RANDOM_STATE = 'dialects.random_state'
RANDOM_STATE_SOURCE = 'tests/rust_core/dialects/accept_rng.py'


def json_bytes(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def ordered(value):
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError('recipe mappings require string keys')
        return {
            'mapping': [[key, ordered(child)] for key, child in value.items()]
        }
    if isinstance(value, list):
        return {'sequence': [ordered(child) for child in value]}
    if type(value) not in {str, int, float, bool, type(None)}:
        raise TypeError('unsupported recipe value')
    json_bytes(value)
    return {'scalar': value}


def seal(recipe):
    tree = ordered(recipe)
    return {
        'format': 'p97-ordered-recipe-v1',
        'tree': tree,
        'input_sha256': hashlib.sha256(json_bytes(tree)).hexdigest(),
    }


def source_hash(root, relative):
    path = root / relative
    if (
        not relative
        or Path(relative).is_absolute()
        or '..' in Path(relative).parts
        or not path.resolve().is_relative_to(root.resolve())
        or any(
            parent.is_symlink()
            for parent in (path, *path.parents)
            if parent.is_relative_to(root)
        )
    ):
        raise ValueError('invalid producer source path')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(root, inventory, settings):
    bindings = inventory['generator_adapter_sources']
    if bindings.get(settings['source']) != settings['source_sha256']:
        raise ValueError('generator source binding differs from inventory')
    for relative, expected in bindings.items():
        if source_hash(root, relative) != expected:
            raise ValueError('generator source hash mismatch: ' + relative)


def original_random_state(root):
    path = root / RANDOM_STATE_SOURCE
    spec = importlib.util.spec_from_file_location('_p97_rng_producer', path)
    if spec is None or spec.loader is None:
        raise ImportError('original random-state producer unavailable')
    previous = list(sys.path)
    try:
        sys.path.insert(0, str(path.parent))
        producer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(producer)
        return producer
    finally:
        sys.path[:] = previous


def random_state_input(producer, seed, ordinal):
    low, high = producer.BOUNDS[ordinal % len(producer.BOUNDS)]
    return {
        'calls': [
            {
                'backend': 'sqlite',
                'function': 'seeded_hash',
                'arguments': {
                    'random_seed': producer.encode(seed),
                    'ref': 'rowid',
                },
            },
            {
                'backend': producer.BACKENDS[ordinal % len(producer.BACKENDS)],
                'function': 'utils.random_state.randint',
                'arguments': {
                    'random_seed': producer.encode(seed),
                    'low': producer.encode(low),
                    'high': producer.encode(high),
                    'size': 1 + ordinal % 12,
                },
            },
        ]
    }


def selected_ordinals(population, minimum, required):
    if (
        type(population) is not int
        or type(minimum) is not int
        or not 2 <= minimum <= population
        or any(
            type(value) is not int or not 0 <= value < population
            for value in required
        )
    ):
        raise ValueError('invalid original population or selection')
    return {
        ordinal * (population - 1) // (minimum - 1)
        for ordinal in range(minimum)
    } | set(required)


def random_state_recipes(root, inventory):
    settings = inventory['generators'][RANDOM_STATE]
    if (
        settings['source'] != RANDOM_STATE_SOURCE
        or settings['callable'] != 'seeds'
    ):
        raise ValueError('unsupported random-state producer')
    verify_sources(root, inventory, settings)
    producer = original_random_state(root)
    if settings['required_stream_seeds'] != [producer.SEED]:
        raise ValueError('random-state stream seed differs from original')
    minimum = settings['minimum_samples_per_stream']
    if type(minimum) is not int or minimum < 256:
        raise ValueError('random-state stream requires at least 256 samples')
    required = set(range(len(producer.EDGES)))
    for stream_seed, ordinal in (
        settings['mandatory_coordinates'] + settings['historical_coordinates']
    ):
        if (
            type(stream_seed) is not int
            or stream_seed != producer.SEED
            or type(ordinal) is not int
            or ordinal < 0
        ):
            raise ValueError('invalid random-state coordinate')
        required.add(ordinal)
    seeds = producer.seeds()
    ordinals = selected_ordinals(len(seeds), minimum, required)
    recipes = []
    for ordinal in sorted(ordinals):
        combined = producer.SEED + ordinal
        recipe = {
            'operation': 'fixture.generator',
            'entrypoint': 'original.generator.consumer',
            'input': typed(
                random_state_input(producer, seeds[ordinal], ordinal)
            ),
            'generator': {
                'name': RANDOM_STATE,
                'stream_seed': producer.SEED,
                'ordinal': ordinal,
            },
            'seed': combined * (combined + 1) // 2 + ordinal,
        }
        recipes.append(
            {
                'id': f'generator/{RANDOM_STATE}/{producer.SEED}/{ordinal}',
                'recipe': seal(recipe),
            }
        )
    verify_sources(root, inventory, settings)
    return recipes


def original_recipes(root, inventory, names):
    from _p97_streams import population_size, samples

    recipes = []
    for name in names:
        if name == RANDOM_STATE:
            recipes.extend(random_state_recipes(root, inventory))
            continue
        settings = inventory['generators'][name]
        verify_sources(root, inventory, settings)
        minimum = settings['minimum_samples_per_stream']
        floor = (
            128
            if name
            in {
                'sampling.random_word_lists',
                'sampling.hashseed_python_jobs',
                'sampling.hashseed_array_jobs',
            }
            else 256
        )
        if type(minimum) is not int or minimum < floor:
            raise ValueError('insufficient original stream sample floor')
        for stream_seed in settings['required_stream_seeds']:
            required = set()
            for selected_seed, ordinal in (
                settings['mandatory_coordinates']
                + settings['historical_coordinates']
            ):
                if (
                    type(selected_seed) is not int
                    or selected_seed not in settings['required_stream_seeds']
                    or type(ordinal) is not int
                    or ordinal < 0
                ):
                    raise ValueError('invalid original stream coordinate')
                if selected_seed == stream_seed:
                    required.add(ordinal)
            population = population_size(
                root,
                name,
                stream_seed,
                array_count=settings.get('array_job_count'),
            )
            selection_count = minimum
            ordinals = selected_ordinals(population, selection_count, required)
            while True:
                generated = samples(
                    root,
                    name,
                    stream_seed,
                    sorted(ordinals),
                    array_count=settings.get('array_job_count'),
                )
                encoded = {
                    ordinal: typed(value)
                    for ordinal, value in generated.items()
                }
                distinct = {json_bytes(value) for value in encoded.values()}
                if len(distinct) >= minimum:
                    break
                if selection_count == population:
                    raise ValueError(
                        'original population lacks distinct sample floor'
                    )
                selection_count = min(
                    population, selection_count + minimum - len(distinct)
                )
                ordinals.update(
                    selected_ordinals(population, selection_count, required)
                )
            for ordinal in sorted(encoded):
                coordinate = {
                    'name': name,
                    'stream_seed': stream_seed,
                    'ordinal': ordinal,
                }
                combined = stream_seed + ordinal
                recipes.append(
                    {
                        'id': f'generator/{name}/{stream_seed}/{ordinal}',
                        'recipe': seal(
                            {
                                'operation': 'fixture.generator',
                                'entrypoint': 'original.generator.consumer',
                                'input': encoded[ordinal],
                                'generator': coordinate,
                                'seed': combined * (combined + 1) // 2
                                + ordinal,
                            }
                        ),
                    }
                )
        verify_sources(root, inventory, settings)
    return recipes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--stream', action='append')
    arguments = parser.parse_args()
    inventory = json.loads(arguments.inventory.read_bytes())
    if arguments.stream is None:
        from _p97_public import recipes as public_recipes

        recipes = public_recipes(ROOT, inventory, arguments.seed, seal)
        arguments.out.write_bytes(json_bytes(recipes))
        return 0
    names = (
        sorted(inventory['generators'])
        if arguments.stream == ['all']
        else arguments.stream
    )
    if len(names) != len(set(names)) or any(
        name not in inventory['generators'] for name in names
    ):
        parser.error('unknown or duplicate original stream')
    recipes = original_recipes(ROOT, inventory, names)
    arguments.out.write_bytes(json_bytes(recipes))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
