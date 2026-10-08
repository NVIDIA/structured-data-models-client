# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare product input adapters with independently retained stream inputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ASSEMBLY = (
    ROOT
    / 'tests/rust_core/freeze/assembly/investigations/p97-acceptance-assembly'
)
sys.path.insert(0, str(ASSEMBLY))
sys.path.insert(0, str(ROOT / 'scripts/parity'))

from _p97_streams import population_size  # noqa: E402
from generator_inputs import construct, source_bindings  # noqa: E402
from p97_generate import (  # noqa: E402
    json_bytes,
    original_recipes,
    selected_ordinals,
)
from recipes import open_recipe  # noqa: E402

POLICY = json.loads(
    (ROOT / 'tests/rust_core/freeze/inventory-policy.json').read_bytes()
)['generator_policy']
STREAMS = [
    (name, seed)
    for name, settings in POLICY.items()
    if name != 'dialects.random_state'
    for seed in settings['required_stream_seeds']
]


@pytest.mark.parametrize(
    ('name', 'seed'), STREAMS, ids=[f'{name}/{seed}' for name, seed in STREAMS]
)
def test_original_stream_inputs(name, seed):
    settings = POLICY[name]
    arguments = {'array_count': settings.get('array_job_count')}
    bindings = source_bindings(ROOT)
    inventory = {
        'generator_adapter_sources': bindings,
        'generators': {
            name: {
                **settings,
                'source_sha256': bindings[settings['source']],
            }
        },
    }
    cases = [
        case
        for case in original_recipes(ROOT, inventory, [name])
        if open_recipe(case['recipe'])['generator']['stream_seed'] == seed
    ]
    population = population_size(ROOT, name, seed, **arguments)
    ordinals = sorted(
        selected_ordinals(
            population,
            settings['minimum_samples_per_stream'],
            {
                ordinal
                for stream, ordinal in (
                    settings['mandatory_coordinates']
                    + settings['historical_coordinates']
                )
                if stream == seed
            },
        )
    )
    assert ordinals[0] == 0
    assert ordinals[-1] == population - 1
    actual = {
        open_recipe(case['recipe'])['generator']['ordinal']: open_recipe(
            case['recipe']
        )['input']
        for case in cases
    }
    assert set(ordinals) <= set(actual)
    ordinals = sorted(actual)
    assert len({case['id'] for case in cases}) == len(cases)
    assert (
        len({json_bytes(value) for value in actual.values()})
        >= settings['minimum_samples_per_stream']
    )
    expected = construct(ROOT, name, seed, ordinals, bindings, **arguments)
    assert set(actual) == set(expected) == set(ordinals)
    for ordinal in ordinals:
        encoded = actual[ordinal]
        assert encoded == expected[ordinal]['input'], (name, seed, ordinal)
        json.dumps(encoded, allow_nan=False)


def test_spread_selection_preserves_population_and_required_coordinates():
    assert selected_ordinals(11, 4, {1, 9}) == {0, 1, 3, 6, 9, 10}
    assert selected_ordinals(4, 4, set()) == {0, 1, 2, 3}


@pytest.mark.parametrize(
    'population,minimum,required',
    [
        (3, 4, set()),
        (10, 1, set()),
        (10, 4, {-1}),
        (10, 4, {10}),
        (True, 2, set()),
        (10, True, set()),
    ],
)
def test_invalid_spread_selection_is_rejected(population, minimum, required):
    with pytest.raises(ValueError):
        selected_ordinals(population, minimum, required)
