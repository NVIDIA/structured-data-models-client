# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise input generation against the independently locked constructors."""

from __future__ import annotations

import copy
import json
import subprocess
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

import p97_generate as generator  # noqa: E402
from generator_inputs import construct, source_bindings  # noqa: E402
from generator_records import decode  # noqa: E402
from recipes import open_recipe  # noqa: E402
from support import Rejected  # noqa: E402


@pytest.fixture
def inventory():
    policy = json.loads(
        (ROOT / 'tests/rust_core/freeze/inventory-policy.json').read_bytes()
    )
    sources = source_bindings(ROOT)
    settings = {
        **policy['generator_policy'][generator.RANDOM_STATE],
        'source_sha256': sources[generator.RANDOM_STATE_SOURCE],
    }
    return {
        'generators': {generator.RANDOM_STATE: settings},
        'generator_adapter_sources': sources,
    }


def test_original_input_stream_and_edges(inventory):
    recipes = generator.random_state_recipes(ROOT, inventory)
    ordinals = sorted(generator.selected_ordinals(10000, 256, range(18)))
    original = construct(
        ROOT,
        generator.RANDOM_STATE,
        20261004,
        ordinals,
        inventory['generator_adapter_sources'],
    )
    assert len(recipes) == len(ordinals)
    assert len({case['id'] for case in recipes}) == len(ordinals)
    assert len({case['recipe']['input_sha256'] for case in recipes}) == len(
        ordinals
    )
    assert set(range(18)) <= set(ordinals)
    assert ordinals[-1] == 9999
    for ordinal, case in zip(ordinals, recipes, strict=True):
        assert set(case) == {'id', 'recipe'}
        recipe = open_recipe(case['recipe'])
        assert recipe['input'] == original[ordinal]['input']
        assert recipe['generator'] == {
            'name': generator.RANDOM_STATE,
            'stream_seed': 20261004,
            'ordinal': ordinal,
        }
        assert recipe['seed'] == (
            (20261004 + ordinal) * (20261005 + ordinal) // 2 + ordinal
        )


def test_cli_is_deterministic_and_input_only(inventory, tmp_path):
    inventory_path = tmp_path / 'inventory.json'
    inventory_path.write_text(json.dumps(inventory))
    output = tmp_path / 'recipes.json'
    command = [
        sys.executable,
        '-B',
        str(ROOT / 'scripts/parity/p97_generate.py'),
        '--inventory',
        str(inventory_path),
        '--seed',
        '20261005',
        '--stream',
        generator.RANDOM_STATE,
        '--out',
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)
    first = output.read_bytes()
    subprocess.run(command, check=True, capture_output=True)
    assert output.read_bytes() == first
    assert all(set(case) == {'id', 'recipe'} for case in json.loads(first))


@pytest.mark.parametrize(
    'change',
    ['source', 'dependency', 'stream', 'ordinal', 'population', 'floor'],
)
def test_invalid_source_or_coordinate_is_rejected(inventory, change):
    mutated = copy.deepcopy(inventory)
    settings = mutated['generators'][generator.RANDOM_STATE]
    if change == 'source':
        settings['source_sha256'] = '0' * 64
    elif change == 'dependency':
        mutated['generator_adapter_sources'][
            'tests/rust_core/dialects/_dialects.py'
        ] = '0' * 64
    elif change == 'stream':
        settings['required_stream_seeds'] = [20261005]
    elif change == 'ordinal':
        settings['historical_coordinates'] = [[20261004, -1]]
    elif change == 'population':
        settings['historical_coordinates'] = [[20261004, 10000]]
    else:
        settings['minimum_samples_per_stream'] = 255
    with pytest.raises(ValueError):
        generator.random_state_recipes(ROOT, mutated)


def test_historical_coordinate_extends_spread_selection(inventory):
    baseline = generator.random_state_recipes(ROOT, inventory)
    inventory['generators'][generator.RANDOM_STATE][
        'historical_coordinates'
    ] = [[20261004, 9998]]
    recipes = generator.random_state_recipes(ROOT, inventory)
    assert len(recipes) == len(baseline) + 1
    recipe = open_recipe(recipes[-2]['recipe'])
    original = construct(
        ROOT,
        generator.RANDOM_STATE,
        20261004,
        [9998],
        inventory['generator_adapter_sources'],
    )
    assert recipe['input'] == original[9998]['input']


def test_original_outcomes_decode_and_mutations_fail(inventory, tmp_path):
    recipes = generator.random_state_recipes(ROOT, inventory)
    ordinals = [
        open_recipe(case['recipe'])['generator']['ordinal'] for case in recipes
    ]
    original = construct(
        ROOT,
        generator.RANDOM_STATE,
        20261004,
        ordinals,
        inventory['generator_adapter_sources'],
        observe_outcomes=True,
    )
    import pandas as pd

    major = pd.__version__.split('.')[0]
    cases = []
    for ordinal, case in zip(ordinals, recipes, strict=True):
        recipe = open_recipe(case['recipe'])
        observed = original[ordinal]
        assert observed['public_calls'] == 0
        assert observed['consumer_kind'] == 'original-python'
        assert len(observed['source_consumer_events']) == 2
        assert all(
            event['native_calls'] == 0
            for event in observed['source_consumer_events']
        )
        cases.append(
            {
                'id': case['id'] + '/pandas-' + major,
                'recipe': case['recipe'],
                'reference': observed['outcome'],
                'expected': observed['outcome'],
                'corrected_reference': None,
                'replay_input': {
                    'generator': recipe['generator'],
                    'generator_input': recipe['input'],
                },
                'depends_on': [],
                'consumer_kind': observed['consumer_kind'],
            }
        )
    path = 'tests/parity/frozen-generators/dialects.random_state.json'

    def contents(records):
        return {
            path: generator.json_bytes(
                {'format': 'p97-generators-v1', 'cases': records}
            )
        }

    recorded = contents(cases)
    destination = tmp_path / path
    destination.parent.mkdir(parents=True)
    destination.write_bytes(recorded[path])
    expected, inputs = decode(recorded)
    assert len(expected) == len(inputs) == len(recipes)
    for field, replacement, error in (
        (
            'consumer_kind',
            'native-consistency',
            'generator.record_consumer_kind',
        ),
        ('depends_on', ['absent.json'], 'generator.record_dependencies'),
        ('id', cases[1]['id'], 'generator.record_identity'),
    ):
        mutated = copy.deepcopy(cases)
        mutated[0][field] = replacement
        with pytest.raises(Rejected, match=error):
            decode(contents(mutated))


@pytest.mark.parametrize('closed', ['left', 'right', 'both', 'neither'])
@pytest.mark.parametrize(
    'kind',
    [
        'integer',
        'float',
        'timestamp',
        'timedelta_s',
        'timedelta_ms',
        'timedelta_us',
        'timedelta_ns',
    ],
)
def test_interval_fixture_roundtrip(closed, kind):
    import numpy as np
    import pandas as pd
    from _fixture_ops import decode as decode_value
    from _fixture_ops import encode as encode_value

    endpoints = {
        'integer': (-(2**63), 2**63 - 1),
        'float': (float('-inf'), 0.125),
        'timestamp': (
            pd.Timestamp('2024-01-01T00:00:00.000000001+05:30'),
            pd.Timestamp('2024-01-01T00:00:00.000000002+05:30'),
        ),
    }
    for unit in ('s', 'ms', 'us', 'ns'):
        endpoints['timedelta_' + unit] = (
            pd.Timedelta(np.timedelta64(-5, unit)),
            pd.Timedelta(np.timedelta64(2, unit)),
        )
    original = pd.Interval(*endpoints[kind], closed=closed)
    encoded = encode_value(original)
    actual = decode_value(json.loads(json.dumps(encoded)))
    assert actual == original
    assert type(actual.left) is type(original.left)
    assert type(actual.right) is type(original.right)
    assert actual.closed == closed
    if kind.startswith('timedelta_'):
        assert actual.left.unit == original.left.unit
        assert actual.right.unit == original.right.unit
        assert actual.left.asm8 == original.left.asm8
        assert actual.right.asm8 == original.right.asm8


def test_interval_public_matrix_constructor():
    import _fixture_ops as operations
    import pandas as pd
    from _parity.data import generate
    from dimensions import construct as construct_dimensions

    policy = json.loads(
        (ROOT / 'tests/rust_core/freeze/inventory-policy.json').read_bytes()
    )
    templates = policy['required']['matrix_templates']
    name = 'memory-feature-regression'
    template = templates[name]
    frames = generate('shop', 0)
    values = [
        {
            '$interval': {
                'left': ordinal,
                'right': ordinal + 1,
                'closed': 'right',
            }
        }
        for ordinal in range(len(frames[template['table']]))
    ]
    recipe = {
        'entrypoint': template['entrypoint'],
        'operation': template['operation'],
        'input': template['input'],
        'matrix': {
            'template': name,
            'dtype': 'pandas/interval_int64',
            'values': values,
        },
        'seed': 20261005,
    }
    measured = construct_dimensions(recipe, templates, frames, operations)
    assert measured['dtype'] == 'pandas/interval_int64'
    series = frames[template['table']][template['column']]
    assert series.dtype == pd.IntervalDtype('int64', closed='right')
    assert series.tolist() == [
        pd.Interval(ordinal, ordinal + 1, closed='right')
        for ordinal in range(len(series))
    ]
