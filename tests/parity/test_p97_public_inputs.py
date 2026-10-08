# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise every public matrix dtype through the locked construction path."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/parity'))
sys.path.insert(0, str(ROOT / 'tests/parity'))
sys.path.insert(
    0,
    str(
        ROOT
        / 'tests/rust_core/freeze/assembly/investigations/p97-acceptance-assembly'
    ),
)

import _fixture_ops as operations  # noqa: E402
from _p97_public import recipes  # noqa: E402
from dimensions import construct  # noqa: E402
from p97_generate import json_bytes, seal  # noqa: E402
from recipes import open_recipe  # noqa: E402

INVENTORY = json.loads(
    (ROOT / 'tests/rust_core/freeze/inventory-policy.json').read_bytes()
)['required']


@pytest.fixture(scope='module')
def generated():
    return recipes(ROOT, INVENTORY, 20261005, seal)


def test_public_recipe_dimensions_and_determinism(generated):
    assert len(generated) >= 2000
    assert len({case['id'] for case in generated}) == len(generated)
    assert all(set(case) == {'id', 'recipe'} for case in generated)
    assert json_bytes(generated) == json_bytes(
        recipes(ROOT, INVENTORY, 20261005, seal)
    )
    dimensions = []
    for case in generated:
        recipe = open_recipe(case['recipe'])
        matrix = recipe['matrix']
        template = INVENTORY['matrix_templates'][matrix['template']]
        dimensions.append((matrix['dtype'], template['role']))
    assert set(dimensions) == {
        (dtype, role)
        for dtype in INVENTORY['dtype_ids']
        for role in INVENTORY['roles']
    }


def test_public_cli_is_deterministic_and_input_only(generated, tmp_path):
    inventory_path = tmp_path / 'inventory.json'
    inventory_path.write_bytes(json_bytes(INVENTORY))
    output = tmp_path / 'recipes.json'
    command = [
        sys.executable,
        '-B',
        str(ROOT / 'scripts/parity/p97_generate.py'),
        '--inventory',
        str(inventory_path),
        '--seed',
        '20261005',
        '--out',
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)
    first = output.read_bytes()
    assert first == json_bytes(generated)
    subprocess.run(command, check=True, capture_output=True)
    assert output.read_bytes() == first


@pytest.mark.parametrize('dtype', INVENTORY['dtype_ids'])
def test_public_dtype_constructs(dtype, generated):
    recipe = next(
        open_recipe(case['recipe'])
        for case in generated
        if case['id'].startswith('sweep/' + dtype + '/feature/')
    )
    frames = {
        name: operations.build_frame(spec)
        for name, spec in recipe['input']['tables'].items()
    }
    measured = construct(
        recipe, INVENTORY['matrix_templates'], frames, operations
    )
    assert measured['dtype'] == dtype
    assert measured['role'] == 'feature'
    assert measured['input_data_sha256']


@pytest.mark.parametrize(
    'dtype',
    [
        'pandas/string_python',
        'pandas/category_ordered_False',
        'pandas/category_ordered_True',
        'numpy/bool',
        'pandas/boolean',
        'arrow/bool8',
        'arrow/fixed_binary_3',
        'numpy/S6',
    ],
)
def test_primary_key_values_are_unique_and_nonmissing(dtype, generated):
    recipe = next(
        open_recipe(case['recipe'])
        for case in generated
        if case['id'].startswith('sweep/' + dtype + '/primary_key/')
    )
    frames = {
        name: operations.build_frame(spec)
        for name, spec in recipe['input']['tables'].items()
    }
    measured = construct(
        recipe, INVENTORY['matrix_templates'], frames, operations
    )
    column = frames[measured['table']][measured['column']]
    assert not column.isna().any()
    assert len(set(column.tolist())) == len(column)
