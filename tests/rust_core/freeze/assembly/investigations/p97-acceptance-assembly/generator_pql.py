# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-bound PQL input constructors, replayed from the original stream start."""

from __future__ import annotations

import ast
import itertools
import random
import subprocess

from support import checked_path, load, require, sha, unique_json

SOURCE = 'tests/rust_core/pql/_generate.py'
HELPER = 'tests/rust_core/pql/_pql.py'
CUSTOM = 'rust/crates/pql/tests/custom_profiles.py'
RETAINED = 'rust/crates/pql/tests/custom_profiles.json'


def source_bindings(root):
    return {
        path: sha(checked_path(root, path).read_bytes())
        for path in (SOURCE, HELPER, CUSTOM, RETAINED)
    }


def original_schemas(root):
    helper = load('_p97_original_pql_helper', checked_path(root, HELPER))
    schemas = helper.schemas()
    require(bool(schemas), 'generator.pql_schemas')
    historical = {}
    for name in ('valid.json', 'invalid.json'):
        path = f'tests/parity/pql/{name}'
        raw = subprocess.check_output(
            ['git', 'show', f'{helper.BASE}:{path}'], cwd=root
        )
        historical[f'{helper.BASE}:{path}'] = sha(raw)
    return schemas, historical


def custom_constants(root):
    tree = ast.parse(checked_path(root, CUSTOM).read_bytes())
    names = {'CAPABILITIES', 'MODES', 'MODEL'}
    constants = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in names
    }
    require(set(constants) == names, 'generator.custom_constants')
    return constants


def samples(root, name, stream_seed, ordinals, bindings):
    require(
        name in {'pql.queries', 'pql.date_literals'}, 'generator.pql_family'
    )
    require(
        type(stream_seed) is int
        and stream_seed
        in ({20261002, 31, 32} if name == 'pql.queries' else {20261003}),
        'generator.pql_seed',
    )
    require(
        bool(ordinals)
        and all(type(value) is int and value >= 0 for value in ordinals),
        'generator.pql_ordinals',
    )
    require(source_bindings(root) == bindings, 'generator.pql_source')
    producer = load('_p97_original_pql_generator', checked_path(root, SOURCE))
    count = max(ordinals) + 1
    require(
        count <= (50000 if name == 'pql.queries' else 100000),
        'generator.pql_population',
    )
    historical = {}
    if name == 'pql.date_literals':
        generated = (
            {'literal': literal}
            for literal in producer.date_literals(stream_seed, count)
        )
    else:
        schemas, historical = original_schemas(root)
        generated = producer.queries(stream_seed, count, schemas)
    reconstructed = stream_seed in {31, 32}
    constants = custom_constants(root) if reconstructed else None
    profile_rng = random.Random(stream_seed)
    wanted = set(ordinals)
    result = {}
    for ordinal, arguments in enumerate(itertools.islice(generated, count)):
        if reconstructed:
            arguments = {
                'query': arguments['query'],
                'schema': arguments['schema'],
                'mode': profile_rng.choice(constants['MODES']),
                **(
                    {
                        'capabilities': {
                            'base': 'nothing',
                            'set': {
                                name: bool(profile_rng.getrandbits(1))
                                for name in constants['CAPABILITIES']
                            },
                            'model': constants['MODEL'],
                        }
                    }
                    if stream_seed == 31
                    else {
                        'allow_array_targets': bool(profile_rng.getrandbits(1)),
                        'allow_timestamp_arrays': bool(
                            profile_rng.getrandbits(1)
                        ),
                    }
                ),
            }
        if ordinal in wanted:
            result[ordinal] = {
                'operation': 'pql.date'
                if name == 'pql.date_literals'
                else 'pql.profile'
                if stream_seed == 31
                else 'pql.options'
                if stream_seed == 32
                else 'pql.validate',
                'arguments': arguments,
            }
    require(set(result) == wanted, 'generator.pql_selection')
    require(source_bindings(root) == bindings, 'generator.pql_source_changed')
    return {
        'samples': result,
        'historical_corpus': historical,
        'reconstruction': 'reviewed-custom-profile-reconstitution-required'
        if reconstructed
        else None,
        'retained_custom_cases': {
            key: len(
                unique_json(checked_path(root, RETAINED).read_bytes())[key]
            )
            for key in ('profiles', 'options')
        }
        if reconstructed
        else None,
    }
