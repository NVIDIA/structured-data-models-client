# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join complete freshly captured semantic inputs without adopting saved output."""

import hashlib
import json
from pathlib import Path

import generator_host
from generator_semantic_presentation import validate as presentation
from support import checked_path, require, same_json, sha

FAMILY = 'schema.semantic_views'
SOURCES = {
    'tests/rust_core/schema/accept_views.py',
    'tests/rust_core/request/_request.py',
    'rust/bindings/python/python/sdm_core/describe.py',
    'packages/structured-data-models-client/src/sdm_client/_schema_core.py',
    'packages/structured-data-models-client/src/sdm_client/schema.py',
}


def input_digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def assets(value):
    found = set()
    if type(value) is dict:
        if len(value) == 1:
            tag, identity = next(iter(value.items()))
            if tag in {'$table', '$batch', '$array', '$chunked'}:
                require(
                    type(identity) is str
                    and len(identity) == 64
                    and all(
                        character in '0123456789abcdef'
                        for character in identity
                    ),
                    'semantic.arrow_identity',
                )
                found.add('tables/' + identity + '.arrow')
        for child in value.values():
            found.update(assets(child))
    elif type(value) is list:
        for child in value:
            found.update(assets(child))
    return found


def validate(value, witness, original_input):
    generator_host.fields(
        value, {'host', 'external'}, 'semantic.witness_fields'
    )
    generator_host.descriptor(value['host'])
    external = value['external']
    generator_host.fields(
        external,
        {'original_input_sha256', 'initial_schema', 'queries'},
        'semantic.external_fields',
    )
    require(
        external['original_input_sha256'] == input_digest(original_input),
        'semantic.original_input_digest',
    )
    initial = external['initial_schema']
    generator_host.fields(
        initial, {'function', 'args'}, 'semantic.initial_fields'
    )
    require(
        initial['function'] == 'schema_source_run'
        and type(initial['args']) is list
        and len(initial['args']) == 3
        and all(type(argument) is str for argument in initial['args']),
        'semantic.initial_call',
    )
    require(
        type(external['queries']) is list and bool(external['queries']),
        'semantic.query_tape',
    )
    for query in external['queries']:
        generator_host.fields(
            query, {'sql', 'params', 'outcome'}, 'semantic.query_fields'
        )
        require(type(query['sql']) is str, 'semantic.query_sql')
        require(
            type(query['outcome']) is dict
            and set(query['outcome']) in ({'data'}, {'error'}),
            'semantic.query_outcome',
        )
    assets(external)
    require(same_json(value, witness), 'semantic.complete_original_witness')


def joined_case(root, case, coordinate, original):
    if coordinate['name'] != FAMILY:
        return case
    require(root is not None, 'semantic.source_root')
    major = case['id'].rsplit('/pandas-', 1)[1]
    fresh = original.get('semantic_public_correction')
    require(
        ('semantic_public_correction' in case) == (fresh is not None),
        'semantic.public_correction_presence',
    )
    supplied = case.get('semantic_public_correction')
    require(same_json(supplied, fresh), 'semantic.public_correction_witness')
    presentation(
        root,
        major,
        {'generator': coordinate, 'generator_input': original['input']},
        original['outcome'],
        fresh['native'] if fresh is not None else original['outcome'],
        correction=supplied,
    )
    case = {
        key: value
        for key, value in case.items()
        if key != 'semantic_public_correction'
    }
    replay = case['replay_input']
    generator_host.fields(
        replay,
        {'generator', 'generator_input', 'host_inputs'},
        'semantic.replay_fields',
    )
    require(
        same_json(replay['generator'], coordinate)
        and same_json(replay['generator_input'], original['input']),
        'semantic.original_input',
    )
    require(
        'host_inputs_witness' in original and 'semantic_assets' in original,
        'semantic.missing_witness',
    )
    validate(
        replay['host_inputs'],
        original['host_inputs_witness'],
        original['input'],
    )
    captured = original['semantic_assets']
    require(
        type(captured) is dict
        and set(captured) == assets(replay['host_inputs']['external']),
        'semantic.captured_asset_closure',
    )
    dependencies = case['depends_on']
    require(
        type(dependencies) is list
        and len(dependencies) == len(set(dependencies)),
        'semantic.dependencies',
    )
    actual = {}
    owners = set()
    for relative in dependencies:
        path = Path(relative)
        require(
            path.parent.name == 'tables'
            and relative.startswith('tests/parity/frozen-generators/'),
            'semantic.asset_owner',
        )
        owners.add(str(path.parent.parent))
        identity = 'tables/' + path.name
        require(identity not in actual, 'semantic.asset_membership')
        raw = checked_path(root, relative).read_bytes()
        require(sha(raw) == path.stem, 'semantic.asset_digest')
        if identity in captured:
            require(raw.hex() == captured[identity], 'semantic.asset_bytes')
        actual[identity] = raw.hex()
    require(
        len(owners) <= 1 and set(captured) <= set(actual),
        'semantic.asset_closure',
    )
    return {
        **case,
        'replay_input': {
            'generator': replay['generator'],
            'generator_input': replay['generator_input'],
        },
    }
