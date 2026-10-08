# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite public presentation witnesses, separate from fixture outcomes."""

import copy
import hashlib
import json

from support import checked_path, contract, require, same_json, sha, unique_json

POLICY = 'tests/rust_core/freeze/semantic-public-reference-policy.json'
FAMILY = 'schema.semantic_views'
SEED = 20261009
SOURCE = 'rust/crates/schema/src/diagnostics.rs'
INTERVENTION = 'retain-owned-public-ValueError-presentation'


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(',', ':'),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def envelope(result):
    return {
        'status': 'success',
        'schema': [],
        'payloads': [],
        'sql': [],
        'result': result,
        'error': None,
        'warnings': [],
    }


def check_witness(value, entry, source_digest):
    require(
        set(value)
        >= {
            'dv',
            'intervention',
            'source_sha256',
            'actual_old',
            'corrected',
            'native',
        },
        'semantic.presentation_fields',
    )
    require(
        value['dv'] == 'DV127'
        and value['intervention'] == INTERVENTION
        and value['source_sha256'] == source_digest,
        'semantic.presentation_provenance',
    )
    require(
        same_json(value['actual_old'], envelope(entry['original_result'])),
        'semantic.presentation_original',
    )
    require(
        same_json(value['corrected'], envelope(entry['public_result']))
        and same_json(value['native'], value['corrected']),
        'semantic.presentation_exact_public',
    )


def validate(root, major, given, original, observed, correction=None):
    require(major in {'2', '3'}, 'semantic.presentation_major')
    require(
        set(given) == {'generator', 'generator_input'},
        'semantic.presentation_input_fields',
    )
    coordinate = given['generator']
    require(
        set(coordinate) == {'name', 'stream_seed', 'ordinal'}
        and coordinate['name'] == FAMILY
        and coordinate['stream_seed'] == SEED
        and type(coordinate['ordinal']) is int,
        'semantic.presentation_coordinate',
    )
    policy = unique_json(checked_path(root, POLICY).read_bytes())
    require(
        policy['format'] == 'p97-semantic-public-reference-v1'
        and len(policy['cases']) == 44,
        'semantic.presentation_policy',
    )
    for name, expected in policy['sources'].items():
        require(
            sha(checked_path(root, name).read_bytes()) == expected,
            'semantic.presentation_source',
        )
    identity = f'{major}/{coordinate["ordinal"]}'
    entry = policy['cases'].get(identity)
    if entry is None:
        require(
            same_json(original, observed) and correction is None,
            'semantic.same_original_observation',
        )
        return None
    require(
        entry['input_sha256'] == digest(given['generator_input']),
        'semantic.presentation_same_input',
    )
    value = {
        'dv': 'DV127',
        'intervention': INTERVENTION,
        'source_sha256': policy['sources'][SOURCE],
        'actual_old': original,
        'corrected': envelope(entry['public_result']),
        'native': observed,
    }
    check_witness(value, entry, policy['sources'][SOURCE])
    negatives = {}
    for name in (
        'wrong_correction',
        'unrelated_failure',
        'missing_native',
        'changed_observation',
    ):
        changed = copy.deepcopy(value)
        if name == 'wrong_correction':
            changed['corrected'] = copy.deepcopy(original)
        elif name == 'unrelated_failure':
            for field in ('actual_old', 'corrected', 'native'):
                changed[field]['result'] = {'scalar': 'unrelated failure'}
        elif name == 'missing_native':
            del changed['native']
        else:
            changed['native']['warnings'] = [
                {'category': 'UserWarning', 'message': 'changed'}
            ]
        try:
            check_witness(changed, entry, policy['sources'][SOURCE])
        except contract.Rejected:
            negatives[name] = True
        else:
            negatives[name] = False
    value['negative_controls'] = negatives
    contract.validate_corrected_reference(value, ['DV127'])
    if correction is not None:
        require(
            same_json(correction, value), 'semantic.presentation_witness_join'
        )
    return value
