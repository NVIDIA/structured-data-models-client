# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Join independently decoded cases, original finite inventories and actual calls."""

from __future__ import annotations

from expectations import validate_dv87
from inline_old import validate_c29
from recipes import open_recipe
from support import checked_path, contract, require, same_json, sha


def observations(expected: dict) -> dict:
    return {
        identity: value['observation']
        for identity, value in expected.items()
        if value['family']
        in {
            'public-observation-v1',
            'recorded-native-v1',
            'original-generator-v1',
        }
    }


def inventory_join(
    root, inventory: dict, index: dict, decoded: dict, inputs: dict
) -> dict:
    require(
        set(index) == {'carry', 'edges', 'generators', 'sampling_cases'},
        'freeze.index_fields',
    )
    expected = observations(decoded)
    require(
        type(index['sampling_cases']) is list
        and len(index['sampling_cases']) == len(set(index['sampling_cases']))
        and set(index['sampling_cases']) == set(inventory['sampling_case_ids'])
        and all(
            identity in decoded
            and decoded[identity]['family'] == 'recorded-native-v1'
            for identity in index['sampling_cases']
        ),
        'freeze.sampling_identity_family',
    )
    prepared = {**index, 'expected': expected}
    translated = {**inventory, 'generators': {}}
    for label, field in (('carry', 'carry_cases'), ('edges', 'edge_cases')):
        require(
            same_json(index[label], inventory[field]),
            'freeze.fixed_case_mapping',
        )
        expanded = {
            name: [
                f'{identity}/pandas-{major}'
                for identity in identities
                for major in ('2', '3')
            ]
            for name, identities in inventory[field].items()
        }
        translated[field] = expanded
        prepared[label] = expanded
    claimed = set()
    for name, settings in inventory['generators'].items():
        require(
            sha(checked_path(root, settings['source']).read_bytes())
            == settings['source_sha256'],
            'freeze.generator_original_source',
        )
        require(name in index['generators'], 'freeze.generator_missing')
        records = index['generators'][name]
        coordinates = set()
        generated_inputs = {}
        for record in records:
            require(
                set(record) == {'stream_seed', 'ordinal', 'seed', 'case_id'},
                'freeze.coordinate_fields',
            )
            stream, ordinal = record['stream_seed'], record['ordinal']
            require(
                type(stream) is int
                and stream >= 0
                and type(ordinal) is int
                and ordinal >= 0,
                'freeze.coordinate_types',
            )
            require(
                record['seed']
                == ((stream + ordinal) * (stream + ordinal + 1)) // 2 + ordinal
                and type(record['seed']) is int,
                'freeze.paired_sample_identity',
            )
            coordinate = (stream, ordinal)
            require(
                coordinate not in coordinates
                and record['case_id'] not in claimed,
                'freeze.duplicate_coordinate_or_input',
            )
            coordinates.add(coordinate)
            claimed.add(record['case_id'])
            require(
                record['case_id'] in expected, 'freeze.generated_case_absent'
            )
            require(
                record['case_id'] in inputs, 'freeze.generated_input_absent'
            )
            recipe = open_recipe(inputs[record['case_id']]['recipe'])
            require(
                same_json(
                    recipe.get('generator'),
                    {
                        'name': name,
                        'stream_seed': stream,
                        'ordinal': ordinal,
                    },
                )
                and type(recipe.get('seed')) is int
                and recipe['seed'] == record['seed'],
                'freeze.generated_coordinate_join',
            )
            fingerprint = contract.digest(
                {
                    key: recipe[key]
                    for key in ('operation', 'entrypoint', 'input')
                }
            )
            generated_inputs.setdefault(stream, set()).add(fingerprint)
        require(
            {stream for stream, _ordinal in coordinates}
            == set(settings['required_stream_seeds']),
            'freeze.generator_streams',
        )
        require(
            all(
                len(generated_inputs[stream])
                >= settings['minimum_samples_per_stream']
                for stream in settings['required_stream_seeds']
            ),
            'freeze.generator_stream_floor',
        )
        mandatory = {
            tuple(value) for value in settings['mandatory_coordinates']
        }
        historical = {
            tuple(value) for value in settings['historical_coordinates']
        }
        require(
            mandatory | historical <= coordinates, 'freeze.required_coordinate'
        )
        translated['generators'][name] = {
            **settings,
            'historical_mismatch_seeds': [
                ((stream + ordinal) * (stream + ordinal + 1)) // 2 + ordinal
                for stream, ordinal in historical
            ],
        }
    contract.validate_carry_and_generators(translated, prepared)
    return expected


def carry_proofs(
    root, inventory: dict, inputs: dict, joined: dict, expected: dict
) -> dict:
    wanted = {
        f'{identity}/pandas-{major}'
        for cases in inventory['carry_cases'].values()
        for identity in cases
        for major in ('2', '3')
    }
    require(wanted <= set(expected), 'carry.expected_missing')
    expanded = {
        **inventory,
        'carry_cases': {},
        'accepted_dvs': [*inventory['accepted_dvs'], 'DV87'],
    }
    proofs = {}
    decisions = {
        (item['case_id'], item['pandas_major']): item
        for item in inventory['decisions']
    }
    require(
        len(decisions) == len(inventory['decisions']),
        'carry.duplicate_decision',
    )
    for carry, identities in inventory['carry_cases'].items():
        expanded['carry_cases'][carry] = []
        proofs[carry] = {}
        for identity in identities:
            for major in ('2', '3'):
                key = f'{identity}/pandas-{major}'
                require(
                    key in inputs and key in joined,
                    'carry.actual_execution_missing',
                )
                recipe = open_recipe(inputs[key]['recipe'])
                pair = joined[key]
                old, core = pair['python'], pair['core']
                require(
                    same_json(old['observation'], inputs[key]['actual_old'])
                    and same_json(core['observation'], inputs[key]['expected'])
                    and same_json(inputs[key]['expected'], expected[key]),
                    'carry.frozen_observation_join',
                )
                for selector, record in pair.items():
                    require(
                        record['selector'] == selector
                        and record['pandas_major'] == major
                        and all(
                            type(record[field]) is int and record[field] >= 0
                            for field in (
                                'public_calls',
                                'legacy_calls',
                                'native_calls',
                            )
                        )
                        and record['public_calls'] > 0,
                        'carry.measured_lane',
                    )
                require(set(pair) == {'python', 'core'}, 'carry.selector_map')
                decision = decisions.get((identity, major))
                if decision is not None and decision['dv'] == 'DV87':
                    require(carry == 'C05', 'dv87.exact_carry')
                    validate_dv87(
                        key,
                        inputs[key]['recipe'],
                        old['observation'],
                        core['observation'],
                        inventory.get('approved_difference_scopes'),
                    )
                backend = recipe['input'].get('backend', 'memory')
                proof = {
                    'sha': inventory['candidate'],
                    'entrypoint': recipe['entrypoint'],
                    'backend': backend,
                    'python_legacy_calls': old['legacy_calls'],
                    'core_native_calls': core['native_calls'],
                    'python_native_calls': old['native_calls'],
                    'core_legacy_calls': core['legacy_calls'],
                    'actual_old': old['observation'],
                    'native': core['observation'],
                    'dv': None if decision is None else decision['dv'],
                    'decision_old_sha256': None
                    if decision is None
                    else decision['old_sha256'],
                    'decision_core_sha256': None
                    if decision is None
                    else decision['core_sha256'],
                }
                if carry == 'C29':
                    validate_c29(
                        root,
                        inputs[key]['recipe'],
                        old,
                        core,
                        expected[key],
                    )
                    proof['legacy_branch_events'] = old['legacy_branch_events']
                    if not same_json(proof['actual_old'], proof['native']):
                        require(
                            proof['dv'] in inventory['accepted_dvs'],
                            'carry.unapproved_difference',
                        )
                        require(
                            proof['decision_old_sha256']
                            == contract.digest(proof['actual_old'])
                            and proof['decision_core_sha256']
                            == contract.digest(proof['native']),
                            'carry.decision_hash',
                        )
                    else:
                        require(proof['dv'] is None, 'carry.stale_difference')
                expanded['carry_cases'][carry].append(key)
                proofs[carry][key] = proof
    lane_expected = {
        key: expected[key] for proof in proofs.values() for key in proof
    }
    contract.validate_carry_proofs(
        {**expanded, 'carry_cases': {**expanded['carry_cases'], 'C29': []}},
        {**proofs, 'C29': {}},
        lane_expected,
    )
    return proofs


def sweep_frozen(sweep_expected: dict, decoded: dict):
    require(
        set(sweep_expected) <= set(decoded), 'bundle.sweep_missing_from_frozen'
    )
    require(
        all(
            decoded[key]['family'] == 'public-observation-v1'
            and same_json(decoded[key]['observation'], value)
            for key, value in sweep_expected.items()
        ),
        'bundle.sweep_frozen_disagree',
    )
