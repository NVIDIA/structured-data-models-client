# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-owned generator references, never mislabelled as public calls."""

from __future__ import annotations

import generator_dates
import generator_descriptor
import generator_host
import generator_semantic
from generator_execution import capture
from generator_inputs import plan, source_bindings
from generator_native import consumer_kind
from recipes import open_recipe
from support import contract, require, same_json, unique_json

PREFIX = 'tests/parity/frozen-generators/'
FAMILY = 'original-generator-v1'


def decode(contents):
    expected, inputs = {}, {}
    for path, raw in contents.items():
        if not path.startswith(PREFIX) or not path.endswith('.json'):
            continue
        document = unique_json(raw)
        require(
            set(document) == {'format', 'cases'}
            and document['format'] == 'p97-generators-v1',
            'generator.record_format',
        )
        for case in document['cases']:
            require(
                set(case) - {'semantic_public_correction'}
                == {
                    'id',
                    'recipe',
                    'reference',
                    'expected',
                    'corrected_reference',
                    'replay_input',
                    'depends_on',
                    'consumer_kind',
                },
                'generator.record_fields',
            )
            identity = case['id']
            require(
                type(identity) is str and identity not in inputs,
                'generator.record_identity',
            )
            recipe = open_recipe(case['recipe'])
            name = recipe['generator']['name']
            require(name in plan(), 'generator.record_family')
            require(
                'semantic_public_correction' not in case
                or (
                    name == generator_semantic.FAMILY
                    and type(case['semantic_public_correction']) is dict
                ),
                'generator.semantic_public_correction_scope',
            )
            kind = consumer_kind(name)
            require(
                case['consumer_kind'] == kind, 'generator.record_consumer_kind'
            )
            require(
                recipe['operation'] == 'fixture.generator'
                and recipe['entrypoint'] == 'original.generator.consumer',
                'generator.record_entrypoint',
            )
            require(
                type(case['depends_on']) is list
                and set(case['depends_on']) <= set(contents),
                'generator.record_dependencies',
            )
            contract.observation(case['reference'])
            contract.observation(case['expected'])
            require(
                case['replay_input'] is not None,
                'generator.record_nonexecutable',
            )
            expected[identity], inputs[identity] = case['expected'], case
    return expected, inputs


def requests(root, inventory, index):
    require(
        set(index) == set(inventory['generators']) == set(plan()),
        'generator.original_inventory',
    )
    sources = source_bindings(root)
    require(
        same_json(sources, inventory['generator_adapter_sources']),
        'generator.original_source_closure',
    )
    streams = []
    for name, records in index.items():
        settings = inventory['generators'][name]
        require(
            settings['source'] == plan()[name]['source']
            and settings['source_sha256'] == sources[settings['source']],
            'generator.original_source_join',
        )
        for seed in settings['required_stream_seeds']:
            selected = [
                record['ordinal']
                for record in records
                if record['stream_seed'] == seed
            ]
            require(
                bool(selected) and len(selected) == len(set(selected)),
                'generator.selected_coordinates',
            )
            streams.append(
                {
                    'name': name,
                    'seed': seed,
                    'ordinals': sorted(selected),
                    'array_count': settings.get('array_job_count'),
                }
            )
    return {'sources': sources, 'streams': streams}


def join(inventory, index, inputs, receipt, *, root=None, native_sha256=None):
    require(
        receipt['candidate'] == inventory['candidate']
        and set(receipt['samples']) == {'2', '3'},
        'generator.reference_candidate',
    )
    wanted = set()
    for name, records in index.items():
        for record in records:
            stem, separator, major = record['case_id'].rpartition('/pandas-')
            require(
                separator and stem and major in {'2', '3'},
                'generator.index_lane',
            )
            for major in ('2', '3'):
                identity = f'{stem}/pandas-{major}'
                require(
                    identity in inputs and identity not in wanted,
                    'generator.reference_missing_duplicate',
                )
                wanted.add(identity)
                case = inputs[identity]
                recipe = open_recipe(case['recipe'])
                coordinate = {
                    'name': name,
                    'stream_seed': record['stream_seed'],
                    'ordinal': record['ordinal'],
                }
                original = receipt['samples'][major][
                    f'{name}/{record["stream_seed"]}'
                ][str(record['ordinal'])]
                generator_dates.validate(root, case, receipt)
                case = generator_semantic.joined_case(
                    root, case, coordinate, original
                )
                case = generator_descriptor.joined_case(
                    case, coordinate, original
                )
                case = generator_host.joined_case(case, coordinate, original)
                require(
                    same_json(recipe['generator'], coordinate)
                    and recipe['seed'] == record['seed'],
                    'generator.reference_coordinate',
                )
                require(
                    same_json(recipe['input'], original['input'])
                    and same_json(
                        case['replay_input'],
                        {
                            'generator': coordinate,
                            'generator_input': original['input'],
                        },
                    ),
                    'generator.original_input_join',
                )
                require(
                    case['consumer_kind'] == original['consumer_kind']
                    and same_json(case['reference'], original['outcome']),
                    'generator.original_outcome_join',
                )
                if case['corrected_reference'] is None:
                    require(
                        same_json(case['reference'], case['expected']),
                        'generator.unapproved_difference',
                    )
                else:
                    require(
                        case['consumer_kind']
                        in {'original-python', 'original-cpp'},
                        'generator.native_consistency_correction',
                    )
                    correction = case['corrected_reference']
                    if correction['dv'] == 'DV12':
                        import generator_syntax

                        require(root is not None, 'pql.syntax_root')
                        generator_syntax.validate(
                            root,
                            inventory,
                            recipe,
                            case,
                            original,
                            native_sha256,
                        )
                    else:
                        contract.validate_corrected_reference(
                            correction,
                            inventory['accepted_dvs']
                            + (
                                ['DV114'] if correction['dv'] == 'DV114' else []
                            ),
                        )
                    require(
                        same_json(correction['actual_old'], case['reference'])
                        and same_json(correction['native'], case['expected']),
                        'generator.correction_join',
                    )
    require(wanted == set(inputs), 'generator.exact_record_map')


def authenticate(root, binding, inventory, index, inputs):
    receipt = capture(root, binding, requests(root, inventory, index))
    join(
        inventory,
        index,
        inputs,
        receipt,
        root=root,
        native_sha256=binding['native_build']['native_sha256'],
    )
    return receipt
