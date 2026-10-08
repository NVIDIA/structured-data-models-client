# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Localized pre-LOCK DV85/86 extensions; original historical predicates stay intact."""

from __future__ import annotations

import itertools
import re

import schema_boundary
from expectations import DECISIONS, FORMAT, sweep_entry
from recipes import open_recipe
from routes import validate_capture_route
from support import contract, require, same_json


def sweep(
    inventory: dict, cases: list, decisions: list, route_map: dict
) -> dict:
    require(re.fullmatch('[0-9a-f]{40}', inventory['base']) is not None, 'base')
    for key, wanted in [
        ('roles', contract.ROLES),
        ('backends', contract.BACKENDS),
        ('tasks', contract.TASKS),
        ('pandas_majors', contract.MAJORS),
        ('carry_ids', contract.CARRY),
        ('edge_ids', contract.EDGES),
    ]:
        require(set(inventory[key]) == wanted, f'inventory.{key}')
    dtypes = inventory['dtype_ids']
    require(
        bool(dtypes) and len(dtypes) == len(set(dtypes)), 'inventory.dtypes'
    )
    support = inventory['supported_controls']
    require(
        set(support)
        == {
            f'{backend}/{task}'
            for backend, task in itertools.product(
                contract.BACKENDS, contract.TASKS
            )
        }
        and all(type(value) is bool for value in support.values())
        and any(support.values()),
        'inventory.support',
    )
    require(len(cases) >= 2000, 'sweep.minimum')
    ids, inputs, cells, controls = set(), set(), set(), set()
    expected, differences, decimal_scopes = {}, {}, {}
    finite_scopes = {}
    for case in cases:
        identity, value = case['id'], case['input']
        require(
            type(identity) is str and bool(identity) and identity not in ids,
            'sweep.duplicate_id',
        )
        ids.add(identity)
        require(
            set(value)
            == {'dtype', 'role', 'backend', 'task', 'data', 'options'},
            'input.fields',
        )
        require(
            value['dtype'] in dtypes
            and value['role'] in contract.ROLES
            and value['backend'] in contract.BACKENDS
            and value['task'] in contract.TASKS,
            'input.dimensions',
        )
        fingerprint = contract.digest(value)
        require(
            case['input_sha256'] == fingerprint and fingerprint not in inputs,
            'input.hash_or_duplicate',
        )
        inputs.add(fingerprint)
        require(type(case['seed']) is int, 'input.seed')
        cells.add((value['dtype'], value['role']))
        require(set(case['runs']) == contract.MAJORS, 'sweep.both_majors')
        envelope = case['recipe']
        recipe = open_recipe(envelope)
        require(
            same_json(value['data'], envelope),
            'input.executable_recipe_binding',
        )
        for major, pair in case['runs'].items():
            require(set(pair) == {'python', 'core'}, 'sweep.both_paths')
            key = f'{identity}/pandas-{major}'
            require(key in route_map, 'sweep.route_missing')
            binding = route_map[key]
            require(
                binding['input_sha256'] == envelope['input_sha256']
                and binding['entrypoint'] == recipe['entrypoint'],
                'sweep.route_input',
            )
            for selector, run in pair.items():
                require(
                    run['selector'] == selector
                    and run['pandas_major'] == major
                    and run['input_sha256'] == fingerprint
                    and run['source_sha'] == inventory['candidate'],
                    'provenance.binding',
                )
                require(
                    run['entrypoint']
                    in {
                        'PredictiveQuery.materialize',
                        'SDMClient.predict',
                        'FittedModel.predict',
                    }
                    and run['entrypoint'] == recipe['entrypoint'],
                    'provenance.public',
                )
                require(
                    type(run['process_id']) is int
                    and run['process_id'] > 0
                    and run['process_id']
                    != pair['core' if selector == 'python' else 'python'][
                        'process_id'
                    ],
                    'provenance.isolation',
                )
                if run.get('schema_preparation') is not None:
                    schema_boundary.validate(envelope, run)
                    if selector == 'core':
                        validate_capture_route(binding, run, run['host_event'])
                    continue
                require(
                    all(
                        type(run[key]) is int and run[key] >= 0
                        for key in (
                            'public_calls',
                            'native_calls',
                            'legacy_calls',
                            'materialized_rows',
                        )
                    )
                    and run['public_calls'] > 0,
                    'provenance.counts',
                )
                contract.observation(run['observation'])
                if selector == 'python':
                    require(
                        run['legacy_calls'] > 0 and run['native_calls'] == 0,
                        'provenance.oracle',
                    )
                    if major == '2' and binding['legacy_copy_required']:
                        require(
                            type(run['copied_legacy_calls']) is int
                            and run['copied_legacy_calls'] > 0,
                            'provenance.copied_oracle',
                        )
                else:
                    validate_capture_route(binding, run, run['host_event'])
            old, core = (
                pair['python']['observation'],
                pair['core']['observation'],
            )
            for decision in ('DV131', 'DV133', 'DV134', 'DV136'):
                scope = inventory['approved_difference_scopes'].get(
                    decision, {}
                )
                entry = scope.get('cases', {}).get(key)
                if entry is not None:
                    require(
                        key not in decimal_scopes
                        and same_json(envelope, entry['recipe'])
                        and same_json(old, entry['actual_old'])
                        and same_json(core, entry['expected'])
                        and all(
                            pair[selector]['materialized_rows']
                            == entry['materialized_rows'][selector]
                            for selector in ('python', 'core')
                        ),
                        'decimal.sweep_scope',
                    )
                    decimal_scopes[key] = decision
            finite = sweep_entry(
                key,
                envelope,
                old,
                core,
                {
                    selector: pair[selector]['materialized_rows']
                    for selector in ('python', 'core')
                },
                inventory['approved_difference_scopes'],
                [*inventory['accepted_dvs'], *DECISIONS],
            )
            if finite:
                require(
                    key not in finite_scopes and key not in decimal_scopes,
                    'finite.duplicate_scope_case',
                )
                finite_scopes[key] = finite
            expected[key] = core
            if contract.digest(old) != contract.digest(core):
                differences[(identity, major)] = (
                    contract.digest(old),
                    contract.digest(core),
                )
            if (
                old['status'] == core['status'] == 'success'
                and old['payloads']
                and core['payloads']
                and pair['python']['materialized_rows'] > 1
                and pair['core']['materialized_rows'] > 1
                and (value['backend'] == 'memory' or old['sql'] and core['sql'])
            ):
                controls.add((f'{value["backend"]}/{value["task"]}', major))
    require(
        set(decimal_scopes)
        == {
            identity
            for decision in ('DV131', 'DV133', 'DV134', 'DV136')
            for identity in inventory['approved_difference_scopes']
            .get(decision, {})
            .get('cases', {})
        },
        'decimal.missing_scope_case',
    )
    require(
        set(finite_scopes)
        == {
            identity
            for scope in inventory['approved_difference_scopes'].values()
            if scope.get('format') == FORMAT
            for identity in scope.get('cases', {})
        },
        'finite.missing_scope_case',
    )
    require(set(route_map) == set(expected), 'sweep.route_case_map')
    require(
        cells == set(itertools.product(dtypes, contract.ROLES)),
        'sweep.missing_cell',
    )
    require(
        all(
            (cell, major) in controls
            for cell, supported in support.items()
            if supported
            for major in contract.MAJORS
        ),
        'sweep.missing_success',
    )
    accepted = {}
    for decision in decisions:
        key = (decision['case_id'], decision['pandas_major'])
        require(
            key not in accepted and key in differences, 'dv.stale_or_duplicate'
        )
        require(
            (
                decision['dv'] in inventory['accepted_dvs']
                or finite_scopes.get(
                    f'{decision["case_id"]}/pandas-{decision["pandas_major"]}'
                )
                == decision['dv']
                or decimal_scopes.get(
                    f'{decision["case_id"]}/pandas-{decision["pandas_major"]}'
                )
                == decision['dv']
            )
            and bool(decision['reason'].strip()),
            'dv.unaccepted',
        )
        accepted[key] = (decision['old_sha256'], decision['core_sha256'])
    require(accepted == differences, 'parity.unapproved_or_wrong_difference')
    return expected


def fixture_correction(
    scope: dict,
    correction: dict,
    executed: dict,
    fixture_case: dict,
    accepted_dvs: list,
):
    oracle = scope['oracle']
    require(
        set(oracle)
        == {
            'kind',
            'fixture_path',
            'fixture_case_id',
            'pandas_major',
            'input_sha256',
        }
        and oracle['kind'] == 'fixture',
        'corrected.fixture_oracle',
    )
    require(
        oracle['fixture_case_id'] == fixture_case['id']
        and oracle['input_sha256'] == contract.digest(fixture_case['input']),
        'corrected.fixture_input',
    )
    require(
        executed['fixture_path'] == oracle['fixture_path']
        and executed['fixture_case_id'] == oracle['fixture_case_id']
        and executed['pandas_major'] == oracle['pandas_major']
        and executed['input_sha256'] == oracle['input_sha256']
        and executed['source_sha'] == scope['candidate'],
        'corrected.executed_fixture',
    )
    require(
        correction['dv'] == scope['dv']
        and correction['intervention'] == scope['intervention']
        and correction['source_sha256'] == scope['source_sha256'],
        'corrected.exact_intervention',
    )
    contract.validate_corrected_reference(correction, accepted_dvs)
    require(
        same_json(correction['actual_old'], executed['actual_old'])
        and same_json(correction['native'], executed['native']),
        'corrected.execution_join',
    )
