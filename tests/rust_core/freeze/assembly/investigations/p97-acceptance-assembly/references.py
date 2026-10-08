# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact approved existing-fixture projection, never a sweep or carry alias."""

from __future__ import annotations

import ast
import contextlib
import copy
import inspect
import os
import sys
from pathlib import Path

from catalog import VettedCatalog
from recipes import open_recipe, seal
from support import (
    INVESTIGATIONS,
    Rejected,
    checked_path,
    contract,
    load,
    require,
    same_json,
    sha,
    unique_json,
)

FIXTURE = 'tests/parity/fixtures/driver.json'
SOURCE = 'tests/rust_core/rfm/test_driver_fixtures.py'


def original_outcome(root: Path, case: dict):
    warning_source = 'tests/rust_core/_fixed_contracts.py'
    namespace = {}
    for path, name in (
        (warning_source, 'missing_index_warning'),
        (SOURCE, '_old_fixed_outcome'),
        (SOURCE, '_current_original_fixed_outcome'),
    ):
        tree = ast.parse(checked_path(root, path).read_bytes())
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == name
        ]
        require(len(functions) == 1, 'reference.original_function')
        function = functions[0]
        if name == '_old_fixed_outcome':
            imports = [
                node
                for node in function.body
                if isinstance(node, ast.ImportFrom)
            ]
            require(
                len(imports) == 1
                and imports[0].module == '_fixed_contracts'
                and [(alias.name, alias.asname) for alias in imports[0].names]
                == [('missing_index_warning', None)],
                'reference.original_dependency',
            )
            function.body.remove(imports[0])
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module='__future__',
                    names=[ast.alias(name='annotations')],
                    level=0,
                ),
                function,
            ],
            type_ignores=[],
        )
        exec(
            compile(ast.fix_missing_locations(module), path, 'exec'), namespace
        )
    return projection(*namespace['_current_original_fixed_outcome'](case))


def scope_case(root: Path, scope_id: str, major: str, scopes: dict):
    identity = f'{scope_id}/pandas-{major}'
    require(identity in scopes, 'reference.unapproved_scope')
    scope = scopes[identity]
    require(
        scope['source'] == SOURCE and scope['dv'] == 'DV57',
        'reference.source_scope',
    )
    require(
        sha(checked_path(root, SOURCE).read_bytes()) == scope['source_sha256'],
        'reference.approved_source',
    )
    document = unique_json(checked_path(root, FIXTURE).read_bytes())
    cases = [case for case in document['cases'] if case['id'] == scope_id]
    require(
        len(cases) == 1 and cases[0]['operation'] == 'driver.materialize',
        'reference.exact_fixture_case',
    )
    case = cases[0]
    oracle = {
        'kind': 'fixture',
        'fixture_path': FIXTURE,
        'fixture_case_id': scope_id,
        'pandas_major': major,
        'input_sha256': contract.digest(case['input']),
    }
    require(
        'oracle' not in scope or same_json(scope['oracle'], oracle),
        'reference.original_input',
    )
    return {**scope, 'oracle': oracle}, case


def projection(kind: str, actual: dict):
    from common import outcome

    require(
        kind in {'expected', 'expected_error'} and type(actual) is dict,
        'reference.original_outcome_kind',
    )
    error = None
    if kind == 'expected_error':
        error = {name: actual[name] for name in ('class', 'code', 'message')}
        error['details'] = {'fixture_details_keys': actual['details_keys']}
    return outcome(
        {
            'family': 'fixture-operation-v1',
            'operation': 'driver.materialize',
            'kind': kind,
            'value': actual,
        },
        error=error,
        warnings=[
            {
                'class': category,
                'code': None,
                'message': message,
                'details': {'projection': 'fixture-category-message'},
            }
            for category, message in actual.get('warnings', [])
        ],
    )


def capture(
    root: Path,
    envelope: dict,
    scopes: dict,
    *,
    postgres_connection_factory=None,
):
    sys.path.insert(0, str(INVESTIGATIONS / 'p97-real-adapter-staging'))
    from common import helpers, observe_public_call

    recipe = open_recipe(envelope)
    require(
        recipe['operation'] == 'fixture.reference'
        and recipe['entrypoint'] == 'fixture.driver.materialize',
        'reference.operation',
    )
    value = recipe['input']
    require(
        set(value) == {'scope_id', 'corrected'}
        and type(value['corrected']) is bool,
        'reference.recipe_fields',
    )
    switch, _driver, _request = helpers(root)
    import pandas as pd
    import sdm_core._native as native

    selector = os.environ[switch.SELECTOR]
    major = pd.__version__.split('.')[0]
    require(selector in {'python', 'core'}, 'reference.selector')
    scope, case = scope_case(root, value['scope_id'], major, scopes)
    require(
        case['input']['backend'] in {'memory', 'sqlite', 'duckdb', 'postgres'},
        'reference.backend',
    )
    provider_context = contextlib.nullcontext()
    if case['input']['backend'] == 'postgres':
        from _parity import runner
        from providers import reference_connection

        provider_context = reference_connection(
            runner, postgres_connection_factory
        )
    helper = load('_p97_approved_corrected_reference', root / SOURCE)
    require(
        case['id'].rsplit('/', 1)[-1] in helper.FIXED_IN_THE_CORE,
        'reference.named_fix',
    )
    classifier = switch.PortedCode()
    if major == '2':
        classifier = VettedCatalog(
            classifier,
            root
            / 'packages/structured-data-models-relational-engine/src/sdm_relational_engine',
            Path(os.environ['P97_LEGACY_COPY_ROOT']),
        )
    natives = tuple(
        value for value in vars(native).values() if inspect.isbuiltin(value)
    )
    require(bool(natives), 'reference.native_exports')
    corrected = value['corrected'] and selector == 'python'
    with (
        provider_context,
        helper._current_corrected_reference(case)
        if corrected
        else contextlib.nullcontext(),
    ):
        (kind, actual), counts = observe_public_call(
            lambda: helper._outcome(case),
            helper._outcome.__code__,
            classifier.symbol,
            natives,
        )
    if isinstance(classifier, VettedCatalog):
        classifier.verify()
    require(
        counts['public_calls'] == 1, 'reference.fixture_operation_not_executed'
    )
    require(
        (
            selector == 'python'
            and counts['native_calls'] == 0
            and counts['legacy_calls'] > 0
        )
        or (
            selector == 'core'
            and counts['native_calls'] > 0
            and counts['legacy_calls'] == 0
        ),
        'reference.actual_route',
    )
    return {
        'observation': projection(kind, actual),
        'family': 'fixture-operation-v1',
        **counts,
        'public_calls': 0,
        'fixture_operation_calls': counts['public_calls'],
        'selector': selector,
        'pandas_major': major,
        'materialized_rows': 0,
        'native_sha256': sha(Path(native.__file__).read_bytes()),
        'copied_legacy_calls': getattr(classifier, 'copied_calls', 0),
        'legacy_branch_events': [],
        'host_event': None,
        'dimensions': None,
        'reference': {
            'scope_id': value['scope_id'],
            'corrected': corrected,
            **scope['oracle'],
            'source_sha256': scope['source_sha256'],
            'intervention': scope['intervention'],
        },
    }


def validate(root: Path, envelope: dict, captured: dict, scopes: dict):
    recipe = open_recipe(envelope)
    value = recipe['input']
    scope, _case = scope_case(
        root, value['scope_id'], captured['pandas_major'], scopes
    )
    expected_binding = {
        'scope_id': value['scope_id'],
        'corrected': value['corrected'] and captured['selector'] == 'python',
        **scope['oracle'],
        'source_sha256': scope['source_sha256'],
        'intervention': scope['intervention'],
    }
    require(
        same_json(captured['reference'], expected_binding),
        'reference.execution_binding',
    )
    require(
        captured['family'] == 'fixture-operation-v1'
        and type(captured['fixture_operation_calls']) is int
        and captured['fixture_operation_calls'] == 1
        and captured['public_calls'] == 0
        and captured['materialized_rows'] == 0
        and not captured['legacy_branch_events']
        and captured['host_event'] is None
        and captured['dimensions'] is None,
        'reference.family_boundary',
    )
    result = captured['observation']['result']
    require(
        type(result) is dict
        and set(result) == {'family', 'operation', 'kind', 'value'},
        'reference.projection_fields',
    )
    require(
        same_json(
            captured['observation'], projection(result['kind'], result['value'])
        ),
        'reference.projection_join',
    )
    require(
        (
            captured['selector'] == 'python'
            and captured['native_calls'] == 0
            and captured['legacy_calls'] > 0
        )
        or (
            captured['selector'] == 'core'
            and captured['native_calls'] > 0
            and captured['legacy_calls'] == 0
        ),
        'reference.measured_route',
    )


def recipes(scopes: dict) -> list:
    identities = sorted(
        {identity.rsplit('/pandas-', 1)[0] for identity in scopes}
    )
    require(
        set(scopes)
        == {
            f'{identity}/pandas-{major}'
            for identity in identities
            for major in ('2', '3')
        },
        'reference.scope_lanes',
    )
    return [
        {
            'id': f'{identity}/reference-{kind}',
            'recipe': seal(
                {
                    'entrypoint': 'fixture.driver.materialize',
                    'operation': 'fixture.reference',
                    'input': {
                        'scope_id': identity,
                        'corrected': kind == 'corrected',
                    },
                }
            ),
        }
        for identity in identities
        for kind in ('original', 'corrected')
    ]


def compare(
    root: Path,
    identity: str,
    major: str,
    scopes: dict,
    original: dict,
    corrected: dict,
):
    scope, case = scope_case(root, identity, major, scopes)
    envelopes = {
        open_recipe(item['recipe'])['input']['corrected']: item['recipe']
        for item in recipes(
            {
                key: value
                for key, value in scopes.items()
                if key.rsplit('/pandas-', 1)[0] == identity
            }
        )
    }
    for flag, pair in ((False, original), (True, corrected)):
        require(set(pair) == {'python', 'core'}, 'reference.both_selectors')
        for selector, record in pair.items():
            require(
                record['selector'] == selector
                and record['pandas_major'] == major,
                'reference.lane_join',
            )
            validate(root, envelopes[flag], record, scopes)
    old = original['python']['observation']
    fixed = corrected['python']['observation']
    native = original['core']['observation']
    require(
        same_json(old, original_outcome(root, case)), 'reference.original_old'
    )
    kind = 'expected' if 'expected' in case else 'expected_error'
    require(
        same_json(native, projection(kind, case[kind])),
        'reference.governed_fixture',
    )
    require(
        same_json(native, corrected['core']['observation']),
        'reference.core_repeat',
    )
    require(not same_json(old, fixed), 'reference.relabelled_original')
    require(same_json(fixed, native), 'reference.corrected_mismatch')
    return {
        'dv': scope['dv'],
        'intervention': scope['intervention'],
        'source_sha256': scope['source_sha256'],
        'actual_old': old,
        'corrected': fixed,
        'native': native,
        'oracle': scope['oracle'],
    }


def controlled_comparison(
    root, identity, major, scopes, original, corrected, accepted_dvs
):
    result = compare(root, identity, major, scopes, original, corrected)
    mutations = {
        'wrong_correction': lambda value: value['corrected']['python'][
            'reference'
        ].update(intervention='wrong'),
        'unrelated_failure': lambda value: value['corrected']['python'][
            'reference'
        ].update(fixture_case_id='wrong'),
        'missing_native': lambda value: value['original']['core'].update(
            native_calls=0
        ),
        'changed_observation': lambda value: value['corrected']['python'][
            'observation'
        ].update(result=None),
    }
    negatives = {}
    for name, mutate in mutations.items():
        altered = copy.deepcopy({'original': original, 'corrected': corrected})
        mutate(altered)
        try:
            compare(
                root,
                identity,
                major,
                scopes,
                altered['original'],
                altered['corrected'],
            )
        except Rejected:
            negatives[name] = True
        else:
            raise Rejected('reference.negative_accepted')
    result['negative_controls'] = negatives
    contract.validate_corrected_reference(result, accepted_dvs)
    return result


def verify(
    root,
    binding,
    scopes,
    accepted_dvs,
    *,
    provider_path=None,
    provider_sha256=None,
):
    from execution import batch, capture_chunk

    require(bool(scopes), 'reference.scopes_absent')
    selected = recipes(scopes)
    executions, references = [], {}
    for major in ('2', '3'):
        records = {selector: {} for selector in ('python', 'core')}
        for start in range(0, len(selected), batch.MAX_CASES):
            execution = capture_chunk(
                root,
                selected[start : start + batch.MAX_CASES],
                major,
                binding,
                reference_scopes=scopes,
                provider_path=provider_path,
                provider_sha256=provider_sha256,
            )
            executions.append({'pandas_major': major, 'lanes': execution})
            for selector in records:
                require(
                    not (
                        set(records[selector])
                        & set(execution[selector]['cases'])
                    ),
                    'reference.duplicate_execution',
                )
                records[selector].update(execution[selector]['cases'])
        for lane_id in scopes:
            identity, _separator, lane_major = lane_id.rpartition('/pandas-')
            if lane_major != major:
                continue
            pairs = {
                kind: {
                    selector: records[selector][f'{identity}/reference-{kind}'][
                        'capture'
                    ]
                    for selector in records
                }
                for kind in ('original', 'corrected')
            }
            references[lane_id] = controlled_comparison(
                root,
                identity,
                major,
                scopes,
                pairs['original'],
                pairs['corrected'],
                accepted_dvs,
            )
    require(set(references) == set(scopes), 'reference.exact_scope_execution')
    return {
        'candidate': binding['candidate'],
        'references': references,
        'executions': executions,
    }
