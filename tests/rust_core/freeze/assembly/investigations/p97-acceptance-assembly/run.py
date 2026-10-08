# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent executable acceptance components; no BASE/LOCK is implied."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from candidate import context, host_routes
from case_joins import carry_proofs, inventory_join, observations, sweep_frozen
from checks import dialect_keys, names, run_preserved
from dimensions import template_binding
from execution import batch, capture_chunk, source
from expectations import (
    decode_semantics,
    join_recordings,
    public_expected,
    read_manifest,
    verify_pickles,
    verify_semantic_dispositions,
)
from negative_replay import exercise
from predicates import sweep
from recipes import open_recipe
from replay import execute_snapshots
from routes import replay_join
from support import (
    INVESTIGATIONS,
    Rejected,
    contract,
    json_bytes,
    load,
    require,
    same_json,
    sha,
    unique_json,
)


def recording_recipes(inputs: dict) -> list:
    recipes = {}
    lanes = {}
    for identity, case in inputs.items():
        stem, separator, major = identity.rpartition('/pandas-')
        require(
            bool(stem) and bool(separator) and major in {'2', '3'},
            'recording.lane_identity',
        )
        if stem in recipes:
            require(
                same_json(recipes[stem], case['recipe']),
                'recording.lane_recipe',
            )
        recipes[stem] = case['recipe']
        lanes.setdefault(stem, set()).add(major)
    require(
        all(majors == {'2', '3'} for majors in lanes.values()),
        'recording.both_majors',
    )
    return [
        {'id': identity, 'recipe': recipe}
        for identity, recipe in recipes.items()
    ]


def executed_cases(receipt: dict, recipes: list, candidate: str) -> dict:
    require(receipt['candidate'] == candidate, 'capture.candidate')
    expected = {case['id']: case['recipe'] for case in recipes}
    require(len(expected) == len(recipes), 'capture.duplicate_recipe')
    joined = {}
    for execution in receipt['executions']:
        major = execution['pandas_major']
        require(
            major in {'2', '3'}
            and set(execution['lanes']) == {'python', 'core'},
            'capture.lane_map',
        )
        old, core = execution['lanes']['python'], execution['lanes']['core']
        require(set(old['cases']) == set(core['cases']), 'capture.pair_map')
        require(
            type(old['process_id']) is int
            and type(core['process_id']) is int
            and old['process_id'] > 0
            and core['process_id'] > 0
            and old['process_id'] != core['process_id'],
            'capture.pair_pids',
        )
        for identity in old['cases']:
            key = f'{identity}/pandas-{major}'
            require(
                identity in expected and key not in joined,
                'capture.extra_duplicate',
            )
            pair = {}
            for selector, lane in execution['lanes'].items():
                record = lane['cases'][identity]
                require(
                    lane['source_commit'] == candidate
                    and record['id'] == identity
                    and same_json(record['recipe'], expected[identity]),
                    'capture.executed_binding',
                )
                captured = record['capture']
                require(
                    captured['selector'] == selector
                    and captured['pandas_major'] == major
                    and captured['native_sha256']
                    == lane['metadata']['native_sha256'],
                    'capture.measured_lane',
                )
                batch.validate_materialized_rows(captured)
                pair[selector] = {
                    **captured,
                    'process_id': lane['process_id'],
                    'source_sha': candidate,
                }
            value = {
                'recipe': expected[identity],
                'old': pair['python']['observation'],
                'core': pair['core']['observation'],
            }
            require(
                same_json(receipt['cases'][key], value),
                'capture.summary_execution_join',
            )
            joined[key] = pair
    require(
        set(joined)
        == {
            f'{identity}/pandas-{major}'
            for identity in expected
            for major in ('2', '3')
        }
        and set(receipt['cases']) == set(joined),
        'capture.exact_execution_map',
    )
    return joined


def capture(
    root: Path,
    binding: dict,
    recipes: list,
    templates: dict | None = None,
    *,
    provider_path: Path | None = None,
    provider_sha256: str | None = None,
) -> dict:
    require(type(recipes) is list and bool(recipes), 'capture.empty')
    identities = [case['id'] for case in recipes]
    require(len(set(identities)) == len(identities), 'capture.duplicate_case')
    results = {major: {} for major in ('2', '3')}
    executions = []
    for major in ('2', '3'):
        for start in range(0, len(recipes), batch.MAX_CASES):
            chunk = recipes[start : start + batch.MAX_CASES]
            execution = capture_chunk(
                root,
                chunk,
                major,
                binding,
                templates,
                provider_path=provider_path,
                provider_sha256=provider_sha256,
            )
            executions.append({'pandas_major': major, 'lanes': execution})
            for case in chunk:
                identity = case['id']
                results[major][identity] = {
                    'recipe': case['recipe'],
                    'old': execution['python']['cases'][identity]['capture'][
                        'observation'
                    ],
                    'core': execution['core']['cases'][identity]['capture'][
                        'observation'
                    ],
                }
    result = {
        'candidate': binding['candidate'],
        'executions': executions,
        'cases': {
            f'{identity}/pandas-{major}': value
            for major, cases in results.items()
            for identity, value in cases.items()
        },
    }
    executed_cases(result, recipes, binding['candidate'])
    return result


def sweep_cases(receipt: dict, recipes: list, inventory: dict) -> list:
    joined = executed_cases(receipt, recipes, inventory['candidate'])
    results = []
    measured_inputs = {major: set() for major in ('2', '3')}
    for case in recipes:
        envelope = case['recipe']
        recipe = open_recipe(envelope)
        if 'matrix' not in recipe:
            continue
        template = template_binding(recipe, inventory['matrix_templates'])
        require(template is not None, 'sweep.no_dimension_template')
        dimensions = {key: template[key] for key in ('role', 'backend', 'task')}
        dimensions['dtype'] = recipe['matrix']['dtype']
        value = {
            **dimensions,
            'data': envelope,
            'options': {'template': recipe['matrix']['template']},
        }
        fingerprint = contract.digest(value)
        runs = {}
        for major in ('2', '3'):
            pair = joined[f'{case["id"]}/pandas-{major}']
            for record in pair.values():
                measured = record['dimensions']
                require(
                    type(measured) is dict
                    and all(
                        measured[key] == wanted
                        for key, wanted in dimensions.items()
                    )
                    and measured['template'] == recipe['matrix']['template']
                    and measured['table'] == template['table']
                    and measured['column'] == template['column'],
                    'sweep.measured_dimensions',
                )
                record['input_sha256'] = fingerprint
                record['entrypoint'] = recipe['entrypoint']
            digest = pair['python']['dimensions']['input_data_sha256']
            require(
                digest == pair['core']['dimensions']['input_data_sha256']
                and digest not in measured_inputs[major],
                'sweep.duplicate_or_changed_actual_data',
            )
            measured_inputs[major].add(digest)
            runs[major] = pair
        results.append(
            {
                'id': case['id'],
                'recipe': envelope,
                'input': value,
                'input_sha256': fingerprint,
                'seed': recipe['seed'],
                'runs': runs,
            }
        )
    return results


def generator(
    root: Path, inventory: Path, python: Path, output: Path, environment: dict
):
    script = root / 'scripts/parity/p97_generate.py'
    require(script.is_file(), 'generator.absent')
    command = [
        str(python),
        '-B',
        str(script),
        '--inventory',
        str(inventory),
        '--seed',
        '20261005',
        '--out',
        str(output),
    ]
    batch.bounded(command, b'', environment, root)
    require(output.is_file(), 'generator.failed')
    raw = output.read_bytes()
    batch.bounded(command, b'', environment, root)
    require(
        output.read_bytes() == raw,
        'generator.nondeterministic',
    )
    document = unique_json(raw)
    require(type(document) is list and bool(document), 'generator.recipes')
    for case in document:
        require(
            set(case) == {'id', 'recipe'}, 'generator.no_expected_or_counters'
        )
        open_recipe(case['recipe'])
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'command',
        choices=[
            'capture',
            'generate',
            'sweep',
            'verify-recordings',
            'replay',
            'frozen',
            'manifest',
            'names',
            'preserved',
        ],
    )
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--candidate-binding', type=Path)
    parser.add_argument('--recipes', type=Path)
    parser.add_argument('--capture', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--case-index', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--only', default='')
    parser.add_argument('--route', choices=['rust', 'binding'])
    parser.add_argument('--python', type=Path)
    parser.add_argument('--rust-binary', type=Path)
    parser.add_argument('--native-review', type=Path)
    parser.add_argument('--postgres-provider', type=Path)
    parser.add_argument('--postgres-provider-sha256')
    arguments = parser.parse_args()
    try:
        require(
            (arguments.postgres_provider is None)
            == (arguments.postgres_provider_sha256 is None),
            'provider.arguments',
        )
        provider = {
            'provider_path': arguments.postgres_provider,
            'provider_sha256': arguments.postgres_provider_sha256,
        }
        inventory = unique_json(arguments.inventory.read_bytes())
        actual_head = (
            subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=arguments.root
            )
            .decode()
            .strip()
        )
        binding = (
            unique_json(arguments.candidate_binding.read_bytes())
            if arguments.candidate_binding
            else None
        )
        if arguments.command in {
            'capture',
            'sweep',
            'verify-recordings',
            'replay',
            'frozen',
            'manifest',
        }:
            require(binding is not None, 'candidate.execution_binding_required')
            context(arguments.root, binding)
        inventory = {**inventory, 'candidate': actual_head}
        if binding is not None:
            require(binding['candidate'] == actual_head, 'candidate.head')
        source(arguments.root, inventory['candidate'])
        if arguments.command == 'generate':
            require(
                arguments.python is not None and arguments.out is not None,
                'generator.arguments',
            )
            with tempfile.TemporaryDirectory(
                prefix='p97-generator-'
            ) as temporary:
                environment = batch.environment(
                    arguments.root,
                    arguments.python.parent.parent,
                    Path(temporary),
                    'core',
                    '3',
                )
                result = generator(
                    arguments.root,
                    arguments.inventory,
                    arguments.python,
                    arguments.out,
                    environment,
                )
            source(arguments.root, inventory['candidate'])
            return 0
        elif arguments.command == 'capture':
            result = capture(
                arguments.root,
                binding,
                unique_json(arguments.recipes.read_bytes()),
                inventory.get('matrix_templates', {}),
                **provider,
            )
        elif arguments.command == 'sweep':
            require(
                arguments.recipes is not None and arguments.recipes.is_file(),
                'sweep.recipes_absent',
            )
            recipes = unique_json(arguments.recipes.read_bytes())
            retained = capture(
                arguments.root,
                binding,
                recipes,
                inventory['matrix_templates'],
                **provider,
            )
            measured = sweep_cases(retained, recipes, inventory)
            expected = sweep(
                inventory,
                measured,
                inventory['decisions'],
                host_routes(inventory['case_routes'], binding),
            )
            result = {
                'criterion': 'P97.sweep',
                'expected': expected,
                'capture': retained,
            }
        elif arguments.command == 'names':
            names(
                arguments.root,
                inventory.get('name_exemptions', []),
                inventory.get('accepted_dvs', []),
            )
            dialect_keys(
                arguments.root, inventory.get('dialect_keys_disposition', {})
            )
            result = {
                'criterion': 'P97.names',
                'candidate': inventory['candidate'],
            }
        elif arguments.command == 'preserved':
            return run_preserved(arguments.root, arguments.only.split(','))
        else:
            arguments.manifest = (
                arguments.manifest
                or arguments.root / 'tests/parity/frozen-public/manifest.json'
            )
            require(arguments.manifest.is_file(), 'manifest.absent')
            manifest = unique_json(arguments.manifest.read_bytes())
            contents = read_manifest(
                arguments.root, manifest, inventory['locked_files']
            )
            discovered = contract.fixture_files(arguments.root, inventory)
            require(
                set(discovered) == set(contents),
                'manifest.exact_discovered_files',
            )
            if arguments.command == 'manifest':
                decoded = decode_semantics(contents, arguments.root)
                verify_semantic_dispositions(manifest, contents, decoded)
                routes = host_routes(inventory['replay_routes'], binding)
                require(
                    set(routes) == set(decoded), 'manifest.independent_case_map'
                )
                require(
                    arguments.native_review is not None
                    and arguments.artifacts is not None
                    and arguments.rust_binary is not None,
                    'manifest.measurement_arguments',
                )
                require(
                    sha(arguments.native_review.read_bytes())
                    == binding['native_review_sha256'],
                    'manifest.review_binding',
                )
                helper = load(
                    '_p97_scaffold_measurement',
                    INVESTIGATIONS
                    / 'p97-scaffold-measurement-staging/scaffold_measurement.py',
                )
                report = helper.measure(
                    arguments.root,
                    manifest,
                    contents,
                    binding,
                    arguments.artifacts,
                    routes=routes,
                    binary=arguments.rust_binary,
                    no_python_binding=binding['no_python_binding'],
                    review=unique_json(arguments.native_review.read_bytes()),
                    toolchain={
                        key: Path(value).expanduser()
                        for key, value in binding['coverage_toolchain'].items()
                    },
                    expected=decoded,
                )
                contract.validate_scaffold_report(
                    report, inventory['candidate']
                )
                result = {
                    'criterion': 'P97.manifest',
                    'file_count': len(contents),
                    'candidate': inventory['candidate'],
                    'scaffold_report': report,
                }
            elif arguments.command == 'verify-recordings':
                _expected, inputs = public_expected(contents)
                recipes = recording_recipes(inputs)
                retained = capture(
                    arguments.root,
                    binding,
                    recipes,
                    inventory.get('matrix_templates', {}),
                    **provider,
                )
                join_recordings(
                    inputs,
                    retained['cases'],
                    inventory['accepted_dvs'],
                    inventory['approved_difference_scopes'],
                )
                result = {
                    'criterion': 'P97.match',
                    'case_count': len(inputs),
                    'candidate': inventory['candidate'],
                }
            else:
                expected = decode_semantics(contents, arguments.root)
                routes = host_routes(inventory['replay_routes'], binding)
                require(
                    set(routes) == set(expected), 'replay.independent_case_map'
                )
                review = (
                    unique_json(arguments.native_review.read_bytes())
                    if arguments.native_review
                    else None
                )
                if review is not None:
                    require(
                        sha(arguments.native_review.read_bytes())
                        == binding['native_review_sha256'],
                        'replay.review_authority_binding',
                    )
                selections = (
                    ('rust', 'binding')
                    if arguments.command == 'frozen'
                    else (arguments.route,)
                )
                require(None not in selections, 'replay.route_required')
                executions = {
                    route: execute_snapshots(
                        arguments.root,
                        contents,
                        routes,
                        manifest=manifest,
                        route=route,
                        python=arguments.python,
                        binary=arguments.rust_binary,
                        native_sha256=binding['native_build']['native_sha256'],
                        no_python_binding=binding.get('no_python_binding'),
                        review=review,
                    )
                    for route in selections
                }
                if arguments.command == 'frozen':
                    verify_semantic_dispositions(manifest, contents, expected)
                    require(
                        arguments.case_index is not None,
                        'freeze.case_index_required',
                    )
                    _public_expected, inputs = public_expected(contents)
                    from generator_records import (
                        authenticate,
                    )
                    from generator_records import (
                        decode as decode_generators,
                    )

                    _generator_expected, generator_inputs = decode_generators(
                        contents
                    )
                    require(
                        not set(inputs).intersection(generator_inputs),
                        'freeze.generator_public_overlap',
                    )
                    case_index = unique_json(arguments.case_index.read_bytes())
                    inventory_join(
                        arguments.root,
                        inventory,
                        case_index,
                        expected,
                        {**inputs, **generator_inputs},
                    )
                    executions['original_generators'] = authenticate(
                        arguments.root,
                        binding,
                        inventory,
                        case_index['generators'],
                        generator_inputs,
                    )
                    recipes = recording_recipes(inputs)
                    public_run = capture(
                        arguments.root,
                        binding,
                        recipes,
                        inventory.get('matrix_templates', {}),
                        **provider,
                    )
                    join_recordings(
                        inputs,
                        public_run['cases'],
                        inventory['accepted_dvs'],
                        inventory['approved_difference_scopes'],
                    )
                    joined = executed_cases(
                        public_run, recipes, inventory['candidate']
                    )
                    executions['public_capture'] = public_run
                    executions['carry_proofs'] = carry_proofs(
                        arguments.root,
                        inventory,
                        inputs,
                        joined,
                        observations(expected),
                    )
                    from references import verify as verify_references

                    reference_scopes = inventory.get(
                        'corrected_reference_sources', {}
                    )
                    require(
                        set(reference_scopes)
                        == set(inventory.get('corrected_reference_ids', []))
                        and bool(reference_scopes),
                        'reference.locked_scope_map',
                    )
                    executions['corrected_references'] = verify_references(
                        arguments.root,
                        binding,
                        reference_scopes,
                        inventory['accepted_dvs'],
                        **provider,
                    )
                    matrix_recipes = [
                        case
                        for case in recipes
                        if 'matrix' in open_recipe(case['recipe'])
                    ]
                    matrix_ids = {case['id'] for case in matrix_recipes}
                    matrix_expected = sweep(
                        inventory,
                        sweep_cases(public_run, recipes, inventory),
                        inventory['decisions'],
                        host_routes(inventory['case_routes'], binding),
                    )
                    require(matrix_ids, 'freeze.sweep_absent')
                    sweep_frozen(matrix_expected, expected)
                    replay_join(
                        expected,
                        routes,
                        executions['binding']['result'],
                        executions['rust']['result'],
                        contents,
                    )
                    pickle_exit = run_preserved(arguments.root, ['P9.pickle'])
                    verify_pickles(
                        manifest,
                        contents,
                        inventory['pickle_contract'],
                        {'P9.pickle': pickle_exit},
                    )
                    for selected_route in selections:

                        def run_negative(
                            changed, selected_route=selected_route
                        ):
                            return execute_snapshots(
                                arguments.root,
                                changed,
                                routes,
                                manifest=manifest,
                                route=selected_route,
                                python=arguments.python,
                                binary=arguments.rust_binary,
                                native_sha256=binding['native_build'][
                                    'native_sha256'
                                ],
                                no_python_binding=binding.get(
                                    'no_python_binding'
                                ),
                                review=review,
                            )

                        native_ids = (
                            {
                                key
                                for key, value in routes.items()
                                if value['kind'] == 'native_semantic'
                            }
                            if selected_route == 'rust'
                            else None
                        )
                        executions[selected_route]['negative_controls'] = (
                            exercise(
                                arguments.root,
                                contents,
                                run_negative,
                                native_ids,
                            )
                        )
                else:
                    actual = executions[arguments.route]['result']
                    native_expected = {
                        key: value
                        for key, value in expected.items()
                        if arguments.route == 'binding'
                        or routes[key]['kind'] == 'native_semantic'
                    }
                    require(
                        same_json(actual['observations'], native_expected),
                        'replay.wrong_observation',
                    )
                result = {
                    'criterion': 'P97.frozen'
                    if arguments.command == 'frozen'
                    else 'development-replay',
                    'executions': executions,
                    'candidate': inventory['candidate'],
                }
        source(arguments.root, inventory['candidate'])
        require(arguments.out is not None, 'driver.output_required')
        arguments.out.write_bytes(json_bytes(result) + b'\n')
    except Rejected as error:
        print(f'REJECTED: {error}')
        return 1
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
    ):
        print('REJECTED: assembly.invalid_or_unavailable_input')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
