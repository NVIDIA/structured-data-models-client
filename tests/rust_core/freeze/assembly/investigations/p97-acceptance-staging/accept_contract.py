# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staged P97 evidence predicates; not a phase lock or a replay runner."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any

ROLES = {'primary_key', 'foreign_key', 'time_column', 'feature', 'target'}
BACKENDS = {'memory', 'duckdb', 'sqlite'}
TASKS = {'regression', 'classification', 'ranking', 'link_prediction'}
MAJORS = {'2', '3'}
CARRY = {f'C{number:02d}' for number in (*range(1, 16), 22, 24, 29)}
EDGES = {
    'float_keys_near_1e30',
    'unicode_date_digits',
    'arabic_indic_date_digits',
    'uint64_above_i64',
    'arrow_time_names',
    'arrow_duration_names',
}
LEGACY = {'sdm_relational_engine', 'predictive_query', 'relationallib'}
OBSERVABLES = {
    'status',
    'schema',
    'payloads',
    'result',
    'sql',
    'error',
    'warnings',
}


class Rejected(ValueError):
    """An evidence predicate failed without echoing potentially unsafe values."""


def require(condition: bool, category: str) -> None:
    if not condition:
        raise Rejected(category)


def digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def path_ok(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and '..' not in path.parts


def exact_keys(value: dict[str, Any], keys: set[str], category: str) -> None:
    require(isinstance(value, dict) and set(value) == keys, category)


def observation(value: dict[str, Any]) -> None:
    exact_keys(value, OBSERVABLES, 'observation.fields')
    require(value['status'] in {'success', 'refused'}, 'observation.status')
    require(isinstance(value['schema'], list), 'observation.schema')
    for column in value['schema']:
        exact_keys(
            column,
            {'table', 'column', 'dtype', 'stype', 'is_time'},
            'observation.column',
        )
    require(isinstance(value['payloads'], list), 'observation.payloads')
    require(isinstance(value['sql'], list), 'observation.sql')
    for query in value['sql']:
        exact_keys(query, {'statement', 'parameters'}, 'observation.sql_fields')
    require(isinstance(value['warnings'], list), 'observation.warnings')
    for warning in value['warnings']:
        exact_keys(
            warning, {'class', 'code', 'message', 'details'}, 'warning.fields'
        )
    if value['status'] == 'refused':
        exact_keys(
            value['error'],
            {'class', 'code', 'message', 'details'},
            'error.fields',
        )
        require(bool(value['error']['class']), 'error.class')
    else:
        require(value['error'] is None, 'error.unexpected')
    digest(value)


def validate_sweep(
    inventory: dict[str, Any],
    cases: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    require(
        re.fullmatch(r'[0-9a-f]{40}', inventory['base']) is not None, 'base'
    )
    require(set(inventory['roles']) == ROLES, 'inventory.roles')
    require(set(inventory['backends']) == BACKENDS, 'inventory.backends')
    require(set(inventory['tasks']) == TASKS, 'inventory.tasks')
    require(set(inventory['pandas_majors']) == MAJORS, 'inventory.pandas')
    dtype_ids = inventory['dtype_ids']
    require(
        bool(dtype_ids) and len(dtype_ids) == len(set(dtype_ids)),
        'inventory.dtypes',
    )
    require(set(inventory['carry_ids']) == CARRY, 'inventory.carry')
    require(set(inventory['edge_ids']) == EDGES, 'inventory.edges')
    support = inventory['supported_controls']
    require(
        set(support)
        == {
            f'{backend}/{task}'
            for backend, task in itertools.product(BACKENDS, TASKS)
        },
        'inventory.support',
    )
    require(
        all(type(value) is bool for value in support.values()),
        'inventory.support_types',
    )
    require(any(support.values()), 'inventory.no_success_support')
    require(len(cases) >= 2000, 'sweep.minimum')
    ids: set[str] = set()
    inputs: set[str] = set()
    cells: set[tuple[str, str]] = set()
    controls: set[tuple[str, str]] = set()
    expected: dict[str, Any] = {}
    differences: dict[tuple[str, str], tuple[str, str]] = {}
    for case in cases:
        case_id = case['id']
        require(
            isinstance(case_id, str) and bool(case_id) and case_id not in ids,
            'sweep.duplicate_id',
        )
        ids.add(case_id)
        value = case['input']
        exact_keys(
            value,
            {'dtype', 'role', 'backend', 'task', 'data', 'options'},
            'input.fields',
        )
        require(
            value['dtype'] in dtype_ids and value['role'] in ROLES,
            'input.dtype_role',
        )
        require(
            value['backend'] in BACKENDS and value['task'] in TASKS,
            'input.backend_task',
        )
        fingerprint = digest(value)
        require(case['input_sha256'] == fingerprint, 'input.hash')
        require(fingerprint not in inputs, 'sweep.duplicate_input')
        inputs.add(fingerprint)
        require(type(case['seed']) is int, 'input.seed')
        cells.add((value['dtype'], value['role']))
        require(set(case['runs']) == MAJORS, 'sweep.both_majors')
        for major, pair in case['runs'].items():
            require(set(pair) == {'python', 'core'}, 'sweep.both_paths')
            for selector, run in pair.items():
                require(run['selector'] == selector, 'provenance.selector')
                require(run['pandas_major'] == major, 'provenance.pandas')
                require(run['input_sha256'] == fingerprint, 'provenance.input')
                require(
                    run['entrypoint']
                    in {
                        'PredictiveQuery.materialize',
                        'SDMClient.predict',
                        'FittedModel.predict',
                    },
                    'provenance.public',
                )
                require(run['public_calls'] > 0, 'provenance.public_calls')
                require(
                    run['source_sha'] == inventory['candidate'],
                    'provenance.sha',
                )
                require(
                    run['process_id']
                    != pair['core' if selector == 'python' else 'python'][
                        'process_id'
                    ],
                    'provenance.isolation',
                )
                if selector == 'python':
                    require(
                        run['legacy_calls'] > 0 and run['native_calls'] == 0,
                        'provenance.oracle',
                    )
                else:
                    require(
                        run['native_calls'] > 0 and run['legacy_calls'] == 0,
                        'provenance.native',
                    )
                observation(run['observation'])
            old, core = (
                pair['python']['observation'],
                pair['core']['observation'],
            )
            expected[f'{case_id}/pandas-{major}'] = core
            if digest(old) != digest(core):
                differences[(case_id, major)] = (digest(old), digest(core))
            if (
                old['status'] == core['status'] == 'success'
                and old['payloads']
                and core['payloads']
                and pair['python']['materialized_rows'] > 1
                and pair['core']['materialized_rows'] > 1
                and (
                    value['backend'] == 'memory' or (old['sql'] and core['sql'])
                )
            ):
                controls.add((f'{value["backend"]}/{value["task"]}', major))
    require(
        cells == set(itertools.product(dtype_ids, ROLES)), 'sweep.missing_cell'
    )
    require(
        all(
            (cell, major) in controls
            for cell, supported in support.items()
            if supported
            for major in MAJORS
        ),
        'sweep.missing_success',
    )
    accepted: dict[tuple[str, str], tuple[str, str]] = {}
    for decision in decisions:
        key = (decision['case_id'], decision['pandas_major'])
        require(key not in accepted, 'dv.duplicate')
        require(decision['dv'] in inventory['accepted_dvs'], 'dv.unaccepted')
        require(bool(decision['reason'].strip()), 'dv.reason')
        require(key in differences, 'dv.stale_or_wildcard')
        accepted[key] = (decision['old_sha256'], decision['core_sha256'])
    require(accepted == differences, 'parity.unapproved_or_wrong_difference')
    return expected


def validate_carry_and_generators(
    inventory: dict[str, Any],
    frozen: dict[str, Any],
) -> None:
    require(set(frozen['carry']) == CARRY, 'freeze.carry_inventory')
    require(set(frozen['edges']) == EDGES, 'freeze.edge_inventory')
    require(
        set(frozen['generators']) == set(inventory['generators']),
        'freeze.generator_inventory',
    )
    require(frozen['carry'] == inventory['carry_cases'], 'freeze.carry_mapping')
    require(frozen['edges'] == inventory['edge_cases'], 'freeze.edge_mapping')
    expected = frozen['expected']
    for mapping in (frozen['carry'], frozen['edges']):
        for identities in mapping.values():
            require(
                bool(identities) and set(identities) <= set(expected),
                'freeze.missing_case',
            )
    for name, settings in inventory['generators'].items():
        records = frozen['generators'][name]
        require(
            len(records) >= settings['minimum_samples'],
            'freeze.generator_minimum',
        )
        seeds = [record['seed'] for record in records]
        require(len(seeds) == len(set(seeds)), 'freeze.duplicate_seed')
        require(
            len({record['case_id'] for record in records}) == len(records),
            'freeze.duplicate_sample',
        )
        require(
            set(settings['historical_mismatch_seeds']) <= set(seeds),
            'freeze.historical_seed',
        )
        require(
            all(record['case_id'] in expected for record in records),
            'freeze.generator_case',
        )
    require(bool(frozen['sampling_cases']), 'freeze.sampling_empty')
    require(
        set(frozen['sampling_cases']) == set(inventory['sampling_case_ids']),
        'freeze.sampling_inventory',
    )
    require(
        set(frozen['sampling_cases']) <= set(expected), 'freeze.sampling_case'
    )
    for value in expected.values():
        observation(value)


def fixture_files(root: Path, inventory: dict[str, Any]) -> dict[str, str]:
    found: dict[str, str] = {}
    for prefix in inventory['fixture_roots']:
        require(path_ok(prefix), 'manifest.root_path')
        folder = root / prefix
        require(
            folder.is_dir() and not folder.is_symlink(), 'manifest.root_missing'
        )
        for path in sorted(folder.rglob('*')):
            require(not path.is_symlink(), 'manifest.symlink')
            if path.is_file() and path.suffix in inventory['fixture_suffixes']:
                found[path.relative_to(root).as_posix()] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    require(bool(found), 'manifest.empty')
    require(
        set(inventory['fixture_paths_at_base']) <= set(found),
        'manifest.base_file_missing',
    )
    return found


def validate_frozen(
    inventory: dict[str, Any],
    manifest: list[dict[str, Any]],
    files: dict[str, str],
    expected: dict[str, Any],
    reports: list[dict[str, Any]],
) -> None:
    indexed = {entry['path']: entry for entry in manifest}
    require(len(indexed) == len(manifest), 'manifest.duplicate')
    require(set(indexed) == set(files), 'manifest.missing_or_extra')
    for path, entry in indexed.items():
        require(path_ok(path), 'manifest.path')
        require(entry['sha256'] == files[path], 'manifest.hash')
        require(
            entry['kind'] in {'contract', 'scaffold'}, 'manifest.classification'
        )
        require(set(entry['depends_on']) <= set(indexed), 'manifest.transitive')
    require(
        set(report['route'] for report in reports) == {'rust', 'binding'}
        and len(reports) == 2,
        'frozen.routes',
    )
    for report in reports:
        require(report['sha'] == inventory['candidate'], 'frozen.sha')
        require(
            report['exit_code'] == 0 and report['skipped'] == 0,
            'frozen.failed_or_skipped',
        )
        require(
            report['expected_sha256'] == digest(expected),
            'frozen.same_expectations',
        )
        require(
            set(report['files_consumed']) == set(files),
            'frozen.unconsumed_file',
        )
        require(
            len(report['files_consumed']) == len(files), 'frozen.duplicate_file'
        )
        require(set(report['observed']) == set(expected), 'frozen.missing_case')
        require(
            digest(report['observed']) == digest(expected),
            'frozen.wrong_output',
        )
        if report['route'] == 'binding':
            require(
                report['absent_modules'] == sorted(LEGACY),
                'frozen.oracle_present',
            )
            require(
                report['absent_distributions']
                == ['sdm-pql', 'sdm-relational-engine'],
                'frozen.distribution_present',
            )
            require(
                report['native_calls'] >= len(expected), 'frozen.native_calls'
            )
        else:
            require(report['python_processes'] == 0, 'frozen.rust_uses_python')


def validate_names(
    sources: dict[str, str],
    exemptions: list[dict[str, Any]],
    accepted_dvs: list[str],
) -> None:
    require(bool(sources), 'names.no_sources')
    hits = {
        (path, number, digest(line))
        for path, content in sources.items()
        for number, line in enumerate(content.splitlines(), 1)
        if any(name in line for name in LEGACY)
    }
    approved = set()
    for entry in exemptions:
        require(
            entry['dv'] in accepted_dvs and bool(entry['reason'].strip()),
            'names.unapproved',
        )
        key = (entry['path'], entry['line'], entry['line_sha256'])
        require(key not in approved and key in hits, 'names.stale_or_duplicate')
        approved.add(key)
    require(approved == hits, 'names.legacy_reference')


def validate_corrected_reference(
    value: dict[str, Any], accepted_dvs: list[str]
) -> None:
    require(value['dv'] in accepted_dvs, 'corrected.unaccepted')
    require(
        digest(value['actual_old']) != digest(value['corrected']),
        'corrected.relabelled_old',
    )
    for outcome in (value['actual_old'], value['corrected'], value['native']):
        observation(outcome)
    require(
        digest(value['corrected']) == digest(value['native']),
        'corrected.mismatch',
    )
    require(
        bool(value['intervention'])
        and re.fullmatch(r'[0-9a-f]{64}', value['source_sha256']) is not None,
        'corrected.provenance',
    )
    require(
        set(value['negative_controls'])
        == {
            'wrong_correction',
            'unrelated_failure',
            'missing_native',
            'changed_observation',
        },
        'corrected.controls',
    )
    require(
        all(result is True for result in value['negative_controls'].values()),
        'corrected.negative_failed',
    )


def validate_corrected_references(
    inventory: dict[str, Any],
    references: dict[str, Any],
    cases: list[dict[str, Any]],
    proofs: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    sources = inventory['corrected_reference_sources']
    require(
        set(references)
        == set(sources)
        == set(inventory['corrected_reference_ids']),
        'bundle.corrected_inventory',
    )
    retained = {
        f'{case["id"]}/pandas-{major}': (case['id'], major, pair['python'])
        for case in cases
        for major, pair in case['runs'].items()
    }
    for identity, reference in references.items():
        validate_corrected_reference(reference, inventory['accepted_dvs'])
        scope = sources[identity]
        exact_keys(
            scope,
            {'oracle', 'dv', 'intervention', 'source_sha256'},
            'corrected.scope_fields',
        )
        require(
            all(
                reference[key] == scope[key]
                for key in ('dv', 'intervention', 'source_sha256')
            ),
            'corrected.scope_disagree',
        )
        source = scope['oracle']
        if identity in retained:
            case_id, major, run = retained[identity]
            require(
                source
                == {'kind': 'sweep', 'case_id': case_id, 'pandas_major': major},
                'corrected.oracle_mapping',
            )
            actual_old = run['observation']
        else:
            exact_keys(
                source,
                {'kind', 'carry_id', 'case_id'},
                'corrected.oracle_mapping',
            )
            require(
                source['kind'] == 'carry'
                and source['carry_id'] in proofs
                and source['case_id'] == identity
                and identity in proofs[source['carry_id']],
                'corrected.oracle_mapping',
            )
            actual_old = proofs[source['carry_id']][identity]['actual_old']
        require(
            digest(reference['actual_old']) == digest(actual_old),
            'corrected.old_disagree',
        )
        require(
            identity in expected
            and digest(reference['native']) == digest(expected[identity]),
            'bundle.corrected_frozen_disagree',
        )


def validate_carry_proofs(
    inventory: dict[str, Any], proofs: dict[str, Any], expected: dict[str, Any]
) -> None:
    require(set(proofs) == CARRY, 'carry.proof_inventory')
    for carry_id, required_cases in inventory['carry_cases'].items():
        require(
            set(proofs[carry_id]) == set(required_cases), 'carry.proof_cases'
        )
        for case_id, proof in proofs[carry_id].items():
            require(proof['sha'] == inventory['candidate'], 'carry.sha')
            require(
                proof['entrypoint'].startswith(
                    (
                        'Schema.',
                        'PredictiveQuery.',
                        'SDMClient.',
                        'FittedModel.',
                        'PredictionResult.',
                    )
                ),
                'carry.public',
            )
            require(
                proof['python_legacy_calls'] > 0
                and proof['core_native_calls'] > 0,
                'carry.provenance',
            )
            require(
                proof['python_native_calls'] == 0
                and proof['core_legacy_calls'] == 0,
                'carry.mixed_path',
            )
            observation(proof['actual_old'])
            observation(proof['native'])
            require(
                case_id in expected
                and digest(proof['native']) == digest(expected[case_id]),
                'carry.frozen_disagree',
            )
            if digest(proof['actual_old']) != digest(proof['native']):
                require(
                    proof['dv'] in inventory['accepted_dvs'],
                    'carry.unapproved_difference',
                )
                require(
                    proof['decision_old_sha256'] == digest(proof['actual_old'])
                    and proof['decision_core_sha256']
                    == digest(proof['native']),
                    'carry.decision_hash',
                )
            else:
                require(proof['dv'] is None, 'carry.stale_difference')
            if carry_id == 'C24':
                require(
                    proof['backend'] == 'postgres'
                    and proof['entrypoint'] == 'Schema.from_source',
                    'carry.c24_public_postgres',
                )
            if carry_id == 'C29':
                require(
                    proof['actual_old']['status']
                    == proof['native']['status']
                    == 'success',
                    'carry.c29_success',
                )


def validate_scaffold_report(report: dict[str, Any], candidate: str) -> None:
    require(report['sha'] == candidate, 'scaffold.sha')
    require(
        report['measurement'] == 'instrumented-core-branches',
        'scaffold.unmeasured',
    )
    require(bool(report['measurement_artifacts']), 'scaffold.no_artifact')
    contract = set(report['contract_branches'])
    scaffold = set(report['scaffold_branches'])
    require(
        set(report['scaffold_only_branches']) == scaffold - contract,
        'scaffold.wrong_difference',
    )
