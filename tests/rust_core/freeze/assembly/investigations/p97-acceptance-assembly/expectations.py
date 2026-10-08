# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Byte-closed manifest joins and independently decoded executable public cases."""

from __future__ import annotations

from pathlib import Path

import fixture_inputs
from recipes import open_recipe
from support import (
    INVESTIGATIONS,
    checked_path,
    contract,
    load,
    require,
    same_json,
    sha,
    unique_json,
)


def read_manifest(root: Path, manifest: list, locked_files: dict) -> dict:
    require(type(manifest) is list and bool(manifest), 'manifest.empty')
    records = {}
    contents = {}
    for record in manifest:
        require(
            type(record) is dict
            and set(record)
            == {
                'path',
                'sha256',
                'kind',
                'depends_on',
                'semantic_cases',
                'artifact_assertion',
            },
            'manifest.fields',
        )
        name = record['path']
        require(name not in records, 'manifest.duplicate')
        require(record['kind'] in {'contract', 'scaffold'}, 'manifest.kind')
        for key in ('depends_on', 'semantic_cases'):
            require(
                type(record[key]) is list
                and all(type(item) is str for item in record[key])
                and len(record[key]) == len(set(record[key])),
                'manifest.list',
            )
        raw = checked_path(root, name).read_bytes()
        require(sha(raw) == record['sha256'], 'manifest.digest')
        contents[name] = raw
        records[name] = record
    require(set(locked_files) <= set(records), 'manifest.base_omitted')
    for name, binding in locked_files.items():
        require(
            contents[name] and records[name]['sha256'] == binding['sha256'],
            'manifest.base_changed',
        )
        require(
            set(binding['depends_on']) <= set(records[name]['depends_on']),
            'manifest.base_dependency',
        )
        require(records[name]['kind'] == binding['kind'], 'manifest.base_kind')
    dependency_inputs = fixture_inputs.dispositions(records, contents)
    for name, record in records.items():
        require(
            bool(record['semantic_cases'])
            or record['artifact_assertion'] is not None
            or bool(record['depends_on'])
            or name in dependency_inputs,
            'manifest.no_disposition',
        )
        require(
            set(record['depends_on']) <= set(records),
            'manifest.dependency_missing',
        )
        require(name not in record['depends_on'], 'manifest.self_dependency')
    return contents


def verify_pickles(
    records: list, contents: dict, locked: dict, executed_checks: dict
) -> None:
    by_name = {record['path']: record for record in records}
    expected_path = locked['pickle_expected_path']
    require(
        sha(contents[expected_path]) == locked['pickle_expected_sha256'],
        'pickle.expected_changed',
    )
    expected = unique_json(contents[expected_path])
    wanted = {record['path'] for record in locked['pickles']}
    require(
        {name for name in contents if name.endswith('.pkl')} == wanted,
        'pickle.files',
    )
    for binding in locked['pickles']:
        name, identity = binding['path'], binding['fixture_identity']
        record = by_name[name]
        require(
            sha(contents[name]) == binding['sha256'] and identity in expected,
            'pickle.binding',
        )
        require(
            record['artifact_assertion'] == f'P9.pickle/{identity}'
            and record['semantic_cases'] == []
            and expected_path in record['depends_on'],
            'pickle.disposition',
        )
    require(executed_checks.get('P9.pickle') == 0, 'pickle.preserved_check')


METADATA_ASSERTIONS = {
    'tests/golden/public_api.json': 'P9.public',
    'tests/parity/schema/transcript.schema.json': 'P9.acceptance',
    'rust/crates/wire/tests/edge/contract.json': 'P97.wire-generator+P9.rust-test',
    'tests/rust_core/switch/pickles/expected.json': 'P9.pickle',
}


def verify_metadata_dispositions(manifest: list):
    by_name = {item['path']: item for item in manifest}
    for name, assertion in METADATA_ASSERTIONS.items():
        require(
            name in by_name
            and by_name[name]['artifact_assertion'] == assertion
            and by_name[name]['semantic_cases'] == [],
            'manifest.metadata_disposition',
        )
    for item in manifest:
        if (
            item['artifact_assertion'] is not None
            and item['path'] not in METADATA_ASSERTIONS
        ):
            require(
                item['path'].endswith('.pkl')
                and item['artifact_assertion'].startswith('P9.pickle/'),
                'manifest.semantic_reclassified_as_artifact',
            )


def verify_semantic_dispositions(
    manifest: list, contents: dict, expected: dict
):
    verify_metadata_dispositions(manifest)
    require(
        {identity for item in manifest for identity in item['semantic_cases']}
        == set(expected),
        'manifest.semantic_case_union',
    )
    for item in manifest:
        name = item['path']
        if not name.endswith('.json') or name in METADATA_ASSERTIONS:
            continue
        document = unique_json(contents[name])
        if name.startswith('tests/parity/requests/') and name.endswith(
            '/calls.json'
        ):
            wanted = {
                f'requests/{document["scenario"]}/call-{ordinal}'
                for ordinal in range(len(document['calls']))
            }
        elif name.startswith('tests/parity/sampling/') and name.endswith(
            '/boundary.json'
        ):
            wanted = {
                f'sampling/{document["scenario"]}/call-{ordinal}'
                for ordinal in range(len(document['calls']))
            }
        elif document.get('format') in {'p97-public-v2', 'p97-generators-v1'}:
            wanted = {case['id'] for case in document['cases']}
        else:
            wanted = {
                identity
                for identity in expected
                if identity == name or identity.startswith(name + '::')
            }
        require(
            (
                bool(wanted)
                or fixture_inputs.empty_boundary(name, contents[name])
            )
            and set(item['semantic_cases']) == wanted,
            'manifest.semantic_owner',
        )


def public_expected(contents: dict[str, bytes]) -> tuple[dict, dict]:
    expected, inputs = {}, {}
    for name, raw in contents.items():
        if not name.startswith(
            'tests/parity/frozen-public/'
        ) or not name.endswith('.json'):
            continue
        if name.endswith('/manifest.json'):
            continue
        document = unique_json(raw)
        require(
            set(document) == {'format', 'cases'}
            and document['format'] == 'p97-public-v2',
            'recording.format',
        )
        for case in document['cases']:
            require(
                set(case)
                == {
                    'id',
                    'recipe',
                    'actual_old',
                    'expected',
                    'corrected_reference',
                    'replay_input',
                    'depends_on',
                },
                'recording.fields',
            )
            identity = case['id']
            require(
                type(identity) is str and identity not in expected,
                'recording.identity',
            )
            open_recipe(case['recipe'])
            contract.observation(case['actual_old'])
            contract.observation(case['expected'])
            require(case['replay_input'] is not None, 'recording.nonexecutable')
            require(
                type(case['depends_on']) is list
                and set(case['depends_on']) <= set(contents),
                'recording.dependency',
            )
            expected[identity] = case['expected']
            inputs[identity] = case
    require(bool(expected), 'recording.absent')
    return expected, inputs


def fixture_operation_expected(
    contents: dict[str, bytes], correction=None
) -> dict:
    expected = {}
    for name, raw in contents.items():
        if not name.startswith('tests/parity/fixtures/') or not name.endswith(
            '.json'
        ):
            continue
        document = unique_json(raw)
        require(set(document) == {'source', 'cases'}, 'fixture.document')
        for case in document['cases']:
            fields = {'id', 'source', 'operation', 'input', 'expected_warnings'}
            require(
                set(case)
                in (fields | {'expected'}, fields | {'expected_error'}),
                'fixture.case_fields',
            )
            identity = f'{name}::{case["id"]}'
            require(identity not in expected, 'fixture.duplicate_case')
            if correction is not None and case['id'] == correction.CASE_ID:
                require(
                    name == correction.FIXTURE_PATH, 'fixture.correction_file'
                )
                case = correction.expected_case(case, core=True)
            expected[identity] = {
                'operation': case['operation'],
                'result': case.get('expected'),
                'error': case.get('expected_error'),
                'warnings': case['expected_warnings'],
            }
    return expected


def pql_expected(contents: dict[str, bytes]) -> dict:
    expected = {}
    for name in (
        'tests/parity/pql/valid.json',
        'tests/parity/pql/invalid.json',
    ):
        if name not in contents:
            continue
        document = unique_json(contents[name])
        require(set(document) == {'cases'}, 'pql.document')
        corrections = (
            pql_fixture_syntax_policy(name, contents[name], document)
            if name.endswith('/invalid.json')
            else {}
        )
        for case in document['cases']:
            identity = f'{name}::{case["id"]}'
            require(identity not in expected, 'pql.duplicate_case')
            if name.endswith('/valid.json'):
                require(
                    set(case)
                    in (
                        {'id', 'query', 'schema', 'profile', 'expected_tree'},
                        {
                            'id',
                            'query',
                            'schema',
                            'profile',
                            'expected_tree',
                            'expected_warnings',
                        },
                    ),
                    'pql.valid_fields',
                )
                expected[identity] = {
                    'tree': case['expected_tree'],
                    'warnings': case.get('expected_warnings', []),
                }
            else:
                require(
                    set(case)
                    == {'id', 'query', 'schema', 'profile', 'expected_errors'},
                    'pql.invalid_fields',
                )
                expected[identity] = {
                    'errors': pql_fixture_syntax_errors(case, corrections)
                }
    return expected


def dialect_expected(contents: dict[str, bytes]) -> dict:
    expected = {}
    for name, raw in contents.items():
        if not name.startswith('tests/parity/dialects/') or not name.endswith(
            '.json'
        ):
            continue
        document = unique_json(raw)
        require(
            set(document) == {'format', 'backend', 'calls'}
            and document['format'] == 1,
            'dialect.document',
        )
        for ordinal, call in enumerate(document['calls']):
            require(
                set(call)
                in (
                    {'function', 'args', 'sources', 'sql', 'params'},
                    {'function', 'args', 'sources', 'result'},
                ),
                'dialect.output_fields',
            )
            expected[f'{name}::call-{ordinal}'] = (
                {'sql': call['sql'], 'params': call['params']}
                if 'sql' in call
                else {'result': call['result']}
            )
    return expected


def transcript_expected(contents: dict[str, bytes]) -> dict:
    expected = {}
    for name, raw in contents.items():
        if not name.startswith(
            'tests/parity/transcripts/'
        ) or not name.endswith('/transcript.json'):
            continue
        document = unique_json(raw)
        require(
            {
                'format_version',
                'name',
                'backend',
                'task_kind',
                'data',
                'calls',
                'steps',
                'payloads',
                'responses',
                'result',
                'warnings',
            }
            <= set(document),
            'transcript.fields',
        )
        expected[name] = {
            key: document[key]
            for key in ('steps', 'payloads', 'responses', 'result', 'warnings')
        }
    return expected


def pql_crate_expected(contents: dict[str, bytes]) -> dict:
    expected = {}
    documents = {
        'rust/crates/pql/tests/accept_quirks.json': ('queries', 'dates'),
        'rust/crates/pql/tests/custom_profiles.json': ('profiles', 'options'),
    }
    for name, sections in documents.items():
        if name not in contents:
            continue
        document = unique_json(contents[name])
        for section in sections:
            for case in document[section]:
                identity = f'{name}::{section}/{case["id"]}'
                require(
                    identity not in expected and 'python' in case,
                    'pql_crate.case',
                )
                expected[identity] = case['python']
    return expected


def decode_semantics(contents: dict[str, bytes], repo: Path) -> dict:
    expected = {}

    def merge(family, cases):
        require(not set(cases).intersection(expected), 'decode.duplicate_case')
        expected.update(
            {
                identity: {'family': family, 'observation': value}
                for identity, value in cases.items()
            }
        )

    correction_path = repo / 'tests/rust_core/request/_fixture_expectations.py'
    correction = (
        load('_p97_governed_fixture_correction', correction_path)
        if correction_path.is_file()
        else None
    )
    if correction is not None:
        entry = correction.load_correction()
        fixture = unique_json(contents[correction.FIXTURE_PATH])
        cases = [
            case
            for case in fixture['cases']
            if case['id'] == correction.CASE_ID
        ]
        require(
            len(cases) == 1 and same_json(cases[0], entry['original_case']),
            'fixture.correction_original',
        )
    merge(
        'fixture-operation-v1', fixture_operation_expected(contents, correction)
    )
    for family, decoder in (
        ('pql-compiler-v1', pql_expected),
        ('dialect-call-v1', dialect_expected),
        ('transcript-v1', transcript_expected),
        ('pql-crate-v1', pql_crate_expected),
    ):
        merge(family, decoder(contents))
    if any(
        name.startswith('tests/parity/frozen-public/')
        and not name.endswith('/manifest.json')
        and name.endswith('.json')
        for name in contents
    ):
        public, _inputs = public_expected(contents)
        merge('public-observation-v1', public)
    from generator_records import FAMILY
    from generator_records import decode as decode_generators

    generated, _generated_inputs = decode_generators(contents)
    merge(FAMILY, generated)
    subset = {
        name: raw
        for name, raw in contents.items()
        if name.startswith(('tests/parity/requests/', 'tests/parity/sampling/'))
    }
    if subset:
        import sys

        sys.path.insert(0, str(INVESTIGATIONS / 'p97-real-adapter-staging'))
        from arrow_decode import request_decoder
        from common import helpers
        from fixtures import governed_fixture_decode_v1

        _switch, _driver, request = helpers(repo)
        decoded = governed_fixture_decode_v1(subset, request_decoder(request))
        merge('recorded-native-v1', decoded)
    wire = 'tests/golden/wire_payloads.json'
    if wire in contents:
        merge(
            'wire-roundtrip-v1',
            {
                f'{wire}::{identity}': value
                for identity, value in unique_json(contents[wire]).items()
            },
        )
    require(bool(expected), 'decode.no_semantics')
    return expected


def validate_dv87(identity, recipe, old, core, scopes):
    require(type(scopes) is dict and 'DV87' in scopes, 'dv87.scope_missing')
    scope = scopes['DV87']
    require(
        identity
        in {f'{scope["case_id"]}/pandas-{major}' for major in ('2', '3')}
        and same_json(open_recipe(recipe), scope['recipe']),
        'dv87.exact_input',
    )
    for observed, error in (
        (old, scope['old_error']),
        (core, scope['core_error']),
    ):
        require(
            observed['status'] == 'refused'
            and same_json(observed['error'], error)
            and observed['warnings'] == [],
            'dv87.exact_refusal',
        )
    require(
        same_json(
            {key: value for key, value in old.items() if key != 'error'},
            {key: value for key, value in core.items() if key != 'error'},
        ),
        'dv87.other_fields_unchanged',
    )


DECIMAL_CORRECTIONS = {
    'DV131': 'whole-decimal-entity-keys',
    'DV133': 'finite-whole-decimal-target-object',
}
DECIMAL_REFUSALS = {
    'DV134': 'decimal-time-retained-refusal',
    'DV136': 'decimal-link-retained-refusal',
}


def decimal_difference(identity, recipe, old, core, correction, scopes):
    matches = [
        (decision, scope['cases'][identity])
        for decision, scope in (scopes or {}).items()
        if decision in {*DECIMAL_CORRECTIONS, *DECIMAL_REFUSALS}
        and identity in scope['cases']
    ]
    require(len(matches) <= 1, 'decimal.ambiguous_scope')
    if not matches:
        return None
    decision, entry = matches[0]
    scope = scopes[decision]
    require(same_json(recipe, entry['recipe']), 'decimal.exact_input')
    require(
        same_json(old, entry['actual_old'])
        and same_json(core, entry['expected'])
        and not same_json(old, core),
        'decimal.exact_observations',
    )
    if decision in DECIMAL_REFUSALS:
        require(
            scope['kind'] == DECIMAL_REFUSALS[decision]
            and correction is None
            and core['status'] == 'refused'
            and core['payloads'] == [],
            'decimal.retained_refusal',
        )
    else:
        require(
            scope['kind'] == 'corrected_original'
            and type(correction) is dict
            and correction['dv'] == decision
            and correction['intervention'] == DECIMAL_CORRECTIONS[decision]
            and correction['source_sha256'] == scope['source_sha256'],
            'decimal.corrected_source',
        )
        require(
            same_json(correction['actual_old'], old)
            and same_json(correction['corrected'], entry['corrected'])
            and same_json(correction['native'], core)
            and same_json(entry['corrected'], core),
            'decimal.corrected_observation',
        )
    return decision


def join_recordings(
    inputs: dict, captured: dict, accepted_dvs: list, scopes: dict | None = None
) -> None:
    require(set(inputs) == set(captured), 'recording.capture_case_map')
    for identity, case in inputs.items():
        actual = captured[identity]
        require(
            same_json(actual['recipe'], case['recipe']),
            'recording.executed_input',
        )
        require(
            same_json(actual['old'], case['actual_old']), 'recording.actual_old'
        )
        require(
            same_json(actual['core'], case['expected']), 'recording.actual_core'
        )
        finite = difference(
            identity,
            case['recipe'],
            actual['old'],
            actual['core'],
            case['corrected_reference'],
            scopes or {},
            [*accepted_dvs, *DECISIONS],
        )
        if finite and scopes[finite]['kind'] == 'retained_refusal':
            continue
        decimal = decimal_difference(
            identity,
            case['recipe'],
            actual['old'],
            actual['core'],
            case['corrected_reference'],
            scopes,
        )
        if decimal in DECIMAL_REFUSALS:
            continue
        if case['corrected_reference'] is None:
            require(
                same_json(actual['old'], actual['core']),
                'recording.unapproved_difference',
            )
        else:
            correction = case['corrected_reference']
            permitted = [*accepted_dvs, decimal] if decimal else accepted_dvs
            if finite:
                permitted = [*permitted, finite]
            if correction['dv'] == 'DV87':
                validate_dv87(
                    identity,
                    case['recipe'],
                    actual['old'],
                    actual['core'],
                    scopes,
                )
                require(
                    correction['source_sha256']
                    == scopes['DV87']['source_sha256'],
                    'dv87.correction_source',
                )
                permitted = [*accepted_dvs, 'DV87']
            contract.validate_corrected_reference(correction, permitted)
            require(
                same_json(correction['actual_old'], actual['old'])
                and same_json(correction['native'], actual['core']),
                'recording.correction_join',
            )


def pql_fixture_syntax_policy(name, raw, document):
    policy = unique_json(
        (
            Path(__file__).resolve().parents[3] / 'inventory-policy.json'
        ).read_bytes()
    )['pql_fixture_syntax_corrections']
    require(
        policy['dv'] == 'DV141'
        and policy['existing_authority'] == 'DV12'
        and policy['fixture'] == name
        and policy['fixture_sha256'] == sha(raw),
        'pql.fixture_syntax_source',
    )
    require(
        len(policy['cases']) == 40
        and set(policy['cases']) <= {case['id'] for case in document['cases']},
        'pql.fixture_syntax_scope',
    )
    helper_path = (
        Path(__file__).resolve().parents[6] / 'tests/rust_core/pql/_pql.py'
    )
    require(
        sha(helper_path.read_bytes()) == policy['comparator_sha256'],
        'pql.fixture_syntax_comparator',
    )
    helper = load('_p97_fixture_syntax_comparator', helper_path)
    require(
        all(
            helper.mismatch(
                {'errors': correction['actual_old'], 'warnings': []},
                {'errors': correction['expected'], 'warnings': []},
            )
            is None
            for correction in policy['cases'].values()
        ),
        'pql.fixture_syntax_existing_policy',
    )
    return policy['cases']


def pql_fixture_syntax_errors(case, corrections):
    errors = case['expected_errors']
    if case['id'] not in corrections:
        return errors
    correction = corrections[case['id']]
    require(
        set(correction) == {'actual_old', 'expected'}
        and same_json(errors, correction['actual_old']),
        'pql.fixture_syntax_original',
    )
    expected = correction['expected']
    require(
        type(expected) is list
        and len(errors) == len(expected)
        and all(
            set(old) == set(new)
            and same_json(
                {key: value for key, value in old.items() if key != 'message'},
                {key: value for key, value in new.items() if key != 'message'},
            )
            and (
                old['message'] == new['message']
                or old['code'] == 'PQL_PARSE_SYNTAX'
            )
            for old, new in zip(errors, expected, strict=True)
        ),
        'pql.fixture_syntax_diagnostic',
    )
    return expected


FORMAT = 'p97-finite-envelope-hashes-v1'
DECISIONS = {'DV147', 'DV150', 'DV153', 'DV154', 'DV155', 'DV157'}


def exact_entry(entry, recipe, old, core):
    require(
        set(entry)
        == {
            'recipe_sha256',
            'actual_old_sha256',
            'expected_sha256',
            'materialized_rows',
        }
        and contract.digest(recipe) == entry['recipe_sha256']
        and contract.digest(old) == entry['actual_old_sha256']
        and contract.digest(core) == entry['expected_sha256'],
        'finite.exact_envelopes',
    )


def difference(identity, recipe, old, core, correction, scopes, accepted_dvs):
    matches = [
        (decision, scope)
        for decision, scope in scopes.items()
        if scope.get('format') == FORMAT and identity in scope.get('cases', {})
    ]
    require(len(matches) <= 1, 'finite.ambiguous_scope')
    if not matches:
        return None
    decision, scope = matches[0]
    require(
        decision in DECISIONS and decision in accepted_dvs,
        'finite.unapproved_decision',
    )
    exact_entry(scope['cases'][identity], recipe, old, core)
    require(not same_json(old, core), 'finite.stale_difference')
    if scope['kind'] == 'retained_refusal':
        require(
            correction is None
            and all(
                observation['status'] == 'refused'
                and observation['payloads'] == []
                and observation['result'] is None
                for observation in (old, core)
            ),
            'finite.retained_refusal',
        )
    else:
        require(scope['kind'] == 'authored_semantic', 'finite.kind')
        require(
            type(correction) is dict
            and correction['dv'] == decision
            and correction['source_sha256'] == scope['source_sha256']
            and correction['intervention'] == scope['intervention']
            and same_json(correction['actual_old'], old)
            and same_json(correction['corrected'], core)
            and same_json(correction['native'], core),
            'finite.authored_reference',
        )
        contract.validate_corrected_reference(correction, accepted_dvs)
    return decision


def sweep_entry(identity, recipe, old, core, rows, scopes, accepted_dvs):
    matches = [
        (decision, scope)
        for decision, scope in scopes.items()
        if scope.get('format') == FORMAT and identity in scope.get('cases', {})
    ]
    require(len(matches) <= 1, 'finite.ambiguous_scope')
    if not matches:
        return None
    decision, scope = matches[0]
    require(
        decision in DECISIONS and decision in accepted_dvs,
        'finite.unapproved_decision',
    )
    entry = scope['cases'][identity]
    exact_entry(entry, recipe, old, core)
    require(same_json(rows, entry['materialized_rows']), 'finite.exact_rows')
    return decision


def preentry(identity, recipe, lanes, scopes):
    matches = [
        (decision, scope['preentry_cases'][identity])
        for decision, scope in scopes.items()
        if scope.get('format') == FORMAT
        and identity in scope.get('preentry_cases', {})
    ]
    require(len(matches) <= 1, 'finite.ambiguous_preentry')
    if not matches:
        return None
    decision, entry = matches[0]
    require(decision in {'DV153', 'DV154', 'DV157'}, 'finite.preentry_decision')
    require(
        contract.digest(recipe) == entry['recipe_sha256'],
        'finite.preentry_recipe',
    )
    require(set(lanes) == {'python', 'core'}, 'finite.preentry_lanes')
    for selector, lane in lanes.items():
        require(same_json(lane['recipe'], recipe), 'finite.preentry_input')
        wanted = entry['lanes'][selector]
        entered = 'capture' in lane
        require(entered == wanted['entered'], 'finite.preentry_status')
        if entered:
            capture = lane['capture']
            require(
                all(
                    type(capture[name]) is int and capture[name] >= 0
                    for name in (
                        'public_calls',
                        'legacy_calls',
                        'native_calls',
                        'materialized_rows',
                    )
                )
                and capture['public_calls'] > 0
                and capture[
                    'legacy_calls' if selector == 'python' else 'native_calls'
                ]
                > 0
                and capture[
                    'native_calls' if selector == 'python' else 'legacy_calls'
                ]
                == 0,
                'finite.preentry_public_provenance',
            )
            observed = capture['observation']
            require(
                same_json(
                    capture['materialized_rows'], wanted['materialized_rows']
                ),
                'finite.preentry_rows',
            )
        else:
            require(
                (lane['phase'], lane['class'], lane['message'])
                == (
                    'public_observation',
                    'Rejected',
                    'capture.no_public_entry',
                ),
                'finite.preentry_unknown_failure',
            )
            partial = lane['partial']
            require(
                contract.digest(partial) == wanted['partial_sha256'],
                'finite.preentry_partial',
            )
            require(
                partial['counts']
                == {'public_calls': 0, 'legacy_calls': 0, 'native_calls': 0}
                and partial['event'] is None
                and partial['preparation'] is None
                and partial['legacy_branch_events'] == []
                and partial['dimensions'] is not None,
                'finite.preentry_absence',
            )
            observed = partial['observed']
            require(
                observed['schema'] == []
                and same_json(partial['error'], observed['error']),
                'finite.preentry_error',
            )
        contract.observation(observed)
        require(
            observed['status'] == 'refused'
            and observed['payloads'] == []
            and observed['result'] is None
            and contract.digest(observed) == wanted['observation_sha256'],
            'finite.preentry_observation',
        )
    require(
        not all(lane['entered'] for lane in entry['lanes'].values()),
        'finite.preentry_not_public',
    )
    return decision
