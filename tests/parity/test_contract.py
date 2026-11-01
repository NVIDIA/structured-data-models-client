# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Permanent evidence gates cannot turn missing or partial checks green."""

import copy
import json
import subprocess
import sys
import threading
import types

import pytest
from compare import assess as compare_assess
from compare import difference, matrix
from routes import LegacyGuard

from contract import (
    BASELINE,
    HERE,
    cells,
    digest,
    fixture_hash,
    manifest,
    records,
)


def assess(*args, **kwargs):
    return compare_assess(*args, candidate_commit='candidate', **kwargs)


def one():
    data = manifest()
    data['scenarios'] = [copy.deepcopy(data['scenarios'][0])]
    data['scenarios'][0]['connectors'] = ['memory']
    data['scenarios'][0]['models'] = ['relational']
    data['scenarios'][0]['comparison'] = 'exact'
    return data


def observation(case, phase='candidate', language='python'):
    fixture = json.loads((HERE / case['fixture']).read_text())
    return {
        'version': 1,
        'case': case['id'],
        'connector': 'memory',
        'model': 'relational',
        'language': language,
        'phase': phase,
        'status': 'passed',
        'execution': 'live',
        'fixture_hash': fixture_hash(case),
        'artifact': {
            'commit': BASELINE if phase == 'baseline' else 'candidate',
            'package_hash': 'package',
            'native_hash': 'native',
            'contract': 54,
        },
        'source_snapshot': 'snapshot',
        'reader': {
            name: {
                'schema': [[col, 'fixture-type'] for col in table['columns']],
                'data': table,
            }
            for name, table in fixture['tables'].items()
        },
        'endpoint': {
            'kind': 'live',
            'model': 'relational',
            'version': 'model-v1',
            'transport': 'nim',
        },
        'error': None,
        'semantic': {
            'schema': {'users': {}},
            'request': {},
            'results': [[1, 2.0]],
            'lifecycle': ['materialize', 'predict'],
        },
        'route': {
            'installed': True,
            'scope': 'whole-journey',
            'workers': ['pid:thread'],
            'legacy_hits': [],
            'unobserved_subprocesses': [],
        },
    }


def paired(data):
    case = data['scenarios'][0]
    old = observation(case, 'baseline')
    new = observation(case)
    identity = tuple(next(cells(data)))
    return {(*identity[:3], 'python'): old}, {
        (*identity[:3], 'python'): new,
        (*identity[:3], 'java'): observation(case, language='java'),
    }


def test_missing_scope_and_local_runs_cannot_finish_the_goal():
    data = one()
    old, new = paired(data)
    assert assess(data, old, new)['goal_complete']
    selected = {next(iter(new))}
    scoped = assess(data, old, new, selected=selected)
    assert scoped['scope_complete'] and not scoped['goal_complete']
    assert not assess(data, old, {}, selected=selected)['scope_complete']
    assert not assess(data, old, new, live=False)['goal_complete']


@pytest.mark.parametrize(
    'field,value',
    [
        ('status', 'incomplete'),
        ('status', 'deferred'),
        ('fixture_hash', 'other'),
        ('source_snapshot', 'other'),
        ('execution', 'local'),
        ('endpoint', {'kind': 'stub'}),
        ('route', {'legacy_hits': []}),
        ('artifact', {'commit': 'candidate'}),
    ],
)
def test_incomplete_or_mismatched_evidence_never_passes(field, value):
    data = one()
    old, new = paired(data)
    next(iter(new.values()))[field] = value
    assert not assess(data, old, new)['goal_complete']


def test_refusal_requires_expected_stage_code_and_rule():
    data = one()
    case = data['scenarios'][0]
    case['expected'] = {
        'status': 'refused',
        'stage': 'materialize',
        'code': 'UNSUPPORTED_REQUEST',
        'rule': 'model',
    }
    old, new = paired(data)
    for record in [*old.values(), *new.values()]:
        record.update(
            status='refused',
            error={k: case['expected'][k] for k in ('stage', 'code', 'rule')},
        )
    assert assess(data, old, new)['goal_complete']
    next(iter(new.values()))['error']['stage'] = 'predict'
    assert not assess(data, old, new)['goal_complete']


def test_numbers_preserve_identity_order_and_explicit_tolerances():
    exact = {'absolute': 0, 'relative': 0}
    assert difference([True], [1], exact)
    assert difference([2**60], [2**60 + 1], exact)
    assert difference([1, 2], [2, 1], exact)
    assert difference([1.0], [1.0001], exact)
    assert (
        difference([1.0], [1.0001], {'absolute': 0.001, 'relative': 0}) is None
    )
    assert difference([float('nan')], [float('nan')], exact)


def test_duplicate_records_fail_and_generated_matrix_is_reproducible(tmp_path):
    data = one()
    old, new = paired(data)
    path = tmp_path / 'duplicate.jsonl'
    line = json.dumps(next(iter(new.values()))) + '\n'
    path.write_text(line * 2)
    with pytest.raises(ValueError, match='duplicate'):
        records(path)
    report = assess(data, old, new)
    assert matrix(report) == matrix(copy.deepcopy(report))
    assert 'Python Relational' in matrix(report) and 'Java Tabular' in matrix(
        report
    )


def test_fixture_hash_covers_options_schema_rows_and_order():
    case = copy.deepcopy(manifest()['scenarios'][0])
    initial = fixture_hash(case)
    case['options']['random_seed'] += 1
    assert fixture_hash(case) != initial
    fixture = json.loads((HERE / case['fixture']).read_text())
    changed = copy.deepcopy(fixture)
    changed['tables']['users']['rows'].reverse()
    assert digest(fixture) != digest(changed)


def test_python_api_mappings_exist_on_installed_public_surface():
    import sdm_client

    for case in manifest()['scenarios']:
        for reference in case['apis']['python']:
            cls, method = reference.split('.')
            assert callable(getattr(getattr(sdm_client, cls), method)), (
                reference
            )


def test_tripwires_catch_worker_aliases_even_when_worker_catches_error(
    monkeypatch,
):
    module = types.ModuleType('parity_tripwire_probe')
    exec('def legacy():\n    return 7\n', module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    alias = module.legacy
    guard = LegacyGuard({module.__name__: ('legacy',)})
    with pytest.raises(AssertionError, match='invoked and caught'):
        with guard:

            def worker():
                try:
                    alias()
                except AssertionError:
                    pass

            thread = threading.Thread(target=worker)
            thread.start()
            thread.join()
    assert guard.report['legacy_hits']
    assert len(guard.report['workers']) >= 2
    assert module.legacy() == 7


def test_subprocesses_are_not_silently_counted_as_observed():
    with LegacyGuard({}) as guard:
        subprocess.run([sys.executable, '-c', 'pass'], check=True)
    assert guard.report['unobserved_subprocesses']


def test_pytest_tripwire_wraps_fixture_setup_and_teardown(
    tmp_path, monkeypatch
):
    # Run a deliberately failing public setup in a child pytest. Its local
    # conftest uses the exact permanent hook, not a hand-written approximation.
    (tmp_path / 'conftest.py').write_text((HERE / 'conftest.py').read_text())
    (tmp_path / 'test_probe.py').write_text("""import pytest
import predictive_query
@pytest.fixture
def data():
    predictive_query.validate(None)
    return 1
@pytest.mark.parity_default_routes
def test_public(data):
    assert data == 1
""")
    monkeypatch.setenv('PYTHONPATH', str(HERE))
    ran = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', str(tmp_path)],
        text=True,
        capture_output=True,
    )
    assert ran.returncode != 0
    assert 'legacy policy executed' in ran.stdout + ran.stderr


def test_helper_api_registry_has_real_behavior_tests_and_declared_scope():
    from api_contracts import validate

    assert validate(manifest()) == {
        'schema-edits',
        'direct-dtypes-options',
        'readiness-discovery',
        'results-merge',
        'persistence',
        'readers',
        'provider-contract',
        'large-values',
        'public-serving-refusals',
        'request-inspection',
    }


def test_candidate_commit_and_artifact_mixing_fail_closed():
    data = one()
    old, new = paired(data)
    next(iter(new.values()))['artifact']['commit'] = 'another-commit'
    with pytest.raises(ValueError, match='requested commit'):
        assess(data, old, new)


def test_model_tolerance_never_changes_input_or_reader_comparison():
    data = one()
    data['scenarios'][0]['tolerance'] = {'absolute': 0.1, 'relative': 0}
    old, new = paired(data)
    for record in [*old.values(), *new.values()]:
        record['semantic']['request'] = {'feature': 1.0}
    next(iter(new.values()))['semantic']['request']['feature'] = 1.01
    assert not assess(data, old, new)['goal_complete']


def test_explicit_improvement_still_checks_the_exact_old_error():
    data = one()
    case = data['scenarios'][0]
    case['expected'] = {
        'status': 'refused',
        'outcome': 'failed',
        'stage': 'restore',
        'code': 'TypeError',
        'rule': None,
    }
    case['expected_by_language'] = {
        'java': {
            'status': 'refused',
            'stage': 'restore',
            'code': 'UNSUPPORTED_REQUEST',
            'rule': None,
            'verdict': 'Improvement',
        }
    }
    case['differences'] = ['named-source-persistence-refusal']
    old, new = paired(data)
    for record in [*old.values(), *new.values()]:
        spec = case['expected_by_language'].get(
            record['language'], case['expected']
        )
        record.update(
            status=spec.get('outcome', 'refused'),
            error={field: spec[field] for field in ('stage', 'code', 'rule')},
        )
    assert assess(data, old, new)['goal_complete']
    next(iter(old.values()))['error']['code'] = 'ConnectionError'
    assert not assess(data, old, new)['goal_complete']


def test_connector_variant_is_explicit_and_does_not_change_other_cells():
    from contract import effective_case

    case = manifest()['scenarios'][0]
    standard = effective_case(case, 'memory')
    variant = effective_case(case, 'snowflake-semantic')
    assert 'users.VALUE' in variant['pql']
    assert 'users.value' in standard['pql']
    assert fixture_hash(standard) == fixture_hash(case)
    assert fixture_hash(variant) != fixture_hash(standard)


def test_identity_mapping_is_bijective_and_does_not_hide_feature_edge_or_order_changes():
    from identity import IdentityError, canonical

    fixture = json.loads((HERE / 'fixtures/shop.json').read_text())

    def envelope(synthetic):
        definitions, tables = {}, {}
        for name, positions in [
            ('users', [1]),
            ('events', [1, 25]),
            ('items', [1]),
        ]:
            source = fixture['tables'][name]
            key = fixture['primary_keys'][name]
            wire_key = '__node_id' if synthetic and name != 'users' else key
            columns = [
                'instance_id',
                *[wire_key if col == key else col for col in source['columns']],
            ]
            rows = []
            for wire_id, position in enumerate(positions):
                values = dict(
                    zip(
                        source['columns'], source['rows'][position], strict=True
                    )
                )
                if synthetic and name != 'users':
                    values[key] = wire_id
                if synthetic and name == 'events':
                    values['item'] = 0
                rows.append([0, *[values[col] for col in source['columns']]])
            types = {
                col: {
                    'dtype': 'timestamp[us]'
                    if col == 'event_time'
                    else 'string'
                    if col == 'class'
                    else 'bool'
                    if col == 'flag'
                    else 'float64'
                    if col in ('value', 'feature', 'price')
                    else 'int64'
                }
                for col in columns
            }
            definitions[name] = {
                'columns': types,
                'primary_key': ['instance_id', wire_key],
            }
            tables[name] = {
                'columns': columns,
                'rows': rows,
                'format': 'arrays',
            }
        links = [
            {
                'source_table': 'events',
                'source_columns': ['id'],
                'target_table': 'users',
                'target_columns': ['id'],
            },
            {
                'source_table': 'events',
                'source_columns': ['item'],
                'target_table': 'items',
                'target_columns': ['__node_id' if synthetic else 'item'],
            },
        ]
        return [
            {
                'schema': {
                    'related_tables': definitions,
                    'relationships': links,
                },
                'context': {'related_tables': tables},
            }
        ]

    old, new = envelope(True), envelope(False)
    expected = canonical(old, fixture)
    assert canonical(new, fixture) == expected
    corrupted = copy.deepcopy(new)
    table = corrupted[0]['context']['related_tables']['events']
    table['rows'][0][table['columns'].index('item')] = 2
    with pytest.raises(IdentityError, match='edge'):
        canonical(corrupted, fixture)
    reordered = copy.deepcopy(new)
    reordered[0]['context']['related_tables']['events']['rows'].reverse()
    assert canonical(reordered, fixture) != expected
    duplicated = copy.deepcopy(old)
    duplicated[0]['context']['related_tables']['events']['rows'].append(
        old[0]['context']['related_tables']['events']['rows'][0]
    )
    with pytest.raises(IdentityError, match='bijective'):
        canonical(duplicated, fixture)


def test_manifest_rejects_nonzero_unscoped_model_tolerance(tmp_path):
    data = manifest()
    data['scenarios'][0]['tolerance']['absolute'] = 0.001
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError, match='prediction-column policy'):
        manifest(path)


def test_runner_accepts_only_explicit_expected_failure():
    from python_runner import expected_outcome

    case = {
        'expected': {
            'status': 'refused',
            'outcome': 'failed',
            'stage': 'restore',
            'code': 'TypeError',
            'rule': None,
        }
    }
    record = {
        'phase': 'baseline',
        'status': 'failed',
        'error': {'stage': 'restore', 'code': 'TypeError', 'rule': None},
    }
    assert expected_outcome(case, record)
    record['error']['stage'] = 'predict'
    assert not expected_outcome(case, record)
    record['status'] = 'passed'
    assert not expected_outcome(case, record)
    assert not expected_outcome(
        {'expected': {'status': 'supported'}},
        {'phase': 'candidate', 'status': 'refused'},
    )
