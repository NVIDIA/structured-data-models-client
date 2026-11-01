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
        'result-items',
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


def test_zero_tolerance_never_rounds_mixed_numeric_values():
    exact = {'absolute': 0, 'relative': 0}
    assert difference(2**53 + 1, float(2**53), exact) is not None
    assert difference(1.0, 1.0000000000000002, exact) is not None
    assert difference(2, 2.0, exact) is None


def test_reader_timestamp_units_require_identical_values_and_exact_precision():
    from compare import equivalent_reader_time_units

    def reading(unit, value='2024-01-01T00:00:00.123456', zone=''):
        return {
            'schema': [['time', f'timestamp[{unit}{zone}]']],
            'data': {'columns': ['time'], 'rows': [[value]]},
        }

    old, new = reading('us'), reading('ns')
    equivalent_reader_time_units(old, new)
    assert old == new
    for new in (
        reading('ns', '2024-01-01T00:00:00.123457'),
        reading('ns', zone=', tz=UTC'),
    ):
        old = reading('us')
        equivalent_reader_time_units(old, new)
        assert old != new
    old = reading('us', '2024-01-01T00:00:00.123456789')
    new = reading('ns', '2024-01-01T00:00:00.123456789')
    equivalent_reader_time_units(old, new)
    assert old != new


def test_equal_raw_requests_do_not_need_fixture_identity_rewriting(monkeypatch):
    import compare

    data = one()
    data['scenarios'][0]['comparison'] = 'fixture-source-identity'
    old, new = paired(data)
    for record in [*old.values(), *new.values()]:
        record['semantic']['request'] = {'already': 'identical'}

    def unexpected(*args):
        raise AssertionError('identical raw requests need no identity rewrite')

    monkeypatch.setattr(compare, 'canonical', unexpected)
    assert assess(data, old, new)['goal_complete']


def test_temporal_entity_ref_improvement_is_exact_and_role_bound():
    from compare import equivalent_derived_entity_refs

    name = '__kumo_entity_ref_0'
    fixture = {'primary_keys': {'items': 'id'}}

    def payload(dtype, value):
        table = {'columns': [name], 'rows': [[value], [None]]}
        return {
            'task': {
                'kind': 'temporal_link_prediction',
                'entity_table_names': ['items'],
            },
            'schema': {
                'instance_table': {
                    'columns': {name: {'dtype': dtype, 'stype': 'ID'}}
                },
                'relationships': [
                    {
                        'source_columns': [name],
                        'target_columns': ['id'],
                        'target_table': 'items',
                    }
                ],
            },
            'context': {'instance_table': copy.deepcopy(table)},
            'predict': {'instance_table': copy.deepcopy(table)},
        }

    old, new = payload('float64', 2.0), payload('int64', 2)
    equivalent_derived_entity_refs([old], [new], fixture)
    assert old == new
    for value, integer in [(1.5, 1), (float(2**53), 2**53 + 1), (True, 1)]:
        old, new = payload('float64', value), payload('int64', integer)
        equivalent_derived_entity_refs([old], [new], fixture)
        assert (
            old['schema']['instance_table']['columns'][name]['dtype']
            == 'float64'
        )
    old, new = payload('float64', 2.0), payload('int64', 2)
    for record in (old, new):
        record['schema']['relationships'][0]['source_table'] = 'ordinary_source'
    equivalent_derived_entity_refs([old], [new], fixture)
    assert (
        old['schema']['instance_table']['columns'][name]['dtype'] == 'float64'
    )
    old, new = payload('float64', 2.0), payload('int64', 2)
    new['predict']['instance_table']['rows'][1][0] = 0
    equivalent_derived_entity_refs([old], [new], fixture)
    assert (
        old['schema']['instance_table']['columns'][name]['dtype'] == 'float64'
    )


def test_derived_reference_alignment_precedes_identity_proof_without_hiding_changes(
    monkeypatch,
):
    import compare

    data = one()
    case = data['scenarios'][0]
    case['id'] = 'relational.temporal-link'
    case['comparison'] = 'fixture-source-identity'
    fixture = json.loads((HERE / case['fixture']).read_text())
    name = '__kumo_entity_ref_0'
    user = fixture['tables']['users']
    columns = user['columns']
    values = user['rows'][0]
    kinds = ['int64', 'float64', 'string', 'bool']
    related = {
        'columns': {
            'instance_id': {'dtype': 'int64'},
            **{
                column: {'dtype': dtype}
                for column, dtype in zip(columns, kinds, strict=True)
            },
        },
        'primary_key': ['instance_id', 'id'],
    }

    def payload(dtype, value):
        half = {
            'instance_table': {
                'columns': ['instance_id', name],
                'rows': [[0, value], [1, None]],
            },
            'related_tables': {
                'users': {
                    'columns': ['instance_id', *columns],
                    'rows': [[0, *values]],
                }
            },
        }
        return {
            'task': {
                'kind': 'temporal_link_prediction',
                'entity_table_names': ['items'],
            },
            'schema': {
                'instance_table': {
                    'columns': {name: {'dtype': dtype, 'stype': 'ID'}}
                },
                'related_tables': {'users': copy.deepcopy(related)},
                'relationships': [
                    {
                        'source_columns': [name],
                        'target_columns': ['item'],
                        'target_table': 'items',
                    }
                ],
            },
            'context': copy.deepcopy(half),
            'predict': copy.deepcopy(half),
        }

    old, new = paired(data)
    for record in old.values():
        record['semantic']['request'] = [payload('float64', 2.0)]
        record['endpoint']['transport'] = 'private-bridge'
    for record in new.values():
        record['semantic']['request'] = [payload('int64', 2)]
        record['endpoint']['transport'] = 'private-bridge'
    raw_before = copy.deepcopy((old, new))
    calls = []
    canonical = compare.canonical

    def observed(*args):
        calls.append(True)
        return canonical(*args)

    monkeypatch.setattr(compare, 'canonical', observed)
    report = assess(data, old, new)
    assert report['goal_complete']
    assert not calls
    assert all(
        row['verdict'] == 'Improvement (private bridge)'
        for row in report['cells']
    )
    assert (old, new) == raw_before
    # A changed source identity or feature must still invoke the original proof
    # and fail, even though the derived nullable reference itself is equivalent.
    for column in ('id', 'value'):
        changed = copy.deepcopy(new)
        record = next(iter(changed.values()))
        table = record['semantic']['request'][0]['context']['related_tables'][
            'users'
        ]
        table['rows'][0][table['columns'].index(column)] = 999
        calls.clear()
        report = assess(data, old, changed)
        assert calls
        assert not report['goal_complete']
        assert any(
            row.get('reason', '').startswith('identity proof failed:')
            for row in report['cells']
        )


def test_snowflake_platform_probe_requires_exact_command_and_driver_origin():
    pytest.importorskip('snowflake.connector')
    code = """
import json, subprocess
from routes import LegacyGuard
with LegacyGuard({}) as guard:
    import snowflake.connector
assert guard.report['unobserved_subprocesses'] == []
assert [item['command'] for item in guard.report['observed_subprocesses']] == [['uname', '-p']]
with LegacyGuard({}) as outside:
    subprocess.run(['uname', '-p'], check=True, capture_output=True)
assert outside.report['observed_subprocesses'] == []
assert outside.report['unobserved_subprocesses'] == ['subprocess.Popen']
print(json.dumps(guard.report))
"""
    ran = subprocess.run(
        [sys.executable, '-c', code],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=True,
    )
    observed = json.loads(ran.stdout)['observed_subprocesses']
    assert (
        observed[0]['origin']
        == 'snowflake.connector.description -> platform._Processor.from_subprocess'
    )


def test_snowflake_probe_near_misses_remain_unobserved():
    from routes import snowflake_platform_probe

    assert not snowflake_platform_probe(
        'subprocess.Popen', ('uname', ['uname', '-a'], None, None)
    )
    assert not snowflake_platform_probe(
        'subprocess.Popen', ('uname', ['uname', '-p'], '/', None)
    )
    assert not snowflake_platform_probe(
        'subprocess.Popen', ('uname', ['uname', '-p'], None, {})
    )
    assert not snowflake_platform_probe(
        'os.posix_spawn', ('uname', ['uname', '-p'], None, None)
    )
