# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Permanent evidence gates cannot turn missing or partial checks green."""

import copy
import json
import os
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
    record = {
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

    if case.get('input_mode') == 'raw-reader':
        record['semantic']['wire_calls'] = [
            {
                'operation': 'predict',
                'payload': {
                    'metadata': {'request_id': 'sdm-000000000001'},
                    'value': 1,
                },
            }
        ]
    return record


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
            if cls == 'PredictionResult' and method in {
                'items',
                'metadata',
                'explanation',
            }:
                result = sdm_client.PredictionResult(
                    model='kumo-relational', request_id='inspection', items=[]
                )
                assert hasattr(result, method), reference
            else:
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
@pytest.fixture
def data():
    import predictive_query
    return 1
@pytest.mark.parity_default_routes
def test_public(data):
    assert data == 1
""")
    monkeypatch.setenv(
        'PYTHONPATH',
        os.pathsep.join(
            filter(None, [str(HERE), os.environ.get('PYTHONPATH')])
        ),
    )
    ran = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', str(tmp_path)],
        text=True,
        capture_output=True,
    )
    assert ran.returncode != 0
    assert 'retired runtime imported' in ran.stdout + ran.stderr


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
        'sessions',
        'errors',
        'wire-vocabulary',
        'sql-helpers',
        'release-metadata',
        'relational-materialization',
        'in-process-models',
        'relbench',
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

    from identity import canonical_scoring_payload

    old_score = {'predict': old[0]['context']}
    new_score = {'predict': new[0]['context']}
    assert canonical_scoring_payload(
        old_score, old[0]['schema'], fixture
    ) == canonical_scoring_payload(new_score, new[0]['schema'], fixture)
    with pytest.raises(IdentityError, match='edge'):
        canonical_scoring_payload(
            {'predict': corrupted[0]['context']}, new[0]['schema'], fixture
        )
    with pytest.raises(IdentityError, match='redeclares'):
        canonical_scoring_payload(
            {'schema': new[0]['schema'], **new_score}, new[0]['schema'], fixture
        )


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
        'model': 'relational',
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
        {'phase': 'candidate', 'model': 'relational', 'status': 'refused'},
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


@pytest.mark.parametrize('linked_system_tool', [False, True])
def test_snowflake_platform_probe_requires_exact_command_and_driver_origin(
    linked_system_tool,
):
    pytest.importorskip('snowflake.connector')
    code = """
import json, subprocess
from routes import LegacyGuard
with LegacyGuard({}) as guard:
    import snowflake.connector
assert guard.report['unobserved_subprocesses'] == [], guard.report
assert all(item['command'] == ['uname', '-p'] for item in guard.report['observed_subprocesses'])
with LegacyGuard({}) as outside:
    subprocess.run(['uname', '-p'], check=True, capture_output=True)
assert outside.report['observed_subprocesses'] == []
assert outside.report['unobserved_subprocesses'] == ['subprocess.Popen']
print(json.dumps(guard.report))
"""
    if linked_system_tool:
        code = (
            """
from pathlib import Path
original_resolve = Path.resolve
def system_resolve(self, *args, **kwargs):
    resolved = original_resolve(self, *args, **kwargs)
    if resolved == Path('/usr/bin/uname'):
        return Path('/usr/lib/cargo/bin/coreutils/uname')
    return resolved
Path.resolve = system_resolve
"""
            + code
        )
    ran = subprocess.run(
        [sys.executable, '-c', code],
        cwd=HERE,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
    observed = json.loads(ran.stdout)['observed_subprocesses']
    assert all(
        item['origin']
        == 'snowflake.connector.description -> platform._Processor.from_subprocess'
        for item in observed
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


def test_snowflake_reader_integer_encoding_requires_lossless_ordered_values():
    from compare import equivalent_snowflake_reader_integers

    def reading(dtype, values):
        return {
            'schema': [['id', dtype]],
            'data': {'columns': ['id'], 'rows': [[v] for v in values]},
        }

    for connector in ('snowflake', 'snowflake-semantic'):
        for dtype in ('int16', 'int64', 'decimal128(38, 0)'):
            old, new = (
                reading('int8', [1, None, 127]),
                reading(dtype, [1, None, 127]),
            )
            assert equivalent_snowflake_reader_integers(old, new, connector)
            assert old == new
    for dtype, before, after, connector in (
        ('decimal128(38, 0)', [1], [2], 'snowflake'),
        ('decimal128(38, 0)', [128], [128], 'snowflake'),
        ('decimal128(1, 0)', [10], [10], 'snowflake'),
        ('decimal128(38, 1)', [1], [1], 'snowflake'),
        ('decimal128(38, 0)', [1.0], [1.0], 'snowflake'),
        ('decimal128(38, 0)', [True], [True], 'snowflake'),
        ('decimal128(38, 0)', [1, None], [None, 1], 'snowflake'),
        ('decimal128(38, 0)', [1, 2], [2, 1], 'snowflake'),
        ('decimal128(38, 0)', [1], [1], 'postgres'),
    ):
        old, new = reading('int8', before), reading(dtype, after)
        assert not equivalent_snowflake_reader_integers(old, new, connector)
        assert new['schema'][0][1] == dtype


@pytest.mark.parametrize('outcome', ['passed', 'refused'])
def test_raw_reader_identity_uses_equal_observations_and_rejects_changed_rows(
    monkeypatch,
    outcome,
):
    import compare

    data = one()
    case = data['scenarios'][0]
    case['input_mode'] = 'raw-reader'
    case['comparison'] = 'fixture-source-identity'
    if outcome == 'refused':
        case['expected'] = {
            'status': 'refused',
            'stage': 'predict',
            'code': 'UNSUPPORTED_DTYPE',
            'rule': None,
        }
    old, new = paired(data)
    for record in [*old.values(), *new.values()]:
        record['status'] = outcome
        if outcome == 'refused':
            record['error'] = {
                'stage': 'predict',
                'code': 'UNSUPPORTED_DTYPE',
                'rule': None,
            }
        record['reader']['users']['data']['rows'][0][2] = 0
        record['semantic']['request'] = [
            {'tag': 'old' if record['phase'] == 'baseline' else 'new'}
        ]
    captured = []

    def checked(payload, fixture):
        captured.append(copy.deepcopy(fixture))
        assert fixture['tables']['users']['rows'][0][2] == 0
        return ['proven equivalent identity']

    monkeypatch.setattr(compare, 'canonical', checked)
    before = copy.deepcopy((old, new))
    assert assess(data, old, new)['goal_complete']
    assert captured
    assert (old, new) == before
    for record in new.values():
        record['reader']['users']['data']['rows'][0][1] = 999.0
    captured.clear()
    report = assess(data, old, new)
    assert not report['goal_complete']
    assert not captured
    assert all(
        row['reason'] == 'raw reader rows differ before identity proof'
        for row in report['cells']
    )


def test_model_expectations_do_not_leak_across_models_or_into_baseline():
    from contract import expectation

    supported = {'status': 'supported'}
    refused = {
        'status': 'refused',
        'stage': 'predict',
        'code': 'UNSUPPORTED_DTYPE',
        'rule': None,
    }
    improved = {
        'status': 'refused',
        'stage': 'schema',
        'code': 'UNSUPPORTED_REQUEST',
        'rule': None,
    }
    case = {
        'expected': supported,
        'expected_by_model': {'tabular': refused},
        'expected_by_language': {'java': improved},
    }
    assert expectation(case, 'relational') == supported
    assert expectation(case, 'tabular') == refused
    assert expectation(case, 'tabular', 'python') == refused
    assert expectation(case, 'tabular', 'java') == improved
    assert expectation(case, 'relational') == supported


def test_model_supported_override_and_endpoint_refusals_require_live_evidence():
    data = one()
    case = data['scenarios'][0]
    case['expected'] = {
        'status': 'refused',
        'stage': 'schema',
        'code': 'UNSUPPORTED_REQUEST',
        'rule': None,
    }
    case['expected_by_model'] = {'relational': {'status': 'supported'}}
    old, new = paired(data)
    assert assess(data, old, new)['goal_complete']
    next(iter(new.values()))['execution'] = 'local'
    assert not assess(data, old, new)['goal_complete']

    case['expected_by_model']['relational'] = {
        'status': 'refused',
        'stage': 'predict',
        'code': 'UNSUPPORTED_DTYPE',
        'rule': None,
    }
    old, new = paired(data)
    for record in [*old.values(), *new.values()]:
        record['status'] = 'refused'
        record['error'] = {
            'stage': 'predict',
            'code': 'UNSUPPORTED_DTYPE',
            'rule': None,
        }
    assert assess(data, old, new)['goal_complete']
    next(iter(new.values()))['endpoint'] = None
    assert not assess(data, old, new)['goal_complete']


def test_source_identity_conversion_proofs_are_typed_and_exact():
    from identity import same

    assert same(None, {'float': 'nan'}, 'float64')
    assert not same(None, {'float': 'inf'}, 'float64')
    assert not same(None, {'float': 'nan'}, 'string')
    assert same('9007199254740993', 9007199254740993, 'int64')
    assert not same('9007199254740992', 9007199254740993, 'int64')
    assert not same('01', 1, 'int64')
    assert not same(str(2**63), 2**63, 'int64')
    assert not same('1', 1, 'string')
    assert same(15.123456, {'decimal': '15.123456'}, 'float64')
    assert not same(15.123457, {'decimal': '15.123456'}, 'float64')
    assert not same(float('inf'), {'decimal': '1e999'}, 'float64')
    assert not same(15.123456, {'decimal': '15.123456'}, 'string')
    assert same(
        '2024-03-10T10:00:00.987654321Z',
        '2024-03-10T03:00:00.987654321-07:00',
        'timestamp[ns]',
    )
    assert not same(
        '2024-03-10T10:00:00.987654321Z',
        '2024-03-10T10:00:00.987654320Z',
        'timestamp[ns]',
    )


def test_tokenized_text_is_not_an_identity_anchor_but_stays_exact_payload_data():
    from identity import IdentityError, canonical

    fixture = {
        'primary_keys': {'users': 'id'},
        'links': [],
        'tables': {
            'users': {
                'columns': ['id', 'score', 'description'],
                'rows': [[7, 1.25, 'Original sentence.'], [8, 2.5, None]],
            }
        },
    }
    payload = [
        {
            'schema': {
                'related_tables': {
                    'users': {
                        'primary_key': ['instance_id', '__node_id'],
                        'columns': {
                            'instance_id': {'dtype': 'int64'},
                            '__node_id': {'dtype': 'int64'},
                            'score': {'dtype': 'float64'},
                            'description': {'dtype': 'stringlist'},
                        },
                    }
                }
            },
            'context': {
                'related_tables': {
                    'users': {
                        'columns': [
                            'instance_id',
                            '__node_id',
                            'score',
                            'description',
                        ],
                        'rows': [
                            [0, 0, 1.25, ['original', 'sentence']],
                            [0, 1, 2.5, []],
                        ],
                    }
                }
            },
        }
    ]
    original = canonical(payload, fixture)
    changed = copy.deepcopy(payload)
    changed[0]['context']['related_tables']['users']['rows'][0][3] = ['changed']
    assert canonical(changed, fixture) != original
    ambiguous = copy.deepcopy(fixture)
    ambiguous['tables']['users']['rows'][1][1] = 1.25
    ambiguous['tables']['users']['rows'][1][2] = 'Different text.'
    with pytest.raises(IdentityError, match='not uniquely'):
        canonical(payload, ambiguous)


def test_timestamp_source_identity_uses_only_the_declared_wire_precision():
    from identity import same

    assert same(
        '2024-03-10T10:00:00.987654Z',
        '2024-03-10T03:00:00.987654321-07:00',
        'timestamp[us]',
    )
    assert not same(
        '2024-03-10T10:00:00.987654Z',
        '2024-03-10T10:00:00.987653999Z',
        'timestamp[us]',
    )
    assert not same(
        '2024-03-10T10:00:00.987654Z',
        '2024-03-10T10:00:00.987654321Z',
        'timestamp[ns]',
    )


def test_raw_observation_equivalence_is_slot_typed_exact_and_detached():
    import copy

    from compare import difference, equivalent_raw_observations

    fixture = {
        'primary_keys': {'users': 'id'},
        'tables': {
            'users': {
                'columns': ['id', 'number', 'empty', 'fraction', 'time'],
                'types': {'time': 'timestamp[ns,America/Los_Angeles]'},
                'rows': [
                    [
                        1,
                        None,
                        None,
                        '1.123456',
                        '2024-03-10T03:00:00.987654321-07:00',
                    ]
                ],
            }
        },
    }
    frame = {
        'columns': fixture['tables']['users']['columns'],
        'rows': [
            [
                1,
                {'float': 'nan'},
                None,
                {'decimal': '1.123456'},
                '2024-03-10T03:00:00.987654321-07:00',
            ]
        ],
    }
    old = {
        'reader': {
            'users': {
                'schema': [
                    ['id', 'int64'],
                    ['number', 'double'],
                    ['empty', 'null'],
                    ['fraction', 'decimal128(7, 6)'],
                    ['time', 'timestamp[ns, tz=America/Los_Angeles]'],
                ],
                'raw_schema': [['preserved', 'original']],
                'data': frame,
            }
        },
        'semantic': {
            'request': {
                'context': copy.deepcopy(frame),
                'predict': copy.deepcopy(frame),
                'target': 'value',
                'task': 'regression',
            },
            'results': [{'float': 'nan'}],
        },
    }
    new = copy.deepcopy(old)
    new['reader']['users']['schema'][2][1] = 'string'
    new['reader']['users']['schema'][3][1] = 'decimal128(20, 6)'
    for data in [
        new['reader']['users']['data'],
        *[new['semantic']['request'][x] for x in ('context', 'predict')],
    ]:
        data['rows'][0][1] = None
        data['rows'][0][4] = 1710064800987654321
    untouched = copy.deepcopy((old, new))
    a, b = copy.deepcopy((old, new))
    equivalent_raw_observations(a, b, fixture)
    assert difference(a, b, {'absolute': 0, 'relative': 0}) is None
    assert (old, new) == untouched
    assert a['reader']['users']['raw_schema'] == [['preserved', 'original']]
    assert a['semantic']['results'] == [{'float': 'nan'}]

    for mutation in (
        'nonnull',
        'tick',
        'scale',
        'precision',
        'key',
        'zone',
        'infinity',
    ):
        a, b = copy.deepcopy((old, new))
        declared = copy.deepcopy(fixture)
        if mutation == 'nonnull':
            declared['tables']['users']['rows'][0][1] = 2.0
        elif mutation == 'tick':
            b['reader']['users']['data']['rows'][0][4] += 1
        elif mutation == 'scale':
            b['reader']['users']['schema'][3][1] = 'decimal128(20, 5)'
        elif mutation == 'precision':
            a['reader']['users']['schema'][3][1] = 'decimal128(6, 6)'
        elif mutation == 'key':
            b['reader']['users']['data']['rows'][0][0] = 2
        elif mutation == 'zone':
            b['reader']['users']['schema'][4][1] = 'timestamp[ns, tz=Not/AZone]'
        else:
            a['reader']['users']['data']['rows'][0][1] = {'float': 'inf'}
        equivalent_raw_observations(a, b, declared)
        assert difference(a, b, {'absolute': 0, 'relative': 0}) is not None, (
            mutation
        )


def test_decimal_improvement_requires_exact_old_cast_and_closer_new_rounding():
    from decimal import Decimal

    import pyarrow as pa
    from identity import equivalent_legacy_decimal_cast

    exact = Decimal('21.123456')
    legacy = (
        pa.array([exact], type=pa.decimal128(20, 6))
        .cast(pa.float64())[0]
        .as_py()
    )
    nearest = float(exact)
    assert legacy != nearest
    fixture = {
        'primary_keys': {'users': 'id'},
        'tables': {
            'users': {
                'columns': ['id', 'fraction'],
                'types': {'fraction': 'decimal128(20,6)'},
                'rows': [[21, str(exact)], [22, None]],
            }
        },
    }

    def payload(value):
        table = {
            'columns': ['instance_id', 'id', 'fraction'],
            'rows': [[0, 21, value], [0, 22, None]],
        }
        return [
            {
                'schema': {
                    'related_tables': {
                        'users': {
                            'primary_key': ['instance_id', 'id'],
                            'columns': {'fraction': {'dtype': 'float64'}},
                        }
                    }
                },
                'context': {'related_tables': {'users': copy.deepcopy(table)}},
                'predict': {'related_tables': {'users': copy.deepcopy(table)}},
            }
        ]

    old, new = payload(legacy), payload(nearest)
    assert equivalent_legacy_decimal_cast(old, new, fixture)
    assert old == new
    for wrong_old, wrong_new in [
        (nearest, legacy),
        (legacy, nearest + 1e-12),
        (legacy - 1e-12, nearest),
    ]:
        old, new = payload(wrong_old), payload(wrong_new)
        before = copy.deepcopy(old)
        assert not equivalent_legacy_decimal_cast(old, new, fixture)
        assert old == before
    old, new = payload(legacy), payload(nearest)
    new[0]['predict']['related_tables']['users']['rows'][0][1] = 22
    assert not equivalent_legacy_decimal_cast(old, new, fixture)


def test_timezone_display_labels_require_exact_instants_and_keep_raw_metadata():
    from compare import equivalent_raw_observations

    fixture = {
        'primary_keys': {'users': 'id'},
        'tables': {
            'users': {
                'columns': ['id', 'time'],
                'types': {'time': 'timestamp[ns,America/Los_Angeles]'},
                'rows': [[1, '2024-03-10T03:00:00.987654321-07:00']],
            }
        },
    }
    old = {
        'reader': {
            'users': {
                'schema': [
                    ['id', 'int64'],
                    ['time', 'timestamp[ns, tz=America/Los_Angeles]'],
                ],
                'raw_schema': [['time', 'original']],
                'data': {
                    'columns': ['id', 'time'],
                    'rows': [[1, '2024-03-10T03:00:00.987654321-07:00']],
                },
            }
        }
    }
    new = copy.deepcopy(old)
    new['reader']['users']['schema'][1][1] = 'timestamp[ns, tz=Etc/UTC]'
    new['reader']['users']['data']['rows'][0][1] = 1710064800987654321
    notes = equivalent_raw_observations(old, new, fixture)
    assert old == new
    assert 'timezone display label differs; exact instants preserved' in notes
    assert old['reader']['users']['raw_schema'] == [['time', 'original']]


def test_date_inspection_and_list_null_proofs_do_not_erase_times_or_numbers():
    from compare import equivalent_raw_observations

    fixture = {
        'primary_keys': {'users': 'id'},
        'tables': {
            'users': {
                'columns': ['id', 'date', 'sequence'],
                'types': {'date': 'date32', 'sequence': 'list<float64>'},
                'rows': [[1, '1969-12-31', [3.0, None]]],
            }
        },
    }
    frame = {
        'columns': ['id', 'date', 'sequence'],
        'rows': [[1, '1969-12-31', [3.0, {'float': 'nan'}]]],
    }
    old = {
        'reader': {
            'users': {
                'schema': [
                    ['id', 'int64'],
                    ['date', 'date32[day]'],
                    ['sequence', 'list<item: double>'],
                ],
                'data': frame,
            }
        },
        'semantic': {
            'request': {
                'context': copy.deepcopy(frame),
                'predict': copy.deepcopy(frame),
                'target': 'label',
                'task': 'regression',
            }
        },
    }
    new = copy.deepcopy(old)
    new['reader']['users']['data']['rows'][0][2][1] = None
    for half in ['context', 'predict']:
        new['semantic']['request'][half]['rows'][0][1] = '1969-12-31T00:00:00'
        new['semantic']['request'][half]['rows'][0][2][1] = None
    a, b = copy.deepcopy((old, new))
    equivalent_raw_observations(a, b, fixture)
    assert a == b
    for mutation in ['hour', 'number', 'unexpected_nan']:
        a, b = copy.deepcopy((old, new))
        if mutation == 'hour':
            b['semantic']['request']['context']['rows'][0][1] = (
                '1969-12-31T12:00:00'
            )
        elif mutation == 'number':
            b['reader']['users']['data']['rows'][0][2][1] = 7.0
        else:
            a['reader']['users']['data']['rows'][0][2][0] = {'float': 'nan'}
        equivalent_raw_observations(a, b, fixture)
        assert a != b


@pytest.mark.parametrize(
    'module', ['predictive_query', 'sdm_relational_engine']
)
def test_default_guard_blocks_retired_package_imports(module):
    import importlib

    with LegacyGuard(raise_caught=False) as guard:
        with pytest.raises(AssertionError, match='retired runtime imported'):
            importlib.import_module(module)
    assert guard.report['legacy_hits'] == [module]
    assert guard.report['installed']


@pytest.mark.parametrize(
    'module', ['predictive_query', 'sdm_relational_engine']
)
def test_default_guard_rejects_preloaded_retired_packages(module, monkeypatch):
    import sys
    import types

    monkeypatch.setitem(sys.modules, module, types.ModuleType(module))
    guard = LegacyGuard()
    with pytest.raises(
        AssertionError, match='retired runtime already imported'
    ):
        with guard:
            pytest.fail('a preloaded retired package must prevent the journey')
    assert guard.report['legacy_hits'] == [module]
    assert not guard.report['installed']
