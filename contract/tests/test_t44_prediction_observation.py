# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Finite common meaning requires actual typed prediction evidence."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODEL = 'kumo-relational'
ANSWER_MODEL = 'endpoint-relational-revision'
CALL_ID = 'sdm-000000000001'


def reader():
    name = 't44_prediction_observation_reader'
    if name not in sys.modules:
        path = Path(__file__).parents[1] / 'runners/prediction_observation.py'
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def actual_observation(call_id=CALL_ID, endpoint_id='endpoint-r1'):
    """One call, two native source rows, and a complete typed final read.

    This is independently authored data in the actual DTO/observer shapes,
    not a frozen capture or a claim that the SDK executed this seed.
    """
    provider = {
        'provider': 'sql',
        'dialect': 'databricks',
        'namespace': ['test_catalog', 'test_schema'],
        'view': {'name': 'metric_view', 'trust_expressions': False},
    }
    source = {
        'tables': [{'name': 'users', 'source': {'source': ['users']}}],
        'primary_keys': {'users': 'user_id'},
        'time_columns': {},
    }
    query = 'PREDICT users.age FOR EACH users.user_id'
    payload = {
        'model': MODEL,
        'task': {
            'kind': 'regression',
            'target': {'column_name': 'TARGET', 'dtype': 'int64'},
        },
        'schema': {'instance_table': {'primary_key': 'instance_id'}},
        'context': {
            'instance_table': {
                'format': 'arrays',
                'columns': ['instance_id', 'TARGET'],
                'rows': [[0, 21], [1, 22]],
            }
        },
        'predict': {
            'instance_table': {
                'format': 'arrays',
                'columns': ['instance_id', '__kumo_entity_ref_0'],
                'rows': [[2, 7], [3, 8]],
            }
        },
        'metadata': {'source_query': query},
    }
    envelope = copy.deepcopy(payload)
    envelope['metadata']['request_id'] = call_id
    envelope['output'] = {'fields': ['prediction']}
    encoded = json.dumps(envelope).encode()
    body = json.loads(encoded, parse_int=lambda token: {'number': token})
    bound = {
        'phase': 'bound',
        'version': 1,
        'model': MODEL,
        'outputs': ['prediction'],
        'named': False,
        'shape': {
            'kind': 'regression',
            'anchored': False,
            'relationships': False,
            'tables': 1,
        },
        'kind': 'regression',
        'context_rows': 2,
    }
    scope = {
        'version': 1,
        'call_request_id': call_id,
        'model': ANSWER_MODEL,
        'request_id': endpoint_id,
        'response_rows': 2,
    }
    response = {
        'scope': scope,
        'response': 2,
        'rows': 3,
        'entity_reference_prefix': '__kumo_entity_ref_',
        'forecast_kind': 'forecasting',
    }
    part = {
        'data': response,
        'attribution': {'scope': scope, 'source_rows': 2, 'instances': True},
        'rows': 4,
        'target': {
            'kind': list(map(ord, 'regression')),
            'classification': False,
            'dtype_truth': None,
            'numeric_prefix': None,
            'dtype': None,
        },
        'target_rows': 5,
    }
    call_input = {
        'stage': 'call_relational',
        'input': {
            'call': bound,
            'body': 2,
            'request_id': call_id,
            'model_endpoint': None,
        },
    }
    finish_input = {
        'stage': 'finish_relational_response',
        'input': {
            'bound': bound,
            'aggregation': {
                'kind': 'calls',
                'parts': [part],
                'counted': [2],
                'known_empty': False,
            },
            'report_absent': True,
        },
    }
    data = {
        'predicted': {
            'result': 8,
            'metadata': 9,
            'model': ANSWER_MODEL,
            'request_id': endpoint_id,
            'outputs': ['prediction'],
        },
        'lineage': [
            {'call_request_id': call_id, 'answer_ordinal': at, 'source_row': at}
            for at in range(2)
        ],
        'answer_details': {'calls_answered': 1, 'rows_answered': 2},
    }
    arrow = {
        'columns': [
            {'name': name, 'type': kind}
            for name, kind in [
                ('row_index', 'int64'),
                ('id', 'int64'),
                ('prediction', 'float64'),
                ('answer_index', 'int64'),
            ]
        ],
        'rows': [
            [
                {'i64': '0'},
                {'i64': '7'},
                {'f64_bits': '4035000000000000'},
                {'i64': '0'},
            ],
            [
                {'i64': '1'},
                {'i64': '8'},
                {'f64_bits': '4036000000000000'},
                {'i64': '1'},
            ],
        ],
        'explanation': '{"source":"endpoint"}',
    }
    metadata = {'endpoint': 'answer metadata, not replay input'}
    call = {
        'run': 3,
        'request_id': call_id,
        'input_body_sha256': hashlib.sha256(encoded).hexdigest(),
        'input_body_base64': base64.b64encode(encoded).decode('ascii'),
        'body': body,
        'body_sha256': hashlib.sha256(encoded).hexdigest(),
        'scope': scope,
    }
    final = {
        'run': 4,
        'input': finish_input,
        'data': data,
        'metadata': metadata,
        'result': arrow,
    }
    starts = [
        {
            'op': 'describe_source',
            'provider': provider,
            'source': source,
            'check_keys': False,
        },
        {
            'op': 'materialize',
            'provider': provider,
            'indices': [{'t': 'i64', 'v': '7'}, {'t': 'i64', 'v': '8'}],
            'anchor_time': None,
            'seed': 7,
            'schema': {'held': 'must be rebuilt'},
            'query': query,
        },
        {
            'op': 'prepare_request',
            'run': 2,
            'entry': 0,
            'input': {'stage': 'model', 'model': MODEL},
            'result': {'phase': 'model_resolved', 'version': 1, 'model': MODEL},
            'error': None,
        },
        {
            'op': 'prepare_request',
            'run': 3,
            'entry': 0,
            'input': copy.deepcopy(call_input),
            'result': copy.deepcopy(response),
            'error': None,
        },
        {
            'op': 'prepare_request',
            'run': 4,
            'entry': 0,
            'input': copy.deepcopy(finish_input),
            'result': copy.deepcopy(data),
            'error': None,
        },
    ]
    entry = {
        'ordinal': 0,
        'parent': None,
        'argument': {'form': 'unobserved'},
        'completion': 'returned',
        'runs': [2, 3, 4],
        'calls': [copy.deepcopy(call)],
        'results': [copy.deepcopy(final)],
        'final': 0,
    }
    core = {
        'inputs': starts,
        'fetches': [],
        'accepted': True,
        'described': {'tables': []},
        'materialized': {'version': 1, 'query': query, 'payload': payload},
        'predicted': {
            'body': copy.deepcopy(body),
            'model': ANSWER_MODEL,
            'request_id': endpoint_id,
            'metadata': copy.deepcopy(metadata),
            'result': copy.deepcopy(arrow),
        },
        'error': None,
        'predictions': [entry],
    }
    return core, ['describe_source', 'materialize'] + ['prepare_request'] * 3


def legacy_observation():
    core, _ = actual_observation()
    core.pop('predictions')
    core['inputs'] = core['inputs'][:2] + [
        {
            'op': 'predict',
            'model': MODEL,
            'state': copy.deepcopy(core['materialized']),
        }
    ]
    return core, ['describe_source', 'materialize', 'predict']


def test_common_meaning_preserves_actual_producer_evidence_without_mutation():
    actual, operations = actual_observation()
    old, old_operations = legacy_observation()
    untouched = copy.deepcopy((actual, operations, old, old_operations))
    new_view = reader().view(actual, operations)
    old_view = reader().view(old, old_operations)
    assert new_view['operations'] == old_view['operations'] == old_operations
    assert (
        new_view['inputs']
        == old_view['inputs']
        == [
            *actual['inputs'][:2],
            {'op': 'predict', 'model': MODEL},
        ]
    )
    assert new_view['predictions'] == [
        {
            'ordinal': 0,
            'parent': None,
            'model': MODEL,
            'native': True,
            'completion': 'returned',
            'failure': None,
        }
    ]
    assert (actual, operations, old, old_operations) == untouched
    assert 'predict' not in operations
    assert actual['predicted']['model'] == ANSWER_MODEL != MODEL


@pytest.mark.parametrize(
    'missing', ['calls', 'body', 'scope', 'results', 'metadata', 'arrow']
)
def test_erased_actual_evidence_cannot_become_a_common_prediction(missing):
    core, operations = actual_observation()
    entry = core['predictions'][0]
    if missing in ('calls', 'results'):
        entry[missing] = []
    elif missing in ('body', 'scope'):
        entry['calls'][0][missing] = None
    else:
        entry['results'][0]['result' if missing == 'arrow' else missing] = None
    with pytest.raises(reader().ObservationError):
        reader().view(core, operations)


@pytest.mark.parametrize(
    'changed',
    [
        'input-id',
        'body-id',
        'scope-id',
        'body-hash',
        'lineage-id',
        'lineage-ordinal',
        'answer-index',
        'arrow-row',
        'metadata',
        'result-data',
        'run',
        'entry-owner',
        'model',
    ],
)
def test_contradictory_actual_binding_is_refused(changed):
    core, operations = actual_observation()
    entry = core['predictions'][0]
    call = entry['calls'][0]
    final = entry['results'][0]
    if changed == 'input-id':
        core['inputs'][3]['input']['input']['request_id'] = 'stale-input'
    elif changed == 'body-id':
        call['body']['metadata']['request_id'] = 'stale-body'
        core['predicted']['body'] = copy.deepcopy(call['body'])
    elif changed == 'scope-id':
        call['scope']['call_request_id'] = 'stale-scope'
    elif changed == 'body-hash':
        call['body_sha256'] = 'f' * 64
    elif changed == 'lineage-id':
        final['data']['lineage'][0]['call_request_id'] = 'stale-lineage'
        core['inputs'][4]['result'] = copy.deepcopy(final['data'])
    elif changed == 'lineage-ordinal':
        final['data']['lineage'][0]['answer_ordinal'] = 2
        core['inputs'][4]['result'] = copy.deepcopy(final['data'])
    elif changed == 'answer-index':
        final['result']['rows'][1][-1] = {'i64': '0'}
        core['predicted']['result'] = copy.deepcopy(final['result'])
    elif changed == 'arrow-row':
        final['result']['rows'].pop()
        core['predicted']['result'] = copy.deepcopy(final['result'])
    elif changed == 'metadata':
        final['metadata']['endpoint'] = 'changed final metadata'
    elif changed == 'result-data':
        final['data']['predicted']['request_id'] = 'changed-result'
    elif changed == 'run':
        entry['runs'].remove(3)
    elif changed == 'entry-owner':
        core['inputs'][3]['entry'] = None
    else:
        core['inputs'][2]['result']['model'] = 'kumo-tabular'
    with pytest.raises(reader().ObservationError):
        reader().view(core, operations)


def test_entry_only_failure_is_not_native_prediction_execution():
    core, operations = actual_observation()
    core['inputs'] = core['inputs'][:2]
    core['predicted'] = None
    core['predictions'][0].update(
        completion='raised', runs=[], calls=[], results=[], final=None
    )
    projected = reader().view(core, operations[:2])
    assert projected['operations'] == operations[:2]
    [prediction] = projected['predictions']
    assert prediction['native'] is False
    assert prediction['model'] is None
    assert prediction['completion'] == 'raised'
    assert all(
        value.get('model') is None
        for value in projected['inputs']
        if value['op'] == 'predict'
    )


def test_unrelated_generic_preparation_cannot_supply_prediction_role_or_model():
    core, operations = actual_observation()
    core.pop('predictions')
    core['predicted'] = None
    core['inputs'] = core['inputs'][:2] + [
        {
            'op': 'prepare_request',
            'run': 2,
            'entry': None,
            'input': {'stage': 'model', 'model': MODEL},
            'result': {'phase': 'model_resolved', 'version': 1, 'model': MODEL},
            'error': None,
        }
    ]
    projected = reader().view(core, operations[:3])
    assert projected['operations'] == operations[:3]
    assert projected['inputs'] == core['inputs']
    assert projected['predictions'] == []


def unresolved_observation():
    core, operations = actual_observation()
    failure = {'kind': 'not_found', 'code': 'UNKNOWN_MODEL'}
    core['inputs'] = core['inputs'][:2] + [
        {
            'op': 'prepare_request',
            'run': 2,
            'entry': 0,
            'input': {'stage': 'model_lookup', 'position': None},
            'result': None,
            'error': failure,
        }
    ]
    core['predictions'][0].update(
        argument={'form': 'exact_string', 'value': 'unknown-model'},
        completion='raised',
        runs=[2],
        calls=[],
        results=[],
        final=None,
        failure={'run': 2, 'relation': 'selected'},
    )
    core['error'] = failure
    core['predicted'] = None
    return core, operations[:3]


def test_failed_model_resolution_retains_none_instead_of_argument_or_expected_model():
    core, operations = unresolved_observation()
    projected = reader().view(core, operations)
    [prediction] = projected['predictions']
    assert prediction['model'] is None
    assert prediction['native'] is True
    assert prediction['failure'] == {'run': 2, 'relation': 'selected'}
    assert all(
        value.get('model') is None
        for value in projected['inputs']
        if value['op'] == 'predict'
    )


def two_observed_predictions():
    core, operations = actual_observation()
    second, _ = actual_observation('sdm-000000000002', 'endpoint-r2')
    for started in second['inputs'][2:]:
        started['run'] += 3
        started['entry'] = 1
        core['inputs'].append(started)
    entry = second['predictions'][0]
    entry['ordinal'] = 1
    entry['runs'] = [run + 3 for run in entry['runs']]
    entry['calls'][0]['run'] += 3
    entry['results'][0]['run'] += 3
    core['predictions'].append(entry)
    core['predicted'] = second['predicted']
    return core, operations + ['prepare_request'] * 3


def test_sequential_independent_entries_keep_separate_calls_and_common_inputs():
    core, operations = two_observed_predictions()
    projected = reader().view(core, operations)
    assert projected['operations'] == [
        'describe_source',
        'materialize',
        'predict',
        'predict',
    ]
    assert [
        item for item in projected['inputs'] if item['op'] == 'predict'
    ] == [
        {'op': 'predict', 'model': MODEL},
        {'op': 'predict', 'model': MODEL},
    ]
    assert [entry['ordinal'] for entry in projected['predictions']] == [0, 1]
    assert all(entry['parent'] is None for entry in projected['predictions'])
