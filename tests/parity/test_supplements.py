# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supplemental evidence must prove exclusions across the sampled graph."""

import copy

import pytest
from supplements import assert_holdout_payloads


def test_uploaded_holdout_checks_related_rows_and_removed_entity():
    original = {
        'schema': {
            'relationships': [
                {'source_columns': ['entity'], 'target_table': 'users'}
            ]
        },
        'context': {
            'instance_table': {
                'columns': ['instance_id', 'entity'],
                'rows': [[0, 10], [1, 20]],
            },
            'related_tables': {
                'users': {
                    'columns': ['instance_id', 'id'],
                    'rows': [[0, 10], [1, 20]],
                }
            },
        },
    }
    fitted = copy.deepcopy(original)
    fitted['context']['instance_table']['rows'] = [[1, 20]]
    fitted['context']['related_tables']['users']['rows'] = [[1, 20]]
    calls = [
        {'operation': 'fit', 'payload': fitted},
        {'operation': 'score', 'payload': {}},
        {'operation': 'score', 'payload': {}},
        {'operation': 'close', 'payload': None},
    ]
    assert_holdout_payloads([original], 2, [10], calls)
    leaked = copy.deepcopy(calls)
    leaked[0]['payload']['context']['related_tables']['users']['rows'].append(
        [0, 10]
    )
    with pytest.raises(AssertionError):
        assert_holdout_payloads([original], 2, [10], leaked)
    wrong = copy.deepcopy(calls)
    wrong[0]['payload']['context']['instance_table']['rows'] = [[0, 10]]
    with pytest.raises(AssertionError):
        assert_holdout_payloads([original], 2, [10], wrong)
    with pytest.raises(AssertionError):
        assert_holdout_payloads([original], 2, [], calls)


def test_response_ids_require_ordered_real_call_correlation_before_normalizing():
    from supplements import normalize_response_ids, response_groups

    case = {'supplement': {'kind': 'batch'}}
    original = {
        'transport': [
            {'operation': 'predict', 'response_id': 'a'},
            {'operation': 'predict', 'response_id': 'b'},
        ],
        'result_details': [
            {
                'request_id': 'a',
                'items': [{'id': 'a'}],
                'metadata': {
                    'batches': 2,
                    'batch_request_ids': ['a', 'b'],
                    'unknown': {'token': 'a'},
                },
            }
        ],
    }
    original['request'] = [{'metadata': {'part': 0}}, {'metadata': {'part': 1}}]
    for index, call in enumerate(original['transport']):
        call['payload'] = {
            'metadata': {'part': index, 'request_id': f'sdm-{index:012x}'}
        }
    normalized = copy.deepcopy(original)
    normalize_response_ids(normalized, case)
    assert normalized['result_details'][0]['request_id'] == 'call:0'
    assert normalized['result_details'][0]['metadata']['batch_request_ids'] == [
        'call:0',
        'call:1',
    ]
    assert normalized['result_details'][0]['metadata']['unknown'] == {
        'token': 'a'
    }
    assert normalized['result_details'][0]['items'] == [{'id': 'a'}]
    wrong = copy.deepcopy(original)
    wrong['result_details'][0]['metadata']['batch_request_ids'].reverse()
    with pytest.raises(ValueError, match='ordered'):
        response_groups(wrong, case)
    wrong = copy.deepcopy(original)
    wrong['transport'][1]['response_id'] = 'a'
    with pytest.raises(ValueError, match='repeated'):
        response_groups(wrong, case)
    wrong = copy.deepcopy(original)
    wrong['result_details'][0]['request_id'] = 'b'
    with pytest.raises(ValueError, match='first actual'):
        response_groups(wrong, case)


def test_fitted_predictions_have_separate_response_groups():
    from supplements import response_groups

    semantic = {
        'transport': [
            {'operation': 'fit'},
            {'operation': 'score', 'response_id': 'a'},
            {'operation': 'score', 'response_id': 'b'},
            {'operation': 'close'},
        ],
        'result_details': [
            {'request_id': 'a', 'metadata': {}},
            {'request_id': 'b', 'metadata': {}},
        ],
    }
    assert response_groups(semantic, {'supplement': {'kind': 'holdout'}})[
        1
    ] == [['a'], ['b']]
    semantic['result_details'][1]['request_id'] = 'a'
    with pytest.raises(ValueError, match='first actual'):
        response_groups(semantic, {'supplement': {'kind': 'holdout'}})


def test_request_nonces_bind_exact_body_metadata_and_never_share_response_ids():
    from supplements import request_ids

    case = {'supplement': {'kind': 'batch'}}
    semantic = {
        'request': [{'metadata': {'part': 0}}, {'metadata': {'part': 1}}],
        'transport': [
            {
                'operation': 'predict',
                'payload': {
                    'metadata': {
                        'part': index,
                        'request_id': f'sdm-{index:012x}',
                    }
                },
            }
            for index in range(2)
        ],
    }
    assert len(request_ids(semantic, case)) == 2
    wrong = copy.deepcopy(semantic)
    wrong['transport'][1]['payload']['metadata']['request_id'] = (
        'sdm-000000000000'
    )
    with pytest.raises(ValueError, match='reused'):
        request_ids(wrong, case)
    wrong = copy.deepcopy(semantic)
    wrong['transport'][1]['payload']['metadata']['part'] = 0
    with pytest.raises(ValueError, match='source request'):
        request_ids(wrong, case)
    wrong = copy.deepcopy(semantic)
    wrong['transport'][0]['payload']['metadata']['request_id'] = 'pred_response'
    with pytest.raises(ValueError, match='invalid generated'):
        request_ids(wrong, case)


def test_session_proof_requires_actual_handle_and_unchanged_pinned_schema():
    from supplements import session_proof

    schema = {'instance_table': {'columns': {'id': {'dtype': 'int64'}}}}
    semantic = {
        'request': [{'schema': schema}],
        'transport': [
            {
                'operation': 'fit',
                'session_id': 'actual',
                'payload': {'schema': schema},
            },
            {'operation': 'score', 'session_id': 'actual', 'payload': {}},
            {'operation': 'score', 'session_id': 'actual', 'payload': {}},
            {'operation': 'close', 'session_id': 'actual', 'payload': None},
        ],
    }
    case = {'supplement': {'kind': 'holdout'}}
    session_proof(semantic, case)
    wrong = copy.deepcopy(semantic)
    wrong['transport'][1]['session_id'] = 'other'
    with pytest.raises(ValueError, match='actual fitted session'):
        session_proof(wrong, case)
    wrong = copy.deepcopy(semantic)
    wrong['transport'][0]['payload']['schema'] = {}
    with pytest.raises(ValueError, match='pinned schema differs'):
        session_proof(wrong, case)
    wrong = copy.deepcopy(semantic)
    wrong['transport'][1]['payload']['schema'] = schema
    with pytest.raises(ValueError, match='redeclares'):
        session_proof(wrong, case)


def test_actual_input_capture_includes_refused_requests_without_normalizing_data():
    from supplements import wire_inputs

    record = {
        'status': 'refused',
        'error': {'stage': 'predict'},
        'semantic': {
            'request': {
                'context': {},
                'predict': {},
                'target': 'label',
                'task': 'regression',
            },
            'wire_calls': [
                {
                    'operation': 'predict',
                    'payload': {
                        'metadata': {'request_id': 'sdm-000000000001'},
                        'context': {'value': 2.5},
                    },
                    'response_id': None,
                }
            ],
        },
    }
    original = copy.deepcopy(record)
    case = {'journey': 'predict'}
    calls = wire_inputs(record, case)
    assert calls == [
        {
            'operation': 'predict',
            'payload': {
                'metadata': {'request_id': 'request:0'},
                'context': {'value': 2.5},
            },
        }
    ]
    assert record == original
    for missing in ('wire_calls',):
        wrong = copy.deepcopy(record)
        wrong['semantic'].pop(missing)
        with pytest.raises(ValueError):
            wire_inputs(wrong, case)
    wrong = copy.deepcopy(record)
    wrong['semantic']['wire_calls'][0]['payload']['metadata']['caller_data'] = (
        'unproven'
    )
    with pytest.raises(ValueError, match='source request'):
        wire_inputs(wrong, case)
    wrong = copy.deepcopy(record)
    wrong['error']['stage'] = 'schema'
    with pytest.raises(ValueError, match='before'):
        wire_inputs(wrong, case)
