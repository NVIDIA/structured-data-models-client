# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen public entity-fold and keyless-history frame journeys."""

from __future__ import annotations

import copy
import json
import warnings
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from t45_frame_routes import FRAME_ROUTES
from test_t44_text_frames import p3e2_warning_codes

from sdm_client import KumoRelational, PredictiveQuery, Schema, SDMClient
from sdm_client.materializers.relational import RelationalMaterializer

ROUTES = FRAME_ROUTES


def _frozen_fold_product(expected):
    product = copy.deepcopy(expected)
    caller_warning = {
        'type': 'Pandas4Warning',
        'message': "The default 'epoch' date format is deprecated and will be removed in a future version, please use 'iso' date format instead.",
    }
    if product['case'] == 'fold-entity' and product['warnings'][-1:] == [
        caller_warning
    ]:
        product['warnings'].pop()
    return product


def _assert_fold_profile(actual, original, expected, *, core):
    paired = copy.deepcopy(original)
    pinned = _frozen_fold_product(expected)
    if core:
        paired['warnings'] = p3e2_warning_codes(paired['warnings'])
        pinned['warnings'] = p3e2_warning_codes(pinned['warnings'])
    assert actual == paired
    assert [part['phase'] for part in actual['serialization']] == [
        'result.to_json',
        'merged.to_json',
    ]
    product = copy.deepcopy(actual)
    product.pop('serialization')
    assert product == pinned


def _serialized(frame, phase, record):
    # Capture only the caller's actual default-format serialization, while
    # retaining its complete native warnings in the same-runtime observation.
    with warnings.catch_warnings(record=True) as native:
        try:
            encoded = frame.to_json(orient='split')
        finally:
            record['serialization'].append(
                {
                    'phase': phase,
                    'warnings': [
                        {
                            'type': type(item.message).__name__,
                            'module': item.category.__module__,
                            'message': str(item.message),
                            'filename': item.filename,
                        }
                        for item in native
                    ],
                }
            )
    return json.loads(encoded)


@pytest.mark.parametrize(
    'changed', ['warning', 'phase', 'input', 'body', 'result']
)
def test_fold_profile_pair_keeps_caller_warning_and_every_pinned_product(
    changed,
):
    expected = json.loads(
        Path(__file__)
        .with_name('t44_fold_keyless_pandas3_reference.json')
        .read_text()
    )['records']['fold-entity']
    original = _frozen_fold_product(expected)
    original['serialization'] = [
        {'phase': 'result.to_json', 'warnings': []},
        {
            'phase': 'merged.to_json',
            'warnings': [
                {
                    'type': 'Pandas4Warning',
                    'message': 'actual native words',
                    'module': 'pandas.errors',
                    'filename': __file__,
                }
            ],
        },
    ]
    _assert_fold_profile(
        original, copy.deepcopy(original), expected, core=False
    )
    actual = copy.deepcopy(original)
    if changed == 'warning':
        actual['serialization'][1]['warnings'][0]['message'] += ' changed'
    elif changed == 'phase':
        actual['serialization'].reverse()
    elif changed == 'input':
        actual['input']['entity_counts'][0] += 1
    elif changed == 'body':
        actual['sent'][0]['metadata']['source_query'] += ' '
    else:
        actual['result']['data'][0][-1] += 1
    if changed in ('input', 'body', 'result'):
        # Matching twins must still satisfy every unchanged pinned product.
        original = copy.deepcopy(actual)
    with pytest.raises(AssertionError):
        _assert_fold_profile(actual, original, expected, core=False)


def question(case):
    users = pd.DataFrame(
        {'user_id': np.arange(1, 9), 'age': np.arange(21, 29, dtype=np.int32)}
    )
    if case == 'fold-entity':
        # Both repeated 01:30 wall times are actual distinct UTC instants.
        instants = (
            pd.to_datetime(
                [
                    '2024-11-03T04:30:00Z',
                    '2024-11-03T05:15:00Z',
                    '2024-11-03T05:30:00Z',
                    '2024-11-03T05:45:00Z',
                    '2024-11-03T06:15:00Z',
                    '2024-11-03T06:30:00Z',
                    '2024-11-03T06:45:00Z',
                    '2024-11-03T07:30:00Z',
                ],
                utc=True,
            )
            .as_unit('ns')
            .tz_convert('America/New_York')
        )
        users['signup'] = pd.Series(instants.take([0, 1, 2, 3, 4, 6, 2, 5]))
        assert users.loc[6, 'signup'].hour == users.loc[7, 'signup'].hour == 1
        assert (
            users.loc[6, 'signup'].minute == users.loc[7, 'signup'].minute == 30
        )
        assert [users.loc[i, 'signup'].fold for i in (6, 7)] == [0, 1]
    else:
        assert case == 'keyless-history'
        instants = pd.to_datetime(
            [
                '1970-01-01T00:00:00',
                '1970-01-01T00:00:00',
                '1970-01-01T00:00:01',
                '1970-01-01T00:00:01',
                '1970-01-01T00:00:02',
                '1970-01-01T00:00:02',
                '1970-01-01T00:00:03',
                '1970-01-01T00:00:03',
            ]
        ).as_unit('ns')
    events = pd.DataFrame(
        {
            'event_id': np.arange(64),
            'user_id': np.repeat(np.arange(1, 9), 8),
            'marker': np.arange(64),
            'amount': np.arange(64, dtype=np.float32),
            'at': pd.Series(instants.take(np.arange(64) % 8)),
        }
    )
    data = {'users': users, 'events': events}
    schema = Schema.from_data(
        data,
        primary_keys={'users': 'user_id', 'events': 'event_id'},
        time_columns={
            'users': 'signup' if case == 'fold-entity' else None,
            'events': 'at',
        },
    )
    if case == 'keyless-history':
        schema['events'].primary_key = None
    assert schema['events'].primary_key == (
        None if case == 'keyless-history' else 'event_id'
    )
    assert schema['users'].time_column == (
        'signup' if case == 'fold-entity' else None
    )
    facts = {
        'history_primary_key': schema['events'].primary_key,
        'entity_time': schema['users'].time_column,
        'history_time': schema['events'].time_column,
        'history_dtype': str(events['at'].dtype),
        'history_counts': events['at'].astype('int64').tolist(),
    }
    if case == 'fold-entity':
        facts['entity_dtype'] = str(users['signup'].dtype)
        facts['entity_counts'] = users['signup'].astype('int64').tolist()
        facts['scored_folds'] = [users.loc[i, 'signup'].fold for i in (6, 7)]
    return (
        PredictiveQuery('PREDICT users.age FOR EACH users.user_id', schema),
        data,
        facts,
    )


class Scripted:
    url = 'scripted://frozen-frame-fold-keyless'
    serves_sessions = True

    def __init__(self):
        self.calls = []

    def predict(self, payload):
        self.calls.append(copy.deepcopy(payload))
        table = payload['predict']['instance_table']
        at = table['columns'].index('instance_id')
        return {
            'id': 'frame-fold-keyless',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {'row_index': i, 'id': str(row[at]), 'prediction': 101.0 + i}
                for i, row in enumerate(table['rows'])
            ],
        }

    def close(self):
        pass


def observe(case, expected=None):
    script = Scripted()
    record = {'case': case, 'serialization': []}
    stage = 'schema'
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            asked, data, facts = question(case)
            record['input'] = facts
            stage = 'materialize'
            request = asked.materialize(
                indices=[7, 8],
                num_context_samples=3,
                num_neighbors=[8] if case == 'fold-entity' else [3],
                num_hops=1,
                random_seed=7,
                anchor_time='entity'
                if case == 'fold-entity'
                else pd.Timestamp('1970-01-01T00:00:04'),
            )
            assert type(request.payloads) is tuple
            record['payloads'] = copy.deepcopy(list(request.payloads))
            record['inspection'] = {
                'context_rows': request.context_rows,
                'predict_rows': request.predict_rows,
                'nothing_to_score': request.nothing_to_score,
            }
            stage = 'predict'
            identifiers = (
                [body['metadata']['request_id'] for body in expected['sent']]
                if expected is not None
                else None
            )
            with (
                patch(
                    'sdm_client.adapters.relational._request_id',
                    side_effect=identifiers,
                )
                if identifiers is not None
                else nullcontext()
            ):
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = script
                    answer = client.predict(request, model=KumoRelational())
            result = answer.to_pandas()
            record['result'] = _serialized(result, 'result.to_json', record)
            record['merged'] = _serialized(
                result.merge(data['users'], left_on='id', right_on='user_id'),
                'merged.to_json',
                record,
            )
            record['outcome'] = 'success'
        except Exception as error:
            record.update(
                outcome='error',
                stage=stage,
                error={
                    'type': type(error).__name__,
                    'code': getattr(error, 'code', None),
                    'message': str(error),
                },
            )
        record['warnings'] = [
            {'type': type(item.message).__name__, 'message': str(item.message)}
            for item in notices
        ]
        record['sent'] = script.calls
    return record


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize('case', ['fold-entity', 'keyless-history'])
def test_public_frame_fold_and_keyless_neighbors_keep_complete_frozen_journeys(
    monkeypatch, core, case
):
    cell = 'pandas2' if int(pd.__version__.split('.')[0]) == 2 else 'pandas3'
    expected = copy.deepcopy(
        json.loads(
            Path(__file__)
            .with_name(f't44_fold_keyless_{cell}_reference.json')
            .read_text()
        )['records'][case]
    )
    assert expected['outcome'] == 'success'

    def observed_route(core_route):
        with monkeypatch.context() as leg:
            leg.setenv('SDM_CORE_ROUTES', ROUTES if core_route else '')
            if core_route:

                def old_engine(*args, **kwargs):
                    raise AssertionError(
                        'core fold/keyless journey called the old engine'
                    )

                leg.setattr(RelationalMaterializer, '_get_model', old_engine)
            return observe(case, expected)

    original = observed_route(False)
    actual = observed_route(True) if core else original
    _assert_fold_profile(actual, original, expected, core=core)
