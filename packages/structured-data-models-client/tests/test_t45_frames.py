# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public frame LINK dtype, shape and response observations from the frozen client."""

from __future__ import annotations

import copy
import json
import warnings
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from t38_forced_routes_record import bare
from t45_frame_routes import FRAME_ROUTES

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Relationship,
    Schema,
    SDMClient,
)
from sdm_client.materializers.relational import RelationalMaterializer

PANDAS = int(pd.__version__.split('.')[0])
REFERENCE = json.loads(
    Path(__file__)
    .with_name(f't45_frames_pandas{PANDAS}_reference.json')
    .read_text()
)


def _request_identifiers(kind, shape, expected):
    recorded = [call['metadata']['request_id'] for call in expected['sent']]
    if (kind, shape) == ('category', 'ordinary') and not recorded:
        # This exact historical refusal can now reach its one paid request.
        # A second request must still exhaust the finite script.
        return ['t45-category-ordinary-profile']
    return recorded


def test_only_newly_reached_category_ordinary_has_one_finite_authored_id():
    empty = {'sent': []}
    identifiers = iter(_request_identifiers('category', 'ordinary', empty))
    assert next(identifiers) == 't45-category-ordinary-profile'
    with pytest.raises(StopIteration):
        next(identifiers)
    assert _request_identifiers('category', 'window', empty) == []
    assert _request_identifiers('object', 'ordinary', empty) == []
    recorded = {'sent': [{'metadata': {'request_id': 'actual-frozen-id'}}]}
    assert _request_identifiers('category', 'ordinary', recorded) == [
        'actual-frozen-id'
    ]
    assert empty == {'sent': []}


def keys(kind, values):
    if kind == 'object':
        return pd.Series(
            [None if v is None else 'item-' + str(v) for v in values],
            dtype=object,
        )
    if kind in ('string-python', 'string-pyarrow'):
        return pd.Series(
            [None if v is None else 'item-' + str(v) for v in values],
            dtype=pd.StringDtype(storage=kind.split('-')[1]),
        )
    if kind in ('arrow-string', 'arrow-large-string'):
        dtype = pa.string() if kind == 'arrow-string' else pa.large_string()
        return pd.Series(
            pd.arrays.ArrowExtensionArray(
                pa.array(
                    [None if v is None else 'item-' + str(v) for v in values],
                    type=dtype,
                )
            )
        )
    if kind == 'category':
        return pd.Series(
            [None if v is None else 'item-' + str(v) for v in values],
            dtype='category',
        )
    if kind == 'decimal-object':
        return pd.Series(
            [
                None if v is None else Decimal(v) + Decimal('0.25')
                for v in values
            ],
            dtype=object,
        )
    if kind == 'decimal128':
        return pd.Series(
            pd.arrays.ArrowExtensionArray(
                pa.array(
                    [
                        None if v is None else Decimal(v) + Decimal('0.25')
                        for v in values
                    ],
                    type=pa.decimal128(20, 2),
                )
            )
        )
    if kind == 'timestamp':
        return pd.Series(
            [
                pd.NaT
                if v is None
                else pd.Timestamp('2020-01-01') + pd.Timedelta(days=v)
                for v in values
            ]
        )
    if kind == 'uint64-high':
        return pd.Series(
            [None if v is None else 2**63 + v for v in values], dtype='UInt64'
        )
    if kind == 'int64-high':
        return pd.Series(
            [None if v is None else 2**53 + v for v in values], dtype='Int64'
        )
    if kind == 'float64-close':
        return pd.Series(
            [None if v is None else 1 + v * 1e-8 for v in values],
            dtype='float64',
        )
    return pd.Series(values, dtype=kind)


def frames(kind='int64', shape='ordinary'):
    users = pd.DataFrame(
        {
            'user_id': np.arange(8, dtype='int64'),
            'age': np.arange(20, 28, dtype='int64'),
        }
    )
    items = pd.DataFrame(
        {'item_id': keys(kind, list(range(4))), 'amount': [3.0, 4.0, 5.0, 6.0]}
    )
    buys = pd.DataFrame(
        {
            'buy_id': np.arange(120, dtype='int64'),
            'user_id': np.arange(120) % 8,
            'item_id': keys(kind, [(i * 3) % 4 for i in range(120)]),
            'ts': pd.date_range('2023-01-01', periods=120, freq='5D'),
        }
    )
    if shape == 'null':
        buys.loc[9, 'item_id'] = None
    if shape == 'window':
        extra = pd.DataFrame(
            {
                'buy_id': [999000, 999001, 999002, 999003, 999004],
                'user_id': [0] * 5,
                'item_id': keys(kind, [0, 1, 1, 3, 2]),
                'ts': pd.to_datetime(
                    [
                        '2023-01-30',
                        '2023-02-01',
                        '2023-02-02',
                        '2023-03-01',
                        '2023-03-01 00:00:00.000001',
                    ],
                    format='mixed',
                ),
            }
        )
        buys = pd.concat([buys, extra], ignore_index=True)
    return {'users': users, 'items': items, 'buys': buys}


def configure(kind, shape):
    data = frames(kind, shape)
    primary = {'users': 'user_id', 'items': 'item_id', 'buys': 'buy_id'}
    times = {'users': None, 'items': None, 'buys': 'ts'}
    links = None
    if shape in ('composite-rhs', 'explicit-composite-rhs'):
        data['items']['part'] = 0
        primary['items'] = ['item_id', 'part']
    if shape in ('composite-lhs', 'explicit-composite-lhs'):
        data['users']['part'] = 0
        primary['users'] = ['user_id', 'part']
    if shape == 'composite-history':
        data['buys']['part'] = 0
        primary['buys'] = ['buy_id', 'part']
    if shape.startswith('static-'):
        times['buys'] = None
    if shape == 'static-no-ts':
        data['buys'] = data['buys'].drop(columns='ts')
    if shape == 'unicode-ids':
        identities = ['', '\u00e9', '\u6771\u4eac', '\U0001f63a']
        data['items']['item_id'] = pd.Series(identities, dtype=object)
        data['buys']['item_id'] = pd.Series(
            [identities[(i * 3) % 4] for i in range(120)], dtype=object
        )
    if shape in REFERENCE['explicit_shapes']:
        links = [
            Relationship('buys', 'user_id', 'users', 'user_id'),
            Relationship('buys', 'item_id', 'items', 'item_id'),
        ]
    query = 'PREDICT LIST_DISTINCT(buys.item_id, 0, 30, days) FOR EACH users.user_id'
    if shape == 'static' or shape.startswith('static-'):
        query = 'PREDICT LIST_DISTINCT(buys.item_id) FOR EACH users.user_id'
    if shape in ('window', 'empty-filter'):
        condition = (
            'buys.buy_id >= 999000'
            if shape == 'window'
            else 'buys.buy_id >= 999999'
        )
        query = f'PREDICT LIST_DISTINCT(buys.item_id WHERE {condition}, 0, 30, days) FOR EACH users.user_id'
    if shape == 'empty-one-neighborhood':
        query = 'PREDICT LIST_DISTINCT(buys.item_id WHERE buys.user_id != 0, 0, 30, days) FOR EACH users.user_id'
    if shape == 'reverse':
        query = 'PREDICT LIST_DISTINCT(buys.user_id, 0, 30, days) FOR EACH items.item_id'
    return data, primary, times, links, query


class Scripted:
    url = 'scripted://frozen-full-link'
    serves_sessions = True

    def __init__(self):
        self.calls = []
        self.replies = []

    def predict(self, payload):
        self.calls.append(copy.deepcopy(payload))
        frame = payload['predict']['instance_table']
        at = frame['columns'].index('instance_id')
        reply = {
            'id': 'frozen-full-link',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {
                    'row_index': i,
                    'id': str(row[at]),
                    'prediction': None,
                    'rankings': [{'item': 'display-rhs', 'score': 0.9}],
                }
                for i, row in enumerate(frame['rows'])
            ],
        }
        self.replies.append(copy.deepcopy(reply))
        return reply

    def close(self):
        pass


def observe(kind, shape, expected):
    script = Scripted()
    record = {}
    stage = 'setup'
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        try:
            data, primary, times, links, query = configure(kind, shape)
            stage = 'schema'
            schema = Schema.from_data(
                data,
                primary_keys=primary,
                time_columns=times,
                relationships=links,
            )
            stage = 'query'
            question = PredictiveQuery(query, schema)
            stage = 'materialize'
            indices = (
                [i % 8 for i in range(201)]
                if shape == '201-repeated'
                else [0, 1]
            )
            request = question.materialize(
                indices=indices,
                anchor_time=pd.Timestamp('2023-03-01'),
                random_seed=7,
            )
            record['payloads'] = copy.deepcopy(request.payloads)
            record['inspect'] = {
                'context_rows': request.context_rows,
                'predict_rows': request.predict_rows,
                'nothing_to_score': request.nothing_to_score,
            }
            stage = 'predict'
            # Request identifiers are host randomness, replayed from the
            # actual frozen call without masking any transmitted body field.
            identifiers = _request_identifiers(kind, shape, expected)
            with patch(
                'sdm_client.adapters.relational._request_id',
                side_effect=identifiers,
            ):
                with SDMClient('http://localhost:1', max_retries=0) as client:
                    client._transport.close()
                    client._transport = script
                    product = client.predict(request, model=KumoRelational())
                    answer = product.to_pandas()
            if (kind, shape) == ('category', 'ordinary') and not expected[
                'sent'
            ]:
                record['decoded'] = {
                    'model': product.model,
                    'request_id': product.request_id,
                    'metadata': copy.deepcopy(product.metadata),
                    'outputs': list(product.outputs),
                    'dtypes': [str(dtype) for dtype in answer.dtypes],
                    'items': [
                        {
                            'row_index': item.row_index,
                            'id': int(item.id),
                            'id_type': [
                                type(item.id).__module__,
                                type(item.id).__name__,
                            ],
                            'prediction': item.prediction,
                            'rankings': [
                                dict(entry) for entry in item.rankings
                            ],
                        }
                        for item in product.items
                    ],
                }
                record['replies'] = copy.deepcopy(script.replies)
            record['result'] = json.loads(answer.to_json(orient='split'))
            if 'merged' in expected:
                record['merged'] = json.loads(
                    answer.merge(
                        data['users'], left_on='id', right_on='user_id'
                    ).to_json(orient='split')
                )
            record['outcome'] = 'success'
        except Exception as error:
            record.update(
                outcome='error',
                stage=stage,
                error={
                    'type': type(error).__name__,
                    'module': type(error).__module__,
                    'code': getattr(error, 'code', None),
                    'message': str(error),
                },
            )
        record['warnings'] = [
            {'type': type(n.message).__name__, 'message': str(n.message)}
            for n in notices
        ]
        record['sent'] = script.calls
    return json.loads(json.dumps(record))


@pytest.mark.parametrize('core', [False, True], ids=['original', 'core'])
@pytest.mark.parametrize(
    'case',
    REFERENCE['cases'],
    ids=lambda case: case['kind'] + '/' + case['shape'],
)
def test_public_link_frame_journey_matches_frozen_client(
    monkeypatch, core, case
):
    bounded = (case['kind'], case['shape']) == (
        'category',
        'ordinary',
    ) and not case['expected']['sent']

    def observed_route(core_route):
        with monkeypatch.context() as leg:
            leg.setenv('SDM_CORE_ROUTES', FRAME_ROUTES if core_route else '')
            if core_route:

                def old_engine_was_called(*args, **kwargs):
                    raise AssertionError(
                        'core LINK materialization called the Python relational engine'
                    )

                leg.setattr(
                    RelationalMaterializer, '_get_model', old_engine_was_called
                )
            return observe(case['kind'], case['shape'], case['expected'])

    original = observed_route(False) if bounded else None
    observed = observed_route(core) if core or not bounded else original
    expected = copy.deepcopy(case['expected'])
    if core:
        # P3E-2 changes only a leading warning code, using the same pinned
        # comparison as the existing forced-route differential consumers.
        for warning in observed['warnings']:
            warning['message'] = bare(warning['message'])
        if (case['kind'], case['shape']) == ('uint64-high', 'ordinary'):
            # P2D-4 omits this captured data value and preserves the exact
            # class/code, table/column/row, range and remedy around it.
            message = expected['error']['message']
            captured = 'value 9223372036854775808'
            assert message.count(captured) == 1
            expected['error']['message'] = message.replace(
                captured, 'a value', 1
            )
    if bounded and original['outcome'] == 'success':
        paired = copy.deepcopy(original)
        if core:
            for warning in paired['warnings']:
                warning['message'] = bare(warning['message'])
        assert observed == paired
        assert observed['outcome'] == 'success'
        assert observed['inspect'] == {
            'context_rows': 6,
            'predict_rows': 2,
            'nothing_to_score': False,
        }
        assert (
            len(observed['payloads'])
            == len(observed['sent'])
            == len(observed['replies'])
            == 1
        )
        assert observed['warnings'] == expected['warnings']
        assert (
            observed['sent'][0]['metadata']['request_id']
            == 't45-category-ordinary-profile'
        )
        assert (
            observed['decoded']['request_id']
            == observed['replies'][0]['id']
            == 'frozen-full-link'
        )
        assert (
            observed['decoded']['model']
            == observed['replies'][0]['model']
            == 'kumo-relational'
        )
        assert (
            observed['decoded']['metadata']
            == observed['replies'][0]['metadata']
            == {}
        )
        rankings = [{'item': 'display-rhs', 'score': 0.9}]
        assert observed['result'] == {
            'columns': ['row_index', 'id', 'rankings'],
            'index': [0, 1],
            'data': [[0, 0, rankings], [1, 1, rankings]],
        }
        assert [item['id'] for item in observed['decoded']['items']] == [0, 1]
        assert [item['rankings'] for item in observed['decoded']['items']] == [
            rankings,
            rankings,
        ]
    else:
        if bounded:
            assert original == case['expected']
        assert observed == expected
