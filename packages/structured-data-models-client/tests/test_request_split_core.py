# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public request inspection uses shared Rust policy without altering the request."""

import copy
import pickle
import warnings

import pandas as pd
import pyarrow as pa
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Request, Schema


@pytest.fixture(autouse=True)
def default_routes_without_legacy_decoder(monkeypatch):
    import sdm_client.request as module

    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)

    def legacy(*args, **kwargs):
        pytest.fail('public inspection called the legacy Python decoder')

    monkeypatch.setattr(module, '_split_from', legacy)


def users():
    schema = Schema.from_data(
        {
            'users': pd.DataFrame(
                {'id': range(24), 'value': [i * 1.25 for i in range(24)]}
            )
        },
        primary_keys={'users': 'id'},
    )
    schema['users']['value'].stype = 'numerical'
    return schema


def test_materialize_inspect_and_pickle_preserve_payloads_and_independent_frames():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(
            'PREDICT users.value FOR EACH users.id', users()
        ).materialize(indices=[1, 2], random_seed=7, model=KumoRelational())
    payloads = copy.deepcopy(request.payloads)
    first = request.context
    second = request.context
    prediction = request.predict
    assert len(first.instance_table) == request.context_rows
    assert len(prediction.instance_table) == request.predict_rows == 2
    assert 'users' in first.related_tables
    pd.testing.assert_frame_equal(first.instance_table, second.instance_table)
    pd.testing.assert_frame_equal(
        first.related_tables['users'], second.related_tables['users']
    )
    first.instance_table.iloc[0, 0] = -999
    first.related_tables['users'].iloc[0, 0] = -999
    pd.testing.assert_frame_equal(
        request.context.instance_table, second.instance_table
    )
    pd.testing.assert_frame_equal(
        request.context.related_tables['users'], second.related_tables['users']
    )
    assert request.payloads == payloads
    restored = pickle.loads(pickle.dumps(request))
    assert restored.payloads == payloads
    pd.testing.assert_frame_equal(
        restored.context.instance_table, second.instance_table
    )
    pd.testing.assert_frame_equal(
        restored.predict.instance_table, prediction.instance_table
    )
    assert restored.payloads == payloads


def test_inspection_keeps_nullable_wide_integers_exact():
    # Public already-materialized requests carry declared writer dtypes.
    # Decimal strings above 2**53 must not travel through pandas float inference.
    wide = 2**53 + 1
    declaration = {
        'columns': {
            'instance_id': {'dtype': 'int64'},
            'wide': {'dtype': 'int64'},
            'empty': {'dtype': 'int64'},
        }
    }
    payload = {
        'schema': {
            'instance_table': declaration,
            'related_tables': {'users': declaration},
        },
        'task': {
            'kind': 'regression',
            'target': {'column_name': 'wide', 'dtype': 'int64'},
        },
        'context': {
            'instance_table': {
                'format': 'arrays',
                'columns': ['instance_id', 'wide', 'empty'],
                'rows': [[0, str(wide), None], [1, None, None]],
            },
            'related_tables': {
                'users': {
                    'format': 'arrays',
                    'columns': ['instance_id', 'wide', 'empty'],
                    'rows': [[0, str(wide), None], [1, None, None]],
                }
            },
        },
    }
    original = copy.deepcopy(payload)
    request = Request.materialized(
        payloads=(payload,),
        schema=users(),
        query='PREDICT users.value FOR EACH users.id',
        indices=[],
        entity_table='users',
    )
    for frame in [
        request.context.instance_table,
        request.context.related_tables['users'],
    ]:
        assert frame['wide'].dtype == pd.ArrowDtype(pa.int64())
        assert frame['empty'].dtype == pd.ArrowDtype(pa.int64())
        assert frame['wide'].iloc[0] == wide
        assert pd.isna(frame['wide'].iloc[1])
        assert frame['empty'].isna().all()
    assert request.predict.instance_table.empty
    assert request.predict.instance_table['wide'].dtype == pd.ArrowDtype(
        pa.int64()
    )
    assert payload == original
