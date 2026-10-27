# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public text feature and scalar class journeys, frozen before implementation."""

from __future__ import annotations

import copy
import json
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
    SDMWarning,
    StructuredDataError,
)
from sdm_client._routing import RouteError
from sdm_client.materializers.relational import RelationalMaterializer

REFERENCE = json.loads(
    Path(__file__).with_name('t44_text_reference.json').read_text()
)
ROUTES = ','.join(
    (
        'schema.from-data/frames/frames',
        'schema.from-data/mapping/frames',
        'query.materialize/indices-regression/frames',
        'client.predict/regression/frames',
    )
)


@pytest.fixture(params=[False, True], ids=['original', 'core'])
def core(request, monkeypatch):
    selected = request.param
    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES if selected else '')
    if selected:

        def old_engine_was_called(*args, **kwargs):
            raise AssertionError('the core text journey called the old engine')

        monkeypatch.setattr(
            RelationalMaterializer, '_get_model', old_engine_was_called
        )
    return selected


def text(values, storage, chunks=False):
    if storage in ('python', 'pyarrow'):
        return pd.Series(values, dtype=pd.StringDtype(storage=storage))
    dtype = pa.string() if storage == 'utf8' else pa.large_string()
    array = (
        pa.chunked_array(
            [pa.array(values[:3], type=dtype), pa.array(values[3:], type=dtype)]
        )
        if chunks
        else pa.array(values, type=dtype)
    )
    return pd.Series(pd.arrays.ArrowExtensionArray(array))


def shop(storage='python', chunks=False, empty=False):
    values = ['', None, 'é', '😺', 'a\x00b', '東京', 'red', 'blue']
    return {
        'users': pd.DataFrame(
            {
                'user_id': np.arange(1, 9),
                'age': np.arange(21, 29, dtype=np.int32),
                'category': text(
                    ['red', 'blue', 'red', None, 'blue', 'red', 'blue', 'red'],
                    storage,
                    chunks,
                ),
                'note': text([None] * 8 if empty else values, storage, chunks),
            }
        ),
        'orders': pd.DataFrame(
            {
                'order_id': np.arange(64),
                'user_id': np.repeat(np.arange(1, 9), 8),
                'marker': np.arange(64),
                'note': text(
                    ([None] * 8 if empty else values) * 8, storage, chunks
                ),
            }
        ),
    }


def question(frames, target='age', prose=False):
    schema = Schema.from_data(
        frames,
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
        time_columns={},
    )
    if prose:
        schema['users']['note'].stype = 'text'
        schema['orders']['note'].stype = 'text'
    return PredictiveQuery(
        f'PREDICT users.{target} FOR EACH users.user_id', schema
    )


def materialize(asked, target='age', **options):
    return asked.materialize(
        indices=[7, 8],
        num_context_samples=6 if target == 'category' else 3,
        num_neighbors=[3],
        num_hops=1,
        random_seed=7,
        **options,
    )


def inspection(request):
    return {
        'payloads': copy.deepcopy(request.payloads),
        'context_rows': request.context_rows,
        'predict_rows': request.predict_rows,
    }


class Scripted:
    url = 'scripted://text-feature-reference'
    serves_sessions = True

    def __init__(self, target):
        self.calls = []
        self.target = target

    def predict(self, payload):
        self.calls.append(copy.deepcopy(payload))
        table = payload['predict']['instance_table']
        at = table['columns'].index('instance_id')
        return {
            'id': 'text-feature-reference',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {
                    'row_index': i,
                    'id': str(row[at]),
                    'prediction': 'red'
                    if self.target == 'category'
                    else 100.0 + len(self.calls),
                }
                for i, row in enumerate(table['rows'])
            ],
        }

    def close(self):
        pass


def predict(request, target, frames, expected):
    scripted = Scripted(target)
    # Replay captured UUID host facts, keeping the complete body comparison.
    ids = [call['metadata']['request_id'] for call in expected['sent']]
    with patch('sdm_client.adapters.relational._request_id', side_effect=ids):
        with SDMClient('http://localhost:1', max_retries=0) as client:
            client._transport.close()
            client._transport = scripted
            result = client.predict(request, model=KumoRelational()).to_pandas()
    assert scripted.calls == expected['sent']
    assert json.loads(result.to_json(orient='split')) == expected['result']
    joined = result.merge(frames['users'], left_on='id', right_on='user_id')
    assert json.loads(joined.to_json(orient='split')) == expected['merged']


@pytest.mark.parametrize('storage', ['python', 'pyarrow', 'utf8'])
@pytest.mark.parametrize('target', ['age', 'category'])
@pytest.mark.parametrize(
    'variant', ['ordinary', 'chunks', 'allnull', 'batched', 'prose']
)
def test_text_journey_keeps_complete_frozen_payloads_and_results(
    core, storage, target, variant
):
    expected = REFERENCE['records'][f'{storage}-{target}-{variant}']
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        frames = shop(
            storage, chunks=variant == 'chunks', empty=variant == 'allnull'
        )
        asked = question(frames, target, prose=variant == 'prose')
        request = materialize(
            asked,
            target,
            **({'rows_per_call': 1} if variant == 'batched' else {}),
        )
        before = inspection(request)
        assert before == {key: expected[key] for key in before}
        # The public request's frames must expose the selected native values.
        assert request.predict.related_tables['users']['user_id'].tolist() == [
            7,
            8,
        ]
        if core:
            assert request._core is not None
        predict(request, target, frames, expected)
        assert inspection(request) == before
    assert [
        {'type': type(w.message).__name__, 'message': str(w.message)}
        for w in caught
    ] == expected['warnings']


@pytest.mark.parametrize('unknown', [False, True])
def test_text_snapshot_survives_mutation_and_failed_first_lookup(core, unknown):
    expected = REFERENCE['records'][f'mutation-{unknown}']
    frames = shop()
    asked = question(frames)
    if unknown:
        with pytest.raises(StructuredDataError) as failed:
            asked.materialize(
                indices=[9999],
                num_context_samples=3,
                num_neighbors=[3],
                num_hops=1,
                random_seed=7,
            )
        assert {
            'type': type(failed.value).__name__,
            'code': failed.value.code,
            'message': str(failed.value),
        } == expected['failure']
    else:
        assert inspection(materialize(asked)) == expected['initial']
    frames['orders']['note'] = text(['changed'] * 64, 'python')
    assert inspection(materialize(asked)) == expected['cached']
    assert inspection(materialize(question(frames))) == expected['fresh']


def test_text_sanitization_warning_precedes_unknown_entity_error(core):
    expected = REFERENCE['records']['drop-and-unknown']
    frames = shop()
    asked = question(frames)
    frames['orders'].loc[1, 'order_id'] = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with pytest.raises(StructuredDataError) as failed:
            asked.materialize(
                indices=[9999],
                num_context_samples=3,
                num_neighbors=[3],
                num_hops=1,
                random_seed=7,
            )
    assert {
        'type': type(failed.value).__name__,
        'code': failed.value.code,
        'message': str(failed.value),
    } == expected['error']
    assert [
        {'type': type(w.message).__name__, 'message': str(w.message)}
        for w in caught
    ] == expected['warnings']


def test_text_warning_as_error_keeps_the_detached_snapshot(core):
    frames = shop()
    asked = question(frames)
    frames['orders'].loc[1, 'order_id'] = 0
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'error', message='the graph was built without', category=SDMWarning
        )
        with pytest.raises(SDMWarning, match='the graph was built without'):
            materialize(asked)
    frames['orders']['note'] = text(['changed'] * 64, 'python')
    with pytest.warns(SDMWarning, match='the graph was built without'):
        cached = materialize(asked)
    assert (
        'changed'
        not in cached.predict.related_tables['orders']['note'].tolist()
    )


@pytest.mark.parametrize('storage', ['python', 'pyarrow', 'utf8', 'largeutf8'])
def test_text_graph_keys_remain_an_explicit_unfinished_core_family(
    monkeypatch, storage
):
    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES)
    frames = shop(storage)
    frames['users']['user_id'] = text([f'u{i}' for i in range(1, 9)], storage)
    frames['orders']['user_id'] = text(
        [f'u{i}' for i in range(1, 9) for _ in range(8)], storage
    )
    asked = question(frames)
    with pytest.raises((NotImplementedError, RouteError)):
        asked.materialize(
            indices=['u7', 'u8'],
            num_context_samples=3,
            num_neighbors=[3],
            random_seed=7,
        )


def test_exact_largeutf8_native_materialization_keeps_its_frozen_failure_boundary(
    core,
):
    expected = REFERENCE['records']['largeutf8-age-ordinary']
    asked = question(shop('largeutf8'))
    if core:
        with pytest.raises(RouteError, match='large.*string'):
            materialize(asked)
    else:
        with pytest.raises(StructuredDataError) as failed:
            materialize(asked)
        assert {
            'type': type(failed.value).__name__,
            'code': failed.value.code,
            'message': str(failed.value),
        } == expected['error']
