# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""First public numeric-frame consumer, before its core implementation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
    StructuredDataError,
)
from sdm_client.materializers.relational import RelationalMaterializer

REFERENCE = json.loads(
    Path(__file__).with_name('t44_numeric_static_reference.json').read_text()
)
CORE = ','.join(
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
    monkeypatch.setenv('SDM_CORE_ROUTES', CORE if selected else '')
    if selected:

        def old_engine_was_called(*args, **kwargs):
            raise AssertionError('the core frame journey called the old engine')

        monkeypatch.setattr(
            RelationalMaterializer, '_get_model', old_engine_was_called
        )
    return selected


def shop():
    users = pd.DataFrame(
        {
            'user_id': np.arange(1, 9),
            'age': np.arange(21, 29, dtype=np.int32),
        }
    )
    owners = np.repeat(np.arange(1, 9), np.arange(6, 14))
    orders = pd.DataFrame(
        {
            'order_id': np.arange(len(owners)),
            'user_id': owners,
            'amount': np.arange(1, len(owners) + 1, dtype=np.float32),
            'marker': np.arange(len(owners), dtype=np.int64),
        }
    )
    return {'users': users, 'orders': orders}


def query(frames=None, target='users.age'):
    frames = shop() if frames is None else frames
    schema = Schema.from_data(
        frames,
        primary_keys={'users': 'user_id', 'orders': 'order_id'},
        time_columns={},
    )
    return PredictiveQuery(f'PREDICT {target} FOR EACH users.user_id', schema)


def materialize(question, *, seed=7, indices=(7, 8), **options):
    return question.materialize(
        indices=list(indices),
        num_context_samples=3,
        num_neighbors=[3],
        num_hops=1,
        random_seed=seed,
        **options,
    )


def markers(request):
    return request.predict.related_tables['orders']['marker'].tolist()


class Scripted:
    url = 'scripted://numeric-static-journey'
    serves_sessions = True

    def __init__(self):
        self.calls = []

    def predict(self, payload):
        self.calls.append(copy.deepcopy(payload))
        table = payload['predict']['instance_table']
        at = table['columns'].index('instance_id')
        return {
            'id': 'numeric-static-journey',
            'model': 'kumo-relational',
            'metadata': {},
            'predictions': [
                {
                    'row_index': i,
                    'id': str(row[at]),
                    'prediction': 100.0 + len(self.calls),
                }
                for i, row in enumerate(table['rows'])
            ],
        }

    def close(self):
        pass


def predict(request):
    scripted = Scripted()
    with SDMClient('http://localhost:1', max_retries=0) as client:
        client._transport.close()
        client._transport = scripted
        result = client.predict(request, model=KumoRelational())
    return result.to_pandas(), scripted.calls


def test_numeric_static_journey_inspects_and_predicts(core):
    request = materialize(query())
    assert markers(request) == REFERENCE['seed7_markers']
    assert request.context.instance_table['TARGET'].tolist() == [21]
    assert 'TARGET' not in request.predict.instance_table
    assert request.predict.related_tables['users']['user_id'].tolist() == [7, 8]
    assert request.context_rows == 1
    assert request.predict_rows == 2
    if core:
        assert request._core is not None
    before = copy.deepcopy(request.payloads)
    result, sent = predict(request)
    assert result['row_index'].tolist() == [0, 1]
    assert result['id'].tolist() == [7, 8]
    assert result['prediction'].tolist() == [101.0, 101.0]
    assert len(sent) == 1
    assert request.payloads == before


def test_numeric_static_multi_call_predictions_merge_in_input_order(core):
    frames = shop()
    request = materialize(query(frames), rows_per_call=1)
    assert len(request.payloads) == 2
    selected = []
    for payload in request.payloads:
        table = payload['predict']['related_tables']['orders']
        at = table['columns'].index('marker')
        selected.append([row[at] for row in table['rows']])
    assert selected == REFERENCE['batched_markers']
    result, sent = predict(request)
    assert len(sent) == 2
    assert result['row_index'].tolist() == [0, 1]
    assert result['id'].tolist() == [7, 8]
    assert result['prediction'].tolist() == REFERENCE['merged_predictions']
    joined = result.merge(
        frames['users'], left_on='id', right_on='user_id', validate='one_to_one'
    )
    assert joined['age'].tolist() == [27, 28]


def test_core_df1_adjacency_preserves_reversed_source_row_ties(monkeypatch):
    monkeypatch.setenv('SDM_CORE_ROUTES', CORE)

    def old_engine_was_called(*args, **kwargs):
        raise AssertionError('the DF-1 frame journey called the old engine')

    monkeypatch.setattr(
        RelationalMaterializer, '_get_model', old_engine_was_called
    )
    frames = shop()
    frames['orders'] = frames['orders'].iloc[::-1].reset_index(drop=True)
    request = materialize(query(frames))
    # DF-1 keeps source-row order within a destination's neighborhood.
    # Reverse each original group at the frozen selected positions.
    assert markers(request) == [61, 53, 55, 70, 66, 65]


def test_numeric_static_neighbor_stream_resets_and_continues_per_query(core):
    question = query()
    first = materialize(question, seed=None)
    second = materialize(question, seed=None)
    assert markers(first) == REFERENCE['unseeded_markers'][0]
    assert markers(second) == REFERENCE['unseeded_markers'][1]
    assert markers(materialize(query(), seed=None)) == markers(first)
    seeded = materialize(question, seed=7)
    repeated = materialize(question, seed=7)
    assert markers(seeded) == markers(repeated) == REFERENCE['seed7_markers']
    continued = materialize(question, seed=None)
    assert markers(continued) == REFERENCE['after_seed7_markers']


def test_unknown_entity_failure_does_not_consume_neighbor_draws(core):
    question = query()
    first = materialize(question, seed=None)
    with pytest.raises(StructuredDataError, match='unresolved row') as failed:
        materialize(question, seed=None, indices=[9999])
    assert failed.value.code == 'INVALID_REQUEST'
    after = materialize(question, seed=None)
    assert markers(first) == REFERENCE['unseeded_markers'][0]
    assert markers(after) == REFERENCE['unseeded_markers'][1]


def test_first_numeric_materialization_keeps_its_detached_snapshot(core):
    frames = shop()
    question = query(frames)
    initial = materialize(question)
    amounts = initial.predict.related_tables['orders']['amount'].tolist()
    frames['orders']['amount'] = np.full(
        len(frames['orders']), 999, dtype=np.float32
    )
    cached = materialize(question)
    assert cached.predict.related_tables['orders']['amount'].tolist() == amounts
    rebuilt = materialize(query(frames))
    assert (
        rebuilt.predict.related_tables['orders']['amount'].tolist() == [999] * 6
    )
    assert (
        initial.predict.related_tables['orders']['amount'].tolist() == amounts
    )


@pytest.mark.parametrize('aggregation', ['COUNT', 'SUM'])
def test_numeric_static_aggregation_keeps_the_missing_time_refusal(
    core, aggregation
):
    question = query(target=f'{aggregation}(orders.amount)')
    with pytest.raises(
        StructuredDataError, match='missing a time range'
    ) as failed:
        materialize(question)
    assert failed.value.code == 'INVALID_REQUEST'
