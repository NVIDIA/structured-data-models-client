# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wide public requests use bounded native preparation without changing types."""

import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoTabular,
    PredictiveQuery,
    Request,
    Schema,
    StructuredDataError,
)
from sdm_client._rust import direct


def test_wide_schema_and_long_query_materialize_once(monkeypatch):
    target = 'x' * 70_000
    frame = pd.DataFrame(
        {
            'id': [1, 2, 3, 4],
            **{f'c{i}': [0.5, 1.5, 2.5, 3.5] for i in range(5_000)},
            target: [1.5, 2.0, 3.0, 4.0],
        }
    )
    text = f'PREDICT users.{target} FOR EACH users.id'
    assert len(text.encode()) > 64 * 1024
    operations = []
    scored_checks = []
    original = direct._run

    def run(operation, *args, **kwargs):
        operations.append(operation)
        if (
            operation == 'check_direct_preparation'
            and args[0].get('kind') == 'scored_column'
        ):
            scored_checks.append(args[0])
        return original(operation, *args, **kwargs)

    monkeypatch.setattr(direct, '_run', run)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'users': frame})
        request = PredictiveQuery(text, schema).materialize(indices=[1, 2])
    assert request.predict.instance_table['id'].tolist() == [1, 2]
    assert request.context.instance_table['id'].tolist() == [3, 4]
    assert request.context.instance_table[target].tolist() == [3.0, 4.0]
    assert len(request.context.instance_table.columns) == 5002
    assert operations.count('prepare_direct') == 1
    assert scored_checks == []


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('values', [[np.nan, np.nan], [np.nan, 2.0]])
def test_float_batch_keeps_nulls_and_schema(dtype, values):
    from test_tabular_direct_core import Endpoint, client

    context = pd.DataFrame(
        {
            'id': [1, 2],
            'value': np.asarray(values, dtype=dtype),
            'target': [0.0, 1.0],
        }
    )
    scoring = pd.DataFrame(
        {'id': [3], 'value': np.asarray([np.nan], dtype=dtype)}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'data': context}, primary_keys={'data': 'id'}
        )
        request = Request(
            context,
            scoring,
            target='target',
            task='regression',
            schema=schema,
            entity_table='data',
        )
    assert request.context.instance_table is context
    assert request.predict.instance_table is scoring
    assert request.schema['data']['value'].dtype == dtype
    assert (
        request.context.instance_table['value'].isna().tolist()
        == pd.isna(values).tolist()
    )
    assert request.predict.instance_table['value'].isna().all()
    endpoint = Endpoint(
        {
            'model': 'kumo-tabular',
            'id': 'float-result',
            'metadata': {},
            'predictions': [{'row_index': 0, 'prediction': 1.0}],
        }
    )
    with client(endpoint) as model_client:
        assert len(model_client.predict(request, model=KumoTabular())) == 1
    assert len(endpoint.calls) == 1
    body = endpoint.calls[0]
    encoded = body['context']['instance_table']
    at = encoded['columns'].index('value')
    assert [row[at] for row in encoded['rows']] == [
        None if pd.isna(value) else value for value in values
    ]
    encoded = body['predict']['instance_table']
    at = encoded['columns'].index('value')
    assert [row[at] for row in encoded['rows']] == [None]


def test_unsigned_64_with_floats_keeps_ordered_preparation(monkeypatch):
    context = pd.DataFrame(
        {
            'id': [1, 2],
            'value': [0.5, 1.5],
            'wide': np.array([2**63, 2**63 + 1], dtype='uint64'),
            'target': [0, 1],
        }
    )
    scoring = pd.DataFrame(
        {
            'id': [3],
            'value': [2.5],
            'wide': np.array([2**63 + 2], dtype='uint64'),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'data': context}, primary_keys={'data': 'id'}
        )
    operations = []
    original = direct._run

    def run(operation, *args, **kwargs):
        operations.append((operation, args[0].get('kind'), args[0].get('name')))
        return original(operation, *args, **kwargs)

    monkeypatch.setattr(direct, '_run', run)
    Request(
        context,
        scoring,
        target='target',
        task='regression',
        schema=schema,
        entity_table='data',
    )
    assert ('check_direct_preparation', 'scored_column', 'wide') in operations
    assert not any(
        operation == 'prepare_direct' for operation, _, _ in operations
    )


def test_early_scored_refusal_precedes_later_custom_column():
    context = pd.DataFrame(
        {
            'id': [1, 2],
            'early': [0.0, 1.0],
            'later': [2.0, 3.0],
            'target': [0, 1],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'data': context}, primary_keys={'data': 'id'}
        )
    schema['data']['early'].dtype = 'string'
    schema['data']['early'].stype = 'text'

    class Scoring(pd.DataFrame):
        def __getitem__(self, key):
            if key == 'later':
                raise AssertionError('later column must not be observed')
            return super().__getitem__(key)

    scoring = Scoring({'id': [3], 'early': [4.0], 'later': [5.0]})
    with pytest.raises(StructuredDataError, match='early'):
        Request(
            context,
            scoring,
            target='target',
            task='regression',
            schema=schema,
            entity_table='data',
        )
