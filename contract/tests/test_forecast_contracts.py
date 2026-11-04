# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forecast limits, request identity and lag types through the current client."""

import importlib.util
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import yaml
from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
    StructuredDataError,
    decompose,
)
from sdm_connectors import DuckDBSource

ROOT = Path(__file__).resolve().parents[2]


def _endpoint():
    name = 'forecast_contract_endpoint'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'tests/support/scripted_endpoint.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].serve()


@pytest.fixture
def schema():
    statements = yaml.safe_load((ROOT / 'contract/bases.yaml').read_text())[
        'bases'
    ]['shop']
    with duckdb.connect() as connection:
        for statement in statements:
            connection.execute(statement)
        yield Schema.from_source(DuckDBSource(connection))


def _forecast(schema, steps):
    return PredictiveQuery(
        f'PREDICT SUM(orders.price, 0, 7, days) FORECAST {steps} '
        'TIMEFRAMES FOR EACH users.user_id',
        schema,
    ).materialize(indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01'))


def _predict(request):
    with _endpoint() as (url, received), SDMClient(url) as client:
        # This fixture models stateless prediction replies, not session creation.
        client._transport.serves_sessions = False
        client.predict(request, model=KumoRelational())
    return [json.loads(body) for _, _, body in received.requests]


def test_forecast_calls_have_distinct_request_ids(schema):
    bodies = _predict(_forecast(schema, 3))
    ids = [body['metadata']['request_id'] for body in bodies]
    assert len(ids) == 3
    assert all(isinstance(value, str) and value for value in ids)
    assert len(set(ids)) == 3


def test_caller_can_lower_and_raise_forecast_limit(schema, monkeypatch):
    monkeypatch.setattr(decompose, 'MOST_PARTS', 2)
    with pytest.raises(StructuredDataError) as refused:
        _forecast(schema, 3)
    assert refused.value.code == 'INVALID_REQUEST'
    monkeypatch.setattr(decompose, 'MOST_PARTS', 3)
    assert len(_predict(_forecast(schema, 3))) == 3


@pytest.mark.parametrize('steps', [201, 1_000_000_000])
def test_excessive_forecast_is_refused_before_materialization(schema, steps):
    with pytest.raises(StructuredDataError) as refused:
        _forecast(schema, steps)
    assert refused.value.code == 'INVALID_REQUEST'


def test_lag_types_are_declared_on_each_call(schema):
    request = PredictiveQuery(
        'PREDICT AVG(orders.price, 0, 28, days) > 20 FOR EACH users.user_id',
        schema,
    ).materialize(
        indices=[0, 1],
        anchor_time=pd.Timestamp('2023-03-01'),
        lag_timesteps=2,
        rows_per_call=1,
    )
    types = [
        {
            name: column['dtype']
            for name, column in body['schema']['instance_table'][
                'columns'
            ].items()
            if name.startswith('__kumo_arl')
        }
        for body in _predict(request)
    ]
    assert types == [{'__kumo_arl0__': 'bool', '__kumo_arl1__': 'bool'}] * 2
