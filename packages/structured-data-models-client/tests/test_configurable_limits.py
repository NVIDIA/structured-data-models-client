# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The size limits a client holds requests to are settings.

Each defaults to the serving layer's own, and a deployment configured to take
more is matched by raising it: with an argument, else its environment
variable. A call past the most bytes one takes is split rather than refused,
and only a context too large on its own is refused, naming the setting.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from test_limits_and_absent_outputs import gathered, query, table
from test_tabular_direct_core import Endpoint, client

from sdm_client import KumoTabular, Limits, Request, StructuredDataError
from sdm_client.limits import settings

DEFAULTS = {
    'max_relational_context_rows': 10_000,
    'max_relational_predict_rows': 10_000,
    'max_temporal_link_predict_rows': 200,
    'max_tabular_context_rows': 10_000,
    'max_request_bytes': 30 * 1024 * 1024,
}


@pytest.fixture(autouse=True)
def _no_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for setting in settings():
        monkeypatch.delenv(setting.env, raising=False)


def test_the_defaults_are_the_serving_layers() -> None:
    assert {s.name: s.default for s in settings()} == DEFAULTS
    assert {s.name: s.env for s in settings()} == {
        name: f'SDM_CLIENT_{name.upper()}' for name in DEFAULTS
    }
    assert Limits().resolved() == DEFAULTS


def test_an_argument_comes_before_the_environment_before_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv('SDM_CLIENT_MAX_TABULAR_CONTEXT_ROWS', '200000')
    monkeypatch.setenv('SDM_CLIENT_MAX_REQUEST_BYTES', '64_000_000')
    from_env = Limits().resolved()
    assert from_env['max_tabular_context_rows'] == 200_000
    assert from_env['max_request_bytes'] == 64_000_000
    assert from_env['max_relational_context_rows'] == 10_000
    given = Limits(max_tabular_context_rows=50_000).resolved()
    assert given['max_tabular_context_rows'] == 50_000


@pytest.mark.parametrize('text', ['0', '-1', '2.5', 'lots', '1e6'])
def test_an_environment_variable_that_is_not_a_whole_number_is_refused(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    monkeypatch.setenv('SDM_CLIENT_MAX_RELATIONAL_PREDICT_ROWS', text)
    with pytest.raises(StructuredDataError) as refused:
        Limits().resolved()
    assert refused.value.code == 'INVALID_CONFIGURATION'
    assert 'SDM_CLIENT_MAX_RELATIONAL_PREDICT_ROWS' in str(refused.value)


@pytest.mark.parametrize('value', [0, -5, True, 2.5, '10'])
def test_a_setting_given_that_is_not_a_whole_number_is_refused(value) -> None:
    with pytest.raises(StructuredDataError) as refused:
        Limits(max_request_bytes=value)
    assert refused.value.code == 'INVALID_CONFIGURATION'


def _direct(context_rows: int) -> Request:
    context = pd.DataFrame(
        {
            'row_id': np.arange(context_rows),
            'feature': np.arange(context_rows) % 7,
            'target': np.arange(context_rows) % 2,
        }
    )
    scoring = pd.DataFrame({'row_id': [90, 91], 'feature': [1, 2]})
    return Request(context, scoring, target='target', task='classification')


def test_a_tabular_context_of_200_000_rows_is_sent_when_the_limit_is_raised() -> (
    None
):
    request = _direct(200_000)
    endpoint = Endpoint()
    with pytest.raises(StructuredDataError) as refused:
        client(endpoint).predict(request, model=KumoTabular())
    assert 'SDM_CLIENT_MAX_TABULAR_CONTEXT_ROWS' in str(refused.value)
    assert endpoint.calls == []

    raised = client(endpoint)
    raised._limits = Limits(max_tabular_context_rows=200_000)
    raised.predict(request, model=KumoTabular())
    (sent,) = endpoint.calls
    assert len(sent['context']['instance_table']['rows']) == 200_000


def test_materialize_samples_to_a_raised_tabular_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = table(12_000)
    with pytest.raises(StructuredDataError):
        query(schema).materialize(indices=[0, 1], num_context_samples=11_000)
    request = query(schema).materialize(
        indices=[0, 1],
        num_context_samples=11_000,
        limits=Limits(max_tabular_context_rows=20_000),
    )
    assert request.context_rows == 11_000
    monkeypatch.setenv('SDM_CLIENT_MAX_TABULAR_CONTEXT_ROWS', '20000')
    assert (
        query(schema)
        .materialize(indices=[0, 1], num_context_samples=11_000)
        .context_rows
        == 11_000
    )


def test_a_raised_relational_predict_limit_and_a_lowered_one_set_the_calls() -> (
    None
):
    from sdm_client import KumoRelational, PredictiveQuery, Schema

    frame = pd.DataFrame(
        {'uid': range(40), 'label': [*range(30), *[None] * 10]}
    )
    schema = Schema.from_data({'users': frame}, primary_keys={'users': 'uid'})
    schema['users']['label'].stype = 'numerical'
    asked = PredictiveQuery('PREDICT users.label FOR EACH users.uid', schema)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lowered = asked.materialize(
            indices=list(range(30, 40)),
            model=KumoRelational(),
            limits=Limits(max_relational_predict_rows=4),
        )
    assert [
        len(p['predict']['instance_table']['rows']) for p in lowered.payloads
    ] == [
        4,
        4,
        2,
    ]


def _relational(predict_rows: int):
    return gathered(context_rows=30, run_mode='fast', predict_rows=predict_rows)


def _sent(request, limits: Limits) -> list[dict]:
    class RelationalEndpoint(Endpoint):
        def predict_body(self, body):
            payload = json.loads(body)
            self.calls.append(payload)
            self.sizes.append(len(body))
            table = payload['predict']['instance_table']
            key = table['columns'].index('instance_id')
            return json.dumps(
                {
                    'id': 'answer',
                    'model': 'kumo-relational',
                    'metadata': {},
                    'predictions': [
                        {'id': str(row[key]), 'row_index': i, 'prediction': 1.0}
                        for i, row in enumerate(table['rows'])
                    ],
                }
            ).encode()

    endpoint = RelationalEndpoint()
    endpoint.sizes = []
    sending = client(endpoint)
    sending._limits = limits
    result = sending.predict(request, model='kumo-relational')
    assert list(result.to_pandas()['row_index']) == list(range(8))
    return endpoint


def test_a_call_past_the_byte_limit_is_split_and_each_part_fits() -> None:
    request = _relational(8)
    whole = _sent(request, Limits())
    (size,) = whole.sizes
    split = _sent(request, Limits(max_request_bytes=size - 1))
    assert len(split.calls) > 1
    assert all(sent <= size - 1 for sent in split.sizes)
    assert (
        sum(len(c['predict']['instance_table']['rows']) for c in split.calls)
        == 8
    )
    # Every call carries the whole context.
    contexts = {json.dumps(c['context'], sort_keys=True) for c in split.calls}
    assert len(contexts) == 1
    # At the limit exactly, one call.
    assert len(_sent(request, Limits(max_request_bytes=size)).calls) == 1


def test_a_context_too_large_on_its_own_is_refused_naming_the_setting() -> None:
    request = _relational(8)
    with pytest.raises(StructuredDataError) as refused:
        _sent(request, Limits(max_request_bytes=1_000))
    said = str(refused.value)
    assert 'context alone' in said
    assert 'SDM_CLIENT_MAX_REQUEST_BYTES' in said
    assert 'max_request_bytes' in said


def test_more_than_six_hops_and_fifteen_tables_are_taken() -> None:
    from test_query_engine_boundary import gathered_from, star

    from sdm_client import KumoRelational

    assert len(gathered_from(star(satellites=20, joining=20))) == 21
    request = _relational(1)
    assert request.payloads
    frame = pd.DataFrame({'uid': range(12), 'label': [*range(10), None, None]})
    from sdm_client import PredictiveQuery, Schema

    schema = Schema.from_data({'users': frame}, primary_keys={'users': 'uid'})
    schema['users']['label'].stype = 'numerical'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for options in ({'num_hops': 8}, {'num_neighbors': [4] * 9}):
            built = PredictiveQuery(
                'PREDICT users.label FOR EACH users.uid', schema
            ).materialize(indices=[10, 11], model=KumoRelational(), **options)
            assert built.payloads
