# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public direct calls, compared on both routes without observing row indexes."""

from __future__ import annotations

import builtins
import dataclasses
import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from sdm_client import KumoTabular, Request, Schema, SDMClient, StructuredDataError

DIRECT = ','.join(
    (
        'request.tabular/classification/frames',
        'request.tabular/regression/frames',
        'client.predict/tabular/frames',
    )
)


class Endpoint:
    url = 'http://localhost'

    def __init__(self, reply: dict[str, Any] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.reply = reply

    def predict_body(self, body: bytes) -> bytes:
        payload = json.loads(body)
        self.calls.append(payload)
        return json.dumps(self.reply or answer()).encode()

    def predict(self, payload: dict[str, Any]) -> dict[str, Any]:
        return json.loads(self.predict_body(json.dumps(payload).encode()))

    def close(self) -> None:
        pass


def answer() -> dict[str, Any]:
    return {
        'model': 'kumo-tabular',
        'id': 'answer',
        'predictions': [
            {'row_index': 1, 'id': '[91]', 'prediction': 1},
            {'row_index': 0, 'id': '[90]', 'prediction': 0},
        ],
        'metadata': {'source': 'scripted'},
    }


def client(endpoint: Endpoint) -> SDMClient:
    made = SDMClient.__new__(SDMClient)
    made._start(endpoint)
    return made


def frames(dtype: str = 'int64') -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.DataFrame(
            {
                'row_id': np.array([0, 1, 2, 3], dtype=dtype),
                'feature': np.array([10, 11, 12, 13], dtype=dtype),
                'target': np.array([0, 1, 0, 1], dtype=dtype),
            }
        ),
        pd.DataFrame(
            {
                'row_id': np.array([90, 91], dtype=dtype),
                'feature': np.array([20, 21], dtype=dtype),
            }
        ),
    )


@pytest.fixture(params=['', DIRECT], ids=['old', 'core'])
def route(request: Any, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv('SDM_CORE_ROUTES', request.param)
    return str(request.param)


@pytest.mark.parametrize('dtype', ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'])
def test_direct_columns_and_typed_results(route: str, dtype: str) -> None:
    context, scoring = frames(dtype)
    request = Request(context, scoring, target='target', task='classification')
    assert request.context.instance_table is context
    assert request.predict.instance_table is scoring
    assert request.task.target.dtype == 'int64'
    endpoint = Endpoint()
    result = client(endpoint).predict(request, model=KumoTabular())
    payload = endpoint.calls[0]
    assert payload['context']['instance_table']['rows'] == context.values.tolist()
    assert payload['task']['target']['dtype'] == payload['schema']['instance_table']['columns']['target']['dtype']
    assert [(item.row_index, item.id, item.prediction) for item in result.items] == [(1, 91, 1), (0, 90, 0)]
    assert list(result.to_pandas()['row_index']) == [0, 1]


def test_boolean_features_and_target(route: str) -> None:
    context = pd.DataFrame({'feature': [True, False, True], 'target': [False, True, False]})
    scoring = pd.DataFrame({'feature': [False, True]})
    request = Request(context, scoring, target='target', task='classification')
    endpoint = Endpoint()
    client(endpoint).predict(request, model='kumo-tabular')
    columns = endpoint.calls[0]['schema']['instance_table']['columns']
    assert columns['feature'] == {'dtype': 'bool', 'stype': 'categorical'}
    assert columns['target'] == {'dtype': 'bool', 'stype': 'categorical'}


@pytest.mark.parametrize('kind', ['duplicates', 'nonmonotonic', 'multi'])
def test_direct_indexes_do_not_affect_model_rows(route: str, kind: str) -> None:
    context, scoring = frames()
    context.index = {
        'duplicates': pd.Index(['same'] * 4),
        'nonmonotonic': pd.Index(['z', 'a', 'x', 'b']),
        'multi': pd.MultiIndex.from_tuples([('b', 2), ('a', 1), ('a', 1), ('c', 0)]),
    }[kind]
    scoring.index = pd.Index([object(), object()])
    endpoint = Endpoint()
    client(endpoint).predict(Request(context, scoring, target='target', task='classification'), model='kumo-tabular')
    assert endpoint.calls[0]['predict']['instance_table']['columns'] == ['row_id', 'feature']
    assert endpoint.calls[0]['context']['instance_table']['rows'] == context.values.tolist()


def test_uint64_values_travel_as_exact_text(route: str) -> None:
    large = 2**63 + 17
    context, scoring = frames()
    context['feature'] = np.array([large + i for i in range(4)], dtype='uint64')
    scoring['feature'] = np.array([large + 9, large + 10], dtype='uint64')
    endpoint = Endpoint()
    request = Request(context, scoring, target='target', task='regression')
    client(endpoint).predict(request, model='kumo-tabular')
    payload = endpoint.calls[0]
    assert payload['schema']['instance_table']['columns']['feature']['dtype'] == 'string'
    assert payload['predict']['instance_table']['rows'][0][1] == str(large + 9)


def test_supplied_schema_and_live_frames_keep_their_owners(route: str) -> None:
    context, scoring = frames()
    schema = Schema.from_data({'instance': context}, relationships=())
    request = Request(context, scoring, target='target', task='regression', schema=schema)
    assert request.schema is schema
    scoring.loc[0, 'feature'] = 700
    request.task = dataclasses.replace(request.task, kind='classification')
    endpoint = Endpoint()
    client(endpoint).predict(request, model='kumo-tabular')
    assert endpoint.calls[0]['predict']['instance_table']['rows'][0][1] == 700
    assert endpoint.calls[0]['task']['kind'] == 'classification'


def test_zero_column_scoring_retains_nonzero_row_count(route: str) -> None:
    request = Request(pd.DataFrame({'target': [0, 1, 0]}), pd.DataFrame(index=['x', 'x']), target='target', task='classification')
    endpoint = Endpoint()
    client(endpoint).predict(request, model='kumo-tabular')
    assert endpoint.calls[0]['predict']['instance_table'] == {'format': 'arrays', 'columns': [], 'rows': [[], []]}


@pytest.mark.parametrize('model, outputs, task', [('absent', None, 'classification'), ('kumo-tabular', [], 'classification'), ('kumo-tabular', None, 'binary_classification')])
def test_model_preflight_precedes_empty_no_send(route: str, model: str, outputs: Any, task: str) -> None:
    context, scoring = frames()
    request = Request(context, scoring.iloc[:0], target='target', task=task)
    endpoint = Endpoint()
    with pytest.raises(StructuredDataError):
        client(endpoint).predict(request, model=model, outputs=outputs)
    assert endpoint.calls == []


def test_empty_call_skips_late_inference_conversion(route: str) -> None:
    context, scoring = frames()
    request = Request(context, scoring.iloc[:0], target='target', task='classification', inference={'opaque': object()})
    endpoint = Endpoint()
    result = client(endpoint).predict(request, model='kumo-tabular')
    assert result.items == []
    assert endpoint.calls == []


def test_constructor_guard_order(route: str) -> None:
    context, scoring = frames()
    with pytest.raises(StructuredDataError, match='task must'):
        Request(context.iloc[:0], scoring.assign(target=0), target='absent', task='absent')
    with pytest.raises(StructuredDataError, match='no rows'):
        Request(context.iloc[:0], scoring.assign(target=0), target='absent', task='regression')
    with pytest.raises(StructuredDataError, match='no target column'):
        Request(context, scoring.assign(target=0), target='absent', task='regression')


@pytest.mark.parametrize('change, fragment', [('missing', 'missing a row_index'), ('bounds', 'names row_index'), ('duplicate', 'more than once'), ('unanswered', 'unanswered'), ('wrong_key', 'pairs row_index')])
def test_response_attribution_failures(route: str, change: str, fragment: str) -> None:
    reply = answer()
    rows = reply['predictions']
    if change == 'missing':
        rows[0].pop('row_index')
    elif change == 'bounds':
        rows[0]['row_index'] = 4
    elif change == 'duplicate':
        rows[1] = dict(rows[0])
    elif change == 'unanswered':
        rows.pop()
    else:
        rows[0]['id'] = '[90]'
    context, scoring = frames()
    with pytest.raises(StructuredDataError, match=fragment):
        client(Endpoint(reply)).predict(Request(context, scoring, target='target', task='classification'), model='kumo-tabular')


def test_native_literal_and_opaque_ids(route: str) -> None:
    reply = answer()
    reply['predictions'][0]['id'] = "[+91]"
    reply['predictions'][1]['id'] = 'service-generated'
    context, scoring = frames()
    result = client(Endpoint(reply)).predict(Request(context, scoring, target='target', task='classification'), model='kumo-tabular')
    assert [item.id for item in result.items] == [91, 90]


def test_direct_core_does_not_import_legacy_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('SDM_CORE_ROUTES', DIRECT)
    original = builtins.__import__

    def blocked(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith(('sdm_relational_engine', 'predictive_query')):
            raise AssertionError(f'legacy inference reached: {name}')
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', blocked)
    context, scoring = frames()
    request = Request(context, scoring, target='target', task='classification')
    client(Endpoint()).predict(request, model='kumo-tabular')
