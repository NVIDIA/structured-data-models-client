# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public client gradient handling over HTTP, including fit and nested merges."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd
import pytest
from sdm_api.generated.tfm_api import PredictionItem

from sdm_client import KumoTabular, PredictionResult, Request, SDMClient


@pytest.fixture
def endpoint(request):
    mode = getattr(request, 'param', 'ok')
    calls = []
    fitted = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(
                self.rfile.read(int(self.headers['Content-Length']))
            )
            calls.append(payload)
            if self.path == '/v1/sessions':
                fitted.update(payload)
                answer = {'session_id': 'gradient-session'}
            else:
                if (
                    mode == 'partial'
                    and sum('predict' in call for call in calls) > 1
                ):
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b'{}')
                    return
                table = payload['predict']['instance_table']
                columns = [
                    column for column in table['columns'] if column == 'age'
                ]
                predictions = []
                for row in range(len(table['rows'])):
                    predictions.append(
                        {
                            'row_index': row,
                            'prediction': 1,
                            'probabilities': {'0': 0.2, '1': 0.8},
                            'explanation': {
                                'format': 'sdm_gradient_v1',
                                'details': {
                                    'input_space': 'preprocessed_numerical',
                                    'outputs': [
                                        {'kind': 'probability', 'class': '0'},
                                        {'kind': 'probability', 'class': '1'},
                                    ],
                                    'instance_table': {
                                        'row_indices': [row],
                                        'columns': columns,
                                        'gradients': [
                                            [[0.25] * len(columns)],
                                            [[0.75] * len(columns)],
                                        ],
                                    },
                                    'related_tables': {},
                                },
                            },
                        }
                    )
                model = payload.get(
                    'model', fitted.get('model', 'kumo-tabular')
                )
                for at, item in enumerate(predictions):
                    details = item['explanation']['details']
                    if mode == 'malformed':
                        details['instance_table']['row_indices'] = [-1]
                    if model == 'kumo-relational':
                        item['id'] = str(
                            table['rows'][at][
                                table['columns'].index('instance_id')
                            ]
                        )
                        item.pop('probabilities')
                        details['outputs'] = [
                            {'kind': 'prediction', 'statistic': 'mean'}
                        ]
                        details['instance_table']['columns'] = []
                        details['instance_table']['gradients'] = [[[]]]
                answer = {
                    'id': 'gradient-answer',
                    'metadata': {},
                    'model': model,
                    'predictions': predictions,
                }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(answer).encode())

        def do_DELETE(self):
            self.send_response(204)
            self.end_headers()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def request():
    return Request(
        pd.DataFrame({'age': range(20), 'churn': [i % 2 for i in range(20)]}),
        pd.DataFrame({'age': [31, 41]}),
        target='churn',
        task='classification',
    )


def test_public_multiple_rows_class_labels_and_nested_merge(endpoint):
    url, calls = endpoint
    with SDMClient(url=url) as client:
        result = client.predict(
            request(),
            model=KumoTabular(),
            outputs=['prediction', 'probabilities', 'explanation'],
        )
    assert len(result.explanations) == 2
    frame = result.explanations.to_pandas()
    assert set(frame.output) == {'probability_0', 'probability_1'}
    assert set(frame.output) <= set(result.to_pandas().columns)
    merged = PredictionResult.merge(
        [PredictionResult.merge([result, result]), result]
    )
    assert len(merged.explanations) == 6
    assert set(merged.explanations) == set(range(6))
    assert calls[0]['output']['fields'] == [
        'prediction',
        'probabilities',
        'explanation',
    ]
    assert result.explanation is not None


def test_ordinary_merge_does_not_serialize_user_ids():
    result = PredictionResult(
        model='kumo-tabular',
        request_id='a',
        items=[PredictionItem(row_index=0, id=np.int64(4), prediction=1)],
    )
    assert PredictionResult.merge([result]) is result
    assert len(PredictionResult.merge([result, result]).items) == 2


@pytest.mark.parametrize('model_name', ['tabular', 'relational'])
@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_pretrained_local_model_gradients(model_name, task):
    """Opt-in GPU E2E with actual pretrained weights; no model-server fixture."""
    import os

    if os.environ.get('SDM_RUN_LOCAL_MODEL') != '1':
        pytest.skip(
            'set SDM_RUN_LOCAL_MODEL=1 with the optional model runtime and GPU'
        )
    from sample_tables import graph

    from sdm_client import KumoRelational, PredictiveQuery

    if model_name == 'tabular':
        context = pd.DataFrame(
            {
                'x': [float(i) for i in range(16)],
                'target': [
                    float(i * 2) if task == 'regression' else str(i % 2)
                    for i in range(16)
                ],
            }
        )
        req = Request(
            context, pd.DataFrame({'x': [2.5, 5.5]}), target='target', task=task
        )
        model = KumoTabular()
    else:
        target = 'age' if task == 'regression' else 'tier'
        req = PredictiveQuery(
            f'EXPLAIN PREDICT users.{target} FOR EACH users.user_id', graph()
        ).materialize(
            indices=[0, 1],
            num_context_samples=12,
            num_neighbors=[2, 2],
            random_seed=7,
        )
        model = KumoRelational()
    with SDMClient.in_process(device='cuda') as client:
        result = client.predict(
            req,
            model=model,
            outputs=['prediction', 'explanation']
            + (['quantiles'] if task == 'regression' else ['probabilities']),
        )
        assert len(result) == len(result.explanations) == 2
        frame = result.explanations.to_pandas()
        assert not frame.empty
        assert set(frame.output) <= set(result.to_pandas().columns)
        assert all(
            item.explanation['format'] == 'sdm_gradient_v1'
            for item in result.items
        )
        if model_name == 'relational':
            assert any('__' in name for name in frame.column)


@pytest.mark.parametrize('endpoint', ['malformed'], indirect=True)
def test_malformed_gradient_is_not_silently_discarded(endpoint):
    from sdm_client.errors import StructuredDataError

    url, _ = endpoint
    with (
        SDMClient(url=url) as client,
        pytest.raises(StructuredDataError) as caught,
    ):
        client.predict(request(), model=KumoTabular(), outputs=['explanation'])
    assert caught.value.code == 'INVALID_RESPONSE'
    assert caught.value.details['field'] == 'prediction.explanation'


def test_tabular_session_explain_is_an_output_not_a_new_question(endpoint):
    from sdm_client import PredictiveQuery, Schema

    url, calls = endpoint
    schema = Schema.from_data(
        {
            'users': pd.DataFrame(
                {
                    'id': range(20),
                    'age': range(20),
                    'churn': [i % 2 for i in range(20)],
                }
            )
        },
        primary_keys={'users': 'id'},
    )
    schema['users']['churn'].stype = 'categorical'
    query = 'PREDICT users.churn FOR EACH users.id'

    def materialize(text):
        return PredictiveQuery(text, schema).materialize(
            model=KumoTabular(), indices=[18, 19], num_context_samples=12
        )

    plain = materialize(query)
    explained = materialize('EXPLAIN\n' + query)
    with SDMClient(url=url) as client:
        assert (
            len(client.predict(explained, model=KumoTabular()).explanations)
            == 2
        )
        assert 'explanation' in calls[-1]['output']['fields']
        with client.fit(plain, model=KumoTabular()) as fitted:
            result = fitted.predict(explained)
    assert len(result.explanations) == 2
    assert 'explanation' in calls[-1]['output']['fields']
    assert 'context' not in calls[-1]


@pytest.mark.parametrize('endpoint', ['partial'], indirect=True)
def test_partial_relational_batches_keep_completed_explanations(endpoint):
    from test_automatic_batch_sessions import request as relational_request

    from sdm_client import KumoRelational
    from sdm_client.errors import StructuredDataError

    url, _ = endpoint
    with (
        SDMClient(url=url) as client,
        pytest.raises(StructuredDataError) as caught,
    ):
        client.predict(
            relational_request(rows=1),
            model=KumoRelational(),
            outputs=['prediction', 'explanation'],
        )
    partial = caught.value.details['answered']
    assert len(partial.explanations) == 1


def test_explain_forecast_keeps_step_identity(endpoint):
    from test_frame_materialize_core import link_schema

    from sdm_client import KumoRelational, PredictiveQuery

    url, calls = endpoint
    req = PredictiveQuery(
        'EXPLAIN PREDICT COUNT(orders.*, 0, 30, days) FORECAST 2 TIMEFRAMES FOR EACH users.user_id',
        link_schema(),
    ).materialize(
        indices=[1, 2],
        num_context_samples=12,
        random_seed=7,
        model=KumoRelational(),
    )
    with SDMClient(url=url) as client:
        result = client.predict(req, model=KumoRelational())
    assert len(result.explanations) == 4
    assert {key[1] for key in result.explanations} == {1, 2}
    assert all(
        'explanation' in call['output']['fields']
        for call in calls
        if 'predict' in call
    )


def test_native_reply_nonfinite_gradient_uses_shared_refusal():
    from test_tabular_direct_core import Endpoint, client

    from sdm_client.errors import StructuredDataError

    reply = {
        'model': 'kumo-tabular',
        'id': 'bad-gradient',
        'metadata': {},
        'predictions': [
            {
                'row_index': row,
                'prediction': 1,
                'explanation': {
                    'format': 'sdm_gradient_v1',
                    'details': {
                        'input_space': 'preprocessed_numerical',
                        'outputs': [{'kind': 'probability', 'class': '1'}],
                        'instance_table': {
                            'row_indices': [row],
                            'columns': ['age'],
                            'gradients': [[[float('nan')]]],
                        },
                        'related_tables': {},
                    },
                },
            }
            for row in range(2)
        ],
    }
    with (
        client(Endpoint(reply)) as model_client,
        pytest.raises(StructuredDataError) as caught,
    ):
        model_client.predict(
            request(), model=KumoTabular(), outputs=['explanation']
        )
    assert caught.value.code == 'INVALID_RESPONSE'
    assert caught.value.details['field'] == 'prediction.explanation'


def test_explain_candidate_ranking_is_refused_before_model_call(endpoint):
    from test_frame_materialize_core import link_schema

    from sdm_client import (
        KumoRelational,
        PredictiveQuery,
        UnsupportedRequestError,
    )

    url, calls = endpoint
    req = PredictiveQuery(
        'EXPLAIN PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) RANK TOP 2 FOR EACH users.user_id',
        link_schema(),
    ).materialize(
        indices=[1, 2],
        candidates=[1, 2],
        num_context_samples=12,
        random_seed=7,
        model=KumoRelational(),
    )
    with SDMClient(url=url) as client, pytest.raises(UnsupportedRequestError):
        client.predict(req, model=KumoRelational())
    assert not calls
