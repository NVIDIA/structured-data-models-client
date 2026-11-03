# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public requests through actual HTTP: Rust lifecycle, host transport policy."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd
import pytest

from sdm_client import KumoRelational, RelationalTask, Schema, SDMClient
from sdm_client.errors import StructuredDataError

FIXTURE = json.loads(
    (
        Path(__file__).resolve().parents[3]
        / 'tests/fixtures/batch-sessions/scenarios.json'
    ).read_text()
)


def request(*, rows=1):
    def frame(data):
        return pd.DataFrame(data['rows'], columns=data['columns'])

    schema = Schema.from_data(
        {name: frame(data) for name, data in FIXTURE['tables'].items()},
        primary_keys={'users': 'id'},
    )
    schema['users']['value'].stype = 'numerical'
    return schema.materialize_examples(
        RelationalTask('regression', 'users'),
        frame(FIXTURE['context']),
        frame(FIXTURE['predict']),
        rows_per_call=rows,
    )


class Endpoint:
    def __init__(self, mode='ok'):
        self.mode, self.calls, self.created, self.scored = mode, [], 0, 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                payload = json.loads(
                    self.rfile.read(int(self.headers['Content-Length']))
                )
                outer.calls.append((self.command, self.path, payload))
                status, answer = 200, {}
                if self.path == '/v1/sessions':
                    outer.created += 1
                    if outer.mode in ('unsupported', 'uncertain'):
                        status = 405 if outer.mode == 'unsupported' else 503
                    else:
                        answer = {'session_id': f's{outer.created}'}
                        if outer.mode == 'alias':
                            answer = {
                                'session_id': None,
                                'id': f's{outer.created}',
                            }
                else:
                    outer.scored += 1
                    if outer.mode == 'transport' and outer.scored == 2:
                        self.close_connection = True
                        return
                    if (outer.mode == 'expired' and outer.scored == 2) or (
                        outer.mode == 'expired_twice' and outer.scored >= 2
                    ):
                        status = 404
                    elif outer.mode == 'malformed' and outer.scored == 2:
                        answer = {'predictions': 'invalid'}
                    elif outer.mode == 'fail' and outer.scored == 2:
                        status = 500
                    elif outer.mode == 'retry' and outer.scored == 1:
                        status = 408
                    else:
                        table = payload['predict']['instance_table']
                        index = table['columns'].index('instance_id')
                        answer = {
                            'model': 'kumo-relational',
                            'id': 'answer',
                            'metadata': {},
                            'predictions': [
                                {
                                    'id': str(row[index]),
                                    'row_index': i,
                                    'prediction': 5.0,
                                }
                                for i, row in enumerate(table['rows'])
                            ],
                        }
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(answer).encode())

            def do_DELETE(self):
                outer.calls.append((self.command, self.path, None))
                self.send_response(
                    400 if outer.mode in ('malformed', 'transport') else 204
                )
                self.end_headers()

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.thread.start()

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_port}'

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


@pytest.mark.parametrize(
    'mode,creates,scores,deletes',
    [
        tuple(case[k] for k in ('mode', 'creates', 'scores', 'deletes'))
        for case in json.loads(
            (
                Path(__file__).resolve().parents[3]
                / 'tests/fixtures/batch-sessions/scenarios.json'
            ).read_text()
        )['cases']
    ],
)
def test_public_batch_lifecycle(mode, creates, scores, deletes):
    endpoint = Endpoint(mode)
    try:
        with SDMClient(url=endpoint.url, max_retries=1) as client:
            result = client.predict(request(), model=KumoRelational())
            assert len(result.to_pandas()) == 3
        assert endpoint.created == creates
        assert endpoint.scored == scores
        assert (
            sum(method == 'DELETE' for method, _, _ in endpoint.calls)
            == deletes
        )
        if mode != 'unsupported':
            assert all(
                'context' not in body
                for method, path, body in endpoint.calls
                if method == 'POST' and path.endswith('/predictions')
            )
    finally:
        endpoint.close()


def test_failed_batch_preserves_paid_answers_and_cleanup():
    endpoint = Endpoint('fail')
    try:
        with SDMClient(url=endpoint.url, max_retries=1) as client:
            with pytest.raises(StructuredDataError) as caught:
                client.predict(request(), model=KumoRelational())
        assert endpoint.scored == 2  # HTTP500 is not retried.
        assert endpoint.calls[-1][:2] == ('DELETE', '/v1/sessions/s1')
        assert getattr(caught.value, 'status_code', None) == 500
        assert caught.value.details['answered'] is not None
    finally:
        endpoint.close()


def test_uncertain_creation_is_not_retried_or_fallen_back():
    endpoint = Endpoint('uncertain')
    try:
        with SDMClient(url=endpoint.url, max_retries=2) as client:
            with pytest.raises(StructuredDataError):
                client.predict(request(), model=KumoRelational())
        assert endpoint.created == 1
        assert endpoint.scored == 0
    finally:
        endpoint.close()


@pytest.mark.parametrize('rows', [0, -1])
def test_invalid_batch_size(rows):
    with pytest.raises(StructuredDataError):
        request(rows=rows)


def test_malformed_answer_keeps_primary_error_when_cleanup_fails():
    endpoint = Endpoint('malformed')
    try:
        with SDMClient(url=endpoint.url, max_retries=0) as client:
            with pytest.raises(StructuredDataError) as caught:
                client.predict(request(), model=KumoRelational())
        assert caught.value.code == 'INVALID_RESPONSE'
        assert endpoint.calls[-1][0] == 'DELETE'
        assert caught.value.details['answered'] is not None
    finally:
        endpoint.close()


def test_transport_failure_keeps_primary_error_when_cleanup_fails():
    endpoint = Endpoint('transport')
    try:
        with SDMClient(url=endpoint.url, max_retries=0) as client:
            with pytest.raises(StructuredDataError) as caught:
                client.predict(request(), model=KumoRelational())
        assert caught.value.code == 'TRANSPORT_ERROR'
        assert caught.value.details['answered'] is not None
        assert endpoint.calls[-1][0] == 'DELETE'
    finally:
        endpoint.close()


def test_distinct_materialized_contexts_remain_stateless():
    from sdm_client import PredictiveQuery

    schema = Schema.from_data(
        {
            'users': pd.DataFrame(
                {'id': range(12), 'value': [float(i) for i in range(12)]}
            )
        },
        primary_keys={'users': 'id'},
    )
    schema['users']['value'].stype = 'numerical'
    gathered = PredictiveQuery(
        'PREDICT users.value FOR EACH users.id', schema
    ).materialize(
        model=KumoRelational(),
        indices=[6, 7, 8],
        rows_per_call=1,
        num_context_samples=12,
        random_seed=7,
    )
    assert gathered.payloads[0]['context'] != gathered.payloads[1]['context']
    endpoint = Endpoint()
    try:
        with SDMClient(url=endpoint.url) as client:
            assert (
                len(
                    client.predict(gathered, model=KumoRelational()).to_pandas()
                )
                == 3
            )
        assert endpoint.created == 0
        assert endpoint.scored == 3
    finally:
        endpoint.close()


def test_session_and_stateless_scoring_preserve_inference_and_outputs():
    gathered = request()
    sent = []
    for mode in ('ok', 'unsupported'):
        endpoint = Endpoint(mode)
        try:
            with SDMClient(url=endpoint.url) as client:
                client.predict(gathered, model=KumoRelational())
            sent.append(
                [
                    {
                        key: body.get(key)
                        for key in ('predict', 'output', 'inference')
                    }
                    for method, path, body in endpoint.calls
                    if method == 'POST' and path.endswith('/predictions')
                ]
            )
        finally:
            endpoint.close()
    assert sent[0] == sent[1]


def test_expired_session_is_recreated_only_once():
    endpoint = Endpoint('expired_twice')
    try:
        with SDMClient(url=endpoint.url, max_retries=0) as client:
            with pytest.raises(StructuredDataError) as caught:
                client.predict(request(), model=KumoRelational())
        assert endpoint.created == 2
        assert endpoint.scored == 3
        assert sum(method == 'DELETE' for method, _, _ in endpoint.calls) == 2
        assert caught.value.details['answered'] is not None
    finally:
        endpoint.close()


@pytest.mark.live_nim
def test_live_automatic_session_uploads_context_once(monkeypatch):
    import os

    url = os.getenv('SDM_RELATIONAL_URL')
    if not url:
        pytest.skip('SDM_RELATIONAL_URL is required')
    calls = {'create_session': 0, 'session_predict': 0, 'delete_session': 0}
    with SDMClient(url=url, timeout=180) as client:
        for name in calls:
            original = getattr(client._transport, name)

            def tracked(*args, _name=name, _original=original, **kwargs):
                calls[_name] += 1
                return _original(*args, **kwargs)

            monkeypatch.setattr(client._transport, name, tracked)
        gathered = request()
        reused = client.predict(gathered, model=KumoRelational()).to_pandas()
        monkeypatch.setattr(client._transport, 'serves_sessions', False)
        stateless = client.predict(gathered, model=KumoRelational()).to_pandas()
        assert reused.row_index.tolist() == [0, 1, 2]
        assert reused.id.tolist() == [6, 7, 8]
        pd.testing.assert_frame_equal(
            reused, stateless, check_exact=False, rtol=1e-5, atol=1e-6
        )
    assert calls == {
        'create_session': 1,
        'session_predict': 3,
        'delete_session': 1,
    }
