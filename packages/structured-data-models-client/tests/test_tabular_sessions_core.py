# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public fitted Tabular lifecycle through the explicit core session routes."""

import copy

import pandas as pd
import pytest
from sdm_connectors import read

from sdm_client import KumoTabular, Request, Schema, SDMClient
from sdm_client.errors import StructuredDataError

ROUTES = 'client.fit/tabular/frames,fitted.predict/tabular/frames'


class SessionEndpoint:
    url = 'http://localhost'
    serves_sessions = True

    def __init__(self):
        self.created = []
        self.scored = []
        self.deleted = []
        self.delete_failures = 0
        self.closed = False

    def create_session(self, payload):
        self.created.append(copy.deepcopy(payload))
        return {'session_id': f'session-{len(self.created)}'}

    def session_predict(self, session_id, payload):
        self.scored.append((session_id, copy.deepcopy(payload)))
        rows = payload['predict']['instance_table']['rows']
        return {
            'model': 'kumo-tabular',
            'id': 'session-answer',
            'predictions': [
                {'row_index': i, 'prediction': 10.0 + i}
                for i in range(len(rows))
            ],
            'metadata': {},
        }

    def delete_session(self, session_id):
        self.deleted.append(session_id)
        if self.delete_failures:
            self.delete_failures -= 1
            raise StructuredDataError('retry deletion', code='INVALID_REQUEST')

    def close(self):
        self.closed = True


def frame():
    return pd.DataFrame(
        {'id': [1, 2, 3], 'x': [1.0, 2.0, 3.0], 'y': [4.0, 5.0, 6.0]}
    )


def request(context, scores):
    schema = Schema.from_data(
        {'instance': context}, primary_keys={'instance': 'id'}
    )
    return Request(
        context, scores, target='y', task='regression', schema=schema
    )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('SDM_CORE_ROUTES', ROUTES)
    endpoint = SessionEndpoint()
    client = SDMClient.__new__(SDMClient)
    client._start(endpoint)
    try:
        yield client, endpoint
    finally:
        client.close()


def test_fit_two_scores_holdout_and_idempotent_close(client):
    client, endpoint = client
    context = frame()
    opening = request(context, pd.DataFrame({'id': [4], 'x': [4.5]}))
    fitted = client.fit(opening, model=KumoTabular(), holdout=[2])
    assert fitted._tabular_session is not None
    uploaded = endpoint.created[0]
    assert 'predict' not in uploaded
    table = uploaded['context']['instance_table']
    key = table['columns'].index('id')
    assert [row[key] for row in table['rows']] == [1, 3]
    for entity in (4, 5):
        answer = fitted.predict(
            request(context, pd.DataFrame({'id': [entity], 'x': [9.5]}))
        )
        assert answer.to_pandas()['prediction'].tolist() == [10.0]
    assert len(endpoint.created) == 1
    assert [sid for sid, _ in endpoint.scored] == ['session-1', 'session-1']
    assert all(
        'context' not in payload and 'schema' not in payload
        for _, payload in endpoint.scored
    )
    fitted.close()
    fitted.close()
    assert endpoint.deleted == ['session-1']


def test_empty_scoring_does_not_send_model_call(client):
    client, endpoint = client
    context = frame()
    fitted = client.fit(
        request(context, context[['id', 'x']].iloc[:1]), model=KumoTabular()
    )
    result = fitted.predict(request(context, context[['id', 'x']].iloc[:0]))
    assert result.to_pandas().empty
    assert endpoint.scored == []


def test_failed_delete_remains_retryable(client):
    client, endpoint = client
    context = frame()
    fitted = client.fit(
        request(context, pd.DataFrame({'id': [4], 'x': [4.0]})),
        model=KumoTabular(),
    )
    endpoint.delete_failures = 1
    with pytest.raises(StructuredDataError, match='retry deletion'):
        fitted.close()
    fitted.close()
    fitted.close()
    assert endpoint.deleted == ['session-1', 'session-1']


def test_client_cleanup_closes_transport_after_delete_failure(client):
    client, endpoint = client
    context = frame()
    client.fit(
        request(context, pd.DataFrame({'id': [4], 'x': [4.0]})),
        model=KumoTabular(),
    )
    endpoint.delete_failures = 1
    with pytest.raises(StructuredDataError, match='retry deletion'):
        client.close()
    assert endpoint.closed
    assert endpoint.deleted == ['session-1']


@pytest.mark.parametrize('extension', ['csv', 'parquet'])
def test_existing_file_reader_feeds_fitted_predictions(
    client, tmp_path, extension
):
    client, endpoint = client
    path = tmp_path / f'context.{extension}'
    original = frame()
    if extension == 'csv':
        original.to_csv(path, index=False)
    else:
        original.to_parquet(path, index=False)
    context = read(path)
    fitted = client.fit(
        request(context, pd.DataFrame({'id': [4], 'x': [4.0]})),
        model=KumoTabular(),
    )
    result = fitted.predict(
        request(context, pd.DataFrame({'id': [5], 'x': [5.0]}))
    )
    assert result.to_pandas()['prediction'].tolist() == [10.0]
    assert len(endpoint.created) == len(endpoint.scored) == 1


def test_fit_resolves_both_frames_and_later_scores_use_pinned_types(client):
    client, endpoint = client
    context = frame().astype({'x': 'int32'})
    first = pd.DataFrame({'id': [4], 'x': pd.Series([4.5], dtype='float64')})
    fitted = client.fit(request(context, first), model=KumoTabular())
    columns = endpoint.created[0]['schema']['instance_table']['columns']
    assert columns['x']['dtype'] == 'float64'
    later = pd.DataFrame({'id': [5], 'x': pd.Series([7], dtype='int32')})
    fitted.predict(request(context, later))
    table = endpoint.scored[-1][1]['predict']['instance_table']
    assert table['rows'][0][table['columns'].index('x')] == 7.0
    assert (
        endpoint.created[0]['schema']['instance_table']['columns']['x']['dtype']
        == 'float64'
    )


def test_incompatible_pinned_column_fails_before_scoring_call(client):
    client, endpoint = client
    context = frame()
    fitted = client.fit(
        request(context, pd.DataFrame({'id': [4], 'x': [4.0]})),
        model=KumoTabular(),
    )
    invalid = pd.DataFrame({'id': [5], 'x': ['not-a-number']})
    with pytest.raises((StructuredDataError, ValueError, TypeError)):
        fitted.predict(request(context, invalid))
    assert endpoint.scored == []


def test_holdout_does_not_round_high_precision_entity_identity(client):
    from decimal import Decimal, localcontext

    client, endpoint = client
    keys = [
        '1.12345678901234567890123456781',
        '1.12345678901234567890123456782',
    ]
    context = pd.DataFrame({'id': keys, 'x': [1.0, 2.0], 'y': [3.0, 4.0]})
    opening = request(context, pd.DataFrame({'id': ['new'], 'x': [5.0]}))
    with localcontext() as decimal_context:
        decimal_context.prec = 28
        fitted = client.fit(
            opening, model=KumoTabular(), holdout=[Decimal(keys[0])]
        )
    table = endpoint.created[0]['context']['instance_table']
    key_at = table['columns'].index('id')
    assert [row[key_at] for row in table['rows']] == keys[1:]
    fitted.close()
