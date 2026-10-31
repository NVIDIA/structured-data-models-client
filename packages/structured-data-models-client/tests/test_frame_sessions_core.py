# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public materialized-frame sessions use Rust policy and existing transport."""

import copy

import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
    _routing,
)
from sdm_client.errors import StructuredDataError
from sdm_client.wire.keys import _named_by, keys_in


class Endpoint:
    url = 'http://localhost'
    serves_sessions = True

    def __init__(self):
        self.created = []
        self.scored = []
        self.deleted = []
        self.fail_delete = False

    def create_session(self, payload):
        self.created.append(copy.deepcopy(payload))
        return {'session_id': 'frames-1'}

    def session_predict(self, session_id, payload):
        self.scored.append(copy.deepcopy(payload))
        table = payload['predict']['instance_table']
        rows = table['rows']
        instance = table['columns'].index('instance_id')
        return {
            'model': 'kumo-relational',
            'id': 'answer',
            'metadata': {},
            'predictions': [
                {
                    'id': str(rows[i][instance]),
                    'row_index': i,
                    'prediction': 5.0 + i,
                }
                for i in reversed(range(len(rows)))
            ],
        }

    def delete_session(self, session_id):
        self.deleted.append(session_id)
        if self.fail_delete:
            self.fail_delete = False
            raise StructuredDataError('retry deletion', code='INVALID_REQUEST')

    def close(self):
        pass


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv(_routing.ENVIRONMENT, ','.join(_routing.FRAME_ROUTES))
    endpoint = Endpoint()
    client = SDMClient.__new__(SDMClient)
    client._start(endpoint)
    try:
        yield client, endpoint
    finally:
        client.close()


def query():
    schema = Schema.from_data(
        {
            'users': pd.DataFrame(
                {'id': range(20), 'value': [i * 1.25 for i in range(20)]}
            ),
            'events': pd.DataFrame(
                {
                    'event': range(40),
                    'id': [i % 20 for i in range(40)],
                    'feature': [i * 2.5 for i in range(40)],
                }
            ),
        },
        primary_keys={'users': 'id', 'events': 'event'},
    )
    schema['users']['value'].stype = 'numerical'
    return PredictiveQuery('PREDICT users.value FOR EACH users.id', schema)


def gather(asked, indices):
    return asked.materialize(
        indices=indices, model=KumoRelational(), random_seed=7
    )


def test_materialize_fit_holdout_predict_close_and_request_isolation(client):
    client, endpoint = client
    asked = query()
    opening = gather(asked, [1, 2])
    before = copy.deepcopy(opening.payloads)
    fitted = client.fit(opening, model=KumoRelational(), holdout=[3, 4])
    assert fitted._relational_session is not None
    assert opening.payloads == before
    uploaded = endpoint.created[0]
    assert set(uploaded) == {'model', 'task', 'schema', 'context', 'metadata'}
    assert not keys_in(uploaded, 'context') & {_named_by([3]), _named_by([4])}
    instance = uploaded['context']['instance_table']
    at = instance['columns'].index('instance_id')
    surviving = {row[at] for row in instance['rows']}
    for table in uploaded['context']['related_tables'].values():
        at = table['columns'].index('instance_id')
        assert all(row[at] in surviving for row in table['rows'])
    for ids in ([3, 4], [1, 2]):
        scoring = gather(asked, ids)
        scoring._payloads[0]['metadata'] = {'trace': 'caller-tag'}
        result = fitted.predict(scoring)
        assert endpoint.scored[-1]['metadata']['trace'] == 'caller-tag'
        assert result.to_pandas()['prediction'].tolist() == [5.0, 6.0]
    assert all(
        set(payload) <= {'predict', 'output', 'inference', 'metadata'}
        for payload in endpoint.scored
    )
    assert 'context' not in fitted._relational_session_pinned.decode()
    fitted.close()
    fitted.close()
    assert endpoint.deleted == ['frames-1']


def test_session_refuses_changed_question_and_retries_failed_close(client):
    client, endpoint = client
    asked = query()
    fitted = client.fit(gather(asked, [1]), model=KumoRelational())
    changed = PredictiveQuery(
        'PREDICT users.value > 3 FOR EACH users.id', asked.schema
    )
    with pytest.raises(StructuredDataError, match='different question'):
        fitted.predict(gather(changed, [2]))
    assert not endpoint.scored
    endpoint.fail_delete = True
    with pytest.raises(StructuredDataError, match='retry deletion'):
        fitted.close()
    assert not fitted._closed
    fitted.close()
    assert fitted._closed
    assert endpoint.deleted == ['frames-1', 'frames-1']


def test_session_reports_pinned_answer_overlap(client):
    client, _ = client
    asked = query()
    fitted = client.fit(gather(asked, [1]), model=KumoRelational())
    scoring = gather(asked, [3])
    with pytest.warns(UserWarning, match='pinned target'):
        fitted.predict(scoring)


def test_later_scoring_failure_retains_paid_answers(client):
    client, endpoint = client
    asked = query()
    fitted = client.fit(gather(asked, [1]), model=KumoRelational())
    scoring = asked.materialize(
        indices=[2, 3], model=KumoRelational(), random_seed=7, rows_per_call=1
    )
    original = endpoint.session_predict
    calls = 0

    def fail_second(session_id, payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise StructuredDataError(
                'second call failed', code='INVALID_REQUEST'
            )
        return original(session_id, payload)

    endpoint.session_predict = fail_second
    with pytest.raises(
        StructuredDataError, match='second call failed'
    ) as caught:
        fitted.predict(scoring)
    assert caught.value.details['calls_answered'] == 1
    assert caught.value.details['rows_answered'] == 1
    assert len(caught.value.details['answered'].to_pandas()) == 1


@pytest.mark.parametrize('primary', [None, ''])
def test_session_id_fallback(client, primary):
    client, endpoint = client
    endpoint.create_session = lambda payload: {
        'session_id': primary,
        'id': 'fallback',
    }
    fitted = client.fit(gather(query(), [1]), model=KumoRelational())
    assert fitted.session_id == 'fallback'


def test_fit_explanation_validates_context_then_score_validates_rows(client):
    client, endpoint = client
    asked = query()
    fitted = client.fit(
        gather(asked, [1, 2]),
        model=KumoRelational(),
        outputs=['prediction', 'explanation'],
    )
    with pytest.raises(StructuredDataError, match='one|exactly'):
        fitted.predict(gather(asked, [3, 4]))
    assert not endpoint.scored
    fitted.predict(gather(asked, [3]))
    assert len(endpoint.scored) == 1


def test_holding_back_all_context_is_refused_before_upload(client):
    client, endpoint = client
    with pytest.raises(StructuredDataError, match='no context|empty'):
        client.fit(
            gather(query(), [1]), model=KumoRelational(), holdout=range(20)
        )
    assert not endpoint.created


def test_empty_explanation_score_is_local_and_new_context_is_not_validated(
    client,
):
    client, endpoint = client
    asked = query()
    fitted = client.fit(
        gather(asked, [1, 2]),
        model=KumoRelational(),
        outputs=['prediction', 'explanation'],
    )
    scoring = gather(asked, [3])
    table = scoring._payloads[0]['context']['instance_table']
    table['rows'] = [[object()]]  # The unsent context is never converted.
    fitted.predict(scoring)
    assert len(endpoint.scored) == 1
    scoring._payloads[0]['predict']['instance_table']['rows'] = []
    assert fitted.predict(scoring).to_pandas().empty
    assert len(endpoint.scored) == 1


def test_session_string_holdout_is_refused(client):
    client, endpoint = client
    with pytest.raises(StructuredDataError, match='string'):
        client.fit(
            gather(query(), [1, 2]), model=KumoRelational(), holdout='one'
        )
    assert not endpoint.created


def test_session_unsupported_endpoint_is_explained(client, monkeypatch):
    client, endpoint = client

    def unsupported(payload):
        error = RuntimeError('not implemented')
        error.status_code = 501
        raise error

    monkeypatch.setattr(endpoint, 'create_session', unsupported)
    with pytest.raises(StructuredDataError) as refused:
        client.fit(gather(query(), [1, 2]), model=KumoRelational())
    assert refused.value.code == 'SESSIONS_UNSUPPORTED'
