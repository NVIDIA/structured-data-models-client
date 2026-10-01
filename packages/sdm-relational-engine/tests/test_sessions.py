# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

from typing import Any

import pytest
from conftest import MOCK_URL
from predictive_query.query import ValidatedPredictiveQuery
from sdm_relational_engine.client import NimClient
from sdm_relational_engine.client.rfm import RFMAPI
from sdm_relational_engine.exceptions import (
    HTTPException,
    InvalidResponseError,
)
from sdm_relational_engine.rfm import Graph, KumoRelational
from sdm_relational_engine.rfm.inference.payload import (
    INSTANCE_ID,
    session_create_payload,
    session_predict_payload,
)
from sdm_relational_engine.rfm.inference.requests import RFMPredictResponse
from sdm_relational_engine.utils.progress_logger import PlainProgressLogger

_FULL_PAYLOAD = {
    'model': 'kumo-relational',
    'task': {'kind': 'regression'},
    'schema': {'instance_table': {}},
    'context': {'instance_table': {}, 'related_tables': {}},
    'predict': {'instance_table': {}, 'related_tables': {}},
    'output': {'fields': ['prediction']},
    'inference': {'run_mode': 'fast'},
    'metadata': {'source_query': 'PREDICT ...'},
}


class RecordingAPI:
    r"""Fake RFMAPI that records the create/session_predict/delete sequence and
    can simulate an unsupported NIM, an expired session, or a failing predict.
    """

    def __init__(
        self,
        *,
        unsupported: bool = False,
        expire_first_session_predict: bool = False,
        session_predict_status: int | None = None,
    ) -> None:
        self.calls: list[tuple[str, Any]] = []
        self._unsupported = unsupported
        self._expire_first = expire_first_session_predict
        self._session_predict_status = session_predict_status
        self._n_created = 0
        self._n_session_predict = 0

    def _response(self, entity_ids: Any) -> RFMPredictResponse:
        ids = list(entity_ids) if entity_ids is not None else [0]
        return RFMPredictResponse(
            prediction={
                'columns': ['ENTITY', 'TRUE_PROB'],
                'data': [[entity, 0.5] for entity in ids],
            }
        )

    def predict(
        self,
        request: Any,
        *,
        entity_ids: Any = None,
        instance_ids: Any = None,
        anchor_times: Any = None,
    ) -> RFMPredictResponse:
        self.calls.append(('predict', None))
        return self._response(entity_ids)

    def create_session(self, request: Any) -> str:
        if self._unsupported:
            self.calls.append(('create', None))
            raise HTTPException(404, 'sessions not supported')
        self._n_created += 1
        session_id = f'sess_{self._n_created}'
        self.calls.append(('create', session_id))
        return session_id

    def session_predict(
        self,
        session_id: str,
        request: Any,
        *,
        entity_ids: Any = None,
        instance_ids: Any = None,
        anchor_times: Any = None,
    ) -> RFMPredictResponse:
        self._n_session_predict += 1
        self.calls.append(('session_predict', session_id))
        if self._expire_first and self._n_session_predict == 1:
            raise HTTPException(404, 'SESSION_NOT_FOUND')
        if self._session_predict_status is not None:
            raise HTTPException(self._session_predict_status, 'boom')
        return self._response(entity_ids)

    def delete_session(self, session_id: str) -> None:
        self.calls.append(('delete', session_id))


def _kinds(api: RecordingAPI) -> list[str]:
    return [kind for kind, _ in api.calls]


def _model(graph: Graph, api: RecordingAPI) -> KumoRelational:
    model = KumoRelational(graph, verbose=False)
    model._client = api  # type: ignore[assignment]
    return model


def test_session_create_payload_keeps_context_sections() -> None:
    create = session_create_payload(_FULL_PAYLOAD)
    assert set(create) == {'model', 'task', 'schema', 'context', 'metadata'}
    assert create['context'] is _FULL_PAYLOAD['context']


def test_session_predict_payload_keeps_percall_sections() -> None:
    predict = session_predict_payload(_FULL_PAYLOAD)
    assert set(predict) == {'predict', 'output', 'inference', 'metadata'}
    assert 'context' not in predict and 'schema' not in predict


def test_split_covers_every_section() -> None:
    create = session_create_payload(_FULL_PAYLOAD)
    predict = session_predict_payload(_FULL_PAYLOAD)
    assert set(create) | set(predict) == set(_FULL_PAYLOAD)


def test_batch_mode_creates_one_session_and_deletes_it(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = RecordingAPI()
    model = _model(user_store_graph, api)
    with model.batch_mode(batch_size=2):
        df = model.predict(ltv, indices=[0, 1, 2, 3], verbose=False)
    kinds = _kinds(api)
    assert kinds.count('create') == 1
    assert kinds.count('session_predict') == 2
    assert kinds.count('predict') == 0
    assert kinds.count('delete') == 1
    assert kinds[-1] == 'delete'
    assert len(df) == 4


def test_single_predict_does_not_use_a_session(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = RecordingAPI()
    model = _model(user_store_graph, api)
    model.predict(ltv, indices=[0, 1, 2, 3], verbose=False)
    assert _kinds(api) == ['predict']


def test_single_batch_batch_mode_does_not_use_a_session(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = RecordingAPI()
    model = _model(user_store_graph, api)
    with model.batch_mode(batch_size=10):
        model.predict(ltv, indices=[0, 1], verbose=False)
    assert _kinds(api) == ['predict']


def test_random_seed_none_disables_sessions(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = RecordingAPI()
    model = _model(user_store_graph, api)
    with model.batch_mode(batch_size=2):
        model.predict(
            ltv, indices=[0, 1, 2, 3], random_seed=None, verbose=False
        )
    kinds = _kinds(api)
    assert 'create' not in kinds
    assert kinds.count('predict') == 2


def test_random_seed_none_says_the_context_is_re_uploaded(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = RecordingAPI()
    model = _model(user_store_graph, api)
    logger = PlainProgressLogger('Predicting', verbose=False)
    with model.batch_mode(batch_size=2):
        model.predict(
            ltv, indices=[0, 1, 2, 3], random_seed=None, verbose=logger
        )
    assert any(
        'random_seed=None' in msg and 're-uploads' in msg for msg in logger.logs
    )


def test_env_kill_switch_says_the_context_is_re_uploaded(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv('KUMO_RELATIONAL_DISABLE_SESSIONS', '1')
    api = RecordingAPI()
    model = _model(user_store_graph, api)
    logger = PlainProgressLogger('Predicting', verbose=False)
    with model.batch_mode(batch_size=2):
        model.predict(ltv, indices=[0, 1, 2, 3], verbose=logger)
    assert any('KUMO_RELATIONAL_DISABLE_SESSIONS' in msg for msg in logger.logs)


def test_a_session_backed_run_says_nothing_about_sessions(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = RecordingAPI()
    model = _model(user_store_graph, api)
    logger = PlainProgressLogger('Predicting', verbose=False)
    with model.batch_mode(batch_size=2):
        model.predict(ltv, indices=[0, 1, 2, 3], verbose=logger)
    assert not any('Sessions disabled' in msg for msg in logger.logs)


def test_env_kill_switch_disables_sessions(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv('KUMO_RELATIONAL_DISABLE_SESSIONS', '1')
    api = RecordingAPI()
    model = _model(user_store_graph, api)
    with model.batch_mode(batch_size=2):
        model.predict(ltv, indices=[0, 1, 2, 3], verbose=False)
    kinds = _kinds(api)
    assert 'create' not in kinds
    assert kinds.count('predict') == 2


def test_unsupported_sessions_fall_back_to_stateless(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = RecordingAPI(unsupported=True)
    model = _model(user_store_graph, api)
    with model.batch_mode(batch_size=2):
        df = model.predict(ltv, indices=[0, 1, 2, 3], verbose=False)
    kinds = _kinds(api)
    assert kinds.count('create') == 1
    assert kinds.count('session_predict') == 0
    assert kinds.count('predict') == 2
    assert 'delete' not in kinds
    assert len(df) == 4


def test_expired_session_is_recreated_and_retried(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = RecordingAPI(expire_first_session_predict=True)
    model = _model(user_store_graph, api)
    with model.batch_mode(batch_size=2):
        df = model.predict(ltv, indices=[0, 1, 2, 3], verbose=False)
    kinds = _kinds(api)
    assert kinds.count('create') == 2
    assert kinds.count('session_predict') == 3
    assert kinds.count('delete') == 1
    assert len(df) == 4


def test_session_is_deleted_even_when_a_batch_fails(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    api = RecordingAPI(session_predict_status=500)
    model = _model(user_store_graph, api)
    with pytest.raises(RuntimeError):
        with model.batch_mode(batch_size=2, num_retries=0):
            model.predict(ltv, indices=[0, 1, 2, 3], verbose=False)
    kinds = _kinds(api)
    assert 'create' in kinds
    assert kinds[-1] == 'delete'


def test_rfmapi_create_session_returns_id(mock_api: Any) -> None:
    mock_api.post(
        f'{MOCK_URL}/v1/sessions',
        json={'session_id': 'sess_x', 'ttl_seconds': 3600},
    )
    api = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))
    assert api.create_session({'model': 'kumo-relational'}) == 'sess_x'


def test_rfmapi_create_session_requires_id(mock_api: Any) -> None:
    """A ``200`` with no ``session_id`` is the server breaking the contract: the
    request that produced it is the client's own generated payload, so the caller
    cannot have influenced it. Raising the classified type is what puts it in
    the ``INVALID_RESPONSE`` branch of the client's error classifier rather than
    the ``ValueError`` one, which blames the request.
    """
    mock_api.post(f'{MOCK_URL}/v1/sessions', json={'ttl_seconds': 3600})
    api = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))
    with pytest.raises(InvalidResponseError, match='session_id'):
        api.create_session({'model': 'kumo-relational'})


def test_rfmapi_create_session_rejects_a_non_json_body(mock_api: Any) -> None:
    mock_api.post(f'{MOCK_URL}/v1/sessions', text='<html>gateway</html>')
    api = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))
    with pytest.raises(InvalidResponseError):
        api.create_session({'model': 'kumo-relational'})


def test_rfmapi_delete_session_issues_delete(mock_api: Any) -> None:
    matcher = mock_api.delete(f'{MOCK_URL}/v1/sessions/sess_x', status_code=204)
    api = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))
    api.delete_session('sess_x')
    assert matcher.called


def test_rfmapi_session_predict_targets_session_path(mock_api: Any) -> None:
    def _response(request: Any, _context: Any) -> dict[str, Any]:
        table = request.json()['predict']['instance_table']
        index = table['columns'].index(INSTANCE_ID)
        return {
            'id': 'pred',
            'model': 'kumo-relational',
            'predictions': [
                {'id': str(row[index]), 'row_index': i, 'prediction': 0.5}
                for i, row in enumerate(table['rows'])
            ],
            'metadata': {'task_kind': 'regression'},
        }

    matcher = mock_api.post(
        f'{MOCK_URL}/v1/sessions/sess_x/predictions', json=_response
    )
    api = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))
    request = {
        'predict': {
            'instance_table': {'columns': [INSTANCE_ID], 'rows': [[0]]}
        },
        'output': {},
    }
    response = api.session_predict(
        'sess_x', request, entity_ids=[7], instance_ids=[0]
    )
    assert matcher.called
    assert 'sessions/sess_x/predictions' in matcher.last_request.url
    assert response.prediction['data'][0][0] == 7
