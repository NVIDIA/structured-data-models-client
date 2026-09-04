# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import re
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import requests
from conftest import MOCK_URL
from predictive_query.query import ValidatedPredictiveQuery
from predictive_query.schema.task import TaskType
from predictive_query.schema.typing import Stype
from sdm_api.generated import tfm_api as tfm_api_module
from sdm_relational_engine.api.rfm.context import Table
from sdm_relational_engine.client import NimClient
from sdm_relational_engine.client.rfm import RFMAPI
from sdm_relational_engine.rfm import Graph, KumoRelational, TaskTable
from sdm_relational_engine.rfm.payload import (
    ANCHOR_TIME_PREFIX,
    ENTITY_REFERENCE_PREFIX,
    INSTANCE_ID,
    SYNTHETIC_NODE_ID,
    _dtype_name,
    _occurrence_dataframe,
    _target_json_value,
)
from sdm_relational_engine.rfm.rfm import Explanation

# The Universal TFM OpenAPI contract is not vendored here, so the tests that
# replay it skip unless a checkout is pointed at explicitly. The default is the
# sibling-directory layout the maintainers use; anyone else sets
# KUMO_RELATIONAL_CONTRACT_DIR.
_ENV_CONTRACT_DIR = 'KUMO_RELATIONAL_CONTRACT_DIR'
CANONICAL_SPEC = (
    Path(os.environ.get(_ENV_CONTRACT_DIR, '../structured-data-api'))
    / 'nim-sd.openapi.yaml'
)


def test_keyless_table_synthetic_key_is_opaque_on_name_collision() -> None:
    source = pd.DataFrame(
        {
            SYNTHETIC_NODE_ID: [10, 11],
            f'{SYNTHETIC_NODE_ID}_1': [20, 21],
            'VALUE': [1.5, 2.5],
        }
    )
    table = Table(
        df=source,
        row=None,
        batch=np.array([0, 1]),
        num_sampled_nodes=[2],
        stype_dict={'VALUE': Stype.numerical},
        primary_key=None,
    )

    materialized, key_column = _occurrence_dataframe(table)

    assert key_column == f'{SYNTHETIC_NODE_ID}_2'
    assert materialized.columns.tolist() == [
        INSTANCE_ID,
        key_column,
        SYNTHETIC_NODE_ID,
        f'{SYNTHETIC_NODE_ID}_1',
        'VALUE',
    ]
    assert materialized[key_column].tolist() == [0, 1]
    assert materialized[SYNTHETIC_NODE_ID].tolist() == [10, 11]
    assert materialized[f'{SYNTHETIC_NODE_ID}_1'].tolist() == [20, 21]
    assert source.columns.tolist() == [
        SYNTHETIC_NODE_ID,
        f'{SYNTHETIC_NODE_ID}_1',
        'VALUE',
    ]


class JsonPayloadReceptor:
    def __init__(self) -> None:
        self.payload: dict[str, Any] | None = None
        self.headers: requests.structures.CaseInsensitiveDict[str] | None = None

    def __call__(self, request: Any) -> bool:
        self.payload = request.json()
        self.headers = request.headers
        return True


def _correlated_response(item_fields: dict[str, Any]) -> Any:
    def response(request: Any, _context: Any) -> dict[str, Any]:
        payload = request.json()
        table = payload['predict']['instance_table']
        instance_index = table['columns'].index(INSTANCE_ID)
        predictions = [
            {
                'id': str(row[instance_index]),
                'row_index': row_index,
                **item_fields,
            }
            for row_index, row in enumerate(table['rows'])
        ]
        return {
            'id': 'pred-test',
            'model': 'kumo-relational',
            'predictions': predictions,
            'metadata': {
                'task_kind': 'regression',
            },
        }

    return response


def test_predict_posts_universal_json_payload(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    mock_api: Any,
) -> None:
    receptor = JsonPayloadReceptor()
    mock_api.post(
        f'{MOCK_URL}/v1/predictions',
        additional_matcher=receptor,
        json=_correlated_response(
            {
                'prediction': 0.5,
                'embeddings': [0.1, 0.2],
            }
        ),
    )

    model = KumoRelational(user_store_graph, verbose=False)
    model._client = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))  # type: ignore

    result = model.predict(
        ltv,
        indices=[3],
        inference_config={'output_type': 'quantiles'},
        return_embeddings=True,
        verbose=False,
    )

    assert result.to_dict('records') == [
        {
            'ENTITY': 3,
            'ANCHOR_TIMESTAMP': pd.Timestamp('2025-01-09 00:00:00', tz='UTC'),
            'PREDICTION': 0.5,
            'EMBEDDINGS': [0.1, 0.2],
        }
    ]
    assert receptor.headers is not None
    assert receptor.headers['Content-Type'] == 'application/json'

    payload = receptor.payload
    assert payload is not None
    assert 'version' not in payload
    assert payload['model'] == 'kumo-relational'
    assert payload['task']['kind'] == 'regression'
    assert payload['schema']['relationships']
    assert payload['context']['instance_table']['format'] == 'arrays'
    assert payload['predict']['instance_table']['format'] == 'arrays'
    assert (
        payload['task']['target']['column_name']
        in payload['context']['instance_table']['columns']
    )
    assert (
        payload['task']['target']['column_name']
        not in payload['predict']['instance_table']['columns']
    )
    assert payload['schema']['instance_table']['primary_key'] == INSTANCE_ID
    predict_instance_table = payload['predict']['instance_table']
    predict_instance_id_index = predict_instance_table['columns'].index(
        INSTANCE_ID
    )
    assert predict_instance_table['rows'][0][predict_instance_id_index] != 3
    context_instance_table = payload['context']['instance_table']
    instance_id_index = context_instance_table['columns'].index(INSTANCE_ID)
    instance_ids = [
        row[instance_id_index] for row in context_instance_table['rows']
    ]
    assert len(instance_ids) == len(set(instance_ids))
    assert (
        '__kumo_instance_feature'
        not in payload['context']['instance_table']['columns']
    )
    anchor_column = payload['task']['anchor_time_column']
    assert anchor_column.startswith(ANCHOR_TIME_PREFIX)
    assert anchor_column in payload['context']['instance_table']['columns']
    assert anchor_column in payload['predict']['instance_table']['columns']
    assert (
        payload['schema']['instance_table']['columns'][anchor_column]['stype']
        == 'timestamp'
    )
    entity_relationship = payload['schema']['relationships'][0]
    assert 'source_table' not in entity_relationship
    assert entity_relationship['source_columns'][0].startswith(
        ENTITY_REFERENCE_PREFIX
    )
    assert entity_relationship['target_table'] == 'USERS'
    assert payload['schema']['related_tables']['USERS']['primary_key'] == (
        [INSTANCE_ID, 'USER_ID']
    )
    for schema in payload['schema']['related_tables'].values():
        assert INSTANCE_ID in schema['primary_key']
        assert schema['columns'][INSTANCE_ID]['nullable'] is False
    assert (
        payload['schema']['related_tables']['ORDERS']['columns']['TIME'][
            'dtype'
        ]
        == 'timestamp[us]'
    )
    assert payload['context']['related_tables']['ORDERS']['rows'][0][
        payload['context']['related_tables']['ORDERS']['columns'].index('TIME')
    ].endswith('Z')
    assert payload['task']['target']['dtype'] == 'float32'
    assert 'embeddings' in payload['output']['fields']
    assert 'quantiles' in payload['output']['fields']
    assert 'prediction' not in payload['output']['fields']
    assert payload['inference']['run_mode'] == 'fast'
    assert payload['inference']['inference_config']['kind'] == 'regression'
    assert payload['inference']['inference_config']['output_type'] == (
        'quantiles'
    )
    assert 'operation' not in payload['metadata']
    _assert_payload_matches_local_prediction_request_schema(payload)

    payload_text = str(payload)
    assert 'application/x-protobuf' not in payload_text
    assert "'batch'" not in payload_text
    assert "'row'" not in payload_text
    assert "'col'" not in payload_text
    assert "'evaluate'" not in payload_text


def test_forecast_payload_contains_universal_controls(
    user_store_graph: Graph,
    forecast: ValidatedPredictiveQuery,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    task_table = model._get_task_table(
        forecast,
        indices=forecast.get_rfm_entity_id_list(),
        anchor_time=pd.Timestamp('2025-01-05'),
    )

    payload = model.materialize_task(
        task_table,
        inference_config={'output_type': 'quantiles'},
        verbose=False,
    )[0].payload

    assert payload['task']['kind'] == 'forecasting'
    assert payload['task']['step_size'] == 86_400_000_000_000
    assert payload['task']['num_forecasts'] == 4
    assert payload['task']['target']['dtype'] == 'float32'
    assert set(payload['output']['fields']) == {'quantiles'}
    assert payload['inference']['inference_config']['kind'] == 'regression'
    assert payload['inference']['inference_config']['output_type'] == (
        'quantiles'
    )


def test_link_prediction_payload_preserves_sampled_rhs_candidates(
    user_store_graph: Graph,
) -> None:
    task = TaskTable(
        task_type=TaskType.TEMPORAL_LINK_PREDICTION,
        context_df=pd.DataFrame(
            {
                'ENTITY': [0, 1],
                # Deliberately use target IDs that cannot occur in the sampled
                # STORES table. Targets remain supervision on the instance table;
                # payload materialization must not fabricate RHS candidate rows.
                'TARGET': [['missing-0'], ['missing-1']],
                'ANCHOR_TIMESTAMP': pd.to_datetime(
                    [
                        '2025-01-05',
                        '2025-01-05',
                    ]
                ),
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': [3],
                'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-01-05']),
            }
        ),
        entity_table_name=('USERS', 'STORES'),
        entity_column='ENTITY',
        target_column='TARGET',
        time_column='ANCHOR_TIMESTAMP',
    )

    payload = (
        KumoRelational(user_store_graph, verbose=False)
        .materialize_task(
            task,
            top_k=2,
            num_neighbors=[4, 4],
            verbose=False,
        )[0]
        .payload
    )

    assert payload['task']['kind'] == 'temporal_link_prediction'
    assert payload['task']['entity_table_names'] == ['USERS', 'STORES']
    assert payload['task']['top_k'] == 2
    assert payload['task']['target'] == {
        'column_name': 'TARGET',
        'dtype': 'stringlist',
    }
    assert set(payload['output']['fields']) == {'rankings'}
    target_schema = payload['schema']['instance_table']['columns']['TARGET']
    assert target_schema['dtype'] == 'stringlist'
    assert target_schema['stype'] == 'multicategorical'
    assert payload['schema']['related_tables']['USERS']['primary_key'] == (
        [INSTANCE_ID, 'USER_ID']
    )
    assert payload['schema']['related_tables']['STORES']['primary_key'] == (
        [INSTANCE_ID, 'STORE_ID']
    )
    assert 'TARGET' in payload['context']['instance_table']['columns']
    assert 'TARGET' not in payload['predict']['instance_table']['columns']
    target_index = payload['context']['instance_table']['columns'].index(
        'TARGET'
    )
    assert [
        row[target_index]
        for row in payload['context']['instance_table']['rows']
    ] == ([['missing-0'], ['missing-1']])
    stores = payload['context']['related_tables']['STORES']
    instance_index = stores['columns'].index(INSTANCE_ID)
    store_index = stores['columns'].index('STORE_ID')
    store_candidates = {
        (row[instance_index], str(row[store_index])) for row in stores['rows']
    }
    assert all(
        store_id not in {'missing-0', 'missing-1'}
        for _, store_id in store_candidates
    )
    assert {instance_id for instance_id, _ in store_candidates} == {0, 1}


def test_link_prediction_target_values_must_be_stringlist_arrays() -> None:
    task_type = TaskType.TEMPORAL_LINK_PREDICTION

    assert _target_json_value(task_type, np.array([1, '2'])) == ['1', '2']

    with pytest.raises(ValueError, match='stringlist arrays'):
        _target_json_value(task_type, 'item-1')

    with pytest.raises(ValueError, match='must not contain null'):
        _target_json_value(task_type, ['item-1', None])


def test_stringlist_dtype_detection_rejects_mixed_or_null_items() -> None:
    assert _dtype_name(
        pd.Series([['a'], None, np.array(['b'])], dtype=object)
    ) == ('stringlist')

    with pytest.raises(ValueError, match='only arrays or nulls'):
        _dtype_name(pd.Series([['a'], 'b'], dtype=object))

    with pytest.raises(ValueError, match='only arrays or nulls'):
        _dtype_name(pd.Series(['b', ['a']], dtype=object))

    with pytest.raises(ValueError, match='must not contain null items'):
        _dtype_name(pd.Series([['a', None]], dtype=object))


def test_payload_encodes_unsafe_int64_as_base10_strings() -> None:
    large_id = 9007199254740993
    large_feature = 2**62 + 1
    users = pd.DataFrame(
        {
            'USER_ID': np.arange(large_id, large_id + 8, dtype='int64'),
            'BIG_FEATURE': np.full(8, large_feature, dtype='int64'),
            'AGE': np.arange(20, 28, dtype='int64'),
        }
    )
    orders = pd.DataFrame(
        {
            'ORDER_ID': np.arange(16, dtype='int64'),
            'USER_ID': np.repeat(users['USER_ID'].to_numpy(), 2),
            'AMOUNT': np.arange(16, dtype='float64'),
            'TIME': pd.to_datetime(['2025-01-01'] * 16),
        }
    )
    graph = Graph.from_data({'USERS': users, 'ORDERS': orders}, verbose=False)
    task = TaskTable(
        task_type=TaskType.REGRESSION,
        context_df=pd.DataFrame(
            {
                'ENTITY': users['USER_ID'].to_numpy()[:6],
                'TARGET': np.arange(6, dtype='float64'),
                'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-01-05'] * 6),
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': users['USER_ID'].to_numpy()[6:],
                'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-01-05'] * 2),
            }
        ),
        entity_table_name='USERS',
        entity_column='ENTITY',
        target_column='TARGET',
        time_column='ANCHOR_TIMESTAMP',
    )

    payload = (
        KumoRelational(graph, verbose=False)
        .materialize_task(
            task,
            verbose=False,
        )[0]
        .payload
    )

    users_table = payload['context']['related_tables']['USERS']
    users_schema = payload['schema']['related_tables']['USERS']['columns']
    user_id_index = users_table['columns'].index('USER_ID')
    feature_index = users_table['columns'].index('BIG_FEATURE')
    age_index = users_table['columns'].index('AGE')
    assert users_schema['USER_ID']['dtype'] == 'int64'
    assert users_schema['BIG_FEATURE']['dtype'] == 'int64'
    assert users_table['rows'][0][user_id_index] == str(large_id)
    assert users_table['rows'][0][feature_index] == str(large_feature)
    assert users_table['rows'][0][age_index] == 20

    orders_table = payload['context']['related_tables']['ORDERS']
    fkey_index = orders_table['columns'].index('USER_ID')
    assert orders_table['rows'][0][fkey_index] == str(large_id)

    instance_table = payload['context']['instance_table']
    entity_ref_column = next(
        column
        for column in instance_table['columns']
        if column.startswith(ENTITY_REFERENCE_PREFIX)
    )
    entity_index = instance_table['columns'].index(entity_ref_column)
    assert (
        payload['schema']['instance_table']['columns'][entity_ref_column][
            'dtype'
        ]
        == 'int64'
    )
    assert instance_table['rows'][0][entity_index] == str(large_id)

    encoded = json.dumps(payload)
    assert not re.search(rf'(?<![\d"]){large_id}(?![\d"])', encoded)
    assert not re.search(rf'(?<![\d"]){large_feature}(?![\d"])', encoded)


def test_entity_identity_survives_batch_local_row_indexes(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    mock_api: Any,
) -> None:
    session_id = 'sess_test'
    mock_api.post(
        f'{MOCK_URL}/v1/sessions',
        json={
            'session_id': session_id,
            'expires_at': '2099-01-01T00:00:00Z',
            'ttl_seconds': 3600,
        },
    )
    mock_api.post(
        f'{MOCK_URL}/v1/sessions/{session_id}/predictions',
        json=_correlated_response({'prediction': 0.5}),
    )
    mock_api.delete(f'{MOCK_URL}/v1/sessions/{session_id}', status_code=204)
    model = KumoRelational(user_store_graph, verbose=False)
    model._client = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))  # type: ignore

    with model.batch_mode(batch_size=1):
        result = model.predict(ltv, indices=[3, 1], verbose=False)

    assert result['ENTITY'].tolist() == [3, 1]


def test_explain_surfaces_summary_from_nim_wrapped_details(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    mock_api: Any,
) -> None:
    nim_explanation = {
        'format': 'kumo_rfm_v2_1',
        'details': {
            'summary': 'Recent order value drove the prediction.',
            'warning': 'Sparse neighborhood for this entity.',
            'col_grads': {'ORDERS': {'PRICE': 0.42}},
        },
    }
    mock_api.post(
        f'{MOCK_URL}/v1/predictions',
        json=_correlated_response(
            {
                'prediction': 0.5,
                'explanation': nim_explanation,
            }
        ),
    )

    model = KumoRelational(user_store_graph, verbose=False)
    model._client = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))  # type: ignore

    result = model.predict(ltv, indices=[0], explain=True, verbose=False)

    assert isinstance(result, Explanation)
    assert result.summary == 'Recent order value drove the prediction.'
    assert result.warning == 'Sparse neighborhood for this entity.'
    assert result.details == nim_explanation


def test_explain_matches_live_nim_shape_with_no_summary(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    mock_api: Any,
) -> None:
    r"""Captured kumo_rfm_v2_1 shape from a live NIM driver: structured
    attribution only, no server summary. With ``skip_summary`` the client leaves
    ``summary`` empty and the rich details and typed accessors stay available.
    """
    from sdm_relational_engine.rfm.rfm import ExplainConfig

    live_explanation = {
        'format': 'kumo_rfm_v2_1',
        'details': {
            'task_type': 'binary_classification',
            'cohorts': [
                {
                    'table_name': 'accounts',
                    'column_name': 'amount',
                    'hop': 0,
                    'stype': 'numerical',
                    'cohorts': ['[8 - 9.25]', '(280 - 300]'],
                    'populations': [0.5, 0.5],
                    'targets': [0.0, 1.0],
                },
            ],
            'subgraphs': [
                {
                    'seed_id': 0,
                    'seed_table': 'accounts',
                    'seed_time': '2025-02-01T00:00:00',
                    'tables': {},
                    'context_examples': [],
                },
            ],
        },
    }
    mock_api.post(
        f'{MOCK_URL}/v1/predictions',
        json=_correlated_response(
            {
                'prediction': 0.5,
                'explanation': live_explanation,
            }
        ),
    )

    model = KumoRelational(user_store_graph, verbose=False)
    model._client = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))  # type: ignore

    result = model.predict(
        ltv,
        indices=[0],
        explain=ExplainConfig(skip_summary=True),
        verbose=False,
    )

    assert isinstance(result, Explanation)
    assert result.summary == ''
    assert result.warning is None
    assert result.details == live_explanation
    assert result.cohorts[0]['column_name'] == 'amount'
    assert len(result.subgraphs) == 1


def test_explain_requests_explanation_output_field(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    mock_api: Any,
) -> None:
    receptor = JsonPayloadReceptor()
    mock_api.post(
        f'{MOCK_URL}/v1/predictions',
        additional_matcher=receptor,
        json=_correlated_response(
            {
                'prediction': 0.5,
                'explanation': {
                    'format': 'natural_language_summary',
                    'summary': 'Order frequency dropped.',
                    'warning': 'Cross-region fallback used.',
                },
            }
        ),
    )

    model = KumoRelational(user_store_graph, verbose=False)
    model._client = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))  # type: ignore

    result = model.predict(ltv, indices=[0], explain=True, verbose=False)

    assert isinstance(result, Explanation)
    assert result.prediction.to_dict('records') == [
        {
            'ENTITY': 0,
            'ANCHOR_TIMESTAMP': pd.Timestamp('2025-01-09 00:00:00', tz='UTC'),
            'PREDICTION': 0.5,
        }
    ]
    assert result.summary == 'Order frequency dropped.'
    assert result.details == {
        'format': 'natural_language_summary',
        'summary': 'Order frequency dropped.',
        'warning': 'Cross-region fallback used.',
    }
    assert result.warning == 'Cross-region fallback used.'

    assert receptor.headers is not None
    assert receptor.headers['Content-Type'] == 'application/json'
    payload = receptor.payload
    assert payload is not None
    assert 'explanation' in payload['output']['fields']
    assert 'operation' not in payload['metadata']


def _v2_1_explanation() -> dict[str, Any]:
    return {
        'format': 'kumo_rfm_v2_1',
        'details': {
            'task_type': 'regression',
            'cohorts': [
                {
                    'table_name': 'orders',
                    'column_name': 'COUNT(*)',
                    'hop': 1,
                    'cohorts': ['[0-0]', '(0-4+]'],
                    'populations': [0.2, 0.8],
                    'targets': [0.0, 0.5],
                }
            ],
            'subgraphs': [
                {
                    'seed_id': 0,
                    'seed_table': 'users',
                    'tables': {},
                }
            ],
        },
    }


def test_explain_generates_nl_summary_for_v2_1_payload(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    mock_api: Any,
    monkeypatch: Any,
) -> None:
    calls: dict[str, Any] = {}

    def fake_generate(
        *,
        query: str,
        prediction: Any,
        cohorts: Any,
        subgraphs: Any,
        **kwargs: Any,
    ) -> str:
        calls.update(
            query=query,
            prediction=prediction,
            cohorts=cohorts,
            subgraphs=subgraphs,
            **kwargs,
        )
        return 'Generated NL summary.'

    monkeypatch.setattr(
        'sdm_relational_engine.rfm.rfm.generate_summary', fake_generate
    )
    mock_api.post(
        f'{MOCK_URL}/v1/predictions',
        json=_correlated_response(
            {
                'prediction': 0.5,
                'explanation': _v2_1_explanation(),
            }
        ),
    )
    model = KumoRelational(user_store_graph, verbose=False)
    model._client = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))  # type: ignore

    result = model.predict(ltv, indices=[0], explain=True, verbose=False)

    assert isinstance(result, Explanation)
    assert result.summary == 'Generated NL summary.'
    assert len(result.cohorts) == 1
    assert result.cohorts[0]['column_name'] == 'COUNT(*)'
    assert len(result.subgraphs) == 1
    assert result.subgraphs[0]['seed_id'] == 0
    assert calls['query']
    assert calls['cohorts'] == _v2_1_explanation()['details']['cohorts']
    assert calls['subgraphs'] == _v2_1_explanation()['details']['subgraphs']


def test_explain_skips_generation_when_skip_summary(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    mock_api: Any,
    monkeypatch: Any,
) -> None:
    from sdm_relational_engine.rfm.rfm import ExplainConfig

    called = False

    def fake_generate(**kwargs: Any) -> str:
        nonlocal called
        called = True
        return 'x'

    monkeypatch.setattr(
        'sdm_relational_engine.rfm.rfm.generate_summary', fake_generate
    )
    mock_api.post(
        f'{MOCK_URL}/v1/predictions',
        json=_correlated_response(
            {
                'prediction': 0.5,
                'explanation': _v2_1_explanation(),
            }
        ),
    )
    model = KumoRelational(user_store_graph, verbose=False)
    model._client = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))  # type: ignore

    result = model.predict(
        ltv,
        indices=[0],
        explain=ExplainConfig(skip_summary=True),
        verbose=False,
    )

    assert isinstance(result, Explanation)
    assert result.summary == ''
    assert called is False
    assert len(result.cohorts) == 1


def test_explain_prefers_server_summary_over_generation(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
    mock_api: Any,
    monkeypatch: Any,
) -> None:
    called = False

    def fake_generate(**kwargs: Any) -> str:
        nonlocal called
        called = True
        return 'x'

    monkeypatch.setattr(
        'sdm_relational_engine.rfm.rfm.generate_summary', fake_generate
    )
    mock_api.post(
        f'{MOCK_URL}/v1/predictions',
        json=_correlated_response(
            {
                'prediction': 0.5,
                'explanation': {
                    'format': 'natural_language_summary',
                    'summary': 'Server text.',
                },
            }
        ),
    )
    model = KumoRelational(user_store_graph, verbose=False)
    model._client = RFMAPI(NimClient(MOCK_URL, api_key='DISABLED'))  # type: ignore

    result = model.predict(ltv, indices=[0], explain=True, verbose=False)

    assert result.summary == 'Server text.'
    assert called is False


def _assert_payload_matches_local_prediction_request_schema(
    payload: dict[str, Any],
) -> None:
    spec = _load_local_canonical_spec_at_generated_revision()
    if spec is None:
        return

    schema = spec['components']['schemas']['PredictionRequest']
    assert set(payload) == set(schema['properties'])
    assert set(schema['required']) <= set(payload)
    assert 'operation' not in payload.get('metadata', {})
    assert 'evaluate' not in payload


def _load_local_canonical_spec_at_generated_revision() -> dict | None:
    if not CANONICAL_SPEC.exists():
        return None
    # Located through the imported module rather than a path relative to the
    # working directory: the latter resolved to nothing from the directory
    # pytest actually runs in, so this guard returned early and the schema
    # assertions below never ran.
    output = Path(tfm_api_module.__file__)
    if _generated_source_sha(output) != _file_sha256(CANONICAL_SPEC):
        return None
    yaml = pytest.importorskip('yaml')
    return yaml.safe_load(CANONICAL_SPEC.read_text())


def _generated_source_sha(path: Path) -> str:
    match = re.search(
        r'^# Source SHA256: ([0-9a-f]+)$', path.read_text(), flags=re.MULTILINE
    )
    assert match is not None
    return match.group(1)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_explain_gpu_pressure_surfaces_clear_error(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    from sdm_relational_engine.exceptions import HTTPException

    class _FailingAPI:
        def predict(self, *args: Any, **kwargs: Any) -> Any:
            raise HTTPException(503, 'Service Unavailable')

    model = KumoRelational(user_store_graph, verbose=False)
    model._client = _FailingAPI()  # type: ignore

    with pytest.raises(RuntimeError) as err:
        model.predict(ltv, indices=[0], explain=True, verbose=False)
    msg = str(err.value)
    assert 'temporarily unavailable' in msg
    assert 'GPU memory pressure' in msg
    assert 'one at a time' in msg


def test_prediction_connection_drop_surfaces_clear_error(
    user_store_graph: Graph,
    ltv: ValidatedPredictiveQuery,
) -> None:
    import requests

    class _DroppingAPI:
        def predict(self, *args: Any, **kwargs: Any) -> Any:
            raise requests.exceptions.ConnectionError(
                'connection reset by peer'
            )

    model = KumoRelational(user_store_graph, verbose=False)
    model._client = _DroppingAPI()  # type: ignore

    with pytest.raises(RuntimeError) as err:
        model.predict(ltv, indices=[3], verbose=False)
    assert 'temporarily unavailable' in str(err.value)


def _feature_payload(feature: Any) -> dict[str, Any]:
    users = pd.DataFrame(
        {
            'USER_ID': np.arange(8, dtype='int64'),
            'FEATURE': feature,
        }
    )
    orders = pd.DataFrame(
        {
            'ORDER_ID': np.arange(16, dtype='int64'),
            'USER_ID': np.repeat(np.arange(8, dtype='int64'), 2),
            'AMOUNT': np.arange(16, dtype='float64'),
            'TIME': pd.to_datetime(['2025-01-01'] * 16),
        }
    )
    graph = Graph.from_data({'USERS': users, 'ORDERS': orders}, verbose=False)
    task = TaskTable(
        task_type=TaskType.REGRESSION,
        context_df=pd.DataFrame(
            {
                'ENTITY': np.arange(6, dtype='int64'),
                'TARGET': np.arange(6, dtype='float64'),
                'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-01-05'] * 6),
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': np.arange(6, 8, dtype='int64'),
                'ANCHOR_TIMESTAMP': pd.to_datetime(['2025-01-05'] * 2),
            }
        ),
        entity_table_name='USERS',
        entity_column='ENTITY',
        target_column='TARGET',
        time_column='ANCHOR_TIMESTAMP',
    )
    return (
        KumoRelational(graph, verbose=False)
        .materialize_task(
            task,
            verbose=False,
        )[0]
        .payload
    )


@pytest.mark.parametrize('value', [float('inf'), float('-inf')])
def test_non_finite_cell_names_its_table_column_and_row(value: float) -> None:
    # Regression: these used to escape as a bare ValueError out of
    # ``json.dumps`` inside the request-size helper, naming neither table,
    # column nor row.
    with pytest.raises(ValueError, match=r"Column 'USERS.FEATURE' row 3"):
        _feature_payload([1.0, 1.0, 1.0, value] + [1.0] * 4)


def test_decimal_cells_are_named_by_what_they_hold() -> None:
    # Regression: every DB-API driver returns Decimal for NUMERIC columns,
    # which used to raise 'Object of type Decimal is not JSON serializable'.
    # Serializing them is not enough on its own. A DECIMAL column announced as
    # `string` reaches the model as text, so a numeric feature stops being
    # numeric and an id stops being an id. The dtype follows the values, and
    # falls back to `string` only where precision demands it.

    # Fractional: float64, and the cells are numbers rather than strings.
    payload = _feature_payload([Decimal(f'-{index}.50') for index in range(8)])
    table = payload['context']['related_tables']['USERS']
    schema = payload['schema']['related_tables']['USERS']['columns']
    index = table['columns'].index('FEATURE')
    assert schema['FEATURE']['dtype'] == 'float64'
    assert {row[index] for row in table['rows']} <= {
        -position - 0.5 for position in range(8)
    }
    assert -0.5 in {row[index] for row in table['rows']}
    json.dumps(payload)

    # Integral and within int64: int64, so an id stays an id.
    payload = _feature_payload([Decimal(index) for index in range(8)])
    schema = payload['schema']['related_tables']['USERS']['columns']
    assert schema['FEATURE']['dtype'] == 'int64'
    json.dumps(payload)

    # Integral but wider than int64 -- decimal(38, 0) permits 38 digits where
    # int64 holds 19. Named `string`, which keeps every digit, rather than
    # int64, which would declare a width the value does not have.
    # Built from digits rather than by arithmetic: Decimal addition applies the
    # context precision, which is 28 significant digits by default and would
    # round these before they ever reached the payload.
    wide = [
        Decimal(f'1234567890123456789012345678{index:02d}')
        for index in range(8)
    ]
    payload = _feature_payload(wide)
    table = payload['context']['related_tables']['USERS']
    schema = payload['schema']['related_tables']['USERS']['columns']
    index = table['columns'].index('FEATURE')
    assert schema['FEATURE']['dtype'] == 'string'
    # Subset, not equality: the context/predict split does not carry every row.
    # Every digit survives, and none of them arrive in scientific notation.
    emitted = {row[index] for row in table['rows']}
    assert emitted <= {str(value) for value in wide}
    assert '123456789012345678901234567800' in emitted
    json.dumps(payload)


# --- the guard that landed in only one of the two encoders -------------------


def test_a_uint64_past_int64_is_refused_rather_than_sent_as_int64():
    """`_dtype_name` answers int64 for uint64, so without a bounds check the
    payload declares a width the value does not have and sends it anyway.

    The tabular encoder grew this guard; this one did not, so the same column
    was exact on one path and wrong on the other.
    """
    import numpy as np
    from sdm_relational_engine.rfm.payload import _cell_json_value

    with pytest.raises(ValueError) as error:
        _cell_json_value(np.uint64(2**64 - 1), 'int64')
    assert 'outside the range' in str(error.value)
    assert 'strings' in str(error.value)


def test_an_int64_within_range_still_travels():
    from sdm_relational_engine.rfm.payload import _cell_json_value

    # Inside the JS-safe range it stays a number; past it the contract's
    # base-10-string rule applies, and both are still within int64.
    assert _cell_json_value(2**40, 'int64') == 2**40
    assert _cell_json_value(2**62, 'int64') == str(2**62)
    assert _cell_json_value(-(2**63), 'int64') == str(-(2**63))


def test_a_decimal_too_wide_for_a_float_is_refused_not_sent_as_inf():
    """`float(Decimal('1e400'))` is `inf`, which is not a rounding of the
    original in any useful sense and has no JSON form either.
    """
    from decimal import Decimal

    from sdm_relational_engine.rfm.payload import _cell_json_value

    with pytest.raises(ValueError) as error:
        _cell_json_value(Decimal('1e400'), 'float64')
    # Caught by the exactness guard now, which runs on the original value
    # before any conversion, so it names the remedy rather than the symptom.
    assert 'strings' in str(error.value)


def test_a_huge_fractional_decimal_is_refused_before_it_becomes_inf():
    """The conversion happens in `_json_value`, above the encoder's guard.

    A finite Decimal with a 310-digit integer part and a fraction reached
    `float()` there and came back `inf`, which has no JSON form and is not a
    rounding of the original in any useful sense.
    """
    from decimal import Decimal

    from sdm_relational_engine.rfm.payload import _json_value

    huge = Decimal('1' + '0' * 310 + '.1')
    assert huge.is_finite()
    with pytest.raises(ValueError) as error:
        _json_value(huge)
    assert 'too large' in str(error.value)


def test_an_ordinary_fractional_decimal_still_converts():
    from decimal import Decimal

    from sdm_relational_engine.rfm.payload import _json_value

    assert _json_value(Decimal('1.5')) == 1.5


def test_a_python_int_too_large_for_a_float_is_refused_not_raised_raw():
    """The Decimal path returns `inf`; the int path raises OverflowError."""
    from sdm_relational_engine.rfm.payload import _cell_json_value

    with pytest.raises(ValueError) as error:
        _cell_json_value(10**400, 'float64')
    # Refused by the exactness guard before the conversion is even attempted,
    # which is a better answer than catching the OverflowError afterwards.
    assert 'strings' in str(error.value)


def test_an_integer_a_float_cannot_hold_is_refused_not_rounded():
    """This encoder returned 9007199254740992.0 and sent it.

    A float column that silently renumbers an integer is the same defect on
    either encoder, and the wire cannot tell which one produced the cell. The
    matching assertion for the tabular encoder is in `sdm-client`'s
    `test_wire_integer_exactness.py`; the two cannot be checked together
    because neither package's test job installs the other.
    """
    from sdm_relational_engine.rfm.payload import _cell_json_value

    with pytest.raises(ValueError):
        _cell_json_value(2**53 + 1, 'float64')


def test_a_wide_integral_decimal_travels_as_digits_here_too():
    """`str(Decimal)` gives an exponent, and this branch exists to keep the
    digits. The tabular encoder was fixed; this one was not.
    """
    from decimal import Decimal

    from sdm_relational_engine.rfm.payload import _json_value

    sent = _json_value(Decimal('1e400'))
    assert 'E' not in sent and 'e' not in sent
    assert int(sent) == int(Decimal('1e400'))


def test_a_wide_whole_decimal_in_a_float_column_is_not_silently_rounded():
    """The exactness guard ran after `_json_value` had already normalised.

    A whole Decimal outside int64 becomes a *string* there, and a string
    carries no integer for the guard to check, so
    `Decimal('9223372036854775809.00')` reached `float()` and came back one
    short with no error at all.
    """
    from decimal import Decimal

    from sdm_relational_engine.rfm.payload import _cell_json_value

    value = Decimal('9223372036854775809.00')
    with pytest.raises(ValueError) as error:
        _cell_json_value(value, 'float64')
    assert 'strings' in str(error.value)


def test_an_ordinary_decimal_in_a_float_column_still_converts():
    from decimal import Decimal

    from sdm_relational_engine.rfm.payload import _cell_json_value

    assert _cell_json_value(Decimal('1.5'), 'float64') == 1.5
    assert _cell_json_value(42, 'float64') == 42.0


@pytest.mark.parametrize(
    ('value', 'survives'),
    [
        (2**53, True),
        (2**53 - 1, True),
        (-(2**53), True),
        (2**53 + 1, False),
        (-(2**53) - 1, False),
        (10**400, False),
    ],
)
def test_only_an_integer_a_float_cannot_hold_is_refused(value, survives):
    """The rule is a round trip, not the safe-integer ceiling.

    `2**53` is above `JSON_SAFE_INT_MAX` and is still exactly representable,
    so a ceiling comparison refused a value that travels perfectly well, and
    disagreed with the tabular encoder about it.
    """
    from sdm_relational_engine.rfm.payload import _cell_json_value

    if survives:
        assert _cell_json_value(value, 'float64') == float(value)
    else:
        with pytest.raises(ValueError):
            _cell_json_value(value, 'float64')
