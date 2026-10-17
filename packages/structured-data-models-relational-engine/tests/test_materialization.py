# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from typing import Any

import pandas as pd
import pytest
from predictive_query.schema.task import TaskType
from pytest_mock import MockerFixture
from sdm_relational_engine.rfm import (
    Explanation,
    Graph,
    GraphSanitizationReport,
    KumoRelational,
    SanitizationStatus,
    TaskReferenceError,
    TaskTable,
)
from sdm_relational_engine.rfm.inference.payload import (
    ENTITY_REFERENCE_PREFIX,
    INSTANCE_ID,
    MAX_TABLE_ROWS,
    validate_payload_table_rows,
)
from sdm_relational_engine.rfm.inference.requests import RFMPredictResponse


class NoNetworkAPI:
    def predict(self, request: dict[str, Any]) -> RFMPredictResponse:
        raise AssertionError('materialization must not contact the API')


class RecordingAPI:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.entity_ids: list[tuple[Any, ...]] = []
        self.instance_ids: list[tuple[Any, ...]] = []

    def predict(
        self,
        request: dict[str, Any],
        *,
        entity_ids: Sequence[Any],
        instance_ids: Sequence[Any],
        anchor_times: Sequence[Any] | None = None,
    ) -> RFMPredictResponse:
        self.requests.append(request)
        self.entity_ids.append(tuple(entity_ids))
        self.instance_ids.append(tuple(instance_ids))
        return RFMPredictResponse(
            prediction={
                'columns': ['ENTITY', 'PREDICTION'],
                'data': [[entity_id, 0.5] for entity_id in entity_ids],
            }
        )


class ExplanationRecordingAPI:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []

    def predict(
        self,
        request: dict[str, Any],
        *,
        entity_ids: Sequence[Any],
        instance_ids: Sequence[Any],
        anchor_times: Sequence[Any] | None = None,
    ) -> RFMPredictResponse:
        self.requests.append(request)
        return RFMPredictResponse(
            prediction={
                'columns': ['ENTITY', 'PREDICTION', 'EXPLANATION'],
                'data': [
                    [
                        entity_ids[0],
                        0.5,
                        {
                            'format': 'natural_language_summary',
                            'summary': '',
                        },
                    ]
                ],
            }
        )


def _task() -> TaskTable:
    return TaskTable(
        task_type=TaskType.REGRESSION,
        context_df=pd.DataFrame(
            {
                'ENTITY': [0, 1],
                'TARGET': [1.0, 2.0],
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': [0, 1, 2, 3],
                'TARGET': [10.0, 11.0, 12.0, 13.0],
            }
        ),
        entity_table_name='USERS',
        entity_column='ENTITY',
        target_column='TARGET',
    )


def test_materialize_task_matches_live_requests(
    user_store_graph: Graph,
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv('KUMO_RELATIONAL_DISABLE_SESSIONS', '1')
    model = KumoRelational(user_store_graph, verbose=False)
    task = _task()
    context_before = task._context_df.copy()
    prediction_before = task._pred_df.copy()
    model._client = NoNetworkAPI()  # type: ignore
    assert model.sanitization_report.status.value == 'available'

    with model.batch_mode(batch_size=3):
        materialized = model.materialize_task(
            task,
            num_neighbors=[2, 1],
            use_prediction_time=True,
            verbose=False,
        )
        materialized_again = model.materialize_task(
            task,
            num_neighbors=[2, 1],
            use_prediction_time=True,
            verbose=False,
        )

        recorder = RecordingAPI()
        model._client = recorder  # type: ignore
        live_result = model.predict_task(
            task,
            num_neighbors=[2, 1],
            use_prediction_time=True,
            verbose=False,
        )

    assert [
        (request.batch_index, request.prediction_start, request.prediction_stop)
        for request in materialized
    ] == [(0, 0, 3), (1, 3, 4)]
    assert materialized == materialized_again
    assert [request.payload for request in materialized] == recorder.requests
    assert recorder.entity_ids == [(0, 1, 2), (3,)]
    assert live_result['ENTITY'].tolist() == [0, 1, 2, 3]
    assert all(request.request_size_bytes > 0 for request in materialized)
    assert all(
        'TARGET' not in request.payload['predict']['instance_table']['columns']
        for request in materialized
    )
    expected_instance_ids: list[tuple[Any, ...]] = []
    for request in materialized:
        payload = request.payload
        row_count = len(payload['predict']['instance_table']['rows'])
        assert row_count == request.prediction_stop - request.prediction_start
        instance_table = payload['predict']['instance_table']
        columns = instance_table['columns']
        root_relationship = next(
            relationship
            for relationship in payload['schema']['relationships']
            if (
                'source_table' not in relationship
                and relationship['target_table'] == 'USERS'
            )
        )
        entity_ref_column = root_relationship['source_columns'][0]
        assert entity_ref_column.startswith(ENTITY_REFERENCE_PREFIX)
        instance_id_index = columns.index(INSTANCE_ID)
        entity_ref_index = columns.index(entity_ref_column)
        related_table = payload['predict']['related_tables']['USERS']
        related_columns = related_table['columns']
        related_instance_index = related_columns.index(INSTANCE_ID)
        related_entity_index = related_columns.index(
            root_relationship['target_columns'][0]
        )
        occurrences = {
            (row[related_instance_index], row[related_entity_index])
            for row in related_table['rows']
        }
        expected_entities = (
            task._pred_df['ENTITY']
            .iloc[request.prediction_start : request.prediction_stop]
            .tolist()
        )
        request_instance_ids = tuple(
            row[instance_id_index] for row in instance_table['rows']
        )
        expected_instance_ids.append(request_instance_ids)
        assert len(request_instance_ids) == len(set(request_instance_ids))
        for row, expected_entity in zip(
            instance_table['rows'], expected_entities
        ):
            assert row[entity_ref_index] == expected_entity
            assert (
                row[instance_id_index],
                expected_entity,
            ) in occurrences
    assert recorder.instance_ids == expected_instance_ids
    pd.testing.assert_frame_equal(task._context_df, context_before)
    pd.testing.assert_frame_equal(task._pred_df, prediction_before)


def test_materialization_preserves_opaque_colliding_feature_names(
    user_store_graph: Graph,
) -> None:
    internal_looking_feature = f'{ENTITY_REFERENCE_PREFIX}_0'
    task = TaskTable(
        task_type=TaskType.REGRESSION,
        context_df=pd.DataFrame(
            {
                'USER_KEY': [0, 1],
                'TARGET': [1.0, 2.0],
                'ENTITY': [100, 101],
                'TARGET_PRED': ['context-a', 'context-b'],
                'q_0.5': [0.1, 0.2],
                internal_looking_feature: ['feature-a', 'feature-b'],
            }
        ),
        pred_df=pd.DataFrame(
            {
                'USER_KEY': [2],
                'TARGET': [3.0],
                'ENTITY': [102],
                'TARGET_PRED': ['predict'],
                'q_0.5': [0.3],
                internal_looking_feature: ['feature-ref'],
            }
        ),
        entity_table_name='USERS',
        entity_column='USER_KEY',
        target_column='TARGET',
    )

    payload = (
        KumoRelational(
            user_store_graph,
            verbose=False,
        )
        .materialize_task(task, verbose=False)[0]
        .payload
    )
    instance_table = payload['predict']['instance_table']
    columns = instance_table['columns']
    root_relationship = next(
        relationship
        for relationship in payload['schema']['relationships']
        if (
            'source_table' not in relationship
            and relationship['target_table'] == 'USERS'
        )
    )
    entity_ref_column = root_relationship['source_columns'][0]

    assert entity_ref_column.startswith(ENTITY_REFERENCE_PREFIX)
    assert entity_ref_column != internal_looking_feature
    for feature_name in (
        'ENTITY',
        'TARGET_PRED',
        'q_0.5',
        internal_looking_feature,
    ):
        assert feature_name in columns
    row = instance_table['rows'][0]
    assert row[columns.index(entity_ref_column)] == 2
    assert row[columns.index('ENTITY')] == 102
    assert row[columns.index('TARGET_PRED')] == 'predict'
    assert row[columns.index('q_0.5')] == 0.3
    assert row[columns.index(internal_looking_feature)] == 'feature-ref'


@pytest.mark.parametrize(
    ('task_type', 'targets'),
    [
        (TaskType.BINARY_CLASSIFICATION, [False, True]),
        (TaskType.MULTICLASS_CLASSIFICATION, ['red', 'blue']),
        (TaskType.REGRESSION, [1.0, 2.0]),
    ],
)
def test_materialization_covers_task_types_and_request_options(
    user_store_graph: Graph,
    task_type: TaskType,
    targets: list[Any],
) -> None:
    task = TaskTable(
        task_type=task_type,
        context_df=pd.DataFrame(
            {
                'ENTITY': [0, 1],
                'TARGET': targets,
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': [2],
                'TARGET': [targets[0]],
            }
        ),
        entity_table_name='USERS',
        entity_column='ENTITY',
        target_column='TARGET',
    )
    inference_config = (
        {'output_type': 'quantiles'}
        if task_type == TaskType.REGRESSION
        else None
    )

    payload = (
        KumoRelational(
            user_store_graph,
            verbose=False,
        )
        .materialize_task(
            task,
            return_embeddings=True,
            inference_config=inference_config,
            exclude_cols_dict={'USERS': ['AGE']},
            use_prediction_time=False,
            top_k=2,
            verbose=False,
        )[0]
        .payload
    )

    fields = set(payload['output']['fields'])
    assert payload['task']['kind'] == task_type.value
    assert payload['task']['top_k'] == 2
    assert ('prediction' in fields) is (task_type != TaskType.REGRESSION)
    assert 'embeddings' in fields
    assert ('probabilities' in fields) is task_type.is_classification
    assert ('quantiles' in fields) is (task_type == TaskType.REGRESSION)
    assert payload['inference']['use_prediction_time'] is False
    anchor_column = payload['task']['anchor_time_column']
    assert (
        payload['schema']['instance_table']['columns'][anchor_column][
            'nullable'
        ]
        is False
    )
    assert anchor_column in payload['context']['instance_table']['columns']
    assert anchor_column in payload['predict']['instance_table']['columns']
    assert 'TARGET' in payload['context']['instance_table']['columns']
    assert 'TARGET' not in payload['predict']['instance_table']['columns']
    assert 'AGE' not in payload['context']['related_tables']['USERS']['columns']
    assert 'AGE' not in payload['predict']['related_tables']['USERS']['columns']


def test_materialize_task_seed_controls_random_neighborhoods(
    monkeypatch: Any,
) -> None:
    monkeypatch.setenv('KUMO_RELATIONAL_DISABLE_SESSIONS', '1')
    graph = Graph.from_data(
        {
            'USERS': pd.DataFrame(
                {
                    'USER_ID': [0],
                    'VALUE': [1.0],
                }
            ),
            'ORDERS': pd.DataFrame(
                {
                    'ORDER_ID': range(40),
                    'USER_ID': [0] * 40,
                    'AMOUNT': range(40),
                }
            ),
        },
        verbose=False,
    )
    task = TaskTable(
        task_type=TaskType.REGRESSION,
        context_df=pd.DataFrame(
            {
                'ENTITY': [0],
                'TARGET': [1.0],
            }
        ),
        pred_df=pd.DataFrame(
            {
                'ENTITY': [0, 0, 0, 0],
                'TARGET': [2.0, 3.0, 4.0, 5.0],
            }
        ),
        entity_table_name='USERS',
        entity_column='ENTITY',
        target_column='TARGET',
    )
    model = KumoRelational(graph, verbose=False)

    with model.batch_mode(batch_size=2):
        first = model.materialize_task(
            task, num_neighbors=[3], random_seed=7, verbose=False
        )
        different = model.materialize_task(
            task, num_neighbors=[3], random_seed=8, verbose=False
        )
        repeated = model.materialize_task(
            task, num_neighbors=[3], random_seed=7, verbose=False
        )

        recorder = RecordingAPI()
        model._client = recorder  # type: ignore
        model.predict_task(
            task,
            num_neighbors=[3],
            random_seed=7,
            verbose=False,
        )

    assert len(first) == 2
    assert first == repeated
    assert first != different
    assert [request.payload for request in first] == recorder.requests


def test_materialize_task_rejects_prediction_row_count_mismatch(
    user_store_graph: Graph,
    mocker: MockerFixture,
) -> None:
    from sdm_relational_engine.rfm import rfm as rfm_module

    serialize = rfm_module.predict_request_to_json

    def serialize_with_missing_prediction_row(*args: Any, **kwargs: Any):
        payload = serialize(*args, **kwargs)
        payload['predict']['instance_table']['rows'].pop()
        return payload

    mocker.patch.object(
        rfm_module,
        'predict_request_to_json',
        side_effect=serialize_with_missing_prediction_row,
    )

    model = KumoRelational(user_store_graph, verbose=False)
    with pytest.raises(
        RuntimeError,
        match=(r'Request batch 0 serialized 3 prediction rows; expected 4'),
    ):
        model.materialize_task(_task(), verbose=False)


@pytest.mark.parametrize(
    ('batch_size', 'offers_fewer_entities'),
    [(4, True), (1, False)],
    ids=['a_batch_with_rows_to_lose', 'a_batch_of_one'],
)
def test_the_size_limit_offers_fewer_entities_only_where_there_are_some(
    user_store_graph: Graph,
    monkeypatch: Any,
    batch_size: int,
    offers_fewer_entities: bool,
) -> None:
    """A batch of one row cannot be split, so advising fewer entities per call
    names a remedy with nothing left to divide.
    """
    from sdm_relational_engine.rfm import rfm as rfm_module

    monkeypatch.setattr(rfm_module, '_MAX_SIZE', 1)
    model = KumoRelational(user_store_graph, verbose=False)
    with pytest.raises(ValueError) as refused:
        with model.batch_mode(batch_size=batch_size):
            model.materialize_task(_task(), verbose=False)
    said = str(refused.value)
    assert 'exceeds the 30MB limit' in said
    assert ('entities predicted at once' in said) is offers_fewer_entities
    assert 'tables in the graph' in said


def test_materialize_task_rejects_sanitized_entity_reference() -> None:
    graph = Graph.from_data(
        {
            'USERS': pd.DataFrame(
                {
                    'USER_ID': [0, 1],
                    'TIME': ['2025-01-01', None],
                }
            ),
        },
        verbose=False,
    )
    # One date orders nothing, so it is named rather than inferred.
    graph['USERS'].time_column = 'TIME'
    model = KumoRelational(graph, verbose=False)
    task = TaskTable(
        task_type=TaskType.REGRESSION,
        context_df=pd.DataFrame(
            {
                'ENTITY': [0],
                'TARGET': [1.0],
            }
        ),
        pred_df=pd.DataFrame({'ENTITY': [1]}),
        entity_table_name='USERS',
        entity_column='ENTITY',
        target_column='TARGET',
        time_column=TaskTable.ENTITY_TIME,
    )

    with pytest.raises(TaskReferenceError) as exc_info:
        model.materialize_task(task, verbose=False)

    assert exc_info.value.table_name == 'USERS'
    assert exc_info.value.unresolved_rows == 1
    assert 'USER_ID' not in str(exc_info.value)


def test_explain_dict_matches_live_request(
    user_store_graph: Graph,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    task = _task().narrow_prediction(0, 1)

    materialized = model.materialize_task(
        task,
        explain={},
        verbose=False,
    )
    recorder = ExplanationRecordingAPI()
    model._client = recorder  # type: ignore
    result = model.predict_task(task, explain={}, verbose=False)

    assert isinstance(result, Explanation)
    assert [request.payload for request in materialized] == recorder.requests
    assert 'explanation' in materialized[0].payload['output']['fields']


def test_all_references_are_validated_before_first_batch(
    user_store_graph: Graph,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    task = _task()
    task._pred_df.loc[3, 'ENTITY'] = 99
    recorder = RecordingAPI()
    model._client = recorder  # type: ignore

    with model.batch_mode(batch_size=3):
        with pytest.raises(TaskReferenceError) as exc_info:
            model.predict_task(task, verbose=False)

    assert exc_info.value.unresolved_rows == 1
    assert recorder.requests == []


def test_unknown_sanitization_is_distinct_from_zero_drops() -> None:
    report = GraphSanitizationReport.not_available()

    assert report.status == SanitizationStatus.NOT_AVAILABLE
    assert report.tables == {}


@pytest.mark.parametrize(
    'section',
    ['context', 'predict'],
)
def test_payload_row_limit_checks_instance_table_paths(
    section: str,
) -> None:
    empty_table = {'format': 'arrays', 'columns': [], 'rows': []}
    payload = {
        'context': {
            'instance_table': dict(empty_table),
            'related_tables': {},
        },
        'predict': {
            'instance_table': dict(empty_table),
            'related_tables': {},
        },
    }
    payload[section]['instance_table']['rows'] = [[]] * 3
    expected_path = f'{section}.instance_table'

    with pytest.raises(ValueError, match=expected_path):
        validate_payload_table_rows(payload, batch_index=7, limit=2)


@pytest.mark.parametrize(
    'section',
    ['context', 'predict'],
)
def test_payload_row_limit_allows_large_related_tables(section: str) -> None:
    empty_table = {'format': 'arrays', 'columns': [], 'rows': []}
    payload = {
        'context': {
            'instance_table': dict(empty_table),
            'related_tables': {},
        },
        'predict': {
            'instance_table': dict(empty_table),
            'related_tables': {},
        },
    }
    payload[section]['related_tables']['T'] = {
        **empty_table,
        'rows': [[]] * (MAX_TABLE_ROWS + 1),
    }

    validate_payload_table_rows(payload, batch_index=7)


def test_default_payload_row_limit_boundary() -> None:
    table = {'format': 'arrays', 'columns': [], 'rows': []}
    payload = {
        'context': {
            'instance_table': table,
            'related_tables': {},
        },
        'predict': {
            'instance_table': {
                'format': 'arrays',
                'columns': [],
                'rows': [[]] * 10_000,
            },
            'related_tables': {},
        },
    }

    validate_payload_table_rows(payload, batch_index=0)
    payload['predict']['instance_table']['rows'].append([])

    with pytest.raises(ValueError, match='10,001 rows'):
        validate_payload_table_rows(payload, batch_index=0)


@pytest.mark.parametrize('entry_point', ['materialize_task', 'predict_task'])
def test_explain_guard_is_identical_on_both_entry_points(
    user_store_graph: Graph,
    entry_point: str,
) -> None:
    # `materialize_task` and `predict_task` share one guard; this pins the two
    # to the same behaviour so a change to one cannot drift from the other.
    model = KumoRelational(user_store_graph, verbose=False)
    model._client = ExplanationRecordingAPI()  # type: ignore
    call = getattr(model, entry_point)

    with pytest.warns(UserWarning, match='only supported for run mode'):
        call(
            _task().narrow_prediction(0, 1),
            explain=True,
            run_mode='best',
            verbose=False,
        )

    with pytest.raises(ValueError, match='more than a single entity'):
        call(_task(), explain=True, verbose=False)


def test_explain_guard_warning_points_at_the_caller(
    user_store_graph: Graph,
) -> None:
    model = KumoRelational(user_store_graph, verbose=False)
    model._client = ExplanationRecordingAPI()  # type: ignore

    with pytest.warns(UserWarning) as record:
        model.predict_task(
            _task().narrow_prediction(0, 1),
            explain=True,
            run_mode='best',
            verbose=False,
        )

    assert record[0].filename == __file__
