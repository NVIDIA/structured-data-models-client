# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A NIM and a model in this process answer a request alike.

The same calls run against each backend. The model in this process always
runs, with small random-weight models standing in for the published ones, so
nothing is downloaded. A NIM runs too under ``-m live_nim``, when
``SDM_TABULAR_URL`` and ``SDM_RELATIONAL_URL`` point at one of each:

    SDM_TABULAR_URL=http://localhost:8000 \\
    SDM_RELATIONAL_URL=http://localhost:8001 pytest -m live_nim

The values differ, since the weights do. What both are held to is everything
else a caller reads: which rows come back under which keys, the columns, what
a session answers, and the code a refusal carries.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd
import pytest
from sample_tables import graph, people

from sdm_client import (
    KumoRelational,
    KumoTabular,
    PredictiveQuery,
    Request,
    Schema,
    SDMClient,
    StructuredDataError,
)

pytestmark = pytest.mark.filterwarnings('ignore')

URLS = {
    'kumo-tabular': os.environ.get('SDM_TABULAR_URL'),
    'kumo-relational': os.environ.get('SDM_RELATIONAL_URL'),
}

COUNT = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
BINARY = 'PREDICT COUNT(orders.*, 0, 30, days) > 2 FOR EACH users.user_id'
MULTICLASS = 'PREDICT users.tier FOR EACH users.user_id'


@pytest.fixture(
    params=['in-process', pytest.param('nim', marks=pytest.mark.live_nim)]
)
def client_for(request):
    r"""Opens a client for one model, on this backend."""
    if request.param == 'in-process':
        return lambda model: SDMClient.in_process(
            request.getfixturevalue(model.removeprefix('kumo-'))
        )

    def opened(model: str) -> SDMClient:
        if not URLS[model]:
            pytest.skip(f'point SDM_{model[5:].upper()}_URL at a {model} NIM')
        return SDMClient(url=URLS[model], timeout=300.0)

    return opened


def asked(target: str, task: str, start: int = 0, **options) -> Request:
    frame = people(start)[['person_id', 'age', 'segment', 'visits', target]]
    return Request(
        context=frame.iloc[:45],
        predict=frame.iloc[45:].drop(columns=[target]),
        target=target,
        task=task,
        **options,
    )


@pytest.mark.parametrize(
    ('target', 'task', 'outputs', 'columns'),
    [
        (
            'churn',
            'classification',
            None,
            ['prediction', 'probability_0', 'probability_1'],
        ),
        (
            'spend',
            'regression',
            ['prediction', 'quantiles'],
            ['prediction'] + [f'quantile_0.{n}' for n in range(1, 10)],
        ),
    ],
)
def test_a_table_is_answered_row_by_row_under_its_keys(
    client_for, target, task, outputs, columns
):
    request = asked(target, task)
    with client_for('kumo-tabular') as client:
        answer = client.predict(request, model=KumoTabular(), outputs=outputs)

    frame = answer.to_pandas()
    assert list(frame.columns) == ['row_index', 'id', *columns]
    assert frame['id'].tolist() == list(range(45, 60))


@pytest.mark.parametrize(
    ('query', 'columns'),
    [
        (COUNT, ['prediction']),
        (BINARY, ['prediction', 'probability_True', 'probability_False']),
        (
            MULTICLASS,
            [
                'prediction',
                'probability_bronze',
                'probability_silver',
                'probability_gold',
            ],
        ),
    ],
)
def test_an_entity_is_answered_under_its_key_and_a_session_agrees(
    client_for, query, columns
):
    request = PredictiveQuery(query, graph()).materialize(indices=[0, 1, 2, 3])
    with client_for('kumo-relational') as client:
        once = client.predict(request, model=KumoRelational()).to_pandas()
        with client.fit(request, model=KumoRelational()) as fitted:
            pinned = fitted.predict(request).to_pandas()

    assert list(once.columns) == ['row_index', 'id', *columns]
    assert once['id'].tolist() == [0, 1, 2, 3]
    pd.testing.assert_frame_equal(once, pinned)


def test_a_key_wider_than_a_double_comes_back_exactly(client_for):
    start = 2**53 + 1
    request = asked('churn', 'classification', start=start)
    with client_for('kumo-tabular') as client:
        answer = client.predict(request, model=KumoTabular())

    assert [item.id for item in answer.items] == list(
        range(start + 45, start + 60)
    )


def test_a_key_of_several_columns_answers_under_its_values(client_for):
    frame = people()[['person_id', 'age', 'visits', 'churn']].assign(
        tenant=['a', 'b'] * 30
    )
    schema = Schema.from_data(
        {'instance': frame}, primary_keys={'instance': ['tenant', 'person_id']}
    )
    request = Request(
        context=frame.iloc[:45],
        predict=frame.iloc[45:].drop(columns=['churn']),
        target='churn',
        task='classification',
        schema=schema,
    )
    with client_for('kumo-tabular') as client:
        answer = client.predict(request, model=KumoTabular())

    assert [item.id for item in answer.items] == [
        f'["{tenant}",{person}]'
        for tenant, person in zip(
            frame['tenant'][45:], frame['person_id'][45:], strict=True
        )
    ]


@pytest.mark.parametrize(
    ('inference', 'code'),
    [
        ({'use_prediction_time': 'no'}, None),
        ({'use_prediction_time': np.bool_(False)}, None),
        ({'use_prediction_time': None}, 'INVALID_REQUEST'),
        ({'run_mode': 7}, 'INVALID_REQUEST'),
        ({'inference_config': None}, 'INVALID_REQUEST'),
    ],
)
def test_an_inference_option_is_read_as_the_type_it_is(
    client_for, inference, code
):
    """A boolean spelled ``'no'`` asks for nothing, and a value that cannot
    be read as its type is refused before it is sent."""
    request = asked('churn', 'classification', inference=inference)
    with client_for('kumo-tabular') as client:
        if code is None:
            assert len(client.predict(request, model=KumoTabular()).items) == 15
            return
        with pytest.raises(StructuredDataError) as refused:
            client.predict(request, model=KumoTabular())

    assert refused.value.code == code


def test_a_fault_in_the_client_is_raised_as_itself(client_for, monkeypatch):
    """What the client makes of an answer is the same code on both backends,
    so a fault in it is raised as it is, not reported as the model's."""
    from sdm_client.adapters.tabular import TabularAdapter

    def broken(*args, **kwargs):
        raise RuntimeError('broken')

    monkeypatch.setattr(TabularAdapter, 'answered', broken)
    with client_for('kumo-tabular') as client:
        with pytest.raises(RuntimeError, match='broken'):
            client.predict(
                asked('churn', 'classification'), model=KumoTabular()
            )


def test_a_long_question_is_sent_within_the_contract_and_answered(client_for):
    from sdm_client.adapters.relational import RelationalAdapter

    query = f'{COUNT} WHERE ' + ' OR '.join(
        f'users.age > {i}' for i in range(60)
    )
    request = PredictiveQuery(query, graph()).materialize(indices=[0, 1])
    logged = request.payloads[0]['metadata']['source_query']
    sent = RelationalAdapter().prediction_plan(request, ('prediction',))
    assert len(logged) > 1024
    assert [p['metadata']['source_query'] for p in sent] == [logged[:1024]]
    with client_for('kumo-relational') as client:
        answer = client.predict(request, model=KumoRelational())

    assert [item.id for item in answer.items] == [0, 1]


def _a_table_named_like_the_scored_rows() -> dict[str, pd.DataFrame]:
    r"""``instance_table`` is the wire's own name for the rows scored."""
    tables = graph()
    tables['instance_table'] = tables.pop('items').rename(
        columns={'item_id': 'instance_table_id'}
    )
    tables['orders'] = tables['orders'].rename(
        columns={'item_id': 'instance_table_id'}
    )
    return tables


def _scored_without_its_key() -> Request:
    frame = people()[['person_id', 'age', 'visits', 'churn']].assign(
        tenant=['a', 'b'] * 30
    )
    schema = Schema.from_data(
        {'instance': frame}, primary_keys={'instance': ['tenant', 'person_id']}
    )
    return Request(
        context=frame.iloc[:45],
        predict=frame.iloc[45:][['age', 'visits']],
        target='churn',
        task='classification',
        schema=schema,
    )


def _a_session_asked_for_prediction_time(client: SDMClient) -> None:
    r"""A session is pinned with no inference options, so its context is
    fitted without prediction time, and a prediction cannot ask for it."""
    request = PredictiveQuery(BINARY, graph()).materialize(
        indices=[0, 1], use_prediction_time=True
    )
    with client.fit(request, model=KumoRelational()) as fitted:
        fitted.predict(request)


def _text_schema() -> Schema:
    schema = Schema.from_data(graph())
    schema['users']['tier'].stype = 'text'
    return schema


@pytest.mark.parametrize(
    ('model', 'ask', 'code'),
    [
        (
            'kumo-tabular',
            lambda client: client.predict(
                asked('churn', 'classification'),
                model=KumoTabular(),
                outputs=['prediction', 'quantiles'],
            ),
            'UNSUPPORTED_OUTPUT',
        ),
        (
            'kumo-relational',
            lambda client: client.predict(
                PredictiveQuery(COUNT, graph()).materialize(indices=[0]),
                model=KumoRelational(),
                outputs=['prediction', 'explanation'],
            ),
            'UNSUPPORTED_OUTPUT',
        ),
        (
            'kumo-relational',
            lambda client: client.predict(
                PredictiveQuery(
                    'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) '
                    'FOR EACH users.user_id',
                    graph(),
                ).materialize(indices=[0, 1]),
                model=KumoRelational(),
            ),
            'UNSUPPORTED_TASK',
        ),
        (
            'kumo-tabular',
            lambda client: client.predict(
                asked(
                    'churn',
                    'classification',
                    inference={'use_prediction_time': True},
                ),
                model=KumoTabular(),
            ),
            'INVALID_REQUEST',
        ),
        (
            'kumo-relational',
            lambda client: client.predict(
                PredictiveQuery(COUNT, graph()).materialize(
                    indices=[0, 1], inference_config={'num_estimators': 4}
                ),
                model=KumoRelational(),
            ),
            'INVALID_REQUEST',
        ),
        (
            'kumo-relational',
            lambda client: client.predict(
                PredictiveQuery(COUNT, _text_schema()).materialize(
                    indices=[0, 1]
                ),
                model=KumoRelational(),
            ),
            'UNSUPPORTED_DTYPE',
        ),
        (
            'kumo-relational',
            lambda client: client.predict(
                PredictiveQuery(
                    COUNT, _a_table_named_like_the_scored_rows()
                ).materialize(indices=[0, 1]),
                model=KumoRelational(),
            ),
            'INVALID_SCHEMA',
        ),
        (
            'kumo-tabular',
            lambda client: client.predict(
                _scored_without_its_key(), model=KumoTabular()
            ),
            'INVALID_SCHEMA',
        ),
        (
            'kumo-relational',
            _a_session_asked_for_prediction_time,
            'INVALID_REQUEST',
        ),
    ],
)
def test_a_request_the_model_cannot_answer_is_refused_with_the_same_code(
    client_for, model, ask, code
):
    with client_for(model) as client:
        with pytest.raises(StructuredDataError) as refused:
            ask(client)

    assert refused.value.code == code


def hand_built(change) -> Request:
    r"""A gathered request with its payload changed, as one passed to
    ``Request.materialized`` by hand may be."""
    gathered = PredictiveQuery(COUNT, graph()).materialize(indices=[0, 1])
    payload = copy.deepcopy(gathered.payloads[0])
    change(payload)
    return Request.materialized(
        payloads=(payload,),
        schema=gathered.schema,
        query=gathered.query,
        indices=gathered.indices,
    )


GHOST = {'format': 'arrays', 'columns': ['k'], 'rows': []}


def _with(table: dict, column: str, value: object) -> None:
    table['columns'].append(column)
    for row in table['rows']:
        row.append(value)


def _without(table: dict, column: str) -> None:
    position = table['columns'].index(column)
    del table['columns'][position]
    for row in table['rows']:
        del row[position]


def _undeclared(payload: dict, name: str, column: str) -> None:
    r"""``column`` taken out of table ``name`` everywhere: from its rows in
    both halves and from its schema."""

    def holding(part: dict) -> dict:
        return part if name == 'instance_table' else part['related_tables']

    for half in ('context', 'predict'):
        _without(holding(payload[half])[name], column)
    del holding(payload['schema'])[name]['columns'][column]


def _first(table: dict, column: str, value: object) -> None:
    table['rows'][0][table['columns'].index(column)] = value


def _counted(payload: dict, value: object) -> None:
    r"""``users.age`` declared ``int64``, with ``value`` in its first row."""
    payload['schema']['related_tables']['users']['columns']['age']['dtype'] = (
        'int64'
    )
    for half in ('context', 'predict'):
        users = payload[half]['related_tables']['users']
        position = users['columns'].index('age')
        for row in users['rows']:
            row[position] = int(row[position])
    _first(payload['context']['related_tables']['users'], 'age', value)


@pytest.mark.parametrize(
    ('change', 'code'),
    [
        (
            lambda p: p['task'].update(entity_table_names=['orders']),
            'TASK_SCHEMA_MISMATCH',
        ),
        (lambda p: p['task'].update(top_k=3), 'UNSUPPORTED_FEATURE'),
        (
            lambda p: p['task'].update(anchor_time_column='nowhere'),
            'TASK_SCHEMA_MISMATCH',
        ),
        (
            lambda p: p['task']['target'].update(dtype='timestamp[us]'),
            'TASK_SCHEMA_MISMATCH',
        ),
        (
            lambda p: p['predict']['related_tables'].pop('orders'),
            'INVALID_SCHEMA',
        ),
        (
            lambda p: p['predict']['related_tables'].update(ghost=GHOST),
            'INVALID_SCHEMA',
        ),
        (
            lambda p: [
                p[half]['related_tables'].update(ghost=GHOST)
                for half in ('context', 'predict')
            ],
            'INVALID_SCHEMA',
        ),
        (
            lambda p: _undeclared(p, 'instance_table', 'instance_id'),
            'INVALID_SCHEMA',
        ),
        (lambda p: _undeclared(p, 'users', 'instance_id'), 'INVALID_SCHEMA'),
        (
            lambda p: _without(p['predict']['related_tables']['orders'], 'ts'),
            'INVALID_SCHEMA',
        ),
        (
            lambda p: _without(p['context']['related_tables']['users'], 'age'),
            'INVALID_SCHEMA',
        ),
        (
            lambda p: _with(
                p['context']['related_tables']['users'], 'ghost', 1.0
            ),
            'INVALID_SCHEMA',
        ),
        (
            lambda p: _with(
                p['predict']['related_tables']['users'], 'ghost', 1.0
            ),
            'INVALID_SCHEMA',
        ),
        (
            lambda p: _with(
                p['context']['related_tables']['users'], 'age', 1.0
            ),
            'VALIDATION_FAILED',
        ),
        (
            lambda p: p['context']['related_tables']['users']['rows'][0].pop(),
            'INVALID_SCHEMA',
        ),
        (
            lambda p: p['context']['related_tables']['users']['rows'][0].append(
                1.0
            ),
            'INVALID_SCHEMA',
        ),
        (
            lambda p: _first(
                p['context']['related_tables']['orders'], 'user_id', 'abc'
            ),
            'VALIDATION_FAILED',
        ),
        (
            lambda p: _first(
                p['predict']['related_tables']['orders'], 'user_id', '9' * 30
            ),
            'VALIDATION_FAILED',
        ),
        (lambda p: _counted(p, '9' * 30), 'VALIDATION_FAILED'),
    ],
)
def test_a_request_built_by_hand_is_refused_where_it_states_another_question(
    client_for, change, code
):
    """The model scores the table its rows link to, from a timestamp the
    instance table declares, over the tables the context carries. A request
    saying otherwise is refused rather than answered as something else."""
    request = hand_built(change)
    with client_for('kumo-relational') as client:
        with pytest.raises(StructuredDataError) as refused:
            client.predict(request, model=KumoRelational())

    assert refused.value.code == code


@pytest.mark.parametrize(
    'value', [np.bool_(False), np.int64(0), np.int64(1), np.bool_(True)]
)
def test_a_numpy_boolean_built_by_hand_is_read_as_the_boolean_it_holds(
    client_for, value
):
    """A frame hands back NumPy scalars and a hand-built payload keeps them,
    where the tabular path is converted on its way out."""
    request = hand_built(
        lambda p: p.setdefault('inference', {}).update(
            use_prediction_time=value
        )
    )
    with client_for('kumo-relational') as client:
        assert len(client.predict(request, model=KumoRelational()).items) == 2
