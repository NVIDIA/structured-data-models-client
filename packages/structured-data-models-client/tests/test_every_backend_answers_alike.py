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

import os
import warnings

import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    KumoTabular,
    PredictiveQuery,
    Request,
    Schema,
    SDMClient,
    StructuredDataError,
)

URLS = {
    'kumo-tabular': os.environ.get('SDM_TABULAR_URL'),
    'kumo-relational': os.environ.get('SDM_RELATIONAL_URL'),
}

COUNT = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
BINARY = 'PREDICT COUNT(orders.*, 0, 30, days) > 2 FOR EACH users.user_id'
MULTICLASS = 'PREDICT users.tier FOR EACH users.user_id'


@pytest.fixture(scope='module')
def models():
    sdm = pytest.importorskip('sdm', reason='needs the in-process extra')
    import torch

    torch.manual_seed(0)
    return {
        'kumo-tabular': sdm.models.KumoTabular(
            size='small', pretrained=False, device='cpu'
        ),
        'kumo-relational': sdm.models.KumoRelational(
            pretrained=False, device='cpu'
        ),
    }


@pytest.fixture(
    params=['in-process', pytest.param('nim', marks=pytest.mark.live_nim)]
)
def client_for(request):
    r"""Opens a client for one model, on this backend."""
    if request.param == 'in-process':
        loaded = request.getfixturevalue('models')
        return lambda model: SDMClient.in_process(loaded[model])

    def opened(model: str) -> SDMClient:
        if not URLS[model]:
            pytest.skip(f'point SDM_{model[5:].upper()}_URL at a {model} NIM')
        return SDMClient(url=URLS[model], timeout=300.0)

    return opened


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def people(start: int = 0) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            'person_id': range(start, start + 60),
            'age': [20.0 + (i * 7) % 40 for i in range(60)],
            'segment': ['north' if i % 3 else 'south' for i in range(60)],
            'visits': [(i * 5) % 11 for i in range(60)],
        }
    )
    frame['churn'] = (frame['visits'] > 5).astype(int)
    frame['spend'] = frame['age'] * 2.5 + frame['visits']
    return frame


def graph() -> dict[str, pd.DataFrame]:
    users = pd.DataFrame(
        {
            'user_id': range(40),
            'tier': [('gold', 'silver', 'bronze')[i % 3] for i in range(40)],
            'age': [20.0 + (i * 7) % 30 for i in range(40)],
        }
    )
    items = pd.DataFrame(
        {'item_id': range(6), 'price': [5.0 + i for i in range(6)]}
    )
    rows = []
    for user in range(40):
        busy = user % 2 == 0
        for k in range(40 if busy else 4):
            rows.append(
                {
                    'order_id': len(rows),
                    'user_id': user,
                    'item_id': (user + k) % 6,
                    'ts': pd.Timestamp('2024-01-01')
                    + pd.Timedelta(days=(k * 5 if busy else k * 50) + user % 9),
                }
            )
    return {'users': users, 'items': items, 'orders': pd.DataFrame(rows)}


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


def test_a_question_longer_than_its_logged_copy_is_answered(client_for):
    query = f'{COUNT} WHERE ' + ' OR '.join(
        f'users.age > {i}' for i in range(60)
    )
    assert len(query) > 1024
    request = PredictiveQuery(query, graph()).materialize(indices=[0, 1])
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
    ],
)
def test_a_request_the_model_cannot_answer_is_refused_with_the_same_code(
    client_for, model, ask, code
):
    with client_for(model) as client:
        with pytest.raises(StructuredDataError) as refused:
            ask(client)

    assert refused.value.code == code
