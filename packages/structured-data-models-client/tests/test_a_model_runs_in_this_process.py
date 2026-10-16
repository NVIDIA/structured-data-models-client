# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The models run in this Python process, answering as their NIM does.

Small random-weight models stand in for the published ones, so nothing is
downloaded and no GPU is needed. What is checked is that a request reaches the
model as its own API would be called, and comes back as a NIM answers it.
"""

from __future__ import annotations

import copy
import sys
import warnings

import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    KumoTabular,
    MissingExtraError,
    NimRequestError,
    PredictiveQuery,
    Request,
    Schema,
    SDMClient,
    StructuredDataError,
)

COUNT = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
BINARY = 'PREDICT COUNT(orders.*, 0, 30, days) > 2 FOR EACH users.user_id'
MULTICLASS = 'PREDICT users.tier FOR EACH users.user_id'


@pytest.fixture(scope='module')
def sdm():
    return pytest.importorskip('sdm', reason='needs the in-process extra')


@pytest.fixture(scope='module')
def tabular(sdm):
    import torch

    torch.manual_seed(0)
    return sdm.models.KumoTabular(size='small', pretrained=False, device='cpu')


@pytest.fixture(scope='module')
def relational(sdm):
    pytest.importorskip('sdm_relational_engine', reason='needs [relational]')
    import torch

    torch.manual_seed(0)
    return sdm.models.KumoRelational(pretrained=False, device='cpu')


@pytest.fixture(autouse=True)
def _quiet():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


def people() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            'person_id': range(60),
            'age': [20.0 + (i * 7) % 40 for i in range(60)],
            'segment': ['north' if i % 3 else 'south' for i in range(60)],
            'visits': [(i * 5) % 11 for i in range(60)],
        }
    )
    frame['churn'] = (frame['visits'] > 5).astype(int)
    frame['plan'] = [
        'pro' if age > 40 else 'plus' if visits > 3 else 'free'
        for age, visits in zip(frame['age'], frame['visits'], strict=True)
    ]
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


def refused(call) -> NimRequestError:
    with pytest.raises(NimRequestError) as caught:
        call()
    return caught.value


@pytest.mark.parametrize(
    ('target', 'task'),
    [
        ('churn', 'classification'),
        ('plan', 'classification'),
        ('spend', 'regression'),
    ],
)
def test_a_tabular_answer_is_the_one_the_model_gives_when_called_directly(
    sdm, tabular, target, task
):
    import torch

    frame = people()[['person_id', 'age', 'segment', 'visits', target]]
    context, scored = frame.iloc[:45], frame.iloc[45:].drop(columns=[target])
    with SDMClient.in_process(tabular) as client:
        request = Request(
            context=context, predict=scored, target=target, task=task
        )
        answer = client.predict(request, model=KumoTabular()).to_pandas()

    features = {
        'age': 'numerical',
        'segment': 'categorical',
        'visits': 'numerical',
    }
    label = 'categorical' if task == 'classification' else 'numerical'
    fitted = copy.copy(tabular)
    fitted.fit(
        sdm.TableTensor.from_pandas(context, features),
        sdm.TableTensor.from_pandas(context, {target: label}),
        num_estimators=1,
        generator=torch.Generator().manual_seed(42),
    )
    output = fitted.predict(sdm.TableTensor.from_pandas(scored, features))
    expected = output.numerical.numpy()

    assert answer['id'].tolist() == scored['person_id'].tolist()
    if task == 'regression':
        assert answer['prediction'].tolist() == expected.mean(axis=1).tolist()
        return
    classes = output.columns[sdm.Stype.numerical]
    for position, name in enumerate(classes):
        assert answer[f'probability_{name}'].tolist() == (
            expected[:, position].tolist()
        )
    assert [str(value) for value in answer['prediction']] == [
        classes[position] for position in expected.argmax(axis=1)
    ]


@pytest.mark.parametrize('query', [COUNT, BINARY, MULTICLASS])
def test_a_session_answers_as_one_call_does(relational, query):
    request = PredictiveQuery(query, graph()).materialize(indices=[0, 1, 2, 3])
    with SDMClient.in_process(relational) as client:
        once = client.predict(request, model=KumoRelational()).to_pandas()
        again = client.predict(request, model=KumoRelational()).to_pandas()
        with client.fit(request, model=KumoRelational()) as fitted:
            pinned = fitted.predict(request).to_pandas()

    assert once['id'].tolist() == [0, 1, 2, 3]
    pd.testing.assert_frame_equal(once, again)
    pd.testing.assert_frame_equal(once, pinned)


@pytest.mark.parametrize(
    ('query', 'rows'),
    [
        (
            'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) RANK TOP 2 '
            'FOR EACH users.user_id',
            3,
        ),
        (
            'PREDICT COUNT(orders.*, 0, 30, days) FORECAST 3 TIMEFRAMES '
            'FOR EACH users.user_id',
            9,
        ),
    ],
)
def test_a_ranking_and_a_forecast_are_asked_as_the_questions_it_serves(
    relational, query, rows
):
    tables = graph()
    asked = PredictiveQuery(query, tables)
    request = (
        asked.materialize(
            indices=[0, 1, 2], candidates=tables['items']['item_id']
        )
        if 'RANK' in query
        else asked.materialize(indices=[0, 1, 2])
    )
    with SDMClient.in_process(relational) as client:
        answer = client.predict(request, model=KumoRelational()).to_pandas()

    assert len(answer) == rows
    if 'RANK' in query:
        assert all(len(ranked) <= 2 for ranked in answer['prediction'])
    else:
        assert sorted(set(answer['forecast_step'])) == [1, 2, 3]


def test_each_refusal_is_the_one_a_nim_answers_with(relational, tabular):
    tables = graph()
    request = PredictiveQuery(COUNT, tables).materialize(indices=[0, 1])
    one = PredictiveQuery(BINARY, tables).materialize(indices=[0])
    texty = Schema.from_data(tables)
    texty['users']['tier'].stype = 'text'
    with SDMClient.in_process(relational, tabular) as client:
        cases = {
            ('UNSUPPORTED_OUTPUT', 422): lambda: client.predict(
                request,
                model=KumoRelational(),
                outputs=['prediction', 'scores'],
            ),
            ('UNSUPPORTED_OUTPUT', 422, 'explanation'): lambda: client.predict(
                one,
                model=KumoRelational(),
                outputs=['prediction', 'explanation'],
            ),
            ('UNSUPPORTED_DTYPE', 422, 'tier'): lambda: client.predict(
                PredictiveQuery(COUNT, texty).materialize(indices=[0, 1]),
                model=KumoRelational(),
            ),
            ('INVALID_REQUEST', 400): lambda: client.predict(
                PredictiveQuery(COUNT, tables).materialize(
                    indices=[0, 1], inference_config={'num_estimators': 4}
                ),
                model=KumoRelational(),
            ),
            ('UNSUPPORTED_TASK', 422): lambda: client.predict(
                PredictiveQuery(
                    'PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) '
                    'FOR EACH users.user_id',
                    tables,
                ).materialize(indices=[0, 1]),
                model=KumoRelational(),
            ),
            ('UNSUPPORTED_DTYPE', 422, 'seen'): lambda: client.predict(
                Request(
                    context=people()
                    .assign(seen=pd.Timestamp('2024-01-01'))
                    .iloc[:40][['person_id', 'age', 'seen', 'churn']],
                    predict=people()
                    .assign(seen=pd.Timestamp('2024-01-01'))
                    .iloc[40:][['person_id', 'age', 'seen']],
                    target='churn',
                    task='classification',
                ),
                model=KumoTabular(),
            ),
        }
        for (code, status, *named), call in cases.items():
            error = refused(call)
            assert (error.code, error.status_code) == (code, status), named
            assert all(name in error.message for name in named)


def test_a_model_passed_in_keeps_what_was_fitted_on_it(sdm, tabular):
    frame = people()
    features = {'age': 'numerical', 'visits': 'numerical'}
    tabular.fit(
        sdm.TableTensor.from_pandas(frame.iloc[:30], features),
        sdm.TableTensor.from_pandas(frame.iloc[:30], {'churn': 'categorical'}),
    )
    query = sdm.TableTensor.from_pandas(frame.iloc[30:], features)
    before = tabular.predict(query).numerical.clone()
    try:
        with SDMClient.in_process(tabular) as client:
            request = Request(
                context=frame.iloc[:45][['person_id', 'age', 'plan']],
                predict=frame.iloc[45:][['person_id', 'age']],
                target='plan',
                task='classification',
            )
            client.predict(request, model=KumoTabular())
            with client.fit(request, model=KumoTabular()) as fitted:
                fitted.predict(request)
        assert tabular.predict(query).numerical.equal(before)
    finally:
        tabular.clear()


def test_a_model_is_loaded_the_first_time_it_is_asked_for(
    sdm, tabular, monkeypatch
):
    import torch

    loads = []

    def load(**kwargs):
        loads.append(kwargs)
        return tabular

    monkeypatch.setattr(sdm.models, 'KumoTabular', load)
    frame = people()
    request = Request(
        context=frame.iloc[:45][['person_id', 'age', 'churn']],
        predict=frame.iloc[45:][['person_id', 'age']],
        target='churn',
        task='classification',
    )
    with SDMClient.in_process() as client:
        assert client.url == 'in-process'
        assert client.models() == ['kumo-tabular', 'kumo-relational']
        assert loads == []
        client.predict(request, model=KumoTabular())
        client.predict(request, model=KumoTabular())

    expected = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert loads == [{'device': expected}]


def test_a_model_that_will_not_load_says_so_with_its_cause(sdm, monkeypatch):
    def load(**kwargs):
        raise OSError('no route to the hub')

    monkeypatch.setattr(sdm.models, 'KumoTabular', load)
    frame = people()
    request = Request(
        context=frame.iloc[:45][['person_id', 'age', 'churn']],
        predict=frame.iloc[45:][['person_id', 'age']],
        target='churn',
        task='classification',
    )
    with SDMClient.in_process() as client:
        error = refused(lambda: client.predict(request, model=KumoTabular()))

    assert (error.code, error.status_code) == ('MODEL_NOT_READY', 503)
    assert isinstance(error.__cause__, OSError)


def test_what_cannot_be_run_in_process_is_refused_when_opened(sdm, tabular):
    other = sdm.models.TabICLv2(pretrained=False, device='cpu')
    for models, device in (
        ((other,), None),
        ((tabular, tabular), None),
        ((tabular,), 'cpu'),
        ((), 'no-such-device'),
    ):
        with pytest.raises(StructuredDataError) as caught:
            SDMClient.in_process(*models, device=device)
        assert caught.value.code == 'INVALID_CONFIGURATION'


def test_a_model_left_in_training_mode_is_named_not_run(tabular):
    frame = people()
    request = Request(
        context=frame.iloc[:45][['person_id', 'age', 'churn']],
        predict=frame.iloc[45:][['person_id', 'age']],
        target='churn',
        task='classification',
    )
    tabular.train()
    try:
        with SDMClient.in_process(tabular) as client:
            with pytest.raises(StructuredDataError) as caught:
                client.predict(request, model=KumoTabular())
        assert caught.value.code == 'INVALID_CONFIGURATION'
        assert 'eval()' in caught.value.message
    finally:
        tabular.eval()


def test_without_the_extra_the_extra_is_named(monkeypatch):
    monkeypatch.setitem(sys.modules, 'sdm', None)
    with pytest.raises(MissingExtraError) as caught:
        SDMClient.in_process()
    assert caught.value.details['extra'] == 'in-process'
