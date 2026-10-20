# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A model run in this process, as structured-data-models runs it.

Small random-weight models stand in for the published ones, so nothing is
downloaded and no GPU is needed. What a NIM answers alike is held in
``test_every_backend_answers_alike``; this is what only a model in this
process has: the answer the model itself gives, how a model is loaded and
kept, and a refusal that carries no HTTP status, since nothing was sent.
"""

from __future__ import annotations

import copy
import subprocess
import sys
import warnings

import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    KumoTabular,
    NimRequestError,
    PredictiveQuery,
    Request,
    SDMClient,
    StructuredDataError,
)

COUNT = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
BINARY = 'PREDICT COUNT(orders.*, 0, 30, days) > 2 FOR EACH users.user_id'


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


def refused(call) -> StructuredDataError:
    with pytest.raises(StructuredDataError) as caught:
        call()
    assert not isinstance(caught.value, NimRequestError)
    assert not hasattr(caught.value, 'status_code')
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


def test_a_session_keeps_the_question_it_pinned(relational):
    """A request changed after it was pinned leaves the session alone, as a
    NIM's is: the model still scores the regression and the tables it pinned,
    and each answer is still read under the key's own type."""
    request = PredictiveQuery(COUNT, graph()).materialize(indices=[0, 1])
    with SDMClient.in_process(relational) as client:
        with client.fit(request, model=KumoRelational()) as fitted:
            pinned = request.payloads[0]
            pinned['task']['kind'] = 'classification'
            users = pinned['schema']['related_tables']['users']['columns']
            users['user_id']['dtype'] = 'bool'
            del users['age']
            answer = fitted.predict(request)

    assert [repr(item.id) for item in answer.items] == ['0', '1']


def test_what_a_holdout_leaves_is_refused_when_it_cannot_be_learned_from(
    relational,
):
    """A holdout edits the context after the client has checked it, so an
    empty context, or a binary one left a single class, is refused here."""
    tables = graph()
    with SDMClient.in_process(relational) as client:
        for request, holdout, code in (
            (
                PredictiveQuery(COUNT, tables).materialize(indices=[0, 1]),
                list(range(40)),
                'VALIDATION_FAILED',
            ),
            (
                PredictiveQuery(BINARY, tables).materialize(indices=[1, 3]),
                list(range(0, 40, 2)),
                'TASK_SCHEMA_MISMATCH',
            ),
        ):
            error = refused(
                lambda request=request, holdout=holdout: client.fit(
                    request, model=KumoRelational(), holdout=holdout
                )
            )
            assert error.code == code


@pytest.mark.parametrize(
    ('frame', 'code'),
    [
        (people()[['person_id', 'churn']], 'VALIDATION_FAILED'),
        (
            people().assign(seen=pd.Timestamp('2024-01-01'))[
                ['person_id', 'age', 'seen', 'churn']
            ],
            'UNSUPPORTED_DTYPE',
        ),
    ],
)
def test_a_table_the_model_cannot_read_is_refused(tabular, frame, code):
    request = Request(
        context=frame.iloc[:40],
        predict=frame.iloc[40:].drop(columns=['churn']),
        target='churn',
        task='classification',
    )
    with SDMClient.in_process(tabular) as client:
        error = refused(lambda: client.predict(request, model=KumoTabular()))
    assert error.code == code


def test_a_task_the_model_was_not_loaded_for_is_refused(sdm):
    regressor = sdm.models.KumoTabular(
        task='regression', size='small', pretrained=False, device='cpu'
    )
    frame = people()
    request = Request(
        context=frame.iloc[:45][['person_id', 'age', 'churn']],
        predict=frame.iloc[45:][['person_id', 'age']],
        target='churn',
        task='classification',
    )
    with SDMClient.in_process(regressor) as client:
        error = refused(lambda: client.predict(request, model=KumoTabular()))
    assert error.code == 'UNSUPPORTED_TASK'
    assert "['regression']" in error.message


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

    assert error.code == 'MODEL_NOT_READY'
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


def test_without_the_extra_the_extra_is_named():
    """Run in a fresh interpreter, so nothing the extra brings is already
    imported. On Python 3.10, where the extra installs nothing, the refusal
    says which Python it needs."""
    script = (
        'import sys\n'
        "for name in ('sdm', 'torch', 'pyarrow'):\n"
        '    sys.modules[name] = None\n'
        'from sdm_client import SDMClient, StructuredDataError\n'
        'try:\n'
        '    SDMClient.in_process()\n'
        'except StructuredDataError as error:\n'
        "    print(error.code, error.details['extra'], '3.11' in str(error))\n"
    )
    run = subprocess.run(
        [sys.executable, '-c', script],
        capture_output=True,
        text=True,
        check=False,
    )
    names_the_python = sys.version_info < (3, 11)
    assert run.stdout.split() == [
        'MISSING_EXTRA',
        'in-process',
        str(names_the_python),
    ], run.stderr
