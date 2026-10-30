# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Asking ``kumo-tabular`` for an explanation.

The endpoint explains every scored row, and the client hands each row's
explanation back under the key naming that row. A query that starts with
`EXPLAIN` is answered as if `outputs` named ``'explanation'``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable

import pandas as pd
import pytest
from sample_tables import people

from sdm_client import (
    KumoTabular,
    PredictiveQuery,
    Request,
    Schema,
    SDMClient,
)
from sdm_client.adapters.tabular import TabularAdapter
from sdm_client.base import check_request_against_model
from sdm_client.core.backend import NimBackend
from sdm_client.errors import StructuredDataError
from sdm_client.models import KUMO_TABULAR

EXPLAIN = ('prediction', 'explanation')
CLASSES = [
    {'kind': 'probability', 'class': '0'},
    {'kind': 'probability', 'class': '1'},
]
CHURN = 'PREDICT people.churn FOR EACH people.person_id'
#: What a classification asks for when the caller names nothing.
DEFAULTS = ['prediction', 'probabilities']


def response(**record) -> dict:
    """A one-record ``kumo-tabular`` response carrying ``record``'s fields."""
    return {
        'id': 'r',
        'model': 'kumo-tabular',
        'predictions': [{'row_index': 0, 'prediction': 1, **record}],
        'metadata': {},
    }


@pytest.mark.parametrize('task', ['classification', 'regression'])
def test_allowed(tabular_request, task):
    """kumo-tabular accepts explanation as an output on both of its tasks.

    Assumptions:
    - The request scores one row.
    """
    check_request_against_model(
        tabular_request(task=task), KUMO_TABULAR, EXPLAIN
    )


def test_one_row_sent(tabular_request):
    """An explanation over one row is sent with explanation in its fields.

    Assumptions:
    - The request scores exactly one row.
    """
    (envelope,) = TabularAdapter().prediction_plan(tabular_request(), EXPLAIN)
    assert envelope['output']['fields'] == list(EXPLAIN)


def test_rows_sent(tabular_request):
    """An explanation over two rows is sent with explanation in its fields.

    Assumptions:
    - The endpoint explains every scored row.
    """
    (envelope,) = TabularAdapter().prediction_plan(
        tabular_request(rows=2), EXPLAIN
    )
    assert envelope['output']['fields'] == list(EXPLAIN)


def test_pin_sends_no_rows(tabular_request):
    """Pinning a session context for explanations sends no predict rows."""
    pinned = TabularAdapter().session_plan(tabular_request(rows=2), EXPLAIN)
    assert 'predict' not in pinned


def test_session_rows_sent(tabular_request):
    """Session scoring sends an explanation over two rows.

    Assumptions:
    - The endpoint explains every scored row.
    """
    (envelope,) = TabularAdapter().session_prediction_plan(
        tabular_request(rows=2), EXPLAIN
    )
    assert envelope['output']['fields'] == list(EXPLAIN)


def test_result_carries_explanation(gradient_explanation):
    """The row's explanation reaches the result under its row_index.

    Assumptions:
    - The endpoint returns one record, with no id, and its explanation.
    """
    explanation = gradient_explanation(
        outputs=[{'kind': 'prediction', 'statistic': 'mean'}],
        row=0,
        columns=['x'],
        gradients=[[0.5]],
    )
    result = TabularAdapter().normalize(
        response(explanation=explanation), EXPLAIN
    )
    assert result.explanations.gradients == {
        0: {
            'outputs': [{'kind': 'prediction', 'statistic': 'mean'}],
            'instance': {'columns': ['x'], 'gradients': [[0.5]]},
        }
    }


def test_result_without_explanation():
    """A response without an explanation leaves the result's explanations
    empty.

    Assumptions:
    - The record carries no explanation field.
    """
    result = TabularAdapter().normalize(response(), ('prediction',))
    assert result.explanations.gradients == {}


def answered(request, gradient_explanation, *, keyed=True, outputs=None):
    """Send ``request`` through the adapter and read back a mocked answer.

    Each scored row ``i`` comes back as a NIM answers it: under ``'[<key>]'``
    when the request has a key, with probabilities, and with its own
    explanation, whose gradients are ``-(i + 1)`` and ``i + 1`` per class.
    """
    adapter = TabularAdapter()
    (envelope,) = adapter.prediction_plan(request, EXPLAIN)
    table = envelope['predict']['instance_table']
    records = []
    for index, row in enumerate(table['rows']):
        record = {
            'row_index': index,
            'prediction': 1,
            'probabilities': {'0': 0.25, '1': 0.75},
            'explanation': gradient_explanation(
                outputs=outputs or CLASSES,
                row=index,
                columns=['age'],
                gradients=[[-(index + 1.0)], [index + 1.0]],
            ),
        }
        if keyed:
            record['id'] = f'[{row[table["columns"].index("row_id")]}]'
        records.append(record)
    return adapter.normalize(
        {
            'id': 'r',
            'model': 'kumo-tabular',
            'predictions': records,
            'metadata': {},
        },
        EXPLAIN,
        envelope,
    )


def test_every_row_is_explained_under_its_id(
    tabular_request, gradient_explanation
):
    """Each scored row's explanation is under that row's id, as the frame
    names it.

    Assumptions:
    - The request's rows carry the key ``row_id`` (90 and 91).
    - The NIM answers each row under ``'[<key>]'`` with its own explanation.
    """
    result = answered(tabular_request(rows=2), gradient_explanation)

    assert sorted(result.explanations.gradients) == [90, 91]
    assert sorted(result.explanations.gradients) == sorted(
        result.to_pandas()['id']
    )


def test_each_row_keeps_its_own_gradients(
    tabular_request, gradient_explanation
):
    """A row's instance gradients are its own, as ``[output][column]``.

    Assumptions:
    - The NIM gives row ``i`` the gradients ``-(i + 1)`` and ``i + 1``.
    """
    result = answered(tabular_request(rows=2), gradient_explanation)

    assert result.explanations.gradients[90]['instance'] == {
        'columns': ['age'],
        'gradients': [[-1.0], [1.0]],
    }
    assert result.explanations.gradients[91]['instance']['gradients'] == [
        [-2.0],
        [2.0],
    ]


def test_rows_without_a_key_are_explained_under_their_row_index(
    tabular_request, gradient_explanation
):
    """With no key column, each explanation is under the row's row_index.

    Assumptions:
    - The request has no ``row_id`` column, so the NIM sends no id.
    """
    result = answered(
        tabular_request(rows=3, keyed=False), gradient_explanation, keyed=False
    )

    assert sorted(result.explanations.gradients) == [0, 1, 2]
    assert result.explanations.gradients[2]['instance']['gradients'] == [
        [-3.0],
        [3.0],
    ]


def test_rows_whose_ids_repeat_are_explained_under_their_row_index(
    tabular_request, gradient_explanation
):
    """Two rows sharing an id are both explained, each under its row_index.

    Assumptions:
    - The request's two rows both carry ``row_id`` 90.
    - The NIM answers each row under ``'[90]'`` with its own explanation.
    """
    built = tabular_request(rows=2)
    request = Request(
        context=built.context.instance_table,
        predict=built.predict.instance_table.assign(row_id=[90, 90]),
        target='churn',
        task='classification',
    )

    result = answered(request, gradient_explanation)

    assert sorted(result.explanations.gradients) == [0, 1]
    assert result.explanations.gradients[1]['instance']['gradients'] == [
        [-2.0],
        [2.0],
    ]
    assert (
        list(result.explanations.gradients.to_pandas().columns)[0]
        == 'row_index'
    )


def test_a_tabular_entry_has_no_related_tables(
    tabular_request, gradient_explanation
):
    """kumo-tabular has one table, so an entry carries no ``related`` key.

    Assumptions:
    - The NIM sends an empty ``related_tables``, as it always does here.
    """
    result = answered(tabular_request(rows=2), gradient_explanation)

    for entry in result.explanations.gradients.values():
        assert set(entry) == {'outputs', 'instance'}


def test_outputs_and_columns_are_passed_through(
    tabular_request, gradient_explanation
):
    """``outputs`` and ``columns`` are kept as the NIM returned them.

    Assumptions:
    - The NIM names a quantile output with a float level and an extra key,
      which the client does not read or rewrite.
    """
    outputs = [
        {'kind': 'prediction', 'statistic': 'median'},
        {'kind': 'quantile', 'level': 0.1, 'note': 'kept'},
    ]
    result = answered(
        tabular_request(rows=1), gradient_explanation, outputs=outputs
    )

    assert result.explanations.gradients[90]['outputs'] == outputs
    assert result.explanations.gradients[90]['instance']['columns'] == ['age']


def test_the_flat_table_has_a_line_per_row_output_and_column(
    tabular_request, gradient_explanation
):
    """``to_pandas()`` gives one line per (row, output, column), keyed as the
    frame is, each on the row's own block at hop 0.

    Assumptions:
    - Two rows, two class outputs and one column: four lines.
    """
    result = answered(tabular_request(rows=2), gradient_explanation)

    flat = result.explanations.gradients.to_pandas()

    assert list(flat.columns) == [
        'id',
        'table',
        'hop',
        'key',
        'row',
        'output',
        'column',
        'gradient',
    ]
    assert len(flat) == 4
    assert set(flat['table']) == {'instance'}
    assert set(flat['hop']) == {0}
    assert list(flat['key']) == list(flat['id'])
    assert flat.loc[flat['id'] == 91, 'gradient'].tolist() == [-2.0, 2.0]


def test_the_flat_table_names_a_keyless_row_by_its_row_index(
    tabular_request, gradient_explanation
):
    """With no key, ``to_pandas()`` names a row by ``row_index``, and its
    ``key`` is empty.

    Assumptions:
    - The request has no key column.
    """
    result = answered(
        tabular_request(rows=2, keyed=False), gradient_explanation, keyed=False
    )

    flat = result.explanations.gradients.to_pandas()

    assert list(flat.columns)[0] == 'row_index'
    assert sorted(set(flat['row_index'])) == [0, 1]
    assert flat['key'].isna().all()


@pytest.mark.parametrize(
    ('task', 'outputs', 'returned'),
    [
        (
            'classification',
            CLASSES,
            {'probabilities': {'0': 0.25, '1': 0.75}},
        ),
        (
            'regression',
            [
                {'kind': 'prediction', 'statistic': 'mean'},
                {'kind': 'quantile', 'level': 0.1},
                {'kind': 'quantile', 'level': 0.9},
            ],
            {'quantiles': {'0.1': 1.0, '0.9': 3.0}},
        ),
    ],
    ids=['probabilities', 'quantiles'],
)
def test_the_flat_table_names_each_output_as_the_frame_does(
    tabular_request, gradient_explanation, task, outputs, returned
):
    """Each explained score is named by the column of ``result.to_pandas()``
    that holds it, so a gradient can be read beside its score.

    Assumptions:
    - The NIM returns the scores it explains: every class probability, or the
      prediction and each asked quantile keyed by its level as text.
    """
    adapter = TabularAdapter()
    (envelope,) = adapter.prediction_plan(tabular_request(task=task), EXPLAIN)
    result = adapter.normalize(
        response(
            id='[90]',
            explanation=gradient_explanation(
                outputs=outputs,
                row=0,
                columns=['age'],
                gradients=[[0.1]] * len(outputs),
            ),
            **returned,
        ),
        EXPLAIN,
        envelope,
    )

    named = result.explanations.gradients.to_pandas()['output']

    assert len(set(named)) == len(outputs)
    assert set(named) <= set(result.to_pandas().columns)


@pytest.mark.parametrize(
    ('classes', 'spelled', 'columns'),
    [
        (
            (False, True),
            ('false', 'true'),
            {'probability_False', 'probability_True'},
        ),
        ((0, 1), ('0', '1'), {'probability_0', 'probability_1'}),
        (('no', 'yes'), ('no', 'yes'), {'probability_no', 'probability_yes'}),
    ],
    ids=['boolean', 'integer', 'string'],
)
def test_each_class_is_named_by_the_frames_column_for_it(
    tabular_request, gradient_explanation, classes, spelled, columns
):
    """An explained class probability is named exactly as the result frame
    names that class's column, whatever the target's type.

    Assumptions:
    - The target holds ``classes``; the NIM spells them ``spelled`` both as
      the keys of ``probabilities`` and as each explained output's ``class``,
      as it does live (a boolean as lowercase ``'true'``/``'false'``).
    """
    adapter = TabularAdapter()
    (envelope,) = adapter.prediction_plan(
        tabular_request(classes=classes), EXPLAIN
    )
    outputs = [{'kind': 'probability', 'class': name} for name in spelled]
    result = adapter.normalize(
        response(
            id='[90]',
            probabilities=dict(zip(spelled, (0.25, 0.75), strict=True)),
            explanation=gradient_explanation(
                outputs=outputs,
                row=0,
                columns=['age'],
                gradients=[[0.1], [0.9]],
            ),
        ),
        EXPLAIN,
        envelope,
    )

    named = set(result.explanations.gradients.to_pandas()['output'])

    assert named == columns
    assert named <= set(result.to_pandas().columns)


def test_a_regressions_outputs_keep_their_names(
    tabular_request, gradient_explanation
):
    """The point prediction and each quantile are named ``prediction`` and
    ``quantile_<level>``, as the frame names them.

    Assumptions:
    - A regression explained by its mean and two quantile levels, which the
      NIM returns keyed by level as text.
    """
    adapter = TabularAdapter()
    (envelope,) = adapter.prediction_plan(
        tabular_request(task='regression'), EXPLAIN
    )
    outputs = [
        {'kind': 'prediction', 'statistic': 'mean'},
        {'kind': 'quantile', 'level': 0.1},
        {'kind': 'quantile', 'level': 0.9},
    ]
    result = adapter.normalize(
        response(
            id='[90]',
            quantiles={'0.1': 1.0, '0.9': 3.0},
            explanation=gradient_explanation(
                outputs=outputs,
                row=0,
                columns=['age'],
                gradients=[[0.1]] * 3,
            ),
        ),
        EXPLAIN,
        envelope,
    )

    named = set(result.explanations.gradients.to_pandas()['output'])

    assert named == {'prediction', 'quantile_0.1', 'quantile_0.9'}


def test_a_boolean_class_keeps_its_spelling_in_outputs(
    tabular_request, gradient_explanation
):
    """Naming a boolean class as the frame does leaves the entry's
    ``outputs`` as the NIM returned them.

    Assumptions:
    - A boolean target, explained under the NIM's lowercase class names.
    """
    adapter = TabularAdapter()
    (envelope,) = adapter.prediction_plan(
        tabular_request(classes=(False, True)), EXPLAIN
    )
    outputs = [
        {'kind': 'probability', 'class': 'false'},
        {'kind': 'probability', 'class': 'true'},
    ]
    result = adapter.normalize(
        response(
            id='[90]',
            probabilities={'false': 0.25, 'true': 0.75},
            explanation=gradient_explanation(
                outputs=outputs,
                row=0,
                columns=['age'],
                gradients=[[0.1], [0.9]],
            ),
        ),
        EXPLAIN,
        envelope,
    )

    assert result.explanations.gradients[90]['outputs'] == outputs


def test_the_flat_table_of_no_explanation_is_empty_with_its_columns(
    tabular_request,
):
    """With nothing explained, ``to_pandas()`` is an empty frame that still
    has its columns.

    Assumptions:
    - The response carries no explanation.
    """
    result = TabularAdapter().normalize(response(), ('prediction',))

    flat = result.explanations.gradients.to_pandas()

    assert isinstance(flat, pd.DataFrame)
    assert flat.empty
    assert 'gradient' in flat.columns


# --- a query asked with `EXPLAIN` -------------------------------------------


def tabular_asked(text: str) -> Request:
    """``text`` over ``people``, scoring persons 50 and 51."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'people': people()}, primary_keys={'people': 'person_id'}
        )
    return PredictiveQuery(text, schema).materialize(
        indices=[50, 51], model=KumoTabular(), task='classification'
    )


class TabularEndpoint:
    """Answers ``kumo-tabular`` calls, explaining each row when asked;
    records every scoring payload."""

    def __init__(self, gradient_explanation: Callable[..., dict]) -> None:
        self.build = gradient_explanation
        self.sent: list[dict] = []

    def predict(self, payload: dict) -> dict:
        self.sent.append(payload)
        explain = 'explanation' in payload['output']['fields']
        records = []
        for index, _ in enumerate(payload['predict']['instance_table']['rows']):
            record = {
                'row_index': index,
                'prediction': 1,
                'probabilities': {'0': 0.4, '1': 0.6},
            }
            if explain:
                record['explanation'] = self.build(
                    outputs=CLASSES,
                    row=index,
                    columns=['age'],
                    gradients=[[-1.0], [1.0]],
                )
            records.append(record)
        return {
            'id': 'r',
            'model': 'kumo-tabular',
            'predictions': records,
            'metadata': {},
        }

    def create_session(self, payload: dict) -> dict:
        return {'session_id': 's1'}

    def session_predict(self, session_id: str, payload: dict) -> dict:
        return self.predict(payload)

    def delete_session(self, session_id: str) -> None:
        return None


@pytest.fixture
def tabular_endpoint(gradient_explanation) -> tuple[SDMClient, TabularEndpoint]:
    client = SDMClient(url='http://localhost:8000')
    endpoint = TabularEndpoint(gradient_explanation)
    client._backend = NimBackend(endpoint)
    return client, endpoint


def test_tabular_asks_for_explanation(tabular_endpoint, fields_sent):
    """An `EXPLAIN` query materialized for tabular asks for the defaults and
    ``'explanation'``, and its answer carries one per row.

    Assumptions:
    - Two persons scored; the endpoint explains when asked.
    """
    client, endpoint = tabular_endpoint

    result = client.predict(
        tabular_asked(f'EXPLAIN {CHURN}'), model=KumoTabular()
    )

    assert fields_sent(endpoint.sent) == [[*DEFAULTS, 'explanation']]
    assert len(result.explanations.gradients) == len(result) == 2


@pytest.mark.parametrize(
    ('outputs', 'expected'),
    [
        (None, [*DEFAULTS, 'explanation']),
        (['prediction'], ['prediction', 'explanation']),
        (['explanation'], ['explanation']),
        (['prediction', 'explanation'], ['prediction', 'explanation']),
    ],
    ids=['defaults', 'appended', 'already named', 'named among others'],
)
def test_explain_adds_explanation(
    tabular_endpoint, fields_sent, outputs, expected
):
    """`EXPLAIN` adds ``'explanation'`` to the defaults or the caller's list,
    once.

    Assumptions:
    - A tabular classification, whose defaults are prediction and
      probabilities.
    """
    client, endpoint = tabular_endpoint

    client.predict(
        tabular_asked(f'EXPLAIN {CHURN}'), model=KumoTabular(), outputs=outputs
    )

    assert fields_sent(endpoint.sent) == [expected]


@pytest.mark.parametrize(
    ('outputs', 'expected'),
    [(None, DEFAULTS), (['prediction'], ['prediction'])],
    ids=['defaults', 'named'],
)
def test_no_prefix_changes_nothing(
    tabular_endpoint, fields_sent, outputs, expected
):
    """Without `EXPLAIN` the call asks for exactly what it did before.

    Assumptions:
    - A tabular classification, whose defaults are prediction and
      probabilities.
    """
    client, endpoint = tabular_endpoint

    client.predict(tabular_asked(CHURN), model=KumoTabular(), outputs=outputs)

    assert fields_sent(endpoint.sent) == [expected]


def test_empty_outputs_still_refused(tabular_endpoint):
    """`EXPLAIN` does not turn ``outputs=[]`` into a request; it is refused
    as before, and nothing is sent.

    Assumptions:
    - A tabular classification.
    """
    client, endpoint = tabular_endpoint

    with pytest.raises(StructuredDataError, match='at least one field'):
        client.predict(
            tabular_asked(f'EXPLAIN {CHURN}'), model=KumoTabular(), outputs=[]
        )
    assert endpoint.sent == []


def test_session_explains_when_asked(tabular_endpoint, fields_sent):
    """A session fitted without `EXPLAIN` accepts the same question asked
    with it, and that call asks for ``'explanation'``.

    Assumptions:
    - The session pinned the plain question with the default outputs.
    """
    client, endpoint = tabular_endpoint
    fitted = client.fit(tabular_asked(CHURN), model=KumoTabular())

    result = fitted.predict(tabular_asked(f'EXPLAIN {CHURN}'))

    assert fields_sent(endpoint.sent) == [[*DEFAULTS, 'explanation']]
    assert len(result.explanations.gradients) == 2


def test_session_keeps_its_question(tabular_endpoint, fields_sent):
    """After an `EXPLAIN` call, a plain call on the same session is accepted
    and asks for the defaults only.

    Assumptions:
    - The session pinned the plain question with the default outputs.
    """
    client, endpoint = tabular_endpoint
    fitted = client.fit(tabular_asked(CHURN), model=KumoTabular())
    fitted.predict(tabular_asked(f'EXPLAIN {CHURN}'))

    fitted.predict(tabular_asked(CHURN))

    assert fields_sent(endpoint.sent)[-1] == DEFAULTS


def test_explain_fit_pins_explanation(tabular_endpoint, fields_sent):
    """A session fitted with `EXPLAIN` explains a later plain call.

    Assumptions:
    - No outputs were named at fit.
    """
    client, endpoint = tabular_endpoint
    fitted = client.fit(tabular_asked(f'EXPLAIN {CHURN}'), model=KumoTabular())

    result = fitted.predict(tabular_asked(CHURN))

    assert fields_sent(endpoint.sent) == [[*DEFAULTS, 'explanation']]
    assert len(result.explanations.gradients) == 2


@pytest.mark.parametrize('prefix', ['', 'EXPLAIN '], ids=['plain', 'explain'])
def test_session_refuses_another_question(tabular_endpoint, prefix):
    """A different question is refused, `EXPLAIN` or not, before it is
    sent.

    Assumptions:
    - The session pinned the churn question; the other asks about segment.
    """
    client, endpoint = tabular_endpoint
    fitted = client.fit(tabular_asked(CHURN), model=KumoTabular())
    other = tabular_asked(
        f'{prefix}PREDICT people.segment FOR EACH people.person_id'
    )

    with pytest.raises(StructuredDataError, match='pinned a different'):
        fitted.predict(other)
    assert endpoint.sent == []
