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

import pytest
from sample_tables import people

from sdm_client import (
    KumoTabular,
    PredictiveQuery,
    Request,
    Schema,
    SDMClient,
)
from sdm_client.errors import StructuredDataError

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


def answered(request, gradient_explanation, *, keyed=True, outputs=None):
    """Send ``request`` through the adapter and read back a mocked answer.

    Each scored row ``i`` comes back as a NIM answers it: under ``'[<key>]'``
    when the request has a key, with probabilities, and with its own
    explanation, whose gradients are ``-(i + 1)`` and ``i + 1`` per class.
    """
    import json

    from test_tabular_direct_core import Endpoint, client

    class ExplainedEndpoint(Endpoint):
        def predict_body(self, body):
            payload = json.loads(body)
            self.calls.append(payload)
            table = payload['predict']['instance_table']
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
            return json.dumps(
                {
                    'id': 'r',
                    'model': 'kumo-tabular',
                    'metadata': {},
                    'predictions': records,
                }
            ).encode()

    with client(ExplainedEndpoint()) as model_client:
        return model_client.predict(
            request, model=KumoTabular(), outputs=EXPLAIN
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

    assert sorted(result.explanations) == [90, 91]
    assert sorted(result.explanations) == sorted(result.to_pandas()['id'])


def test_each_row_keeps_its_own_gradients(
    tabular_request, gradient_explanation
):
    """A row's instance gradients are its own, as ``[output][column]``.

    Assumptions:
    - The NIM gives row ``i`` the gradients ``-(i + 1)`` and ``i + 1``.
    """
    result = answered(tabular_request(rows=2), gradient_explanation)

    assert result.explanations[90]['instance'] == {
        'columns': ['age'],
        'gradients': [[-1.0], [1.0]],
    }
    assert result.explanations[91]['instance']['gradients'] == [[-2.0], [2.0]]


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

    assert sorted(result.explanations) == [0, 1, 2]
    assert result.explanations[2]['instance']['gradients'] == [[-3.0], [3.0]]


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

    assert sorted(result.explanations) == [0, 1]
    assert result.explanations[1]['instance']['gradients'] == [[-2.0], [2.0]]
    assert list(result.explanations.to_pandas().columns)[0] == 'row_index'


def test_a_tabular_entry_has_no_related_tables(
    tabular_request, gradient_explanation
):
    """kumo-tabular has one table, so an entry carries no ``related`` key.

    Assumptions:
    - The NIM sends an empty ``related_tables``, as it always does here.
    """
    result = answered(tabular_request(rows=2), gradient_explanation)

    for entry in result.explanations.values():
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
        tabular_request(rows=1, task='regression'),
        gradient_explanation,
        outputs=outputs,
    )

    assert result.explanations[90]['outputs'] == outputs
    assert result.explanations[90]['instance']['columns'] == ['age']


def test_the_flat_table_has_a_line_per_row_output_and_column(
    tabular_request, gradient_explanation
):
    """``to_pandas()`` gives one line per (row, output, column), keyed as the
    frame is, each on the row's own block at hop 0.

    Assumptions:
    - Two rows, two class outputs and one column: four lines.
    """
    result = answered(tabular_request(rows=2), gradient_explanation)

    flat = result.explanations.to_pandas()

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

    flat = result.explanations.to_pandas()

    assert list(flat.columns)[0] == 'row_index'
    assert sorted(set(flat['row_index'])) == [0, 1]
    assert flat['key'].isna().all()


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
    url = 'http://localhost'
    serves_sessions = True

    def close(self):
        pass

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
    client = SDMClient.__new__(SDMClient)
    endpoint = TabularEndpoint(gradient_explanation)
    client._start(endpoint)
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
    assert len(result.explanations) == len(result) == 2


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
    assert len(result.explanations) == 2


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
    assert len(result.explanations) == 2


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


def test_empty_explanation_view_retains_columns():
    from sdm_client import PredictionResult

    flat = PredictionResult(
        items=(), request_id='r', model='kumo-tabular', outputs=('prediction',)
    ).explanations.to_pandas()
    assert flat.empty
    assert 'gradient' in flat.columns
