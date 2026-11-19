# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading ``kumo-relational`` explanations back as the caller's rows.

The NIM names each related row by its position in the table as that call sent
it. These payloads are shaped as the engine gathers one (an instance table
carrying ``instance_id`` and the entity reference, related tables keyed by
``instance_id`` and their own key, or the made-up ``__node_id`` when a table
has none), and the answers as the NIM user guide gives them, so what is held
here is that each entity gets back its own rows, keys and gradients.

The graph per entity: ``users`` (the entity's own row), ``orders`` linked to
it, ``products`` linked to those orders, and ``events``, which has no key.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest
from sample_tables import graph

from sdm_client import (
    ColumnExplanations,
    KumoRelational,
    PredictiveQuery,
    Relationship,
    Request,
    Schema,
    SDMClient,
    SDMWarning,
)
from sdm_client.adapters.relational import RelationalAdapter
from sdm_client.core.backend import NimBackend
from sdm_client.errors import StructuredDataError, UnsupportedRequestError
from sdm_client.materializers.relational import RelationalMaterializer

EXPLAIN = ('prediction', 'probabilities', 'explanation')
CLASSES = [
    {'kind': 'probability', 'class': 'no'},
    {'kind': 'probability', 'class': 'yes'},
]
#: Past 2**53, so the contract sends it as base-10 text.
WIDE = 9_007_199_254_740_993

#: Each related table as the engine sends it: its columns, and the one
#: numerical column the mocked NIM explains it by.
TABLES = {
    'users': (['instance_id', 'user_id', 'tenure_days'], 'tenure_days'),
    'orders': (
        ['instance_id', 'order_id', 'user_id', 'product_id', 'amount'],
        'amount',
    ),
    'products': (['instance_id', 'product_id', 'price'], 'price'),
    'events': (
        ['instance_id', '__node_id', 'user_id', 'kind', 'seconds'],
        'seconds',
    ),
}

#: Each entity's rows, without ``instance_id``. Both users bought product 77,
#: so it is sampled once per entity; product 88 is sampled for user 5 and no
#: order of theirs names it.
ROWS = {
    5: {
        'users': [[5, 30.0]],
        'orders': [[9003, 5, 77, 12.0], [9004, 5, 77, 13.0]],
        'products': [[77, 4.0], [88, 6.0]],
        'events': [[5, 'login', 41], [5, 'login', 42]],
    },
    7: {
        'users': [[7, 70.0]],
        'orders': [[str(WIDE), 7, 77, 17.0]],
        'products': [[77, 4.5]],
        'events': [[7, 'logout', 71]],
    },
}

RELATIONSHIPS = [
    {
        'source_columns': ['__kumo_entity_ref_0'],
        'target_table': 'users',
        'target_columns': ['user_id'],
    },
    {
        'source_table': 'orders',
        'source_columns': ['user_id'],
        'target_table': 'users',
        'target_columns': ['user_id'],
    },
    {
        'source_table': 'orders',
        'source_columns': ['product_id'],
        'target_table': 'products',
        'target_columns': ['product_id'],
    },
    {
        'source_table': 'events',
        'source_columns': ['user_id'],
        'target_table': 'users',
        'target_columns': ['user_id'],
    },
]

DTYPES = {
    'instance_id': 'int64',
    'user_id': 'int64',
    'order_id': 'int64',
    'product_id': 'int64',
    '__node_id': 'int64',
    'kind': 'string',
    'seconds': 'int64',
}

PRIMARY_KEYS = {
    'users': ['instance_id', 'user_id'],
    'orders': ['instance_id', 'order_id'],
    'products': ['instance_id', 'product_id'],
    'events': ['instance_id', '__node_id'],
}


def gathered(*users: int, first: int = 100, task: str = 'classification'):
    """One call's payload, scoring ``users`` as instances ``first`` on.

    Related rows are interleaved across the entities, so a row's position is
    never its entity's: reading one by position in the wrong table, or the
    wrong call, gives another entity's row.
    """
    instances = {user: first + at for at, user in enumerate(users)}
    labels = [0.0, 1.0, 0.0] if task == 'regression' else ['no', 'yes', 'no']
    related: dict[str, dict[str, Any]] = {}
    for name, (columns, _) in TABLES.items():
        rows: list[list[Any]] = []
        longest = max(len(ROWS[user][name]) for user in users)
        for at in range(longest):
            for user in users:
                if at < len(ROWS[user][name]):
                    rows.append([instances[user], *ROWS[user][name][at]])
        if name == 'events':
            rows = [
                [row[0], position, *row[1:]]
                for position, row in enumerate(rows)
            ]
        related[name] = {'format': 'arrays', 'columns': columns, 'rows': rows}
    return {
        'task': {
            'kind': task,
            'target': {
                'column_name': 'TARGET',
                'dtype': 'string' if task != 'regression' else 'float64',
            },
        },
        'schema': {
            'instance_table': {
                'columns': {
                    'instance_id': {'dtype': 'int64'},
                    '__kumo_entity_ref_0': {'dtype': 'int64'},
                    'TARGET': {'dtype': 'string'},
                },
                'primary_key': 'instance_id',
            },
            'related_tables': {
                name: {
                    'columns': {
                        column: {'dtype': DTYPES.get(column, 'float64')}
                        for column in columns
                    },
                    'primary_key': PRIMARY_KEYS[name],
                }
                for name, (columns, _) in TABLES.items()
            },
            'relationships': RELATIONSHIPS,
        },
        'context': {
            'instance_table': {
                'format': 'arrays',
                'columns': ['instance_id', '__kumo_entity_ref_0', 'TARGET'],
                'rows': [
                    [at, at + 1, label] for at, label in enumerate(labels)
                ],
            },
        },
        'predict': {
            'instance_table': {
                'format': 'arrays',
                'columns': ['instance_id', '__kumo_entity_ref_0'],
                'rows': [[instances[user], user] for user in users],
            },
            'related_tables': related,
        },
    }


def nim_answer(payload: dict, gradient_explanation) -> dict:
    """What the NIM answers ``payload`` with, explanations included.

    Each record names its related rows by position in this call's tables,
    starting at 0, and explains each one by its numerical column ``v`` as
    ``[-v, v]`` per class. The instance block's gradient is the entity's
    ``row_index + 1``.
    """
    predict = payload['predict']
    records = []
    for index, (instance, _) in enumerate(predict['instance_table']['rows']):
        related = {}
        for name, (columns, explained) in TABLES.items():
            rows = predict['related_tables'][name]['rows']
            mine = [at for at, row in enumerate(rows) if row[0] == instance]
            values = [rows[at][columns.index(explained)] for at in mine]
            related[name] = (
                mine,
                [explained],
                [[[-value] for value in values], [[value] for value in values]],
            )
        records.append(
            {
                'id': str(instance),
                'row_index': index,
                'prediction': 'yes',
                'probabilities': {'no': 0.4, 'yes': 0.6},
                'explanation': gradient_explanation(
                    outputs=CLASSES,
                    row=index,
                    columns=['anchor_dayofweek'],
                    gradients=[[-(index + 1.0)], [index + 1.0]],
                    related=related,
                ),
            }
        )
    return {
        'id': 'r',
        'model': 'kumo-relational',
        'predictions': records,
        'metadata': {},
    }


@pytest.fixture
def explained(gradient_explanation):
    """Users 5 and 7, scored in one call and explained."""
    payload = gathered(5, 7)
    return RelationalAdapter().normalize(
        nim_answer(payload, gradient_explanation), EXPLAIN, payload
    )


class Transport:
    """Answers each call as the NIM would, recording what was sent."""

    def __init__(self, gradient_explanation) -> None:
        self.build = gradient_explanation
        self.sent: list[dict] = []
        self.pinned: dict = {}

    def predict(self, payload: dict) -> dict:
        self.sent.append(payload)
        return nim_answer(payload, self.build)

    def create_session(self, payload: dict) -> dict:
        self.pinned = payload
        return {'session_id': 's1'}

    def session_predict(self, session_id: str, payload: dict) -> dict:
        """Answered as a one-shot call over the pinned half and ``payload``."""
        return self.predict({**self.pinned, **payload})

    def delete_session(self, session_id: str) -> None:
        return None

    def close(self) -> None:
        return None


def client_answering(gradient_explanation) -> tuple[SDMClient, Transport]:
    client = SDMClient(url='http://localhost:8000')
    transport = Transport(gradient_explanation)
    client._backend = NimBackend(transport)
    return client, transport


def split(*calls: tuple[int, ...]) -> Request:
    """One request gathered as several calls, each numbered from 0."""
    return Request.materialized(
        payloads=tuple(gathered(*users) for users in calls),
        schema=Schema.from_data({'users': _users_frame()}),
        query='PREDICT users.churn FOR EACH users.user_id',
        indices=[user for users in calls for user in users],
        entity_table='users',
    )


def _users_frame():
    import pandas as pd

    return pd.DataFrame({'user_id': [5, 7], 'churn': ['no', 'yes']})


# --- each entity, under its own key -----------------------------------------


def test_every_entity_is_explained_under_its_id(explained):
    """Each scored entity's explanation is under its id, as the frame names it.

    Assumptions:
    - The call scores users 5 and 7, and the NIM explains both.
    """
    assert sorted(explained.explanations.gradients) == [5, 7]
    assert sorted(explained.explanations.gradients) == sorted(
        explained.to_pandas()['id']
    )


def test_the_engines_question_row_is_not_an_entrys_block(explained):
    """An entry has no ``instance``: that block explains the engine's question
    row, not a row the caller sent, and the entity's own row is in ``related``.

    Assumptions:
    - The NIM explains every record's ``instance_table`` as well.
    """
    for entry in explained.explanations.gradients.values():
        assert set(entry) == {'outputs', 'related'}


def test_an_entity_has_a_block_for_every_related_table(explained):
    """Every related table sent is in the entry, under its own name.

    Assumptions:
    - The NIM sends one block per related table for every record.
    """
    for entry in explained.explanations.gradients.values():
        assert set(entry['related']) == set(TABLES)


def test_related_rows_are_the_entitys_own_values_as_sent(explained):
    """A related row comes back as the values sent, less the engine's
    ``instance_id`` and ``__node_id``.

    Assumptions:
    - The rows of the two entities are interleaved in every table, so a
      position read as the wrong row names the other entity's.
    """
    related = explained.explanations.gradients[5]['related']

    assert related['users']['rows'] == [{'user_id': 5, 'tenure_days': 30.0}]
    assert related['orders']['rows'] == [
        {'order_id': 9003, 'user_id': 5, 'product_id': 77, 'amount': 12.0},
        {'order_id': 9004, 'user_id': 5, 'product_id': 77, 'amount': 13.0},
    ]
    assert related['events']['rows'] == [
        {'user_id': 5, 'kind': 'login', 'seconds': 41},
        {'user_id': 5, 'kind': 'login', 'seconds': 42},
    ]
    assert explained.explanations.gradients[7]['related']['products'][
        'rows'
    ] == [{'product_id': 77, 'price': 4.5}]


def test_a_keyed_table_gives_each_row_its_key_as_declared(explained):
    """A table with a key of its own gives each row that key, read back as its
    declared dtype, while ``rows`` keep the value as sent.

    Assumptions:
    - ``orders.order_id`` is declared int64, and user 7's order id is past
      2**53, so it travels as text.
    """
    orders_7 = explained.explanations.gradients[7]['related']['orders']

    assert explained.explanations.gradients[5]['related']['orders']['keys'] == [
        9003,
        9004,
    ]
    assert orders_7['keys'] == [WIDE]
    assert isinstance(orders_7['keys'][0], int)
    assert orders_7['rows'][0]['order_id'] == str(WIDE)
    assert explained.explanations.gradients[5]['related']['users']['keys'] == [
        5
    ]


def test_a_keyless_table_gives_rows_without_keys(explained):
    """A table keyed only by the engine's ``__node_id`` has no ``keys``.

    Assumptions:
    - ``events`` has no key of its own; the engine numbers its rows.
    """
    events = explained.explanations.gradients[5]['related']['events']

    assert 'keys' not in events
    assert all('__node_id' not in row for row in events['rows'])


def test_related_gradients_are_aligned_with_rows(explained):
    """A related row's gradients sit at its position in ``rows``, as
    ``[output][row][column]``.

    Assumptions:
    - The NIM explains each related row by its numerical column ``v`` as
      ``[-v, v]``, so a gradient names the row it belongs to.
    """
    for entry in explained.explanations.gradients.values():
        for name, block in entry['related'].items():
            column = TABLES[name][1]
            values = [row[column] for row in block['rows']]
            assert block['columns'] == [column]
            assert block['gradients'] == [
                [[-value] for value in values],
                [[value] for value in values],
            ]


def test_outputs_and_columns_are_passed_through(explained):
    """``outputs`` and every ``columns`` are kept as the NIM returned them.

    Assumptions:
    - The NIM names its outputs and preprocessed columns; the client does not
      read or rename them.
    """
    entry = explained.explanations.gradients[5]

    assert entry['outputs'] == CLASSES
    assert entry['related']['events']['columns'] == ['seconds']


def test_no_explanation_asked_for_leaves_explanations_empty(
    gradient_explanation,
):
    """Without ``'explanation'`` in the outputs, ``result.explanations`` is an
    empty dict.

    Assumptions:
    - The NIM answers without an explanation when none is asked for.
    """
    payload = gathered(5, 7)
    answer = nim_answer(payload, gradient_explanation)
    for record in answer['predictions']:
        del record['explanation']

    result = RelationalAdapter().normalize(answer, ('prediction',), payload)

    assert result.explanations.gradients == {}


# --- a request sent in several calls ----------------------------------------


def test_a_split_request_reads_each_call_against_its_own_rows(
    gradient_explanation,
):
    """A request sent in two calls, each numbering its rows from 0, gives
    every entity its own related rows and keys.

    Assumptions:
    - The request is gathered as two calls, user 5 then user 7, and both
      calls number their instances and related rows from the same start.
    - The NIM answers each call with positions in that call's tables.
    """
    client, transport = client_answering(gradient_explanation)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = client.predict(
            split((5,), (7,)), model=KumoRelational(), outputs=list(EXPLAIN)
        )

    assert len(transport.sent) == 2
    assert sorted(result.explanations.gradients) == [5, 7]
    seven = result.explanations.gradients[7]['related']
    assert seven['users']['keys'] == [7]
    assert seven['orders']['keys'] == [WIDE]
    assert seven['events']['rows'] == [
        {'user_id': 7, 'kind': 'logout', 'seconds': 71}
    ]
    assert result.explanations.gradients[5]['related']['orders']['keys'] == [
        9003,
        9004,
    ]


def test_a_split_request_whose_calls_repeat_an_id_keys_by_row_index(
    gradient_explanation,
):
    """When two calls each explain the same id, both entries are kept, each
    under its row_index across the whole request.

    Assumptions:
    - The request is gathered as two calls, each scoring user 5, so each
      call's ids are unique but repeat across the calls.
    - The client numbers the second call's row after the first call's one.
    """
    client, _ = client_answering(gradient_explanation)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = client.predict(
            split((5,), (5,)), model=KumoRelational(), outputs=list(EXPLAIN)
        )

    assert sorted(result.explanations.gradients) == [0, 1]
    assert result.explanations.gradients[1]['related']['users']['keys'] == [5]
    assert (
        list(result.explanations.gradients.to_pandas().columns)[0]
        == 'row_index'
    )


def test_a_split_request_keeps_row_index_keys_apart(gradient_explanation):
    """Explanations kept under ``row_index`` move with their rows when a later
    call's rows are numbered after the earlier ones.

    Assumptions:
    - Each call's answer has no entity key, so it is keyed by ``row_index``,
      and each call numbers its rows from 0.
    - The client numbers the second call's rows after the first call's two.
    """
    from sdm_client.adapters.tabular import TabularAdapter
    from sdm_client.client import _numbered_across_calls
    from sdm_client.result import PredictionResult

    def answer(payload: dict) -> PredictionResult:
        return TabularAdapter().normalize(
            {
                'id': 'r',
                'model': 'kumo-tabular',
                'predictions': [
                    {
                        'row_index': index,
                        'prediction': 1,
                        'explanation': gradient_explanation(
                            outputs=CLASSES,
                            row=index,
                            columns=['age'],
                            gradients=[[-row[0]], [row[0]]],
                        ),
                    }
                    for index, row in enumerate(
                        payload['predict']['instance_table']['rows']
                    )
                ],
                'metadata': {},
            },
            ('prediction', 'explanation'),
        )

    calls = [
        {'predict': {'instance_table': {'rows': [[10.0], [11.0]]}}},
        {'predict': {'instance_table': {'rows': [[12.0], [13.0]]}}},
    ]

    merged = PredictionResult.merge(_numbered_across_calls(calls, answer))

    assert sorted(merged.explanations.gradients) == [0, 1, 2, 3]
    assert merged.explanations.gradients[2]['instance']['gradients'] == [
        [-12.0],
        [12.0],
    ]
    assert sorted(merged.explanations.gradients) == list(
        merged.to_pandas()['row_index']
    )


def _entry(gradient: float) -> dict:
    """A decoded tabular entry: one regression output over ``age``."""
    return {
        'outputs': [{'kind': 'regression'}],
        'instance': {'columns': ['age'], 'gradients': [[gradient]]},
    }


def test_hand_built_parts_with_unique_ids_merge_under_their_ids(
    hand_built_result,
):
    """Hand-built parts whose ids differ merge keyed by id, as before.

    Assumptions:
    - Each part is built by hand with a plain-dict ``explanations``, so no
      row positions are recorded.
    - The parts explain users ``u5`` and ``u7``, at rows 0 and 1.
    """
    from sdm_client.result import PredictionResult

    merged = PredictionResult.merge(
        [
            hand_built_result([(0, 'u5')], {'u5': _entry(1.0)}),
            hand_built_result([(1, 'u7')], {'u7': _entry(2.0)}),
        ]
    )

    assert sorted(merged.explanations.gradients) == ['u5', 'u7']
    assert list(merged.explanations.gradients.to_pandas().columns)[0] == 'id'


def test_hand_built_parts_repeating_an_id_merge_by_row_index(
    hand_built_result,
):
    """Hand-built parts that both explain ``u5`` keep both entries, each under
    the ``row_index`` its row has in the merged frame.

    Assumptions:
    - Each part is built by hand with a plain-dict ``explanations``, so no
      row positions are recorded; the items carry them.
    - The first part's row is 0 and the second's is 1.
    """
    from sdm_client.result import PredictionResult

    merged = PredictionResult.merge(
        [
            hand_built_result([(0, 'u5')], {'u5': _entry(1.0)}),
            hand_built_result([(1, 'u5')], {'u5': _entry(2.0)}),
        ]
    )

    assert sorted(merged.explanations.gradients) == list(
        merged.to_pandas()['row_index']
    )
    assert sorted(merged.explanations.gradients) == [0, 1]
    assert merged.explanations.gradients[1]['instance']['gradients'] == [[2.0]]
    assert (
        list(merged.explanations.gradients.to_pandas().columns)[0]
        == 'row_index'
    )


def test_hand_built_forecast_parts_repeating_an_id_merge_by_row_and_step(
    hand_built_result,
):
    """Hand-built forecast parts that both explain ``('u5', 1)`` are keyed by
    ``(row_index, forecast_step)``, as a forecast from ``predict()`` is.

    Assumptions:
    - Each part is built by hand, keyed by ``(id, forecast_step)``, and its
      item carries ``forecast_step``.
    - The first part's row is 0 and the second's is 1, both at step 1.
    """
    from sdm_client.result import PredictionResult

    merged = PredictionResult.merge(
        [
            hand_built_result([(0, 'u5', 1)], {('u5', 1): _entry(1.0)}),
            hand_built_result([(1, 'u5', 1)], {('u5', 1): _entry(2.0)}),
        ]
    )

    assert sorted(merged.explanations.gradients) == [(0, 1), (1, 1)]
    flat = merged.explanations.gradients.to_pandas()
    assert list(flat.columns)[:2] == ['row_index', 'forecast_step']


@pytest.mark.parametrize('key', ['u9', ('u5', 9)])
def test_hand_built_parts_whose_rows_cannot_be_found_are_refused(
    hand_built_result, key
):
    """Repeated ids whose rows cannot be found from the items raise a
    ``StructuredDataError`` naming the remedy, not a ``KeyError``.

    Assumptions:
    - Both parts explain ``u5``, so the merge must key by row.
    - The second part's key matches none of its items: an id it does not
      score, or a step it does not carry.
    """
    from sdm_client.errors import StructuredDataError
    from sdm_client.result import PredictionResult

    parts = [
        hand_built_result([(0, 'u5')], {'u5': _entry(1.0)}),
        hand_built_result([(1, 'u5')], {'u5': _entry(2.0), key: _entry(3.0)}),
    ]

    with pytest.raises(StructuredDataError) as raised:
        PredictionResult.merge(parts)

    assert raised.value.code == 'INVALID_REQUEST'
    assert 'predict()' in str(raised.value)
    assert 'row_index' in str(raised.value)


# --- a forecast ---------------------------------------------------------------


@pytest.fixture
def forecast(monkeypatch, relational_endpoint):
    """A two-step forecast of users 5 and 7 through ``predict()``, each step a
    regression the endpoint answers and explains as the NIM would."""
    client, _ = relational_endpoint()
    request = relational_asked(
        monkeypatch, FORECAST.removeprefix('EXPLAIN '), kind='regression'
    )
    return client.predict(
        request, model=KumoRelational(), outputs=['prediction', 'explanation']
    )


def test_a_forecast_is_explained_under_id_and_step(forecast):
    """Each point of a forecast is explained under ``(id, forecast_step)``,
    and nothing is gathered under ``by_forecast_step``.

    Assumptions:
    - Two steps, built by the real ``materialize()`` with gathering stood
      in for, each answered with an explanation for users 5 and 7.
    """
    result = forecast

    assert sorted(result.explanations.gradients) == [
        (5, 1),
        (5, 2),
        (7, 1),
        (7, 2),
    ]
    assert 'by_forecast_step' not in result.explanations.gradients
    frame = result.to_pandas()
    assert sorted(result.explanations.gradients) == sorted(
        zip(frame['id'], frame['forecast_step'], strict=True)
    )


def test_a_forecasts_flat_table_names_the_step(forecast):
    """``to_pandas()`` of a forecast's explanations starts with ``id`` and
    ``forecast_step``.

    Assumptions:
    - Two steps, built by the real ``materialize()``, each explained.
    """
    flat = forecast.explanations.gradients.to_pandas()

    assert list(flat.columns)[:3] == ['id', 'forecast_step', 'table']
    assert sorted(set(flat['forecast_step'])) == [1, 2]


# --- a query asked with `EXPLAIN` -------------------------------------------

TIER = 'PREDICT users.tier FOR EACH users.user_id'
FORECAST = (
    'EXPLAIN PREDICT COUNT(orders.*, 0, 30, days) FORECAST 2 TIMEFRAMES '
    'FOR EACH users.user_id'
)
RANK = (
    'EXPLAIN PREDICT LIST_DISTINCT(orders.item_id, 0, 30, days) RANK TOP 2 '
    'FOR EACH users.user_id'
)
#: What a classification asks for when the caller names nothing.
DEFAULTS = ['prediction', 'probabilities']


def relational_asked(
    monkeypatch, text: str, *, kind: str = 'classification', **how: Any
) -> Request:
    """``text`` over the sample graph, each gather answered by users 5 and 7.

    ``labels``, when given, replaces the context's target column.
    """
    labels = how.pop('labels', None)

    def gather(self, query, indices, num_context_samples, options):
        payload = gathered(5, 7, task=kind)
        if labels is not None:
            rows = payload['context']['instance_table']['rows']
            for row, label in zip(rows, labels, strict=True):
                row[2] = label
        return (payload,)

    monkeypatch.setattr(RelationalMaterializer, '_gather', gather)
    return materialized(text, **how)


def materialized(text: str, **how: Any) -> Request:
    """``text`` materialized over the sample graph for users 5 and 7."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            graph(),
            primary_keys={
                'users': 'user_id',
                'items': 'item_id',
                'orders': 'order_id',
            },
            time_columns={'orders': 'ts'},
            relationships=[
                Relationship('orders', 'user_id', 'users', 'user_id'),
                Relationship('orders', 'item_id', 'items', 'item_id'),
            ],
        )
        return PredictiveQuery(text, schema).materialize(
            indices=[5, 7], model=KumoRelational(), **how
        )


class RelationalEndpoint(Transport):
    """``Transport``, answering a regression with a number, and leaving the
    explanation off the calls numbered in ``bare``."""

    def __init__(self, gradient_explanation, bare: frozenset[int]) -> None:
        super().__init__(gradient_explanation)
        self.bare = bare

    def predict(self, payload: dict) -> dict:
        answer = super().predict(payload)
        for record in answer['predictions']:
            if payload['task']['kind'] == 'regression':
                record['prediction'] = 1.0
                del record['probabilities']
            if len(self.sent) - 1 in self.bare:
                del record['explanation']
        return answer


@pytest.fixture
def relational_endpoint(
    gradient_explanation,
) -> Callable[..., tuple[SDMClient, RelationalEndpoint]]:
    """A client on a ``RelationalEndpoint``; ``bare`` as it takes."""

    def build(bare: frozenset[int] = frozenset()):
        client = SDMClient(url='http://localhost:8000')
        endpoint = RelationalEndpoint(gradient_explanation, bare)
        client._backend = NimBackend(endpoint)
        return client, endpoint

    return build


def test_relational_asks_for_explanation(
    monkeypatch, relational_endpoint, fields_sent
):
    """An `EXPLAIN` query materialized for relational asks for the defaults
    and ``'explanation'``, and its answer carries one per entity.

    Assumptions:
    - The engine's gathering is stood in for; users 5 and 7 are scored.
    """
    client, endpoint = relational_endpoint()

    result = client.predict(
        relational_asked(monkeypatch, f'EXPLAIN {TIER}'),
        model=KumoRelational(),
    )

    assert fields_sent(endpoint.sent) == [[*DEFAULTS, 'explanation']]
    assert sorted(result.explanations.gradients) == [5, 7]


@pytest.fixture
def gathers(monkeypatch) -> list[Any]:
    """Each query the engine is asked to gather, answered by users 5 and 7."""
    asked: list[Any] = []

    def gather(self, query, indices, num_context_samples, options):
        asked.append(query)
        return (gathered(5, 7),)

    monkeypatch.setattr(RelationalMaterializer, '_gather', gather)
    return asked


def test_ranking_refused_before_gathering(gathers):
    """An `EXPLAIN` ranking is refused by ``materialize()``, saying a ranking
    gives back no explanation, before any candidate's context is gathered.

    Assumptions:
    - Two candidates; gathering is stood in for by a recorder.
    """
    with pytest.raises(StructuredDataError) as refused:
        materialized(RANK, candidates=[0, 1])

    assert refused.value.code == 'INVALID_REQUEST'
    assert "'explanation' is not among them" in str(refused.value)
    assert gathers == []


def test_explain_without_ranking_is_gathered(gathers):
    """An `EXPLAIN` query that ranks nothing is gathered and keeps its text.

    Assumptions:
    - Gathering is stood in for by a recorder.
    """
    request = materialized(f'EXPLAIN {TIER}')

    assert len(gathers) == 1
    assert request.query == f'EXPLAIN {TIER}'


def test_forecast_explains_every_step(
    monkeypatch, relational_endpoint, fields_sent
):
    """Every step of an `EXPLAIN` forecast asks for ``'explanation'``, and
    each step's points are explained under ``(id, step)``.

    Assumptions:
    - Two steps, built by the real ``materialize()``; gathering is stood in
      for, each step a regression over users 5 and 7.
    """
    client, endpoint = relational_endpoint()

    result = client.predict(
        relational_asked(monkeypatch, FORECAST, kind='regression'),
        model=KumoRelational(),
    )

    assert fields_sent(endpoint.sent) == [['prediction', 'explanation']] * 2
    assert sorted(result.explanations.gradients) == [
        (5, 1),
        (5, 2),
        (7, 1),
        (7, 2),
    ]


def test_forecast_names_an_unexplained_step(monkeypatch, relational_endpoint):
    """An `EXPLAIN` forecast whose second step comes back unexplained says
    so, naming the step.

    Assumptions:
    - Two steps; the endpoint leaves the explanation off the second call.
    """
    client, _ = relational_endpoint(bare=frozenset({1}))
    request = relational_asked(monkeypatch, FORECAST, kind='regression')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        client.predict(request, model=KumoRelational())

    said = [str(item.message) for item in caught]
    assert any('step 2 carries none' in message for message in said), said


def test_a_forecast_keeps_every_steps_request_id(
    monkeypatch, gradient_explanation
):
    """A forecast's result records each step's request id under its step.

    Rules out the old behavior, where only the first step's id survived.

    Assumptions:
    - Three steps, built by the real ``materialize()``; gathering is stood
      in for, each step one call over users 5 and 7.
    - The endpoint stamps call ``n`` (from 0) with the id ``call-n``.
    """

    class Stamping(RelationalEndpoint):
        def predict(self, payload: dict) -> dict:
            answer = super().predict(payload)
            answer['id'] = f'call-{len(self.sent) - 1}'
            return answer

    client = SDMClient(url='http://localhost:8000')
    client._backend = NimBackend(Stamping(gradient_explanation, frozenset()))
    query = FORECAST.removeprefix('EXPLAIN ').replace(
        'FORECAST 2', 'FORECAST 3'
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = client.predict(
            relational_asked(monkeypatch, query, kind='regression'),
            model=KumoRelational(),
            outputs=['prediction'],
        )

    assert result.metadata['steps'] == [1, 2, 3]
    assert result.metadata['step_request_ids'] == [
        ['call-0'],
        ['call-1'],
        ['call-2'],
    ]
    assert result.request_id == 'call-0'


def test_one_class_refusal_names_the_column(monkeypatch, relational_endpoint):
    """A one-class context is refused naming the predicted column, not the
    wire's ``TARGET``, when the query starts with `EXPLAIN`.

    Assumptions:
    - Every context row is labelled ``'no'``; gathering is stood in for.
    """
    client, endpoint = relational_endpoint()
    request = relational_asked(
        monkeypatch, f'EXPLAIN {TIER}', labels=['no', 'no', 'no']
    )

    with pytest.raises(UnsupportedRequestError) as refused:
        client.predict(request, model=KumoRelational())

    assert 'every context row has the same users.tier' in str(refused.value)
    assert endpoint.sent == []


# --- the flat table ---------------------------------------------------------


def test_the_flat_table_has_the_documented_columns(explained):
    """``to_pandas()`` names the row, then where each gradient is.

    Assumptions:
    - The entities are keyed by id.
    """
    assert list(explained.explanations.gradients.to_pandas().columns) == [
        'id',
        'table',
        'hop',
        'key',
        'row',
        'output',
        'column',
        'gradient',
    ]


def test_the_flat_table_has_a_line_per_row_table_output_and_column(explained):
    """One line per (entity, table, related row, output, column).

    Assumptions:
    - Each block has one column and two outputs. User 5 has 1 + 2 + 2 + 2
      related rows and user 7 has 1 + 1 + 1 + 1; the question row is dropped.
    """
    flat = explained.explanations.gradients.to_pandas()

    assert len(flat) == 2 * 7 + 2 * 4
    assert len(flat[(flat['id'] == 5) & (flat['table'] == 'orders')]) == 4


def test_the_flat_table_counts_hops_from_the_entity(explained):
    """``hop`` is 0 for the entity's own row, 1 for a row
    linked to it, and 2 for a row linked through one of those.

    Assumptions:
    - ``users`` holds the entity, ``orders`` and ``events`` link
      to ``users``, and ``products`` is linked from ``orders``.
    """
    flat = explained.explanations.gradients.to_pandas()
    linked = flat[(flat['id'] == 5) & (flat['key'] != 88)]

    hops = linked.groupby('table')['hop'].unique()

    assert {name: list(found) for name, found in hops.items()} == {
        'users': [0],
        'orders': [1],
        'events': [1],
        'products': [2],
    }


def test_the_flat_table_leaves_the_hop_of_an_unlinked_row_empty(explained):
    """A row no relationship sent reaches from the entity has no hop.

    Assumptions:
    - Product 88 is sampled for user 5, and no order of theirs names it.
    """
    flat = explained.explanations.gradients.to_pandas()

    unlinked = flat[(flat['table'] == 'products') & (flat['key'] == 88)]

    assert len(unlinked) == 2
    assert unlinked['hop'].isna().all()


def test_the_flat_table_names_rows_by_key_and_position(explained):
    """``key`` is a related row's key, empty for a keyless table, and the id
    for the entity's own row; ``row`` is the position in ``rows``.

    Assumptions:
    - User 5 has two orders and two keyless events.
    """
    flat = explained.explanations.gradients.to_pandas()
    five = flat[flat['id'] == 5]

    orders = five[five['table'] == 'orders']
    assert sorted(set(zip(orders['row'], orders['key'], strict=True))) == [
        (0, 9003),
        (1, 9004),
    ]
    assert five[five['table'] == 'events']['key'].isna().all()
    assert set(five[five['table'] == 'users']['key']) == {5}
    assert 'instance' not in set(flat['table'])
    gradient = orders[
        (orders['row'] == 1) & (orders['output'] == 'probability_yes')
    ]['gradient']
    assert gradient.tolist() == [13.0]


def test_the_flat_table_names_each_output_as_the_frame_does(explained):
    """``output`` is the name of the column of ``result.to_pandas()`` holding
    that score.

    Assumptions:
    - The NIM explains both class probabilities and returns them.
    """
    named = set(explained.explanations.gradients.to_pandas()['output'])

    assert named == {'probability_no', 'probability_yes'}
    assert named <= set(explained.to_pandas().columns)


def test_a_boolean_class_is_named_by_the_frames_column_for_it(
    gradient_explanation,
):
    """With a boolean target, ``output`` names the frame's
    ``probability_True``/``probability_False``, not the NIM's lowercase class.

    Assumptions:
    - The target is declared ``bool``; the NIM keys ``probabilities`` and
      names each explained ``class`` as ``'false'``/``'true'``, as seen live.
    """
    payload = gathered(5, 7)
    payload['task']['target']['dtype'] = 'bool'
    answer = nim_answer(payload, gradient_explanation)
    for record in answer['predictions']:
        record['prediction'] = 'true'
        record['probabilities'] = {'false': 0.4, 'true': 0.6}
        for output, name in zip(
            record['explanation']['details']['outputs'],
            ('false', 'true'),
            strict=True,
        ):
            output['class'] = name
    result = RelationalAdapter().normalize(answer, EXPLAIN, payload)

    named = set(result.explanations.gradients.to_pandas()['output'])

    assert named == {'probability_False', 'probability_True'}
    assert named <= set(result.to_pandas().columns)


# --- column analysis, made once per request -------------------------------
#
# Each labelled context below gives every entity one ``users`` row, so the
# analysis has one ``COUNT(*)`` cohort holding every entity, and its mean per
# class is that class's share of the context's labels.

#: Two of four entities labelled ``'yes'``.
EVEN = [(1, 'no'), (2, 'yes'), (3, 'no'), (4, 'yes')]
#: One of three entities labelled ``'yes'``.
SKEWED = [(1, 'no'), (2, 'yes'), (3, 'no')]


def labelled(payload: dict, entities: list[tuple[int, Any]]) -> dict:
    """``payload`` with its context replaced by ``entities``, each an
    ``(entity, label)`` with one linked ``users`` row."""
    payload['context'] = {
        'instance_table': {
            'format': 'arrays',
            'columns': ['instance_id', '__kumo_entity_ref_0', 'TARGET'],
            'rows': [
                [at, entity, label]
                for at, (entity, label) in enumerate(entities)
            ],
        },
        'related_tables': {
            'users': {
                'format': 'arrays',
                'columns': TABLES['users'][0],
                'rows': [
                    [at, entity, 1.0] for at, (entity, _) in enumerate(entities)
                ],
            }
        },
    }
    return payload


def gathering(monkeypatch, *calls: tuple[dict, ...]) -> None:
    """Answer each gather with the next of ``calls``, a tuple of payloads."""
    handed = iter(calls)
    monkeypatch.setattr(
        RelationalMaterializer,
        '_gather',
        lambda self, query, indices, num_context_samples, options: next(handed),
    )


def asked_over(monkeypatch, entities, text: str = TIER) -> Request:
    """``text`` scoring users 5 and 7 over the context ``entities``."""
    gathering(monkeypatch, (labelled(gathered(5, 7), entities),))
    return materialized(text)


def label_means(result) -> dict[Any, float]:
    """Per class, the mean in the ``users`` ``COUNT(*)`` cohort."""
    frame = result.explanations.columns.to_pandas()
    count = frame[(frame['table'] == 'users') & (frame['column'] == 'COUNT(*)')]
    assert list(count['cohort'].unique()) == ['[1 - 1+]']
    assert set(count['share']) == {1.0}
    return dict(zip(count['class'], count['mean'], strict=True))


@pytest.mark.parametrize(
    ('outputs', 'analysed'),
    [(['prediction'], False), (['prediction', 'explanation'], True)],
    ids=['not asked', 'asked'],
)
def test_relational_one_shot_analysis_only_when_explained(
    monkeypatch, relational_endpoint, outputs, analysed
):
    """A one-shot relational answer carries a column analysis exactly when an
    explanation was asked for.

    Assumptions:
    - Gathering is stood in for; the endpoint explains every call.
    """
    client, _ = relational_endpoint()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = client.predict(
            asked_over(monkeypatch, EVEN),
            model=KumoRelational(),
            outputs=outputs,
        )

    assert (result.explanations.columns is not None) is analysed


def test_relational_one_shot_predicts_analyse_apart(
    monkeypatch, relational_endpoint
):
    """Separate ``predict()`` calls each analyse their own request's context.

    Assumptions:
    - The first request's context is ``EVEN`` (yes 1/2), the second's
      ``SKEWED`` (yes 1/3).
    """
    client, _ = relational_endpoint()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        even, skewed = (
            client.predict(
                asked_over(monkeypatch, entities),
                model=KumoRelational(),
                outputs=['prediction', 'explanation'],
            )
            for entities in (EVEN, SKEWED)
        )

    assert label_means(even) == pytest.approx({'no': 1 / 2, 'yes': 1 / 2})
    assert label_means(skewed) == pytest.approx({'no': 2 / 3, 'yes': 1 / 3})


def test_relational_fits_analyse_apart_and_after_the_holdout(
    monkeypatch, relational_endpoint
):
    """Separate ``fit()`` calls each analyse what they pin, and a holdout is
    out of it: holding entity 4 back leaves yes at 1/3, not 1/2.

    Assumptions:
    - Both sessions are fitted over ``EVEN``; the second holds 4 back.
    """
    client, _ = relational_endpoint()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        whole, held = (
            client.fit(
                asked_over(monkeypatch, EVEN),
                model=KumoRelational(),
                outputs=['prediction', 'explanation'],
                holdout=holdout,
            ).predict(asked_over(monkeypatch, EVEN))
            for holdout in (None, [4])
        )

    assert label_means(whole) == pytest.approx({'no': 1 / 2, 'yes': 1 / 2})
    assert label_means(held) == pytest.approx({'no': 2 / 3, 'yes': 1 / 3})


def test_a_relational_session_keeps_its_fit_time_analysis(
    monkeypatch, relational_endpoint
):
    """A session's answers all carry the one analysis made at fit, even for a
    request re-materialized over a different context.

    Assumptions:
    - Fitted over ``EVEN``; the later requests are gathered over ``EVEN``
      and then ``SKEWED``, and both are accepted as the pinned question.
    """
    client, _ = relational_endpoint()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fitted = client.fit(
            asked_over(monkeypatch, EVEN),
            model=KumoRelational(),
            outputs=['prediction', 'explanation'],
        )
        same = fitted.predict(asked_over(monkeypatch, EVEN))
        other = fitted.predict(asked_over(monkeypatch, SKEWED))

    assert other.explanations.columns is same.explanations.columns
    assert label_means(other) == pytest.approx({'no': 1 / 2, 'yes': 1 / 2})


#: Five labelled entities, two of them the scored users 5 and 7.
GATHERED_AS = [(1, 'no'), (5, 'yes'), (7, 'no'), (2, 'yes'), (3, 'no')]


def test_a_split_request_analyses_the_rows_it_sends(
    monkeypatch, relational_endpoint
):
    """A request gathered as two calls is analysed once, over the labelled
    rows it sends, after the scored entities' rows were dropped.

    Assumptions:
    - Both calls share one context object, ``GATHERED_AS``; the leak check
      drops 5 and 7 from it, leaving 1, 2, 3 in both: no 2/3, yes 1/3.
    """
    first = labelled(gathered(5), GATHERED_AS)
    second = gathered(7)
    second['context'] = first['context']
    gathering(monkeypatch, (first, second))
    client, endpoint = relational_endpoint()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = client.predict(
            materialized(TIER),
            model=KumoRelational(),
            outputs=['prediction', 'explanation'],
        )

    assert [
        [row[1] for row in sent['context']['instance_table']['rows']]
        for sent in endpoint.sent
    ] == [[1, 2, 3], [1, 2, 3]]
    assert label_means(result) == pytest.approx({'no': 2 / 3, 'yes': 1 / 3})


def test_a_split_request_analyses_each_sent_instance_once(
    monkeypatch, relational_endpoint
):
    """Calls carrying their own copies of one labelled draw are analysed over
    the union of what they send, each instance counted once.

    Assumptions:
    - Each call gets its own copy of ``GATHERED_AS`` under the same instance
      ids, as the engine sends it; each loses only its own scored entity.
    - Sent: call 1 has 1, 7, 2, 3 and call 2 has 1, 5, 2, 3. The union
      1, 5, 7, 2, 3 gives no 3/5, yes 2/5. Counting overlaps twice would
      give yes 3/8; reading call 1 alone, 1/4.
    """
    first = labelled(gathered(5), GATHERED_AS)
    second = labelled(gathered(7), GATHERED_AS)
    gathering(monkeypatch, (first, second))
    client, endpoint = relational_endpoint()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = client.predict(
            materialized(TIER),
            model=KumoRelational(),
            outputs=['prediction', 'explanation'],
        )

    assert [
        [row[1] for row in sent['context']['instance_table']['rows']]
        for sent in endpoint.sent
    ] == [[1, 7, 2, 3], [1, 5, 2, 3]]
    assert label_means(result) == pytest.approx({'no': 3 / 5, 'yes': 2 / 5})


def test_a_forecast_is_analysed_over_every_steps_rows(
    monkeypatch, relational_endpoint
):
    """A forecast carries one analysis over every step's labelled rows, each
    step's entities kept apart: the mean label is over all six rows.

    Assumptions:
    - Two steps gathered over three entities each, labelled 0, 1, 0 and
      then 3, 3, 3: 10/6 over both, 1/3 over the first alone.
    - Both steps number their instances from 0; one ``COUNT(*)`` cohort at
      1 shows no entity took another step's ``users`` row.
    """
    gathering(
        monkeypatch,
        *(
            (labelled(gathered(5, 7, task='regression'), labels),)
            for labels in (
                [(1, 0.0), (2, 1.0), (3, 0.0)],
                [(1, 3.0), (2, 3.0), (3, 3.0)],
            )
        ),
    )
    client, _ = relational_endpoint()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = client.predict(materialized(FORECAST), model=KumoRelational())

    assert label_means(result) == pytest.approx({None: 10 / 6})


def test_a_forecast_analyses_a_table_only_some_steps_reached(
    monkeypatch, relational_endpoint
):
    """A related table one step's context lacks is still analysed, each of
    its rows read against the entity of the step that carried it.

    Assumptions:
    - Two steps over entities 1, 2, 3 (instances 0-2 in each), labelled
      0, 10, 0 and then 100, 200, 300.
    - ``orders`` (``amount`` 1, 2, 3) is reached only in step 1, ``events``
      (``seconds`` 10, 20, 30) only in step 2, one row per entity; both
      columns are numerical. An entity with no row of a table drops out of
      that column's cohorts.
    - Each step's schema names only ``users`` and the table it reached, with
      their links, as the engine sends it.
    - Hand-computed: deciles of 10, 20, 30 are 10, 12, ..., 30; the values
      land in the bins ending 12, 20 and 30, so the kept cohorts are
      [10 - 12], (12 - 20], (20 - 30], one entity each, means 100, 200,
      300. Read against step 1's entities instead, they would be 0, 10, 0;
      dropped, ``events`` would be absent.
    """

    def step(labels: list[float], table: str, values: list[float]) -> dict:
        entities = list(zip([1, 2, 3], labels, strict=True))
        payload = labelled(gathered(5, 7, task='regression'), entities)
        columns = TABLES[table][0]
        # As the engine sends it: the schema names only the tables this
        # step reached, and the links between them.
        schema = payload['schema']
        kept = {'users', table}
        schema['related_tables'] = {
            name: described
            for name, described in schema['related_tables'].items()
            if name in kept
        }
        schema['relationships'] = [
            link
            for link in schema['relationships']
            if link.get('source_table', 'users') in kept
            and link['target_table'] in kept
        ]
        described = schema['related_tables'][table]
        described['columns'][TABLES[table][1]]['stype'] = 'numerical'
        rows = {
            'orders': lambda at, entity, value: [
                at,
                9000 + at,
                entity,
                77,
                value,
            ],
            'events': lambda at, entity, value: [
                at,
                at,
                entity,
                'login',
                value,
            ],
        }[table]
        payload['context']['related_tables'][table] = {
            'format': 'arrays',
            'columns': columns,
            'rows': [
                rows(at, entity, value)
                for at, ((entity, _), value) in enumerate(
                    zip(entities, values, strict=True)
                )
            ],
        }
        return payload

    gathering(
        monkeypatch,
        (step([0.0, 10.0, 0.0], 'orders', [1.0, 2.0, 3.0]),),
        (step([100.0, 200.0, 300.0], 'events', [10.0, 20.0, 30.0]),),
    )
    client, _ = relational_endpoint()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = client.predict(materialized(FORECAST), model=KumoRelational())

    frame = result.explanations.columns.to_pandas()
    analysed = {
        column: (
            list(cohorts['cohort']),
            list(cohorts['share']),
            list(cohorts['mean']),
        )
        for column, cohorts in frame.groupby('column')
        if column in ('amount', 'seconds')
    }
    assert analysed['seconds'][0] == ['[10 - 12]', '(12 - 20]', '(20 - 30]']
    assert analysed['seconds'][1] == pytest.approx([1 / 3] * 3)
    assert analysed['seconds'][2] == pytest.approx([100.0, 200.0, 300.0])
    assert analysed['amount'][0] == ['[1 - 1.20]', '(1.20 - 2]', '(2 - 3]']
    assert analysed['amount'][2] == pytest.approx([0.0, 10.0, 0.0])


def test_time_cohorts_hold_every_rows_age():
    """Each related row's age before its own entity's anchor falls in the
    time cohort whose label contains it: the oldest in the closed last
    cohort, one exactly on an edge in the cohort that edge opens.

    Assumptions:
    - Entity 1 (label 100) is anchored 2024-01-31 with one event on
      2024-01-01, 30 days old; entity 2 (label 10) is anchored 2024-01-11
      with events on 2024-01-05 and 2024-01-01, 6 and 10 days old.
    - Cohort width is the oldest age over ten, rounded up, so ten cohorts
      cover it; an age on an edge goes in the cohort that edge opens, and
      the last cohort is closed.
    - Hand-computed: the oldest age, 30 days, gives 3-day cohorts ending at
      30. 6 opens [6, 9), 10 is in [9, 12), 30 closes [27, 30]. Each entity weighs 1, split over its rows, so the shares are
      1/4, 1/4, 1/2 and the means 10, 10, 100.
    - The raw timestamps span 4 days, which gave 1-day cohorts up to 10
      days: 30 fell in [9, 10) and 6 in [5, 6).
    """
    events = [(1, '2024-01-01'), (2, '2024-01-05'), (2, '2024-01-01')]
    payload = labelled(
        gathered(5, 7, task='regression'), [(1, 100.0), (2, 10.0)]
    )
    payload['task']['anchor_time_column'] = 'anchor'
    instance = payload['context']['instance_table']
    instance['columns'].append('anchor')
    for row, anchor in zip(
        instance['rows'], ['2024-01-31', '2024-01-11'], strict=True
    ):
        row.append(anchor)
    payload['schema']['related_tables']['events'] = {
        'columns': {
            'instance_id': {'dtype': 'int64'},
            '__node_id': {'dtype': 'int64'},
            'user_id': {'dtype': 'int64'},
            'at': {'dtype': 'datetime64[ns]', 'stype': 'timestamp'},
        },
        'primary_key': PRIMARY_KEYS['events'],
    }
    payload['context']['related_tables']['events'] = {
        'format': 'arrays',
        'columns': ['instance_id', '__node_id', 'user_id', 'at'],
        'rows': [
            [entity - 1, node, entity, at]
            for node, (entity, at) in enumerate(events)
        ],
    }

    frame = ColumnExplanations(payload).to_pandas()

    cohorts = frame[frame['column'] == 'at']
    assert list(cohorts['cohort']) == [
        '[6 days, 9 days)',
        '[9 days, 12 days)',
        '[27 days, 30 days]',
    ]
    assert list(cohorts['share']) == pytest.approx([1 / 4, 1 / 4, 1 / 2])
    assert list(cohorts['mean']) == pytest.approx([10.0, 10.0, 100.0])


def numbered(values: list[float]) -> pd.DataFrame:
    """The analysis of a regression context whose ``users.tenure_days``,
    analysed as numerical, holds ``values``, one entity each."""
    entities = [(at + 1, float(at)) for at in range(len(values))]
    payload = labelled(gathered(5, 7, task='regression'), entities)
    users = payload['schema']['related_tables']['users']
    users['columns']['tenure_days']['stype'] = 'numerical'
    rows = payload['context']['related_tables']['users']['rows']
    for row, value in zip(rows, values, strict=True):
        row[2] = value
    return ColumnExplanations(payload).to_pandas()


def test_large_numeric_edges_keep_distinct_labels():
    """Cohort edges of large, close values print in full, so no two cohort
    labels collide and none is in scientific notation.

    Assumptions:
    - Ten entities with ``tenure_days`` 10000 to 10009, one row each.
    - Hand-computed: the deciles step by 0.9 (10000, 10000.90, ...); four
      significant digits would print most of them as ``1e+04``.
    """
    frame = numbered([10000.0 + at for at in range(10)])

    labels = list(frame[frame['column'] == 'tenure_days']['cohort'].unique())
    assert len(labels) == 10
    assert all('e+' not in label and 'e-' not in label for label in labels)


def test_whole_number_edges_print_without_decimals():
    """Whole-number cohort edges, numerical and ``COUNT(*)``, print with no
    ``.00``.

    Assumptions:
    - Eleven entities with ``tenure_days`` 0 to 10, one ``users`` row each.
    - Hand-computed: the deciles are 0, 1, ..., 10, each value but 0 ending
      its own bin; every entity has 1 linked row, so ``COUNT(*)`` is one
      cohort, open-ended.
    """
    frame = numbered([float(at) for at in range(11)])

    tenure = list(frame[frame['column'] == 'tenure_days']['cohort'].unique())
    count = list(frame[frame['column'] == 'COUNT(*)']['cohort'].unique())
    assert tenure == [
        '[0 - 1]',
        *(f'({at} - {at + 1}]' for at in range(1, 10)),
    ]
    assert count == ['[1 - 1+]']


def test_fractional_edges_keep_two_decimals():
    """A fractional cohort edge prints rounded to two decimals.

    Assumptions:
    - Two entities with ``tenure_days`` 0 and 1660 / 7 (237.142857...).
    - Hand-computed: the deciles step by 23.714285...; 0 falls in the first
      bin and 1660 / 7 in the last, and the empty bins between merge into
      it, so the second cohort opens at 23.714285...
    """
    frame = numbered([0.0, 1660 / 7])

    labels = list(frame[frame['column'] == 'tenure_days']['cohort'].unique())
    assert labels == ['[0 - 23.71]', '(23.71 - 237.14]']


def test_tiny_edges_gain_decimals_until_distinct():
    """Cohort edges too close for two decimals print with more, so every
    cohort label is distinct and none is in scientific notation.

    Assumptions:
    - Ten entities with ``tenure_days`` 0, 0.0001, ..., 0.0009, one row each.
    - Hand-computed: the deciles step by 0.00009, so at two decimals every
      edge would print as ``0``; each of the ten bins holds an entity.
    """
    frame = numbered([0.0001 * at for at in range(10)])

    labels = list(frame[frame['column'] == 'tenure_days']['cohort'].unique())
    assert len(labels) == 10
    assert all('e+' not in label and 'e-' not in label for label in labels)


def test_single_value_edges_keep_two_decimals():
    """A one-value column, whose two edges are equal, prints them at two
    decimals rather than escalating.

    Assumptions:
    - Two entities with ``tenure_days`` 0.3 each.
    - Hand-computed: every decile is 0.3, so the column is one cohort.
    """
    frame = numbered([0.3, 0.3])

    labels = list(frame[frame['column'] == 'tenure_days']['cohort'].unique())
    assert labels == ['[0.30 - 0.30]']


def categorised(kinds: int) -> pd.DataFrame:
    """The ``cat`` cohorts of a 300-row regression table: every fifth row
    has no ``cat`` and outcome 1, the rest cycle ``c0``.. over ``kinds``
    categories with outcome 0."""
    cat = [f'c{i % kinds}' if i % 5 else None for i in range(300)]
    payload = {
        'task': {'kind': 'regression', 'target': {'column_name': 'y'}},
        'schema': {
            'instance_table': {
                'primary_key': 'pk',
                'columns': {
                    'pk': {'stype': 'ID'},
                    'cat': {'stype': 'categorical'},
                    'y': {'stype': 'numerical'},
                },
            }
        },
        'context': {
            'instance_table': {
                'columns': ['pk', 'cat', 'y'],
                'rows': [
                    [i, value, 1.0 if value is None else 0.0]
                    for i, value in enumerate(cat)
                ],
            }
        },
    }
    frame = ColumnExplanations(payload).to_pandas()
    return frame[frame['column'] == 'cat']


def test_missing_values_keep_their_own_cohort_past_the_merge():
    """A categorical column with more cohorts than fit keeps its missing
    values as an ``N/A`` row of their own, after ``OTHER(*)``, which merges
    only rare categories.

    Assumptions:
    - Twelve categories plus missing values: 13 cohorts, past the 10 shown.
    - Hand-computed: missing rows are 60 of 300 (share 0.2), all with
      outcome 1; every categorised row has outcome 0.
    """
    cohorts = categorised(12)

    assert len(cohorts) == ColumnExplanations.NUM_COHORTS
    assert list(cohorts['cohort'])[-2:] == ['OTHER(*)', 'N/A']
    na = cohorts[cohorts['cohort'] == 'N/A']
    assert na['share'].item() == pytest.approx(0.2)
    assert na['mean'].item() == pytest.approx(1.0)
    other = cohorts[cohorts['cohort'] == 'OTHER(*)']
    assert other['mean'].item() == pytest.approx(0.0)


def test_a_narrow_column_shows_missing_values_without_a_merge():
    """A categorical column within the cohorts shown lists every category
    and ``N/A``, with no ``OTHER(*)``.

    Assumptions:
    - Three categories plus missing values: 4 cohorts.
    """
    cohorts = categorised(3)

    assert list(cohorts['cohort']) == ['c0', 'c1', 'c2', 'N/A']
    assert cohorts['mean'].iloc[-1] == pytest.approx(1.0)


def test_a_forecast_is_validated_before_it_is_analysed(
    monkeypatch, relational_endpoint
):
    """An explained forecast asking for an output no model gives is refused
    before any column analysis starts.

    Assumptions:
    - Two steps over users 5 and 7, gathering stood in for.
    - The analysis is replaced by a spy that records each start.
    """
    started: list[Any] = []
    monkeypatch.setattr(
        'sdm_client.client.ColumnExplanations',
        lambda *args, **kwargs: started.append(args),
    )
    client, _ = relational_endpoint()
    request = relational_asked(monkeypatch, FORECAST, kind='regression')

    with pytest.raises(StructuredDataError):
        client.predict(
            request,
            model=KumoRelational(),
            outputs=['prediction', 'explanation', 'bogus'],
        )

    assert started == []


def test_a_forecast_not_explained_is_not_analysed(
    monkeypatch, relational_endpoint
):
    """A forecast asked without an explanation carries no analysis.

    Assumptions:
    - Two steps, asked for the prediction only.
    """
    client, _ = relational_endpoint()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = client.predict(
            relational_asked(
                monkeypatch,
                FORECAST.removeprefix('EXPLAIN '),
                kind='regression',
            ),
            model=KumoRelational(),
            outputs=['prediction'],
        )

    assert result.explanations.columns is None


def test_a_failed_analysis_does_not_block_a_forecast(
    monkeypatch, relational_endpoint, failing_analysis
):
    """An explained forecast whose analysis raises warns, naming the error,
    and still answers every step with no column analysis.

    Assumptions:
    - The analysis is made to raise; two steps over users 5 and 7, gathering
      stood in for.
    """
    client, _ = relational_endpoint()
    request = relational_asked(monkeypatch, FORECAST, kind='regression')

    with pytest.warns(
        SDMWarning, match=f'this request failed.*{failing_analysis}'
    ):
        result = client.predict(
            request,
            model=KumoRelational(),
            outputs=['prediction', 'explanation'],
        )

    assert sorted(result.explanations.gradients) == [
        (5, 1),
        (5, 2),
        (7, 1),
        (7, 2),
    ]
    assert result.explanations.columns is None
