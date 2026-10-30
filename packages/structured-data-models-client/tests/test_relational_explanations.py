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
from types import SimpleNamespace
from typing import Any

import pytest

from sdm_client import KumoRelational, Request, Schema, SDMClient
from sdm_client.adapters.relational import RelationalAdapter
from sdm_client.core.backend import NimBackend

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
                'rows': [[0, 1, 'no'], [1, 2, 'yes'], [2, 3, 'no']],
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

    def predict(self, payload: dict) -> dict:
        self.sent.append(payload)
        return nim_answer(payload, self.build)

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
    assert sorted(explained.explanations) == [5, 7]
    assert sorted(explained.explanations) == sorted(explained.to_pandas()['id'])


def test_the_engines_question_row_is_not_an_entrys_block(explained):
    """An entry has no ``instance``: that block explains the engine's question
    row, not a row the caller sent, and the entity's own row is in ``related``.

    Assumptions:
    - The NIM explains every record's ``instance_table`` as well.
    """
    for entry in explained.explanations.values():
        assert set(entry) == {'outputs', 'related'}


def test_an_entity_has_a_block_for_every_related_table(explained):
    """Every related table sent is in the entry, under its own name.

    Assumptions:
    - The NIM sends one block per related table for every record.
    """
    for entry in explained.explanations.values():
        assert set(entry['related']) == set(TABLES)


def test_related_rows_are_the_entitys_own_values_as_sent(explained):
    """A related row comes back as the values sent, less the engine's
    ``instance_id`` and ``__node_id``.

    Assumptions:
    - The rows of the two entities are interleaved in every table, so a
      position read as the wrong row names the other entity's.
    """
    related = explained.explanations[5]['related']

    assert related['users']['rows'] == [{'user_id': 5, 'tenure_days': 30.0}]
    assert related['orders']['rows'] == [
        {'order_id': 9003, 'user_id': 5, 'product_id': 77, 'amount': 12.0},
        {'order_id': 9004, 'user_id': 5, 'product_id': 77, 'amount': 13.0},
    ]
    assert related['events']['rows'] == [
        {'user_id': 5, 'kind': 'login', 'seconds': 41},
        {'user_id': 5, 'kind': 'login', 'seconds': 42},
    ]
    assert explained.explanations[7]['related']['products']['rows'] == [
        {'product_id': 77, 'price': 4.5}
    ]


def test_a_keyed_table_gives_each_row_its_key_as_declared(explained):
    """A table with a key of its own gives each row that key, read back as its
    declared dtype, while ``rows`` keep the value as sent.

    Assumptions:
    - ``orders.order_id`` is declared int64, and user 7's order id is past
      2**53, so it travels as text.
    """
    orders_7 = explained.explanations[7]['related']['orders']

    assert explained.explanations[5]['related']['orders']['keys'] == [
        9003,
        9004,
    ]
    assert orders_7['keys'] == [WIDE]
    assert isinstance(orders_7['keys'][0], int)
    assert orders_7['rows'][0]['order_id'] == str(WIDE)
    assert explained.explanations[5]['related']['users']['keys'] == [5]


def test_a_keyless_table_gives_rows_without_keys(explained):
    """A table keyed only by the engine's ``__node_id`` has no ``keys``.

    Assumptions:
    - ``events`` has no key of its own; the engine numbers its rows.
    """
    events = explained.explanations[5]['related']['events']

    assert 'keys' not in events
    assert all('__node_id' not in row for row in events['rows'])


def test_related_gradients_are_aligned_with_rows(explained):
    """A related row's gradients sit at its position in ``rows``, as
    ``[output][row][column]``.

    Assumptions:
    - The NIM explains each related row by its numerical column ``v`` as
      ``[-v, v]``, so a gradient names the row it belongs to.
    """
    for entry in explained.explanations.values():
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
    entry = explained.explanations[5]

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

    assert result.explanations == {}


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
    assert sorted(result.explanations) == [5, 7]
    seven = result.explanations[7]['related']
    assert seven['users']['keys'] == [7]
    assert seven['orders']['keys'] == [WIDE]
    assert seven['events']['rows'] == [
        {'user_id': 7, 'kind': 'logout', 'seconds': 71}
    ]
    assert result.explanations[5]['related']['orders']['keys'] == [9003, 9004]


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

    assert sorted(result.explanations) == [0, 1]
    assert result.explanations[1]['related']['users']['keys'] == [5]
    assert list(result.explanations.to_pandas().columns)[0] == 'row_index'


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

    assert sorted(merged.explanations) == [0, 1, 2, 3]
    assert merged.explanations[2]['instance']['gradients'] == [[-12.0], [12.0]]
    assert sorted(merged.explanations) == list(merged.to_pandas()['row_index'])


# --- a forecast ---------------------------------------------------------------


def forecast(gradient_explanation, steps: int = 2):
    """A forecast of users 5 and 7 over ``steps`` steps, through the client's
    decomposed route, each step answered by the adapter as the NIM would."""
    from sdm_client.decompose import Decomposition

    rewritten = Decomposition(
        'forecast', tuple((f'q{step}', step) for step in range(1, steps + 1))
    )
    request = SimpleNamespace(
        _decomposition=rewritten, parts=tuple(object() for _ in range(steps))
    )
    adapter = RelationalAdapter()
    answers = []
    for _ in range(steps):
        payload = gathered(5, 7, task='regression')
        answer = nim_answer(payload, gradient_explanation)
        for record in answer['predictions']:
            record['prediction'] = 1.0
            del record['probabilities']
        answers.append(adapter.normalize(answer, EXPLAIN, payload))
    client = SDMClient.__new__(SDMClient)
    handed = iter(answers)
    client.predict = lambda part, model, outputs: next(handed)
    return SDMClient._answer_in_parts(
        client, request, 'kumo-relational', ['prediction', 'explanation']
    )


def test_a_forecast_is_explained_under_id_and_step(gradient_explanation):
    """Each point of a forecast is explained under ``(id, forecast_step)``,
    and nothing is gathered under ``by_forecast_step``.

    Assumptions:
    - Two steps, each answered with an explanation for users 5 and 7.
    """
    result = forecast(gradient_explanation)

    assert sorted(result.explanations) == [(5, 1), (5, 2), (7, 1), (7, 2)]
    assert 'by_forecast_step' not in result.explanations
    frame = result.to_pandas()
    assert sorted(result.explanations) == sorted(
        zip(frame['id'], frame['forecast_step'], strict=True)
    )


def test_a_forecasts_flat_table_names_the_step(gradient_explanation):
    """``to_pandas()`` of a forecast's explanations starts with ``id`` and
    ``forecast_step``.

    Assumptions:
    - Two steps, each explained.
    """
    flat = forecast(gradient_explanation).explanations.to_pandas()

    assert list(flat.columns)[:3] == ['id', 'forecast_step', 'table']
    assert sorted(set(flat['forecast_step'])) == [1, 2]


# --- the flat table ---------------------------------------------------------


def test_the_flat_table_has_the_documented_columns(explained):
    """``to_pandas()`` names the row, then where each gradient is.

    Assumptions:
    - The entities are keyed by id.
    """
    assert list(explained.explanations.to_pandas().columns) == [
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
    flat = explained.explanations.to_pandas()

    assert len(flat) == 2 * 7 + 2 * 4
    assert len(flat[(flat['id'] == 5) & (flat['table'] == 'orders')]) == 4


def test_the_flat_table_counts_hops_from_the_entity(explained):
    """``hop`` is 0 for the entity's own row, 1 for a row
    linked to it, and 2 for a row linked through one of those.

    Assumptions:
    - ``users`` holds the entity, ``orders`` and ``events`` link
      to ``users``, and ``products`` is linked from ``orders``.
    """
    flat = explained.explanations.to_pandas()
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
    flat = explained.explanations.to_pandas()

    unlinked = flat[(flat['table'] == 'products') & (flat['key'] == 88)]

    assert len(unlinked) == 2
    assert unlinked['hop'].isna().all()


def test_the_flat_table_names_rows_by_key_and_position(explained):
    """``key`` is a related row's key, empty for a keyless table, and the id
    for the entity's own row; ``row`` is the position in ``rows``.

    Assumptions:
    - User 5 has two orders and two keyless events.
    """
    flat = explained.explanations.to_pandas()
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
    named = set(explained.explanations.to_pandas()['output'])

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

    named = set(result.explanations.to_pandas()['output'])

    assert named == {'probability_False', 'probability_True'}
    assert named <= set(result.to_pandas().columns)
