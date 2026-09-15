# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The context must not contain the question's own answer.

On a target with no time window, the entity being scored is sampled into the
context as well, at the same anchor, carrying the value being asked for.
Measured against the live endpoint on a label that is pure noise, a hundred
scored entities came back 0.660 correct where chance is 0.500; with those rows
removed, 0.440.

The distinction that makes this safe to fix is the anchor. An entity appearing
in the context at an *earlier* anchor is not a leak -- it is an earlier
example, and it is what temporal in-context learning runs on. Dropping those
would delete the examples an aggregation target depends on, to fix a leak that
is not there.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Relationship, Schema
from sdm_client.materializers.relational import (
    _drop_context_that_answers_the_question,
    _entities_at_anchors,
    _instances_matching,
    _remove_instances,
)

STATIC = 'PREDICT users.label FOR EACH users.uid'
WINDOWED = 'PREDICT SUM(events.amt, 0, 30, days) FOR EACH users.uid'


def graph(rows: int = 300) -> Schema:
    rng = np.random.default_rng(17)
    users = pd.DataFrame(
        {
            'uid': range(rows),
            'seg': rng.choice(list('xy'), rows),
            'label': rng.integers(0, 2, rows).astype(str),
        }
    )
    events = pd.DataFrame(
        {
            'eid': range(rows * 6),
            'uid': rng.integers(0, rows, rows * 6),
            'amt': rng.gamma(2.0, 10.0, rows * 6),
            'ts': pd.to_datetime('2023-01-01')
            + pd.to_timedelta(rng.integers(0, 200, rows * 6), 'D'),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            {'users': users, 'events': events},
            primary_keys={'users': 'uid', 'events': 'eid'},
            time_columns={'events': 'ts'},
            relationships=[Relationship('events', 'uid', 'users', 'uid')],
        )


def gathered(query: str, indices=(1, 2, 3, 4, 5), **options):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        request = PredictiveQuery(query, graph()).materialize(
            indices=list(indices), model=KumoRelational(), **options
        )
    return request, [str(w.message) for w in caught]


def entities_in_context(request) -> set:
    context = request.payloads[0]['context']['instance_table']
    columns = context['columns']
    reference = next(
        i
        for i, n in enumerate(columns)
        if str(n).startswith('__kumo_entity_ref')
    )
    return {row[reference] for row in context['rows']}


def same_anchor_overlap(request) -> list:
    payload = request.payloads[0]
    context = payload['context']['instance_table']
    predict = payload['predict']['instance_table']
    asked = _entities_at_anchors(predict)
    columns = context['columns']
    reference = next(
        i
        for i, n in enumerate(columns)
        if str(n).startswith('__kumo_entity_ref')
    )
    anchor = columns.index('__kumo_anchor_time')
    return [
        row[reference]
        for row in context['rows']
        if (row[reference], row[anchor]) in asked
    ]


# --- the leak ---------------------------------------------------------------


def test_a_static_target_leaves_no_row_answering_its_own_question():
    request, _ = gathered(STATIC)
    assert same_anchor_overlap(request) == []


def test_removing_them_is_said_out_loud():
    """The context is smaller than it would otherwise have been, and that is a
    change to what the model sees."""
    _, said = gathered(STATIC)
    assert any('being scored' in s for s in said), said


def test_the_rows_removed_are_counted():
    _, said = gathered(STATIC)
    complaint = next(s for s in said if 'being scored' in s)
    assert complaint.startswith('5 of the rows')


# --- what must not be removed ----------------------------------------------


def test_an_aggregation_keeps_the_entity_at_earlier_anchors():
    """Those rows are earlier examples, with features cut at their own anchor
    and a label window that closes before the prediction window opens."""
    request, said = gathered(WINDOWED)
    payload = request.payloads[0]
    context = payload['context']['instance_table']
    columns = context['columns']
    reference = next(
        i
        for i, n in enumerate(columns)
        if str(n).startswith('__kumo_entity_ref')
    )
    present = {row[reference] for row in context['rows']}

    assert {1, 2, 3, 4, 5} <= present
    assert same_anchor_overlap(request) == []
    assert not any('being scored' in s for s in said)


def test_an_aggregation_keeps_every_context_row_it_gathered():
    request, _ = gathered(WINDOWED)
    assert len(request.payloads[0]['context']['instance_table']['rows']) > 300


def test_a_static_target_keeps_everything_that_is_not_the_answer():
    request, _ = gathered(STATIC)
    rows = request.payloads[0]['context']['instance_table']['rows']
    assert len(rows) == 295


# --- the related rows go with the row they hang off ------------------------


def test_related_rows_of_a_removed_instance_go_with_it():
    """Each half carries its own `instance_id` space, so a context row left
    behind by its instance is an orphan."""
    request, _ = gathered(STATIC)
    context = request.payloads[0]['context']
    instances = {
        row[context['instance_table']['columns'].index('instance_id')]
        for row in context['instance_table']['rows']
    }
    for table in context['related_tables'].values():
        at = table['columns'].index('instance_id')
        assert {row[at] for row in table['rows']} <= instances


def test_the_predict_half_is_untouched():
    request, _ = gathered(STATIC)
    predict = request.payloads[0]['predict']
    assert len(predict['instance_table']['rows']) == 5


# --- the pieces, directly ---------------------------------------------------


def table(columns, rows) -> dict:
    return {'columns': list(columns), 'rows': [list(r) for r in rows]}


def test_an_entity_at_a_different_anchor_is_not_matched():
    predict = table(
        ['instance_id', '__kumo_entity_ref_0', '__kumo_anchor_time'],
        [[0, 7, 'T2']],
    )
    context = table(
        ['instance_id', '__kumo_entity_ref_0', '__kumo_anchor_time'],
        [[10, 7, 'T1'], [11, 7, 'T2'], [12, 8, 'T2']],
    )
    matched = _instances_matching(context, _entities_at_anchors(predict))
    assert matched == {11}


@pytest.mark.parametrize(
    'missing',
    [
        {'columns': ['instance_id'], 'rows': [[1]]},
        {'columns': [], 'rows': []},
        {'rows': [[1]]},
        {},
        None,
    ],
)
def test_a_table_without_the_columns_matches_nothing(missing):
    assert _entities_at_anchors(missing) == set()
    assert _instances_matching(missing, {(1, 'T')}) == set()


def test_removing_from_a_table_without_instance_ids_does_nothing():
    plain = table(['a'], [[1], [2]])
    _remove_instances(plain, {1})
    assert plain['rows'] == [[1], [2]]


def test_a_payload_with_no_predict_half_is_left_alone():
    payload = {
        'context': {
            'instance_table': table(
                ['instance_id', '__kumo_entity_ref_0', '__kumo_anchor_time'],
                [[1, 7, 'T']],
            )
        }
    }
    _drop_context_that_answers_the_question((payload,))
    assert len(payload['context']['instance_table']['rows']) == 1


# --- a static label does not move with the anchor --------------------------


def test_a_static_target_removes_the_entity_at_any_anchor():
    """A static column is not a window over anything: its value is the same at
    every anchor, so the same entity at an earlier one carries the answer just
    as plainly. Matching on the anchor alone let that straight through, and a
    caller reaches it by setting `context_anchor_time`."""
    request, said = gathered(
        STATIC,
        anchor_time=pd.Timestamp('2023-06-01'),
        context_anchor_time=pd.Timestamp('2023-05-02'),
    )
    assert entities_in_context(request).isdisjoint({1, 2, 3, 4, 5})
    assert any('being scored' in s for s in said)


def test_an_aggregation_keeps_earlier_anchors_even_when_they_are_chosen():
    """The same option on a windowed target is asking for earlier examples,
    which is the thing this must not delete."""
    request, said = gathered(
        WINDOWED,
        anchor_time=pd.Timestamp('2023-06-01'),
        context_anchor_time=pd.Timestamp('2023-04-01'),
    )
    assert {1, 2, 3, 4, 5} <= entities_in_context(request)
    assert not any('being scored' in s for s in said)


def test_a_static_target_removes_the_entity_from_the_default_gather_too():
    request, _ = gathered(STATIC)
    assert entities_in_context(request).isdisjoint({1, 2, 3, 4, 5})


# --- the anchor column is read, not assumed --------------------------------


def test_a_suffixed_anchor_column_is_still_found():
    """The engine suffixes the name when it would collide with a real column,
    and looking only for the bare name went blind on exactly those graphs."""
    payload = {
        'task': {'anchor_time_column': '__kumo_anchor_time_1'},
        'predict': {
            'instance_table': table(
                ['instance_id', '__kumo_entity_ref_0', '__kumo_anchor_time_1'],
                [[0, 7, 'T2']],
            )
        },
        'context': {
            'instance_table': table(
                ['instance_id', '__kumo_entity_ref_0', '__kumo_anchor_time_1'],
                [[10, 7, 'T1'], [11, 7, 'T2'], [12, 8, 'T2']],
            ),
            'related_tables': {
                'users': table(
                    ['instance_id', 'v'], [[10, 1], [11, 2], [12, 3]]
                )
            },
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _drop_context_that_answers_the_question((payload,), windowed=True)

    kept = [row[0] for row in payload['context']['instance_table']['rows']]
    assert kept == [10, 12]
    assert [
        row[0] for row in payload['context']['related_tables']['users']['rows']
    ] == [
        10,
        12,
    ]


def test_a_static_target_with_a_suffixed_anchor_removes_every_anchor():
    payload = {
        'task': {'anchor_time_column': '__kumo_anchor_time_1'},
        'predict': {
            'instance_table': table(
                ['instance_id', '__kumo_entity_ref_0', '__kumo_anchor_time_1'],
                [[0, 7, 'T2']],
            )
        },
        'context': {
            'instance_table': table(
                ['instance_id', '__kumo_entity_ref_0', '__kumo_anchor_time_1'],
                [[10, 7, 'T1'], [11, 7, 'T2'], [12, 8, 'T2']],
            )
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _drop_context_that_answers_the_question((payload,), windowed=False)

    assert [row[0] for row in payload['context']['instance_table']['rows']] == [
        12
    ]


def test_a_task_naming_no_anchor_column_falls_back_to_the_plain_name():
    payload = {
        'predict': {
            'instance_table': table(
                ['instance_id', '__kumo_entity_ref_0', '__kumo_anchor_time'],
                [[0, 7, 'T2']],
            )
        },
        'context': {
            'instance_table': table(
                ['instance_id', '__kumo_entity_ref_0', '__kumo_anchor_time'],
                [[10, 7, 'T1'], [11, 7, 'T2']],
            )
        },
    }
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _drop_context_that_answers_the_question((payload,), windowed=True)
    assert [row[0] for row in payload['context']['instance_table']['rows']] == [
        10
    ]
