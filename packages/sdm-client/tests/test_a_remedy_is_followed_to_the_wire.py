# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advice that survives being taken.

Four refusals named a remedy that did not work. One offered a call this
package does not have; one offered two that disagreed about which link to
keep, so following them in order kept the wrong one; one offered a semantic
type the model's own class limit then refused; one suggested a timestamp that
does not parse. A remedy has to be followed all the way to the wire before it
is printed.

Two more were about the same question said differently: the relational route
narrows the context it gathers and the tabular one says when it does, and a
holdout that names nothing looked exactly like a holdout that named
everything.
"""

from __future__ import annotations

import warnings
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    KumoTabular,
    PredictiveQuery,
    Relationship,
    Request,
    Schema,
)
from sdm_client.client import _held_back
from sdm_client.errors import StructuredDataError
from sdm_client.request import warn_of_rows_scored_from_their_own_answers


def _quietly(build):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return build()


@pytest.fixture
def two_ways_to_one_table() -> Schema:
    rng = np.random.default_rng(0)
    users = pd.DataFrame(
        {'user_id': range(60), 'age': rng.integers(20, 70, 60)}
    )
    orders = pd.DataFrame(
        {
            'order_id': range(300),
            'user_id': rng.integers(0, 60, 300),
            'ship_user_id': rng.integers(0, 60, 300),
            'price': rng.random(300) * 100,
            'ts': pd.date_range('2024-01-01', periods=300, freq='D'),
        }
    )
    schema = _quietly(
        lambda: Schema.from_data(
            {'users': users, 'orders': orders},
            primary_keys={'users': 'user_id', 'orders': 'order_id'},
            time_columns={'orders': 'ts'},
        )
    )
    schema.relationships = [
        Relationship('orders', 'user_id', 'users', 'user_id'),
        Relationship('orders', 'ship_user_id', 'users', 'user_id'),
    ]
    return schema


def test_the_ambiguous_link_refusal_offers_one_remedy_and_it_runs(
    two_ways_to_one_table: Schema,
) -> None:
    query = PredictiveQuery(
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        two_ways_to_one_table,
    )
    with pytest.raises(StructuredDataError) as raised:
        query.materialize(indices=[1], model=KumoRelational())

    said = str(raised.value)
    assert 'unlink(' not in said
    assert said.count('schema.relationships = ') == 1

    remedy = said[said.index('schema.relationships = ') :].rstrip('. ')
    exec(remedy, {'schema': two_ways_to_one_table})
    assert [
        link.source_column for link in two_ways_to_one_table.relationships
    ] == ['ship_user_id']


def test_a_refusal_written_any_other_way_keeps_its_words() -> None:
    from sdm_client.query import _without_a_remedy_this_caller_cannot_run

    said = 'the anchor is after every row; move it earlier.'
    assert _without_a_remedy_this_caller_cannot_run(said) == said


def test_a_text_target_past_the_class_limit_is_not_sent_to_categorical() -> (
    None
):
    rng = np.random.default_rng(1)
    entity = pd.DataFrame(
        {
            'id': range(400),
            'x': rng.random(400),
            'note': [f'label-{index % 90}' for index in range(400)],
        }
    )
    other = pd.DataFrame({'uid': range(400), 'id': range(400)})
    schema = _quietly(
        lambda: Schema.from_data(
            {'t': entity, 'u': other}, primary_keys={'t': 'id', 'u': 'uid'}
        )
    )
    schema.relationships = [Relationship('u', 'id', 't', 'id')]

    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery('PREDICT t.note FOR EACH t.id', schema).materialize(
            indices=[1], model=KumoRelational()
        )

    said = str(raised.value)
    assert 'classifies at most 10 classes' in said
    assert "stype = 'categorical'" not in said


def test_a_text_target_within_the_class_limit_still_gets_the_remedy() -> None:
    rng = np.random.default_rng(2)
    entity = pd.DataFrame(
        {
            'id': range(400),
            'x': rng.random(400),
            'note': [f'label-{index % 70}' for index in range(400)],
        }
    )
    schema = _quietly(
        lambda: Schema.from_data({'t': entity}, primary_keys={'t': 'id'})
    )

    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery('PREDICT t.note FOR EACH t.id', schema).materialize(
            indices=[1], model=KumoTabular()
        )

    said = str(raised.value)
    assert "stype = 'categorical'" in said
    assert 'classifies at most' not in said


def test_an_anchor_time_remedy_is_a_call_that_parses() -> None:
    pytest.importorskip('sdm_relational_engine')
    from sdm_relational_engine.rfm.rfm import _a_conversion_that_works

    assert _a_conversion_that_works('not a date') == ''
    assert _a_conversion_that_works(12345) == ''
    suggested = _a_conversion_that_works('2025-03-01')
    assert suggested == "; try pd.Timestamp('2025-03-01')"
    assert pd.Timestamp('2025-03-01') == pd.Timestamp(year=2025, month=3, day=1)


def test_indices_naming_nothing_is_refused_in_this_package_s_own_voice() -> (
    None
):
    """The relational route cannot gather a neighbourhood around no entity,
    and said so in the engine's words -- 'Provide at least one entity to
    predict for' -- which names nothing a caller can do. The single-table
    route answers an empty selection as the no-op it is, and that is where the
    refusal points."""
    frame = pd.DataFrame(
        {
            'id': range(40),
            'x': [float(index) for index in range(40)],
            'y': [index % 2 for index in range(40)],
        }
    )
    schema = _quietly(
        lambda: Schema.from_data({'t': frame}, primary_keys={'t': 'id'})
    )
    query = PredictiveQuery('PREDICT t.y FOR EACH t.id', schema)

    assert query.materialize(indices=[], model=KumoTabular()).nothing_to_score

    with pytest.raises(StructuredDataError) as raised:
        query.materialize(indices=[], model=KumoRelational())
    said = str(raised.value)
    assert 'Provide at least one entity' not in said
    assert 'Request(predict=<no rows>)' in said

    empty = Request(
        context=frame,
        predict=frame.head(0)[['id', 'x']],
        target='y',
        task='classification',
    )
    assert empty.nothing_to_score


def test_a_holdout_that_matched_nothing_says_so() -> None:
    """Silence looked the same whether nothing went or everything did."""
    payload = {
        'schema': {'instance_table': {'primary_key': 'id'}},
        'context': {
            'instance_table': {
                'columns': ['id', 'y'],
                'rows': [[1.0, 0], [2.0, 1], [3.0, 0]],
            }
        },
    }
    with warnings.catch_warnings(record=True) as heard:
        warnings.simplefilter('always')
        assert _held_back(payload, [90, 91]) == 0
    assert any('matched a row' in str(one.message) for one in heard)


def test_a_holdout_written_as_an_integer_holds_back_a_float_key() -> None:
    """The same entity written two ways is one entity. Comparing the
    spellings held nothing back and said nothing, which is how a holdout came
    to be a silent no-op."""
    payload = {
        'schema': {'instance_table': {'primary_key': 'id'}},
        'context': {
            'instance_table': {
                'columns': ['id', 'y'],
                'rows': [[1.0, 0], [2.0, 1], [3.0, 0]],
            }
        },
    }
    with warnings.catch_warnings(record=True) as heard:
        warnings.simplefilter('always')
        assert _held_back(payload, [1, 2]) == 2
    assert not [one for one in heard if 'matched a row' in str(one.message)]
    assert payload['context']['instance_table']['rows'] == [[3.0, 0]]


def test_a_holdout_given_as_one_string_is_refused_not_spelled_out() -> None:
    payload = {
        'schema': {'instance_table': {'primary_key': 'id'}},
        'context': {'instance_table': {'columns': ['id'], 'rows': [['abc']]}},
    }
    with pytest.raises(StructuredDataError) as raised:
        _held_back(payload, 'abc')
    assert "pass ['abc']" in str(raised.value)


def test_the_relational_default_context_budget_is_said_out_loud() -> None:
    from sdm_client.materializers.relational import (
        _warn_of_a_context_the_default_mode_capped,
    )

    class Task:
        def __init__(self, gathered: int) -> None:
            self.num_context_examples = gathered

    with warnings.catch_warnings(record=True) as heard:
        warnings.simplefilter('always')
        _warn_of_a_context_the_default_mode_capped(Task(1_000), None)
    assert any('num_context_samples' in str(one.message) for one in heard)

    for task, asked in ((Task(999), None), (Task(1_000), 1_000)):
        with warnings.catch_warnings(record=True) as heard:
            warnings.simplefilter('always')
            _warn_of_a_context_the_default_mode_capped(task, asked)
        assert not heard


def test_the_limits_page_records_the_default_the_code_has() -> None:
    from pathlib import Path

    page = (
        Path(__file__).resolve().parents[3] / 'docs' / 'reference' / 'limits.md'
    ).read_text()
    assert '1,000 when explaining, otherwise 10,000' not in page
    assert '1,000 by default, up to 10,000 with `num_context_samples`' in page


def test_a_timestamp_that_parses_to_nothing_is_not_suggested() -> None:
    """Parsing without raising is not enough: `pd.Timestamp('NaT')` returns
    `NaT`, which the check the remedy is meant to satisfy still refuses."""
    pytest.importorskip('sdm_relational_engine')
    from sdm_relational_engine.rfm.rfm import (
        _a_conversion_that_works,
        _check_anchor_time,
    )

    assert _a_conversion_that_works('NaT') == ''
    with pytest.raises(TypeError):
        _check_anchor_time(pd.Timestamp('NaT'), 'anchor_time')

    suggested = _a_conversion_that_works('2025-03-01')
    assert suggested
    _check_anchor_time(pd.Timestamp('2025-03-01'), 'anchor_time')


def test_the_leak_warning_counts_rows_and_not_keys() -> None:
    """The sentence counts rows being scored, and two rows asking about one
    entity are two rows whose answer is being shown."""
    context = pd.DataFrame(
        {'id': [1, 2, 3, 4], 'x': [0.1, 0.2, 0.3, 0.4], 'y': [0, 1, 0, 1]}
    )
    predict = pd.DataFrame({'id': [1, 1], 'x': [0.1, 0.1]})

    with warnings.catch_warnings(record=True) as heard:
        warnings.simplefilter('always')
        warn_of_rows_scored_from_their_own_answers(
            Request(
                context=context,
                predict=predict,
                target='y',
                task='classification',
            )
        )

    said = [
        str(one.message)
        for one in heard
        if 'rows being scored' in str(one.message)
    ]
    assert said and said[0].startswith('2 of the 2 rows being scored')


def test_the_leak_warning_needs_every_part_of_a_composite_key() -> None:
    """Reducing a composite key to the columns both frames happen to carry
    compares a part of an identity, and two rows sharing a part are not the
    same row: one tenant's entity 1 was reported as another tenant's."""
    context = pd.DataFrame(
        {
            'tenant': ['a', 'a', 'b', 'b'],
            'id': [1, 2, 1, 2],
            'x': [0.1, 0.2, 0.3, 0.4],
            'y': [0, 1, 0, 1],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'instance': context}, primary_keys={'instance': ['tenant', 'id']}
        )

    def heard_for(predict: pd.DataFrame) -> list[str]:
        with warnings.catch_warnings(record=True) as heard:
            warnings.simplefilter('always')
            warn_of_rows_scored_from_their_own_answers(
                Request(
                    context=context,
                    predict=predict,
                    schema=schema,
                    target='y',
                    task='classification',
                )
            )
        return [
            str(one.message)
            for one in heard
            if 'rows being scored' in str(one.message)
        ]

    assert not heard_for(pd.DataFrame({'id': [1], 'x': [0.9]}))
    assert not heard_for(pd.DataFrame({'tenant': ['z'], 'id': [9], 'x': [0.9]}))
    assert heard_for(pd.DataFrame({'tenant': ['a'], 'id': [1], 'x': [0.9]}))


@pytest.mark.parametrize(
    ('context_keys', 'predict_keys'),
    [
        ([1, 2, 3], [1]),
        ([1, 2, 3], ['1']),
        ([1.0, 2.0, 3.0], [1]),
        ([1, 2, 3], [1.0]),
        ([2**60 + 1, 2, 3], [2**60 + 1]),
        ([2**60 + 1, 2, 3], [str(2**60 + 1)]),
        (['a', 'b', 'c'], ['a']),
        (['a', 'b', 'c'], ['z']),
        ([1, 2, 3], [9]),
        (
            [Decimal('1.50'), Decimal('2'), Decimal('3')],
            [Decimal('1.5')],
        ),
    ],
)
def test_the_warning_agrees_with_the_wire_on_every_way_of_writing_a_key(
    context_keys: list, predict_keys: list
) -> None:
    """Approximating the encoder with `str()` disagreed with it: a key held as
    1.0 in the context and 1 in the rows being scored reaches the model as one
    value, and nothing was said. The encoder is asked instead of imitated."""
    from sdm_client.adapters.tabular import TabularAdapter

    context = pd.DataFrame(
        {
            'k': context_keys,
            'x': [0.1] * len(context_keys),
            'y': [index % 2 for index in range(len(context_keys))],
        }
    )
    predict = pd.DataFrame({'k': predict_keys, 'x': [0.1] * len(predict_keys)})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'instance': context}, primary_keys={'instance': 'k'}
        )

    with warnings.catch_warnings(record=True) as heard:
        warnings.simplefilter('always')
        request = Request(
            context=context,
            predict=predict,
            schema=schema,
            target='y',
            task='classification',
        )
        warn_of_rows_scored_from_their_own_answers(request)
    warned = any('rows being scored' in str(one.message) for one in heard)

    envelope = next(
        iter(TabularAdapter().prediction_plan(request, ('prediction',)))
    )
    halves = {}
    for half in ('context', 'predict'):
        table = envelope[half]['instance_table']
        at = table['columns'].index('k')
        halves[half] = {row[at] for row in table['rows']}

    assert warned is bool(halves['context'] & halves['predict'])


def test_the_leak_warning_compares_keys_as_the_wire_spells_them() -> None:
    """A key column whose halves hold different python types -- 1 in the
    context and '1' in the rows being scored -- is one column, encoded one
    way, so values that differ in python are the same value by the time the
    model reads them."""
    from sdm_client.adapters.tabular import TabularAdapter

    context = pd.DataFrame(
        {'tenant': [1, 2], 'id': [10, 20], 'x': [0.1, 0.2], 'y': [0, 1]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'instance': context}, primary_keys={'instance': ['tenant', 'id']}
        )

    def heard_for(predict: pd.DataFrame) -> tuple[list[str], set]:
        with warnings.catch_warnings(record=True) as heard:
            warnings.simplefilter('always')
            request = Request(
                context=context,
                predict=predict,
                schema=schema,
                target='y',
                task='classification',
            )
            warn_of_rows_scored_from_their_own_answers(request)
        said = [
            str(one.message)
            for one in heard
            if 'rows being scored' in str(one.message)
        ]
        envelope = next(
            iter(TabularAdapter().prediction_plan(request, ('prediction',)))
        )
        halves = {}
        for half in ('context', 'predict'):
            table = envelope[half]['instance_table']
            at = [table['columns'].index(name) for name in ('tenant', 'id')]
            halves[half] = {
                tuple(row[index] for index in at) for row in table['rows']
            }
        return said, halves['context'] & halves['predict']

    said, overlapping = heard_for(
        pd.DataFrame({'tenant': ['1'], 'id': [10], 'x': [0.1]})
    )
    assert overlapping
    assert said

    said, overlapping = heard_for(
        pd.DataFrame({'tenant': ['9'], 'id': [99], 'x': [0.1]})
    )
    assert not overlapping
    assert not said


def test_the_warning_uses_the_adapter_s_own_widening_rule() -> None:
    """A float column that is float only because pandas has no integer holding
    a null is the integer it was declared to be. The adapter knows that and a
    fresh inference does not, so a second rule written here chose float64
    where the adapter chose int64, could not encode a wide value the adapter
    encodes fine, and reported no leak where there was one."""
    from sdm_client.adapters.tabular import TabularAdapter

    context = pd.DataFrame(
        {
            'k': pd.array([2**53 + 1, 2, 3], dtype='int64'),
            'x': [0.1, 0.2, 0.3],
            'y': [0, 1, 0],
        }
    )
    predict = pd.DataFrame(
        {'k': pd.array([2.0, None], dtype='Float64'), 'x': [0.1, 0.2]}
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'instance': context}, primary_keys={'instance': 'k'}
        )

    with warnings.catch_warnings(record=True) as heard:
        warnings.simplefilter('always')
        request = Request(
            context=context,
            predict=predict,
            schema=schema,
            target='y',
            task='classification',
        )
        warn_of_rows_scored_from_their_own_answers(request)

    envelope = next(
        iter(TabularAdapter().prediction_plan(request, ('prediction',)))
    )
    halves = {}
    for half in ('context', 'predict'):
        table = envelope[half]['instance_table']
        at = table['columns'].index('k')
        halves[half] = {row[at] for row in table['rows']}

    assert halves['context'] & halves['predict']
    assert any('rows being scored' in str(one.message) for one in heard)
