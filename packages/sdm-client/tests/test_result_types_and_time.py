# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What comes back should be what went in, whichever model answered.

`id` and `prediction` both came back as text. The values were right; the types
were not, and they differed between the two models for the same question. The
documented one-line join raised, and `row.prediction == 0` counted nothing on
one model -- silently, and in the direction of "this class never occurs".

And a string time column was read with `pd.to_datetime`'s defaults, which take
an ambiguous slash date month-first. A day-first column was silently re-dated:
twelve monthly events across a year became twelve daily events in one week of
February, and the only place it surfaced was an unrelated complaint about
anchor times.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Schema
from sdm_client.errors import StructuredDataError
from sdm_client.wire import as_declared

# --- the type the caller gave -----------------------------------------------


@pytest.mark.parametrize(
    'value, dtype, expected',
    [
        ('1', 'int64', 1),
        (1, 'int64', 1),
        ('2', 'int32', 2),
        ('1.5', 'float64', 1.5),
        (1.5, 'float64', 1.5),
        ('9007199254740993', 'int64', 9007199254740993),
    ],
)
def test_a_value_is_read_back_as_what_the_column_holds(value, dtype, expected):
    assert as_declared(value, dtype) == expected


@pytest.mark.parametrize(
    'value, dtype',
    [
        ('not a number', 'int64'),
        ('1.9', 'int64'),
        (1.9, 'int64'),
        (None, 'int64'),
        ('abc', 'float64'),
        ('1', None),
        ('1', 'string'),
        ([1], 'int64'),
    ],
)
def test_anything_that_does_not_convert_exactly_is_left_alone(value, dtype):
    """A cast, not a parse. `int(1.9)` does not raise and is not this integer
    under another name."""
    assert (
        as_declared(value, dtype) is value or as_declared(value, dtype) == value
    )


def test_a_whole_float_is_read_as_the_integer_it_is():
    assert as_declared(2.0, 'int64') == 2


# --- a time column that says one thing --------------------------------------


def graph(times) -> Schema:
    rng = np.random.default_rng(3)
    users = pd.DataFrame(
        {'user_id': range(1, 21), 'age': rng.integers(18, 70, 20)}
    )
    events = pd.DataFrame(
        {
            'eid': np.arange(len(times)),
            'user_id': rng.integers(1, 21, len(times)),
            'amount': rng.gamma(2.0, 20.0, len(times)).round(2),
            'ts': times,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'users': users, 'events': events},
            primary_keys={'users': 'user_id', 'events': 'eid'},
        )
    schema['events'].time_column = 'ts'
    return schema


def on_the_wire(schema: Schema) -> list[str]:
    request = PredictiveQuery(
        'PREDICT COUNT(events.*, 0, 30, days) FOR EACH users.user_id', schema
    ).materialize(indices=[1, 2], model=KumoRelational())
    table = request.payloads[0]['context']['related_tables']['events']
    column = table['columns'].index('ts')
    return sorted({str(row[column])[:10] for row in table['rows']})


TRUTH = pd.to_datetime([f'2020-{month:02d}-02' for month in range(1, 13)] * 10)


@pytest.mark.parametrize(
    'form',
    [None, '%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d', '%Y-%m-%dT%H:%M:%SZ'],
    ids=['datetime', 'iso date', 'iso datetime', 'slash', 'iso utc'],
)
def test_a_year_first_column_reads_the_same_as_real_timestamps(form):
    rendered = (
        pd.Series(TRUTH) if form is None else pd.Series(TRUTH).dt.strftime(form)
    )
    assert on_the_wire(graph(rendered)) == on_the_wire(graph(TRUTH))


def test_a_day_first_column_is_refused_rather_than_re_dated():
    """Read month-first, all 24 of these dates move: twelve monthly events
    across a year become twelve daily events in one week of February."""
    ambiguous = pd.Series(TRUTH).dt.strftime('%d/%m/%Y %H:%M:%S')
    with pytest.raises(StructuredDataError) as raised:
        on_the_wire(graph(ambiguous))
    message = str(raised.value)
    assert 'events.ts' in message
    assert 'day-first' in message
    assert 'dayfirst=True' in message


def test_a_time_column_that_is_already_timestamps_is_untouched():
    assert on_the_wire(graph(TRUTH))[0] == '2020-01-02'


def test_an_empty_time_column_is_not_refused():
    from sdm_client.materializers.relational import _to_time

    values = pd.Series([None, None], dtype=object)
    assert _to_time(values, 't.ts') is values


# --- a composite key the language cannot read -------------------------------


def composite_schema() -> Schema:
    rng = np.random.default_rng(6)
    users = pd.DataFrame(
        {'user_id': range(1, 41), 'age': rng.integers(18, 70, 40)}
    )
    events = pd.DataFrame(
        {
            'eid': np.arange(400),
            'user_id': rng.integers(1, 41, 400),
            'amount': rng.gamma(2.0, 20.0, 400).round(2),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            {'users': users, 'events': events},
            primary_keys={'users': 'user_id', 'events': ['user_id', 'eid']},
        )


def test_a_dropped_composite_key_says_so():
    """`TableDefinition.pkey` is one column name, so the language has no form
    for a composite key. It was dropped in silence, and the schema went on
    reporting it."""
    schema = composite_schema()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        PredictiveQuery(
            'PREDICT users.age FOR EACH users.user_id', schema
        ).materialize(indices=[1, 2], model=KumoRelational())
    said = [str(w.message) for w in caught]
    complaint = [s for s in said if 'composite primary key' in s]
    assert complaint, said
    assert "['user_id', 'eid']" in complaint[0]
    assert 'as though it had no key' in complaint[0]


def test_the_composite_key_still_reaches_the_request():
    """Dropped for the query, not for the engine, which does support one."""
    schema = composite_schema()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(
            'PREDICT users.age FOR EACH users.user_id', schema
        ).materialize(indices=[1, 2], model=KumoRelational())
    assert request.predict_rows == 2
    assert schema['events'].primary_key == ['user_id', 'eid']


def test_a_single_column_key_is_not_mentioned():
    rng = np.random.default_rng(6)
    users = pd.DataFrame(
        {'user_id': range(1, 41), 'age': rng.integers(18, 70, 40)}
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        schema = Schema.from_data(
            {'users': users}, primary_keys={'users': 'user_id'}
        )
        PredictiveQuery('PREDICT users.age FOR EACH users.user_id', schema)
    assert not any('composite' in str(w.message) for w in caught)


# --- the relational path puts the key back with its type too ---------------


def test_the_relational_key_keeps_the_type_it_was_sent_as():
    """Both adapters restore the caller's key; both used to stringify it, so
    both had to stop."""
    import dataclasses

    from sdm_client.adapters.relational import _named_entities

    @dataclasses.dataclass
    class Item:
        id: str
        row_index: int
        forecast_step: int | None = None

    payload = {
        'predict': {
            'instance_table': {
                'columns': ['instance_id', '__kumo_entity_ref_0'],
                'rows': [[300, 7], [301, 11]],
            }
        },
        'task': {'kind': 'regression'},
    }
    named = _named_entities((Item('300', 0), Item('301', 1)), payload)

    assert [item.id for item in named] == [7, 11]


def test_the_relational_key_is_left_alone_when_it_is_text():
    import dataclasses

    from sdm_client.adapters.relational import _named_entities

    @dataclasses.dataclass
    class Item:
        id: str
        row_index: int
        forecast_step: int | None = None

    payload = {
        'predict': {
            'instance_table': {
                'columns': ['instance_id', '__kumo_entity_ref_0'],
                'rows': [[300, 'ATL'], [301, 'ORD']],
            }
        },
        'task': {'kind': 'regression'},
    }
    named = _named_entities((Item('300', 0), Item('301', 1)), payload)

    assert [item.id for item in named] == ['ATL', 'ORD']


# --- the date check reads whole values, and all of them --------------------


def test_a_code_that_merely_opens_with_a_date_is_not_read_as_one():
    """A part code can begin with something date-shaped and continue."""
    codes = pd.Series(['2020-01-02-REV-7'] * 120)
    with pytest.raises(StructuredDataError):
        on_the_wire(graph(codes))


def test_an_ambiguous_value_past_the_sample_is_still_found():
    """Reading a sample would let the last row be the one that says something
    else, and a column read two ways is the whole defect."""
    values = pd.Series(
        [f'2020-01-{1 + i % 28:02d}' for i in range(1500)] + ['02/01/2020']
    )
    with pytest.raises(StructuredDataError) as raised:
        on_the_wire(graph(values))
    assert 'day-first' in str(raised.value)


# --- exact means exact, whatever kind of number it is ----------------------


def test_a_number_that_is_not_whole_is_never_read_as_an_integer():
    """`int(1.9)` does not raise, so the guard has to be a comparison -- and it
    has to be asked of the value, not of its type: a Decimal and a numpy float
    truncate as quietly as a Python one."""
    import decimal

    for value in (
        1.9,
        np.float32(1.9),
        decimal.Decimal('1.9'),
        np.float64(0.5),
    ):
        assert as_declared(value, 'int64') == value
        assert not isinstance(as_declared(value, 'int64'), int)


def test_a_bool_is_not_read_as_the_integer_python_says_it_is():
    assert as_declared(True, 'int64') is True
    assert as_declared(False, 'int64') is False


def test_a_wide_key_is_read_back_exactly():
    wide = 9007199254740993
    assert as_declared(str(wide), 'int64') == wide


# --- only a class label is reshaped ----------------------------------------


def item(prediction, index=0):
    import dataclasses

    @dataclasses.dataclass
    class Item:
        prediction: object
        row_index: int
        id: object = None
        forecast_step: int | None = None

    return Item(prediction, index)


@pytest.mark.parametrize(
    'kind, expected',
    [
        ('binary_classification', 2),
        ('multiclass_classification', 2),
        ('classification', 2),
        ('regression', '2'),
        ('forecasting', '2'),
        ('temporal_link_prediction', '2'),
    ],
)
def test_only_a_class_label_is_given_the_targets_type(kind, expected):
    """A class label is a value the column held, so giving it back as the
    column's type restores something. A regression's answer is a number the
    model computed, and reshaping it would be this client deciding how precise
    the answer is."""
    from sdm_client.adapters.relational import _as_declared_target

    payload = {
        'task': {'kind': kind, 'target': {'dtype': 'int64'}},
    }
    assert _as_declared_target([item('2')], payload)[0].prediction == expected


# --- a column of date objects is not text ----------------------------------


def test_an_object_column_of_datetimes_is_not_read_as_text():
    """`infer_tfm_dtype` already calls these `timestamp[us]`. Reading the
    dtype alone sent them down the text path and refused them for an ambiguity
    they do not have."""
    import datetime

    from sdm_client.materializers.relational import _holds_text

    assert not _holds_text(
        pd.Series(
            [datetime.datetime(2020, 1, 2), datetime.datetime(2020, 2, 2)]
        )
    )
    assert not _holds_text(pd.Series([datetime.date(2020, 1, 2)]))
    assert not _holds_text(pd.Series(pd.to_datetime(['2020-01-02'])))
    assert _holds_text(pd.Series(['2020-01-02']))


def test_a_time_column_of_date_objects_materializes():
    import datetime

    objects = pd.Series(
        [datetime.datetime(2020, (i % 12) + 1, 2) for i in range(120)]
    )
    assert on_the_wire(graph(objects))[0] == '2020-01-02'


def test_the_relational_path_reads_a_wide_key_back_exactly():
    """The reference column is declared `string` -- it is a transport key --
    so the dtype has to come from the table it points at."""
    import dataclasses

    from sdm_client.adapters.relational import _key_dtype, _named_entities

    wide = 9007199254740993

    @dataclasses.dataclass
    class Item:
        id: str
        row_index: int
        forecast_step: int | None = None

    payload = {
        'task': {'kind': 'regression'},
        'schema': {
            'relationships': [
                {
                    'source_columns': ['__kumo_entity_ref_0'],
                    'target_table': 'users',
                    'target_columns': ['user_id'],
                }
            ],
            'related_tables': {
                'users': {'columns': {'user_id': {'dtype': 'int64'}}}
            },
        },
        'predict': {
            'instance_table': {
                'columns': ['instance_id', '__kumo_entity_ref_0'],
                'rows': [[300, str(wide)], [301, str(wide + 1)]],
            }
        },
    }
    assert _key_dtype(payload, '__kumo_entity_ref_0') == 'int64'

    named = _named_entities((Item('300', 0), Item('301', 1)), payload)

    assert [each.id for each in named] == [wide, wide + 1]


def test_a_reference_with_no_relationship_is_left_as_it_came():
    from sdm_client.adapters.relational import _key_dtype

    assert _key_dtype({'schema': {}}, '__kumo_entity_ref_0') is None
    assert (
        _key_dtype(
            {
                'schema': {
                    'relationships': [
                        {
                            'source_columns': ['__kumo_entity_ref_0'],
                            'target_table': 'users',
                            'target_columns': ['a', 'b'],
                        }
                    ]
                }
            },
            '__kumo_entity_ref_0',
        )
        is None
    )
