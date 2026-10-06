# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Things the SDK worked out and then did not act on, or did not say.

`exclude_cols_dict` was accepted, checked, and then not passed on. The count of
rows the engine dropped while building the graph was computed and shown only
when something else had already failed. The hook that names the column behind a
cardinality refusal was keyed on one of the two wordings deployments use. A key
containing a character above U+FFFF could never be matched back, because JSON
writes it as a surrogate pair and a Python literal reader takes the halves
apart. And an ordinary multiclass label was refused for having more than
sixty-four classes, naming a semantic type the caller never chose.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    KumoTabular,
    PredictiveQuery,
    Relationship,
    Schema,
)
from sdm_client.adapters.relational import RelationalAdapter
from sdm_client.adapters.tabular import _key_named_by
from sdm_client.errors import NimRequestError, StructuredDataError

RNG = np.random.default_rng(11)


def graph(*, null_times: int = 0, duplicate_keys: int = 0) -> Schema:
    """A two-table graph, with as many repeated keys or null times as asked.

    600 users, so that a handful of repeats stays inside the tolerance
    inference allows. An explicit key is held to the same bar and refused
    above it, which is a different bug's territory.
    """
    users = pd.DataFrame(
        {
            'uid': [*range(600 - duplicate_keys), *range(duplicate_keys)],
            'seg': RNG.choice(list('xy'), 600),
        }
    )
    events = pd.DataFrame(
        {
            'eid': range(600),
            'uid': RNG.integers(0, 600 - duplicate_keys, 600),
            'amt': RNG.gamma(2.0, 10.0, 600),
            'secret': RNG.normal(0, 1, 600),
            'ts': pd.to_datetime('2023-01-01')
            + pd.to_timedelta(RNG.integers(0, 200, 600), 'D'),
        }
    )
    if null_times:
        events.loc[events.index[:null_times], 'ts'] = pd.NaT
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            {'users': users, 'events': events},
            primary_keys={'users': 'uid', 'events': 'eid'},
            time_columns={'events': 'ts'},
            relationships=[Relationship('events', 'uid', 'users', 'uid')],
        )


def gather(schema: Schema, **options):
    return PredictiveQuery(
        'PREDICT SUM(events.amt, 0, 30, days) FOR EACH users.uid', schema
    ).materialize(indices=[1, 2], model=KumoRelational(), **options)


def wire_columns(request, table: str = 'events') -> dict:
    return request.payloads[0]['schema']['related_tables'][table]['columns']


# --- an option the caller gave is used -------------------------------------


def test_a_column_the_caller_excluded_does_not_reach_the_model():
    """It was accepted, checked against the options this route takes, and then
    the query's own answer was sent in its place."""
    pytest.importorskip('sdm_relational_engine')

    schema = graph()
    request = gather(schema, exclude_cols_dict={'events': ['secret']})

    columns = wire_columns(request)
    assert 'secret' not in columns
    assert 'amt' in columns


def test_a_column_nobody_excluded_still_reaches_the_model():
    pytest.importorskip('sdm_relational_engine')

    assert 'secret' in wire_columns(gather(graph()))


def test_excluding_one_column_keeps_the_rest():
    pytest.importorskip('sdm_relational_engine')

    request = gather(graph(), exclude_cols_dict={'events': ['secret']})
    assert {'amt', 'ts'} <= set(wire_columns(request))


def test_the_caller_and_the_query_are_both_honoured():
    """Merged rather than replaced: the query excludes columns for reasons of
    its own, the target among them, and a caller's list must not stand in for
    that."""
    from sdm_client.materializers.relational import _columns_to_leave_out

    class Query:
        @staticmethod
        def get_exclude_cols_dict():
            return {'events': ['target'], 'users': ['private']}

    merged = _columns_to_leave_out(
        Query(), {'exclude_cols_dict': {'events': ['secret']}}
    )
    assert sorted(merged['events']) == ['secret', 'target']
    assert merged['users'] == ['private']


def test_no_exclusions_at_all_is_still_the_query_s_own():
    from sdm_client.materializers.relational import _columns_to_leave_out

    class Query:
        @staticmethod
        def get_exclude_cols_dict():
            return {'events': ['target']}

    assert _columns_to_leave_out(Query(), {}) == {'events': ['target']}


# --- what the graph was built without --------------------------------------


def test_rows_dropped_for_a_missing_time_are_reported():
    """The count and the reason were worked out either way and shown only when
    something else had already failed."""
    pytest.importorskip('sdm_relational_engine')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gather(graph(null_times=120))
    said = [
        str(w.message)
        for w in caught
        if 'without some of the rows' in str(w.message)
    ]
    assert said, [str(w.message) for w in caught]
    assert '120 of 600 rows dropped' in said[0]
    assert 'no time' in said[0]


def test_rows_dropped_for_a_repeated_key_are_reported():
    """Inference tolerates a nearly-unique key, and the rows it tolerates are
    the ones the engine then drops."""
    pytest.importorskip('sdm_relational_engine')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gather(graph(duplicate_keys=5))
    said = [
        str(w.message)
        for w in caught
        if 'without some of the rows' in str(w.message)
    ]
    assert said
    assert 'repeated primary key' in said[0]
    assert 'users:' in said[0]


def test_a_graph_that_lost_nothing_says_nothing():
    pytest.importorskip('sdm_relational_engine')

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gather(graph())
    assert not any('without some of the rows' in str(w.message) for w in caught)


# --- a refusal that names the columns, whichever way it is worded ----------


def counted_payload() -> dict:
    """What the NIM's guard counts is string cells only, across the predict
    half as well as the context. A numeric key is not a categorical column
    however many distinct values it holds, and counting it here would name a
    column the endpoint was not talking about."""
    return {
        'context': {
            'instance_table': {
                'columns': ['id'],
                'rows': [[i] for i in range(50)],
            },
            'related_tables': {
                'biz': {
                    'columns': ['bid', 'addr', 'city'],
                    'rows': [
                        [i, f'{i} Main St', 'PHL' if i % 2 else 'NYC']
                        for i in range(120)
                    ],
                }
            },
        },
        'predict': {
            'instance_table': {'columns': ['id'], 'rows': [[0]]},
        },
    }


def named_payload() -> dict:
    return {
        'schema': {
            'instance_table': {'columns': {'id': {'stype': 'ID'}}},
            'related_tables': {'biz': {'columns': {'addr': {'stype': 'text'}}}},
        }
    }


def refusal(message: str, status: int = 422) -> NimRequestError:
    return NimRequestError(status, code='UNSUPPORTED_SHAPE', message=message)


def _counted_refusal():
    """The explained cardinality refusal the three checks below are about."""
    pytest.importorskip('sdm_relational_engine')
    return RelationalAdapter().explain_error(
        refusal('categorical cardinality 120 exceeds limit 100.'),
        counted_payload(),
    )


def test_a_cardinality_refusal_names_the_columns_that_are_over():
    """The deployments word this two ways and only one was matched, so on the
    one that counts, the hook never ran."""
    explained = str(_counted_refusal())
    assert 'biz.addr (120 values)' in explained
    assert 'biz.city' not in explained
    assert 'biz.bid' not in explained


def test_a_cardinality_refusal_advises_something_that_helps():
    """These columns are already categorical, so telling their owner to make
    them categorical would change nothing."""
    explained = str(_counted_refusal())
    assert 'exclude_cols_dict' in explained
    assert "stype = 'categorical'" not in explained


def test_a_refusal_keeps_its_status_and_code():
    explained = _counted_refusal()
    assert explained.status_code == 422
    assert explained.code == 'UNSUPPORTED_SHAPE'


def test_the_stype_wording_still_names_its_columns():
    explained = RelationalAdapter().explain_error(
        refusal("semantic type 'text' is unsupported"), named_payload()
    )
    assert 'biz.addr' in str(explained)
    assert "stype = 'categorical'" in str(explained)


def test_an_unrelated_error_is_returned_untouched():
    error = refusal('Unexpected server error.', status=500)
    assert RelationalAdapter().explain_error(error, counted_payload()) is error


# --- a key above the basic plane -------------------------------------------


@pytest.mark.parametrize(
    'key',
    ['\U0001f600key', 'plain', 'ünïcøde', '\U0001f600\U0001f601', '3'],
    ids=['emoji', 'ascii', 'latin1', 'two emoji', 'digits'],
)
def test_a_key_survives_the_json_the_endpoint_sends(key):
    """JSON writes a character above U+FFFF as a surrogate pair, and reading it
    as a Python literal takes the halves apart so they never recombine."""
    assert _key_named_by(json.dumps([key])) == key


def test_a_key_survives_the_python_rendering_too():
    """The other wording an endpoint may answer with."""
    assert _key_named_by(repr(['\U0001f600key'])) == '\U0001f600key'


@pytest.mark.parametrize(
    'rendered', ['hello', '[1, 2]', '[]', '{"a": 1}', 'x' * 2000]
)
def test_something_that_is_not_one_key_is_left_alone(rendered):
    assert _key_named_by(rendered) is None


# --- a label with more classes than the threshold --------------------------


def labelled(classes: int) -> Schema:
    rows = classes * 8
    frame = pd.DataFrame(
        {
            'rid': range(rows),
            'x': RNG.normal(0, 1, rows),
            'cat': [f'c{i % classes}' for i in range(rows)],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})


def test_a_label_typed_text_is_refused_with_the_reason_and_the_fix():
    """It named a semantic type the caller never chose, and whether it happened
    depended on how many rows were handed to `from_data`."""
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(
            'PREDICT t.cat FOR EACH t.rid', labelled(200)
        ).materialize(indices=[1, 2], model=KumoTabular())
    message = str(raised.value)
    assert '200 distinct values' in message
    assert "schema['t']['cat'].stype = 'categorical'" in message


def test_the_fix_the_refusal_names_makes_the_query_answerable():
    schema = labelled(200)
    schema['t']['cat'].stype = 'categorical'
    request = PredictiveQuery(
        'PREDICT t.cat FOR EACH t.rid', schema
    ).materialize(indices=[1, 2], model=KumoTabular())
    assert request.predict_rows == 2


def test_a_label_under_the_threshold_is_untouched():
    request = PredictiveQuery(
        'PREDICT t.cat FOR EACH t.rid', labelled(60)
    ).materialize(indices=[1, 2], model=KumoTabular())
    assert request.predict_rows == 2


def test_a_declared_text_column_is_not_second_guessed():
    """Only an inferred `text` is explained; a caller who said `text` meant
    it."""
    schema = labelled(200)
    schema['t']['cat'].stype = 'text'
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery('PREDICT t.cat FOR EACH t.rid', schema).materialize(
            indices=[1, 2], model=KumoTabular()
        )
    assert 'distinct values' not in str(raised.value)


def test_a_numeric_column_is_not_named_however_many_values_it_holds():
    """The guard counts string cells. Counting every cell instead named a
    numeric key as the offender, which is a column the endpoint was not
    talking about."""
    explained = RelationalAdapter().explain_error(
        refusal('categorical cardinality 120 exceeds limit 100.'),
        counted_payload(),
    )
    assert 'bid' not in str(explained)


def test_a_payload_with_nothing_over_the_limit_is_left_alone():
    error = refusal('categorical cardinality 120 exceeds limit 100000.')
    assert RelationalAdapter().explain_error(error, counted_payload()) is error


# --- an override is held to the limits the query was -----------------------


@pytest.mark.parametrize('given', [0, -1, 21, 999, 1.5, True, 'five'])
def test_a_top_k_the_model_cannot_rank_is_refused(given):
    """`TOP K` written in the query is checked; passed as an option it went
    straight to the task, so an override could put a number in a request that
    the same number written in the query would have been refused for."""
    from sdm_client.materializers.relational import _candidates_to_rank

    class Query:
        top_k = 5

    with pytest.raises(StructuredDataError):
        _candidates_to_rank(Query(), {'top_k': given})


@pytest.mark.parametrize('given', [1, 5, 20])
def test_a_top_k_the_model_can_rank_is_used(given):
    from sdm_client.materializers.relational import _candidates_to_rank

    class Query:
        top_k = 5

    assert _candidates_to_rank(Query(), {'top_k': given}) == given


def test_no_override_leaves_the_query_s_own():
    from sdm_client.materializers.relational import _candidates_to_rank

    class Query:
        top_k = 7

    assert _candidates_to_rank(Query(), {}) == 7


@pytest.mark.parametrize(
    'given',
    [
        ['events'],
        'events',
        {'events': 'secret'},
        {'events': 3},
        [],
        '',
        False,
        0,
    ],
    ids=[
        'list',
        'string',
        'string value',
        'number value',
        'empty list',
        'empty string',
        'false',
        'zero',
    ],
)
def test_an_exclusion_that_is_not_a_mapping_of_names_is_refused(given):
    """It reached the engine and raised `AttributeError` from inside the guard
    that turns engine failures into client errors, so a bad argument came back
    as an internal failure."""
    from sdm_client.materializers.relational import _columns_to_leave_out

    class Query:
        @staticmethod
        def get_exclude_cols_dict():
            return {}

    with pytest.raises(StructuredDataError) as raised:
        _columns_to_leave_out(Query(), {'exclude_cols_dict': given})
    assert raised.value.code == 'INVALID_REQUEST'


@pytest.mark.parametrize('given', [None, {}])
def test_nothing_to_exclude_is_allowed(given):
    """Absent and explicitly nothing both mean nothing."""
    from sdm_client.materializers.relational import _columns_to_leave_out

    class Query:
        @staticmethod
        def get_exclude_cols_dict():
            return {'events': ['target']}

    assert _columns_to_leave_out(Query(), {'exclude_cols_dict': given}) == {
        'events': ['target']
    }


def test_the_refusal_names_a_target_and_not_the_first_text_column():
    """A schema with a free-text feature and a many-class label has two text
    columns, and naming the wrong one is a remedy that changes nothing."""
    classes = 200
    rows = classes * 8
    frame = pd.DataFrame(
        {
            'rid': range(rows),
            'free_text': [f'a long note number {i}' for i in range(rows)],
            'cat': [f'c{i % classes}' for i in range(rows)],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})
    assert schema['t']['free_text'].stype == 'text'

    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery('PREDICT t.cat FOR EACH t.rid', schema).materialize(
            indices=[1, 2], model=KumoTabular()
        )
    message = str(raised.value)
    assert "schema['t']['cat'].stype = 'categorical'" in message
    assert 'free_text' not in message
