# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``id`` holds the caller's key, whichever model answered.

`result` tells callers to join on the key when it is there. The relational
path already made that true: entity values never travel, so it puts the
caller's own key back on each row before returning. The tabular path returned
whatever the endpoint echoed, and the endpoint echoes the stringified list of
primary-key values -- `'[1]'` where the caller asked about `1`. Same client,
same schema, same query, and an identity column that only joins on one of the
two models.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Schema
from sdm_client.adapters.tabular import TabularAdapter
from sdm_client.errors import StructuredDataError

QUERY = 'PREDICT users.churn FOR EACH users.user_id'


def schema_for(keys: list[int] | list[str]) -> Schema:
    frame = pd.DataFrame(
        {
            'user_id': keys,
            'churn': ['yes', 'no'] * (len(keys) // 2),
            'tenure': [float(index) for index in range(len(keys))],
        }
    )
    schema = Schema.from_data({'users': frame}, relationships=[])
    schema['users'].primary_key = 'user_id'
    return schema


def payload_for(schema: Schema, indices: list) -> dict:
    request = PredictiveQuery(QUERY, schema).materialize(indices=indices)
    adapter = TabularAdapter()
    plans = list(adapter.prediction_plan(request, ('prediction',)))
    assert len(plans) == 1
    return plans[0]


def answer(ids: list[str], *, row_indices: list[int] | None = None) -> dict:
    """What the endpoint sends back: an id that is a stringified key list."""
    positions = range(len(ids)) if row_indices is None else row_indices
    return {
        'object': 'prediction.response',
        'id': 'pred_1',
        'model': 'kumo-tabular',
        'metadata': {},
        'predictions': [
            {'id': value, 'row_index': index, 'prediction': 'yes'}
            for index, value in zip(positions, ids, strict=True)
        ],
    }


# --- the key the caller asked about comes back ------------------------------


def test_the_endpoints_stringified_list_becomes_the_key():
    schema = schema_for(list(range(1, 21)))
    payload = payload_for(schema, [3, 7])
    result = TabularAdapter().normalize(
        answer(['[3]', '[7]']), ('prediction',), payload
    )
    assert [item.id for item in result.items] == ['3', '7']


def test_the_frame_joins_back_onto_the_rows_that_were_scored():
    schema = schema_for(list(range(1, 21)))
    payload = payload_for(schema, [3, 7])
    frame = (
        TabularAdapter()
        .normalize(answer(['[3]', '[7]']), ('prediction',), payload)
        .to_pandas()
    )
    assert list(frame['id']) == ['3', '7']


def test_a_string_key_is_returned_as_that_string():
    schema = schema_for(['a', 'b', 'c', 'd'])
    payload = payload_for(schema, ['b', 'd'])
    result = TabularAdapter().normalize(
        answer(["['b']", "['d']"]), ('prediction',), payload
    )
    assert [item.id for item in result.items] == ['b', 'd']


def test_a_key_beyond_the_safe_integer_range_stays_exact():
    wide = 9007199254740993
    schema = schema_for([wide + index for index in range(20)])
    payload = payload_for(schema, [wide, wide + 1])
    result = TabularAdapter().normalize(
        answer([f'[{wide}]', f'[{wide + 1}]']), ('prediction',), payload
    )
    assert [item.id for item in result.items] == [str(wide), str(wide + 1)]


# --- rewriting an id must not misattribute an answer ------------------------


def test_a_row_index_outside_the_batch_is_refused():
    schema = schema_for(list(range(1, 21)))
    payload = payload_for(schema, [3, 7])
    with pytest.raises(StructuredDataError) as error:
        TabularAdapter().normalize(
            answer(['[3]', '[7]'], row_indices=[0, 9]),
            ('prediction',),
            payload,
        )
    assert error.value.code == 'INVALID_RESPONSE'


def test_the_same_row_answered_twice_is_refused():
    schema = schema_for(list(range(1, 21)))
    payload = payload_for(schema, [3, 7])
    with pytest.raises(StructuredDataError) as error:
        TabularAdapter().normalize(
            answer(['[3]', '[3]'], row_indices=[0, 0]),
            ('prediction',),
            payload,
        )
    assert error.value.code == 'INVALID_RESPONSE'


def test_an_answer_with_no_row_index_is_refused():
    schema = schema_for(list(range(1, 21)))
    payload = payload_for(schema, [3, 7])
    response = answer(['[3]', '[7]'])
    response['predictions'][0].pop('row_index')
    with pytest.raises(StructuredDataError) as error:
        TabularAdapter().normalize(response, ('prediction',), payload)
    assert error.value.code == 'INVALID_RESPONSE'


# --- when there is no key to put back, nothing is invented ------------------


def test_rows_supplied_without_the_key_keep_the_endpoints_id():
    schema = schema_for(list(range(1, 21)))
    request = PredictiveQuery(QUERY, schema).materialize(
        predict=pd.DataFrame({'tenure': [3.0, 4.0]})
    )
    adapter = TabularAdapter()
    payload = list(adapter.prediction_plan(request, ('prediction',)))[0]
    result = adapter.normalize(answer(['x', 'y']), ('prediction',), payload)
    assert [item.id for item in result.items] == ['x', 'y']


def test_a_response_read_without_its_payload_is_unchanged():
    result = TabularAdapter().normalize(answer(['[3]']), ('prediction',), None)
    assert [item.id for item in result.items] == ['[3]']
