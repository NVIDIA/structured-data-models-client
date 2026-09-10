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
    # Named for the repeat rather than for the row it left unanswered as a
    # consequence, so this pins the duplicate guard and not the later one.
    assert 'more than once' in str(error.value)


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


def test_a_row_whose_key_is_null_keeps_the_endpoints_id():
    """A null names no row, so there is nothing to put back. Turning it into
    the string 'None' would read as a key that was asked about."""
    schema = schema_for(list(range(1, 21)))
    request = PredictiveQuery(QUERY, schema).materialize(
        predict=pd.DataFrame({'user_id': [None, 902], 'tenure': [3.0, 4.0]})
    )
    adapter = TabularAdapter()
    payload = list(adapter.prediction_plan(request, ('prediction',)))[0]
    result = adapter.normalize(
        answer(['[None]', '[902.0]']), ('prediction',), payload
    )
    assert [item.id for item in result.items] == ['[None]', '902.0']


def test_a_response_read_without_its_payload_is_unchanged():
    result = TabularAdapter().normalize(answer(['[3]']), ('prediction',), None)
    assert [item.id for item in result.items] == ['[3]']


# --- a session answers the same way a one-shot call does --------------------


class StubTransport:
    """Enough of the transport for a session, recording what was sent."""

    def __init__(self, response: dict) -> None:
        self.response = response
        self.scored: list[dict] = []

    def create_session(self, payload: dict) -> dict:
        self.created = payload
        return {'session_id': 'sess_1'}

    def session_predict(self, session_id: str, payload: dict) -> dict:
        self.scored.append(payload)
        return self.response

    def delete_session(self, session_id: str) -> None:
        return None


def fitted_on(schema: Schema, response: dict):
    from sdm_client import SDMClient

    client = SDMClient(url='http://localhost:8000')
    transport = StubTransport(response)
    client._transport = transport
    request = PredictiveQuery(QUERY, schema).materialize(indices=[3, 7])
    return client.fit(request, model='kumo-tabular'), transport, request


def test_a_session_returns_the_same_key_a_one_shot_call_does():
    schema = schema_for(list(range(1, 21)))
    fitted, _, request = fitted_on(schema, answer(['[3]', '[7]']))
    assert [item.id for item in fitted.predict(request).items] == ['3', '7']


def test_a_session_scoring_call_still_sends_only_what_the_contract_allows():
    schema = schema_for(list(range(1, 21)))
    fitted, transport, request = fitted_on(schema, answer(['[3]', '[7]']))
    fitted.predict(request)
    assert set(transport.scored[0]) <= {
        'predict',
        'output',
        'inference',
        'metadata',
    }


def test_the_adapter_reads_the_answer_against_the_whole_request():
    """The pinned half names the task and schema. Without them an adapter
    cannot tell a forecast's repeated rows from a duplicate answer, nor say
    which column an id stands for."""
    schema = schema_for(list(range(1, 21)))
    seen: list[dict] = []

    class Recording(TabularAdapter):
        def normalize(self, response, outputs, payload=None):
            seen.append(payload or {})
            return super().normalize(response, outputs, payload)

    from sdm_client import SDMClient

    client = SDMClient(url='http://localhost:8000')
    client._transport = StubTransport(answer(['[3]', '[7]']))
    client._adapters['kumo-tabular'] = Recording()
    request = PredictiveQuery(QUERY, schema).materialize(indices=[3, 7])
    fitted = client.fit(request, model='kumo-tabular')
    fitted.predict(request)

    assert seen, 'normalize was never reached'
    assert 'task' in seen[0]
    assert 'schema' in seen[0]
    assert 'predict' in seen[0]


# --- row_index is trusted only as far as it can be checked ------------------


def test_a_batch_left_partly_unanswered_is_refused():
    """Relabelling by row_index assumes the batch was answered in full;
    a short response would silently drop the rows it omitted."""
    schema = schema_for(list(range(1, 21)))
    payload = payload_for(schema, [3, 7])
    with pytest.raises(StructuredDataError) as error:
        TabularAdapter().normalize(
            answer(['[3]'], row_indices=[0]), ('prediction',), payload
        )
    assert error.value.code == 'INVALID_RESPONSE'


def test_an_answer_carrying_another_rows_key_is_refused():
    """The endpoint answers under the key it scored. If that key belongs to a
    different row of this batch, row_index and id disagree and relabelling by
    row_index would attach the answer to the wrong entity."""
    schema = schema_for(list(range(1, 21)))
    payload = payload_for(schema, [3, 7])
    with pytest.raises(StructuredDataError) as error:
        TabularAdapter().normalize(
            answer(['[7]', '[3]']), ('prediction',), payload
        )
    assert error.value.code == 'INVALID_RESPONSE'


def test_an_answer_carrying_another_rows_bare_key_is_refused():
    """The endpoint may answer under the key itself rather than under a list of
    one -- that is the shape the relational endpoint uses. Both are keys this
    batch can be checked against."""
    schema = schema_for(list(range(1, 21)))
    payload = payload_for(schema, [3, 7])
    with pytest.raises(StructuredDataError) as error:
        TabularAdapter().normalize(answer(['7', '3']), ('prediction',), payload)
    assert error.value.code == 'INVALID_RESPONSE'


def test_an_id_in_a_shape_this_client_does_not_recognise_is_left_to_row_index():
    """Only a recognisable key can disagree. An endpoint that formats its id
    some other way is not second-guessed."""
    schema = schema_for(list(range(1, 21)))
    payload = payload_for(schema, [3, 7])
    result = TabularAdapter().normalize(
        answer(['row-0', 'row-1']), ('prediction',), payload
    )
    assert [item.id for item in result.items] == ['3', '7']


def test_rows_sharing_a_key_are_each_answered_under_it():
    """Several rows can carry the same entity value -- the key is inferred and
    warned about, not refused -- and all of them are scored. The endpoint
    answers each under that shared key, and none of them is a mismatch."""
    frame = pd.DataFrame(
        {
            'user_id': list(range(200)),
            'churn': ['yes', 'no'] * 100,
            'tenure': [float(index) for index in range(200)],
        }
    )
    frame.loc[[7, 8], 'user_id'] = 6
    with pytest.warns(UserWarning):
        query = PredictiveQuery(QUERY, {'users': frame})
    request = query.materialize(indices=[6])
    adapter = TabularAdapter()
    payload = list(adapter.prediction_plan(request, ('prediction',)))[0]
    scored = len(payload['predict']['instance_table']['rows'])
    assert scored == 3

    result = adapter.normalize(
        answer(['[6]'] * scored), ('prediction',), payload
    )
    assert [item.id for item in result.items] == ['6'] * scored


# --- checking the batch does not depend on having a key to put back ---------


def keyless_payload() -> dict:
    """Two rows to score, supplied without the key column."""
    schema = schema_for(list(range(1, 21)))
    request = PredictiveQuery(QUERY, schema).materialize(
        predict=pd.DataFrame({'tenure': [3.0, 4.0]})
    )
    return list(TabularAdapter().prediction_plan(request, ('prediction',)))[0]


def test_a_keyless_batch_left_partly_unanswered_is_still_refused():
    with pytest.raises(StructuredDataError) as error:
        TabularAdapter().normalize(
            answer(['x'], row_indices=[0]), ('prediction',), keyless_payload()
        )
    assert error.value.code == 'INVALID_RESPONSE'


def test_a_keyless_batch_answered_twice_for_one_row_is_still_refused():
    with pytest.raises(StructuredDataError) as error:
        TabularAdapter().normalize(
            answer(['x', 'y'], row_indices=[0, 0]),
            ('prediction',),
            keyless_payload(),
        )
    assert error.value.code == 'INVALID_RESPONSE'


def test_a_keyless_batch_naming_a_row_outside_it_is_still_refused():
    with pytest.raises(StructuredDataError) as error:
        TabularAdapter().normalize(
            answer(['x', 'y'], row_indices=[0, 5]),
            ('prediction',),
            keyless_payload(),
        )
    assert error.value.code == 'INVALID_RESPONSE'


def test_a_keyless_batch_answered_in_full_keeps_every_id():
    result = TabularAdapter().normalize(
        answer(['x', 'y']), ('prediction',), keyless_payload()
    )
    assert [item.id for item in result.items] == ['x', 'y']


def test_a_batch_the_endpoint_answers_without_an_id_keeps_none():
    """With no key column supplied the endpoint sends no id at all, and there
    is none to invent. `result` tells callers to join on row_index there."""
    payload = keyless_payload()
    response = answer(['x', 'y'])
    for item in response['predictions']:
        item.pop('id')
    result = TabularAdapter().normalize(response, ('prediction',), payload)
    assert [item.id for item in result.items] == [None, None]
    assert 'id' not in result.to_pandas().columns
