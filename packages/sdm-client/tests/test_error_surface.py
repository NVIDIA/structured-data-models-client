# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The holes in an otherwise complete error surface.

The client documents two exception types and raises them almost everywhere. The
exceptions are the interesting part: an unhashable index came out of pandas as
`TypeError: unhashable type: 'dict'`, the session path reached for an attribute
on whatever it was handed, and everything the relational engine does while
*building* a graph escaped as a bare `ValueError` from a package the caller
never imported -- because the guard that converts those failures took the built
graph as an argument and so could not wrap the line that built it.

Two more are about a call that should not fail at all. Releasing a session
marked the handle closed before the release that could fail, so a failure was
permanent and the endpoint's own advice -- retry -- was the one thing the handle
could not do. And a zero-row `predict`, which `Request` allows on purpose and
calls a reasonable no-op, was sent to an endpoint that cannot encode zero rows.
"""

from __future__ import annotations

import contextlib
import warnings

import pandas as pd
import pytest

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Request,
    Schema,
)
from sdm_client.client import FittedModel, _empty_answer
from sdm_client.core.transport import _build_retry
from sdm_client.errors import (
    NimRequestError,
    StructuredDataError,
    UnsupportedRequestError,
)
from sdm_client.schema import Relationship


def rows(count: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'rid': range(count),
            'amount': [float(i % 7) for i in range(count)],
            'region': ['N', 'S'] * (count // 2),
        }
    )


def query(frame: pd.DataFrame | None = None) -> PredictiveQuery:
    frame = rows() if frame is None else frame
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data({'t': frame}, primary_keys={'t': 'rid'})
    return PredictiveQuery('PREDICT t.amount FOR EACH t.rid', schema)


# --- an index that cannot be a key ------------------------------------------


@pytest.mark.parametrize(
    'bad',
    [
        [{'rowId': 1}, {'rowId': 2}],
        [[1], [2]],
        [{1}, {2}],
        [pd.Series([1]), pd.Series([2])],
        [1, {'rowId': 2}],
    ],
    ids=['dicts', 'lists', 'sets', 'series', 'one bad among good'],
)
def test_an_unhashable_index_is_refused_like_every_other_bad_shape(bad):
    """`indices='1,2'` names the type and says what to pass. These came out of
    pandas as `unhashable type: 'dict'`, with no code and nothing saying it was
    a client refusal."""
    with pytest.raises(StructuredDataError) as raised:
        query().materialize(indices=bad)
    assert raised.value.code == 'INVALID_REQUEST'


def test_the_refusal_names_the_shape_an_agent_will_actually_pass():
    """A list of dicts is what `json.loads` and `to_dict('records')` give
    back, so it is the shape a program produces rather than a person."""
    with pytest.raises(StructuredDataError) as raised:
        query().materialize(indices=[{'rid': 1}, {'rid': 2}])
    message = str(raised.value)
    assert "['dict']" in message
    assert 'to_dict("records")' in message
    assert 'hashable' in message


def test_keys_that_can_be_hashed_are_untouched():
    for good in ([1, 2], ['a', 'b'], [(1, 2)], [None]):
        try:
            query().materialize(indices=good)
        except StructuredDataError as error:
            assert 'hashable' not in str(error)


# --- the session path checks what it was handed ----------------------------


def handle(**kwargs) -> FittedModel:
    return FittedModel(
        client=kwargs.pop('client', None),
        adapter=kwargs.pop('adapter', None),
        session_id=kwargs.pop('session_id', 's1'),
        outputs=kwargs.pop('outputs', ()),
        **kwargs,
    )


@pytest.mark.parametrize(
    'given', [pd.DataFrame({'a': [1]}), None, 'a request', 42, {'a': 1}]
)
def test_the_session_path_names_the_type_it_was_given(given):
    """The one-shot path does this. The session path reached for
    `request.payloads` and raised `AttributeError`, which is the one exception
    class the client otherwise never emits."""
    with pytest.raises(StructuredDataError) as raised:
        handle().predict(given)
    assert raised.value.code == 'INVALID_REQUEST'
    assert type(given).__name__ in str(raised.value)


def test_the_type_is_checked_before_the_closed_state():
    """Both are refusals; the argument is wrong either way, and saying the
    handle is closed would send the caller after the wrong problem."""
    closed = handle(client=Client(Transport()))
    closed.close()
    with pytest.raises(StructuredDataError) as raised:
        closed.predict('not a request')
    assert 'must be a Request' in str(raised.value)


# --- a release that failed can be tried again ------------------------------


class Transport:
    def __init__(self, fail_times: int = 0) -> None:
        self.calls = 0
        self._fail_times = fail_times

    def delete_session(self, session_id: str) -> None:
        self.calls += 1
        if self.calls <= self._fail_times:
            raise NimRequestError(
                429,
                code='RATE_LIMIT_EXCEEDED',
                message='Session maintenance is already pending; retry the '
                'request later.',
            )


class Client:
    def __init__(self, transport: Transport) -> None:
        self._transport = transport


class Adapter:
    """Enough of an adapter to name the model, and nothing that sends."""

    class spec:  # noqa: N801 - a stand-in for a ModelSpec
        model_id = 'kumo-tabular'


def test_a_failed_close_leaves_the_handle_open_to_try_again():
    transport = Transport(fail_times=1)
    fitted = handle(client=Client(transport))

    with pytest.raises(NimRequestError):
        fitted.close()
    assert transport.calls == 1

    fitted.close()

    assert transport.calls == 2


def test_a_close_that_worked_is_not_repeated():
    transport = Transport()
    fitted = handle(client=Client(transport))
    fitted.close()
    fitted.close()
    assert transport.calls == 1


def test_a_failure_to_release_is_reported_as_one():
    """It is a real failure, and raising it is what the block did before the
    masking case below was handled. A caller who wraps the block in `try` is
    relying on hearing about it."""
    transport = Transport(fail_times=99)
    with pytest.raises(NimRequestError):
        with handle(client=Client(transport)):
            pass


def test_the_handle_is_still_open_after_a_failed_exit():
    """The release can be tried again, which is the whole of the fix: a
    handle whose `__exit__` failed is not left marked closed."""
    transport = Transport(fail_times=1)
    fitted = handle(client=Client(transport))
    with contextlib.suppress(NimRequestError):
        with fitted:
            pass
    fitted.close()
    assert transport.calls == 2


def test_releasing_a_session_is_retried_by_the_transport():
    """Idempotent by contract -- the NIM answers 204 whether or not the session
    is there -- and the status it gives when it cannot release one now is a 429
    asking the caller to retry."""
    for retry_post in (True, False):
        allowed = _build_retry(3, 0.1, retry_post=retry_post).allowed_methods
        assert 'DELETE' in allowed
        assert 'GET' in allowed
        assert ('POST' in allowed) is retry_post


# --- graph construction is inside the error surface ------------------------


def relational_schema(child: pd.DataFrame) -> Schema:
    parent = pd.DataFrame({'ID': [1, 2, 3], 'label': ['a', 'b', 'a']})
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'parent': parent, 'child': child},
            relationships=[Relationship('child', 'ID', 'parent', 'ID')],
        )
    schema['parent'].primary_key = 'ID'
    schema['child'].primary_key = None
    return schema


def test_a_failure_while_building_the_graph_is_a_client_error():
    """`Graph.from_data`, the key and stype assignments, the links and
    `graph.validate()` all ran one line above the guard that converts their
    failures."""
    pytest.importorskip('sdm_relational_engine')

    schema = relational_schema(
        pd.DataFrame(
            {
                'ID': pd.Series([], dtype='int64'),
                'v': pd.Series([], dtype='object'),
            }
        )
    )
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(
            'PREDICT parent.label FOR EACH parent.ID', schema
        ).materialize(
            indices=[1], num_context_samples=2, model=KumoRelational()
        )
    assert raised.value.code == 'INVALID_REQUEST'
    assert 'empty' in str(raised.value).lower()


def test_a_working_graph_is_not_affected():
    """The child needs a key of its own; a table whose primary key is also the
    link column is refused for a different reason."""
    pytest.importorskip('sdm_relational_engine')

    parent = pd.DataFrame({'ID': range(40), 'label': ['a', 'b'] * 20})
    child = pd.DataFrame(
        {
            'cid': range(200),
            'ID': [i % 40 for i in range(200)],
            'v': [float(i % 9) for i in range(200)],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'parent': parent, 'child': child},
            primary_keys={'parent': 'ID', 'child': 'cid'},
            relationships=[Relationship('child', 'ID', 'parent', 'ID')],
        )
    request = PredictiveQuery(
        'PREDICT parent.label FOR EACH parent.ID', schema
    ).materialize(indices=[1], num_context_samples=2, model=KumoRelational())
    assert request.predict_rows == 1


# --- nothing to score is not a server error --------------------------------


def test_a_zero_row_predict_is_answered_without_asking():
    """`Request` allows it on purpose and calls it a reasonable no-op. Sent, it
    came back `500 Unexpected server error`, because the endpoint cannot encode
    zero rows."""
    frame = rows()
    request = query(frame).materialize(
        predict=frame.head(0).drop(columns=['amount'])
    )
    assert request.predict_rows == 0

    answer = _empty_answer('kumo-tabular', ('prediction',))
    built = answer.to_pandas()

    assert len(built) == 0
    assert list(built.columns) == ['row_index', 'prediction']


def test_selecting_no_rows_by_key_is_the_same_no_op_as_an_empty_frame():
    """The two selectors describe the same empty batch, so a filter that
    matched nothing had to be handled one way as a frame and another as keys."""
    frame = rows()
    by_key = query(frame).materialize(indices=[])
    by_frame = query(frame).materialize(
        predict=frame.head(0).drop(columns=['amount'])
    )

    assert by_key.nothing_to_score is by_frame.nothing_to_score is True
    assert by_key.predict_rows == by_frame.predict_rows == 0


def test_the_empty_answer_carries_the_columns_that_were_asked_for():
    built = _empty_answer('kumo-tabular', ('prediction', 'scores')).to_pandas()
    assert set(built.columns) == {'row_index', 'prediction', 'scores'}


def test_the_empty_answer_reports_no_missing_fields():
    answer = _empty_answer('kumo-tabular', ('prediction',))
    assert answer.absent(['prediction']) == ()


def test_a_relational_request_reports_the_rows_it_carries():
    """These two counts are the only description of a request before it is
    sent, and a relational one keeps its rows in the gathered envelopes rather
    than in `context`/`predict`, so both read nought for every graph request.

    This is load-bearing beyond the report: `predict()` short-circuits a
    zero-row request without sending, so a relational request that claims to
    have no rows gets an empty answer and never reaches the endpoint.
    """
    pytest.importorskip('sdm_relational_engine')

    parent = pd.DataFrame({'ID': range(40), 'label': ['a', 'b'] * 20})
    child = pd.DataFrame(
        {
            'cid': range(200),
            'ID': [i % 40 for i in range(200)],
            'v': [float(i % 9) for i in range(200)],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_data(
            {'parent': parent, 'child': child},
            primary_keys={'parent': 'ID', 'child': 'cid'},
            relationships=[Relationship('child', 'ID', 'parent', 'ID')],
        )
    request = PredictiveQuery(
        'PREDICT parent.label FOR EACH parent.ID', schema
    ).materialize(indices=[1, 2, 3], model=KumoRelational())

    assert request.is_relational
    assert request.predict_rows == 3
    assert request.context_rows > 0
    assert request.predict_rows == len(
        request.payloads[0]['predict']['instance_table']['rows']
    )
    assert '3 to score' in repr(request)


def test_a_split_request_counts_every_call_it_will_make():
    """Both halves are summed across envelopes: each gathered part carries its
    own context as well as its own share of the rows being scored."""
    payloads = (
        {
            'context': {'instance_table': {'rows': [[1], [2], [3]]}},
            'predict': {'instance_table': {'rows': [[4], [5]]}},
        },
        {
            'context': {'instance_table': {'rows': [[1], [2], [3]]}},
            'predict': {'instance_table': {'rows': [[6]]}},
        },
    )
    from sdm_client.request import _envelope_rows

    assert sum(_envelope_rows(p, 'predict') for p in payloads) == 3
    assert _envelope_rows(payloads[0], 'context') == 3


@pytest.mark.parametrize(
    'payload',
    [{}, {'predict': {}}, {'predict': {'instance_table': {}}}, None, 'text'],
)
def test_a_payload_that_cannot_be_read_says_so_rather_than_nought(payload):
    """`None` and `0` are opposite answers where it matters: a request that
    asks about no rows is answered without sending, and one whose rows cannot
    be counted must reach the endpoint and be refused."""
    from sdm_client.request import _envelope_rows

    assert _envelope_rows(payload, 'predict') is None


def test_an_unreadable_payload_is_not_treated_as_empty():
    """The whole reason the count distinguishes the two."""
    from sdm_client.request import Request

    request = Request.__new__(Request)
    request._payloads = ({'predict': {'instance_table': {}}},)

    assert request.nothing_to_score is False
    assert request.predict_rows == 0


def test_a_readable_empty_payload_is_treated_as_empty():
    from sdm_client.request import Request

    request = Request.__new__(Request)
    request._payloads = ({'predict': {'instance_table': {'rows': []}}},)

    assert request.nothing_to_score is True


def test_one_unreadable_part_is_enough_to_send():
    from sdm_client.request import Request

    request = Request.__new__(Request)
    request._payloads = (
        {'predict': {'instance_table': {'rows': []}}},
        {'predict': {'instance_table': {}}},
    )

    assert request.nothing_to_score is False


def test_a_split_request_counts_the_context_of_every_part():
    """Each gathered part carries its own context -- which is why a session
    refuses a split request rather than pinning the first one's -- so reading
    only the first underreports."""
    from sdm_client.request import Request

    request = Request.__new__(Request)
    request._payloads = (
        {
            'context': {'instance_table': {'rows': [[1], [2], [3]]}},
            'predict': {'instance_table': {'rows': [[4], [5]]}},
        },
        {
            'context': {'instance_table': {'rows': [[6], [7]]}},
            'predict': {'instance_table': {'rows': [[8]]}},
        },
    )

    assert request.context_rows == 5
    assert request.predict_rows == 3


# --- a value that claims to be hashable and is not -------------------------


class LiesAboutHashing:
    def __hash__(self):
        raise TypeError('not really hashable')


def test_a_value_that_claims_to_hash_and_raises_is_refused():
    """`isinstance(value, Hashable)` is structural, so a class that defines
    `__hash__` and raises passes it -- and the TypeError then surfaces from
    inside pandas, which is the shape of failure this check exists to
    prevent."""
    from collections.abc import Hashable

    assert isinstance(LiesAboutHashing(), Hashable)

    with pytest.raises(StructuredDataError) as raised:
        query().materialize(indices=[LiesAboutHashing()])
    assert raised.value.code == 'INVALID_REQUEST'


# --- a cleanup warning must not replace the body's exception ---------------


def test_a_failed_release_says_nothing_while_the_body_is_raising():
    """Under `-W error` the warning would raise from `__exit__` and replace
    the exception the block was raising."""
    transport = Transport(fail_times=99)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(ValueError, match='the real failure'):
            with handle(client=Client(transport)):
                raise ValueError('the real failure')
    assert transport.calls == 1


def test_a_failed_release_is_raised_when_the_body_succeeded():
    """A failure to release is a real failure, and a caller wrapping the block
    in `try` is relying on hearing about it. That is what the block did before
    the masking case was handled, and it still does."""
    transport = Transport(fail_times=99)
    with pytest.raises(NimRequestError):
        with handle(client=Client(transport)):
            pass


@pytest.mark.parametrize('rows', [{}, '', 0, None, {'a': 1}, 'ab'])
def test_rows_that_are_not_a_list_cannot_be_counted(rows):
    """`Sized` admits `{}` and `''`, which would read as an envelope carrying
    no rows rather than as one that cannot be read -- and those are the two
    answers this has to keep apart."""
    from sdm_client.request import _envelope_rows

    payload = {'predict': {'instance_table': {'rows': rows}}}
    assert _envelope_rows(payload, 'predict') is None


@pytest.mark.parametrize('rows', [[], (), [[1]], [[1], [2]]])
def test_rows_that_are_a_list_are_counted(rows):
    from sdm_client.request import _envelope_rows

    payload = {'predict': {'instance_table': {'rows': rows}}}
    assert _envelope_rows(payload, 'predict') == len(rows)


# --- refusals that show their own machinery ---------------------------------


def windowed_schema() -> Schema:
    parent = pd.DataFrame({'ID': range(20)})
    child = pd.DataFrame(
        {
            'cid': range(60),
            'ID': [i % 20 for i in range(60)],
            'ts': pd.to_datetime(
                pd.date_range('2024-01-01', periods=60, freq='D')
            ),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_data(
            {'parent': parent, 'child': child},
            primary_keys={'parent': 'ID', 'child': 'cid'},
            time_columns={'child': 'ts'},
            relationships=[Relationship('child', 'ID', 'parent', 'ID')],
        )


@pytest.mark.parametrize('window', ['0, 0, days', '30, 0, days'])
def test_a_window_the_grammar_validates_is_refused_in_words(window):
    """A zero-width window and an inverted one are caught by the grammar's own
    models rather than its parser, so the refusal arrived as a pydantic trace
    around the sentence rather than as the sentence."""
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(
            f'PREDICT COUNT(child.*, {window}) FOR EACH parent.ID',
            windowed_schema(),
        ).materialize(indices=[1], model=KumoRelational())
    message = str(raised.value)
    assert 'needs to be smaller than the end date offset' in message
    assert 'validation error' not in message
    assert 'pydantic' not in message
    assert 'input_value' not in message


def test_a_window_the_parser_refuses_keeps_its_own_words():
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(
            'PREDICT COUNT(child.*, -5, 30, days) FOR EACH parent.ID',
            windowed_schema(),
        ).materialize(indices=[1], model=KumoRelational())
    assert 'aggregation in the past' in str(raised.value)


# --- a request no path can send --------------------------------------------


def test_related_frames_are_refused_where_they_are_offered():
    """Neither model can be sent a request built this way, so the argument
    that builds one is refused rather than answered one call later."""
    context = pd.DataFrame({'ID': [1, 2, 3], 'label': [0, 1, 0]})
    with pytest.raises(UnsupportedRequestError) as raised:
        Request(
            context=context,
            predict=pd.DataFrame({'ID': [4]}),
            target='label',
            task='classification',
            related_tables={'child': pd.DataFrame({'ID': [1], 'v': [1.0]})},
        )
    message = str(raised.value)
    assert 'child' in message
    assert 'materialize' in message


# --- except StructuredDataError is enough, as the module promises -----------


@pytest.mark.parametrize(
    'call',
    [
        lambda c, r: c.predict(object(), model='kumo-tabular'),
        lambda c, r: c.fit(object(), model='kumo-tabular'),
        lambda c, r: c.predict(r, model='kumo-tabular', outputs=123),
        lambda c, r: c.predict(r, model='kumo-tabular', outputs='prediction'),
        lambda c, r: c.predict(r, model='kumo-tabular', outputs=[1, 2]),
    ],
    ids=['predict', 'fit', 'outputs int', 'outputs str', 'outputs of ints'],
)
def test_a_wrong_argument_is_reported_as_this_client_reports_things(call):
    """`_prepare` held the right message and ran after the first attribute was
    dereferenced, so the wrong type escaped as an `AttributeError`."""
    import pandas as pd

    from sdm_client import Request, SDMClient

    context = pd.DataFrame(
        {
            'id': range(20),
            'x': [1.0 * i for i in range(20)],
            'y': [i % 2 for i in range(20)],
        }
    )
    request = Request(
        context=context,
        predict=pd.DataFrame({'id': [99], 'x': [9.0]}),
        target='y',
        task='classification',
    )
    client = SDMClient('http://127.0.0.1:1', api_key='none')
    try:
        with pytest.raises(StructuredDataError):
            call(client, request)
    finally:
        client.close()


def test_a_primary_key_that_is_not_a_name_is_reported_the_same_way():
    import pandas as pd

    from sdm_client import Schema

    frame = pd.DataFrame({'id': range(5), 'x': [1.0] * 5})
    with pytest.raises(StructuredDataError, match='names a column'):
        Schema.from_data({'t': frame}, primary_keys={'t': 1})
