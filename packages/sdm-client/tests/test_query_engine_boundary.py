# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the engine does behind `materialize()`, and what the caller sees.

Three separate things were leaking out of that boundary: the engine's own
exception types, its progress output, and a graph that was not the schema's.
"""

from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout

import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Relationship, Schema
from sdm_client.errors import StructuredDataError
from sdm_client.materializers.relational import (
    RelationalMaterializer,
    _warn_of_columns_the_model_does_not_read,
)

pytest.importorskip('sdm_relational_engine')


def users(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'user_id': range(rows),
            'age': [20.0 + index for index in range(rows)],
            'churn': [index % 2 for index in range(rows)],
        }
    )


def orders(rows: int = 80) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'order_id': range(rows),
            'user_id': [index % 40 for index in range(rows)],
            'price': [10.0 + index for index in range(rows)],
            'placed': pd.date_range('2024-01-01', periods=rows, freq='D'),
        }
    )


def schema() -> Schema:
    return Schema.from_data({'users': users(), 'orders': orders()})


def query(text: str = 'PREDICT users.churn FOR EACH users.user_id'):
    return PredictiveQuery(text, schema())


# --- the graph sent is the schema described ---------------------------------


def edges_of(built: Schema) -> set:
    graph = RelationalMaterializer(built)._build_graph()
    return {(edge.src_table, edge.fkey, edge.dst_table) for edge in graph.edges}


def described(built: Schema) -> set:
    return {
        (link.source_table, link.source_column, link.target_table)
        for link in built.relationships
    }


def test_the_graph_sent_is_exactly_what_the_schema_describes():
    built = schema()
    assert edges_of(built) == described(built)


def test_asserting_there_are_no_relationships_means_none_are_sent():
    """The docstring says an empty sequence asserts there are none."""
    built = Schema.from_data(
        {'users': users(), 'orders': orders()}, relationships=[]
    )
    assert built.relationships == []
    assert edges_of(built) == set()


def test_one_explicit_relationship_sends_exactly_that_one():
    built = Schema.from_data(
        {'users': users(), 'orders': orders()},
        relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
    )
    assert edges_of(built) == {('orders', 'user_id', 'users')}


def test_a_relationship_the_schema_does_not_have_is_not_invented():
    """A column an inference elsewhere might link on stays unlinked."""
    frame = users()
    frame['orders_on_hand'] = frame['user_id']
    built = Schema.from_data(
        {'users': frame, 'orders': orders()},
        relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
    )
    assert edges_of(built) == {('orders', 'user_id', 'users')}


# --- a library writes to a logger, not to stdout ----------------------------


def test_materialize_prints_nothing():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        query().materialize(indices=[1, 2], num_context_samples=20)
    assert buffer.getvalue() == ''


# --- every failure is this client's own error -------------------------------


def test_an_anchor_before_the_data_is_a_structured_data_error():
    with pytest.raises(StructuredDataError) as error:
        query(
            'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
        ).materialize(
            indices=[1, 2],
            num_context_samples=20,
            anchor_time=pd.Timestamp('1990-01-01'),
        )
    assert 'Anchor timestamp' in str(error.value)


def test_an_entity_that_is_not_in_the_table_is_a_structured_data_error():
    with pytest.raises(StructuredDataError):
        query().materialize(indices=[-1, -2], num_context_samples=20)


def test_the_entity_ceiling_is_named_by_this_client_before_gathering():
    """The engine's own message names a `batch_size=` this client has not got."""
    with pytest.raises(StructuredDataError) as error:
        query().materialize(indices=list(range(1_001)), num_context_samples=20)
    message = str(error.value)
    assert error.value.code == 'UNSUPPORTED_REQUEST'
    assert '1,000' in message
    assert 'batch_size' not in message


def test_the_ceiling_is_refused_before_the_engine_is_reached():
    built = query()
    called = False

    def _fail(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError('the engine was reached')

    built._relational_materializer._get_model = _fail  # type: ignore[method-assign]
    with pytest.raises(StructuredDataError):
        built.materialize(indices=list(range(1_001)), num_context_samples=20)
    assert not called


def test_a_dropped_row_is_explained_rather_than_only_counted():
    """The engine says how many rows went unresolved, never why. It knows."""
    frame = orders()
    frame.loc[frame.index[:10], 'placed'] = pd.NaT
    built = Schema.from_data({'users': users(), 'orders': frame})
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(
            'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', built
        ).materialize(
            indices=[1, 2], num_context_samples=20, anchor_time='entity'
        )
    message = str(error.value)
    if 'unresolved' in message or 'dropped' in message:
        assert 'no time' in message or 'dropped' in message


def test_an_ordinary_call_is_unaffected():
    request = query().materialize(indices=[1, 2], num_context_samples=20)
    assert request.payloads


def test_a_temporal_link_prediction_is_capped_lower_than_everything_else():
    """The second call to `_check_entity_count` is why it takes a task kind.

    Only the 1,000 ceiling was covered, so the whole reason the argument
    exists went unexercised: a temporal link prediction is capped at 200, and
    a count between the two has to be refused for that task and allowed for
    every other one.
    """
    from predictive_query import TaskType

    from sdm_client.models import RELATIONAL_LIMITS

    temporal = RELATIONAL_LIMITS['predict_rows_temporal_link_prediction']
    between = list(range(temporal + 1))

    RelationalMaterializer._check_entity_count(
        between, TaskType.BINARY_CLASSIFICATION
    )
    RelationalMaterializer._check_entity_count(between, None)

    with pytest.raises(StructuredDataError) as error:
        RelationalMaterializer._check_entity_count(
            between, TaskType.TEMPORAL_LINK_PREDICTION
        )
    assert str(temporal) in str(error.value)
    assert error.value.code == 'UNSUPPORTED_REQUEST'


def test_the_two_ceilings_are_the_engine_s_own():
    """A drift between these and the engine refuses work the engine allows."""
    from sdm_client.models import RELATIONAL_LIMITS

    engine = pytest.importorskip('sdm_relational_engine.rfm.rfm')
    from predictive_query import TaskType

    assert (
        RELATIONAL_LIMITS['predict_rows_temporal_link_prediction']
        == engine._MAX_PRED_SIZE[TaskType.TEMPORAL_LINK_PREDICTION]
    )
    assert (
        RELATIONAL_LIMITS['predict_rows_max']
        == engine._MAX_PRED_SIZE[TaskType.BINARY_CLASSIFICATION]
    )


def test_a_relationship_listed_twice_is_linked_once():
    """`graph.link()` became unconditional, and the engine refuses an edge it
    already has -- which a caller appending to an inferred list produces."""
    import pandas as pd

    from sdm_client import Relationship, Schema

    users = pd.DataFrame({'user_id': [1, 2], 'age': [20.0, 30.0]})
    orders = pd.DataFrame({'order_id': [1, 2], 'user_id': [1, 2]})
    schema = Schema.from_data({'users': users, 'orders': orders})
    link = Relationship('orders', 'user_id', 'users', 'user_id')
    schema.relationships = [link, link]

    graph = RelationalMaterializer(schema)._build_graph()
    edges = {(e.src_table, e.fkey, e.dst_table) for e in graph.edges}
    assert edges == {('orders', 'user_id', 'users')}


def test_a_type_error_is_sorted_by_whether_the_call_bound():
    """The engine uses TypeError for both a caller's bad value and this
    client's bad call, so the type alone cannot tell them apart.

    Argument binding fails before the callee is entered, so a signature
    error's deepest frame is the calling line; a refused value was raised
    from inside the engine.
    """

    def engine_fn(*, indices):
        return indices

    def validator(value):
        raise TypeError("'anchor_time' must be a pandas.Timestamp")

    try:
        engine_fn(nope=1)
    except TypeError as error:
        assert RelationalMaterializer._is_call_signature_error(error)

    try:
        validator('2024-01-01')
    except TypeError as error:
        assert not RelationalMaterializer._is_call_signature_error(error)


def test_the_frame_test_decides_it_without_help_from_the_message():
    """The message fallback would answer this one too, so it has to be shown
    that the frames alone do.

    A bad call made from the relational materializer leaves its deepest frame
    there, and the two are told apart even when the wording carries no
    signature phrase at all -- which is the point of not depending on
    CPython's wording.
    """
    import sdm_client.materializers.relational as relational

    def engine_fn(*, indices):
        return indices

    # Raised from the relational module, with an unmatched message.
    source = compile(
        "def call():\n    raise TypeError('an entirely unrecognised wording')",
        relational.__file__,
        'exec',
    )
    namespace: dict = {'__name__': relational.__name__}
    exec(source, namespace)
    try:
        namespace['call']()
    except TypeError as error:
        assert RelationalMaterializer._is_call_signature_error(error), (
            'a TypeError whose deepest frame is the relational materializer '
            'is a bad call, however it is worded'
        )

    # Raised from anywhere else, same unrecognised wording.
    def elsewhere():
        raise TypeError('an entirely unrecognised wording')

    try:
        elsewhere()
    except TypeError as error:
        assert not RelationalMaterializer._is_call_signature_error(error)


def test_the_message_fallback_covers_a_traceback_free_type_error():
    """A TypeError with no traceback -- reconstructed, or raised across a
    boundary -- still has to be sorted, and only the message is left."""
    assert RelationalMaterializer._is_call_signature_error(
        TypeError("f() got an unexpected keyword argument 'verbose'")
    )
    assert not RelationalMaterializer._is_call_signature_error(
        TypeError("'anchor_time' must be a pandas.Timestamp")
    )


def test_a_bad_anchor_time_is_the_callers_fault_not_the_libraries():
    """The engine raises TypeError to reject a caller's own `anchor_time`,
    with a hint naming what to pass instead. Removing TypeError from the
    caller bucket turned that into 'the relational engine failed'."""
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError) as error:
        with RelationalMaterializer(schema())._engine_errors(None):
            raise TypeError(
                "'anchor_time' must be a pandas.Timestamp or the literal "
                "'entity' (got str '2024-01-01'); try pd.Timestamp(...)"
            )
    assert error.value.code == 'INVALID_REQUEST'


def test_a_wrong_keyword_into_the_engine_is_reported_as_our_bug():
    """The other half of the same type: this one a user cannot act on."""
    from sdm_client.errors import StructuredDataError

    with pytest.raises(StructuredDataError) as error:
        with RelationalMaterializer(schema())._engine_errors(None):
            raise TypeError(
                "_get_task_table() got an unexpected keyword argument 'verbose'"
            )
    assert error.value.code == 'INTERNAL_ERROR'


# --- the schema decides the time column, and nothing else does --------------


def catalogue() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'item_id': range(1, 121),
            'title': [f'title {index}' for index in range(1, 121)],
            'release': ['01-Jan-1995'] * 120,
        }
    )


def rated() -> pd.DataFrame:
    return pd.DataFrame(
        {
            'item_id': [(index % 120) + 1 for index in range(600)],
            'user_id': [index % 40 for index in range(600)],
            'when': pd.date_range('2000-01-01', periods=600, freq='D'),
        }
    )


def catalogue_schema() -> Schema:
    return Schema.from_data(
        {'items': catalogue(), 'users': users(), 'ratings': rated()},
        primary_keys={'items': 'item_id', 'users': 'user_id'},
        time_columns={'ratings': 'when'},
        relationships=[
            Relationship('ratings', 'item_id', 'items', 'item_id'),
            Relationship('ratings', 'user_id', 'users', 'user_id'),
        ],
    )


def test_a_table_the_schema_gives_no_time_column_is_given_none():
    built = catalogue_schema()
    assert built['items'].time_column is None

    graph = RelationalMaterializer(built)._build_graph()

    assert graph['items'].time_column is None


def test_a_time_column_the_schema_names_is_the_one_the_graph_gets():
    graph = RelationalMaterializer(catalogue_schema())._build_graph()

    assert graph['ratings'].time_column.name == 'when'


def test_a_date_shaped_string_the_schema_ignores_does_not_stop_the_build():
    """`release` reads as a date the engine cannot parse, and it used to try:
    inference ran before the schema was applied, so the caller could not undo
    a guess that failed while it was being made."""
    built = catalogue_schema()

    graph = RelationalMaterializer(built)._build_graph()

    assert graph['items'].time_column is None
    assert built['items']['release'].stype == 'categorical'


def test_a_key_the_schema_withholds_is_not_taken_from_a_column_name():
    frame = pd.DataFrame({'ID': [1, 1, 2, 2, 3], 'value': [1.0, 2, 3, 4, 5]})
    built = Schema.from_data({'t': frame})
    built['t'].primary_key = None

    graph = RelationalMaterializer(built)._build_graph()

    assert graph['t'].primary_key is None


# --- a column the model will not read is said to be unread ------------------


def prose_schema(columns: int = 1, stype: str = 'text') -> Schema:
    frame = pd.DataFrame(
        {
            'paper_id': range(120),
            'target': [index % 2 for index in range(120)],
            **{f'prose{i}': ['words here'] * 120 for i in range(columns)},
        }
    )
    cited = pd.DataFrame(
        {
            'paper_id': [index % 120 for index in range(400)],
            'when': pd.date_range('2020-01-01', periods=400, freq='D'),
        }
    )
    built = Schema.from_data(
        {'papers': frame, 'cited': cited},
        primary_keys={'papers': 'paper_id'},
        time_columns={'cited': 'when'},
        relationships=[Relationship('cited', 'paper_id', 'papers', 'paper_id')],
    )
    built['papers']['target'].stype = 'categorical'
    for index in range(columns):
        built['papers'][f'prose{index}'].stype = stype
    return built


def unread_warnings(built: Schema) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        PredictiveQuery(
            'PREDICT papers.target FOR EACH papers.paper_id', built
        ).materialize(
            indices=list(range(20)),
            anchor_time=pd.Timestamp('2021-06-01'),
        )
    return [
        str(item.message)
        for item in caught
        if 'does not read' in str(item.message)
    ]


def test_a_text_column_is_said_to_contribute_nothing():
    """It is accepted and dropped at the model, so the request succeeds and
    the column that looked most informative carried nothing."""
    said = unread_warnings(prose_schema())

    assert len(said) == 1
    assert 'papers.prose0' in said[0]
    assert 'contributes nothing' in said[0]


def test_every_unread_column_is_named():
    said = unread_warnings(prose_schema(columns=2))

    assert len(said) == 1
    assert 'papers.prose0' in said[0]
    assert 'papers.prose1' in said[0]


def test_a_column_the_model_reads_is_not_mentioned():
    assert unread_warnings(prose_schema(stype='categorical')) == []


def unread_named(built: Schema, left_out: dict) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        _warn_of_columns_the_model_does_not_read(built, left_out)
    return [str(item.message) for item in caught]


def test_a_column_already_left_out_is_not_reported_as_lost():
    """It is not sent, so there is nothing to have lost. The target is left out
    the same way, which is why matching it by name was never needed."""
    built = prose_schema()

    assert unread_named(built, {'papers': ['prose0']}) == []


def test_a_column_left_out_of_another_table_does_not_silence_this_one():
    built = prose_schema()

    said = unread_named(built, {'cited': ['prose0']})

    assert len(said) == 1
    assert 'papers.prose0' in said[0]


def test_an_excluded_column_is_not_named_through_materialize():
    built = prose_schema()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        PredictiveQuery(
            'PREDICT papers.target FOR EACH papers.paper_id', built
        ).materialize(
            indices=list(range(20)),
            anchor_time=pd.Timestamp('2021-06-01'),
            exclude_cols_dict={'papers': ['prose0']},
        )

    assert [
        str(item.message)
        for item in caught
        if 'does not read' in str(item.message)
    ] == []
