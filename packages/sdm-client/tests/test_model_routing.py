# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""Which model a request is built for is the caller's choice, not a side
effect of how many tables the schema happens to have.

`materialize()` used to pick the route by counting tables, and the model was
named later at `predict()`. For a one-table schema those two decisions could
disagree with no way to reconcile them: the request was always built for the
tabular model, so asking the relational model to answer it was refused, and
`materialize()` took no argument that could say otherwise.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Schema
from sdm_client.adapters.tabular import TabularAdapter
from sdm_client.errors import StructuredDataError, UnknownModelError
from sdm_client.models import KumoRelational, KumoTabular
from sdm_client.schema import Relationship

ONE_TABLE_QUERY = 'PREDICT users.churn FOR EACH users.user_id'
TWO_TABLE_QUERY = (
    'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'
)


def engine_or_skip() -> None:
    """The relational route builds through the native sampler."""
    pytest.importorskip('sdm_relational_engine')


def one_table(rows: int = 60) -> Schema:
    frame = pd.DataFrame(
        {
            'user_id': range(1, rows + 1),
            'churn': ['yes', 'no'] * (rows // 2),
            'tenure': [float(index % 24) for index in range(rows)],
        }
    )
    schema = Schema.from_data({'users': frame}, relationships=[])
    schema['users'].primary_key = 'user_id'
    return schema


def two_tables(orders: int = 240, users: int = 12) -> Schema:
    people = pd.DataFrame(
        {'user_id': range(1, users + 1), 'segment': list('abc') * (users // 3)}
    )
    sales = pd.DataFrame(
        {
            'order_id': range(orders),
            'user_id': [1 + index % users for index in range(orders)],
            'amount': [10.0 + index for index in range(orders)],
            'placed_at': pd.to_datetime('2025-01-01')
            + pd.to_timedelta([index * 2 for index in range(orders)], unit='D'),
        }
    )
    schema = Schema.from_data(
        {'users': people, 'orders': sales},
        relationships=[Relationship('orders', 'user_id', 'users', 'user_id')],
    )
    schema['users'].primary_key = 'user_id'
    schema['orders'].primary_key = 'order_id'
    schema['orders'].time_column = 'placed_at'
    return schema


def spending(rows: int = 60) -> Schema:
    """Like :func:`one_table`, with a numerical target the tabular model
    also serves, so a refusal cannot be the task kind in disguise."""
    frame = pd.DataFrame(
        {
            'user_id': range(1, rows + 1),
            'churn': ['yes', 'no'] * (rows // 2),
            'spend': [100.0 + index * 3 for index in range(rows)],
        }
    )
    schema = Schema.from_data({'users': frame}, relationships=[])
    schema['users'].primary_key = 'user_id'
    return schema


# --- the model the caller names decides the route ---------------------------


def test_one_table_can_be_built_for_the_relational_model():
    engine_or_skip()
    request = PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
        indices=[1, 2, 3], model='kumo-relational'
    )
    assert request.payloads is not None


def test_one_table_still_builds_for_the_tabular_model_when_asked():
    request = PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
        indices=[1, 2, 3], model='kumo-tabular'
    )
    assert request.payloads is None


def test_a_model_handle_names_the_route_as_well_as_an_id():
    request = PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
        indices=[1, 2, 3], model=KumoTabular()
    )
    assert request.payloads is None


def test_naming_the_relational_model_by_handle_routes_relationally():
    engine_or_skip()
    request = PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
        indices=[1, 2, 3], model=KumoRelational()
    )
    assert request.payloads is not None


# --- leaving it unsaid keeps the shape-based default ------------------------


def test_one_table_defaults_to_tabular():
    request = PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
        indices=[1, 2, 3]
    )
    assert request.payloads is None


def test_several_tables_default_to_relational():
    engine_or_skip()
    request = PredictiveQuery(TWO_TABLE_QUERY, two_tables()).materialize(
        indices=[1, 2, 3]
    )
    assert request.payloads is not None


# --- a route the schema cannot support is refused ---------------------------


def test_the_tabular_model_is_refused_a_schema_with_relationships():
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(TWO_TABLE_QUERY, two_tables()).materialize(
            indices=[1, 2, 3], model='kumo-tabular'
        )
    assert error.value.code == 'INVALID_REQUEST'
    assert 'kumo-tabular' in str(error.value)


def test_the_tabular_model_is_refused_a_second_table_carrying_no_relationship():
    """Two unrelated tables are still two. This is the count alone, with no
    relationship for the neighbouring check to catch instead."""
    schema = Schema.from_data(
        {
            'users': one_table()['users'].data,
            'notes': pd.DataFrame({'note_id': [1, 2], 'body': ['a', 'b']}),
        },
        relationships=[],
    )
    assert not schema.relationships
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(ONE_TABLE_QUERY, schema).materialize(
            indices=[1, 2, 3], model='kumo-tabular'
        )
    assert 'answers from a single table' in str(error.value)


def test_a_supplied_frame_is_refused_for_the_relational_model():
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
            predict=pd.DataFrame({'user_id': [901], 'tenure': [3.0]}),
            model='kumo-relational',
        )
    assert error.value.code == 'INVALID_REQUEST'


def test_an_unknown_model_is_refused_before_anything_is_gathered():
    with pytest.raises(UnknownModelError):
        PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
            indices=[1, 2, 3], model='gpt-4'
        )


def test_a_model_that_is_neither_a_handle_nor_an_id_is_refused():
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
            indices=[1, 2, 3], model=object()
        )
    assert error.value.code == 'INVALID_REQUEST'


# --- gathering options follow the route, not the table count ----------------


def test_sampling_options_are_refused_on_the_tabular_route():
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
            indices=[1, 2, 3], model='kumo-tabular', num_neighbors=[8, 4]
        )
    assert error.value.code == 'INVALID_REQUEST'
    assert 'num_neighbors' in str(error.value)


def test_sampling_options_are_accepted_when_one_table_goes_relational():
    engine_or_skip()
    request = PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
        indices=[1, 2, 3], model='kumo-relational', num_neighbors=[8, 4]
    )
    assert request.payloads is not None


# --- a request gathered for one route is not fed to the other ---------------


def test_a_relationally_gathered_request_is_refused_by_the_tabular_model():
    engine_or_skip()
    request = PredictiveQuery(
        'PREDICT users.spend FOR EACH users.user_id', spending()
    ).materialize(indices=[1, 2, 3], model='kumo-relational')
    adapter = TabularAdapter()
    with pytest.raises(StructuredDataError) as error:
        list(adapter.prediction_plan(request, ('prediction',)))
    assert error.value.code == 'UNSUPPORTED_REQUEST'
    assert 'kumo-relational' in str(error.value)


def test_a_tabular_request_refused_by_the_relational_model_says_how_to_gather_it():
    request = PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
        indices=[1, 2, 3], model='kumo-tabular'
    )
    relational = pytest.importorskip('sdm_client.adapters.relational')
    adapter = relational.RelationalAdapter()
    with pytest.raises(StructuredDataError) as error:
        list(adapter.prediction_plan(request, ('prediction',)))
    assert error.value.code == 'UNSUPPORTED_REQUEST'
    assert "model='kumo-relational'" in str(error.value)


# --- relationships make a schema relational, whatever its table count -------


def self_referencing() -> Schema:
    """One table that points at itself: `is_relational` is true on one table."""
    frame = pd.DataFrame(
        {
            'emp_id': range(1, 41),
            'manager_id': [None] + [1 + index % 5 for index in range(39)],
            'left': ['yes', 'no'] * 20,
            'tenure': [float(index % 24) for index in range(40)],
        }
    )
    schema = Schema.from_data(
        {'employees': frame},
        relationships=[
            Relationship('employees', 'manager_id', 'employees', 'emp_id')
        ],
    )
    schema['employees'].primary_key = 'emp_id'
    return schema


SELF_QUERY = 'PREDICT employees.left FOR EACH employees.emp_id'


def test_one_table_with_a_relationship_defaults_to_relational():
    engine_or_skip()
    request = PredictiveQuery(SELF_QUERY, self_referencing()).materialize(
        indices=[2, 3]
    )
    assert request.payloads is not None


def test_the_tabular_model_is_refused_a_relationship_on_one_table():
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(SELF_QUERY, self_referencing()).materialize(
            indices=[2, 3], model='kumo-tabular'
        )
    assert error.value.code == 'INVALID_REQUEST'
    assert 'relationship' in str(error.value)


# --- a refusal names a way out that works -----------------------------------


def test_options_beside_a_supplied_frame_do_not_advise_a_route_that_refuses_it():
    """Naming the relational model alone would be refused for the frame, so
    the remediation has to name indices=[...] as well."""
    frame = pd.DataFrame({'user_id': [901], 'tenure': [3.0]})
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
            predict=frame, num_neighbors=[8, 8]
        )
    message = str(error.value)
    assert 'num_neighbors' in message
    assert 'indices=[...]' in message


def test_asking_with_both_rows_and_keys_says_so_before_anything_about_routes():
    frame = pd.DataFrame({'user_id': [901], 'tenure': [3.0]})
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
            predict=frame, indices=[1], model='kumo-relational'
        )
    assert 'not both' in str(error.value)


# --- a handle is only worth as much as the spec it carries ------------------


def test_a_bare_model_handle_is_refused_as_a_client_error():
    """`Model` is exported, so it can be constructed. It carries no spec, and
    reading one off it left an AttributeError where this client documents a
    refusal."""
    from sdm_client.models import Model

    with pytest.raises(UnknownModelError):
        PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
            indices=[1, 2, 3], model=Model()
        )


def test_a_handle_carrying_an_unregistered_spec_is_refused():
    """A handle is a name for one of the models this contract carries, not a
    way to describe a new one. Its spec decides the route and the limits the
    request is checked against, and the id it names is a closed enum on the
    wire, so an unregistered one would gather work that cannot be sent."""
    from predictive_query import RELATIONAL as RELATIONAL_PROFILE

    from sdm_client.base import ModelSpec, RuntimeContract
    from sdm_client.models import Model

    class Invented(Model):
        spec = ModelSpec(
            model_id='not-a-real-model',
            profile=RELATIONAL_PROFILE,
            runtime=RuntimeContract(
                tasks=frozenset({'classification'}),
                outputs=frozenset({'prediction'}),
            ),
        )

    with pytest.raises(UnknownModelError) as error:
        PredictiveQuery(ONE_TABLE_QUERY, one_table()).materialize(
            indices=[1, 2, 3], model=Invented()
        )
    assert 'not-a-real-model' in str(error.value)


def test_the_shipped_handles_are_still_accepted():
    from sdm_client.models import KUMO_RELATIONAL, KUMO_TABULAR, spec_for

    assert spec_for(KumoTabular()) is KUMO_TABULAR
    assert spec_for(KumoRelational()) is KUMO_RELATIONAL


def test_a_frame_on_a_self_referencing_schema_is_told_what_would_work():
    """Gathering it for the tabular model is refused for the relationship, so
    naming that model is not a way out of this one."""
    frame = pd.DataFrame({'emp_id': [901], 'manager_id': [1], 'tenure': [3.0]})
    with pytest.raises(StructuredDataError) as error:
        PredictiveQuery(SELF_QUERY, self_referencing()).materialize(
            predict=frame
        )
    message = str(error.value)
    assert 'indices=[...]' in message
    assert 'drop the relationship' in message


# --- capabilities only names what a caller can reach -------------------------


def test_no_explanation_format_is_advertised_that_cannot_be_asked_for():
    """Two were, and neither appears in any response: nothing on this client
    takes a format argument, and the endpoint names its own."""
    from sdm_client import KumoRelational

    assert KumoRelational().spec.runtime.explanation_formats == ()


def test_the_model_that_explains_still_says_so_through_its_outputs():
    from sdm_client import KumoRelational

    assert 'explanation' in KumoRelational().spec.runtime.outputs
