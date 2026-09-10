# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
