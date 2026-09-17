# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One table can reach another by more than one column, and often should.

A sales line names the customer it was sold to and the one it ships to, and
those are different facts. But an aggregation across a pair of tables has to
resolve to one column, so a query crossing an ambiguous pair is refused -- and
it was refused at the moment the query was asked, which reads as a problem
with the query rather than with the graph it was written against. Worse, the
one instruction that would fix it named `graph.unlink`, which a caller holding
a `Schema` does not have.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from sdm_client import KumoRelational, PredictiveQuery, Relationship, Schema
from sdm_client.errors import StructuredDataError

RNG = np.random.default_rng(0)
QUERY = (
    'PREDICT COUNT(salesdocumentitem.*, 0, 30, days) FOR EACH customer.CUSTOMER'
)


def schema_reaching_by(columns: list[str], *, quiet: bool = True) -> Schema:
    """`quiet` off for the tests that read what building it says: suppressing
    warnings inside the helper is what makes such a test catch nothing."""
    customer = pd.DataFrame(
        {'CUSTOMER': range(60), 'seg': RNG.choice([*'xy'], 60)}
    )
    item = {
        'ID': range(400),
        'amt': RNG.gamma(2.0, 10.0, 400),
        'ts': pd.to_datetime('2023-01-01')
        + pd.to_timedelta(RNG.integers(0, 150, 400), 'D'),
    }
    for column in columns:
        item[column] = RNG.integers(0, 60, 400)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore' if quiet else 'always')
        return Schema.from_data(
            {'customer': customer, 'salesdocumentitem': pd.DataFrame(item)},
            primary_keys={'customer': 'CUSTOMER', 'salesdocumentitem': 'ID'},
            time_columns={'salesdocumentitem': 'ts'},
            relationships=[
                Relationship('salesdocumentitem', c, 'customer', 'CUSTOMER')
                for c in columns
            ],
        )


def said_building(columns: list[str]) -> str:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        schema_reaching_by(columns, quiet=False)
    return ' '.join(
        str(w.message) for w in caught if 'reaches' in str(w.message)
    )


def test_it_is_said_when_the_graph_is_built_not_when_a_query_runs():
    said = said_building(['SOLDTOPARTY', 'SHIPTOPARTY'])

    assert "'salesdocumentitem' reaches 'customer'" in said
    assert "'SHIPTOPARTY' and 'SOLDTOPARTY'" in said


def test_one_way_to_reach_a_table_says_nothing():
    assert said_building(['SOLDTOPARTY']) == ''


def test_every_competing_column_is_named():
    said = said_building(
        ['SOLDTOPARTY', 'SHIPTOPARTY', 'BILLTOPARTY', 'PAYERPARTY']
    )

    for column in ('SOLDTOPARTY', 'SHIPTOPARTY', 'BILLTOPARTY', 'PAYERPARTY'):
        assert repr(column) in said
    assert 'They are all kept' in said


def test_nothing_is_dropped_on_the_caller_s_behalf():
    """Which link a question means decides what the model sees, so it stays
    the caller's to make."""
    schema = schema_reaching_by(['SOLDTOPARTY', 'SHIPTOPARTY'])

    assert len(schema.relationships) == 2


def test_the_refusal_names_something_a_schema_holder_can_do():
    pytest.importorskip('sdm_relational_engine')

    schema = schema_reaching_by(['SOLDTOPARTY', 'SHIPTOPARTY'])

    with pytest.raises(StructuredDataError) as refused:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            PredictiveQuery(QUERY, schema).materialize(
                indices=[1, 2], model=KumoRelational()
            )

    said = str(refused.value)
    assert (
        'schema.relationships = [link for link in schema.relationships' in said
    )
    assert "'SOLDTOPARTY'" in said


def test_following_the_advice_makes_the_query_resolve():
    """The whole point: the instruction has to work."""
    pytest.importorskip('sdm_relational_engine')

    schema = schema_reaching_by(['SOLDTOPARTY', 'SHIPTOPARTY'])
    schema.relationships = [
        link
        for link in schema.relationships
        if link.source_column not in ['SOLDTOPARTY']
    ]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[1, 2], model=KumoRelational()
        )

    assert request.predict_rows == 2
