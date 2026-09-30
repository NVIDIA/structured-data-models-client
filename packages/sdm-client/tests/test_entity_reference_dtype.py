# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The entity's transport key must be the type of the key it joins to.

`_entity_values` read the key with `df.iloc[row][key]`, which builds a Series
for the whole row, and a row has one dtype. An integer key sitting beside a
float column therefore came back as a float, the payload declared the entity
reference `float64` while the key it joins to was `int64`, and the endpoint
refused the request:

    422 INVALID_SCHEMA relationship joins '__kumo_entity_ref_0' declared
                       'float64' to 'user_id' declared 'int64'

Nothing the caller could change avoided it, and the message named two internal
columns they had never seen.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Schema

pytest.importorskip('sdm_relational_engine')

QUERY = 'PREDICT users.tier FOR EACH users.user_id'


def orders(
    rows: int = 120, users: int = 40, *, timed: bool = True
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            'order_id': range(rows),
            'user_id': [index % users for index in range(rows)],
            'price': [10.0 + index for index in range(rows)],
        }
    )
    if timed:
        frame['placed'] = pd.date_range('2024-01-01', periods=rows, freq='D')
    return frame


def users(rows: int = 40, *, numeric_column: bool = True) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            'user_id': range(rows),
            'tier': [
                'gold' if index % 2 else 'silver' for index in range(rows)
            ],
        }
    )
    if numeric_column:
        # The column that used to promote the key.
        frame['age'] = [20.0 + index for index in range(rows)]
    return frame


def declared(
    schema: Schema, query: str = QUERY, indices=(1, 2)
) -> tuple[str, str]:
    payload = (
        PredictiveQuery(query, schema)
        .materialize(indices=list(indices), num_context_samples=50)
        .payloads[0]
    )
    reference = next(
        spec['dtype']
        for name, spec in payload['schema']['instance_table']['columns'].items()
        if name.startswith('__kumo_entity_ref')
    )
    key = payload['schema']['related_tables']['users']['columns']['user_id'][
        'dtype'
    ]
    return reference, key


def test_a_float_column_beside_the_key_does_not_retype_it():
    reference, key = declared(
        Schema.from_data({'users': users(), 'orders': orders()})
    )
    assert reference == key == 'int64'


def test_the_reference_matches_the_key_without_that_column_too():
    schema = Schema.from_data(
        {'users': users(numeric_column=False), 'orders': orders()}
    )
    reference, key = declared(schema)
    assert reference == key == 'int64'


def test_a_string_key_is_still_a_string():
    frame = users()
    frame['user_id'] = frame['user_id'].astype(str)
    related = orders()
    related['user_id'] = related['user_id'].astype(str)
    reference, key = declared(
        Schema.from_data({'users': frame, 'orders': related}),
        indices=('1', '2'),
    )
    assert reference == key == 'string'


def test_a_graph_with_no_time_column_is_unaffected():
    schema = Schema.from_data({'users': users(), 'orders': orders(timed=False)})
    reference, key = declared(schema)
    assert reference == key == 'int64'


def test_a_temporal_aggregation_is_unaffected():
    schema = Schema.from_data({'users': users(), 'orders': orders()})
    reference, key = declared(
        schema, 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
    )
    assert reference == key == 'int64'


def test_the_reference_carries_the_entity_values_it_stands_for():
    """Correctness, not only the dtype: the wrong read gave 1.0 for 1."""
    schema = Schema.from_data({'users': users(), 'orders': orders()})
    payload = (
        PredictiveQuery(QUERY, schema)
        .materialize(indices=[1, 2, 3], num_context_samples=50)
        .payloads[0]
    )
    table = payload['predict']['instance_table']
    position = next(
        index
        for index, name in enumerate(table['columns'])
        if name.startswith('__kumo_entity_ref')
    )
    assert [row[position] for row in table['rows']] == [1, 2, 3]
