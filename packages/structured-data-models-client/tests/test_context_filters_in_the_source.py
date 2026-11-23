# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A selective query still gathers the context it asks for.

The client draws its context examples from a seeded sample of the entity
table, ten times the context's size. When the query's ``WHERE`` keeps one
entity in sixty, or most entities have no value for a static target, a draw
of every entity keeps a sixtieth, or a twentieth, of what was asked for,
though the table holds many more that qualify. The conditions on the
entity's own row go to the source with the sample, so the draw is of the
entities that may qualify.
"""

from __future__ import annotations

import warnings
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema

USERS = 120_000
ASKED = 1_000


def _connection() -> Any:
    connection = duckdb.connect()
    # One user in sixty is older than 76; one in twenty has a label.
    connection.execute(
        f"""
        CREATE TABLE users AS
        SELECT u AS id,
               18 + u % 60 AS age,
               u % 2 = 0 AS churned,
               CASE WHEN u % 20 = 0 THEN u % 3 = 0 END AS labelled,
               TIMESTAMP '2024-01-01' AS joined
        FROM range({USERS}) t(u)
        """
    )
    connection.execute('ALTER TABLE users ADD PRIMARY KEY (id)')
    return connection


def _context_rows(query: str) -> int:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
        request = PredictiveQuery(query, schema).materialize(
            indices=[1, 2],
            anchor_time=pd.Timestamp('2024-03-01'),
            num_context_samples=ASKED,
        )
    return request.context_rows


@pytest.mark.parametrize(
    'query',
    [
        'PREDICT users.churned FOR EACH users.id WHERE users.age > 76',
        'PREDICT users.churned FOR EACH users.id WHERE users.age IN (77, 200)',
        'PREDICT users.churned FOR EACH users.id WHERE NOT users.age <= 76',
    ],
)
def test_a_selective_filter_on_the_entity_fills_the_context(query: str) -> None:
    # 2,000 users qualify; a draw of every user would keep about 167.
    assert _context_rows(query) == ASKED


def test_a_sparse_static_target_fills_the_context() -> None:
    # 6,000 users have a label; a draw of every user would keep about 500.
    assert _context_rows('PREDICT users.labelled FOR EACH users.id') == ASKED
