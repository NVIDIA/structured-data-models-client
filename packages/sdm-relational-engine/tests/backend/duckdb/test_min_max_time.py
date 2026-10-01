# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The time-range probe labels each ``UNION ALL`` branch by integer position
rather than by pasting the graph table name into a string literal.

The name is not necessarily one the caller chose, discovery adopts whatever
the schema holds, and doubling the quote character does not neutralise the
backslash form on Snowflake or Databricks, where the injection lives; see
``test_snow_min_max_probe_keeps_table_names_out_of_the_sql``. This test covers
the other half: an exotic name must still come back as the dictionary key.
"""

import warnings
from pathlib import Path

import pandas as pd
import pytest
from sdm_relational_engine.rfm import Graph

duckdb = pytest.importorskip(
    'duckdb', reason="'duckdb' extension not installed"
)

from sdm_relational_engine.rfm.backend.duckdb import (  # noqa: E402
    DuckDBSampler,
)

_HOSTILE = "events\\' UNION ALL SELECT 'pwned', NULL, NULL --"


@pytest.fixture(params=['events', _HOSTILE])
def graph(request, tmp_path: Path) -> Graph:
    name = request.param
    path = tmp_path / 'events.duckdb'
    connection = duckdb.connect(path)
    connection.execute(
        f'CREATE TABLE "{name.replace(chr(34), chr(34) * 2)}" AS '
        "SELECT i AS event_id, TIMESTAMP '2020-01-01' + INTERVAL (i) DAY "
        'AS ts FROM range(10) t(i)'
    )
    connection.close()

    return Graph.from_duckdb(
        path,
        tables=[dict(name=name, primary_key='event_id', time_column='ts')],
        verbose=False,
    )


def test_min_max_time_is_keyed_by_table_name(graph: Graph) -> None:
    name = next(iter(graph.tables))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        sampler = DuckDBSampler(graph, verbose=False)

    out = sampler._get_min_max_time_dict([name])

    assert set(out) == {name}
    assert out[name] == (pd.Timestamp('2020-01-01'), pd.Timestamp('2020-01-10'))
