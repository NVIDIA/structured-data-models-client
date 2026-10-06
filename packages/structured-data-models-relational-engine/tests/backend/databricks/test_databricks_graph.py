# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from sdm_relational_engine.rfm import Edge, Graph

try:
    from sdm_relational_engine.rfm.backend.databricks import (
        Connection,
        DatabricksTable,
    )
except ImportError:
    pytest.skip("'databricks' extension not installed", allow_module_level=True)


def test_graph(
    connection: Connection,
    catalog: str,
    schema: str,
) -> None:
    graph = Graph(
        tables=[
            DatabricksTable(
                connection,
                name='customers',
                catalog=catalog,
                schema=schema,
                primary_key='customer_id',
            ),
            DatabricksTable(
                connection,
                name='order_lines',
                catalog=catalog,
                schema=schema,
                time_column='order_date',
            ),
            DatabricksTable(
                connection,
                name='products',
                catalog=catalog,
                schema=schema,
                primary_key='product_id',
            ),
        ],
        edges=[
            ('order_lines', 'customer_id', 'customers'),
            ('order_lines', 'product_id', 'products'),
        ],
    )

    assert len(graph.tables) == 3
    assert graph.backend == 'databricks'

    assert graph['customers'].primary_key is not None
    assert graph['customers'].primary_key.name == 'customer_id'
    assert graph['products'].primary_key is not None
    assert graph['products'].primary_key.name == 'product_id'

    assert graph['order_lines'].time_column is not None
    assert graph['order_lines'].time_column.name == 'order_date'

    assert Edge('order_lines', 'customer_id', 'customers') in graph.edges
    assert Edge('order_lines', 'product_id', 'products') in graph.edges


def test_from_databricks(
    connection: Connection,
    catalog: str,
    schema: str,
) -> None:
    graph = Graph.from_databricks(
        connection=connection,
        catalog=catalog,
        schema=schema,
        tables=['customers', 'order_lines', 'products'],
        verbose=False,
    )

    assert len(graph.tables) == 3

    # Primary keys and links are inferred heuristically, since the ERP dataset
    # declares no constraints:
    assert graph['customers'].primary_key is not None
    assert graph['customers'].primary_key.name == 'customer_id'
    assert graph['products'].primary_key is not None
    assert graph['products'].primary_key.name == 'product_id'

    assert graph['order_lines'].time_column is not None
    assert graph['order_lines'].time_column.name == 'order_date'

    assert Edge('order_lines', 'customer_id', 'customers') in graph.edges
    assert Edge('order_lines', 'product_id', 'products') in graph.edges


def test_from_databricks_all_tables(
    connection: Connection,
    catalog: str,
    schema: str,
) -> None:
    # Discover all tables in the schema (no explicit table list):
    graph = Graph.from_databricks(
        connection=connection,
        catalog=catalog,
        schema=schema,
        infer_metadata=False,
        verbose=False,
    )
    assert len(graph.tables) > 0
    assert 'customers' in graph
    assert graph['customers'].source_name == f'{catalog}.{schema}.customers'
