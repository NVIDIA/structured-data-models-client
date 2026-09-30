# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from sdm_relational_engine.rfm import Graph
    from sdm_relational_engine.rfm.backend.databricks import Connection

# The Databricks integration tests run read-only against a pre-populated ERP
# dataset, configured through the following environment variables:
_REQUIRED_ENV = [
    'DATABRICKS_SERVER_HOSTNAME',
    'DATABRICKS_HTTP_PATH',
    'DATABRICKS_TOKEN',
    'DATABRICKS_CATALOG',
    'DATABRICKS_SCHEMA',
]


_HERE = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    # Marked from the fixture closure rather than by hand, so a new test that
    # reaches the warehouse cannot be left unmarked. This hook sees every item
    # in the session, hence the directory check.
    for item in items:
        if item.path.is_relative_to(_HERE) and 'connection' in getattr(
            item, 'fixturenames', ()
        ):
            item.add_marker(pytest.mark.live_warehouse)


@pytest.fixture(scope='session')
def connection() -> Generator['Connection', None, None]:
    databricks = pytest.importorskip(
        'sdm_connectors.backends.databricks',
        reason="'databricks' extension not installed",
    )

    missing = [name for name in _REQUIRED_ENV if not os.getenv(name)]
    if missing:
        pytest.skip(f'Missing Databricks credentials: {", ".join(missing)}')

    connection = databricks.connect(
        server_hostname=os.environ['DATABRICKS_SERVER_HOSTNAME'],
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN'],
        catalog=os.environ['DATABRICKS_CATALOG'],
        schema=os.environ['DATABRICKS_SCHEMA'],
    )
    yield connection
    connection.close()


@pytest.fixture(scope='session')
def catalog(connection: 'Connection') -> str:
    return os.environ['DATABRICKS_CATALOG']


@pytest.fixture(scope='session')
def schema(connection: 'Connection') -> str:
    return os.environ['DATABRICKS_SCHEMA']


@pytest.fixture(scope='session')
def graph(
    connection: 'Connection',
    catalog: str,
    schema: str,
) -> 'Graph':
    r"""A small curated graph over the ERP dataset:
    ``order_lines`` (facts) linking to ``customers`` and ``products``.
    """
    from sdm_relational_engine.rfm import Graph

    return Graph.from_databricks(
        connection=connection,
        catalog=catalog,
        schema=schema,
        tables=[
            dict(
                name='customers',
                columns=['customer_id', 'customer_type', 'segment', 'region'],
                primary_key='customer_id',
            ),
            dict(
                name='order_lines',
                columns=[
                    'order_id',
                    'product_id',
                    'customer_id',
                    'order_date',
                    'quantity',
                    'unit_price_usd',
                    'line_amount_usd',
                ],
                time_column='order_date',
            ),
            dict(
                name='products',
                columns=['product_id', 'product_line', 'list_price_usd'],
                primary_key='product_id',
            ),
        ],
        edges=[
            ('order_lines', 'customer_id', 'customers'),
            ('order_lines', 'product_id', 'products'),
        ],
        verbose=False,
    )
