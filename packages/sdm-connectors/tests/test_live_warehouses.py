# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-OpenMDW-1.1

"""End-to-end against real Snowflake and Databricks warehouses.

Excluded by default; each backend skips unless its credentials are in the
environment. Run with ``-m live_warehouse``.

These cover what the offline suite cannot: that the SQL this package generates
is accepted by the real engines, that identifiers needing quotes survive the
round trip, and that the two governed object types, a Snowflake **semantic
view** and a Databricks **metric view**, are reachable. Those two are worth
testing precisely because neither behaves like a table.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from sdm_connectors import ConnectorError, connect, read, read_table

pytestmark = pytest.mark.live_warehouse

SNOWFLAKE_KEYS = (
    'SNOWFLAKE_ACCOUNT',
    'SNOWFLAKE_USER',
    'SNOWFLAKE_PASSWORD',
    'SNOWFLAKE_DATABASE',
    'SNOWFLAKE_SCHEMA',
    'SNOWFLAKE_WAREHOUSE',
)
DATABRICKS_KEYS = (
    'DATABRICKS_SERVER_HOSTNAME',
    'DATABRICKS_HTTP_PATH',
    'DATABRICKS_TOKEN',
    'DATABRICKS_CATALOG',
    'DATABRICKS_SCHEMA',
)

requires_snowflake = pytest.mark.skipif(
    not all(os.environ.get(key) for key in SNOWFLAKE_KEYS),
    reason='set SNOWFLAKE_* to run against a real Snowflake account',
)
requires_databricks = pytest.mark.skipif(
    not all(os.environ.get(key) for key in DATABRICKS_KEYS),
    reason='set DATABRICKS_* to run against a real Databricks workspace',
)


def snowflake_args() -> dict:
    return {
        'account': os.environ['SNOWFLAKE_ACCOUNT'],
        'user': os.environ['SNOWFLAKE_USER'],
        'password': os.environ['SNOWFLAKE_PASSWORD'],
        'database': os.environ['SNOWFLAKE_DATABASE'],
        'schema': os.environ['SNOWFLAKE_SCHEMA'],
        'warehouse': os.environ['SNOWFLAKE_WAREHOUSE'],
        'role': os.environ.get('SNOWFLAKE_ROLE'),
    }


def databricks_args() -> dict:
    return {
        'server_hostname': os.environ['DATABRICKS_SERVER_HOSTNAME'],
        'http_path': os.environ['DATABRICKS_HTTP_PATH'],
        'access_token': os.environ['DATABRICKS_TOKEN'],
    }


def databricks_schema() -> str:
    return (
        f'{os.environ["DATABRICKS_CATALOG"]}.{os.environ["DATABRICKS_SCHEMA"]}'
    )


# --- Snowflake ------------------------------------------------------------


@requires_snowflake
def test_snowflake_connects_and_reports_where_it_landed():
    frame = read(
        'snowflake',
        query='SELECT CURRENT_DATABASE() AS db, CURRENT_SCHEMA() AS sc',
        **snowflake_args(),
    )
    # Snowflake folds an unquoted identifier to upper case, so the name it
    # reports back is not character-for-character the one that was sent.
    assert (
        frame.loc[0, 'DB'].upper() == os.environ['SNOWFLAKE_DATABASE'].upper()
    )
    assert frame.loc[0, 'SC'].upper() == os.environ['SNOWFLAKE_SCHEMA'].upper()


@requires_snowflake
def test_snowflake_reads_a_table_by_name():
    frame = read('snowflake', table='GPUS', **snowflake_args())
    assert not frame.empty
    assert len(frame.columns) > 1


@requires_snowflake
def test_snowflake_reads_a_table_through_an_open_connection():
    """``connect`` then ``read_table`` reuses one session for several reads."""
    connection = connect('snowflake', **snowflake_args())
    try:
        first = read_table(connection, table='GPUS')
        second = read_table(connection, table='DATACENTERS')
        assert not first.empty
        assert not second.empty
    finally:
        connection.close()


@requires_snowflake
def test_snowflake_row_counts_agree_between_a_table_read_and_a_count():
    args = snowflake_args()
    rows = len(read('snowflake', table='DATACENTERS', **args))
    counted = read(
        'snowflake', query='SELECT COUNT(*) AS n FROM DATACENTERS', **args
    ).loc[0, 'N']
    assert rows == int(counted)


@requires_snowflake
def test_snowflake_semantic_view_is_described_and_queried():
    """A semantic view is not a table: it is read through ``SEMANTIC_VIEW()``.

    Reaching one at all is the thing worth proving, the connector passes SQL
    through rather than assuming every object can be ``SELECT *``-ed.
    """
    args = snowflake_args()
    described = read(
        'snowflake', query='DESCRIBE SEMANTIC VIEW GPU_FLEET_SV', **args
    )
    assert not described.empty
    dimensions = described[described['object_kind'] == 'DIMENSION']
    metrics = described[described['object_kind'] == 'METRIC']
    assert not dimensions.empty, 'the semantic view declares no dimensions'
    assert not metrics.empty, 'the semantic view declares no metrics'

    frame = read(
        'snowflake',
        query=(
            'SELECT * FROM SEMANTIC_VIEW('
            'GPU_FLEET_SV DIMENSIONS DATACENTERS.REGION '
            'METRICS GPUS.GPU_COUNT) ORDER BY 1'
        ),
        **args,
    )
    assert list(frame.columns) == ['REGION', 'GPU_COUNT']
    assert not frame.empty
    assert (frame['GPU_COUNT'] > 0).all()


@requires_snowflake
def test_snowflake_a_missing_table_is_a_connector_error_not_a_driver_error():
    with pytest.raises(ConnectorError) as error:
        read('snowflake', table='NO_SUCH_TABLE_XYZ', **snowflake_args())
    assert error.value.code in {'QUERY_FAILED', 'NOT_FOUND'}


@requires_snowflake
def test_snowflake_an_identifier_needing_quotes_survives_the_round_trip():
    """A name a bare identifier cannot spell must still be readable."""
    frame = read(
        'snowflake',
        query='SELECT 1 AS "Total Revenue", 2 AS "order count"',
        **snowflake_args(),
    )
    assert list(frame.columns) == ['Total Revenue', 'order count']


# --- Databricks -----------------------------------------------------------


@requires_databricks
def test_databricks_connects_and_reports_where_it_landed():
    frame = read(
        'databricks',
        query='SELECT current_catalog() AS cat, current_schema() AS sc',
        **databricks_args(),
    )
    assert frame.loc[0, 'cat'] == os.environ['DATABRICKS_CATALOG']
    assert frame.loc[0, 'sc'] == os.environ['DATABRICKS_SCHEMA']


@requires_databricks
def test_databricks_reads_a_table_by_name():
    frame = read(
        'databricks',
        table=f'{databricks_schema()}.customers',
        **databricks_args(),
    )
    assert not frame.empty


@requires_databricks
def test_databricks_reads_a_table_through_an_open_connection():
    connection = connect('databricks', **databricks_args())
    try:
        frame = read_table(connection, table=f'{databricks_schema()}.products')
        assert not frame.empty
    finally:
        connection.close()


@requires_databricks
def test_databricks_row_counts_agree_between_a_table_read_and_a_count():
    args = databricks_args()
    schema = databricks_schema()
    rows = len(read('databricks', table=f'{schema}.suppliers', **args))
    counted = read(
        'databricks',
        query=f'SELECT COUNT(*) AS n FROM {schema}.suppliers',
        **args,
    ).loc[0, 'n']
    assert rows == int(counted)


@requires_databricks
def test_databricks_metric_view_needs_measure_and_says_so_plainly():
    """A metric view refuses a plain read, and the refusal is legible.

    This is the server's rule, not a defect in this package: a measure column
    has no value until it is aggregated. What matters here is that the error
    arrives as a ``ConnectorError`` naming the reason rather than as a raw
    driver exception.
    """
    args = databricks_args()
    view = f'{databricks_schema()}.nvidia_erp_mv'

    with pytest.raises(ConnectorError) as error:
        read('databricks', table=view, **args)
    assert 'MEASURE' in str(error.value).upper()

    frame = read(
        'databricks',
        query=f'SELECT MEASURE(`Order Count`) AS order_count FROM {view}',
        **args,
    )
    assert int(frame.loc[0, 'order_count']) > 0


@requires_databricks
def test_databricks_a_missing_table_is_a_connector_error():
    with pytest.raises(ConnectorError) as error:
        read(
            'databricks',
            table=f'{databricks_schema()}.no_such_table_xyz',
            **databricks_args(),
        )
    assert error.value.code in {'QUERY_FAILED', 'NOT_FOUND'}


@requires_databricks
def test_databricks_a_backticked_identifier_survives_the_round_trip():
    frame = read(
        'databricks',
        query='SELECT 1 AS `Total Revenue`, 2 AS `order count`',
        **databricks_args(),
    )
    assert list(frame.columns) == ['Total Revenue', 'order count']


# --- both -----------------------------------------------------------------


@requires_snowflake
@requires_databricks
def test_both_warehouses_produce_frames_a_schema_can_be_inferred_from():
    """The point of a connector here: what comes out feeds a request."""
    snowflake = read('snowflake', table='GPUS', **snowflake_args())
    databricks = read(
        'databricks',
        table=f'{databricks_schema()}.customers',
        **databricks_args(),
    )
    for frame in (snowflake, databricks):
        assert isinstance(frame, pd.DataFrame)
        assert not frame.empty
        assert all(isinstance(name, str) for name in frame.columns)
