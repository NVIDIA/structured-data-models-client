# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest

from sdm_connectors import ConnectorError, read_table

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


@pytest.fixture(scope='module')
def snowflake() -> Iterator[Any]:
    connector = pytest.importorskip('snowflake.connector')
    arguments = {
        'account': os.environ['SNOWFLAKE_ACCOUNT'],
        'user': os.environ['SNOWFLAKE_USER'],
        'password': os.environ['SNOWFLAKE_PASSWORD'],
        'database': os.environ['SNOWFLAKE_DATABASE'],
        'schema': os.environ['SNOWFLAKE_SCHEMA'],
        'warehouse': os.environ['SNOWFLAKE_WAREHOUSE'],
        'role': os.environ.get('SNOWFLAKE_ROLE'),
    }
    connection = connector.connect(
        **{key: value for key, value in arguments.items() if value}
    )
    yield connection
    connection.close()


@pytest.fixture(scope='module')
def databricks() -> Iterator[Any]:
    sql = pytest.importorskip('databricks.sql')
    connection = sql.connect(
        server_hostname=os.environ['DATABRICKS_SERVER_HOSTNAME'],
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN'],
    )
    yield connection
    connection.close()


def databricks_schema() -> str:
    return (
        f'{os.environ["DATABRICKS_CATALOG"]}.{os.environ["DATABRICKS_SCHEMA"]}'
    )


# --- Snowflake ------------------------------------------------------------


@requires_snowflake
def test_snowflake_connects_and_reports_where_it_landed(snowflake):
    frame = read_table(
        snowflake,
        query='SELECT CURRENT_DATABASE() AS db, CURRENT_SCHEMA() AS sc',
    )
    # Snowflake folds an unquoted identifier to upper case, so the name it
    # reports back is not character-for-character the one that was sent.
    assert (
        frame.loc[0, 'DB'].upper() == os.environ['SNOWFLAKE_DATABASE'].upper()
    )
    assert frame.loc[0, 'SC'].upper() == os.environ['SNOWFLAKE_SCHEMA'].upper()


@requires_snowflake
def test_snowflake_reads_a_table_by_name(snowflake):
    frame = read_table(snowflake, table='GPUS')
    assert not frame.empty
    assert len(frame.columns) > 1


@requires_snowflake
def test_snowflake_one_connection_serves_several_reads(snowflake):
    first = read_table(snowflake, table='GPUS')
    second = read_table(snowflake, table='DATACENTERS')
    assert not first.empty
    assert not second.empty


@requires_snowflake
def test_snowflake_row_counts_agree_between_a_table_read_and_a_count(
    snowflake,
):
    rows = len(read_table(snowflake, table='DATACENTERS'))
    counted = read_table(
        snowflake, query='SELECT COUNT(*) AS n FROM DATACENTERS'
    ).loc[0, 'N']
    assert rows == int(counted)


@requires_snowflake
def test_snowflake_semantic_view_is_described_and_queried(snowflake):
    """A semantic view is not a table: it is read through ``SEMANTIC_VIEW()``.

    Reaching one at all is the thing worth proving, the connector passes SQL
    through rather than assuming every object can be ``SELECT *``-ed.
    """
    described = read_table(
        snowflake, query='DESCRIBE SEMANTIC VIEW GPU_FLEET_SV'
    )
    assert not described.empty
    dimensions = described[described['object_kind'] == 'DIMENSION']
    metrics = described[described['object_kind'] == 'METRIC']
    assert not dimensions.empty, 'the semantic view declares no dimensions'
    assert not metrics.empty, 'the semantic view declares no metrics'

    frame = read_table(
        snowflake,
        query=(
            'SELECT * FROM SEMANTIC_VIEW('
            'GPU_FLEET_SV DIMENSIONS DATACENTERS.REGION '
            'METRICS GPUS.GPU_COUNT) ORDER BY 1'
        ),
    )
    assert list(frame.columns) == ['REGION', 'GPU_COUNT']
    assert not frame.empty
    assert (frame['GPU_COUNT'] > 0).all()


@requires_snowflake
def test_snowflake_a_missing_table_is_a_connector_error_not_a_driver_error(
    snowflake,
):
    with pytest.raises(ConnectorError) as error:
        read_table(snowflake, table='NO_SUCH_TABLE_XYZ')
    assert error.value.code in {'QUERY_FAILED', 'NOT_FOUND'}


@requires_snowflake
def test_snowflake_an_identifier_needing_quotes_survives_the_round_trip(
    snowflake,
):
    """A name a bare identifier cannot spell must still be readable."""
    frame = read_table(
        snowflake, query='SELECT 1 AS "Total Revenue", 2 AS "order count"'
    )
    assert list(frame.columns) == ['Total Revenue', 'order count']


# --- Databricks -----------------------------------------------------------


@requires_databricks
def test_databricks_connects_and_reports_where_it_landed(databricks):
    frame = read_table(
        databricks,
        query='SELECT current_catalog() AS cat, current_schema() AS sc',
    )
    assert frame.loc[0, 'cat'] == os.environ['DATABRICKS_CATALOG']
    assert frame.loc[0, 'sc'] == os.environ['DATABRICKS_SCHEMA']


@requires_databricks
def test_databricks_reads_a_table_by_name(databricks):
    frame = read_table(databricks, table=f'{databricks_schema()}.customers')
    assert not frame.empty


@requires_databricks
def test_databricks_one_connection_serves_several_reads(databricks):
    schema = databricks_schema()
    assert not read_table(databricks, table=f'{schema}.products').empty
    assert not read_table(databricks, table=f'{schema}.customers').empty


@requires_databricks
def test_databricks_row_counts_agree_between_a_table_read_and_a_count(
    databricks,
):
    schema = databricks_schema()
    rows = len(read_table(databricks, table=f'{schema}.suppliers'))
    counted = read_table(
        databricks, query=f'SELECT COUNT(*) AS n FROM {schema}.suppliers'
    ).loc[0, 'n']
    assert rows == int(counted)


@requires_databricks
def test_databricks_metric_view_needs_measure_and_says_so_plainly(databricks):
    """A metric view refuses a plain read, and the refusal is legible.

    This is the server's rule, not a defect in this package: a measure column
    has no value until it is aggregated. What matters here is that the error
    arrives as a ``ConnectorError`` naming the reason rather than as a raw
    driver exception.
    """
    view = f'{databricks_schema()}.nvidia_erp_mv'

    with pytest.raises(ConnectorError) as error:
        read_table(databricks, table=view)
    assert 'MEASURE' in str(error.value).upper()

    frame = read_table(
        databricks,
        query=f'SELECT MEASURE(`Order Count`) AS order_count FROM {view}',
    )
    assert int(frame.loc[0, 'order_count']) > 0


@requires_databricks
def test_databricks_a_missing_table_is_a_connector_error(databricks):
    with pytest.raises(ConnectorError) as error:
        read_table(databricks, table=f'{databricks_schema()}.no_such_table_xyz')
    assert error.value.code in {'QUERY_FAILED', 'NOT_FOUND'}


@requires_databricks
def test_databricks_a_backticked_identifier_survives_the_round_trip(
    databricks,
):
    frame = read_table(
        databricks, query='SELECT 1 AS `Total Revenue`, 2 AS `order count`'
    )
    assert list(frame.columns) == ['Total Revenue', 'order count']


# --- both -----------------------------------------------------------------


@requires_snowflake
@requires_databricks
def test_both_warehouses_produce_frames_a_schema_can_be_inferred_from(
    snowflake, databricks
):
    """The point of a connector here: what comes out feeds a request."""
    frames = (
        read_table(snowflake, table='GPUS'),
        read_table(databricks, table=f'{databricks_schema()}.customers'),
    )
    for frame in frames:
        assert isinstance(frame, pd.DataFrame)
        assert not frame.empty
        assert all(isinstance(name, str) for name in frame.columns)


# --- what Databricks records of a connection sdm-client opens ---------------

#: How long query history may take to show a statement.
QUERY_HISTORY_SECONDS = 600


@pytest.fixture(scope='module')
def workspace() -> Any:
    r"""The ambient ``WorkspaceClient``, and the warehouse it may read from."""
    if not os.environ.get('DATABRICKS_WAREHOUSE_ID'):
        pytest.skip(
            'set DATABRICKS_WAREHOUSE_ID, and a host and credential the '
            'Databricks SDK reads, to check what query history records'
        )
    sdk = pytest.importorskip('databricks.sdk')
    try:
        return sdk.WorkspaceClient()
    except Exception as error:  # noqa: BLE001 - no credential is a skip
        pytest.skip(
            f'no ambient Databricks credential ({type(error).__name__})'
        )


def test_query_history_records_the_partner_and_the_sampling_tag(workspace):
    """What Databricks attributes the statement by, read back where it does:
    the client application from the User-Agent, and the session's tags."""
    import time

    from databricks.sdk.service.sql import QueryFilter

    from sdm_connectors.databricks import warehouse

    with warehouse(
        os.environ['DATABRICKS_WAREHOUSE_ID'], workspace=workspace
    ) as connection:
        cursor = connection.cursor()
        cursor.execute('SELECT 1 AS one')
        assert cursor.fetchall()[0][0] == 1
        statement_id = cursor.query_id
    assert statement_id

    deadline = time.monotonic() + QUERY_HISTORY_SECONDS
    while True:
        found = workspace.query_history.list(
            filter_by=QueryFilter(statement_ids=[statement_id]),
            include_metrics=False,
        ).res
        if found or time.monotonic() > deadline:
            break
        time.sleep(15)
    assert found, f'{statement_id} is not in query history'
    [query] = found
    assert query.client_application == 'nvidia_sdm-client'
    tags = {tag.key: tag.value for tag in query.query_tags or []}
    assert tags.get('sdm_client_purpose') == 'sampling'
