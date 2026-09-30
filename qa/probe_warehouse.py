# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snowflake and Databricks: real warehouses, real types, into a prediction.

The interesting question is not "does a SELECT work". It is whether what a
warehouse hands back still means the same thing by the time it reaches a model.
"""

from __future__ import annotations

import os
from decimal import Decimal

from harness import TABULAR, fail, note, probe, save
from sdm_client import KumoTabular, Request, SDMClient
from sdm_client.wire.dtypes import infer_tfm_dtype, serialize_column
from sdm_connectors import ConnectorError, connect, read_table

AREA = 'warehouse'
SAFE = 9_007_199_254_740_991
SUFFIX = os.environ.get('QA_SUFFIX', 'qa1')


def snowflake():
    return connect(
        'snowflake',
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        user=os.environ['SNOWFLAKE_USER'],
        password=os.environ['SNOWFLAKE_PASSWORD'],
        database=os.environ['SNOWFLAKE_DATABASE'],
        schema=os.environ['SNOWFLAKE_SCHEMA'],
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
        role=os.environ.get('SNOWFLAKE_ROLE'),
    )


def databricks():
    return connect(
        'databricks',
        server_hostname=os.environ['DATABRICKS_SERVER_HOSTNAME'],
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN'],
    )


def score(client, frame, target, task='classification'):
    context = frame.head(len(frame) - 3)
    predict = frame.tail(3).drop(columns=[target]).reset_index(drop=True)
    return client.predict(
        Request(context=context, predict=predict, target=target, task=task),
        model=KumoTabular(),
    ).to_pandas()


def run_snowflake(client):
    connection = snowflake()
    table = f'SDM_QA_{SUFFIX}'
    created = []

    with probe(
        AREA,
        'snowflake connects and lands where it was told',
        'the session names the schema',
    ) as e:
        frame = read_table(
            connection,
            query='select current_database() DB, current_schema() SC',
        )
        if (
            frame.loc[0, 'DB'].upper()
            != os.environ['SNOWFLAKE_DATABASE'].upper()
        ):
            fail(e, f'landed in {frame.loc[0, "DB"]}')
        else:
            note(e, f'{frame.loc[0, "DB"]}.{frame.loc[0, "SC"]}')

    with probe(
        AREA,
        'snowflake, a real business table scores',
        'warehouse rows predict',
    ) as e:
        tables = read_table(
            connection,
            query=(
                'select table_name, row_count from information_schema.tables '
                "where table_schema = current_schema() and table_type = 'BASE TABLE' "
                'and row_count > 200 order by row_count limit 1'
            ),
        )
        if tables.empty:
            note(e, 'no table with enough rows in this schema')
        else:
            name = tables['TABLE_NAME'].iloc[0]
            frame = read_table(
                connection, query=f'select * from {name} limit 400'
            )
            note(
                e,
                f'{name} {frame.shape} dtypes={sorted(set(frame.dtypes.astype(str)))}',
            )
            usable = frame.select_dtypes(exclude=['datetime64[ns]']).dropna(
                axis=1, how='all'
            )
            targets = [
                c for c in usable.columns if 1 < usable[c].nunique() <= 8
            ]
            if not targets:
                note(e, 'no low-cardinality target in this table')
            else:
                result = score(client, usable.dropna(), targets[0])
                if len(result) != 3:
                    fail(e, f'{len(result)} rows scored')

    with probe(
        AREA,
        'snowflake NUMBER(38,0) past the JS safe range',
        'exact on the wire',
    ) as e:
        connection.cursor().execute(
            f'create or replace table {table} '
            f'(ID NUMBER(38,0), AMOUNT NUMBER(38,4), LABEL VARCHAR)'
        )
        created.append(table)
        connection.cursor().execute(
            f'insert into {table} values '
            f'({SAFE + 2}, 12.3456, %s), ({SAFE + 3}, 1.0, %s)',
            ('a', 'b'),
        )
        frame = read_table(
            connection, query=f'select * from {table} order by ID'
        )
        cells = serialize_column(frame['ID'], infer_tfm_dtype(frame['ID']))
        if str(cells[0]) != str(SAFE + 2):
            fail(
                e,
                f'{SAFE + 2} became {cells[0]!r} '
                f'(pandas dtype {frame["ID"].dtype}, python '
                f'{type(frame["ID"].iloc[0]).__name__})',
            )
        else:
            note(e, f'exact: {cells}')

    with probe(
        AREA,
        'snowflake DECIMAL keeps its scale',
        'a money column is not truncated',
    ) as e:
        frame = read_table(
            connection, query=f'select AMOUNT from {table} order by AMOUNT desc'
        )
        value = frame['AMOUNT'].iloc[0]
        if Decimal(str(value)) != Decimal('12.3456'):
            fail(e, f'12.3456 became {value!r}')

    with probe(
        AREA,
        'snowflake folds an unquoted identifier to upper case',
        'a lower-case name still resolves',
    ) as e:
        frame = read_table(connection, table=table.lower())
        if len(frame) != 2:
            fail(e, f'{len(frame)} rows')

    with probe(
        AREA,
        'snowflake, a quoted mixed-case identifier',
        'readable through an explicit query',
    ) as e:
        odd = f'"Sdm Qa {SUFFIX}"'
        connection.cursor().execute(
            f'create or replace table {odd} ("Col One" number, "select" number)'
        )
        created.append(odd)
        connection.cursor().execute(f'insert into {odd} values (1, 2)')
        frame = read_table(connection, query=f'select * from {odd}')
        if list(frame.columns) != ['Col One', 'select']:
            fail(e, f'columns came back as {list(frame.columns)}')

    with probe(
        AREA,
        'snowflake, a name that needs quoting via table=',
        'refused, not interpolated',
    ) as e:
        try:
            read_table(connection, table=f'Sdm Qa {SUFFIX}')
            fail(e, 'a spaced identifier was interpolated')
        except ConnectorError as exc:
            note(e, str(exc)[:120])

    with probe(
        AREA,
        'snowflake, an injection attempt through table=',
        'the sentinel survives',
    ) as e:
        try:
            read_table(connection, table=f'{table}; drop table {table}')
            fail(e, 'a second statement was allowed through a table name')
        except ConnectorError as exc:
            note(e, str(exc)[:100])
        # '_' is a LIKE wildcard, so this has to be an equality test against
        # the folded name rather than a pattern.
        still = read_table(
            connection,
            query=(
                f'select count(*) N from information_schema.tables '
                f"where table_name = '{table.upper()}'"
            ),
        )
        if int(still['N'].iloc[0]) != 1:
            fail(e, 'the sentinel table was dropped')

    with probe(
        AREA,
        'snowflake, a semantic view is described',
        'dimensions and metrics are declared',
    ) as e:
        described = read_table(
            connection, query='describe semantic view GPU_FLEET_SV'
        )
        kinds = {k for k in described['object_kind'] if isinstance(k, str)}
        if not {'DIMENSION', 'METRIC'} <= kinds:
            fail(e, f'kinds: {sorted(kinds)}')
        else:
            note(e, f'{sorted(kinds)}')

    with probe(
        AREA,
        'snowflake, a semantic view is queried',
        'SEMANTIC_VIEW() returns its columns',
    ) as e:
        frame = read_table(
            connection,
            query=(
                'select * from semantic_view(GPU_FLEET_SV '
                'dimensions DATACENTERS.REGION metrics GPUS.GPU_COUNT) order by 1'
            ),
        )
        if list(frame.columns) != ['REGION', 'GPU_COUNT']:
            fail(e, f'columns {list(frame.columns)}')
        else:
            note(e, f'{frame.shape}')

    with probe(
        AREA,
        'snowflake, a semantic view feeds a prediction',
        'governed rows score',
    ) as e:
        frame = read_table(
            connection,
            query=(
                'select * from semantic_view(GPU_FLEET_SV '
                'dimensions DATACENTERS.REGION, DATACENTERS.COUNTRY, DATACENTERS.COOLING_TYPE '
                'metrics GPUS.GPU_COUNT)'
            ),
        )
        if len(frame) < 8:
            note(e, f'only {len(frame)} rows in the view; not scoring')
        else:
            result = score(client, frame.dropna(), 'COOLING_TYPE')
            if len(result) != 3:
                fail(e, f'{len(result)} rows scored')
            else:
                note(e, f'{frame.shape} -> {list(result["prediction"])}')

    with probe(
        AREA,
        'snowflake, a semantic view metric is exact',
        'a metric is not rounded to float',
    ) as e:
        frame = read_table(
            connection,
            query=(
                'select * from semantic_view(GPU_FLEET_SV '
                'dimensions DATACENTERS.REGION metrics GPUS.GPU_COUNT) order by 1'
            ),
        )
        kind = type(frame['GPU_COUNT'].iloc[0]).__name__
        note(e, f'GPU_COUNT dtype={frame["GPU_COUNT"].dtype} python={kind}')
        if kind == 'Decimal':
            cells = serialize_column(
                frame['GPU_COUNT'], infer_tfm_dtype(frame['GPU_COUNT'])
            )
            note(e, f'{kind} -> wire {cells[:3]}')

    for name in created:
        connection.cursor().execute(f'drop table if exists {name}')
    print(f'snowflake: dropped {created}')
    connection.close()


def run_databricks(client):
    connection = databricks()
    catalog = os.environ['DATABRICKS_CATALOG']
    schema = os.environ['DATABRICKS_SCHEMA']
    table = f'{catalog}.{schema}.sdm_qa_{SUFFIX}'
    created = []

    with probe(
        AREA, 'databricks connects', 'the session names the catalog'
    ) as e:
        frame = read_table(
            connection, query='select current_catalog() c, current_schema() s'
        )
        note(e, f'{frame.iloc[0, 0]}.{frame.iloc[0, 1]}')

    with probe(AREA, 'databricks samples.tpch', 'a shared dataset reads') as e:
        frame = read_table(
            connection,
            query=(
                'select c_custkey, c_nationkey, c_acctbal, c_mktsegment '
                'from samples.tpch.customer limit 400'
            ),
        )
        if len(frame) != 400:
            fail(e, f'{len(frame)} rows')
        else:
            note(e, f'dtypes={dict(frame.dtypes.astype(str))}')

    with probe(
        AREA,
        'databricks tpch scores through the model',
        'warehouse data predicts',
    ) as e:
        frame = read_table(
            connection,
            query=(
                'select c_nationkey, c_acctbal, c_mktsegment '
                'from samples.tpch.customer limit 400'
            ),
        )
        result = score(client, frame, 'c_mktsegment')
        if len(result) != 3:
            fail(e, f'{len(result)} rows')

    with probe(
        AREA,
        'databricks DECIMAL(38,0) past the JS safe range',
        'exact on the wire',
    ) as e:
        connection.cursor().execute(f'drop table if exists {table}')
        connection.cursor().execute(
            f'create table {table} (id decimal(38,0), amount decimal(38,4), label string)'
        )
        created.append(table)
        connection.cursor().execute(
            f'insert into {table} values ({SAFE + 2}, 12.3456, "a"), ({SAFE + 3}, 1.0, "b")'
        )
        frame = read_table(
            connection, query=f'select * from {table} order by id'
        )
        cells = serialize_column(frame['id'], infer_tfm_dtype(frame['id']))
        if str(cells[0]) != str(SAFE + 2):
            fail(
                e,
                f'{SAFE + 2} became {cells[0]!r} '
                f'(pandas dtype {frame["id"].dtype}, python '
                f'{type(frame["id"].iloc[0]).__name__})',
            )
        else:
            note(e, f'exact: {cells}')

    with probe(
        AREA, 'databricks BIGINT past the JS safe range', 'exact on the wire'
    ) as e:
        big = f'{table}_bigint'
        connection.cursor().execute(f'drop table if exists {big}')
        connection.cursor().execute(f'create table {big} (id bigint, v double)')
        created.append(big)
        connection.cursor().execute(
            f'insert into {big} values ({SAFE + 2}, 1.0)'
        )
        frame = read_table(connection, query=f'select * from {big}')
        cells = serialize_column(frame['id'], infer_tfm_dtype(frame['id']))
        if str(cells[0]) != str(SAFE + 2):
            fail(e, f'{SAFE + 2} became {cells[0]!r}')

    with probe(
        AREA,
        'databricks keeps identifier case',
        'unlike snowflake, names are not folded',
    ) as e:
        frame = read_table(
            connection, query=f'select id as MixedCase from {table} limit 1'
        )
        note(e, f'columns={list(frame.columns)}')

    with probe(
        AREA,
        'databricks, an injection attempt through table=',
        'the sentinel survives',
    ) as e:
        try:
            read_table(connection, table=f'{table}; drop table {table}')
        except ConnectorError as exc:
            note(e, str(exc)[:100])
        frame = read_table(connection, query=f'select count(*) n from {table}')
        if int(frame['n'].iloc[0]) != 2:
            fail(e, 'the sentinel table changed')

    with probe(
        AREA,
        'databricks, a dotted three-part name via table=',
        'catalog.schema.table resolves',
    ) as e:
        frame = read_table(connection, table=table)
        if len(frame) != 2:
            fail(e, f'{len(frame)} rows')

    with probe(
        AREA, 'databricks, a metric view is listed', 'isMetric marks it'
    ) as e:
        views = read_table(
            connection, query=f'show views in {catalog}.{schema}'
        )
        metric = (
            views[views['isMetric']] if 'isMetric' in views else views.head(0)
        )
        if metric.empty:
            note(e, f'no metric view among {len(views)} views')
        else:
            note(e, f'metric views: {list(metric["viewName"])}')

    with probe(
        AREA,
        'databricks, a metric view without MEASURE()',
        'refused by name, not a crash',
    ) as e:
        view = f'{catalog}.{schema}.nvidia_erp_mv'
        try:
            read_table(connection, query=f'select * from {view} limit 5')
            note(e, 'select * was accepted')
        except ConnectorError as exc:
            if 'MEASURE' not in str(exc):
                fail(e, f'refused for an unexpected reason: {str(exc)[:140]}')
            else:
                note(e, 'METRIC_VIEW_MISSING_MEASURE_FUNCTION, surfaced intact')
        except Exception as exc:  # noqa: BLE001
            fail(e, f'{type(exc).__name__} escaped the connector error surface')

    with probe(
        AREA,
        'databricks, a metric view feeds a prediction',
        'governed measures score',
    ) as e:
        view = f'{catalog}.{schema}.nvidia_erp_mv'
        frame = read_table(
            connection,
            query=(
                f'select `Product Family`, `Customer Region`, `Sales Channel`, '
                f'MEASURE(`Total Revenue`) as revenue, '
                f'MEASURE(`Order Count`) as orders '
                f'from {view} group by 1, 2, 3'
            ),
        )
        if len(frame) < 10:
            note(e, f'only {len(frame)} rows; not scoring')
        else:
            result = score(client, frame, 'Sales Channel')
            if len(result) != 3:
                fail(e, f'{len(result)} rows scored')
            else:
                note(e, f'{frame.shape} -> {list(result["prediction"])}')

    for name in created:
        connection.cursor().execute(f'drop table if exists {name}')
    print(f'databricks: dropped {created}')
    connection.close()


if __name__ == '__main__':
    with SDMClient(url=TABULAR, timeout=600.0) as client:
        if all(
            os.environ.get(k) for k in ('SNOWFLAKE_ACCOUNT', 'SNOWFLAKE_USER')
        ):
            run_snowflake(client)
        if all(
            os.environ.get(k)
            for k in ('DATABRICKS_SERVER_HOSTNAME', 'DATABRICKS_TOKEN')
        ):
            run_databricks(client)
    save('warehouse.json')
