# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Recording the live scenarios: Databricks, Lakebase and Snowflake.

These need credentials and a network, so each is ``pending-live`` until
someone records it with::

    python scripts/parity/run_corpus.py record --live

Each loads the shop into a scratch namespace it owns (``parity_shop``,
replaced on every recording), builds the metric or semantic view a scenario
reads, records, and is scrubbed before it is stored.

Credentials, by backend:

* Databricks: ``DATABRICKS_SERVER_HOSTNAME``, ``DATABRICKS_HTTP_PATH`` and
  ``DATABRICKS_TOKEN`` for a SQL warehouse, and ``PARITY_DATABRICKS_CATALOG``,
  a catalog the token may create a schema in.
* Lakebase: ``PGHOST``, ``PGUSER``, ``PGPASSWORD`` and ``PGDATABASE`` (an
  OAuth token as the password), read by libpq as usual.
* Snowflake: ``SNOWFLAKE_ACCOUNT``, ``SNOWFLAKE_USER``, either
  ``SNOWFLAKE_TOKEN`` (a programmatic access token) or
  ``SNOWFLAKE_PASSWORD``, ``SNOWFLAKE_WAREHOUSE`` and
  ``PARITY_SNOWFLAKE_DATABASE``, a database the user may create a schema in.

None of this runs in CI or offline; the replay of a live recording, once
stored, does.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

from _parity import data as datagen
from _parity.runner import CannotRecord, load_postgres

REQUIRED = {
    'databricks': (
        'DATABRICKS_SERVER_HOSTNAME',
        'DATABRICKS_HTTP_PATH',
        'DATABRICKS_TOKEN',
        'PARITY_DATABRICKS_CATALOG',
    ),
    'lakebase': ('PGHOST', 'PGUSER', 'PGPASSWORD', 'PGDATABASE'),
    'snowflake': (
        'SNOWFLAKE_ACCOUNT',
        'SNOWFLAKE_USER',
        'SNOWFLAKE_TOKEN|SNOWFLAKE_PASSWORD',
        'SNOWFLAKE_WAREHOUSE',
        'PARITY_SNOWFLAKE_DATABASE',
    ),
}

DRIVERS = {
    'databricks': ('databricks.sql', 'databricks-sql-connector'),
    'lakebase': ('psycopg', 'psycopg[binary]'),
    'snowflake': ('snowflake.connector', 'snowflake-connector-python'),
}

#: The metric view over the shop, in Databricks' metric view YAML. The
#: sources are quoted: a catalog such as ``example-catalog`` is written in
#: backticks, and YAML takes no plain value starting with one.
METRIC_VIEW = """\
version: 0.1
source: "{q}.orders"
joins:
  - name: users
    source: "{q}.users"
    on: source.user_id = users.user_id
  - name: products
    source: "{q}.products"
    on: source.product_id = products.product_id
dimensions:
  - name: order_id
    expr: source.order_id
  - name: user_id
    expr: source.user_id
  - name: product_id
    expr: source.product_id
  - name: quantity
    expr: source.quantity
  - name: ordered_at
    expr: source.ordered_at
  - name: age
    expr: users.age
  - name: country
    expr: users.country
  - name: category
    expr: products.category
  - name: price
    expr: products.price
measures:
  - name: total_quantity
    expr: SUM(source.quantity)
"""

#: The semantic view over the shop, in Snowflake's DDL. Its tables are the
#: shop's written unquoted, so in upper case (:func:`_load`).
SEMANTIC_VIEW = """\
CREATE OR REPLACE SEMANTIC VIEW {q}.PARITY_SHOP_SEMANTICS
  TABLES (
    users AS {q}.USERS PRIMARY KEY (USER_ID),
    products AS {q}.PRODUCTS PRIMARY KEY (PRODUCT_ID),
    orders AS {q}.ORDERS PRIMARY KEY (ORDER_ID)
  )
  RELATIONSHIPS (
    orders_to_users AS orders (USER_ID) REFERENCES users,
    orders_to_products AS orders (PRODUCT_ID) REFERENCES products
  )
  FACTS (
    orders.quantity AS QUANTITY
  )
  DIMENSIONS (
    users.age AS AGE,
    users.country AS COUNTRY,
    products.category AS CATEGORY,
    products.price AS PRICE,
    orders.ordered_at AS ORDERED_AT
  )
  METRICS (
    orders.total_quantity AS SUM(orders.quantity)
  )
"""


def prepare(transcript: dict[str, Any]) -> dict[str, Any]:
    r"""The transcript with its namespace filled in from the environment.

    Raises:
        CannotRecord: A credential or the driver is missing. Nothing has been
            written or connected to.
    """
    backend = transcript['backend']
    missing = [
        names.replace('|', ' or ')
        for names in REQUIRED[backend]
        if not any(os.environ.get(name) for name in names.split('|'))
    ]
    if missing:
        raise CannotRecord(
            f'{transcript["name"]} is pending-live: recording it needs '
            f'{", ".join(missing)} set'
        )
    module, package = DRIVERS[backend]
    try:
        __import__(module)
    except ImportError as error:
        raise CannotRecord(
            f'recording {transcript["name"]} needs {package} installed'
        ) from error
    namespace = {
        key: os.environ[value[1:]] if value.startswith('$') else value
        for key, value in (transcript['data'].get('namespace') or {}).items()
    }
    return {
        **transcript,
        'data': {**transcript['data'], 'namespace': namespace},
    }


def _literal(value: Any) -> str:
    import pandas as pd

    if value is None:
        return 'NULL'
    if isinstance(value, pd.Timestamp):
        return f"TIMESTAMP '{value.isoformat(sep=' ')}'"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value.item() if hasattr(value, 'item') else value)


def _sql_type(dtype: Any, backend: str) -> str:
    text = str(dtype)
    if text.startswith('int'):
        return 'BIGINT'
    if text.startswith('float'):
        return 'DOUBLE' if backend == 'databricks' else 'FLOAT'
    if text.startswith('datetime'):
        return 'TIMESTAMP' if backend == 'databricks' else 'TIMESTAMP_NTZ'
    return 'STRING'


def _name(name: str, quoted: bool) -> str:
    return f'"{name}"' if quoted else name


def _load(
    cursor: Any,
    qualified: str,
    frames: datagen.Frames,
    backend: str,
    quoted: bool = False,
) -> None:
    r"""Write the shop's tables into ``qualified``, their names ``quoted``.

    Snowflake folds an unquoted name to upper case, so ``user_id`` is
    ``USER_ID`` there: quoted, the names keep the lower case every other
    backend has.
    """
    for table, frame in frames.items():
        name = _name(table, quoted)
        columns = ', '.join(
            f'{_name(column, quoted)} {_sql_type(dtype, backend)}'
            for column, dtype in frame.dtypes.items()
        )
        cursor.execute(
            f'CREATE OR REPLACE TABLE {qualified}.{name} ({columns})'
        )
        rows = list(frame.itertuples(index=False, name=None))
        for start in range(0, len(rows), 500):
            values = ', '.join(
                '(' + ', '.join(_literal(v) for v in row) + ')'
                for row in rows[start : start + 500]
            )
            cursor.execute(f'INSERT INTO {qualified}.{name} VALUES {values}')


@contextlib.contextmanager
def _databricks(
    transcript: dict[str, Any], frames: datagen.Frames
) -> Iterator[Any]:
    from databricks import sql

    namespace = transcript['data']['namespace']
    qualified = f'`{namespace["catalog"]}`.{namespace["schema"]}'
    connection = sql.connect(
        server_hostname=os.environ['DATABRICKS_SERVER_HOSTNAME'],
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN'],
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS {qualified}')
            _load(cursor, qualified, frames, 'databricks')
            if transcript['data'].get('view') == 'metric-view':
                cursor.execute(
                    f'CREATE OR REPLACE VIEW {qualified}.parity_shop_metrics '
                    f'WITH METRICS LANGUAGE YAML AS $$\n'
                    f'{METRIC_VIEW.format(q=qualified)}$$'
                )
        yield connection
    finally:
        connection.close()


@contextlib.contextmanager
def _lakebase(
    transcript: dict[str, Any], frames: datagen.Frames
) -> Iterator[Any]:
    import psycopg

    schema = transcript['data']['namespace']['schema']
    connection = psycopg.connect(sslmode='require', autocommit=True)
    try:
        load_postgres(connection, schema, frames)
        yield connection
    finally:
        connection.close()


@contextlib.contextmanager
def _snowflake(
    transcript: dict[str, Any], frames: datagen.Frames
) -> Iterator[Any]:
    import snowflake.connector

    namespace = transcript['data']['namespace']
    qualified = f'{namespace["database"]}.{namespace["schema"]}'
    token = os.environ.get('SNOWFLAKE_TOKEN')
    credential = (
        {'authenticator': 'PROGRAMMATIC_ACCESS_TOKEN', 'token': token}
        if token
        else {'password': os.environ['SNOWFLAKE_PASSWORD']}
    )
    connection = snowflake.connector.connect(
        account=os.environ['SNOWFLAKE_ACCOUNT'],
        user=os.environ['SNOWFLAKE_USER'],
        warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
        **credential,
    )
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'CREATE SCHEMA IF NOT EXISTS {qualified}')
            # A semantic view reads the shop in upper case, as Snowflake
            # spells unquoted names; the engine reads a semantic view's
            # quoted names as written, quotes included, so the lower-case
            # tables can't back one. Tables are read in the lower case every
            # other backend has. The two sets live side by side.
            view = transcript['data'].get('view') == 'semantic-view'
            _load(cursor, qualified, frames, 'snowflake', quoted=not view)
            if view:
                cursor.execute(SEMANTIC_VIEW.format(q=qualified))
        yield connection
    finally:
        connection.close()


@contextlib.contextmanager
def connection(
    transcript: dict[str, Any], frames: datagen.Frames
) -> Iterator[Any]:
    r"""A live connection holding the shop, for recording ``transcript``."""
    opener = {
        'databricks': _databricks,
        'lakebase': _lakebase,
        'snowflake': _snowflake,
    }[transcript['backend']]
    with opener(transcript, frames) as opened:
        yield opened
