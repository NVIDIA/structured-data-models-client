# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated warehouse fixtures owned by a single CI invocation."""

import contextlib
import datetime
import os
import uuid
from urllib.parse import urlparse

from hosted_fixtures import config_for, fixtures
from python_runner import tables_of

from contract import digest


def quoted(name, databricks):
    quote = '`' if databricks else '"'
    return quote + name.replace(quote, quote * 2) + quote


def sql_type(dtype, databricks):
    import pyarrow as pa

    if pa.types.is_boolean(dtype):
        return 'BOOLEAN'
    if pa.types.is_integer(dtype):
        return 'BIGINT'
    if pa.types.is_floating(dtype):
        return 'DOUBLE'
    if pa.types.is_timestamp(dtype):
        return 'TIMESTAMP_NTZ'
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return 'STRING' if databricks else 'VARCHAR'
    raise ValueError(
        'Warehouse fixture type needs explicit mapping: ' + str(dtype)
    )


def binding_value(value):
    if isinstance(value, datetime.datetime):
        if getattr(value, 'nanosecond', 0):
            raise ValueError(
                'Warehouse fixture timestamp requires submicrosecond binding'
            )
        return datetime.datetime(
            value.year,
            value.month,
            value.day,
            value.hour,
            value.minute,
            value.second,
            value.microsecond,
            tzinfo=value.tzinfo,
        )
    return value


def databricks_connection():
    import requests
    from databricks import sql

    host = os.environ['DATABRICKS_HOST'].rstrip('/')
    token = os.environ['DATABRICKS_TOKEN']
    warehouse = os.environ.get('PARITY_DATABRICKS_WAREHOUSE_ID')
    if not warehouse:
        response = requests.get(
            host + '/api/2.0/sql/warehouses',
            headers={'Authorization': 'Bearer ' + token},
            timeout=30,
        )
        response.raise_for_status()
        warehouses = response.json()['warehouses']
        if len(warehouses) != 1:
            raise ValueError(
                'Set PARITY_DATABRICKS_WAREHOUSE_ID to select the test warehouse'
            )
        warehouse = warehouses[0]['id']
    options = {
        'server_hostname': urlparse(host).hostname,
        'http_path': '/sql/1.0/warehouses/' + warehouse,
        'access_token': token,
    }
    return sql.connect(**options), options


def snowflake_connection():
    import snowflake.connector

    url = os.environ['SNOWFLAKE_URL']
    host = urlparse(url if '://' in url else 'https://' + url).hostname
    options = {
        'account': host.removesuffix('.snowflakecomputing.com'),
        'host': host,
        'user': os.environ['SNOWFLAKE_USERNAME'],
        'password': os.environ['SNOWFLAKE_ACCESS_TOKEN'],
        'login_timeout': 30,
    }
    for name in ('warehouse', 'database', 'role'):
        value = os.environ.get('PARITY_SNOWFLAKE_' + name.upper())
        if value:
            options[name] = value
    return snowflake.connector.connect(**options), options


def semantic_view(cursor, fixture, namespace, databricks):
    view = 'fixture_model'

    def q(name):
        return quoted(name, databricks)

    target = namespace + '.' + q(view)
    if databricks:
        import yaml

        definition = {
            'version': '1.1',
            'source': namespace + '.' + q('events'),
            'joins': [
                {
                    'name': 'users',
                    'source': namespace + '.' + q('users'),
                    'on': 'source.id = users.id',
                },
                {
                    'name': 'items',
                    'source': namespace + '.' + q('items'),
                    'on': 'source.item = items.item',
                },
            ],
            'dimensions': [{'name': 'event', 'expr': 'source.event'}],
            'measures': [{'name': 'event_count', 'expr': 'COUNT(1)'}],
        }
        body = yaml.safe_dump(definition)
        cursor.execute(
            f'CREATE VIEW {target} WITH METRICS LANGUAGE YAML AS $$\n{body}$$'
        )
    else:
        tables = ', '.join(
            f'{q(name)} AS {namespace}.{q(name)} PRIMARY KEY ({q(fixture["primary_keys"][name])})'
            for name in fixture['tables']
        )
        relations = []
        for i, (left, right) in enumerate(fixture['links']):
            lt, lc = left.split('.')
            rt, rc = right.split('.')
            relations.append(
                f'{q("link_" + str(i))} AS {q(lt)} ({q(lc)}) REFERENCES {q(rt)} ({q(rc)})'
            )
        cursor.execute(
            f'CREATE SEMANTIC VIEW {target} TABLES ({tables}) '
            f'RELATIONSHIPS ({", ".join(relations)}) '
            f'DIMENSIONS ({q("users")}.{q("user_id")} AS {q("users")}.{q("ID")})'
        )
    return view


@contextlib.contextmanager
def warehouse(connector):
    import pyarrow as pa

    databricks = connector.startswith('databricks')
    available = fixtures(connector)
    if len(available) != 1:
        raise ValueError(
            'Warehouse provisioning requires an explicit plan for multiple fixture namespaces'
        )
    fixture_name, fixture = next(iter(available.items()))
    factory = databricks_connection if databricks else snowflake_connection
    connection, connect = factory()
    schema = 'sdm_ci_' + uuid.uuid4().hex
    if not databricks:
        schema = schema.upper()

    def q(name):
        return quoted(name, databricks)

    try:
        with connection.cursor() as cursor:
            cursor.execute(
                'SELECT current_catalog()'
                if databricks
                else 'SELECT current_database()'
            )
            database = (
                os.environ.get('PARITY_DATABRICKS_CATALOG')
                if databricks
                else connect.get('database')
            )
            database = database or cursor.fetchone()[0]
            if not database:
                raise ValueError(
                    'Configure a test catalog/database before fixture provisioning'
                )
            namespace = q(database) + '.' + q(schema)
            cursor.execute('CREATE SCHEMA ' + namespace)
            try:
                for table, frame in tables_of(fixture).items():
                    data = pa.Table.from_pandas(frame, preserve_index=False)
                    target = namespace + '.' + q(table)
                    columns = ', '.join(
                        f'{q(field.name)} {sql_type(field.type, databricks)}'
                        for field in data.schema
                    )
                    cursor.execute(f'CREATE TABLE {target} ({columns})')
                    placeholders = ','.join(
                        '?' if databricks else '%s' for _ in data.column_names
                    )
                    rows = data.to_pylist()
                    values = ','.join(f'({placeholders})' for _ in rows)
                    cursor.execute(
                        f'INSERT INTO {target} VALUES {values}',
                        [
                            binding_value(value)
                            for row in rows
                            for value in row.values()
                        ],
                    )
                view = (
                    semantic_view(cursor, fixture, namespace, databricks)
                    if connector in ('databricks-metric', 'snowflake-semantic')
                    else None
                )
                source = {
                    'catalog' if databricks else 'database': database,
                    'schema': schema,
                }
                connect.update(source)
                if databricks:
                    java = {
                        'url': f'jdbc:databricks://{connect["server_hostname"]}:443;httpPath={connect["http_path"]};AuthMech=3;ssl=1',
                        'properties': {
                            'UID': 'token',
                            'PWD': connect['access_token'],
                            'ConnCatalog': database,
                            'ConnSchema': schema,
                        },
                    }
                else:
                    java = {
                        'url': 'jdbc:snowflake://' + connect['host'],
                        'properties': {
                            k: connect[k]
                            for k in ('user', 'password', 'warehouse', 'role')
                            if k in connect
                        }
                        | {'db': database, 'schema': schema},
                    }
                settings = {
                    'snapshot': digest(fixture),
                    'source': source,
                    'connect': connect,
                    'tables': {name: name for name in fixture['tables']},
                    'java': java,
                }
                if view:
                    settings['view'] = view
                if connector == 'databricks-spark':
                    settings['connect'] = {'serverless': True}
                    settings['java'] = {}
                yield config_for(connector, {fixture_name: settings})
            finally:
                # This exact random namespace was created by this invocation.
                cursor.execute('DROP SCHEMA ' + namespace + ' CASCADE')
    finally:
        connection.close()
