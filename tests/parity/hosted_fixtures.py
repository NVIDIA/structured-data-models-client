# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disposable fixture provisioning for hosted and external-maintainer tests."""

import contextlib
import io
import json
import os
import tempfile
import uuid
from pathlib import Path

from python_runner import tables_of

from contract import HERE, digest, effective_case, manifest


def fixtures(connector):
    names = dict.fromkeys(
        effective_case(case, connector)['fixture']
        for case in manifest()['scenarios']
        if connector in case['connectors']
    )
    return {name: json.loads((HERE / name).read_text()) for name in names}


def config_for(connector, settings):
    """Use per-scenario source settings without modifying declared support."""
    return {
        'sources': {
            connector: settings.get(
                'fixtures/shop.json', next(iter(settings.values()), {})
            )
        },
        'fixture_sources': {
            name: {connector: value} for name, value in settings.items()
        },
    }


@contextlib.contextmanager
def parquet():
    """Share exact Arrow files; DuckDB fixture writers lose nanosecond timezones."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory(prefix='sdm-parity-parquet-') as directory:
        settings = {}
        for name, fixture in fixtures('parquet').items():
            paths = {}
            folder = Path(directory) / Path(name).stem
            folder.mkdir()
            for table, frame in tables_of(fixture).items():
                path = folder / (table + '.parquet')
                data = pa.Table.from_pandas(frame, preserve_index=False)
                # Arrow types suffice for both readers; pandas extension metadata
                # is unnecessary and is not portable across pandas backends.
                pq.write_table(data.replace_schema_metadata(None), path)
                paths[table] = str(path)
            settings[name] = {'paths': paths}
        yield config_for('parquet', settings)


@contextlib.contextmanager
def s3(connector):
    import boto3
    import pyarrow as pa
    import pyarrow.parquet as pq
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(ip_address='127.0.0.1', port=0, verbose=False)
    server.start()
    try:
        host, port = server.get_host_and_port()
        endpoint = f'http://{host}:{port}'
        client = boto3.client(
            's3',
            endpoint_url=endpoint,
            region_name='us-east-1',
            aws_access_key_id='testing',
            aws_secret_access_key='testing',
        )
        bucket = 'sdm-ci-' + uuid.uuid4().hex
        client.create_bucket(Bucket=bucket)
        settings = {}
        for name, fixture in fixtures(connector).items():
            paths = {}
            for table, frame in tables_of(fixture).items():
                suffix = 'csv' if connector == 's3-csv' else 'parquet'
                key = f'{Path(name).stem}/{table}.{suffix}'
                if suffix == 'csv':
                    body = frame.to_csv(index=False).encode()
                else:
                    output = io.BytesIO()
                    arrow = pa.Table.from_pandas(frame, preserve_index=False)
                    if any(
                        spec.startswith('list<')
                        for spec in fixture['tables'][table]
                        .get('types', {})
                        .values()
                    ):
                        arrow = arrow.replace_schema_metadata(None)
                    pq.write_table(arrow, output)
                    body = output.getvalue()
                client.put_object(Bucket=bucket, Key=key, Body=body)
                paths[table] = f's3://{bucket}/{key}'
            settings[name] = {
                'paths': paths,
                's3_endpoint': endpoint,
                'storage_options': {
                    'key': 'testing',
                    'secret': 'testing',
                    'client_kwargs': {'endpoint_url': endpoint},
                },
            }
        yield config_for(connector, settings)
    finally:
        server.stop()  # All objects belong to this in-memory service.


def postgres_type(dtype):
    import pyarrow as pa

    if pa.types.is_list(dtype):
        return postgres_type(dtype.value_type) + '[]'
    if pa.types.is_boolean(dtype):
        return 'BOOLEAN'
    if pa.types.is_integer(dtype):
        return 'BIGINT' if dtype.bit_width > 32 else 'INTEGER'
    if pa.types.is_floating(dtype):
        return 'REAL' if dtype.bit_width == 32 else 'DOUBLE PRECISION'
    if pa.types.is_decimal(dtype):
        return f'NUMERIC({dtype.precision},{dtype.scale})'
    if pa.types.is_timestamp(dtype):
        return 'TIMESTAMPTZ' if dtype.tz else 'TIMESTAMP'
    if pa.types.is_date(dtype):
        return 'DATE'
    if pa.types.is_string(dtype) or pa.types.is_large_string(dtype):
        return 'TEXT'
    raise ValueError(
        'Fixture type needs an explicit PostgreSQL mapping: ' + str(dtype)
    )


@contextlib.contextmanager
def postgres():
    import psycopg
    import pyarrow as pa
    from psycopg import sql

    dsn = os.environ['SDM_POSTGRES_TEST_DSN']
    schema = 'sdm_ci_' + uuid.uuid4().hex
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(
            sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema))
        )
        try:
            settings = {}
            info = connection.info.get_parameters()
            for name, fixture in fixtures('postgres').items():
                mappings = {}
                for table, frame in tables_of(fixture).items():
                    actual = Path(name).stem.replace('-', '_') + '_' + table
                    mappings[table] = actual
                    data = pa.Table.from_pandas(frame, preserve_index=False)
                    target = sql.Identifier(schema, actual)
                    columns = sql.SQL(', ').join(
                        sql.SQL('{} {}').format(
                            sql.Identifier(field.name),
                            sql.SQL(postgres_type(field.type)),
                        )
                        for field in data.schema
                    )
                    connection.execute(
                        sql.SQL('CREATE TABLE {} ({})').format(target, columns)
                    )
                    with connection.cursor().copy(
                        sql.SQL('COPY {} FROM STDIN').format(target)
                    ) as copy:
                        for row in data.to_pylist():
                            copy.write_row(tuple(row.values()))
                connect = {**info, 'options': f'-c search_path={schema}'}
                # libpq may omit a password from connection.info; preserve the caller DSN.
                connect = {**psycopg.conninfo.conninfo_to_dict(dsn), **connect}
                host, port = (
                    connect.get('host', 'localhost'),
                    connect.get('port', '5432'),
                )
                settings[name] = {
                    'snapshot': digest(fixture),
                    'tables': mappings,
                    'source': {'schema': schema},
                    'connect': connect,
                    'java': {
                        'url': f'jdbc:postgresql://{host}:{port}/{connect.get("dbname", "postgres")}',
                        'properties': {
                            'user': connect.get('user', 'postgres'),
                            'password': connect.get('password', ''),
                            'currentSchema': schema,
                        },
                    },
                }
            yield config_for('postgres', settings)
        finally:
            connection.execute(
                sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema))
            )
