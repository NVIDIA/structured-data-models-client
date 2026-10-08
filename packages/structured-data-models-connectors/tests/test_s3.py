# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import os

import pandas as pd
import pytest

from sdm_connectors import read
from sdm_connectors.sql import (
    ConnectorError,
    MissingBackendError,
)

# These bind a local socket, which a sandboxed or locked-down runner may
# refuse. Marked so such a runner can deselect them rather than report a
# failure it cannot act on.
pytestmark = pytest.mark.socket


s3fs = pytest.importorskip('s3fs')
moto_server = pytest.importorskip('moto.server')

BUCKET = 'sdm-connectors-test'


@pytest.fixture(scope='module')
def storage_options():
    with pytest.MonkeyPatch.context() as patcher:
        patcher.setenv('AWS_CONFIG_FILE', '/dev/null')
        patcher.setenv('AWS_SHARED_CREDENTIALS_FILE', '/dev/null')
        patcher.setenv('AWS_DEFAULT_REGION', 'us-east-1')
        patcher.setenv('AWS_REGION', 'us-east-1')
        # A profile named in the environment is looked up in the empty config
        # above, and fails there before the explicit key is ever read.
        patcher.delenv('AWS_PROFILE', raising=False)
        patcher.delenv('AWS_DEFAULT_PROFILE', raising=False)
        server = moto_server.ThreadedMotoServer(ip_address='127.0.0.1', port=0)
        server.start()
        try:
            host, port = server.get_host_and_port()
            options = {
                'key': 'testing',
                'secret': 'testing',
                'client_kwargs': {'endpoint_url': f'http://{host}:{port}'},
            }
            frame = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
            s3fs.S3FileSystem(**options).mkdir(BUCKET)
            frame.to_csv(
                f's3://{BUCKET}/table.csv',
                index=False,
                storage_options=options,
            )
            frame.to_parquet(
                f's3://{BUCKET}/nested/table.parquet',
                storage_options=options,
            )
            frame.to_parquet(
                f's3://{BUCKET}/upper/TABLE.PARQUET',
                storage_options=options,
            )
            frame.iloc[:2].to_parquet(
                f's3://{BUCKET}/dataset/part-0.parquet',
                storage_options=options,
            )
            frame.iloc[2:].to_parquet(
                f's3://{BUCKET}/dataset/part-1.parquet',
                storage_options=options,
            )
            yield options
        finally:
            server.stop()


def test_read_s3_csv(storage_options):
    frame = read(
        f's3://{BUCKET}/table.csv',
        storage_options=storage_options,
    )
    assert frame.shape == (3, 2)
    assert list(frame['b']) == ['x', 'y', 'z']


def test_read_s3_parquet(storage_options):
    frame = read(
        f's3://{BUCKET}/nested/table.parquet',
        storage_options=storage_options,
    )
    assert frame.shape == (3, 2)
    assert list(frame['a']) == [1, 2, 3]


def test_read_s3_uppercase_parquet_suffix(storage_options):
    frame = read(
        f's3://{BUCKET}/upper/TABLE.PARQUET',
        storage_options=storage_options,
    )
    assert frame.shape == (3, 2)


def test_read_s3_parquet_dataset_prefix(storage_options):
    frame = read(
        f's3://{BUCKET}/dataset/',
        storage_options=storage_options,
    )
    assert frame.shape == (3, 2)


def test_read_s3_missing_object_maps_to_not_found(storage_options):
    with pytest.raises(ConnectorError) as excinfo:
        read(
            f's3://{BUCKET}/absent.csv',
            storage_options=storage_options,
        )
    assert excinfo.value.code == 'NOT_FOUND'
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_read_s3_unreachable_endpoint_maps_to_read_failed():
    with pytest.raises(ConnectorError) as excinfo:
        read(
            's3://bucket/table.csv',
            storage_options={
                'key': 'x',
                'secret': 'x',
                'client_kwargs': {'endpoint_url': 'http://127.0.0.1:1'},
                'config_kwargs': {
                    'connect_timeout': 1,
                    'retries': {'max_attempts': 0},
                },
            },
        )
    assert excinfo.value.code == 'READ_FAILED'
    assert excinfo.value.details['path'] == 's3://bucket/table.csv'


def test_another_object_store_is_refused_naming_s3():
    with pytest.raises(ConnectorError) as excinfo:
        read('gs://bucket/table.csv')
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert 's3://' in excinfo.value.message


def test_read_s3_missing_driver_raises(monkeypatch):
    def raise_missing(name, *args, **kwargs):
        raise ModuleNotFoundError(f"No module named '{name}'", name=name)

    monkeypatch.setattr(importlib, 'import_module', raise_missing)
    with pytest.raises(MissingBackendError) as excinfo:
        read('s3://bucket/table.csv')
    assert excinfo.value.backend == 's3'
    assert excinfo.value.driver == 's3fs'
    assert 'structured-data-models-connectors[s3]' in str(excinfo.value)


@pytest.mark.skipif(
    'KUMO_RELATIONAL_S3_LIVE_URI' not in os.environ,
    reason='KUMO_RELATIONAL_S3_LIVE_URI not set',
)
def test_read_s3_live():
    anon = os.environ.get('KUMO_RELATIONAL_S3_LIVE_ANON', '').lower()
    options = {'anon': True} if anon in ('1', 'true', 'yes') else None
    frame = read(
        os.environ['KUMO_RELATIONAL_S3_LIVE_URI'],
        storage_options=options,
    )
    assert len(frame) > 0
