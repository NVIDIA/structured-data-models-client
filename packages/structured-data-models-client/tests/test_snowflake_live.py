# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A brief check of the built-in Snowflake source on a real Snowflake account.

Java's ``SnowflakeTest#liveSchemaFromSource...`` is the same check. Both read
the variables live parity uses and run in the live-warehouse workflow, where
``SDM_REQUIRE_SNOWFLAKE=1`` turns a missing variable into a failure.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from typing import Any
from urllib.parse import urlparse

import pytest

from sdm_client import Schema, SnowflakeSource

pytestmark = pytest.mark.live_warehouse


@pytest.fixture
def connection() -> Iterator[Any]:
    url = os.environ.get('SNOWFLAKE_URL', '')
    if not url:
        if os.environ.get('SDM_REQUIRE_SNOWFLAKE') == '1':
            pytest.fail('SDM_REQUIRE_SNOWFLAKE=1 but SNOWFLAKE_URL is not set')
        pytest.skip(
            'requires SNOWFLAKE_URL, SNOWFLAKE_USERNAME and SNOWFLAKE_ACCESS_TOKEN'
        )
    connector = pytest.importorskip('snowflake.connector')
    host = urlparse(url if '://' in url else 'https://' + url).hostname
    options = {
        'account': host.removesuffix('.snowflakecomputing.com'),
        'host': host,
        'user': os.environ['SNOWFLAKE_USERNAME'],
        'password': os.environ['SNOWFLAKE_ACCESS_TOKEN'],
        'database': os.environ['PARITY_SNOWFLAKE_DATABASE'],
        'login_timeout': 30,
    }
    if os.environ.get('PARITY_SNOWFLAKE_WAREHOUSE'):
        options['warehouse'] = os.environ['PARITY_SNOWFLAKE_WAREHOUSE']
    opened = connector.connect(**options)
    yield opened
    opened.close()


def test_from_source_describes_and_checks_keys_through_the_source(
    connection: Any,
) -> None:
    database = os.environ['PARITY_SNOWFLAKE_DATABASE']
    schema = 'SDM_CI_PY_' + uuid.uuid4().hex[:12].upper()
    cursor = connection.cursor()
    cursor.execute(f'CREATE SCHEMA {schema}')
    try:
        cursor.execute(
            f'CREATE TABLE {schema}.USERS (USER_ID NUMBER(38,0) PRIMARY KEY, COUNTRY VARCHAR)'
        )
        cursor.execute(
            f'CREATE TABLE {schema}.EVENTS (USER_ID NUMBER(38,0), KIND VARCHAR, TS TIMESTAMP_NTZ)'
        )
        cursor.execute(
            f"INSERT INTO {schema}.USERS VALUES (1, 'DE'), (2, 'US')"
        )
        cursor.execute(
            f"INSERT INTO {schema}.EVENTS VALUES (1, 'a', '2024-01-01'), "
            f"(1, 'a', '2024-01-01'), (2, 'b', '2024-01-02')"
        )
        described = Schema.from_source(
            SnowflakeSource(connection, database=database, schema=schema)
        )
        assert sorted(described.tables) == ['EVENTS', 'USERS']
        assert described['USERS'].primary_key == 'USER_ID'
        assert described['EVENTS'].primary_key is None
        assert described['EVENTS'].time_column == 'TS'
    finally:
        cursor.execute(f'DROP SCHEMA IF EXISTS {schema}')
        cursor.close()
