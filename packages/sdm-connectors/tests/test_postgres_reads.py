# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Reading over a PostgreSQL connection the caller keeps open.

psycopg opens a transaction on a connection's first statement and keeps it
until a commit, so a read that does not end its own leaves the caller's
connection idle in transaction, holding its locks, and after one failed read
refuses every later one.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Any

import pytest

from sdm_connectors import connect, read_table
from sdm_connectors.sql import ConnectorError

try:
    from psycopg.pq import TransactionStatus
except ImportError:
    if os.getenv('SDM_REQUIRE_POSTGRES'):
        raise
    pytest.skip("'postgres' extension not installed", allow_module_level=True)


@pytest.fixture(scope='module')
def postgres_dsn() -> str:
    r"""The PostgreSQL server these tests read from.

    Skipped where none is configured. CI sets ``SDM_REQUIRE_POSTGRES`` beside
    the DSN, and there a missing DSN fails, since a skip would read as a pass.
    """
    dsn = os.getenv('SDM_POSTGRES_TEST_DSN')
    if dsn:
        return dsn
    if os.getenv('SDM_REQUIRE_POSTGRES'):
        pytest.fail(
            'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
        )
    pytest.skip('SDM_POSTGRES_TEST_DSN is not set')


@pytest.fixture
def connection(postgres_dsn: str) -> Iterator[Any]:
    connection = connect('postgres', postgres_dsn)
    try:
        yield connection
    finally:
        connection.close()


def test_a_read_ends_its_own_transaction(connection: Any) -> None:
    frame = read_table(connection, query='SELECT 1 AS one')

    assert frame['one'].tolist() == [1]
    assert connection.info.transaction_status == TransactionStatus.IDLE


def test_a_failed_read_leaves_the_connection_usable(connection: Any) -> None:
    with pytest.raises(ConnectorError):
        read_table(connection, table='no_such_table')
    assert connection.info.transaction_status == TransactionStatus.IDLE

    frame = read_table(connection, query='SELECT 1 AS one')

    assert frame['one'].tolist() == [1]


def test_a_read_inside_the_callers_transaction_leaves_it_open(
    connection: Any,
) -> None:
    with connection.transaction():
        read_table(connection, query='SELECT 1 AS one')

        assert connection.info.transaction_status == TransactionStatus.INTRANS
