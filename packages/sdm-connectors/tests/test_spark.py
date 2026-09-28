# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A notebook's Spark session, driven through the DB-API calls the Databricks
backend makes. No pyspark here: the session is a double that records what it
was asked and answers from rows it was given."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from sdm_connectors import ConnectorError
from sdm_connectors.backends import spark as spark_backend
from sdm_connectors.backends.spark import SparkConnection


class _Row(tuple):
    r"""pyspark's ``Row`` is a tuple subclass; so is this."""


class _Frame:
    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def first(self) -> _Row | None:
        rows = self.collect()
        return rows[0] if rows else None

    def collect(self) -> list[_Row]:
        columns = [column.to_pylist() for column in self._table.columns]
        return [_Row(values) for values in zip(*columns, strict=True)]

    def toArrow(self) -> pa.Table:  # noqa: N802 - pyspark's name
        return self._table


class _Session:
    version = '4.0.0'

    def __init__(self, table: pa.Table) -> None:
        self.table = table
        self.calls: list[tuple[str, Any]] = []
        self.stopped = False

    def sql(self, query: str, args: Any = None) -> _Frame:
        self.calls.append((query, args))
        return _Frame(self.table)

    def stop(self) -> None:
        self.stopped = True


def _session() -> _Session:
    return _Session(pa.table({'a': [1, 2], 'b': ['x', 'y']}))


def test_parameters_are_bound_positionally_not_formatted() -> None:
    session = _session()
    with SparkConnection(session).cursor() as cursor:
        cursor.execute('SELECT * FROM t WHERE a = ?', parameters=["1' OR 1=1"])
    assert session.calls == [('SELECT * FROM t WHERE a = ?', ["1' OR 1=1"])]


def test_a_statement_without_parameters_passes_none() -> None:
    session = _session()
    with SparkConnection(session).cursor() as cursor:
        cursor.execute('SELECT 1')
    assert session.calls == [('SELECT 1', None)]


def test_rows_come_back_as_plain_tuples() -> None:
    with SparkConnection(_session()).cursor() as cursor:
        cursor.execute('SELECT a, b FROM t')
        assert cursor.fetchone() == (1, 'x')
        rows = cursor.fetchall()
    assert rows == [(1, 'x'), (2, 'y')]
    assert all(type(row) is tuple for row in rows)


def test_arrow_is_what_the_session_returns() -> None:
    session = _session()
    with SparkConnection(session).cursor() as cursor:
        cursor.execute('SELECT a, b FROM t')
        assert cursor.fetchall_arrow().equals(session.table)


def test_an_empty_result_has_no_first_row() -> None:
    session = _Session(pa.table({'a': pa.array([], pa.int64())}))
    with SparkConnection(session).cursor() as cursor:
        cursor.execute('SELECT a FROM t WHERE false')
        assert cursor.fetchone() is None


def test_fetching_before_executing_says_so() -> None:
    cursor = SparkConnection(_session()).cursor()
    with pytest.raises(ConnectorError, match='no statement has been executed'):
        cursor.fetchall()


def test_closing_never_stops_the_notebooks_session() -> None:
    session = _session()
    SparkConnection(session).close()
    assert not session.stopped


def test_something_that_is_not_a_session_is_refused() -> None:
    with pytest.raises(ConnectorError, match='expected a Spark session') as err:
        SparkConnection(object())
    assert err.value.code == 'INVALID_CONNECTOR_ARGS'


def test_a_spark_without_arrow_is_refused_up_front_with_the_way_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OldFrame:
        pass

    monkeypatch.setattr(spark_backend, '_dataframe_type', lambda _: _OldFrame)
    session = _session()
    session.version = '3.5.2'
    with pytest.raises(ConnectorError) as err:
        SparkConnection(session)
    message = str(err.value)
    assert 'Spark 3.5.2' in message
    assert '17.3' in message
    assert "connect('databricks'" in message
    assert err.value.code == 'INVALID_CONNECTOR_ARGS'


def test_a_frame_without_arrow_is_refused_at_fetch() -> None:
    class _NoArrowSession(_Session):
        def sql(self, query: str, args: Any = None) -> object:
            return object()

    with SparkConnection(_NoArrowSession(pa.table({}))).cursor() as cursor:
        cursor.execute('SELECT 1')
        with pytest.raises(ConnectorError, match=r'toArrow\(\)'):
            cursor.fetchall_arrow()
