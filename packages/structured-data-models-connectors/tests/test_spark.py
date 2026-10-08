# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A notebook's Spark session, as the Databricks source runs SQL through it.
No pyspark here: the session is a double that records what it was asked and
answers from rows it was given."""

from __future__ import annotations

from typing import Any

import pyarrow as pa
import pytest

from sdm_connectors import ConnectorError, DatabricksSource
from sdm_connectors.backends import spark as spark_backend
from sdm_connectors.backends.spark import SparkConnection, run


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


@pytest.mark.parametrize(
    'params, bound',
    [(["1' OR 1=1"], ["1' OR 1=1"]), (None, None), ([], None)],
    ids=['bound', 'none', 'empty'],
)
def test_parameters_are_bound_positionally_not_formatted(
    params: Any, bound: Any
) -> None:
    session = _session()
    run(SparkConnection(session), 'SELECT * FROM t WHERE a = ?', params)
    assert session.calls == [('SELECT * FROM t WHERE a = ?', bound)]


def test_a_result_is_read_as_arrow() -> None:
    session = _session()
    assert run(SparkConnection(session), 'SELECT a, b FROM t').equals(
        session.table
    )

    session.table = pa.table({'a': pa.array([], pa.int64())})
    table = run(SparkConnection(session), 'SELECT a FROM t WHERE false')
    assert table.num_rows == 0
    assert table.schema.field('a').type == pa.int64()


class _SparkSession(_Session):
    pass


# The source recognises a notebook's session by its class's name and module.
_SparkSession.__name__ = 'SparkSession'
_SparkSession.__module__ = 'pyspark.sql.session'


def test_the_source_runs_through_the_session() -> None:
    session = _SparkSession(pa.table({'n': [3]}))
    source = DatabricksSource(session, catalog='main', schema='shop')
    assert isinstance(source.connection, SparkConnection)
    table = source._run('SELECT COUNT(*) AS n FROM t WHERE a > ?', [1])
    assert table.to_pylist() == [{'n': 3}]
    assert session.calls == [('SELECT COUNT(*) AS n FROM t WHERE a > ?', [1])]


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
    assert 'databricks.sql.connect' in message
    assert err.value.code == 'INVALID_CONNECTOR_ARGS'


def test_a_frame_without_arrow_is_refused_when_read() -> None:
    class _NoArrowSession(_Session):
        def sql(self, query: str, args: Any = None) -> object:
            return object()

    with pytest.raises(ConnectorError, match=r'toArrow\(\)') as err:
        run(SparkConnection(_NoArrowSession(pa.table({}))), 'SELECT 1')
    assert err.value.code == 'INVALID_CONNECTOR_ARGS'
