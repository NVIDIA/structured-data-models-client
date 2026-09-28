# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A notebook's Spark session, seen as the few DB-API calls the Databricks
backend makes.

Inside a Databricks notebook the session is already there, already signed in
as the notebook's user, and already running on compute that is paid for. Going
through it instead of a SQL warehouse means there is no warehouse to name and
no second bill. The SQL the Databricks samplers issue -- ``QUALIFY``,
``TABLESAMPLE ... REPEATABLE``, ``information_schema``, ``posexplode`` over a
JSON parameter -- is Databricks SQL, which the notebook's Spark runs as it is.

Nothing here imports pyspark. The session is taken as given and only its
``sql`` method is called, so the classic and the Spark Connect session behave
the same, and this module imports on a machine that has neither.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from sdm_connectors.sql import ConnectorError

if TYPE_CHECKING:
    import pyarrow as pa

_NO_ARROW = (
    'cannot return results as Arrow, which sampling through a Spark session '
    'needs: DataFrame.toArrow() arrived in Spark 4.0. Use Databricks Runtime '
    '17.3 LTS or later, or serverless environment 4 or later; on an older '
    "runtime, connect through a SQL warehouse with connect('databricks', ...) "
    'instead.'
)


class SparkCursor:
    r"""One statement at a time, run through ``spark.sql``.

    ``execute`` builds the DataFrame; the fetch methods run it. Spark resolves
    names and types when the DataFrame is built, so a statement that names a
    missing table fails in ``execute``, where a driver would fail too.
    """

    def __init__(self, session: Any) -> None:
        self._session = session
        self._frame: Any = None

    def execute(
        self,
        operation: str,
        parameters: Sequence[Any] | None = None,
    ) -> SparkCursor:
        r"""Run ``operation``, binding ``?`` markers positionally.

        Spark binds positional ``?`` markers from a list since Spark 3.5,
        which every runtime this supports is past.
        """
        if parameters:
            self._frame = self._session.sql(operation, args=list(parameters))
        else:
            self._frame = self._session.sql(operation)
        return self

    def _require_frame(self) -> Any:
        if self._frame is None:
            raise ConnectorError(
                'nothing to fetch: no statement has been executed on this '
                'cursor',
                code='QUERY_FAILED',
            )
        return self._frame

    def fetchone(self) -> tuple[Any, ...] | None:
        row = self._require_frame().first()
        return None if row is None else tuple(row)

    def fetchall(self) -> list[tuple[Any, ...]]:
        return [tuple(row) for row in self._require_frame().collect()]

    def fetchall_arrow(self) -> pa.Table:
        frame = self._require_frame()
        if not hasattr(frame, 'toArrow'):
            raise ConnectorError(
                f'this Spark session {_NO_ARROW}',
                code='INVALID_CONNECTOR_ARGS',
            )
        return frame.toArrow()

    def close(self) -> None:
        self._frame = None

    def __enter__(self) -> SparkCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class SparkConnection:
    r"""A Spark session, shaped like the connection the Databricks backend
    takes.

    Closing it never stops the session: the notebook owns that, and a closed
    session would take every other cell down with it.

    Args:
        session: The notebook's ``spark`` -- a classic ``SparkSession`` or a
            Spark Connect one, as serverless compute provides.

    Raises:
        ConnectorError: ``INVALID_CONNECTOR_ARGS`` if ``session`` is not a
            Spark session, or if its DataFrames cannot return Arrow. ``DataFrame.toArrow`` arrived in Spark 4.0, which is
            Databricks Runtime 17.3 LTS and serverless environment 4; the
            samplers read every result as Arrow, and the pandas route loses
            nullable integers and decimals on the way.
    """

    def __init__(self, session: Any) -> None:
        if not callable(getattr(session, 'sql', None)) or not hasattr(
            session, 'version'
        ):
            raise ConnectorError(
                f'expected a Spark session, such as the `spark` a Databricks '
                f'notebook provides; got {type(session).__name__}',
                code='INVALID_CONNECTOR_ARGS',
            )
        frame_type = _dataframe_type(session)
        if frame_type is not None and not hasattr(frame_type, 'toArrow'):
            raise ConnectorError(
                f'this Spark session (Spark {session.version}) {_NO_ARROW}',
                code='INVALID_CONNECTOR_ARGS',
                details={'spark_version': str(session.version)},
            )
        self._session = session

    @property
    def session(self) -> Any:
        r"""The wrapped Spark session."""
        return self._session

    def cursor(self) -> SparkCursor:
        return SparkCursor(self._session)

    def close(self) -> None:
        r"""Do nothing: the session belongs to the notebook, not to us."""

    def __repr__(self) -> str:
        return f'SparkConnection(Spark {self._session.version})'


def _dataframe_type(session: Any) -> type | None:
    r"""The DataFrame class this session returns, without running a query.

    Read off the class that ``session.sql`` is defined on: a classic session
    lives in ``pyspark.sql.session`` and returns ``pyspark.sql.DataFrame``, a
    Spark Connect one lives in ``pyspark.sql.connect.session`` and returns
    ``pyspark.sql.connect.dataframe.DataFrame``. Anything else -- a test
    double -- is not checked here, and is taken at its word.
    """
    module = type(session).__module__
    try:
        if module.startswith('pyspark.sql.connect'):
            from pyspark.sql.connect.dataframe import DataFrame

            return DataFrame
        if module.startswith('pyspark.sql'):
            from pyspark.sql import DataFrame as ClassicDataFrame

            return ClassicDataFrame
    except ImportError:
        return None
    return None
