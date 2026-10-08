# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""A metric view named as a table is refused before any of its rows are read.

Databricks refuses to read a metric view's measures without ``MEASURE()``, so
reading one as a table failed on the first preview with
``METRIC_VIEW_MISSING_MEASURE_FUNCTION``, which names neither the cause nor
the way to read the view. ``information_schema`` already says what the
relation is, in the same look-up that reads its columns.
"""

from __future__ import annotations

from typing import Any

import pytest
from sdm_relational_engine import MetricViewAsTableError
from sdm_relational_engine.rfm import Graph

try:
    from sdm_relational_engine.rfm.backend.databricks import Connection
except ImportError:
    pytest.skip("'databricks' extension not installed", allow_module_level=True)


class _Cursor:
    r"""Answers the column look-up as a catalog holding one metric view, and
    fails on any other statement, so a read of its rows cannot pass unseen.
    """

    def __init__(self, connection: _Connection) -> None:
        self._connection = connection
        self._rows: list = []

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: Any) -> bool:
        return False

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._connection.statements.append(sql)
        assert 'information_schema.columns' in sql, f'unexpected SQL: {sql}'
        self._rows = [
            (name, dtype, 'YES', 'cat', 'sch', 'sales_metrics', 'METRIC_VIEW')
            for name, dtype in (('region', 'string'), ('revenue', 'double'))
        ]

    def fetchall(self) -> list:
        return self._rows

    def fetchall_arrow(self) -> Any:
        import pyarrow as pa

        return pa.table(
            {
                f'column_{index}': [row[index] for row in self._rows]
                for index in range(len(self._rows[0]))
            }
        )


class _Connection(Connection):
    def __init__(self) -> None:
        self.statements: list[str] = []

    def cursor(self) -> _Cursor:  # type: ignore[override]
        return _Cursor(self)

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        pass


def test_a_metric_view_is_named_as_one_before_its_rows_are_read() -> None:
    connection = _Connection()

    with pytest.raises(MetricViewAsTableError) as info:
        Graph.from_databricks(
            connection=connection,
            catalog='cat',
            schema='sch',
            tables=['Sales_Metrics'],
            infer_metadata=False,
            verbose=False,
        )

    assert info.value.name == 'Sales_Metrics'
    message = str(info.value)
    assert "'cat.sch.Sales_Metrics' is a metric view" in message
    assert (
        "Graph.from_databricks_metric_view('Sales_Metrics', connection, "
        "catalog='cat', schema='sch')"
    ) in message
    assert len(connection.statements) == 1
    # Still a ValueError, as a table that is not there is.
    assert isinstance(info.value, ValueError)
