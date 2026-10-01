# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

pytest.importorskip(
    'snowflake.connector',
    reason="requires the 'snowflake' extra",
)

from sdm_relational_engine.rfm.backend.snow.table import SnowTable


class _FakeCursor:
    def __init__(self, row: tuple[Any, ...]) -> None:
        self._row = row

    def __enter__(self) -> '_FakeCursor':
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def execute(self, sql: str, parameters: Any = None) -> None:
        pass

    def fetchone(self) -> tuple[Any, ...]:
        return self._row


class _FakeConnection:
    def __init__(self, num_rows: int) -> None:
        self._num_rows = num_rows
        self._paramstyle = 'pyformat'

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(('',) * 7 + (self._num_rows,))


class _StubTable:
    def __init__(self, num_rows: int) -> None:
        self._connection = _FakeConnection(num_rows)
        self._source_name = 'AUDIT_EMPTY'
        self._database = 'DB'
        self._schema = 'PUBLIC'
        self.source_name = 'DB.PUBLIC.AUDIT_EMPTY'


def test_get_num_rows_empty_table_names_the_table() -> None:
    # Regression: this message used to print the literal '{self.source_name}'.
    with pytest.raises(RuntimeError, match="'DB.PUBLIC.AUDIT_EMPTY' is empty"):
        SnowTable._get_num_rows(_StubTable(0))  # type: ignore[arg-type]


def test_get_num_rows_non_empty_table() -> None:
    assert SnowTable._get_num_rows(_StubTable(7)) == 7  # type: ignore[arg-type]
