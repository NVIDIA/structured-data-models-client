# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pyarrow as pa
import pytest

pytest.importorskip(
    'snowflake.connector',
    reason="requires the 'snowflake' extra",
)

from sdm_relational_engine.rfm.backend.snow.table import SnowTable


class _StubTable:
    r"""What ``_get_num_rows`` reads: its names, and one ``SHOW TABLES`` row
    holding the count, read by field name.
    """

    _listed = SnowTable.__dict__['_listed']
    _named = SnowTable._named

    def __init__(self, num_rows: int) -> None:
        self._num_rows = num_rows
        self._source_name = 'AUDIT_EMPTY'
        self._database = 'DB'
        self._schema = 'PUBLIC'
        self.source_name = 'DB.PUBLIC.AUDIT_EMPTY'

    def _run(self, statement: Any) -> pa.Table:
        return pa.table(
            {
                'created_on': [''],
                'name': ['AUDIT_EMPTY'],
                'database_name': ['DB'],
                'schema_name': ['PUBLIC'],
                'kind': ['TABLE'],
                'comment': [''],
                'cluster_by': [''],
                'rows': pa.array([self._num_rows], pa.int64()),
            }
        )


def test_get_num_rows_empty_table_names_the_table() -> None:
    # Regression: this message used to print the literal '{self.source_name}'.
    with pytest.raises(RuntimeError, match="'DB.PUBLIC.AUDIT_EMPTY' is empty"):
        SnowTable._get_num_rows(_StubTable(0))  # type: ignore[arg-type]


def test_get_num_rows_non_empty_table() -> None:
    assert SnowTable._get_num_rows(_StubTable(7)) == 7  # type: ignore[arg-type]
