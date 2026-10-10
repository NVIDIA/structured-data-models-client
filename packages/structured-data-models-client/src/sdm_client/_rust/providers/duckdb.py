# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The DuckDB provider: it answers a fetch step by running the statements
its plan carries, in order, on a cursor of the source's connection, and
hands back the last one's result as an Arrow table, every batch the driver
made, an empty result keeping its schema.

The statements are the core's (``sdm-data`` renders DuckDB's SQL); this
module writes none. Each request gets its own cursor, as the reference's
reads do. A plan this provider cannot run, one for a host provider or one
with parameters or key sets, which no request it answers carries yet
(T1.11c), raises, and the run loop hands the core a provider error.

It imports nothing but the standard library: not pandas, not this client,
so it runs where pandas cannot be imported.
"""

from __future__ import annotations

from typing import Any

#: Rows per batch the driver hands over; a larger result comes as several.
BATCH_ROWS = 1_000_000


class PlanNotRun(Exception):
    r"""A plan this provider does not run. Its text names no value."""


class DuckDBProvider:
    r"""Answers fetch steps on ``connection``, a DuckDB connection or what
    stands in for one."""

    def __init__(
        self, connection: Any, *, batch_rows: int = BATCH_ROWS
    ) -> None:
        self._connection = connection
        self._batch_rows = batch_rows

    def __call__(self, step: dict[str, Any]) -> Any:
        plan = step['plan']
        if plan['plan'] != 'sql':
            raise PlanNotRun("a host plan is not a SQL provider's to run")
        statements = plan['statements']
        if not statements or any(
            statement['dialect'] != 'duckdb' or statement['params']
            for statement in statements
        ):
            raise PlanNotRun(
                'a plan with no statement, another dialect or parameters'
            )
        with self._connection.cursor() as cursor:
            for statement in statements:
                cursor.execute(statement['text'])
            return cursor.to_arrow_table(self._batch_rows)
