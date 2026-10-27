# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The DuckDB provider: it answers a fetch step by running the statements
its plan carries, in order, on a cursor of the source's connection, and
hands back the last one's result as an Arrow table, every batch the driver
made, an empty result keeping its schema.

The statements are the core's (``sdm-data`` renders DuckDB's SQL); this
module writes none. Each request gets its own cursor, as the reference's
reads do. A key set a plan names (from T1.11c, the seeds of a ``neighbors``
or ``lookup`` request) is the Arrow attachment the core sent with the step,
which the run loop hands over by id: it is registered on the cursor as the
relation its parameter names, as the reference registers its key tables,
so its rows reach the statement as Arrow, never as text. A plan this
provider cannot run, one for a host provider, one with a scalar parameter,
which no request carries yet, or one naming a key set it was not handed,
raises, and the run loop hands the core a provider error.

It imports nothing of pandas, nor of this client, so it runs where pandas
cannot be imported; it reads a key set with pyarrow, the client's base
dependency, so the relation it registers is an Arrow table, as the
reference's is.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sdm_connectors import providers
from sdm_connectors.providers import PlanNotRun as PlanNotRun
from sdm_connectors.providers import arrow_table

#: Rows per batch the driver hands over; a larger result comes as several.
BATCH_ROWS = 1_000_000


class DuckDBProvider:
    r"""Answers fetch steps on ``connection``, a DuckDB connection or what
    stands in for one."""

    def __init__(
        self, connection: Any, *, batch_rows: int = BATCH_ROWS
    ) -> None:
        self._connection = connection
        self._batch_rows = batch_rows

    def __call__(
        self, step: dict[str, Any], keys: Mapping[int, Any] | None = None
    ) -> Any:
        statements = providers.statements(step, 'duckdb')
        relations = providers.bound_relations(statements, keys)
        with self._connection.cursor() as cursor:
            for relation, attachment in relations.items():
                cursor.register(relation, arrow_table((keys or {})[attachment]))
            for statement in statements:
                cursor.execute(statement['text'])
            return cursor.to_arrow_table(self._batch_rows)
