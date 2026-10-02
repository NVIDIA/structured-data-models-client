# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""How well a table's key identifies its rows, counted where the rows are."""

from collections.abc import Sequence
from typing import Any, NamedTuple

from sdm_relational_engine.rfm.base.table import Table


class KeyCount(NamedTuple):
    r"""What one scan found about a table's key.

    Attributes:
        rows: The rows of the table, or ``None`` where nothing was counted.
        distinct: The distinct values of the key among the rows that hold
            one, or ``None`` where nothing was counted.
        nulls: The rows missing the key or any part of it.
        enforced: Whether the source itself refuses a repeated key. Nothing
            is counted when it also refuses a missing one.
        example: A key value more than one row holds, if any does.
    """

    rows: int | None
    distinct: int | None
    nulls: int
    enforced: bool
    example: Any = None

    @property
    def repeats(self) -> int:
        r"""The rows holding a key value another row holds already."""
        if self.rows is None or self.distinct is None:
            return 0
        return self.rows - self.nulls - self.distinct


def count_keys(keys: Sequence[tuple[Table, Sequence[str]]]) -> list[KeyCount]:
    r"""Count the rows, distinct key values and missing keys of each table.

    Every table is counted in one ``UNION ALL`` statement, so a source pays
    one round trip however many tables it has. A key the source enforces is
    left out of it, or counted for nulls alone where the source lets a key be
    missing. One value that repeats is then read for each key that repeats,
    so a message can show it.

    Args:
        keys: Each table with the columns of its key. The tables share one
            source and so one connection.

    Returns:
        A :class:`KeyCount` for each table, in order.
    """
    counts: list[KeyCount] = [
        KeyCount(None, None, 0, True) for _ in range(len(keys))
    ]
    selects: list[str] = []
    unique_by_label: dict[int, bool] = {}
    for label, (table, names) in enumerate(keys):
        unique, never_null = table._enforces_key(names)
        if unique and never_null:
            continue
        unique_by_label[label] = unique
        selects.append(table._key_count_sql(label, names, unique=unique))
    if not selects:
        return counts

    first = keys[next(iter(unique_by_label))][0]
    for label, rows, distinct, nulls in first._fetch_rows(
        '\nUNION ALL\n'.join(selects)
    ):
        label = int(label)
        counts[label] = KeyCount(
            rows=int(rows),
            distinct=int(distinct),
            nulls=int(nulls),
            enforced=unique_by_label[label],
        )

    for label, count in enumerate(counts):
        if count.repeats <= 0:
            continue
        table, names = keys[label]
        found = table._fetch_rows(table._repeated_key_sql(names))
        if found:
            row = tuple(found[0])
            example = row[0] if len(names) == 1 else row
            counts[label] = count._replace(example=example)
    return counts
