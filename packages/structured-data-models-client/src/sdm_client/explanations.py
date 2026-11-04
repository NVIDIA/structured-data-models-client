# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Each explained row's gradient explanation, keyed as the row is in the frame."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

# The columns of `GradientExplanations.to_pandas()` after the ones naming the row.
_GRADIENT_COLUMNS = (
    'table',
    'hop',
    'key',
    'row',
    'output',
    'column',
    'gradient',
)


class GradientExplanations(dict[Any, dict[str, Any]]):
    r"""Each explained row's gradient explanation, keyed as the row is in the frame."""

    def __init__(
        self,
        entries: Mapping[Any, dict[str, Any]] | None = None,
        *,
        keyed_by: tuple[str, ...] = ('id',),
        hops: Mapping[Any, Mapping[str, Sequence[int | None]]] | None = None,
    ) -> None:
        super().__init__(entries or {})
        self._keyed_by = keyed_by
        self._hops = {key: dict(tables) for key, tables in (hops or {}).items()}
        self._records: list[dict[str, Any]] = []
        self._entries_by_answer: dict[int, dict[str, Any]] = {}

    def to_pandas(self) -> pd.DataFrame:
        r"""Every gradient, one line each, across all explained rows.

        One line per (row, table, related row, output, column). ``table`` is
        ``'instance'`` for a ``kumo-tabular`` row's own block and the table's
        name for a related one, the ``kumo-relational`` entity's own table
        among them. ``hop`` is how many links the row is from the entity: 0
        for the entity itself, and ``None`` where the rows sent do not link
        it. ``key`` is the related row's entry in ``keys``, present only when a
        relationship joins on the table's key column, else ``None``; for an
        instance block it is the row's own ``id``. ``row`` is the
        position in that entry's ``related[table]['rows']``. ``output`` names
        the score as the column of ``result.to_pandas()`` that holds it.
        """
        lines: list[tuple[Any, ...]] = []
        for key, entry in self.items():
            named = key if len(self._keyed_by) > 1 else (key,)
            # Each score's column in `result.to_pandas()`: probabilities by
            # the frame's spelling of the class, quantiles by their level.
            outputs = []
            for output in entry['outputs']:
                kind = output.get('kind')
                if kind == 'probability':
                    label = output.get('class')
                    outputs.append(f'probability_{label}')
                elif kind == 'quantile':
                    outputs.append(f'quantile_{output.get("level")}')
                else:
                    outputs.append(str(kind))
            # A kumo-relational entry has none: its own row is in `related`.
            instance = entry.get('instance')
            own = named[0] if self._keyed_by[0] == 'id' else None
            if instance is not None:
                for output, by_column in zip(
                    outputs, instance['gradients'], strict=True
                ):
                    for column, gradient in zip(
                        instance['columns'], by_column, strict=True
                    ):
                        lines.append(
                            (
                                *named,
                                'instance',
                                0,
                                own,
                                0,
                                output,
                                column,
                                gradient,
                            )
                        )
            hops = self._hops.get(key, {})
            for table, block in entry.get('related', {}).items():
                keys = block.get('keys') or [None] * len(block['rows'])
                at = hops.get(table) or [None] * len(block['rows'])
                for output, by_row in zip(
                    outputs, block['gradients'], strict=True
                ):
                    for row, by_column in enumerate(by_row):
                        for column, gradient in zip(
                            block['columns'], by_column, strict=True
                        ):
                            lines.append(
                                (
                                    *named,
                                    table,
                                    at[row],
                                    keys[row],
                                    row,
                                    output,
                                    column,
                                    gradient,
                                )
                            )
        # Built as objects so a key keeps its own type: an integer key beside a
        # keyless table's `None` would otherwise come back a float.
        return pd.DataFrame(
            lines, columns=[*self._keyed_by, *_GRADIENT_COLUMNS], dtype=object
        ).astype({'hop': 'Int64', 'row': 'int64', 'gradient': 'float64'})

    @classmethod
    def _from_core(
        cls,
        collection: dict[str, Any] | None,
        entries: Mapping[int, dict[str, Any]] | None = None,
    ) -> GradientExplanations:
        if collection is None:
            result = cls()
            result._records = []
            return result
        records = collection['rows']
        stepped = any(row['forecast_step'] is not None for row in records)
        kind = collection['keyed_by']
        result = cls(keyed_by=(kind, 'forecast_step') if stepped else (kind,))
        result._records = [
            dict(
                row,
                explanation='{}' if row['explanation'] is not None else None,
            )
            for row in records
        ]
        for row in records:
            if row['explanation'] is None:
                continue
            identity = (
                json.loads(row['id']) if kind == 'id' else row['row_index']
            )
            key = (identity, row['forecast_step']) if stepped else identity
            entry = (
                entries[row['answer_index']]
                if entries is not None
                else json.loads(row['explanation'])
            )
            result._entries_by_answer[row['answer_index']] = entry
            result._hops[key] = entry.get('hops', {})
            result[key] = {
                name: value for name, value in entry.items() if name != 'hops'
            }
        return result
