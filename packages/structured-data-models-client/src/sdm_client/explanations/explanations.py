# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Each explained row's gradient explanation, keyed as the row is in the frame."""

from __future__ import annotations

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
        self._rows: dict[Any, Any] = {}
        # Each class as the NIM spells it, to the label of its frame column.
        self._labels: dict[Any, str] = {}

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
                    outputs.append(
                        f'probability_{self._labels.get(label, label)}'
                    )
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

    def _renumbered(self, start: int) -> GradientExplanations:
        r"""These, with each ``row_index`` ``start`` places further on.

        A key that is an ``id`` is not a position and is left alone.
        """
        if not start:
            return self
        moved = self._keyed_by == ('row_index',)
        shifted = GradientExplanations(
            {
                key + start if moved else key: entry
                for key, entry in self.items()
            },
            keyed_by=self._keyed_by,
            hops={
                key + start if moved else key: hops
                for key, hops in self._hops.items()
            },
        )
        shifted._rows = {
            key + start if moved else key: row + start
            for key, row in self._rows.items()
        }
        shifted._labels = dict(self._labels)
        return shifted

    def _at_step(self, step: Any) -> GradientExplanations:
        r"""These, as the explanations of one forecast step."""
        stepped = GradientExplanations(
            {(key, step): entry for key, entry in self.items()},
            keyed_by=(*self._keyed_by, 'forecast_step'),
            hops={(key, step): hops for key, hops in self._hops.items()},
        )
        stepped._rows = {
            (key, step): (row, step) for key, row in self._rows.items()
        }
        stepped._labels = dict(self._labels)
        return stepped

    @classmethod
    def _merged(
        cls, parts: Sequence[GradientExplanations]
    ) -> GradientExplanations:
        r"""The explanations of several calls, as one; by ``row_index`` if ids repeat."""
        explained = [part for part in parts if part]
        if not explained:
            return cls()
        keys = [key for part in explained for key in part]
        mixed = len({part._keyed_by for part in explained}) > 1
        if mixed or len(set(keys)) < len(keys):
            # Ids repeat or the parts disagree: every part falls back to its
            # rows' positions, which are unique across the whole result.
            by_row = []
            for part in explained:
                if part._keyed_by[0] == 'row_index':
                    by_row.append(part)
                    continue
                moved = cls(
                    {part._rows[key]: entry for key, entry in part.items()},
                    keyed_by=('row_index', *part._keyed_by[1:]),
                    hops={
                        part._rows[key]: hops
                        for key, hops in part._hops.items()
                    },
                )
                moved._rows = {row: row for row in part._rows.values()}
                moved._labels = dict(part._labels)
                by_row.append(moved)
            explained = by_row
        merged = cls(keyed_by=explained[0]._keyed_by)
        for part in explained:
            merged.update(part)
            merged._hops.update(part._hops)
            merged._rows.update(part._rows)
            merged._labels.update(part._labels)
        return merged
