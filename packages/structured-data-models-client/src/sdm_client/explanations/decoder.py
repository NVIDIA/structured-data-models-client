# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``sdm_gradient_v1`` records decoded per call into GradientExplanations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sdm_api.generated.tfm_api import PredictionItem

from sdm_client.explanations.explanations import GradientExplanations
from sdm_client.wire.answers import class_labels
from sdm_client.wire.dtypes import as_declared


class GradientDecoder:
    r"""Decodes one call's explanation records into GradientExplanations.

    Takes the call's payload for kumo-relational, to map related-row positions
    back to keys and values; kumo-tabular needs none, its explanation is about
    the record's own row. ``labelled_by`` is the payload whose target names
    each class as the frame does; it defaults to ``payload``.
    """

    #: Engine transport columns, not the caller's; stripped from related rows.
    INSTANCE_ID = 'instance_id'
    SYNTHETIC_KEY = '__node_id'

    def __init__(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        labelled_by: Mapping[str, Any] | None = None,
    ) -> None:
        self._payload = payload
        self._labelled_by = payload if labelled_by is None else labelled_by
        self._hops: dict[str, list[int | None]] | None = None

    def decode(self, items: Sequence[PredictionItem]) -> GradientExplanations:
        r"""Each explained item's entry, by ``id`` if those are unique, else ``row_index``."""
        explained = [item for item in items if item.explanation]
        if not explained:
            return GradientExplanations()
        ids = [item.id for item in explained]
        unique = len(set(ids)) == len(ids)
        by_id = unique and any(key is not None for key in ids)
        entries: dict[Any, dict[str, Any]] = {}
        hops: dict[Any, dict[str, list[int | None]]] = {}
        for item in explained:
            key = item.id if by_id else item.row_index
            details = (item.explanation or {}).get('details') or {}
            entry: dict[str, Any] = {
                'outputs': list(details.get('outputs') or [])
            }
            if self._payload is None:
                instance = details.get('instance_table') or {}
                entry['instance'] = {
                    'columns': list(instance.get('columns') or []),
                    # One row, the record's own, so the row axis says nothing.
                    'gradients': [
                        by_row[0] for by_row in instance.get('gradients') or []
                    ],
                }
            else:
                # Its instance_table is the engine's question row, not the caller's.
                entry['related'], hops[key] = self._read_related(
                    self._payload, details.get('related_tables') or {}
                )
            entries[key] = entry
        classes = [
            output['class']
            for entry in entries.values()
            for output in entry['outputs']
            if output.get('kind') == 'probability' and 'class' in output
        ]
        labels = class_labels(list(dict.fromkeys(classes)), self._labelled_by)
        decoded = GradientExplanations(
            entries,
            keyed_by=('id',) if by_id else ('row_index',),
            hops=hops,
        )
        decoded._rows = {
            item.id if by_id else item.row_index: item.row_index
            for item in explained
        }
        decoded._labels = dict(labels)
        return decoded

    def _read_related(
        self,
        payload: Mapping[str, Any],
        blocks: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, list[int | None]]]:
        r"""A record's related blocks as keys and row values, with each row's hops."""
        schema = payload.get('schema') or {}
        sent = (payload.get('predict') or {}).get('related_tables') or {}
        if self._hops is None:
            self._hops = self._hops_from_entity(payload)
        hops = self._hops
        related: dict[str, dict[str, Any]] = {}
        away: dict[str, list[int | None]] = {}
        for name, block in blocks.items():
            table = sent.get(name) or {}
            columns = list(table.get('columns') or [])
            rows = [table['rows'][index] for index in block['row_indices']]
            described = (schema.get('related_tables') or {}).get(name) or {}
            key = self._caller_key(described)
            hidden = {self.INSTANCE_ID, *self._synthetic_keys(described)}
            read_back: dict[str, Any] = {}
            if key is not None and key in columns:
                dtype = (
                    (described.get('columns') or {}).get(key, {}).get('dtype')
                )
                read_back['keys'] = [
                    as_declared(row[columns.index(key)], dtype) for row in rows
                ]
            read_back['rows'] = [
                {
                    column: value
                    for column, value in zip(columns, row, strict=True)
                    if column not in hidden
                }
                for row in rows
            ]
            read_back['columns'] = list(block.get('columns') or [])
            read_back['gradients'] = block.get('gradients') or []
            related[name] = read_back
            away[name] = [hops[name][index] for index in block['row_indices']]
        return related, away

    @classmethod
    def _key_columns(cls, described: Mapping[str, Any]) -> list[str]:
        r"""A related table's primary key, less the instance that scopes it."""
        key = described.get('primary_key') or []
        key = [key] if isinstance(key, str) else list(key)
        return [column for column in key if column != cls.INSTANCE_ID]

    @classmethod
    def _synthetic_keys(cls, described: Mapping[str, Any]) -> list[str]:
        r"""The key columns the engine made up for a table that had none."""
        return [
            column
            for column in cls._key_columns(described)
            if column.startswith(cls.SYNTHETIC_KEY)
        ]

    @classmethod
    def _caller_key(cls, described: Mapping[str, Any]) -> str | None:
        r"""The table's own primary key, or ``None`` if it has none of its own."""
        key = cls._key_columns(described)
        if len(key) != 1 or cls._synthetic_keys(described):
            return None
        return key[0]

    @classmethod
    def _hops_from_entity(
        cls, payload: Mapping[str, Any]
    ) -> dict[str, list[int | None]]:
        r"""Links from the entity per related row of this call; ``None`` if unreached."""
        schema = payload.get('schema') or {}
        predict = payload.get('predict') or {}
        instance = predict.get('instance_table') or {}
        tables = predict.get('related_tables') or {}
        hops: dict[str, list[int | None]] = {
            name: [None] * len(table.get('rows') or [])
            for name, table in tables.items()
        }
        linked: dict[tuple[str, int], list[tuple[str, int]]] = {}
        reached: list[tuple[str, int]] = []
        for link in schema.get('relationships') or []:
            target = link.get('target_table')
            if target not in tables:
                continue
            by_value: dict[Any, list[int]] = {}
            for position, value in enumerate(
                cls._joined_on(tables[target], link.get('target_columns') or [])
            ):
                if value is not None:
                    by_value.setdefault(value, []).append(position)
            source = link.get('source_table')
            columns = link.get('source_columns') or []
            if source is None:
                for value in cls._joined_on(instance, columns):
                    reached += [(target, at) for at in by_value.get(value, ())]
                continue
            if source not in tables:
                continue
            for position, value in enumerate(
                cls._joined_on(tables[source], columns)
            ):
                for at in by_value.get(value, ()):
                    linked.setdefault((source, position), []).append(
                        (target, at)
                    )
                    linked.setdefault((target, at), []).append(
                        (source, position)
                    )

        for name, position in reached:
            hops[name][position] = 0
        frontier = reached
        distance = 0
        while frontier:
            distance += 1
            following = []
            for row in frontier:
                for name, position in linked.get(row, ()):
                    if hops[name][position] is None:
                        hops[name][position] = distance
                        following.append((name, position))
            frontier = following
        return hops

    @classmethod
    def _joined_on(
        cls, table: Mapping[str, Any], columns: Sequence[str]
    ) -> list[Any]:
        r"""Per row, ``(instance, values)`` in ``columns``; ``None`` on a null."""
        names = list(table.get('columns') or [])
        if not columns or any(column not in names for column in columns):
            return []
        at = [names.index(column) for column in columns]
        scope = (
            names.index(cls.INSTANCE_ID) if cls.INSTANCE_ID in names else None
        )
        joined: list[Any] = []
        for row in table.get('rows') or []:
            values = tuple(row[index] for index in at)
            if any(value is None for value in values):
                joined.append(None)
                continue
            joined.append((None if scope is None else row[scope], values))
        return joined
