# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Reading the explanation each prediction record carries.

Both models answer an explanation in one format, ``sdm_gradient_v1``, one per
record, about that record's own row. Its blocks name rows by their position in
the tables of the call that produced it, and a request sent in several calls
starts those positions at 0 in each. So each record is read here, beside the
payload of its own call, into an entry that needs no payload to be read: the
row it is about becomes its key, and a related row its key and its values.

The format is the endpoint's: ``gradients`` is ``[output][row][column]``, and
``columns`` are the model's preprocessed numerical columns, passed through.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sdm_client.result import GradientExplanations
from sdm_client.wire.dtypes import as_declared


class GradientDecoder:
    r"""Reads one call's explanations, beside that call's own payload.

    Constructed once per call: with the payload for ``kumo-relational``, whose
    explanations name rows of the related tables it sent, and with none for
    ``kumo-tabular``, whose explanations are about the record's own row only.
    """

    #: The engine's own transport columns on a related table: the instance a
    #: row was sampled for, and the key it numbers a keyless table's rows by
    #: (`rfm/inference/payload.py`). Neither is a value the caller sent, so a
    #: related row is given back without them.
    INSTANCE_ID = 'instance_id'
    SYNTHETIC_KEY = '__node_id'

    def __init__(self, payload: Mapping[str, Any] | None = None) -> None:
        self._payload = payload
        # How far each related row is from the entity, worked out once per
        # call, and only if some record is explained.
        self._hops: dict[str, list[int | None]] | None = None

    def decode(self, items: Sequence[Any]) -> GradientExplanations:
        r"""Each answered row's explanation, under the key naming that row.

        ``items`` have their ``id`` restored already, so the key is the one the
        frame shows: the ``id``, or the ``row_index`` when the rows carry none.
        A payload is given for a model that reads related tables. Then an entry
        has ``related`` and no ``instance``: that model's ``instance_table`` is
        the question row its engine sends, not the caller's, and the entity's
        own table is among the related ones. Without it an entry has
        ``instance`` and no ``related`` at all, rather than an empty one.
        """
        explained = [item for item in items if item.explanation]
        if not explained:
            return GradientExplanations()
        by_id = any(item.id is not None for item in explained)
        entries: dict[Any, dict[str, Any]] = {}
        hops: dict[Any, dict[str, list[int | None]]] = {}
        for item in explained:
            key = item.id if by_id else item.row_index
            details = item.explanation.get('details') or {}
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
                entry['related'], hops[key] = self._read_related(
                    self._payload, details.get('related_tables') or {}
                )
            entries[key] = entry
        return GradientExplanations(
            entries,
            keyed_by=('id',) if by_id else ('row_index',),
            hops=hops,
        )

    def _read_related(
        self, payload: Mapping[str, Any], blocks: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, list[int | None]]]:
        r"""Read a record's related blocks as rows of this call, not positions.

        The endpoint names each related row by its position in the table as
        this call sent it, which the caller never saw and which a split request
        restarts at 0 in every call. Each is given back as its primary key,
        when the table has one of the caller's, and always as its values as
        sent.

        How far each row is from the entity is worked out once per call, and
        only if some record is explained.
        """
        schema = payload.get('schema') or {}
        sent = (payload.get('predict') or {}).get('related_tables') or {}
        if self._hops is None:
            self._hops = self._hops_from_entity(payload)
        hops = self._hops
        related: dict[str, Any] = {}
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
        r"""How many links each related row of this call is from its entity.

        Read from the relationships and key values sent, the same ones the
        model joins on. A relationship with no ``source_table`` starts at the
        instance table, and the rows it reaches are the entity's own: 0. Every
        other one is followed either way, and a row is as far as its nearest
        path. Joins stay within one instance, as the engine scopes them, so two
        entities sharing a product never meet through it.

        Positions are this call's, one list per related table of its predict
        half. ``None`` for a row no relationship reaches from an entity.
        """
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
        r"""What each row of ``table`` joins on through ``columns``.

        The instance a row belongs to, and its values in ``columns``; ``None``
        for a row holding a null there, which joins nothing. Empty when the
        table does not carry every one of ``columns``.
        """
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
