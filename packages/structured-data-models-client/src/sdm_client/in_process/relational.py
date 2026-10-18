# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``kumo-relational`` in process: each entity scored from its sampled graph.

The client gathers one neighbourhood per scored row and tags every row of it
with that row's ``instance_id``. Each relationship is joined on that tag as
well, so one row's neighbourhood never reaches another's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sdm_client.in_process.columns import (
    Context,
    Shape,
    declared,
    key_columns,
    labelled,
    stype_of,
)
from sdm_client.in_process.refusal import refusal

INSTANCE_TABLE = 'instance_table'
INSTANCE_ID = 'instance_id'

_CLASSIFIED = ('prediction', 'probabilities')


class Relational:
    r"""Reads a ``kumo-relational`` request into the model, as its NIM does."""

    model_id = 'kumo-relational'
    outputs = {
        'classification': _CLASSIFIED,
        'binary_classification': _CLASSIFIED,
        'multiclass_classification': _CLASSIFIED,
        'regression': ('prediction', 'quantiles'),
    }

    @staticmethod
    def load(device: Any) -> Any:
        from sdm.models import KumoRelational

        return KumoRelational(device=device)

    @staticmethod
    def accepts(model: Any) -> bool:
        from sdm.models import KumoRelational

        return isinstance(model, KumoRelational)

    def context(self, payload: Mapping[str, Any]) -> Context:
        from sdm import RelatedTables

        schema = payload['schema']
        task = payload['task']
        edges, links = _graph(schema.get('relationships') or ())
        if len(links) != 1:
            raise refusal(
                'INVALID_SCHEMA',
                f'a relational request needs exactly one relationship from '
                f'the instance table into a related table, got {len(links)}',
            )
        joins = _joins(edges, links)
        split = payload['context']
        spec = schema['instance_table']
        table = split['instance_table']
        declared(spec, table)
        if not table['rows']:
            raise refusal(
                'VALIDATION_FAILED',
                'Kumo Relational context must contain training data. '
                'context.instance_table.rows: At least one context instance '
                'row is required.',
            )
        instance = Shape.of(
            spec,
            _instance_stypes(
                spec,
                table,
                task,
                payload.get('inference'),
                joins[INSTANCE_TABLE],
            ),
            joins[INSTANCE_TABLE],
        )
        y, encoded, labels = labelled(table, task, task['kind'] != 'regression')
        shapes = {}
        tensors = {}
        for name, related in (split.get('related_tables') or {}).items():
            table_schema = schema['related_tables'][name]
            declared(table_schema, related)
            keys = joins.get(name, frozenset()) | key_columns(table_schema)
            shapes[name] = Shape.of(
                table_schema, _related_stypes(table_schema, related, keys), keys
            )
            tensors[name] = shapes[name].tensor(
                related, f'context table {name!r}'
            )
        return Context(
            x=instance.tensor(table, 'the context'),
            y=y,
            instance=instance,
            encoded=encoded,
            labels=labels,
            related=RelatedTables(
                tables=tensors, relationships=edges, task_links=links
            ),
            tables=shapes,
            relationships=tuple(edges),
            task_links=tuple(links),
        )

    @staticmethod
    def query(context: Context, payload: Mapping[str, Any]) -> tuple[Any, Any]:
        from sdm import RelatedTables

        split = payload['predict']
        related = split.get('related_tables') or {}
        tensors = {}
        for name, shape in context.tables.items():
            if name not in related:
                raise refusal(
                    'SESSION_PREDICT_SCHEMA_MISMATCH',
                    f'the rows to score omit related table {name!r}, which '
                    f'the context carries',
                )
            tensors[name] = shape.tensor(
                related[name], f'related table {name!r}'
            )
        return (
            context.instance.tensor(
                split['instance_table'], 'the rows to score'
            ),
            RelatedTables(
                tables=tensors,
                relationships=context.relationships,
                task_links=context.task_links,
            ),
        )


def _graph(relationships: Sequence[Mapping[str, Any]]) -> tuple[list, list]:
    r"""The joins between related tables, and the link from the scored rows.

    Every join also matches on ``instance_id``, the way the model's own
    sampler scopes each example to its own neighbourhood.
    """
    from sdm import Relationship, TaskLink

    edges, links = [], []
    for relationship in relationships:
        source = list(relationship['source_columns'])
        target = list(relationship['target_columns'])
        if INSTANCE_ID not in source:
            source.append(INSTANCE_ID)
            target.append(INSTANCE_ID)
        if relationship.get('source_table', INSTANCE_TABLE) == INSTANCE_TABLE:
            links.append(
                TaskLink(
                    task_columns=tuple(source),
                    table=relationship['target_table'],
                    table_columns=tuple(target),
                )
            )
        else:
            edges.append(
                Relationship(
                    left_table=relationship['source_table'],
                    left_columns=tuple(source),
                    right_table=relationship['target_table'],
                    right_columns=tuple(target),
                )
            )
    return edges, links


def _joins(edges: list, links: list) -> dict[str, frozenset[str]]:
    r"""The columns each table is joined on, by table."""
    joins: dict[str, set[str]] = {INSTANCE_TABLE: set()}
    for link in links:
        joins[INSTANCE_TABLE].update(link.task_columns)
        joins.setdefault(link.table, set()).update(link.table_columns)
    for edge in edges:
        joins.setdefault(edge.left_table, set()).update(edge.left_columns)
        joins.setdefault(edge.right_table, set()).update(edge.right_columns)
    return {table: frozenset(columns) for table, columns in joins.items()}


def _instance_stypes(
    spec: Mapping[str, Any],
    table: Mapping[str, Any],
    task: Mapping[str, Any],
    inference: Mapping[str, Any] | None,
    keys: frozenset[str],
) -> dict[str, str]:
    r"""The scored rows' columns the model reads, and as what.

    The model reads a timestamp in this table as the moment to predict from.
    The anchor is read only when the request asks for prediction time, and
    any other timestamp is refused rather than read as one.
    """
    target = task['target']['column_name']
    anchor = task.get('anchor_time_column')
    reads_time = bool((inference or {}).get('use_prediction_time'))
    primary = key_columns(spec)
    supplied = set(table['columns'])
    stypes = {}
    for name, column in spec['columns'].items():
        if name not in supplied or name == target:
            continue
        if name in keys:
            stypes[name] = 'id'
        elif name == anchor:
            if reads_time:
                stypes[name] = 'datetime'
        elif name not in primary:
            stype = stype_of(name, column)
            if stype == 'datetime':
                raise refusal(
                    'UNSUPPORTED_FEATURE',
                    f'instance table carries timestamp column(s) {[name]} that '
                    f'are not the declared anchor; this model reads any task '
                    f'timestamp as a seed time',
                )
            stypes[name] = stype
    return stypes


def _related_stypes(
    schema: Mapping[str, Any], table: Mapping[str, Any], keys: frozenset[str]
) -> dict[str, str]:
    r"""A related table's columns, and what the model reads each as.

    A key is an identifier whatever its dtype says, because it decides which
    rows meet.
    """
    supplied = set(table['columns'])
    return {
        name: 'id' if name in keys else stype_of(name, column)
        for name, column in schema['columns'].items()
        if name in supplied
    }
