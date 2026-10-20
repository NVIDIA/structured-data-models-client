# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""``kumo-relational`` in process: each entity scored from its sampled graph.

The client gathers one neighbourhood per scored row and tags every row of it
with that row's ``instance_id``. Each relationship is joined on that tag as
well, so one row's neighbourhood never reaches another's.

A request the client gathered is always one the model can read. One built by
hand and passed to ``Request.materialized`` may not be, and is refused where
the model would otherwise answer a question other than the one it states.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from sdm_client.errors import StructuredDataError
from sdm_client.in_process.columns import (
    Context,
    Shape,
    check_supplied,
    key_columns,
    labelled,
    stype_of,
)
from sdm_client.in_process.tables import values

INSTANCE_TABLE = 'instance_table'
INSTANCE_ID = 'instance_id'

_CLASSIFIED = ('prediction', 'probabilities')
_INFERENCE_OPTIONS = frozenset(
    {'run_mode', 'use_prediction_time', 'inference_config'}
)
_NAMED_TASK_OPTIONS = (
    'entity_table_names',
    'anchor_time_column',
    'top_k',
    'step_size',
    'num_forecasts',
)
_TASK_OPTIONS = frozenset({'entity_table_names', 'anchor_time_column'})
_CLASSIFICATIONS = frozenset(
    {'classification', 'binary_classification', 'multiclass_classification'}
)
_NUMERIC_TARGETS = frozenset({'bool', 'int32', 'int64', 'float32', 'float64'})


class Relational:
    r"""Reads a ``kumo-relational`` request into the model."""

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
        if not schema.get('related_tables'):
            raise StructuredDataError(
                f'{self.model_id} scores an entity from its related tables, '
                f'and this request declares none',
                code='INVALID_SCHEMA',
            )
        if (task['target'].get('table_name') or INSTANCE_TABLE) != (
            INSTANCE_TABLE
        ):
            raise StructuredDataError(
                f'{self.model_id} predicts a column of the instance table, '
                f'and this target is on {task["target"]["table_name"]!r}',
                code='UNSUPPORTED_FEATURE',
            )
        self.check_inference(payload.get('inference') or {}, task)
        _check_declared(schema)
        edges, links = _graph(schema.get('relationships') or ())
        if len(links) != 1:
            raise StructuredDataError(
                f'the instance table must link to exactly one related table, '
                f'the entity scored, and it links to {len(links)}',
                code='INVALID_SCHEMA',
            )
        spec = schema['instance_table']
        _check_task(task, spec, links[0].table)
        _check_target(task)
        split = payload['context']
        table = split['instance_table']
        check_supplied(table, spec, 'the context')
        joins = _joins(edges, links)
        shapes = _shapes(split, schema, joins, edges, links)
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
        related = split.get('related_tables') or {}
        tensors = {
            name: shape.tensor(related[name], f'context table {name!r}')
            for name, shape in shapes.items()
        }
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
    def check_inference(
        options: Mapping[str, Any],
        task: Mapping[str, Any],
        settled: bool = False,
    ) -> None:
        r"""Refuse an inference option this model does not act on as asked.

        A session's context is fitted without prediction time, since pinning
        one carries no inference options, so a prediction cannot ask for it
        after.
        """
        unknown = sorted(set(options) - _INFERENCE_OPTIONS)
        if unknown:
            raise StructuredDataError(
                f'unsupported inference options: {unknown}',
                code='INVALID_REQUEST',
            )
        if options.get('inference_config'):
            raise StructuredDataError(
                'inference_config is reserved for API compatibility and must '
                'be omitted or empty',
                code='INVALID_REQUEST',
            )
        if not options.get('use_prediction_time'):
            return
        if not task.get('anchor_time_column'):
            raise StructuredDataError(
                'use_prediction_time needs task.anchor_time_column, which '
                'this request does not name',
                code='TASK_SCHEMA_MISMATCH',
            )
        if settled:
            raise StructuredDataError(
                'use_prediction_time is settled when a session is pinned and '
                'cannot be changed by a prediction',
                code='INVALID_REQUEST',
            )

    @staticmethod
    def query(
        context: Context,
        payload: Mapping[str, Any],
        schema: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        r"""The rows to score, read against the context they are scored
        against, and refused where they do not fit it."""
        from sdm import RelatedTables

        split = payload['predict']
        edges, links = list(context.relationships), list(context.task_links)
        _shapes(
            split, schema, _joins(edges, links), edges, links, context.tables
        )
        related = split.get('related_tables') or {}
        tensors = {
            name: shape.tensor(related[name], f'related table {name!r}')
            for name, shape in context.tables.items()
        }
        return (
            context.instance.tensor(
                split['instance_table'], 'predict instance table'
            ),
            RelatedTables(
                tables=tensors,
                relationships=context.relationships,
                task_links=context.task_links,
            ),
        )

    @staticmethod
    def row_ids(
        table: Mapping[str, Any], spec: Mapping[str, Any]
    ) -> list[str | None]:
        r"""Each row's ``instance_id``, which the client matches its answer
        to before putting the entity's own key on it."""
        return [str(value) for value in values(table, INSTANCE_ID)]


def _check_task(
    task: Mapping[str, Any], spec: Mapping[str, Any], linked: str
) -> None:
    r"""Refuse a task the model would answer as a different question.

    The model scores the table the instance table links to, from the anchor
    the instance table declares, and acts on no other task option.
    """
    options = {
        name: task[name]
        for name in _NAMED_TASK_OPTIONS
        if task.get(name) is not None
    }
    options.update(
        {
            name: value
            for name, value in task.items()
            if name not in {'kind', 'target', *_NAMED_TASK_OPTIONS}
        }
    )
    unsupported = sorted(set(options) - _TASK_OPTIONS)
    if unsupported:
        raise StructuredDataError(
            f'the model does not act on the task options {unsupported}',
            code='UNSUPPORTED_FEATURE',
        )
    requested = list(options.get('entity_table_names') or ())
    if requested != [linked]:
        raise StructuredDataError(
            f'the task names entity tables {requested}, and the model scores '
            f'{linked!r}, the one table the instance table links to',
            code='TASK_SCHEMA_MISMATCH',
        )
    anchor = options.get('anchor_time_column')
    if anchor is None:
        return
    column = spec['columns'].get(anchor)
    if (
        column is None
        or anchor in key_columns(spec)
        or stype_of(anchor, column) != 'datetime'
    ):
        raise StructuredDataError(
            f'the task names {anchor!r} as its anchor time, which is not a '
            f'timestamp the instance table declares',
            code='TASK_SCHEMA_MISMATCH',
        )


def _check_target(task: Mapping[str, Any]) -> None:
    r"""Refuse a target the model cannot read: a number, or for a
    classification a string too."""
    dtype = task['target'].get('dtype')
    readable = _NUMERIC_TARGETS
    if task['kind'] in _CLASSIFICATIONS:
        readable = readable | {'string'}
    if dtype not in readable:
        raise StructuredDataError(
            f'the model cannot read a {task["kind"]!r} target declared as '
            f'{dtype!r}',
            code='TASK_SCHEMA_MISMATCH',
        )


def _check_declared(schema: Mapping[str, Any]) -> None:
    r"""Refuse a relationship joining on a column its table does not declare.

    A join on an undeclared column reads whatever the table holds under that
    name, if anything, so the neighbourhood it builds is not the one stated.
    """
    tables = {
        INSTANCE_TABLE: schema['instance_table'],
        **(schema.get('related_tables') or {}),
    }
    for relationship in schema.get('relationships') or ():
        for side in ('source', 'target'):
            name = relationship.get(f'{side}_table', INSTANCE_TABLE)
            declared = (tables.get(name) or {}).get('columns') or {}
            for column in relationship[f'{side}_columns']:
                if column not in declared:
                    raise StructuredDataError(
                        f'a relationship joins on {column!r}, which '
                        f'{name!r} does not declare',
                        code='INVALID_SCHEMA',
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


def _shapes(
    split: Mapping[str, Any],
    schema: Mapping[str, Any],
    joins: Mapping[str, frozenset[str]],
    edges: list,
    links: list,
    compiled: Mapping[str, Shape] | None = None,
) -> dict[str, Shape]:
    r"""Each related table's shape, and what the model reads it as.

    With ``compiled``, the shapes the context was fitted in, which the rows
    scored against it must match table for table.
    """
    tables = split.get('related_tables') or {}
    shapes = {}
    for name, table in tables.items():
        if compiled is not None:
            if name not in compiled:
                raise StructuredDataError(
                    f'the rows to score carry related table {name!r}, which '
                    f'the context does not',
                    code='INVALID_SCHEMA',
                )
            shapes[name] = compiled[name]
            continue
        table_schema = schema['related_tables'].get(name)
        if table_schema is None:
            raise StructuredDataError(
                f'related table {name!r} carries rows and the schema does '
                f'not declare it',
                code='INVALID_SCHEMA',
            )
        check_supplied(table, table_schema, f'context table {name!r}')
        keys = joins.get(name, frozenset()) | key_columns(table_schema)
        shapes[name] = Shape.of(
            table_schema, _related_stypes(table_schema, table, keys), keys
        )
    referenced = {link.table for link in links}
    for edge in edges:
        referenced.update({edge.left_table, edge.right_table})
    missing = sorted(name for name in referenced if name not in shapes)
    if missing:
        raise StructuredDataError(
            f'relationships reference table(s) {missing} that carry no rows',
            code='INVALID_SCHEMA',
        )
    if compiled is not None:
        omitted = sorted(set(compiled) - set(tables))
        if omitted:
            raise StructuredDataError(
                f'the rows to score leave out related table(s) {omitted}, '
                f'which the context carries',
                code='INVALID_SCHEMA',
            )
    return shapes


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
                raise StructuredDataError(
                    f'instance table carries timestamp column(s) {[name]} that '
                    f'are not the declared anchor; this model reads any task '
                    f'timestamp as a seed time',
                    code='UNSUPPORTED_FEATURE',
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
