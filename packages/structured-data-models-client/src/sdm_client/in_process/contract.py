# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The request contract's own rules, checked as a NIM checks them.

Before anything reads a request, a NIM validates it against the contract:
every name and label it carries is at most 1,024 characters, a table names
each column once, and each table supplies exactly the columns its schema
declares. A request outside those is refused, with the field named.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from sdm_client.errors import format_invalid_params
from sdm_client.in_process.refusal import refusal

_TEXT = 1024
_TOO_LONG = f'String should have at most {_TEXT} characters'


def check_contract(payload: Mapping[str, Any]) -> None:
    r"""Refuse a request the contract cannot carry."""
    problems = [
        {'name': name, 'reason': reason} for name, reason in _problems(payload)
    ]
    if problems:
        raise refusal(
            'VALIDATION_FAILED',
            'Request validation failed.' + format_invalid_params(problems),
        )


def _problems(payload: Mapping[str, Any]) -> Iterator[tuple[str, str]]:
    task = payload.get('task') or {}
    target = task.get('target') or {}
    yield from _texts('task', task, ('kind', 'anchor_time_column'))
    yield from _texts(
        'task.target',
        target,
        ('column_name', 'table_name', 'dtype', 'positive_class'),
    )
    yield from _listed('task.target.classes', target.get('classes'))
    yield from _listed(
        'task.entity_table_names', task.get('entity_table_names')
    )
    schema = payload.get('schema') or {}
    yield from _table_schema(
        'schema.instance_table', schema.get('instance_table')
    )
    for name, table in (schema.get('related_tables') or {}).items():
        yield from _table_schema(f'schema.related_tables.{name}', table)
    for index, relationship in enumerate(schema.get('relationships') or ()):
        where = f'schema.relationships[{index}]'
        yield from _texts(where, relationship, ('source_table', 'target_table'))
        yield from _listed(
            f'{where}.source_columns', relationship.get('source_columns')
        )
        yield from _listed(
            f'{where}.target_columns', relationship.get('target_columns')
        )
    for split in ('context', 'predict'):
        tables = payload.get(split) or {}
        yield from _columns(
            f'{split}.instance_table', tables.get('instance_table')
        )
        for name, table in (tables.get('related_tables') or {}).items():
            yield from _columns(f'{split}.related_tables.{name}', table)
    yield from _texts(
        'metadata',
        payload.get('metadata') or {},
        ('request_id', 'trace_id', 'source_query'),
    )


def _texts(
    where: str, values: Mapping[str, Any], names: tuple[str, ...]
) -> Iterator[tuple[str, str]]:
    for name in names:
        value = values.get(name)
        if isinstance(value, str) and len(value) > _TEXT:
            yield f'{where}.{name}', _TOO_LONG


def _listed(where: str, values: Any) -> Iterator[tuple[str, str]]:
    for index, value in enumerate(values or ()):
        if isinstance(value, str) and len(value) > _TEXT:
            yield f'{where}[{index}]', _TOO_LONG


def _table_schema(where: str, table: Any) -> Iterator[tuple[str, str]]:
    if not isinstance(table, Mapping):
        return
    for name, column in (table.get('columns') or {}).items():
        if isinstance(column, Mapping):
            yield from _texts(
                f'{where}.columns.{name}', column, ('dtype', 'stype')
            )


def _columns(where: str, table: Any) -> Iterator[tuple[str, str]]:
    if not isinstance(table, Mapping):
        return
    seen: set[str] = set()
    for index, column in enumerate(table.get('columns') or ()):
        if isinstance(column, str) and len(column) > _TEXT:
            yield f'{where}.columns[{index}]', _TOO_LONG
        if column in seen:
            yield (
                f'{where}.columns[{index}]',
                f"duplicate column name '{column}'",
            )
        seen.add(column)


def check_columns(
    payload: Mapping[str, Any],
    schema: Mapping[str, Any],
    splits: tuple[str, ...],
) -> None:
    r"""Refuse a table that does not supply exactly its declared columns.

    The rows to score may leave out the target, which they are asked about.
    """
    target = (payload.get('task') or {}).get('target') or {}
    column = target.get('column_name')
    target_table = target.get('table_name') or 'instance_table'
    related_schemas = schema.get('related_tables') or {}
    for split in splits:
        tables = payload.get(split) or {}
        unasked = (
            frozenset({column})
            if split == 'predict' and column
            else frozenset()
        )
        _check_table(
            f'{split}.instance_table',
            tables.get('instance_table'),
            schema.get('instance_table'),
            unasked if target_table == 'instance_table' else frozenset(),
        )
        for name, table in (tables.get('related_tables') or {}).items():
            if name not in related_schemas:
                raise refusal(
                    'INVALID_SCHEMA',
                    f"related table '{name}' is not declared in schema.",
                )
            _check_table(
                f'{split}.related_tables.{name}',
                table,
                related_schemas[name],
                unasked if target_table == name else frozenset(),
            )


def _check_table(
    where: str, table: Any, schema: Any, unasked: frozenset[str]
) -> None:
    if not isinstance(table, Mapping) or not isinstance(schema, Mapping):
        return
    declared = set(schema.get('columns') or {})
    present = set(table.get('columns') or ())
    missing = sorted(declared - present - unasked)
    extra = sorted(present - declared)
    if missing or extra:
        reason = f'missing={missing}, extra={extra}.'
        raise refusal(
            'INVALID_SCHEMA',
            'Table columns do not match the declared inputs; '
            + reason
            + format_invalid_params(
                [{'name': f'{where}.columns', 'reason': reason}]
            ),
        )
