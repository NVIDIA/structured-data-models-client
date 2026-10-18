# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The request contract's own limits, checked as a NIM checks them.

Before anything reads a request, a NIM validates it against the contract:
every name and label it carries is at most 1,024 characters, and a table
names each column once. A request outside those is refused whole, with each
offending field named.
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
