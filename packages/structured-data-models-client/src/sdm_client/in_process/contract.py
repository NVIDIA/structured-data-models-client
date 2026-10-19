# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The request contract's own rules, checked as a NIM checks them.

Before anything reads a request, a NIM validates it against the contract:
every name and label it carries is at most 1,024 characters, a table names
each column once, no object carries a key its contract does not name, and
each table supplies exactly the columns its schema declares. A request
outside those is refused, with the field named.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from sdm_client.errors import format_invalid_params
from sdm_client.in_process.refusal import refusal

_TEXT = 1024
_TOO_LONG = f'String should have at most {_TEXT} characters'


PREDICTION = (
    'model',
    'task',
    'schema',
    'context',
    'predict',
    'output',
    'inference',
    'metadata',
)
SESSION = ('model', 'task', 'schema', 'context', 'metadata')
SESSION_PREDICTION = ('predict', 'output', 'inference', 'metadata')

_EXTRA = 'Extra inputs are not permitted'


def check_contract(
    payload: Mapping[str, Any], fields: tuple[str, ...] = PREDICTION
) -> None:
    r"""Refuse a request the contract cannot carry.

    ``fields`` are the ones the endpoint takes. Problems are listed in the
    order a NIM lists them: field by field, each object's unknown keys last.
    """
    problems = [
        {'name': name, 'reason': reason}
        for name, reason in _problems(payload, fields)
    ]
    if problems:
        raise refusal(
            'VALIDATION_FAILED',
            'Request validation failed.' + format_invalid_params(problems),
        )


def _problems(
    payload: Mapping[str, Any], fields: tuple[str, ...]
) -> Iterator[tuple[str, str]]:
    readers = {
        'task': _task,
        'schema': _schema,
        'context': _split,
        'predict': _split,
        'metadata': _metadata,
    }
    for field in fields:
        value = payload.get(field)
        if field in readers and isinstance(value, Mapping):
            yield from readers[field](field, value)
    yield from _extra('', payload, fields)


def _task(where: str, task: Mapping[str, Any]) -> Iterator[tuple[str, str]]:
    yield from _texts(where, task, ('kind',))
    target = task.get('target')
    if isinstance(target, Mapping):
        yield from _texts(
            f'{where}.target', target, ('column_name', 'table_name', 'dtype')
        )
        yield from _listed(f'{where}.target.classes', target.get('classes'))
        yield from _texts(f'{where}.target', target, ('positive_class',))
    yield from _listed(
        f'{where}.entity_table_names', task.get('entity_table_names')
    )
    yield from _texts(where, task, ('anchor_time_column',))


def _schema(where: str, schema: Mapping[str, Any]) -> Iterator[tuple[str, str]]:
    yield from _table_schema(
        f'{where}.instance_table', schema.get('instance_table')
    )
    for name, table in _entries(schema.get('related_tables')):
        yield from _table_schema(f'{where}.related_tables.{name}', table)
    for index, relationship in enumerate(schema.get('relationships') or ()):
        if not isinstance(relationship, Mapping):
            continue
        at = f'{where}.relationships[{index}]'
        yield from _texts(at, relationship, ('source_table',))
        yield from _listed(
            f'{at}.source_columns', relationship.get('source_columns')
        )
        yield from _texts(at, relationship, ('target_table',))
        yield from _listed(
            f'{at}.target_columns', relationship.get('target_columns')
        )
        yield from _extra(
            at,
            relationship,
            (
                'source_table',
                'source_columns',
                'target_table',
                'target_columns',
            ),
        )
    yield from _extra(
        where, schema, ('instance_table', 'related_tables', 'relationships')
    )


def _split(where: str, split: Mapping[str, Any]) -> Iterator[tuple[str, str]]:
    yield from _columns(f'{where}.instance_table', split.get('instance_table'))
    for name, table in _entries(split.get('related_tables')):
        yield from _columns(f'{where}.related_tables.{name}', table)
    yield from _extra(
        where, split, ('instance_table', 'related_tables', 'metadata')
    )


def _metadata(
    where: str, metadata: Mapping[str, Any]
) -> Iterator[tuple[str, str]]:
    yield from _texts(
        where, metadata, ('request_id', 'trace_id', 'source_query')
    )


def _extra(
    where: str, values: Mapping[str, Any], fields: tuple[str, ...]
) -> Iterator[tuple[str, str]]:
    for name in values:
        if name not in fields:
            yield (f'{where}.{name}' if where else str(name)), _EXTRA


def _entries(values: Any) -> Iterator[tuple[str, Any]]:
    yield from (values.items() if isinstance(values, Mapping) else ())


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
    for name, column in _entries(table.get('columns')):
        if isinstance(column, Mapping):
            yield from _texts(
                f'{where}.columns.{name}', column, ('dtype', 'stype')
            )
    yield from _extra(where, table, ('columns', 'primary_key'))


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
    if table.get('format') == 'arrays':
        yield from _extra(where, table, ('format', 'columns', 'rows'))


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
