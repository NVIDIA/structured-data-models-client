# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Relational endpoint diagnostics and request metadata helpers."""

from __future__ import annotations

import dataclasses
import re
import uuid
from collections.abc import Iterator, Mapping
from typing import Any

from sdm_client.base import ModelSpec
from sdm_client.errors import (
    NimRequestError,
    StructuredDataError,
)
from sdm_client.models import (
    KUMO_RELATIONAL,
)

# What to ask for when the caller named nothing, by task kind. Asking for
# everything would make a regression request carry probabilities it cannot
# have, which the endpoint rejects.
_DEFAULT_OUTPUTS = {
    'classification': ('prediction', 'probabilities'),
    'binary_classification': ('prediction', 'probabilities'),
    'multiclass_classification': ('prediction', 'probabilities'),
    'regression': ('prediction',),
    'forecasting': ('prediction',),
    'temporal_link_prediction': ('rankings',),
}

_SESSION_PINNED = ('model', 'task', 'schema', 'context')
#: Which halves of a request a call puts on the wire. Every judgement made
#: before sending asks only about a half that travels.

_SESSION_SCORING = ('predict', 'output', 'inference', 'metadata')


def _request_id() -> str:
    return f'sdm-{uuid.uuid4().hex[:12]}'


def _keys_the_caller_set(requested: Any) -> set[str]:
    r"""The inference-config keys that were asked for rather than defaulted.

    The engine takes ``InferenceConfig | dict | None`` and this client
    publishes nothing narrower, so a caller may hand over the dataclass. It is
    not iterable, and reading it as a set of keys destroyed a request whose
    payload was already gathered and correct.

    A dataclass carries every field whether or not it was set, so only the
    ones differing from their default are a request for anything. Counting the
    rest would refuse a bare ``RegressionInferenceConfig()``, which asks for
    exactly what passing no config at all asks for.
    """
    if isinstance(requested, Mapping):
        return set(requested)
    if dataclasses.is_dataclass(requested) and not isinstance(requested, type):
        return {
            field.name
            for field in dataclasses.fields(requested)
            if getattr(requested, field.name) != _default_of(field)
        }
    return set()


def _default_of(field: dataclasses.Field) -> Any:
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return field.default


class RelationalAdapter:
    r"""Shapes and reads ``kumo-relational`` requests."""

    spec: ModelSpec = KUMO_RELATIONAL

    def explain_error(
        self, error: Exception, payload: dict[str, Any]
    ) -> Exception:
        r"""Name the columns behind a semantic-type refusal.

        ``semantic type 'text' is unsupported`` is true and unactionable:
        ``Schema.from_data`` marks any string column with more than 64 distinct
        values as ``text``, so a real dataset acquires them in bulk and the
        endpoint names none of them. The payload it just refused does.

        The deployments word this two ways and only one of them was matched, so
        on the one that says ``categorical cardinality 13812 exceeds limit
        10000`` the hook never ran and the caller got a number and a limit
        against a graph of twenty-two string columns. Both forms are read now,
        and the second names the columns by counting the payload rather than by
        reading a declared type, because a cardinality complaint is about the
        values.

        A different complaint needs a different remedy. Where the columns are
        already categorical, telling their owner to make them categorical
        would be advice that changes nothing.
        """
        columns, complaint = _columns_the_endpoint_means(error, payload)
        if not columns:
            return error
        if complaint == 'cardinality':
            message = (
                f'{getattr(error, "message", error)}. In this request that is '
                f'{", ".join(columns)}. A column with that many distinct '
                f'values carries no signal the model can use; leave it out '
                f'with materialize(..., exclude_cols_dict=...), or replace it '
                f'with something coarser.'
            )
            return _reworded(error, message)
        # The payload does not carry the entity table's real name,
        # `instance_table` is what the wire calls it. Thus, an example naming
        # it would render a line the caller cannot run. Prefer a related
        # table, whose name did survive, and fall back to naming no table at
        # all rather than a made-up one.
        named = next(
            (c for c in columns if not c.startswith('instance_table.')), None
        )
        if named is not None:
            table, _, column = named.partition('.')
            suggestion = f"schema[{table!r}][{column!r}].stype = 'categorical'"
        else:
            _, _, column = columns[0].partition('.')
            suggestion = (
                f"schema[<the entity table>][{column!r}].stype = 'categorical'"
            )
        message = (
            f'{getattr(error, "message", error)}. In this request that is '
            f'{", ".join(columns)}. Set a supported type on each, e.g. '
            f'{suggestion}'
        )
        return _reworded(error, message)

    def close(self) -> None:
        r"""Nothing to release: gathering happens in the query, which owns
        whatever it opened, and sending goes through the client's transport."""


_UNSUPPORTED_STYPE = re.compile(r"semantic type '([^']+)' is unsupported")

#: The other wording, from the deployment that counts rather than names.
_TOO_MANY_LABELS = re.compile(
    r'cardinality (\d+) exceeds limit (\d+)', re.IGNORECASE
)


def _columns_the_endpoint_means(
    error: Exception, payload: Mapping[str, Any]
) -> tuple[list[str], str]:
    r"""The columns a refusal is about, and which complaint it was."""
    said = str(error)
    named = _UNSUPPORTED_STYPE.search(said)
    if named is not None:
        return _columns_with_stype(payload, named.group(1)), 'stype'
    counted = _TOO_MANY_LABELS.search(said)
    if counted is None:
        return [], ''
    return _columns_over(payload, int(counted.group(2))), 'cardinality'


def _reworded(error: Exception, message: str) -> Exception:
    r"""The same refusal, saying more."""
    if isinstance(error, NimRequestError):
        return NimRequestError(
            error.status_code,
            code=error.code,
            message=message,
            details=error.details,
        )
    rebuilt = StructuredDataError(
        message,
        code=getattr(error, 'code', None),
        details=getattr(error, 'details', None),
    )
    status = getattr(error, 'status_code', None)
    if status is not None:
        # `fit()` reads this to decide whether an endpoint simply does not
        # serve sessions; losing it turns a clean fallback into a failure.
        rebuilt.status_code = status
    return rebuilt


def _columns_over(payload: Mapping[str, Any], limit: int) -> list[str]:
    r"""Every ``table.column`` holding more distinct values than ``limit``.

    The endpoint reports one number and the limit it broke, which says nothing
    about where to look. The rows it just refused do.

    Counted from the refused payload by the diagnostic helper below. What the guard counts
    is not every cell: it is string cells only, across the predict half as well
    as the context, reduced to the larger of the two. A text cell travels
    as a list of tokens, which it counts individually. Counting it a second way
    here would name columns the endpoint was not talking about, a numeric key
    among them.
    """
    return [
        f'{table}.{column} ({count:,} values)'
        for table, column, count in _high_cardinality_columns(
            dict(payload), limit
        )
    ]


def _columns_with_stype(payload: Mapping[str, Any], stype: str) -> list[str]:
    r"""Every ``table.column`` in the payload declared as ``stype``.

    The endpoint refuses an unsupported semantic type without saying which
    column carries it, and a real dataset acquires them in bulk. Northwind
    produces fifteen. The payload that was just refused knows.

    A relational schema is ``instance_table`` plus ``related_tables``, so both
    shapes are walked; the instance table's own name is not in the payload, and
    ``instance_table`` is what the wire calls it.
    """
    found = []
    schema = payload.get('schema') or {}
    tables = {'instance_table': schema.get('instance_table') or {}}
    tables.update(schema.get('related_tables') or {})
    for table_name, table in tables.items():
        if not isinstance(table, Mapping):
            continue
        for column_name, spec in (table.get('columns') or {}).items():
            if isinstance(spec, Mapping) and spec.get('stype') == stype:
                found.append(f'{table_name}.{column_name}')
    return sorted(found)


def _payload_tables_with_names(
    payload: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    r"""Yield ``(table_name, table)`` for every serialized table in a request.

    Related tables are keyed by the name the caller gave them; the entity rows
    travel as ``instance_table``, which is resolved back to the entity table so
    every name reported is one the caller can look up on their graph.
    """
    task = payload.get('task')
    entity_names = (
        task.get('entity_table_names') if isinstance(task, dict) else None
    )
    instance_name = (
        entity_names[0]
        if isinstance(entity_names, list) and entity_names
        else 'instance_table'
    )
    for section_name in ('context', 'predict'):
        section = payload.get(section_name)
        if not isinstance(section, dict):
            continue
        instance = section.get('instance_table')
        if isinstance(instance, dict):
            yield str(instance_name), instance
        related = section.get('related_tables')
        if isinstance(related, dict):
            for table_name, table in related.items():
                if isinstance(table, dict):
                    yield table_name, table


def _string_values(cell: Any) -> tuple[str, ...]:
    r"""The string values a serialized cell contributes to a column's
    cardinality.

    A text or multicategorical cell travels as a list of tokens rather than a
    bare string, and the NIM counts those tokens too, so both shapes have to be
    unwrapped to arrive at the count it reports.
    """
    if isinstance(cell, str):
        return (cell,)
    if isinstance(cell, (list, tuple)):
        return tuple(value for value in cell if isinstance(value, str))
    return ()


def _high_cardinality_columns(
    payload: dict[str, Any],
    limit: int,
) -> list[tuple[str, str, int]]:
    r"""Return ``(table, column, cardinality)`` for string columns holding more
    than ``limit`` distinct values, most offending first.

    Mirrors the NIM's own cardinality guard, which counts string cells only, so
    a rejection the server reports as a bare number can be traced back to the
    columns that caused it. The same table appears in both the context and the
    predict section, so counts are reduced to the larger of the two.
    """
    counts: dict[tuple[str, str], int] = {}
    for table_name, table in _payload_tables_with_names(payload):
        columns = table.get('columns')
        rows = table.get('rows')
        if not isinstance(columns, list) or not isinstance(rows, list):
            continue
        for index, column in enumerate(columns):
            distinct: set[str] = set()
            for row in rows:
                if not isinstance(row, list) or index >= len(row):
                    continue
                distinct.update(_string_values(row[index]))
            if len(distinct) > limit:
                key = (table_name, str(column))
                counts[key] = max(counts.get(key, 0), len(distinct))
    return sorted(
        ((table, column, count) for (table, column), count in counts.items()),
        key=lambda entry: entry[2],
        reverse=True,
    )
