# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""What both models share: a column's meaning, a table's shape, the labels."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from sdm_client.errors import StructuredDataError
from sdm_client.in_process.tables import to_arrow, values

_DECLARED = {
    'numerical': 'numerical',
    'categorical': 'categorical',
    'timestamp': 'datetime',
    'datetime': 'datetime',
    'id': 'id',
}
_FROM_DTYPE = {
    'int32': 'numerical',
    'int64': 'numerical',
    'float32': 'numerical',
    'float64': 'numerical',
    'bool': 'categorical',
    'string': 'categorical',
    'timestamp[us]': 'datetime',
}


def stype_of(name: str, column: Mapping[str, Any]) -> str:
    r"""What the model reads a column as, refusing one it cannot read.

    The model carries no processor for text, multicategorical or sequence
    columns, so a request carrying one is refused rather than answered
    without it.
    """
    declared = column.get('stype')
    if declared is not None:
        stype = _DECLARED.get(str(declared).lower())
        if stype is None:
            raise StructuredDataError(
                f'column {name!r} declares semantic type {declared!r}, which '
                f'SDM does not read',
                code='UNSUPPORTED_DTYPE',
            )
        return stype
    dtype = column.get('dtype')
    stype = _FROM_DTYPE.get(str(dtype))
    if stype is None:
        raise StructuredDataError(
            f'column {name!r} has no unambiguous SDM semantic type for dtype '
            f'{dtype!r}',
            code='UNSUPPORTED_DTYPE',
        )
    return stype


def key_columns(schema: Mapping[str, Any]) -> frozenset[str]:
    r"""The columns a table's primary key names."""
    key = schema.get('primary_key')
    if isinstance(key, str):
        return frozenset({key})
    return frozenset(key or ())


@dataclass(frozen=True)
class Shape:
    r"""Which columns of a table the model reads, and as what.

    Settled from the context and held to for the rows scored against it,
    because the model refuses rows shaped differently from what it fitted.
    """

    columns: tuple[str, ...]
    dtypes: Mapping[str, str]
    stypes: Mapping[str, str]
    keys: frozenset[str] = frozenset()

    @classmethod
    def of(
        cls,
        schema: Mapping[str, Any],
        stypes: Mapping[str, str],
        keys: frozenset[str] = frozenset(),
    ) -> Shape:
        dtypes = {
            name: str(schema['columns'][name].get('dtype')) for name in stypes
        }
        return cls(tuple(stypes), dtypes, dict(stypes), keys)

    def tensor(self, table: Mapping[str, Any], where: str) -> Any:
        r"""``table`` as a CPU ``TableTensor`` of this shape."""
        from sdm import TableTensor

        absent = [name for name in self.columns if name not in table['columns']]
        if absent:
            raise StructuredDataError(
                f'{where} leaves out column(s) {absent}, which the context '
                f'carries',
                code='INVALID_SCHEMA',
            )
        arrow = to_arrow(table, self.columns, self.dtypes, self.keys)
        try:
            return TableTensor.from_arrow(
                arrow, stypes=self.stypes, device='cpu'
            )
        except (TypeError, ValueError) as error:
            raise StructuredDataError(
                'table cannot be represented by SDM', code='VALIDATION_FAILED'
            ) from error


@dataclass
class Context:
    r"""A context converted for the model, and how rows are read against it.

    ``x``, ``y`` and ``related`` are what the model is fitted on; the rest is
    all scoring needs, so a fitted session keeps only that.
    """

    x: Any
    y: Any
    instance: Shape
    encoded: tuple[Any, ...] = ()
    labels: tuple[Any, ...] = ()
    related: Any = None
    tables: dict[str, Shape] = field(default_factory=dict)
    relationships: tuple[Any, ...] = ()
    task_links: tuple[Any, ...] = ()

    def fitted(self) -> Context:
        r"""This context without the rows it was fitted on."""
        return replace(self, x=None, y=None, related=None)


def labelled(
    table: Mapping[str, Any], task: Mapping[str, Any], classifies: bool
) -> tuple[Any, tuple[Any, ...], tuple[Any, ...]]:
    r"""The context's labels, and for a classification its classes.

    Returns the labels as a ``TableTensor``, each class as the model encodes
    it, and what the caller calls each one. A boolean target is declared
    ``'true'``/``'false'`` and encodes as ``True``/``False``, so a declared
    class is matched to an encoded one by its text.
    """
    from sdm import TableTensor

    target = task['target']
    column = target['column_name']
    if not table['rows']:
        raise StructuredDataError(
            'the context carries no rows, so there is nothing to learn from',
            code='VALIDATION_FAILED',
        )
    if any(label is None for label in values(table, column)):
        raise StructuredDataError(
            f'context target {column!r} contains null labels',
            code='TASK_SCHEMA_MISMATCH',
        )
    stype = 'categorical' if classifies else 'numerical'
    try:
        arrow = to_arrow(
            table, [column], {column: target.get('dtype', 'float64')}
        )
        y = TableTensor.from_arrow(arrow, stypes={column: stype}, device='cpu')
    except (StructuredDataError, TypeError, ValueError) as error:
        raise StructuredDataError(
            'context labels cannot be represented by SDM',
            code='TASK_SCHEMA_MISMATCH',
        ) from error
    if not classifies:
        return y, (), ()
    encoded = tuple(y.categorical.categories[0].tolist())
    declared = target.get('classes')
    _check_classes(task['kind'], declared, encoded)
    if not declared:
        return y, encoded, encoded
    text = _text_of(encoded)
    named = {text(label): label for label in declared}
    return y, encoded, tuple(named.get(text(label), label) for label in encoded)


def _check_classes(
    kind: str, declared: Sequence[Any] | None, encoded: tuple[Any, ...]
) -> None:
    r"""Refuse classes the task contradicts.

    Asked of the context that arrived: a holdout can take every row of a
    class out of it after the client has counted them.
    """
    count = len(encoded)
    if kind == 'binary_classification' and count != 2:
        raise StructuredDataError(
            f'binary_classification needs exactly two classes, the context '
            f'carries {count}',
            code='TASK_SCHEMA_MISMATCH',
        )
    if kind == 'multiclass_classification' and count < 2:
        raise StructuredDataError(
            f'multiclass_classification needs at least two classes, the '
            f'context carries {count}',
            code='TASK_SCHEMA_MISMATCH',
        )
    if declared is None:
        return
    text = _text_of(encoded)
    named = [text(label) for label in declared]
    if not named:
        raise StructuredDataError(
            'the task declares an empty class list, which no label can satisfy',
            code='TASK_SCHEMA_MISMATCH',
        )
    if len(set(named)) != len(named):
        raise StructuredDataError(
            f'declared classes repeat a label: {sorted(named)}',
            code='TASK_SCHEMA_MISMATCH',
        )
    present = {text(label) for label in encoded}
    if set(named) != present:
        raise StructuredDataError(
            f'declared classes {sorted(set(named))} do not match the labels '
            f'present in the context {sorted(present)}',
            code='TASK_SCHEMA_MISMATCH',
        )


def _text_of(encoded: tuple[Any, ...]) -> Callable[[Any], str]:
    r"""How a declared class is matched to an encoded one: by its text, and
    for a boolean target without case, since it is declared ``'true'`` and
    encodes as ``True``."""
    if all(isinstance(label, bool) for label in encoded):
        return lambda label: str(label).casefold()
    return str
