# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable, local views of schema metadata and model explanations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import pandas as pd

from sdm_client.errors import StructuredDataError

_METADATA_COLUMNS = [
    'table',
    'column',
    'dtype',
    'stype',
    'primary_key',
    'time_column',
]
_LINK_COLUMNS = [
    'source_table',
    'source_column',
    'target_table',
    'target_column',
]


class _Printable:
    def to_text(self) -> str:
        raise NotImplementedError

    def print(self) -> None:
        """Print this report locally."""
        print(self.to_text())


@dataclass(frozen=True)
class ColumnMetadata:
    """A column's existing type descriptors."""

    name: str
    dtype: str
    stype: str


@dataclass(frozen=True)
class TableMetadata(_Printable):
    """A detached table descriptor, without row data."""

    name: str
    columns: tuple[ColumnMetadata, ...]
    primary_key: tuple[str, ...]
    time_column: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, 'columns', tuple(self.columns))
        object.__setattr__(self, 'primary_key', tuple(self.primary_key))

    def to_text(self) -> str:
        lines = [self.name]
        for column in self.columns:
            flags = []
            if column.name in self.primary_key:
                flags.append('primary key')
            if column.name == self.time_column:
                flags.append('time column')
            suffix = f' ({", ".join(flags)})' if flags else ''
            lines.append(
                f'  {column.name}: {column.dtype}, {column.stype}{suffix}'
            )
        return '\n'.join(lines)

    def to_pandas(self) -> pd.DataFrame:
        return MetadataReport((self,)).to_pandas()


@dataclass(frozen=True)
class MetadataReport(_Printable):
    """Schema tables in their existing order."""

    tables: tuple[TableMetadata, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'tables', tuple(self.tables))

    def to_text(self) -> str:
        return '\n'.join(table.to_text() for table in self.tables)

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (
                    table.name,
                    column.name,
                    column.dtype,
                    column.stype,
                    column.name in table.primary_key,
                    column.name == table.time_column,
                )
                for table in self.tables
                for column in table.columns
            ],
            columns=_METADATA_COLUMNS,
        )


@dataclass(frozen=True)
class LinkMetadata:
    """An existing directed foreign-key relationship."""

    source_table: str
    source_column: str
    target_table: str
    target_column: str


@dataclass(frozen=True)
class LinkReport(_Printable):
    """Relationships in their existing order."""

    links: tuple[LinkMetadata, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, 'links', tuple(self.links))

    def to_text(self) -> str:
        return '\n'.join(
            f'{v.source_table}.{v.source_column} -> '
            f'{v.target_table}.{v.target_column}'
            for v in self.links
        )

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (
                    v.source_table,
                    v.source_column,
                    v.target_table,
                    v.target_column,
                )
                for v in self.links
            ],
            columns=_LINK_COLUMNS,
        )


@dataclass(frozen=True)
class FeatureImportanceRow:
    """A signed aggregate attribution for one table column."""

    table: str
    column: str
    score: float


@dataclass(frozen=True)
class FeatureImportance(_Printable):
    """Immutable scores, optionally normalized by aggregate absolute mass."""

    rows: tuple[FeatureImportanceRow, ...]
    normalized: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, 'rows', tuple(self.rows))

    def to_text(self) -> str:
        return '\n'.join(
            f'{v.table}.{v.column}: {v.score:.6g}' for v in self.rows
        )

    def to_pandas(self) -> pd.DataFrame:
        return pd.DataFrame(
            [(v.table, v.column, v.score) for v in self.rows],
            columns=['table', 'column', 'score'],
        )


def _invalid(path: str) -> StructuredDataError:
    return StructuredDataError(
        'Malformed explanation field',
        code='INVALID_REQUEST',
        details={'path': path},
    )


def _freeze(
    value: Any,
    path: str = '$',
    ancestors: frozenset[int] = frozenset(),
    depth: int = 0,
) -> Any:
    if depth > 128 or id(value) in ancestors:
        raise _invalid(path)
    if type(value) in (str, int, float, bool, type(None)):
        return value
    ancestors = ancestors | {id(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise _invalid(path)
        return MappingProxyType(
            {
                k: _freeze(v, f'{path}.{k}', ancestors, depth + 1)
                for k, v in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze(v, f'{path}[{i}]', ancestors, depth + 1)
            for i, v in enumerate(value)
        )
    raise _invalid(path)


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _invalid(path)
    return value


@dataclass(frozen=True, init=False)
class ExplanationView(_Printable):
    """An immutable explanation snapshot; field validation is demand-driven."""

    raw: Mapping[str, Any]

    def __init__(self, raw: Mapping[str, Any]) -> None:
        object.__setattr__(self, 'raw', _freeze(_mapping(raw, '$')))

    @property
    def details(self) -> Mapping[str, Any] | None:
        value = self.raw.get('details')
        return None if value is None else _mapping(value, '$.details')

    def _prose(self, names: tuple[str, ...]) -> tuple[Any, str]:
        for name in names:
            if name in self.raw:
                return self.raw[name], f'$.{name}'
        # Malformed unrelated details must not prevent absent prose access.
        nested = self.raw.get('details')
        if isinstance(nested, Mapping):
            for name in names:
                if name in nested:
                    return nested[name], f'$.details.{name}'
        return None, '$'

    @property
    def summary(self) -> str | None:
        value, path = self._prose(('summary',))
        if value is not None and not isinstance(value, str):
            raise _invalid(path)
        return value

    @property
    def warnings(self) -> tuple[str, ...] | None:
        value, path = self._prose(('warnings', 'warning'))
        if value is None:
            return None
        if path.endswith('.warning'):
            if not isinstance(value, str):
                raise _invalid(path)
            return (value,)
        if not isinstance(value, tuple) or any(
            not isinstance(v, str) for v in value
        ):
            raise _invalid(path)
        return value

    def _sequence(self, name: str) -> tuple[Any, ...] | None:
        nested = self.raw.get('format') == 'kumo_rfm_v2_1'
        owner = self.details if nested else self.raw
        value = None if owner is None else owner.get(name)
        if value is not None and not isinstance(value, tuple):
            raise _invalid(f'$.details.{name}' if nested else f'$.{name}')
        return value

    @property
    def cohorts(self) -> tuple[Any, ...] | None:
        return self._sequence('cohorts')

    @property
    def subgraphs(self) -> tuple[Any, ...] | None:
        return self._sequence('subgraphs')

    def to_text(self) -> str:
        return '\n'.join(
            ([self.summary] if self.summary is not None else [])
            + [f'Warning: {v}' for v in self.warnings or ()]
        )

    def feature_importance(
        self, normalize: bool = False
    ) -> FeatureImportance | None:
        """Aggregate supported cell scores lazily using the shared Rust policy."""
        if type(normalize) is not bool:
            raise _invalid('normalize')
        if self.raw.get('format') != 'kumo_rfm_v2_1':
            return None
        subgraphs = self.subgraphs
        if subgraphs is None:
            return None
        cells = []
        for i, entry in enumerate(subgraphs):
            path = f'$.details.subgraphs[{i}]'
            tables = _mapping(
                _mapping(entry, path).get('tables'), path + '.tables'
            )
            for table, nodes in tables.items():
                table_path = path + '.tables.' + table
                for node_name, node in _mapping(nodes, table_path).items():
                    node_path = table_path + '.' + node_name
                    columns = _mapping(
                        _mapping(node, node_path).get('cells'),
                        node_path + '.cells',
                    )
                    for column, cell in columns.items():
                        cell_path = node_path + '.cells.' + column
                        score = _mapping(cell, cell_path).get('score')
                        if score is None:
                            continue
                        # Only scores cross the boundary, never raw cell values.
                        scalar = (
                            score
                            if type(score) in (str, int, float, bool)
                            else {'invalid': True}
                        )
                        cells.append(
                            dict(
                                table=table,
                                column=column,
                                score=json.dumps(scalar),
                                path=cell_path + '.score',
                            )
                        )
        from sdm_client._rust.direct import _call

        values = _call(
            'feature_importance', {'cells': cells, 'normalize': normalize}
        )
        return FeatureImportance(
            tuple(FeatureImportanceRow(**v) for v in values['rows']), normalize
        )
