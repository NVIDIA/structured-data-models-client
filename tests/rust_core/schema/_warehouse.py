# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A stand-in Snowflake warehouse, for reading semantic views offline.

It holds one semantic view's YAML and the tables behind it, and answers the
statements reading a semantic view takes the way Snowflake would: the YAML,
``DESCRIBE TABLE``, ``SHOW TABLES LIKE`` and the inference samples. A quoted
name is the name it spells, case kept; an unquoted one is upper-cased. What it
answers depends only on the statement and its parameters, so today's Python
and the Rust core, asking the same statement, read the same answer.

Today's Python reads it through a real ``SnowflakeSource`` over
:class:`Warehouse`, a connection of the connector's class; the core's host
runs each query through that same source.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow as pa

DATABASE = 'TEST'
SCHEMA = 'PARITY_SHOP'

_QUOTED = r'"((?:[^"]|"")*)"'
_DESCRIBE = re.compile(rf'DESCRIBE TABLE {_QUOTED}\.{_QUOTED}\.{_QUOTED}')
_LIKE = re.compile(rf'SHOW TABLES LIKE \? IN SCHEMA {_QUOTED}\.{_QUOTED}')
_LISTING = re.compile(rf'SHOW TABLES IN SCHEMA {_QUOTED}\.{_QUOTED}')
_ORDER_ALIAS = re.compile(r'\sAS (__kumo_order_\d+__)$')
_NAMED = re.compile(rf'\sAS {_QUOTED}$')
_COLUMN = re.compile(rf'{_QUOTED}$')
_LIMIT = re.compile(r'\sLIMIT (\d+)\s*$')

#: Rows each stand-in table holds.
ROWS = 24


def identifier(text: str) -> str:
    r"""The name an identifier spells: a quoted one as written, inner quotes
    undoubled; an unquoted one upper-cased."""
    text = str(text).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        return text[1:-1].replace('""', '"')
    return text.upper()


def missing(what: str) -> Exception:
    from snowflake.connector.errors import ProgrammingError

    return ProgrammingError(
        msg=f'SQL compilation error:\n{what} does not exist or not authorized.',
        errno=2003,
        sqlstate='42S02',
    )


def split_top_level(text: str) -> list[str]:
    r"""``text`` split at the commas outside parentheses, quotes and
    literals."""
    items, depth, start, i = [], 0, 0, 0
    while i < len(text):
        char = text[i]
        if char in '"\'':
            end = i + 1
            while end < len(text):
                if text[end] == char:
                    if end + 1 < len(text) and text[end + 1] == char:
                        end += 2
                        continue
                    break
                if char == "'" and text[end] == '\\':
                    end += 1
                end += 1
            i = end + 1
            continue
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            items.append(text[start:i].strip())
            start = i + 1
        i += 1
    items.append(text[start:].strip())
    return items


def _top_level_from(sql: str) -> int:
    r"""Where the ``FROM`` of a ``SELECT`` starts, outside parentheses and
    quotes."""
    depth, i = 0, 0
    while i < len(sql):
        char = sql[i]
        if char in '"\'':
            end = sql.index(char, i + 1) if char in sql[i + 1 :] else len(sql)
            i = end + 1
            continue
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif depth == 0 and sql.startswith(' FROM ', i):
            return i
        i += 1
    raise AssertionError(f'no FROM in {sql[:200]!r}')


def _values(kind: str, rows: int, salt: int) -> pa.Array:
    kind = kind.strip().upper()
    if kind.startswith('NUMBER'):
        scale = 0
        found = re.search(r',\s*(\d+)\s*\)', kind)
        if found:
            scale = int(found.group(1))
        if scale:
            return pa.array([(i * 7 + salt) / 4 for i in range(rows)])
        return pa.array([i + 1 + salt * 1000 for i in range(rows)], pa.int64())
    if kind in ('FLOAT', 'DOUBLE', 'REAL') or kind.startswith('DECIMAL'):
        return pa.array([(i * 7 + salt) / 4 for i in range(rows)])
    if kind.startswith('TIMESTAMP'):
        start = dt.datetime(2024, 1, 1) + dt.timedelta(days=salt)
        return pa.array(
            [start + dt.timedelta(hours=13 * i) for i in range(rows)],
            pa.timestamp('ns'),
        )
    if kind == 'DATE':
        start = dt.date(2024, 1, 1) + dt.timedelta(days=salt)
        return pa.array([start + dt.timedelta(days=i) for i in range(rows)])
    if kind == 'BOOLEAN':
        return pa.array([(i + salt) % 3 == 0 for i in range(rows)])
    return pa.array([f'v{(i + salt) % 5}' for i in range(rows)])


class Warehouse:
    r"""The stand-in's state: the YAML of each semantic view, by its
    qualified name as the source asks for it, and each table's columns, by
    ``(database, schema, table)``, each a type as ``DESCRIBE TABLE`` gives
    it."""

    def __init__(
        self,
        views: Mapping[str, str],
        tables: Mapping[tuple[str, str, str], Mapping[str, str]],
    ) -> None:
        self.views = dict(views)
        self.tables = {key: dict(columns) for key, columns in tables.items()}
        self.log: list[tuple[str, Any]] = []
        self.unknown: list[str] = []

    def answer(self, sql: str, params: Sequence[Any] | None) -> pa.Table:
        self.log.append((sql, None if params is None else list(params)))
        flat = ' '.join(sql.split())
        if flat == 'SELECT SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW(?)':
            name = str((params or [''])[0])
            if name not in self.views:
                raise missing(f"Semantic view '{name}'")
            return pa.table({flat[7:]: [self.views[name]]})
        found = _DESCRIBE.fullmatch(flat)
        if found:
            key = tuple(part.replace('""', '"') for part in found.groups())
            columns = self.tables.get(key)
            if columns is None:
                raise missing(f"Table '{'.'.join(key)}'")
            return pa.table(
                {
                    'name': list(columns),
                    'type': list(columns.values()),
                    'kind': ['COLUMN'] * len(columns),
                    'null?': ['Y'] * len(columns),
                    'default': pa.nulls(len(columns), pa.string()),
                    'primary key': ['N'] * len(columns),
                    'unique key': ['N'] * len(columns),
                }
            )
        found = _LIKE.fullmatch(flat)
        if found:
            return self._listing(found.groups(), str((params or [''])[0]))
        found = _LISTING.fullmatch(flat)
        if found:
            return self._listing(found.groups(), '%')
        if flat.startswith('SELECT ') and ' FROM ' in flat:
            return self._sample(flat)
        self.unknown.append(flat)
        raise AssertionError(
            f'the stand-in warehouse cannot answer {flat[:300]!r}'
        )

    def _listing(self, namespace: Sequence[str], like: str) -> pa.Table:
        database, schema = (part.replace('""', '"') for part in namespace)
        pattern = re.compile(
            ''.join(
                '.' if c == '_' else '.*' if c == '%' else re.escape(c)
                for c in like
            ),
            re.IGNORECASE | re.DOTALL,
        )
        names = sorted(
            table
            for (d, s, table) in self.tables
            if (d, s) == (database, schema) and pattern.fullmatch(table)
        )
        count = len(names)
        return pa.table(
            {
                'created_on': pa.nulls(count, pa.string()),
                'name': names,
                'database_name': [database] * count,
                'schema_name': [schema] * count,
                'kind': ['TABLE'] * count,
                'rows': [ROWS] * count,
                'bytes': [ROWS * 64] * count,
                'is_external': ['N'] * count,
                'is_hybrid': ['N'] * count,
                'is_iceberg': ['N'] * count,
                'is_dynamic': ['N'] * count,
            }
        )

    def _sample(self, flat: str) -> pa.Table:
        at = _top_level_from(flat)
        items = split_top_level(flat[len('SELECT ') : at])
        rest = flat[at + len(' FROM ') :]
        found = re.match(rf'{_QUOTED}\.{_QUOTED}\.{_QUOTED}', rest)
        if not found:
            self.unknown.append(flat)
            raise AssertionError(f'a sample of no table: {flat[:300]!r}')
        key = tuple(part.replace('""', '"') for part in found.groups())
        columns = self.tables.get(key)
        if columns is None:
            raise missing(f"Table '{'.'.join(key)}'")
        limit = _LIMIT.search(flat)
        rows = min(ROWS, int(limit.group(1))) if limit else ROWS
        out: dict[str, pa.Array] = {}
        for salt, item in enumerate(items):
            order = _ORDER_ALIAS.search(item)
            named = _NAMED.search(item)
            plain = _COLUMN.fullmatch(item)
            if order:
                name = order.group(1)
                values: pa.Array = pa.array(range(rows), pa.int64())
            elif named:
                name = named.group(1).replace('""', '"')
                values = pa.nulls(rows, pa.string())
            elif plain:
                name = plain.group(1).replace('""', '"')
                if name not in columns:
                    raise missing(f"Column '{name}'")
                values = _values(columns[name], rows, salt)
            else:
                self.unknown.append(flat)
                raise AssertionError(f'a projection it cannot name: {item!r}')
            while name in out:
                name += '_'
            out[name] = values
        return pa.table(out)


def connection(warehouse: Warehouse) -> Any:
    r"""A connection of the Snowflake connector's class whose statements the
    stand-in answers."""
    from sdm_connectors.backends.snowflake import Connection

    class Cursor:
        def __init__(self) -> None:
            self.description: list[tuple[str, ...]] = []
            self._table: pa.Table | None = None

        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *exc: object) -> None:
            return None

        def execute(self, sql: str, params: Any = None) -> Cursor:
            self._table = warehouse.answer(sql, params)
            self.description = [(name,) for name in self._table.column_names]
            return self

        def fetch_arrow_all(self, force_return_table: bool = False) -> pa.Table:
            assert self._table is not None
            return self._table

        def fetchall(self) -> list[tuple[Any, ...]]:
            assert self._table is not None
            return [tuple(row.values()) for row in self._table.to_pylist()]

        def close(self) -> None:
            return None

    class StandIn(Connection):  # type: ignore[misc]
        def __init__(self) -> None:
            self._paramstyle = 'pyformat'

        def cursor(self, *args: Any, **kwargs: Any) -> Cursor:
            return Cursor()

        def close(self) -> None:
            return None

        def __del__(self) -> None:
            return None

    return StandIn()


def source(warehouse: Warehouse) -> Any:
    r"""A ``SnowflakeSource`` over the stand-in, in ``TEST.PARITY_SHOP``."""
    from sdm_connectors import SnowflakeSource

    return SnowflakeSource(
        connection(warehouse), database=DATABASE, schema=SCHEMA
    )


def tables_of(cfg: Any) -> dict[tuple[str, str, str], dict[str, str]]:
    r"""The tables a semantic view's YAML reads, with every column it names
    plainly, as a key, or as a relationship's column, typed as it declares
    them (``NUMBER(38,0)`` otherwise)."""
    tables: dict[tuple[str, str, str], dict[str, str]] = {}
    if not isinstance(cfg, dict):
        return tables
    by_name: dict[Any, tuple[str, str, str]] = {}
    for table in cfg.get('tables') or []:
        if not isinstance(table, dict) or not isinstance(
            table.get('base_table'), dict
        ):
            continue
        base = table['base_table']
        key = (
            identifier(base.get('database', DATABASE)),
            identifier(base.get('schema', SCHEMA)),
            identifier(base.get('table', '')),
        )
        by_name[table.get('name')] = key
        columns = tables.setdefault(key, {})
        for name in (table.get('primary_key') or {}).get('columns') or []:
            columns.setdefault(identifier(name), 'NUMBER(38,0)')
        for section in ('dimensions', 'time_dimensions', 'facts'):
            for column in table.get(section) or []:
                if not isinstance(column, dict) or 'name' not in column:
                    continue
                expr = column.get('expr')
                name = str(column['name'])
                plain = expr is None or str(expr).strip().split('.')[-1] in (
                    name,
                    f'"{identifier(name)}"',
                )
                if plain:
                    columns.setdefault(
                        identifier(name),
                        column.get('data_type') or 'NUMBER(38,0)',
                    )
    for link in cfg.get('relationships') or []:
        if not isinstance(link, dict):
            continue
        for pair in link.get('relationship_columns') or []:
            for side in ('left', 'right'):
                key = by_name.get(link.get(f'{side}_table'))
                column = (
                    pair.get(f'{side}_column')
                    if isinstance(pair, dict)
                    else None
                )
                if key is not None and column is not None:
                    tables[key].setdefault(identifier(column), 'NUMBER(38,0)')
    return tables
