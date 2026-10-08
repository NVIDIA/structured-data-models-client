# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Databricks and Snowflake behind stub drivers, for the phase 5.5 checks.

Adapted from ``tests/seam/_warehouse.py``, which pins today's statements and
may change with this phase. This copy is locked, and it is configurable where
the checks need it to be:

* the time column's declared type (Databricks ``timestamp``,
  ``timestamp_ntz``, ``date`` or ``string``; Snowflake ``TIMESTAMP_NTZ``,
  ``TIMESTAMP_LTZ``, ``TIMESTAMP_TZ`` or ``DATE``);
* what Snowflake's ``SHOW TABLES`` lists for each table (``rows``, ``bytes``,
  ``kind`` and the ``is_*`` flags, in the real column order), and what
  Databricks' ``DESCRIBE DETAIL`` says (``sizeInBytes``, ``numFiles``,
  ``format``);
* how many rows a prefiltered read (``SAMPLE SYSTEM``, ``TABLESAMPLE``, or a
  bounded subquery) returns;
* the tables themselves, so a composite reference can be linked.

Every statement the driver is handed is logged with its parameters. The
connections subclass the real drivers' connection classes, so
``sdm_connectors`` runs them as it runs a real one. A metadata statement this
module does not know (``DESCRIBE``, ``SHOW``, ``information_schema``) fails
with the driver's own error, as an unsupported statement would on a real
warehouse; any other statement it has no answer for raises :class:`Refused`.
"""

from __future__ import annotations

import inspect
import json
import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pyarrow as pa


class Refused(Exception):
    r"""What a stub raises for SQL it has no answer for."""


# The shop ####################################################################

SHOP_TIMES = pd.date_range('2024-01-01 08:00', periods=60, freq='37h')


def shop() -> dict[str, pa.Table]:
    r"""Twelve users and sixty orders; ``ordered_at`` is naive UTC."""
    users = pa.table(
        {
            'user_id': pa.array(range(1, 13), pa.int64()),
            'country': ['DE', 'FR', 'DE', 'US', 'US', 'FR'] * 2,
            'age': pa.array([31, 45, 27, 38, 52, 23] * 2, pa.int64()),
        }
    )
    orders = pa.table(
        {
            'order_id': pa.array(range(100, 160), pa.int64()),
            'user_id': pa.array(
                [(i * 7) % 12 + 1 for i in range(60)], pa.int64()
            ),
            'amount': [round(1.5 + (i * 13) % 17 * 0.75, 2) for i in range(60)],
            'ordered_at': pa.array(SHOP_TIMES, pa.timestamp('us')),
        }
    )
    return {'users': users, 'orders': orders}


SHOP_KEYS = {'users': 'user_id', 'orders': 'order_id'}
SHOP_TIME = {'orders': 'ordered_at'}
SHOP_LINKS = {'orders': ('user_id', 'users', 'user_id')}

#: Questions asked of the shop: (query, materialize options).
QUESTIONS: dict[str, tuple[str, dict[str, Any]]] = {
    'count': (
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
        {
            'indices': [1, 2, 3],
            'random_seed': 7,
            'anchor_time': pd.Timestamp('2024-03-20 12:00:00'),
        },
    ),
    'sum': (
        'PREDICT SUM(orders.amount, 0, 60, days) FOR EACH users.user_id',
        {'indices': [4, 5], 'num_neighbors': [3, 2], 'random_seed': 11},
    ),
}

_DATABRICKS_TYPES = {
    'int64': 'bigint',
    'double': 'double',
    'string': 'string',
    'large_string': 'string',
}
_SNOWFLAKE_TYPES = {
    'int64': 'NUMBER(38,0)',
    'double': 'FLOAT',
    'string': 'VARCHAR(16777216)',
    'large_string': 'VARCHAR(16777216)',
}

#: Snowflake's ``SHOW TABLES`` columns, in the order Snowflake returns them.
SHOW_TABLES_COLUMNS = (
    'created_on',
    'name',
    'database_name',
    'schema_name',
    'kind',
    'comment',
    'cluster_by',
    'rows',
    'bytes',
    'owner',
    'retention_time',
    'automatic_clustering',
    'change_tracking',
    'search_optimization',
    'search_optimization_progress',
    'search_optimization_bytes',
    'is_external',
    'enable_schema_evolution',
    'owner_role_type',
    'is_event',
    'budget',
    'is_hybrid',
    'is_iceberg',
    'is_dynamic',
    'is_immutable',
)

#: Databricks' ``DESCRIBE DETAIL`` columns for a Delta table.
DESCRIBE_DETAIL_COLUMNS = (
    'format',
    'id',
    'name',
    'description',
    'location',
    'createdAt',
    'lastModified',
    'partitionColumns',
    'clusteringColumns',
    'numFiles',
    'sizeInBytes',
    'properties',
    'minReaderVersion',
    'minWriterVersion',
    'tableFeatures',
    'statistics',
)


def listing(
    rows: int,
    bytes_: int,
    *,
    kind: str = 'TABLE',
    is_external: str = 'N',
    is_hybrid: str = 'N',
    is_iceberg: str = 'N',
    is_dynamic: str = 'N',
) -> dict[str, Any]:
    r"""What ``SHOW TABLES`` lists for one table, apart from its name."""
    return {
        'rows': rows,
        'bytes': bytes_,
        'kind': kind,
        'is_external': is_external,
        'is_hybrid': is_hybrid,
        'is_iceberg': is_iceberg,
        'is_dynamic': is_dynamic,
    }


def detail(size_in_bytes: int, num_files: int = 1) -> dict[str, Any]:
    r"""What ``DESCRIBE DETAIL`` says of one Delta table."""
    return {
        'format': 'delta',
        'sizeInBytes': size_in_bytes,
        'numFiles': num_files,
    }


# The configuration ###########################################################


@dataclass
class Statement:
    sql: str
    params: list[Any] | None


@dataclass
class Warehouse:
    r"""One stub warehouse.

    Args:
        kind: ``databricks`` (the SQL connector), ``spark`` (a notebook's
            session) or ``snowflake``.
        time_type: The time column's declared type, as the warehouse's
            catalog spells it; ``None`` for the default naive type.
        listings: Snowflake: per table (lower case), what ``SHOW TABLES``
            lists, or ``None`` to list nothing for it (a view). Tables not
            named list their own row count and 1 MiB.
        details: Databricks: per table (lower case), what ``DESCRIBE DETAIL``
            says, or ``None`` for the statement to fail. Tables not named
            fail it too.
        prefiltered_rows: How many rows a prefiltered inference sample or
            entity draw returns; ``None`` returns what the table holds.
        tables, keys, time, links: The data, primary keys, time columns and
            foreign keys, by lower-case table name.
        lookup_found: Whether a ``... LIMIT 1`` lookup of a parent finds a
            row.
        nullable_keys: Whether the catalog says a key may be null.
    """

    kind: str
    time_type: str | None = None
    listings: Mapping[str, dict[str, Any] | None] = field(default_factory=dict)
    details: Mapping[str, dict[str, Any] | None] = field(default_factory=dict)
    prefiltered_rows: int | None = None
    tables: dict[str, pa.Table] = field(default_factory=shop)
    keys: dict[str, str] = field(default_factory=lambda: dict(SHOP_KEYS))
    time: dict[str, str] = field(default_factory=lambda: dict(SHOP_TIME))
    links: dict[str, tuple[str, str, str]] = field(
        default_factory=lambda: dict(SHOP_LINKS)
    )
    lookup_found: bool = True
    nullable_keys: bool = False
    statements: list[Statement] = field(default_factory=list)

    @property
    def upper(self) -> bool:
        return self.kind == 'snowflake'

    # Names.

    def _table_name(self, name: str) -> str:
        return name.upper() if self.upper else name

    def _column_name(self, name: str) -> str:
        return name.upper() if self.upper else name

    def data(self, table: str) -> pa.Table:
        r"""The table as the warehouse holds it: names, and the time
        column in its declared type."""
        data = self.tables[table]
        time = self.time.get(table)
        columns = []
        for name in data.column_names:
            column = data.column(name)
            if name == time:
                column = self._time_array(column)
            columns.append(column)
        names = [self._column_name(n) for n in data.column_names]
        return pa.Table.from_arrays(columns, names=names)

    def _time_kind(self) -> str:
        t = (self.time_type or '').lower()
        if t.startswith('date'):
            return 'date'
        if t.startswith('string') or t.startswith('varchar'):
            return 'string'
        if 'ltz' in t or 'tz' in t.replace('ntz', ''):
            return 'zoned'
        if self.kind != 'snowflake' and t == 'timestamp':
            return 'zoned'
        return 'naive'

    def _time_array(self, column: pa.ChunkedArray) -> pa.Array:
        values = column.to_pylist()
        kind = self._time_kind()
        if kind == 'zoned':
            return pa.array(
                [pd.Timestamp(v).tz_localize('UTC') for v in values],
                pa.timestamp('us', tz='UTC'),
            )
        if kind == 'date':
            return pa.array([pd.Timestamp(v).date() for v in values])
        if kind == 'string':
            return pa.array(
                [pd.Timestamp(v).strftime('%Y-%m-%d %H:%M:%S') for v in values]
            )
        return pa.array(values, pa.timestamp('us'))

    def _declared_type(self, table: str, name: str, arrow: pa.DataType) -> str:
        if name == self.time.get(table):
            if self.time_type is not None:
                return self.time_type
            return 'TIMESTAMP_NTZ(9)' if self.upper else 'timestamp_ntz'
        types = _SNOWFLAKE_TYPES if self.upper else _DATABRICKS_TYPES
        return types[str(arrow)]

    def type_code(self, data_type: pa.DataType) -> Any:
        if not self.upper:
            return str(data_type)
        if pa.types.is_timestamp(data_type):
            return 6 if data_type.tz is not None else 8
        if pa.types.is_date(data_type):
            return 3
        return {
            pa.int64(): 0,
            pa.float64(): 1,
            pa.string(): 2,
        }.get(data_type, 2)

    # Answers.

    def answer(self, sql: str, params: Any) -> pa.Table:
        flat = _flat(sql)
        params = list(params or [])
        if self.upper:
            return self._snowflake(sql, flat, params)
        return self._databricks(sql, flat, params)

    def _source(self, sql: str) -> str:
        r"""The table a statement reads from."""
        names = '|'.join(re.escape(n) for n in self.tables)
        found = re.search(
            rf'(?:FROM|JOIN)\s+[^\s]*?[`"]?\b({names})\b[`"]?(?=[\s)]|$)',
            sql,
            re.IGNORECASE,
        )
        if found is None:
            raise Refused(sql)
        return found.group(1).lower()

    def _is_lookup(self, flat: str) -> bool:
        return bool(re.search(r'\bLIMIT 1$', flat)) and not re.search(
            r'__kumo_order_|COUNT\(', flat
        )

    def _prefiltered(self, flat: str) -> bool:
        return bool(
            re.search(r'\bSAMPLE SYSTEM\b|\bTABLESAMPLE\b', flat)
            or re.search(r'FROM \(SELECT .* LIMIT \d+\)', flat)
        )

    def _sample(self, flat: str) -> pa.Table:
        r"""An inference sample: the order terms first, then each projection,
        a column or an expression (answered with nulls)."""
        data = self.data(self._source(flat))
        head = flat.split(' FROM ', 1)[0]
        terms = re.findall(r'AS (__kumo_order_\d+__)', head)
        select = head[len('SELECT ') :]
        if terms:
            select = head.rsplit(f'AS {terms[-1]}', 1)[1].lstrip(', ')
        if self._prefiltered(flat) and self.prefiltered_rows is not None:
            data = _cycle(data, self.prefiltered_rows)
        known = {c.lower(): c for c in data.column_names}
        out: dict[str, Any] = {
            t: pa.array([0] * data.num_rows, pa.int64()) for t in terms
        }
        for item in _items(select):
            found = re.fullmatch(r'(.*) AS [`"](\w+)[`"]', item)
            expression, name = (
                (found.group(1), found.group(2)) if found else (item, None)
            )
            bare = re.fullmatch(r'[`"](\w+)[`"]', expression.strip())
            if bare and bare.group(1).lower() in known:
                column = data.column(known[bare.group(1).lower()])
                out[name or known[bare.group(1).lower()]] = column
            elif name is not None:
                out[name] = pa.nulls(data.num_rows, pa.string())
            else:
                raise Refused(flat)
        return pa.table(out)

    def _min_max(self, sql: str) -> pa.Table:
        rows = []
        for select in sql.split('UNION ALL'):
            found = re.search(r'(\d+) as table_index', select)
            if found is None:
                raise Refused(sql)
            table = self._source(select)
            data = self.data(table)
            column = data.column(self._column_name(self.time[table]))
            values = [v for v in column.to_pylist() if v is not None]
            rows.append((int(found.group(1)), min(values), max(values)))
        time_type = (
            self.data(next(iter(self.time)))
            .schema.field(self._column_name(next(iter(self.time.values()))))
            .type
        )
        return pa.table(
            {
                'table_index': pa.array([r[0] for r in rows], pa.int64()),
                'min_date': pa.array([r[1] for r in rows], time_type),
                'max_date': pa.array([r[2] for r in rows], time_type),
            }
        )

    def _key_stats(self, sql: str) -> pa.Table:
        out: dict[str, list[int]] = {'k': [], 'n': [], 'd': [], 'z': []}
        for select in sql.split('UNION ALL'):
            found = re.search(
                r'SELECT (\d+) AS k, COUNT\(\*\) AS n, '
                r'(?:COUNT\(DISTINCT ([^)]*)\)|COUNT\(\*\) - )',
                select,
            )
            if found is None:
                raise Refused(sql)
            data = self.data(self._source(select))
            columns = re.findall(r'[`"](\w+)[`"]', found.group(2) or '')
            if not columns:
                columns = re.findall(r'[`"](\w+)[`"] IS NULL', select)[:1]
            known = {c.lower(): c for c in data.column_names}
            values = list(
                zip(
                    *[
                        data.column(known[c.lower()]).to_pylist()
                        for c in columns
                    ],
                    strict=True,
                )
            )
            complete = [v for v in values if None not in v]
            out['k'].append(int(found.group(1)))
            out['n'].append(len(values))
            out['d'].append(len(set(complete)))
            out['z'].append(len(values) - len(complete))
        return pa.table({k: pa.array(v, pa.int64()) for k, v in out.items()})

    def _by_ids(self, flat: str, rows: list[Any], batch: str) -> pa.Table:
        r"""The rows holding each id, with the id's position as its batch.

        A key list of objects or arrays carries each id's window: the end,
        inclusive, and the start, exclusive.
        """
        joined = flat.split(' JOIN ', 1)[1]
        table = self._source('FROM ' + joined)
        data = self.data(table)
        selected = re.split(r' FROM (?:TMP|__kumo_keys__)', flat)[0]
        names = _projected(selected.split(f'{batch},', 1)[-1], data)
        on = re.search(
            r'ON [^\s]*?[`"](\w+)[`"] = (?:TMP|tmp)\.__KUMO_ID__',
            flat,
            re.IGNORECASE,
        )
        if on is None:
            raise Refused(flat)
        known = {c.lower(): c for c in data.column_names}
        keys = data.column(known[on.group(1).lower()]).to_pylist()
        time = self.time.get(table)
        times = (
            [_naive_utc(t) for t in self.tables[table].column(time).to_pylist()]
            if time
            else None
        )
        limit = re.search(r'\) <= (\d+) ORDER BY', flat)
        batches: list[int] = []
        picked: list[int] = []
        for pos, item in enumerate(rows):
            if isinstance(item, dict):
                ident = item.get('id', item.get('f0'))
                end, start = item.get('e'), item.get('s')
            elif isinstance(item, list):
                ident, end, start = (item + [None, None])[:3]
            else:
                ident, end, start = item, None, None
            end, start = _naive_utc(end), _naive_utc(start)
            hits = [i for i, k in enumerate(keys) if k == ident]
            if times is not None:
                if end is not None:
                    hits = [i for i in hits if times[i] <= end]
                if start is not None:
                    hits = [i for i in hits if times[i] > start]
                hits.sort(key=lambda i: (times[i], i), reverse=True)
            if limit is not None:
                hits = hits[: int(limit.group(1))]
            batches += [pos] * len(hits)
            picked += hits
        out: dict[str, Any] = {batch: pa.array(batches, pa.int64())}
        for name in names:
            out[name] = data.column(name).take(pa.array(picked, pa.int64()))
        return pa.table(out)

    def _entity_rows(self, flat: str, ids: list[Any] | None) -> pa.Table:
        r"""An entity draw: the first rows, or the rows holding ``ids``."""
        table = self._source(flat)
        data = self.data(table)
        names = _projected(flat.split(' FROM ', 1)[0], data)
        key = self._column_name(self.keys[table])
        if ids is None:
            limit = re.search(r'LIMIT (\d+)$', flat) or re.search(
                r'SAMPLE ROW \((\d+) ROWS\)', flat
            )
            count = data.num_rows if limit is None else int(limit.group(1))
            if self._prefiltered(flat) and self.prefiltered_rows is not None:
                data = _cycle(data, self.prefiltered_rows)
            index = list(range(min(count, data.num_rows)))
        else:
            keys = data.column(key).to_pylist()
            index = [keys.index(i) for i in ids if i in keys]
        return pa.table(
            {n: data.column(n).take(pa.array(index, pa.int64())) for n in names}
        )

    def _lookup(self) -> pa.Table:
        return pa.table({'one': pa.array([1] if self.lookup_found else [])})

    # Databricks.

    def _databricks(self, sql: str, flat: str, params: list[Any]) -> pa.Table:
        if flat.startswith('SELECT current_catalog(), current_schema()'):
            return pa.table({'c': ['main'], 's': ['shop']})
        if 'information_schema.tables' in flat:
            return pa.table({'table_name': sorted(self.tables)})
        if 'information_schema.columns' in flat:
            schema, table = (str(p).lower() for p in params[:2])
            data = self.tables[table]
            n = data.num_columns
            return pa.table(
                {
                    'column_name': data.column_names,
                    'full_data_type': [
                        self._declared_type(table, f.name, f.type)
                        for f in data.schema
                    ],
                    'is_nullable': [
                        'NO'
                        if c == self.keys.get(table) and not self.nullable_keys
                        else 'YES'
                        for c in data.column_names
                    ],
                    'table_catalog': ['main'] * n,
                    'table_schema': [schema] * n,
                    'table_name': [table] * n,
                }
            )
        if "constraint_type = 'FOREIGN KEY'" in flat:
            table = str(params[1]).lower()
            link = self.links.get(table)
            found = 1 if link else 0
            return pa.table(
                {
                    'constraint_name': [f'{table}_fk'] * found,
                    'column_name': [link[0] if link else ''] * found,
                    'table_catalog': ['main'] * found,
                    'table_schema': ['shop'] * found,
                    'table_name': [link[1] if link else ''] * found,
                    'ref_column_name': [link[2] if link else ''] * found,
                }
            )
        if 'information_schema.table_constraints' in flat:
            table = str(params[1]).lower()
            key = self.keys.get(table)
            found = 1 if key else 0
            return pa.table(
                {
                    'constraint_type': ['PRIMARY KEY'] * found,
                    'constraint_name': [f'{table}_pk'] * found,
                    'column_name': [key or ''] * found,
                }
            )
        if flat.upper().startswith('DESCRIBE DETAIL'):
            table = self._source('FROM ' + flat.split(None, 2)[2])
            found = self.details.get(table)
            if found is None:
                self._fail(sql)
            row = {c: None for c in DESCRIBE_DETAIL_COLUMNS}
            row.update(
                id='00000000-0000-0000-0000-000000000000',
                name=f'main.shop.{table}',
                location='<location>',
                partitionColumns=[],
                clusteringColumns=[],
                minReaderVersion=1,
                minWriterVersion=2,
            )
            row.update(found)
            return pa.Table.from_pylist([row])
        if flat.upper().startswith(('DESCRIBE', 'SHOW', 'ANALYZE')) or (
            'information_schema' in flat
        ):
            self._fail(sql)
        if self._is_lookup(flat):
            return self._lookup()
        if flat.startswith('SELECT COUNT(*) FROM'):
            n = self.data(self._source(flat)).num_rows
            return pa.table({'n': pa.array([n])})
        if 'COUNT(DISTINCT' in flat and ' AS k,' in flat:
            return self._key_stats(flat)
        if ' AS k, COUNT(*) AS n' in flat:
            return self._key_stats(flat)
        if '__kumo_order_' in flat:
            return self._sample(flat)
        if 'as table_index' in flat:
            return self._min_max(flat)
        if flat.startswith('WITH TMP AS'):
            return self._by_ids(flat, json.loads(params[0]), '__KUMO_BATCH__')
        if 'posexplode' in flat:
            return self._entity_rows(flat, json.loads(params[0]))
        if 'TABLESAMPLE' in flat:
            return self._entity_rows(flat, None)
        if flat.startswith('SELECT') and ' FROM ' in flat:
            return self._entity_rows(flat, None)
        raise Refused(sql)

    # Snowflake.

    def _snowflake(self, sql: str, flat: str, params: list[Any]) -> pa.Table:
        if flat.startswith('SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()'):
            return pa.table({'d': ['DB'], 's': ['SHOP']})
        if 'INFORMATION_SCHEMA.TABLES' in flat:
            return pa.table({'TABLE_NAME': sorted(self._names())})
        if flat.startswith('DESCRIBE TABLE'):
            found = re.fullmatch(
                r'DESCRIBE TABLE "(\w+)"\."(\w+)"\."(\w+)"', flat
            )
            if found is None:
                self._fail(sql)
            table = found.group(3).lower()
            if table not in self.tables:
                self._fail(sql)
            data = self.tables[table]
            key = (self.keys.get(table) or '').upper()
            n = data.num_columns
            names = [c.upper() for c in data.column_names]
            return pa.table(
                {
                    'name': names,
                    'type': [
                        self._declared_type(table, f.name, f.type)
                        for f in data.schema
                    ],
                    'kind': ['COLUMN'] * n,
                    'null?': [
                        'N' if c == key and not self.nullable_keys else 'Y'
                        for c in names
                    ],
                    'default': pa.array([None] * n, pa.string()),
                    'primary key': ['Y' if c == key else 'N' for c in names],
                    'unique key': ['N'] * n,
                }
            )
        if flat.startswith('SHOW IMPORTED KEYS'):
            found = re.search(r'"(\w+)"$', flat)
            table = found.group(1).lower() if found else ''
            link = self.links.get(table)
            rows = [link] if link else []
            return pa.table(
                {
                    'created_on': pa.array(['2024-01-01'] * len(rows)),
                    'pk_database_name': ['DB'] * len(rows),
                    'pk_schema_name': ['SHOP'] * len(rows),
                    'pk_table_name': [r[1].upper() for r in rows],
                    'pk_column_name': [r[2].upper() for r in rows],
                    'fk_database_name': ['DB'] * len(rows),
                    'fk_schema_name': ['SHOP'] * len(rows),
                    'fk_table_name': [table.upper()] * len(rows),
                    'fk_column_name': [r[0].upper() for r in rows],
                    'key_sequence': pa.array([1] * len(rows), pa.int64()),
                    'update_rule': ['NO ACTION'] * len(rows),
                    'delete_rule': ['NO ACTION'] * len(rows),
                    'fk_name': [f'{table}_fk'.upper()] * len(rows),
                    'pk_name': [f'{r[1]}_pk'.upper() for r in rows],
                }
            )
        if flat.startswith('SHOW TABLES IN SCHEMA'):
            return self._show_tables(list(self.tables))
        if flat.startswith('SHOW TABLES LIKE'):
            pattern = str(params[0]).lower() if params else ''
            names = [
                t
                for t in self.tables
                if re.fullmatch(
                    re.escape(pattern).replace('_', '.').replace('%', '.*'), t
                )
            ]
            return self._show_tables(names)
        if flat.upper().startswith(('DESCRIBE', 'SHOW')):
            self._fail(sql)
        if self._is_lookup(flat):
            return self._lookup()
        if flat.startswith('SELECT COUNT(*) FROM'):
            n = self.data(self._source(flat)).num_rows
            return pa.table({'N': pa.array([n])})
        if 'COUNT(DISTINCT' in flat and ' AS k,' in flat:
            return self._key_stats(flat)
        if ' AS k, COUNT(*) AS n' in flat:
            return self._key_stats(flat)
        if '__kumo_order_' in flat:
            return self._sample(flat)
        if 'as table_index' in flat:
            return self._min_max(flat)
        if 'FLATTEN' in flat and flat.upper().startswith('WITH TMP AS'):
            return self._by_ids(flat, json.loads(params[0]), '__KUMO_BATCH__')
        if 'FLATTEN' in flat:
            return self._entity_rows(flat, json.loads(params[0]))
        if ' IN (' in flat:
            return self._entity_rows(flat, list(params))
        if flat.startswith('SELECT') and ' FROM ' in flat:
            return self._entity_rows(flat, None)
        raise Refused(sql)

    def _names(self) -> list[str]:
        return [t.upper() for t in self.tables]

    def _show_tables(self, tables: list[str]) -> pa.Table:
        rows = []
        for table in tables:
            listed = self.listings.get(
                table, listing(self.tables[table].num_rows, 1 << 20)
            )
            if listed is None:
                continue
            row: dict[str, Any] = {c: None for c in SHOW_TABLES_COLUMNS}
            row.update(
                created_on='2024-01-01',
                name=table.upper(),
                database_name='DB',
                schema_name='SHOP',
                comment='',
                cluster_by='',
                owner='OWNER',
                retention_time='1',
                automatic_clustering='OFF',
                change_tracking='OFF',
                search_optimization='OFF',
                enable_schema_evolution='N',
                owner_role_type='ROLE',
                is_event='N',
                is_immutable='N',
            )
            row.update(listed)
            rows.append(row)
        columns = {c: [r[c] for r in rows] for c in SHOW_TABLES_COLUMNS}
        types = {'rows': pa.int64(), 'bytes': pa.int64()}
        return pa.table(
            {
                c: pa.array(v, types.get(c, pa.string()))
                for c, v in columns.items()
            }
        )

    def _fail(self, sql: str) -> None:
        if self.upper:
            from snowflake.connector.errors import ProgrammingError

            raise ProgrammingError(
                msg=f'SQL compilation error: unsupported statement: {sql}',
                errno=2003,
                sqlstate='02000',
            )
        from databricks.sql.exc import ServerOperationError

        raise ServerOperationError(f'[UNSUPPORTED] {sql}')


def _naive_utc(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = (
        pd.Timestamp(str(value))
        if not isinstance(value, pd.Timestamp)
        else value
    )
    if ts.tzinfo is not None:
        ts = ts.tz_convert('UTC').tz_localize(None)
    return ts


def _flat(sql: str) -> str:
    r"""A statement on one line, without comments."""
    return ' '.join(re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL).split())


def _projected(text: str, table: pa.Table) -> list[str]:
    r"""The table's columns a select list names, in order."""
    names = re.findall(r'[`"](\w+)[`"](?![.\w`"])', text)
    known = {c.lower(): c for c in table.column_names}
    out: list[str] = []
    for name in names:
        if name.lower() in known and known[name.lower()] not in out:
            out.append(known[name.lower()])
    return out


def _items(text: str) -> list[str]:
    r"""A select list's items, split at the commas outside parentheses."""
    items, depth, start = [], 0, 0
    for index, char in enumerate(text):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            items.append(text[start:index].strip())
            start = index + 1
    items.append(text[start:].strip())
    return [item for item in items if item]


def _cycle(table: pa.Table, rows: int) -> pa.Table:
    r"""``rows`` rows of ``table``, repeated as often as it takes."""
    if table.num_rows == 0 or rows == 0:
        return table.slice(0, 0)
    index = [i % table.num_rows for i in range(rows)]
    return table.take(pa.array(index, pa.int64()))


# Drivers #####################################################################


def _plain(value: Any) -> Any:
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if hasattr(value, 'item'):
        return value.item()
    return value


def _plain_params(params: Any) -> list[Any] | None:
    if params is None or (not isinstance(params, dict) and len(params) == 0):
        return None
    if isinstance(params, dict):
        raise Refused(f'named parameters are not expected: {params!r}')
    return [_plain(p) for p in params]


def _bind(method: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    return dict(inspect.signature(method).bind(None, *args, **kwargs).arguments)


class _Cursor:
    real: Any

    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self._table: pa.Table | None = None
        self._statement = ''
        self.description: list[tuple[Any, ...]] | None = None
        self.rowcount = -1

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        pass

    def _run(self, sql: str, params: Any) -> None:
        warehouse: Warehouse = self._connection.stub
        self._table = None
        self.description = None
        warehouse.statements.append(Statement(sql, _plain_params(params)))
        self._table = warehouse.answer(sql, params)
        self.description = [
            (f.name, warehouse.type_code(f.type)) + (None,) * 5
            for f in self._table.schema
        ]

    def _rows(self) -> list[tuple[Any, ...]]:
        if self._table is None:
            raise Refused('fetch before execute')
        columns = [c.to_pylist() for c in self._table.columns]
        return list(zip(*columns, strict=True))

    def fetchall(self, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        _bind(self.real.fetchall, *args, **kwargs)
        return self._rows()

    def fetchone(self, *args: Any, **kwargs: Any) -> tuple[Any, ...] | None:
        _bind(self.real.fetchone, *args, **kwargs)
        rows = self._rows()
        return rows[0] if rows else None

    def fetchmany(self, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        bound = _bind(self.real.fetchmany, *args, **kwargs)
        size = bound.get('size') or 1
        rows = self._rows()
        out, self._table = rows[:size], None
        return out


class _DatabricksCursor(_Cursor):
    @property
    def real(self) -> Any:
        from databricks.sql.client import Cursor

        return Cursor

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        bound = _bind(self.real.execute, *args, **kwargs)
        self._run(bound['operation'], bound.get('parameters'))
        return self

    def fetchall_arrow(self, *args: Any, **kwargs: Any) -> pa.Table:
        _bind(self.real.fetchall_arrow, *args, **kwargs)
        if self._table is None:
            raise Refused('fetch before execute')
        return self._table


class _SnowflakeCursor(_Cursor):
    @property
    def real(self) -> Any:
        from snowflake.connector.cursor import SnowflakeCursor

        return SnowflakeCursor

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        bound = _bind(self.real.execute, *args, **kwargs)
        sql = bound['command']
        params = bound.get('params')
        if params and self._connection._paramstyle not in ('qmark', 'numeric'):
            raise Refused(
                f'parameters bound with paramstyle '
                f'{self._connection._paramstyle!r}: {sql}'
            )
        self._statement = _flat(sql)
        self._run(sql, params)
        return self

    def _metadata(self) -> bool:
        return self._statement.upper().startswith(('SHOW', 'DESC'))

    def fetch_arrow_all(self, *args: Any, **kwargs: Any) -> pa.Table | None:
        bound = _bind(self.real.fetch_arrow_all, *args, **kwargs)
        if self._metadata():
            from snowflake.connector.errors import NotSupportedError

            raise NotSupportedError(
                msg='Not supported for metadata statements', errno=255001
            )
        if self._table is None:
            raise Refused('fetch before execute')
        if self._table.num_rows == 0 and not bound.get('force_return_table'):
            return None
        return self._table

    def fetch_pandas_all(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        _bind(self.real.fetch_pandas_all, *args, **kwargs)
        if self._metadata():
            from snowflake.connector.errors import NotSupportedError

            raise NotSupportedError(
                msg='Not supported for metadata statements', errno=255001
            )
        if self._table is None:
            raise Refused('fetch before execute')
        return self._table.to_pandas()


def _databricks_connection(warehouse: Warehouse) -> Any:
    from databricks.sql.client import Connection

    class StubDatabricksConnection(Connection):
        def __init__(self) -> None:
            self.stub = warehouse

        def cursor(self, *args: Any, **kwargs: Any) -> Any:
            _bind(Connection.cursor, *args, **kwargs)
            return _DatabricksCursor(self)

        def close(self) -> None:
            pass

        def __del__(self) -> None:
            pass

    return StubDatabricksConnection()


def _snowflake_connection(warehouse: Warehouse) -> Any:
    from snowflake.connector import SnowflakeConnection

    class StubSnowflakeConnection(SnowflakeConnection):
        def __init__(self) -> None:
            self.stub = warehouse
            self._paramstyle = 'pyformat'

        def cursor(self, *args: Any, **kwargs: Any) -> Any:
            return _SnowflakeCursor(self)

        def close(self) -> None:
            pass

        def __del__(self) -> None:
            pass

    return StubSnowflakeConnection()


class _Row(tuple):
    pass


class _Frame:
    def __init__(self, table: pa.Table) -> None:
        self._table = table

    def toArrow(self) -> pa.Table:
        return self._table

    def collect(self) -> list[_Row]:
        columns = [c.to_pylist() for c in self._table.columns]
        return [_Row(r) for r in zip(*columns, strict=True)]

    def first(self) -> _Row | None:
        rows = self.collect()
        return rows[0] if rows else None

    @property
    def columns(self) -> list[str]:
        return list(self._table.column_names)


def _spark_session(warehouse: Warehouse) -> Any:
    class SparkSession:
        version = '4.0.0'

        def sql(self, sqlQuery: str, args: Any = None, **kwargs: Any) -> _Frame:
            if kwargs:
                raise Refused(f'unexpected spark.sql arguments {kwargs!r}')
            warehouse.statements.append(
                Statement(sqlQuery, _plain_params(args))
            )
            return _Frame(warehouse.answer(sqlQuery, args))

    SparkSession.__module__ = 'pyspark.stub'
    return SparkSession()


def source(warehouse: Warehouse) -> Any:
    r"""A ``DatabricksSource`` or ``SnowflakeSource`` over the stub."""
    from sdm_connectors import DatabricksSource, SnowflakeSource

    if warehouse.kind == 'databricks':
        return DatabricksSource(
            _databricks_connection(warehouse), catalog='main', schema='shop'
        )
    if warehouse.kind == 'spark':
        return DatabricksSource(
            _spark_session(warehouse), catalog='main', schema='shop'
        )
    if warehouse.kind == 'snowflake':
        return SnowflakeSource(
            _snowflake_connection(warehouse), database='DB', schema='SHOP'
        )
    raise ValueError(warehouse.kind)


# Running #####################################################################


def query_for(warehouse: Warehouse, query: str) -> str:
    r"""A shop query, with Snowflake's upper-case names."""
    if not warehouse.upper:
        return query
    return re.sub(
        r'\b(users|orders)\.(\*|\w+)',
        lambda m: f'{m.group(1).upper()}.{m.group(2).upper()}',
        query,
    )


def normalize(payload: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(payload))
    metadata = payload.get('metadata')
    if isinstance(metadata, dict) and 'request_id' in metadata:
        metadata['request_id'] = '<request-id>'
    return payload


def build_schema(warehouse: Warehouse) -> Any:
    from sdm_client import Schema

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return Schema.from_source(source(warehouse))


def predict(
    warehouse: Warehouse, questions: Mapping[str, tuple[str, dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    r"""Each question's payloads, gathered through the public API."""
    from sdm_client import PredictiveQuery
    from sdm_client.adapters.relational import RelationalAdapter

    schema = build_schema(warehouse)
    out: dict[str, list[dict[str, Any]]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        for name, (query, options) in questions.items():
            request = PredictiveQuery(
                query_for(warehouse, query), schema
            ).materialize(**options)
            out[name] = [
                normalize(p)
                for p in RelationalAdapter().prediction_plan(
                    request, ('prediction',)
                )
            ]
    return out


def statements_json(warehouse: Warehouse) -> list[dict[str, Any]]:
    return [{'sql': s.sql, 'params': s.params} for s in warehouse.statements]
