# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Databricks and Snowflake push-down predictions over stub drivers.

The live warehouses have no recordings, so their SQL is pinned here instead:
a stub connection that subclasses the real driver's connection class answers
from a small scripted shop, and a prediction runs through the public API
(``DatabricksSource`` or ``SnowflakeSource``, ``Schema.from_source``,
``PredictiveQuery.materialize``, the payloads). Every statement the driver is
handed is logged with its parameters, the parameter style the connection had
at that moment, and the code that handed it over.

The Spark path runs the same prediction through a stand-in notebook session,
whose ``sql`` method takes the statements.

The answers only have to be well formed and repeatable: a statement this
module has no answer for raises, which fails the run.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
from _seam import call_path

HERE = Path(__file__).resolve().parent

# The shop.

_TIMES = pd.date_range('2024-01-01 08:00', periods=60, freq='37h')

USERS = pa.table(
    {
        'user_id': pa.array(range(1, 13), pa.int64()),
        'country': ['DE', 'FR', 'DE', 'US', 'US', 'FR'] * 2,
        'age': pa.array([31, 45, 27, 38, 52, 23] * 2, pa.int64()),
    }
)
ORDERS = pa.table(
    {
        'order_id': pa.array(range(100, 160), pa.int64()),
        'user_id': pa.array([(i * 7) % 12 + 1 for i in range(60)], pa.int64()),
        'amount': [round(1.5 + (i * 13) % 17 * 0.75, 2) for i in range(60)],
        'ordered_at': pa.array(_TIMES, pa.timestamp('us')),
    }
)
TABLES = {'users': USERS, 'orders': ORDERS}
KEYS = {'users': 'user_id', 'orders': 'order_id'}
TIME = {'orders': 'ordered_at'}
LINKS = {'orders': ('user_id', 'users', 'user_id')}

DATABRICKS_TYPES = {
    pa.int64(): 'bigint',
    pa.float64(): 'double',
    pa.string(): 'string',
    pa.timestamp('us'): 'timestamp_ntz',
}
SNOWFLAKE_TYPES = {
    pa.int64(): 'NUMBER(38,0)',
    pa.float64(): 'FLOAT',
    pa.string(): 'VARCHAR(16777216)',
    pa.timestamp('us'): 'TIMESTAMP_NTZ(9)',
}

# The questions asked of each warehouse, with the options they are asked with.
QUESTIONS: dict[str, tuple[str, dict[str, Any]]] = {
    'count': (
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id',
        {'indices': [1, 2, 3], 'random_seed': 7},
    ),
    'sum-entity-sample': (
        'PREDICT SUM(orders.amount, 0, 60, days) FOR EACH users.user_id',
        {'indices': [4, 5], 'num_neighbors': [3, 2], 'random_seed': 11},
    ),
}


class Refused(Exception):
    r"""What a stub raises for SQL it has no answer for."""


def _flat(sql: str) -> str:
    r"""A statement on one line, without comments (the dialect probe's
    marker among them)."""
    return ' '.join(re.sub(r'/\*.*?\*/', ' ', sql, flags=re.DOTALL).split())


def _upper(table: pa.Table) -> pa.Table:
    return table.rename_columns([c.upper() for c in table.column_names])


def _shop(upper: bool) -> dict[str, pa.Table]:
    return {n: _upper(t) if upper else t for n, t in TABLES.items()}


def _name(table: pa.Table, name: str) -> str:
    known = {c.lower(): c for c in table.column_names}
    return known[name.lower()]


def _source(sql: str) -> str:
    r"""The shop table a statement reads from."""
    found = re.search(
        r'(?:FROM|JOIN)\s+[^\s]*?[`"]?\b(users|orders)\b[`"]?(?=[\s)]|$)',
        sql,
        re.IGNORECASE,
    )
    if found is None:
        raise Refused(sql)
    return found.group(1).lower()


def _projected(text: str, table: pa.Table) -> list[str]:
    r"""The shop columns a select list names, in order."""
    names = re.findall(r'[`"](\w+)[`"](?![.\w`"])', text)
    known = {c.lower(): c for c in table.column_names}
    out: list[str] = []
    for name in names:
        if name.lower() in known and known[name.lower()] not in out:
            out.append(known[name.lower()])
    return out


def _select_list(sql: str) -> str:
    head = sql.split(' FROM ', 1)[0]
    terms = re.findall(r'AS (__kumo_order_\d+__)', head)
    if terms:
        head = head.rsplit(terms[-1], 1)[1]
    return head


def _sample(sql: str, shop: dict[str, pa.Table]) -> pa.Table:
    r"""An inference sample: the order terms first, then the columns."""
    data = shop[_source(sql)]
    head = sql.split(' FROM ', 1)[0]
    terms = re.findall(r'AS (__kumo_order_\d+__)', head)
    names = _projected(_select_list(sql), data)
    out: dict[str, Any] = {
        t: pa.array([0] * data.num_rows, pa.int64()) for t in terms
    }
    out.update({n: data.column(n) for n in names})
    return pa.table(out)


def _min_max(sql: str, shop: dict[str, pa.Table]) -> pa.Table:
    rows = []
    for select in sql.split('UNION ALL'):
        found = re.search(r'(\d+) as table_index', select)
        if found is None:
            raise Refused(sql)
        data = shop[_source(select)]
        values = data.column(_name(data, 'ordered_at')).to_pylist()
        rows.append((int(found.group(1)), min(values), max(values)))
    return pa.table(
        {
            'table_index': pa.array([r[0] for r in rows], pa.int64()),
            'min_date': pa.array([r[1] for r in rows], pa.timestamp('us')),
            'max_date': pa.array([r[2] for r in rows], pa.timestamp('us')),
        }
    )


def _key_stats(sql: str, shop: dict[str, pa.Table]) -> pa.Table:
    r"""``SELECT k, COUNT(*) AS n, COUNT(DISTINCT c) AS d, ... z`` per part."""
    out: dict[str, list[int]] = {'k': [], 'n': [], 'd': [], 'z': []}
    for select in sql.split('UNION ALL'):
        found = re.search(
            r'SELECT (\d+) AS k, COUNT\(\*\) AS n, COUNT\(DISTINCT [`"]?(\w+)',
            select,
        )
        if found is None:
            raise Refused(sql)
        data = shop[_source(select)]
        values = data.column(_name(data, found.group(2))).to_pylist()
        out['k'].append(int(found.group(1)))
        out['n'].append(len(values))
        out['d'].append(len({v for v in values if v is not None}))
        out['z'].append(sum(v is None for v in values))
    return pa.table({k: pa.array(v, pa.int64()) for k, v in out.items()})


def _time_value(text: Any) -> pd.Timestamp | None:
    r"""A window bound as the naive UTC time the shop holds: a bound naming
    a UTC instant (``+00:00``) is the same time."""
    if text is None:
        return None
    value = pd.Timestamp(str(text))
    if value.tzinfo is not None:
        value = value.tz_convert('UTC').tz_localize(None)
    return value


def _by_ids(
    sql: str,
    rows: list[Any],
    shop: dict[str, pa.Table],
    batch_name: str,
    bounds: list[Any],
) -> pa.Table:
    r"""The rows holding each id, with the id's position as its batch.

    A key list of objects carries each id's window: ``e`` the end, inclusive,
    ``s`` the start, exclusive. ``QUALIFY ... <= n`` keeps the latest ``n``.
    """
    flat = _flat(sql)
    joined = flat.split(' JOIN ', 1)[1]
    table = _source('FROM ' + joined)
    data = shop[table]
    selected = flat.split(' FROM TMP')[0]
    names = _projected(selected.split('__KUMO_BATCH__,', 1)[-1], data)
    on = re.search(r'ON [^\s]*?[`"](\w+)[`"] = TMP\.__KUMO_ID__', flat)
    if on is None:
        raise Refused(sql)
    key = _name(data, on.group(1))
    keys = data.column(key).to_pylist()
    time_name = next(
        (c for c in data.column_names if c.lower() == 'ordered_at'), None
    )
    times = data.column(time_name).to_pylist() if time_name else None
    limit = re.search(r'\) <= (\d+) ORDER BY', flat)
    batches: list[int] = []
    picked: list[int] = []
    for pos, item in enumerate(rows):
        if isinstance(item, dict):
            ident, end, start = item['id'], item.get('e'), item.get('s')
        elif isinstance(item, list):
            ident, end, start = (item + [None, None])[:3]
        else:
            ident, end, start = item, None, None
        end, start = _time_value(end), _time_value(start)
        hits = [i for i, k in enumerate(keys) if k == ident]
        if times is not None:
            if end is not None:
                hits = [i for i in hits if pd.Timestamp(times[i]) <= end]
            if start is not None:
                hits = [i for i in hits if pd.Timestamp(times[i]) > start]
            hits.sort(key=lambda i: (pd.Timestamp(times[i]), i), reverse=True)
        if limit is not None:
            hits = hits[: int(limit.group(1))]
        batches += [pos] * len(hits)
        picked += hits
    out: dict[str, Any] = {batch_name: pa.array(batches, pa.int64())}
    for name in names:
        out[name] = data.column(name).take(pa.array(picked, pa.int64()))
    return pa.table(out)


def _entity_rows(
    sql: str, shop: dict[str, pa.Table], ids: list[Any] | None
) -> pa.Table:
    r"""An entity sample: every row, or the rows holding ``ids``, in order."""
    flat = _flat(sql)
    data = shop[_source(flat)]
    names = _projected(flat.split(' FROM ', 1)[0], data)
    key = next(c for c in data.column_names if c.lower() == 'user_id')
    if ids is None:
        limit = re.search(r'LIMIT (\d+)', flat)
        count = data.num_rows if limit is None else int(limit.group(1))
        index = list(range(min(count, data.num_rows)))
    else:
        keys = data.column(key).to_pylist()
        index = [keys.index(i) for i in ids if i in keys]
    return pa.table(
        {n: data.column(n).take(pa.array(index, pa.int64())) for n in names}
    )


def databricks_answer(sql: str, params: list[Any] | None) -> pa.Table:
    flat = _flat(sql)
    shop = _shop(upper=False)
    params = list(params or [])
    if flat.startswith('SELECT current_catalog(), current_schema()'):
        return pa.table({'c': ['main'], 's': ['shop']})
    if 'information_schema.tables' in flat:
        return pa.table({'table_name': sorted(TABLES)})
    if 'information_schema.columns' in flat:
        schema, table = (str(p).lower() for p in params[:2])
        data = TABLES[table]
        n = data.num_columns
        return pa.table(
            {
                'column_name': data.column_names,
                'full_data_type': [
                    DATABRICKS_TYPES[f.type] for f in data.schema
                ],
                'is_nullable': [
                    'NO' if c == KEYS[table] else 'YES'
                    for c in data.column_names
                ],
                'table_catalog': ['main'] * n,
                'table_schema': [schema] * n,
                'table_name': [table] * n,
            }
        )
    if "constraint_type = 'FOREIGN KEY'" in flat:
        table = str(params[1]).lower()
        link = LINKS.get(table)
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
        return pa.table(
            {
                'constraint_type': ['PRIMARY KEY'],
                'constraint_name': [f'{table}_pk'],
                'column_name': [KEYS[table]],
            }
        )
    if flat.startswith('SELECT COUNT(*) FROM'):
        return pa.table({'n': pa.array([shop[_source(flat)].num_rows])})
    if 'COUNT(DISTINCT' in flat and ' AS k,' in flat:
        return _key_stats(flat, shop)
    if '__kumo_order_' in flat:
        return _sample(flat, shop)
    if 'as table_index' in flat:
        return _min_max(flat, shop)
    if flat.startswith('WITH TMP AS'):
        return _by_ids(
            flat, json.loads(params[0]), shop, '__KUMO_BATCH__', params[1:]
        )
    if 'posexplode' in flat:
        return _entity_rows(flat, shop, json.loads(params[0]))
    if 'TABLESAMPLE' in flat:
        return _entity_rows(flat, shop, None)
    raise Refused(sql)


def snowflake_answer(sql: str, params: list[Any] | None) -> pa.Table:
    flat = _flat(sql)
    shop = _shop(upper=True)
    params = list(params or [])
    if flat.startswith('SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()'):
        return pa.table({'d': ['DB'], 's': ['SHOP']})
    if 'INFORMATION_SCHEMA.TABLES' in flat:
        return pa.table({'TABLE_NAME': ['ORDERS', 'USERS']})
    if flat.startswith('DESCRIBE TABLE'):
        found = re.fullmatch(r'DESCRIBE TABLE "(\w+)"\."(\w+)"\."(\w+)"', flat)
        if found is None:
            raise Refused(sql)
        table = found.group(3).lower()
        data = shop[table]
        key = KEYS[table].upper()
        n = data.num_columns
        return pa.table(
            {
                'name': data.column_names,
                'type': [SNOWFLAKE_TYPES[f.type] for f in data.schema],
                'kind': ['COLUMN'] * n,
                'null?': ['N' if c == key else 'Y' for c in data.column_names],
                'default': pa.array([None] * n, pa.string()),
                'primary key': [
                    'Y' if c == key else 'N' for c in data.column_names
                ],
                'unique key': ['N'] * n,
            }
        )
    if flat.startswith('SHOW IMPORTED KEYS'):
        found = re.search(r'"(\w+)"$', flat)
        table = found.group(1).lower() if found else ''
        link = LINKS.get(table)
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
    if flat.startswith('SHOW TABLES IN SCHEMA') or flat.startswith(
        'SHOW TABLES IN'
    ):
        return pa.table(
            {
                'created_on': ['2024-01-01', '2024-01-01'],
                'name': ['ORDERS', 'USERS'],
                'kind': ['TABLE', 'TABLE'],
                'database_name': ['DB', 'DB'],
                'schema_name': ['SHOP', 'SHOP'],
                'is_hybrid': ['N', 'N'],
            }
        )
    if flat.startswith('SHOW TABLES LIKE'):
        table = str(params[0]).lower()
        return pa.table(
            {
                'created_on': ['2024-01-01'],
                'name': [table.upper()],
                'kind': ['TABLE'],
                'database_name': ['DB'],
                'schema_name': ['SHOP'],
                'comment': [''],
                'cluster_by': [''],
                'rows': pa.array([shop[table].num_rows], pa.int64()),
            }
        )
    if flat.startswith('SELECT COUNT(*) FROM'):
        return pa.table({'N': pa.array([shop[_source(flat)].num_rows])})
    if 'COUNT(DISTINCT' in flat and ' AS k,' in flat:
        return _key_stats(flat, shop)
    if '__kumo_order_' in flat:
        return _sample(flat, shop)
    if 'as table_index' in flat:
        return _min_max(flat, shop)
    if 'FLATTEN' in flat and flat.upper().startswith('WITH TMP AS'):
        return _by_ids(
            flat, json.loads(params[0]), shop, '__KUMO_BATCH__', params[1:]
        )
    if 'FLATTEN' in flat:
        return _entity_rows(flat, shop, json.loads(params[0]))
    if ' IN (' in flat:
        return _entity_rows(flat, shop, list(params))
    if flat.startswith('SELECT') and ' FROM ' in flat:
        return _entity_rows(flat, shop, None)
    raise Refused(sql)


# A Databricks metric view over the shop: ``orders`` is its source, ``users``
# a join, and its filter keeps the orders above a minimum amount.

METRIC_VIEW_FILTER = 'source.amount > 3.0'
METRIC_VIEW = f"""
version: 0.1
source: main.shop.orders
filter: {METRIC_VIEW_FILTER}
joins:
  - name: users
    source: main.shop.users
    on: source.user_id = users.user_id
dimensions:
  - name: order_id
    expr: source.order_id
  - name: user_id
    expr: source.user_id
  - name: amount
    expr: source.amount
  - name: ordered_at
    expr: source.ordered_at
  - name: country
    expr: users.country
measures:
  - name: total_amount
    expr: SUM(source.amount)
"""
_FILTERED_ORDERS = (
    '(SELECT * FROM `main`.`shop`.`orders` AS source '
    f'WHERE ({METRIC_VIEW_FILTER}))'
)


def _metric_view_shop() -> dict[str, pa.Table]:
    orders = ORDERS.filter(pc.greater(ORDERS.column('amount'), 3.0))
    return {'users': USERS, 'orders': orders}


def metric_view_answer(sql: str, params: list[Any] | None) -> pa.Table:
    flat = _flat(sql)
    if flat.startswith('DESCRIBE TABLE EXTENDED'):
        rows = [
            ('order_id', 'bigint'),
            ('user_id', 'bigint'),
            ('amount', 'double'),
            ('ordered_at', 'timestamp_ntz'),
            ('country', 'string'),
            ('total_amount', 'double measure'),
            ('', ''),
            ('# Detailed Table Information', ''),
            ('Catalog', 'main'),
            ('Database', 'shop'),
            ('Type', 'METRIC_VIEW'),
            ('View Text', METRIC_VIEW),
        ]
        return pa.table(
            {
                'col_name': [r[0] for r in rows],
                'data_type': [r[1] for r in rows],
                'comment': pa.array([None] * len(rows), pa.string()),
            }
        )
    if 'information_schema' in flat and 'columns' in flat:
        return databricks_answer(sql, params)
    if 'information_schema' in flat:
        return databricks_answer(sql, params)
    if '`main`.`shop`.`orders`' in flat and _FILTERED_ORDERS not in flat:
        raise Refused(f'reads orders around the view filter: {sql}')
    shop = _metric_view_shop()
    if flat.startswith('SELECT COUNT(*) FROM'):
        return pa.table({'n': pa.array([shop[_source(flat)].num_rows])})
    if 'COUNT(DISTINCT' in flat and ' AS k,' in flat:
        return _key_stats(flat, shop)
    if '__kumo_order_' in flat:
        return _sample(flat, shop)
    if 'as table_index' in flat:
        return _min_max(flat, shop)
    if flat.startswith('WITH TMP AS'):
        return _by_ids(
            flat, json.loads(params[0]), shop, '__KUMO_BATCH__', params[1:]
        )
    if 'posexplode' in flat:
        return _entity_rows(flat, shop, json.loads(params[0]))
    if 'TABLESAMPLE' in flat:
        return _entity_rows(flat, shop, None)
    raise Refused(sql)


# A Snowflake semantic view over the shop.

SEMANTIC_VIEW = """
tables:
  - name: USERS
    base_table:
      database: DB
      schema: SHOP
      table: USERS
    primary_key:
      columns:
        - USER_ID
    dimensions:
      - name: COUNTRY
        expr: users.COUNTRY
        data_type: VARCHAR(16777216)
      - name: AGE
        expr: USERS.AGE
        data_type: NUMBER(38,0)
  - name: ORDERS
    base_table:
      database: DB
      schema: SHOP
      table: ORDERS
    primary_key:
      columns:
        - ORDER_ID
    dimensions:
      - name: USER_ID
        expr: orders.USER_ID
        data_type: NUMBER(38,0)
      - name: ORDERED_AT
        expr: orders.ORDERED_AT
        data_type: TIMESTAMP_NTZ(9)
    facts:
      - name: AMOUNT
        expr: orders.AMOUNT
        data_type: FLOAT
relationships:
  - name: ORDERS_TO_USERS
    left_table: ORDERS
    right_table: USERS
    relationship_columns:
      - left_column: USER_ID
        right_column: USER_ID
"""


def semantic_view_answer(sql: str, params: list[Any] | None) -> pa.Table:
    if 'SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW' in _flat(sql):
        if list(params or []) != ['DB.SHOP.SHOP_SEMANTICS']:
            raise Refused(f'no semantic view {params!r}')
        return pa.table({'y': [SEMANTIC_VIEW]})
    return snowflake_answer(sql, params)


# What the driver was handed.


@dataclass
class Statement:
    sql: str
    params: list[Any] | None
    paramstyle: str | None
    caller: str
    entry: str | None


@dataclass
class Log:
    statements: list[Statement] = field(default_factory=list)


def _plain_params(params: Any) -> list[Any] | None:
    # No parameters, however a connector spells it: None, [] or ().
    if params is None or (not isinstance(params, dict) and len(params) == 0):
        return None
    if isinstance(params, dict):
        raise Refused(f'named parameters are not expected: {params!r}')
    return [_plain(p) for p in params]


def _plain(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, 'isoformat'):
        return value.isoformat()
    if hasattr(value, 'item'):
        return value.item()
    return value


def _bind(method: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    r"""A call's arguments, bound as the real driver's method binds them."""
    return dict(inspect.signature(method).bind(None, *args, **kwargs).arguments)


class _Cursor:
    real: Any

    def __init__(self, connection: Any) -> None:
        self._connection = connection
        self._table: pa.Table | None = None
        self.description: list[tuple[Any, ...]] | None = None
        self.rowcount = -1

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        pass

    def _run(self, sql: str, params: Any) -> None:
        caller, entry = call_path()
        self._table = None
        self.description = None
        self._connection.log.statements.append(
            Statement(
                sql=sql,
                params=_plain_params(params),
                paramstyle=getattr(self._connection, '_paramstyle', None),
                caller=caller,
                entry=entry,
            )
        )
        self._table = self._connection.answer(sql, params)
        self.description = [
            (f.name, self._connection.type_code(f.type)) + (None,) * 5
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
        # As the connector does: with a client-side style, ``?`` is not a
        # placeholder, so a bound value would never reach the server.
        if params and self._connection._paramstyle not in ('qmark', 'numeric'):
            raise Refused(
                f'parameters bound with paramstyle '
                f'{self._connection._paramstyle!r}: {sql}'
            )
        self._statement = _flat(sql)
        self._run(sql, params)
        return self

    def fetch_arrow_all(self, *args: Any, **kwargs: Any) -> pa.Table | None:
        bound = _bind(self.real.fetch_arrow_all, *args, **kwargs)
        if self._statement.startswith(('SHOW', 'DESC')):
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
        if self._statement.startswith(('SHOW', 'DESC')):
            from snowflake.connector.errors import NotSupportedError

            raise NotSupportedError(
                msg='Not supported for metadata statements', errno=255001
            )
        if self._table is None:
            raise Refused('fetch before execute')
        return self._table.to_pandas()


def databricks_connection(
    log: Log, answer: Callable[[str, Any], pa.Table] = databricks_answer
) -> Any:
    from databricks.sql.client import Connection

    class StubDatabricksConnection(Connection):
        def __init__(self) -> None:
            self.log = log
            self.answer = answer

        def type_code(self, data_type: pa.DataType) -> Any:
            return DATABRICKS_TYPES[data_type]

        def cursor(self, *args: Any, **kwargs: Any) -> Any:
            _bind(Connection.cursor, *args, **kwargs)
            return _DatabricksCursor(self)

        def close(self) -> None:
            pass

        def __del__(self) -> None:
            pass

    return StubDatabricksConnection()


def snowflake_connection(
    log: Log, answer: Callable[[str, Any], pa.Table] = snowflake_answer
) -> Any:
    from snowflake.connector import SnowflakeConnection

    class StubSnowflakeConnection(SnowflakeConnection):
        def __init__(self) -> None:
            self.log = log
            self.answer = answer
            self._paramstyle = 'pyformat'

        def type_code(self, data_type: pa.DataType) -> Any:
            # The connector's codes: FIXED, REAL, TEXT, TIMESTAMP_NTZ.
            return {
                pa.int64(): 0,
                pa.float64(): 1,
                pa.string(): 2,
                pa.timestamp('us'): 8,
            }.get(data_type, 2)

        def cursor(self, *args: Any, **kwargs: Any) -> Any:
            return _SnowflakeCursor(self)

        def close(self) -> None:
            pass

        def __del__(self) -> None:
            pass

    return StubSnowflakeConnection()


# A notebook's Spark session, recognised by name as ``sdm_connectors`` does.


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


def spark_session(log: Log) -> Any:
    namespace: dict[str, Any] = {}

    class SparkSession:
        version = '4.0.0'

        def sql(self, sqlQuery: str, args: Any = None, **kwargs: Any) -> _Frame:
            if kwargs:
                raise Refused(f'unexpected spark.sql arguments {kwargs!r}')
            caller, entry = call_path()
            log.statements.append(
                Statement(
                    sql=sqlQuery,
                    params=_plain_params(args),
                    paramstyle=None,
                    caller=caller,
                    entry=entry,
                )
            )
            return _Frame(databricks_answer(sqlQuery, args))

    SparkSession.__module__ = 'pyspark.stub'
    namespace['SparkSession'] = SparkSession
    return SparkSession()


# Running a prediction.

WAREHOUSES = (
    'databricks',
    'databricks-spark',
    'databricks-metric-view',
    'snowflake',
    'snowflake-semantic-view',
)


def source_for(warehouse: str, log: Log) -> Any:
    from sdm_connectors import DatabricksSource, SnowflakeSource

    if warehouse == 'databricks':
        return DatabricksSource(
            databricks_connection(log), catalog='main', schema='shop'
        )
    if warehouse == 'databricks-spark':
        return DatabricksSource(
            spark_session(log), catalog='main', schema='shop'
        )
    if warehouse == 'databricks-metric-view':
        source = DatabricksSource(
            databricks_connection(log, metric_view_answer),
            catalog='main',
            schema='shop',
        )
        return source.semantic_model('shop_metrics')
    if warehouse == 'snowflake':
        return SnowflakeSource(
            snowflake_connection(log), database='DB', schema='SHOP'
        )
    if warehouse == 'snowflake-semantic-view':
        source = SnowflakeSource(
            snowflake_connection(log, semantic_view_answer),
            database='DB',
            schema='SHOP',
        )
        return source.semantic_model('SHOP_SEMANTICS')
    raise ValueError(warehouse)


def _query_for(warehouse: str, query: str) -> str:
    if not warehouse.startswith('snowflake'):
        return query
    for name in ('orders.amount', 'orders.*', 'users.user_id'):
        table, column = name.split('.')
        query = query.replace(
            name, f'{table.upper()}.{column.upper() if column != "*" else "*"}'
        )
    return query


def normalize(payload: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(payload))
    metadata = payload.get('metadata')
    if isinstance(metadata, dict) and 'request_id' in metadata:
        metadata['request_id'] = '<request-id>'
    return payload


def run(warehouse: str) -> dict[str, Any]:
    r"""One warehouse's prediction: each statement and the payloads."""
    from sdm_client import PredictiveQuery, Schema
    from sdm_client.adapters.relational import RelationalAdapter

    log = Log()
    out: dict[str, Any] = {'questions': {}}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        source = source_for(warehouse, log)
        schema = Schema.from_source(source)
        out['schema_warnings'] = _warnings(caught)
        for name, (query, options) in QUESTIONS.items():
            start, seen = len(log.statements), len(caught)
            request = PredictiveQuery(
                _query_for(warehouse, query), schema
            ).materialize(**options)
            payloads = [
                normalize(p)
                for p in RelationalAdapter().prediction_plan(
                    request, ('prediction',)
                )
            ]
            out['questions'][name] = {
                'first_statement': start,
                'payloads': payloads,
                'warnings': _warnings(caught[seen:]),
            }
    out['statements'] = log.statements
    out['connection'] = source.connection
    return out


def _warnings(caught: list[warnings.WarningMessage]) -> list[dict[str, str]]:
    return [
        {
            'category': f'{w.category.__module__}.{w.category.__qualname__}',
            'message': str(w.message),
        }
        for w in caught
        if issubclass(w.category, UserWarning)
    ]


def as_golden(result: dict[str, Any]) -> dict[str, Any]:
    r"""What is pinned: each statement's SQL and parameters, the payloads."""
    return {
        'statements': [
            {'sql': s.sql, 'params': s.params} for s in result['statements']
        ],
        'schema_warnings': result['schema_warnings'],
        'questions': result['questions'],
    }


GOLDEN = HERE / 'golden' / 'warehouse_sql.json'


def write_golden(path: Path = GOLDEN) -> None:
    golden = {w: as_golden(run(w)) for w in WAREHOUSES}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden, indent=1, sort_keys=True) + '\n')


def main(argv: list[str]) -> int:
    if argv[1:] == ['write-golden']:
        write_golden()
        return 0
    if len(argv) == 3 and argv[1] == 'show':
        result = run(argv[2])
        for s in result['statements']:
            print(
                repr(_flat(s.sql))[:240],
                s.params,
                s.paramstyle,
                s.caller,
                s.entry,
            )
        return 0
    print(f'usage: {argv[0]} write-golden | show WAREHOUSE', file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
