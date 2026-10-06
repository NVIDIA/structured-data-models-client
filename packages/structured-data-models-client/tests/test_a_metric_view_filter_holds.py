# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A metric view's ``filter`` holds for every row a request is gathered from.

The view's rows are its source table's rows the filter keeps. Every statement
that reads the source table must therefore read it through the filter, from
describing the schema to the last neighbour sampled. The warehouse is faked
with frames: it records each statement and answers the shapes the Databricks
sampler sends.
"""

from __future__ import annotations

import json
import re
import warnings
from typing import Any

import pandas as pd
import pytest

pytest.importorskip('sdm_relational_engine.relationallib')
pytest.importorskip('databricks.sql')
pa = pytest.importorskip('pyarrow')
pytest.importorskip('yaml')

from databricks.sql.client import Connection  # noqa: E402
from sdm_connectors import DatabricksSource  # noqa: E402

from sdm_client import PredictiveQuery, Schema  # noqa: E402

_FILTER = "source.status = 'VALID'"
_FACT = '`cat`.`sch`.`order_lines`'
_FILTERED = f'(SELECT * FROM {_FACT} AS source WHERE ({_FILTER}))'
_CUSTOMERS = '`cat`.`sch`.`customers`'

_VIEW = f"""
version: 0.1
source: cat.sch.order_lines
filter: {_FILTER}
joins:
  - name: customers
    source: cat.sch.customers
    on: source.customer_id = customers.customer_id
dimensions:
  - name: line_id
    expr: source.line_id
  - name: ordered_at
    expr: source.ordered_at
  - name: quantity
    expr: source.quantity
  - name: segment
    expr: customers.segment
measures:
  - name: total_quantity
    expr: SUM(source.quantity)
"""

_TYPES = {
    'order_lines': {
        'line_id': 'INT',
        'customer_id': 'INT',
        'ordered_at': 'TIMESTAMP',
        'quantity': 'INT',
        'status': 'STRING',
    },
    'customers': {'customer_id': 'INT', 'segment': 'STRING'},
}

_ORDERS = pd.DataFrame(
    {
        'line_id': range(1, 121),
        'customer_id': [1 + i % 6 for i in range(120)],
        'ordered_at': pd.date_range('2024-01-01', periods=120, freq='D'),
        'quantity': [1 + i % 4 for i in range(120)],
        # Every third line is void, and the view leaves it out:
        'status': ['VOID' if i % 3 == 0 else 'VALID' for i in range(120)],
    }
)
_KEPT = _ORDERS[_ORDERS['status'] == 'VALID']
_RELATIONS = {
    _FILTERED: _KEPT,
    _CUSTOMERS: pd.DataFrame(
        {'customer_id': range(1, 7), 'segment': list('ababab')}
    ),
}


def _describe(definition: str) -> list[tuple]:
    return [
        ('line_id', 'int', None),
        ('ordered_at', 'timestamp', None),
        ('quantity', 'int', None),
        ('segment', 'string', None),
        ('total_quantity', 'bigint measure', None),
        ('', '', None),
        ('# Detailed Table Information', '', None),
        ('Catalog', 'cat', None),
        ('Database', 'sch', None),
        ('Type', 'METRIC_VIEW', None),
        ('View Text', definition, None),
    ]


def _names(projection: str) -> list[str]:
    r"""The column each item of a SELECT list is named, its last quoted
    name: ``\`a\``` and ``expr AS \`a\``` alike.
    """
    items = re.split(r',\s*(?=[^,]*`)', projection)
    return [re.findall(r'`((?:[^`]|``)+)`', item)[-1] for item in items]


class _Cursor:
    def __init__(self, warehouse: _Warehouse) -> None:
        self._warehouse = warehouse
        self._rows: list = []
        self._arrow: Any = None

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, parameters: Any = None) -> None:
        self._warehouse.statements.append(sql)
        self._rows, self._arrow = [], None
        sql = sql.strip()
        if sql.upper().startswith('DESCRIBE TABLE EXTENDED'):
            self._rows = _describe(self._warehouse.definition)
            return
        if 'information_schema.columns' in sql:
            table = str(parameters[1]).lower()
            self._rows = [
                (name, dtype, 'YES', 'cat', 'sch', table, 'MANAGED')
                for name, dtype in _TYPES.get(table, {}).items()
            ]
            return
        if 'information_schema' in sql:
            return

        if 'as table_index' in sql:
            for part in sql.split('UNION ALL'):
                index = int(re.search(r'(\d+) as table_index', part)[1])
                column = re.search(r'MIN\(`(\w+)`\)', part)[1]
                values = self._relation(part)[column]
                self._rows.append((index, values.min(), values.max()))
            return

        frame = self._relation(sql)
        if sql.upper().startswith('SELECT COUNT(*)'):
            self._rows = [(len(frame),)]
            return
        if sql.startswith('WITH TMP AS'):
            self._arrow = self._lookup(sql, frame, json.loads(parameters[0]))
            return

        projection = sql[len('SELECT ') : sql.index('FROM')].strip()
        if 'posexplode' in sql:
            ids = json.loads(parameters[0])
            key = re.search(r'ON `(\w+)` = __ids__', sql)[1]
            frame = frame[frame[key].isin(ids)]
        if limit := re.search(r'LIMIT (\d+)\s*$', sql):
            frame = frame.head(int(limit[1]))
        self._arrow = pa.Table.from_pandas(
            frame[_names(projection)], preserve_index=False
        )

    @staticmethod
    def _relation(sql: str) -> pd.DataFrame:
        found = [name for name in _RELATIONS if name in sql]
        assert len(found) == 1, f'reads no single known relation: {sql}'
        return _RELATIONS[found[0]]

    @staticmethod
    def _lookup(sql: str, frame: pd.DataFrame, rows: list) -> Any:
        r"""A `_by_pkey`, `_by_fkey` or `_by_time` statement: the rows of
        ``frame`` each requested id joins, numbered by its position.
        """
        start = 'as __KUMO_BATCH__, '
        projection = sql[sql.index(start) + len(start) : sql.index('FROM TMP')]
        key = re.search(r'ON `(\w+)` = TMP.__KUMO_ID__', sql)[1]
        time = re.search(r'AND `(\w+)` <= TMP.__KUMO_END_TIME__', sql)
        most = re.search(r'\) <= (\d+)\n', sql)
        parts = []
        for position, row in enumerate(rows):
            if not isinstance(row, dict):
                row = {'id': row}
            match = frame[frame[key] == row['id']]
            if time is not None and 'e' in row:
                at = match[time[1]]
                match = match[at <= pd.Timestamp(row['e'])]
            if time is not None and 's' in row:
                at = match[time[1]]
                match = match[at > pd.Timestamp(row['s'])]
            if most is not None and time is not None:
                match = match.sort_values(time[1], ascending=False)
                match = match.head(int(most[1]))
            parts.append(
                match[_names(projection)].assign(__KUMO_BATCH__=position)
            )
        out = pd.concat(parts, ignore_index=True)
        out = out[['__KUMO_BATCH__', *_names(projection)]]
        return pa.Table.from_pandas(out, preserve_index=False)

    def fetchall(self) -> list:
        return self._rows

    def fetchone(self) -> tuple:
        return self._rows[0]

    def fetchall_arrow(self) -> Any:
        return self._arrow


class _Warehouse(Connection):
    def __init__(self, definition: str = _VIEW) -> None:
        self.definition = definition
        self.statements: list[str] = []

    def cursor(self) -> _Cursor:  # type: ignore[override]
        return _Cursor(self)

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        pass


def test_every_read_of_the_source_table_goes_through_the_filter() -> None:
    warehouse = _Warehouse()
    source = DatabricksSource(warehouse, catalog='cat', schema='sch')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        schema = Schema.from_source(source.semantic_model('sales_mv'))
        request = PredictiveQuery(
            'PREDICT SUM(order_lines.quantity, 0, 30, days) '
            'FOR EACH customers.customer_id',
            schema,
        ).materialize(indices=[1, 2, 3], random_seed=7)

    reads = [sql for sql in warehouse.statements if _FACT in sql]
    assert len(reads) >= 5, warehouse.statements
    shapes = {
        'preview': 'LIMIT 1000',
        'count': 'SELECT COUNT(*)',
        'time range': 'as table_index',
        'lookup': 'WITH TMP AS',
    }
    for shape, marker in shapes.items():
        assert any(marker in sql for sql in reads), f'no {shape} was read'
    for sql in reads:
        assert sql.count(_FACT) == sql.count(_FILTERED), sql

    # Only kept rows reach the request, as context and as the question:
    sent: set[int] = set()
    for payload in request.payloads:
        for place in ('context', 'predict'):
            table = payload.get(place, {}).get('related_tables', {})
            if 'order_lines' in table:
                lines = table['order_lines']
                at = lines['columns'].index('line_id')
                sent |= {row[at] for row in lines['rows']}
    assert sent and sent <= set(_KEPT['line_id'])


def test_a_filter_on_a_joined_table_is_refused_with_the_workaround() -> None:
    warehouse = _Warehouse(_VIEW.replace(_FILTER, "customers.segment = 'a'"))
    source = DatabricksSource(warehouse, catalog='cat', schema='sch')

    with pytest.raises(ValueError) as caught:
        Schema.from_source(source.semantic_model('sales_mv'))

    message = str(caught.value)
    assert "the joined table 'customers'" in message
    assert (
        'Create a regular view with this WHERE clause and point the metric '
        "view's source at it"
    ) in message
    assert not any(_FACT in sql for sql in warehouse.statements)
