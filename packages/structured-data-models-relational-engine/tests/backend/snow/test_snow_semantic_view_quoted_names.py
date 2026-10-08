# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A Snowflake semantic view over tables whose names need quoting.

Snowflake writes a semantic view's YAML with the names it resolved, so a
view over tables created as ``"orders"`` writes ``name: '"orders"'``. Those
names used to be quoted a second time, and the view could not be read:
``Table 'TEST.SHOP."orders"' does not exist``.
"""

import re

import pytest
from sdm_relational_engine.rfm import Graph

try:
    import pyarrow as pa
    import yaml  # noqa: F401
    from sdm_relational_engine.rfm.backend.snow import Connection
except ImportError:
    pytest.skip("'snowflake' extension not installed", allow_module_level=True)

_TABLES = {
    'orders': {
        'order_id': ('NUMBER(38,0)', [1, 2, 3, 4]),
        'user_id': ('NUMBER(38,0)', [1, 1, 2, 2]),
        'amount': ('FLOAT', [1.5, 2.5, 3.5, 4.5]),
    },
    'users': {
        'user_id': ('NUMBER(38,0)', [1, 2]),
        'country': ('VARCHAR(16)', ['DE', 'US']),
    },
}

_YAML = """
name: SHOP_SEMANTICS
tables:
  - name: '"orders"'
    base_table:
      database: TEST
      schema: SHOP
      table: '"orders"'
    primary_key:
      columns:
        - '"order_id"'
    facts:
      - name: '"amount"'
        expr: '"orders"."amount"'
        data_type: FLOAT
    metrics:
      - name: '"total"'
        expr: SUM("orders"."amount")
  - name: '"users"'
    base_table:
      database: TEST
      schema: SHOP
      table: '"users"'
    primary_key:
      columns:
        - '"user_id"'
    dimensions:
      - name: '"country"'
        expr: '"country"'
        data_type: VARCHAR(16)
relationships:
  - name: '"orders_to_users"'
    left_table: '"orders"'
    right_table: '"users"'
    relationship_columns:
      - left_column: '"user_id"'
        right_column: '"user_id"'
"""

_QUOTED = r'"((?:[^"]|"")*)"'


def _top_level_items(text: str) -> list[str]:
    r"""``text`` split at the commas outside parentheses."""
    items, depth, start = [], 0, 0
    for i, char in enumerate(text):
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        elif char == ',' and depth == 0:
            items.append(text[start:i].strip())
            start = i + 1
    items.append(text[start:].strip())
    return items


class _Cursor:
    def __init__(self, statements: list[str]) -> None:
        self._statements = statements
        self._table: pa.Table | None = None
        self.description: list[tuple[str]] = []

    def __enter__(self) -> '_Cursor':
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, parameters: object = None) -> None:
        self._statements.append(sql)
        flat = ' '.join(sql.split())
        if 'SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW' in flat:
            self._table = pa.table({'yaml': [_YAML]})
        elif flat.startswith('DESCRIBE TABLE'):
            found = re.fullmatch(
                rf'DESCRIBE TABLE {_QUOTED}\.{_QUOTED}\.{_QUOTED}', flat
            )
            assert found is not None, flat
            database, schema, table = found.groups()
            if (database, schema) != ('TEST', 'SHOP') or table not in _TABLES:
                raise RuntimeError(
                    f"SQL compilation error: Table '{table}' does not exist"
                )
            columns = _TABLES[table]
            self._table = pa.table(
                {
                    'name': list(columns),
                    'type': [kind for kind, _ in columns.values()],
                    'kind': ['COLUMN'] * len(columns),
                    'null?': ['Y'] * len(columns),
                    'default': [None] * len(columns),
                    'primary key': ['N'] * len(columns),
                    'unique key': ['N'] * len(columns),
                }
            )
        elif flat.startswith('SHOW TABLES'):
            self._table = pa.table({'name': pa.array([], pa.string())})
        elif flat.startswith('SELECT '):
            table = re.search(rf'FROM "TEST"\."SHOP"\.{_QUOTED}', flat)
            assert table is not None, flat
            columns = _TABLES[table.group(1)]
            projection = flat.split(' FROM ')[0][len('SELECT ') :]
            items = _top_level_items(projection)
            out = {}
            for i, item in enumerate(items):
                if ' AS __kumo_order_' in item:
                    out[f'__kumo_order_{i}__'] = [0] * 4
                    continue
                name = re.fullmatch(_QUOTED, item)
                assert name is not None, item
                values = columns[name.group(1)][1]
                out[name.group(1)] = values
            size = min(len(v) for v in out.values())
            self._table = pa.table({k: v[:size] for k, v in out.items()})
        else:
            raise AssertionError(f'unexpected statement: {flat}')
        self.description = [(name,) for name in self._table.column_names]

    def fetch_arrow_all(self, force_return_table: bool = False) -> pa.Table:
        assert self._table is not None
        return self._table


class _Connection(Connection):
    def __init__(self) -> None:
        self._paramstyle = 'pyformat'
        self.statements: list[str] = []

    def cursor(self) -> _Cursor:  # type: ignore[override]
        return _Cursor(self.statements)

    def close(self) -> None:
        pass

    def __del__(self) -> None:
        pass


def test_a_view_of_quoted_names_reads_the_names_they_spell() -> None:
    connection = _Connection()
    graph = Graph.from_snowflake_semantic_view(
        'TEST.SHOP.SHOP_SEMANTICS', connection=connection, verbose=False
    )

    assert sorted(graph.tables) == ['orders', 'users']
    assert graph['orders'].source_name == 'TEST.SHOP.orders'
    assert [c.name for c in graph['orders'].columns] == [
        'amount',
        'order_id',
        'user_id',
    ]
    assert graph['orders']._primary_key == 'order_id'
    assert graph['users']._primary_key == 'user_id'
    assert [tuple(edge) for edge in graph.edges] == [
        ('orders', 'user_id', 'users')
    ]
    assert graph.conversion_messages == ()
    described = sorted(
        sql for sql in connection.statements if sql.startswith('DESCRIBE')
    )
    assert described == [
        'DESCRIBE TABLE "TEST"."SHOP"."orders"',
        'DESCRIBE TABLE "TEST"."SHOP"."users"',
    ]
