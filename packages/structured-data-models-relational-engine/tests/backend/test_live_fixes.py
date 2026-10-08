# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Phase 5.5's fixes on real Snowflake and Databricks warehouses.

Excluded by default; each backend skips unless its credentials are in the
environment (the ``SNOWFLAKE_*`` and ``DATABRICKS_*`` variables
``packages/sdm-connectors/tests/test_live_warehouses.py`` reads). Run with
``-m live_warehouse``. Each test creates its tables, named uniquely, in the
configured schema and drops them afterwards.

* Time windows: a ``TIMESTAMP_LTZ`` or ``TIMESTAMP_TZ`` time column under
  Snowflake's default ``TIMEZONE``, America/Los_Angeles, sends the payloads a
  ``TIMESTAMP_NTZ`` column holding the same instants sends under UTC, and an
  expression time column of either type is read as zoned and compares as the
  instants its bounds name.
* Multi-column links (``DESIGN-composite-link.md`` A1 and its guards): a child
  referencing only parents outside the parent's own inference sample links;
  columns named in the wrong order, values the parent lacks, and a null part
  are refused.
"""

from __future__ import annotations

import os
import uuid
import warnings
from collections.abc import Callable, Iterator
from typing import Any

import pandas as pd
import pytest

pytestmark = pytest.mark.live_warehouse

SNOWFLAKE_KEYS = (
    'SNOWFLAKE_ACCOUNT',
    'SNOWFLAKE_USER',
    'SNOWFLAKE_PASSWORD',
    'SNOWFLAKE_DATABASE',
    'SNOWFLAKE_SCHEMA',
    'SNOWFLAKE_WAREHOUSE',
)
DATABRICKS_KEYS = (
    'DATABRICKS_SERVER_HOSTNAME',
    'DATABRICKS_HTTP_PATH',
    'DATABRICKS_TOKEN',
    'DATABRICKS_CATALOG',
    'DATABRICKS_SCHEMA',
)


def _require(keys: tuple[str, ...], what: str) -> None:
    missing = [key for key in keys if not os.environ.get(key)]
    if missing:
        pytest.skip(f'set {", ".join(missing)} to run against {what}')


@pytest.fixture(scope='module')
def snowflake() -> Iterator[Any]:
    _require(SNOWFLAKE_KEYS, 'a real Snowflake account')
    connector = pytest.importorskip('snowflake.connector')
    arguments = {
        'account': os.environ['SNOWFLAKE_ACCOUNT'],
        'user': os.environ['SNOWFLAKE_USER'],
        'password': os.environ['SNOWFLAKE_PASSWORD'],
        'database': os.environ['SNOWFLAKE_DATABASE'],
        'schema': os.environ['SNOWFLAKE_SCHEMA'],
        'warehouse': os.environ['SNOWFLAKE_WAREHOUSE'],
        'role': os.environ.get('SNOWFLAKE_ROLE'),
    }
    connection = connector.connect(
        **{key: value for key, value in arguments.items() if value}
    )
    yield connection
    connection.close()


@pytest.fixture(scope='module')
def databricks() -> Iterator[Any]:
    _require(DATABRICKS_KEYS, 'a real Databricks workspace')
    sql = pytest.importorskip('databricks.sql')
    connection = sql.connect(
        server_hostname=os.environ['DATABRICKS_SERVER_HOSTNAME'],
        http_path=os.environ['DATABRICKS_HTTP_PATH'],
        access_token=os.environ['DATABRICKS_TOKEN'],
    )
    yield connection
    connection.close()


def _execute(connection: Any, sql: str) -> None:
    with connection.cursor() as cursor:
        cursor.execute(sql)


def _suffix() -> str:
    return uuid.uuid4().hex[:8].upper()


# Time windows on Snowflake ###################################################

_CLOCK_QUERY = 'PREDICT COUNT(events.*, 0, 6, hours) FOR EACH users.USER_ID'
_CLOCK_ANCHOR = pd.Timestamp('2024-03-15 12:00:00')

#: Ten users, and three weeks of events at minutes past every hour, so a
#: window moved by any whole or half-hour offset holds other events.
_CLOCK_EVENTS = """
SELECT SEQ4() + 1 AS EVENT_ID,
       MOD(SEQ4(), 10) + 1 AS USER_ID,
       TIMEADD(minute, MOD(7 * (MOD(SEQ4(), 10) + 1), 60),
               TIMEADD(hour, FLOOR(SEQ4() / 10),
                       '2024-03-01 00:00:00'::TIMESTAMP_NTZ)) AS AT_NTZ,
       MOD(SEQ4(), 13)::FLOAT AS VALUE
FROM TABLE(GENERATOR(ROWCOUNT => 5040))
"""


def _snowflake_clock(connection: Any, time_type: str) -> tuple[str, str]:
    r"""The clock's users and events, ``HAPPENED_AT`` of ``time_type``
    holding the instants the naive times name in UTC.
    """
    suffix = _suffix()
    users, events = f'SDM_FIX_USERS_{suffix}', f'SDM_FIX_EVENTS_{suffix}'
    happened = {
        'TIMESTAMP_NTZ': 'AT_NTZ',
        'TIMESTAMP_TZ': (
            "TO_TIMESTAMP_TZ(TO_VARCHAR(AT_NTZ, 'YYYY-MM-DD HH24:MI:SS') "
            "|| ' +00:00', 'YYYY-MM-DD HH24:MI:SS TZH:TZM')"
        ),
        'TIMESTAMP_LTZ': (
            "TO_TIMESTAMP_TZ(TO_VARCHAR(AT_NTZ, 'YYYY-MM-DD HH24:MI:SS') "
            "|| ' +00:00', 'YYYY-MM-DD HH24:MI:SS TZH:TZM')::TIMESTAMP_LTZ"
        ),
    }[time_type]
    _execute(
        connection,
        f'CREATE TABLE {users} (USER_ID NUMBER(38,0) PRIMARY KEY) AS '
        'SELECT SEQ4() + 1 FROM TABLE(GENERATOR(ROWCOUNT => 10))',
    )
    _execute(
        connection,
        f'CREATE TABLE {events} AS SELECT EVENT_ID, USER_ID, '
        f'{happened}::{time_type} AS HAPPENED_AT, VALUE '
        f'FROM ({_CLOCK_EVENTS})',
    )
    return users, events


def _payloads(source: Any, users: str, events: str) -> list[dict[str, Any]]:
    from sdm_client import PredictiveQuery, Schema
    from sdm_client.adapters.relational import RelationalAdapter

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        schema = Schema.from_source(
            source,
            {'users': users, 'events': events},
            primary_keys={'users': 'USER_ID', 'events': 'EVENT_ID'},
            time_columns={'events': 'HAPPENED_AT'},
        )
        request = PredictiveQuery(_CLOCK_QUERY, schema).materialize(
            indices=[1, 2, 3, 4, 5],
            num_context_samples=20,
            random_seed=7,
            anchor_time=_CLOCK_ANCHOR,
        )
        payloads = list(
            RelationalAdapter().prediction_plan(request, ('prediction',))
        )
    for payload in payloads:
        metadata = payload.get('metadata')
        if isinstance(metadata, dict):
            metadata.pop('request_id', None)
    return payloads


@pytest.mark.parametrize('time_type', ['TIMESTAMP_LTZ', 'TIMESTAMP_TZ'])
def test_snowflake_zoned_windows_follow_no_session_timezone(
    snowflake: Any, time_type: str
) -> None:
    from sdm_connectors import SnowflakeSource

    def source() -> Any:
        return SnowflakeSource(
            snowflake,
            database=os.environ['SNOWFLAKE_DATABASE'],
            schema=os.environ['SNOWFLAKE_SCHEMA'],
        )

    created: list[str] = []
    try:
        naive = _snowflake_clock(snowflake, 'TIMESTAMP_NTZ')
        zoned = _snowflake_clock(snowflake, time_type)
        created += [*naive, *zoned]
        _execute(snowflake, "ALTER SESSION SET TIMEZONE = 'UTC'")
        want = _payloads(source(), *naive)
        # Snowflake's default, under which an NTZ bound read as LTZ or TZ
        # moved every window by seven or eight hours.
        _execute(
            snowflake, "ALTER SESSION SET TIMEZONE = 'America/Los_Angeles'"
        )
        got = _payloads(source(), *zoned)
        assert got == want
    finally:
        _execute(snowflake, 'ALTER SESSION UNSET TIMEZONE')
        for table in created:
            _execute(snowflake, f'DROP TABLE IF EXISTS {table}')


def test_snowflake_zoned_expression_columns_compare_as_instants(
    snowflake: Any,
) -> None:
    r"""An expression time column read as ``TIMESTAMP_LTZ`` or
    ``TIMESTAMP_TZ`` is known to be zoned from its inference sample, and its
    ``TIMESTAMP_TZ`` bounds select the rows the same instants select of the
    naive column under UTC.
    """
    from predictive_query.schema.typing import Dtype
    from sdm_connectors.backends.snowflake import run
    from sdm_relational_engine.rfm.backend.snow import dialect
    from sdm_relational_engine.rfm.backend.snow.table import SnowTable
    from sdm_relational_engine.rfm.base import ColumnSpec

    def instant(cast: str) -> str:
        return (
            f'TO_TIMESTAMP_{cast}(TO_VARCHAR(HAPPENED_AT, '
            "'YYYY-MM-DD HH24:MI:SS') || ' +00:00', "
            "'YYYY-MM-DD HH24:MI:SS TZH:TZM')"
        )

    expressions = {
        'AT_LTZ': instant('LTZ'),
        'AT_TZ': instant('TZ'),
        'AT_NTZ': 'HAPPENED_AT',
    }
    end, start = '2024-03-15 12:00:00', '2024-03-15 06:00:00'
    created: list[str] = []

    def count(where: str, params: list[str]) -> int:
        # Bound as the engine binds, to `?` markers.
        table = run(
            snowflake, f'SELECT COUNT(*) FROM {events} WHERE {where}', params
        )
        return int(table.column(0)[0].as_py())

    try:
        users, events = _snowflake_clock(snowflake, 'TIMESTAMP_NTZ')
        created += [users, events]
        _execute(snowflake, "ALTER SESSION SET TIMEZONE = 'UTC'")
        want = count(
            'HAPPENED_AT <= ?::TIMESTAMP_NTZ AND HAPPENED_AT > ?::TIMESTAMP_NTZ',
            [end, start],
        )
        assert want > 0
        _execute(
            snowflake, "ALTER SESSION SET TIMEZONE = 'America/Los_Angeles'"
        )
        table = SnowTable(
            snowflake,
            name='events',
            source_name=events,
            database=os.environ['SNOWFLAKE_DATABASE'],
            schema=os.environ['SNOWFLAKE_SCHEMA'],
            columns=[
                ColumnSpec(name='EVENT_ID'),
                *(
                    ColumnSpec(name=name, expr=expr, dtype=Dtype.date)
                    for name, expr in expressions.items()
                ),
            ],
            primary_key='EVENT_ID',
        )
        assert table._zoned_expr_columns == {'AT_LTZ', 'AT_TZ'}
        bound = dialect._outer_bound('TZ')
        zoned_end = end.replace(' ', 'T') + '+00:00'
        zoned_start = start.replace(' ', 'T') + '+00:00'
        for name in ('AT_LTZ', 'AT_TZ'):
            expr = expressions[name]
            got = count(
                f'{expr} <= {bound} AND {expr} > {bound}',
                [zoned_end, zoned_start],
            )
            assert got == want, name
    finally:
        _execute(snowflake, 'ALTER SESSION UNSET TIMEZONE')
        for created_table in created:
            _execute(snowflake, f'DROP TABLE IF EXISTS {created_table}')


# Multi-column links ##########################################################

_REGIONS = ('east', 'north', 'south')


def _parent_rows() -> list[tuple[str, str, str]]:
    return [
        (region, f'c{number:04d}', f'seg{number % 7}')
        for number in range(1000)
        for region in _REGIONS
    ]


def _values(rows: list[tuple[Any, ...]]) -> str:
    def literal(value: Any) -> str:
        if value is None:
            return 'NULL'
        if isinstance(value, str):
            # Generated here: letters, digits and no quote.
            return f"'{value}'"
        return str(value)

    return ', '.join(
        '(' + ', '.join(literal(value) for value in row) + ')' for row in rows
    )


class _Warehouse:
    r"""One warehouse's tables and graphs, with its names' spelling."""

    def __init__(
        self,
        connection: Any,
        graph: Callable[[list[str]], Any],
        name: Callable[[str], str],
        text_type: str,
    ) -> None:
        self.connection = connection
        self._graph = graph
        self.name = name
        self.text_type = text_type
        self.suffix = _suffix()
        self.created: list[str] = []

    def table(self, base: str) -> str:
        return self.name(f'sdm_fix_{base}_{self.suffix}')

    def write(self, base: str, columns: list[str], rows: list[tuple]) -> str:
        table = self.table(base)
        types = ', '.join(
            f'{self.name(column)} '
            f'{"BIGINT" if column == "child_id" else self.text_type}'
            for column in columns
        )
        _execute(self.connection, f'CREATE TABLE {table} ({types})')
        self.created.append(table)
        for start in range(0, len(rows), 500):
            _execute(
                self.connection,
                f'INSERT INTO {table} VALUES {_values(rows[start : start + 500])}',
            )
        return table

    def graph(self, tables: list[str]) -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return self._graph(tables)

    def drop(self) -> None:
        for table in self.created:
            _execute(self.connection, f'DROP TABLE IF EXISTS {table}')


@pytest.fixture(params=['databricks', 'snowflake'])
def warehouse(request: pytest.FixtureRequest) -> Iterator[_Warehouse]:
    import sdm_relational_engine.rfm as rfm

    if request.param == 'snowflake':
        connection = request.getfixturevalue('snowflake')
        database = os.environ['SNOWFLAKE_DATABASE']
        schema = os.environ['SNOWFLAKE_SCHEMA']
        built = _Warehouse(
            connection,
            lambda tables: rfm.Graph.from_snowflake(
                connection,
                tables=tables,
                database=database,
                schema=schema,
                infer_metadata=False,
                verbose=False,
            ),
            str.upper,
            'VARCHAR',
        )
    else:
        connection = request.getfixturevalue('databricks')
        catalog = os.environ['DATABRICKS_CATALOG']
        schema = os.environ['DATABRICKS_SCHEMA']
        _execute(connection, f'USE CATALOG `{catalog}`')
        _execute(connection, f'USE SCHEMA `{schema}`')
        built = _Warehouse(
            connection,
            lambda tables: rfm.Graph.from_databricks(
                connection,
                tables=tables,
                catalog=catalog,
                schema=schema,
                infer_metadata=False,
                verbose=False,
            ),
            str.lower,
            'STRING',
        )
    try:
        yield built
    finally:
        built.drop()


def _keyed(warehouse: _Warehouse, parent: str, child: str) -> Any:
    graph = warehouse.graph([parent, child])
    graph[parent].primary_key = (
        warehouse.name('region'),
        warehouse.name('code'),
    )
    graph[child].primary_key = warehouse.name('child_id')
    return graph


def _outside(warehouse: _Warehouse, parent: str, count: int) -> list[tuple]:
    r"""``count`` parents the parent's own inference sample doesn't hold."""
    from sdm_relational_engine.rfm.base import composite_key

    graph = warehouse.graph([parent])
    key = (warehouse.name('region'), warehouse.name('code'))
    graph[parent].primary_key = key
    sample = set(
        graph[parent]._sample_values(graph[parent].primary_key.name).dropna()
    )
    rows = [row[:2] for row in _parent_rows()]
    folds = composite_key.encode_frame(
        pd.DataFrame(rows, columns=list(key)), list(key)
    )
    outside = [
        row for row, fold in zip(rows, folds, strict=True) if fold not in sample
    ]
    return outside[:count]


def _child(warehouse: _Warehouse, pairs: list[tuple]) -> str:
    rows = [
        (index + 1, region, code)
        for index, (region, code) in enumerate(pairs * 2)
    ]
    return warehouse.write('child', ['child_id', 'region', 'code'], rows)


def _parent(warehouse: _Warehouse) -> str:
    return warehouse.write(
        'parent', ['region', 'code', 'segment'], _parent_rows()
    )


def _fkey(warehouse: _Warehouse, *names: str) -> tuple[str, ...]:
    return tuple(warehouse.name(name) for name in names)


def test_a1_a_link_to_parents_outside_the_parents_sample_holds(
    warehouse: _Warehouse,
) -> None:
    parent = _parent(warehouse)
    child = _child(warehouse, _outside(warehouse, parent, 600))
    graph = _keyed(warehouse, parent, child)
    graph.link(
        src_table=child,
        fkey=_fkey(warehouse, 'region', 'code'),
        dst_table=parent,
    )
    assert len(graph.edges) == 1


def test_columns_in_the_wrong_order_are_refused(warehouse: _Warehouse) -> None:
    parent = _parent(warehouse)
    child = _child(warehouse, _outside(warehouse, parent, 600))
    graph = _keyed(warehouse, parent, child)
    with pytest.raises(ValueError, match='matches no rows'):
        graph.link(
            src_table=child,
            fkey=_fkey(warehouse, 'code', 'region'),
            dst_table=parent,
        )
    assert not graph.edges


def test_values_the_parent_does_not_hold_are_refused(
    warehouse: _Warehouse,
) -> None:
    parent = _parent(warehouse)
    child = _child(warehouse, [('west', f'c{n:04d}') for n in range(600)])
    graph = _keyed(warehouse, parent, child)
    with pytest.raises(ValueError, match='matches no rows'):
        graph.link(
            src_table=child,
            fkey=_fkey(warehouse, 'region', 'code'),
            dst_table=parent,
        )
    assert not graph.edges


def test_a_null_part_is_refused(warehouse: _Warehouse) -> None:
    parent = _parent(warehouse)
    pairs = [
        (region, None if index % 10 == 0 else code)
        for index, (region, code) in enumerate(_outside(warehouse, parent, 600))
    ]
    child = _child(warehouse, pairs)
    graph = _keyed(warehouse, parent, child)
    with pytest.raises(ValueError, match='null'):
        graph.link(
            src_table=child,
            fkey=_fkey(warehouse, 'region', 'code'),
            dst_table=parent,
        )
    assert not graph.edges
