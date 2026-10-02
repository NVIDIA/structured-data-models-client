# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import pandas as pd
import pytest
from sdm_relational_engine.exceptions import GraphConstructionError
from sdm_relational_engine.rfm import Graph
from sdm_relational_engine.rfm.base import KeyCount, count_keys

try:
    import pyarrow as pa
    import yaml  # noqa: F401
    from sdm_relational_engine.rfm.backend.snow import Connection
    from snowflake.connector.errors import ProgrammingError
except ImportError:
    pytest.skip("'snowflake' extension not installed", allow_module_level=True)

_DATABASE = 'DB'
_SCHEMA = 'SCH'

_DATA = {
    'DATACENTERS': pd.DataFrame(
        {
            'DATACENTER_ID': [f'dc_{i}' for i in range(4)],
            'REGION': ['AMER', 'EMEA', 'APAC', 'AMER'],
            'COUNTRY': ['US', 'DE', 'JP', 'CA'],
        }
    ),
    'RACKS': pd.DataFrame(
        {
            'RACK_ID': [f'rack_{i}' for i in range(12)],
            'DATACENTER_ID': [f'dc_{i % 4}' for i in range(12)],
            'HALL': [f'hall_{i % 3}' for i in range(12)],
        }
    ),
}

_COLUMN_TYPES = {
    'DATACENTERS': {
        'DATACENTER_ID': 'VARCHAR(16)',
        'REGION': 'VARCHAR(16)',
        'COUNTRY': 'VARCHAR(16)',
    },
    'RACKS': {
        'RACK_ID': 'VARCHAR(16)',
        'DATACENTER_ID': 'VARCHAR(16)',
        'HALL': 'VARCHAR(16)',
    },
}

_PRIMARY_KEYS = {'DATACENTERS': 'DATACENTER_ID', 'RACKS': 'RACK_ID'}

# `SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW` echoes the author's expressions
# verbatim, so a self-qualifier may appear in any case:
_SEMANTIC_VIEW_YAML = """
tables:
  - name: DATACENTERS
    base_table:
      database: DB
      schema: SCH
      table: DATACENTERS
    primary_key:
      columns:
        - DATACENTER_ID
    dimensions:
      - name: REGION
        expr: datacenters.REGION
        data_type: VARCHAR(16)
      - name: COUNTRY
        expr: DATACENTERS.COUNTRY
        data_type: VARCHAR(16)
  - name: RACKS
    base_table:
      database: DB
      schema: SCH
      table: RACKS
    primary_key:
      columns:
        - RACK_ID
    dimensions:
      - name: HALL
        expr: racks.HALL
        data_type: VARCHAR(16)
      - name: DC_REGION
        expr: datacenters.REGION
        data_type: VARCHAR(16)
relationships:
  - name: RACKS_TO_DATACENTERS
    left_table: RACKS
    right_table: DATACENTERS
    relationship_columns:
      - left_column: DATACENTER_ID
        right_column: DATACENTER_ID
"""


class _FakeCursor:
    def __init__(
        self,
        error: Exception | None = None,
        connection: '_FakeConnection | None' = None,
    ) -> None:
        self._error = error
        self._connection = connection
        self._rows: list = []
        self._arrow: pa.Table | None = None
        self.description: list[tuple[str]] = []

    def __enter__(self) -> '_FakeCursor':
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, parameters: object = None) -> None:
        if self._error is not None:
            raise self._error

        self._rows = []
        self._arrow = None
        if self._connection is not None:
            self._connection.statements.append(sql)
        sql = ' '.join(sql.split())

        if sql.startswith('SHOW TABLES LIKE ?'):
            (table,) = parameters
            hybrid = self._connection is not None and (
                table in self._connection.hybrid
            )
            self.description = [('created_on',), ('name',), ('is_hybrid',)]
            self._rows = [(None, table, 'Y' if hybrid else 'N')]
            return

        if ' AS k, COUNT(*) AS n, ' in sql:
            self._rows = [
                _count_key(select) for select in sql.split(' UNION ALL ')
            ]
            return

        if 'SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW' in sql:
            self._rows = [(_SEMANTIC_VIEW_YAML,)]
            return

        if sql.startswith('SHOW IMPORTED KEYS'):
            return

        if sql.startswith('DESCRIBE TABLE'):
            # NOTE Quoted identifiers are case-sensitive in Snowflake, so only
            # the canonical spelling resolves:
            match = re.fullmatch(
                r'DESCRIBE TABLE "([^"]+)"\."([^"]+)"\."([^"]+)"', sql
            )
            assert match is not None, f'Unexpected SQL: {sql}'
            database, schema, table = match.groups()
            if (database, schema) != (
                _DATABASE,
                _SCHEMA,
            ) or table not in _COLUMN_TYPES:
                raise RuntimeError(
                    f"SQL compilation error: Table '{table}' does not exist"
                )
            self._rows = [
                (
                    name,
                    dtype,
                    'COLUMN',
                    'N' if _PRIMARY_KEYS[table] == name else 'Y',
                    None,
                    'Y' if _PRIMARY_KEYS[table] == name else 'N',
                    'N',
                )
                for name, dtype in _COLUMN_TYPES[table].items()
            ]
            return

        match = re.search(r'FROM "([^"]+)"\."([^"]+)"\."([^"]+)"', sql)
        assert match is not None, f'Unexpected SQL: {sql}'
        assert (match.group(1), match.group(2)) == (_DATABASE, _SCHEMA)
        df = _DATA[match.group(3)]

        names = re.findall(r'"((?:[^"]|"")+)"', sql.split(' FROM ')[0])
        self._arrow = pa.Table.from_pandas(
            pd.DataFrame({name: df[name] for name in names}),
            preserve_index=False,
        )

    def fetchall(self) -> list:
        return self._rows

    def fetchone(self) -> tuple:
        return self._rows[0]

    def fetch_arrow_all(self, force_return_table: bool = False) -> 'pa.Table':
        assert self._arrow is not None
        return self._arrow


def _count_key(select: str) -> tuple[int, int, int, int]:
    r"""Answer one ``SELECT`` of a key count from the tables above."""
    label = int(re.match(r'SELECT (\d+) AS k', select).group(1))
    table = re.search(r'FROM "[^"]+"\."[^"]+"\."([^"]+)"', select).group(1)
    names = re.findall(
        r'"([^"]+)"', re.search(r'COUNT\(DISTINCT ([^)]*)\)', select).group(1)
    )
    keys = list(zip(*(_DATA[table][name] for name in names), strict=True))
    present = [key for key in keys if not any(pd.isna(v) for v in key)]
    return label, len(keys), len(set(present)), len(keys) - len(present)


class _FakeConnection(Connection):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.closed = False
        self._paramstyle = 'pyformat'
        self.statements: list[str] = []
        self.hybrid: set[str] = set()

    def cursor(self) -> _FakeCursor:  # type: ignore[override]
        return _FakeCursor(self.error, self)

    def close(self) -> None:
        self.closed = True

    def __del__(self) -> None:
        pass


def test_from_snowflake_semantic_view_keeps_self_qualified_columns() -> None:
    # Regression test for
    # `graph-snowflake-semantic-view-drops-self-qualified-columns.md`: the
    # owning table's qualifier used to be stripped case-sensitively but tested
    # case-insensitively, so `datacenters.REGION` was dropped as if it
    # referenced another table.
    with pytest.warns(UserWarning) as caught:
        graph = Graph.from_snowflake_semantic_view(
            'GPU_FLEET_SV',
            connection=_FakeConnection(),
            verbose=False,
        )

    datacenters = graph['DATACENTERS']
    assert {column.name for column in datacenters.columns} == {
        'DATACENTER_ID',
        'REGION',
        'COUNTRY',
    }

    racks = graph['RACKS']
    assert {column.name for column in racks.columns} == {
        'RACK_ID',
        'HALL',
        'DATACENTER_ID',
    }

    # A qualifier of another table is still rejected:
    message = str(caught[0].message)
    assert "Failed to add column 'DC_REGION' of table 'RACKS'" in message


def test_from_snowflake_case_insensitive_identifiers() -> None:
    # Regression test:
    # Snowflake folds unquoted identifiers to upper case, so a name typed in
    # any other case must still resolve.
    graph = Graph.from_snowflake(
        connection=_FakeConnection(),
        database=_DATABASE.lower(),
        schema=_SCHEMA.lower(),
        tables=['racks', 'datacenters'],
        edges=[('racks', 'DATACENTER_ID', 'datacenters')],
        infer_metadata=False,
        verbose=False,
    )

    assert graph['racks'].source_name == 'DB.SCH.RACKS'
    assert graph['datacenters'].source_name == 'DB.SCH.DATACENTERS'


def test_from_snowflake_unknown_table() -> None:
    with pytest.raises(ValueError, match='does not exist'):
        Graph.from_snowflake(
            connection=_FakeConnection(),
            database=_DATABASE,
            schema=_SCHEMA,
            tables=['DOES_NOT_EXIST'],
            infer_metadata=False,
            verbose=False,
        )


def test_from_snowflake_propagates_non_lookup_errors() -> None:
    # A failure that says nothing about whether the table exists must not be
    # reported as a missing table:
    error = ProgrammingError(
        msg='Authentication token has expired. The user must authenticate '
        'again.',
        errno=390114,
    )
    with pytest.raises(
        GraphConstructionError, match='Authentication token'
    ) as caught:
        Graph.from_snowflake(
            connection=_FakeConnection(error),
            database=_DATABASE,
            schema=_SCHEMA,
            tables=['RACKS'],
            infer_metadata=False,
            verbose=False,
        )

    assert isinstance(caught.value.__cause__, ProgrammingError)


def test_from_snowflake_reports_unresolvable_object_as_missing_table() -> None:
    error = ProgrammingError(
        msg="SQL compilation error:\nObject 'DB.SCH.RACKS' does not exist or "
        'not authorized.',
        errno=2003,
    )
    with pytest.raises(ValueError, match='does not exist') as caught:
        Graph.from_snowflake(
            connection=_FakeConnection(error),
            database=_DATABASE,
            schema=_SCHEMA,
            tables=['RACKS'],
            infer_metadata=False,
            verbose=False,
        )
    assert caught.value.__cause__ is error


def test_from_snowflake_tracks_internal_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression test: a
    # connection the client opened is a connection the client owns and closes.
    connection = _FakeConnection()
    monkeypatch.setattr(
        'sdm_relational_engine.rfm.backend.snow.connect',
        lambda **kwargs: connection,
    )

    graph = Graph.from_snowflake(
        connection=dict(account='localhost'),
        database=_DATABASE,
        schema=_SCHEMA,
        tables=['RACKS'],
        infer_metadata=False,
        verbose=False,
    )
    assert graph._connection is connection

    graph = Graph.from_snowflake(
        connection=connection,
        database=_DATABASE,
        schema=_SCHEMA,
        tables=['RACKS'],
        infer_metadata=False,
        verbose=False,
    )
    assert graph._connection is None


def test_from_snowflake_closes_internal_connection_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A connection the client opened must not leak in case the graph is never
    # constructed and can therefore never take ownership of it:
    connection = _FakeConnection()
    monkeypatch.setattr(
        'sdm_relational_engine.rfm.backend.snow.connect',
        lambda **kwargs: connection,
    )

    with pytest.raises(ValueError, match='does not exist'):
        Graph.from_snowflake(
            connection=dict(account='localhost'),
            database=_DATABASE,
            schema=_SCHEMA,
            tables=['DOES_NOT_EXIST'],
            infer_metadata=False,
            verbose=False,
        )
    assert connection.closed


def test_from_snowflake_keeps_external_connection_open_on_error() -> None:
    connection = _FakeConnection()

    with pytest.raises(ValueError, match='does not exist'):
        Graph.from_snowflake(
            connection=connection,
            database=_DATABASE,
            schema=_SCHEMA,
            tables=['DOES_NOT_EXIST'],
            infer_metadata=False,
            verbose=False,
        )
    assert not connection.closed


def test_from_snowflake_semantic_view_closes_internal_connection_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = ProgrammingError(msg='Semantic view does not exist', errno=2003)
    connection = _FakeConnection(error)
    monkeypatch.setattr(
        'sdm_relational_engine.rfm.backend.snow.connect',
        lambda **kwargs: connection,
    )

    with pytest.raises(GraphConstructionError):
        Graph.from_snowflake_semantic_view(
            'GPU_FLEET_SV',
            connection=dict(account='localhost'),
            verbose=False,
        )
    assert connection.closed


def test_semantic_view_drops_a_type_mismatched_relationship(
    monkeypatch,
) -> None:
    """A production semantic view declared a relationship between a NUMBER
    foreign key and a TEXT primary key. Detecting that is right; aborting on it
    threw away 9 tables and 21 well-formed edges, in a view the caller does not
    own and cannot repair. The relationship is dropped and reported instead,
    like every other unconvertible element.
    """
    from sdm_relational_engine.rfm import Edge, ViewConversionWarning

    datacenters = _DATA['DATACENTERS'].copy()
    datacenters['DATACENTER_ID'] = range(len(datacenters))
    monkeypatch.setitem(_DATA, 'DATACENTERS', datacenters)
    monkeypatch.setitem(
        _COLUMN_TYPES,
        'DATACENTERS',
        {**_COLUMN_TYPES['DATACENTERS'], 'DATACENTER_ID': 'NUMBER(38,0)'},
    )

    with pytest.warns(ViewConversionWarning):
        graph = Graph.from_snowflake_semantic_view(
            'GPU_FLEET_SV',
            connection=_FakeConnection(),
            verbose=False,
        )

    assert set(graph.tables) == {'DATACENTERS', 'RACKS'}
    assert Edge('RACKS', 'DATACENTER_ID', 'DATACENTERS') not in graph.edges

    dropped = [
        msg
        for msg in graph.conversion_messages
        if 'incompatible data types' in msg
    ]
    assert len(dropped) == 1
    assert "'RACKS'" in dropped[0] and "'DATACENTERS'" in dropped[0]
    assert graph.validate() is graph


def _counted_graph(connection: _FakeConnection) -> Graph:
    return Graph.from_snowflake(
        connection=connection,
        database=_DATABASE,
        schema=_SCHEMA,
        tables=['RACKS', 'DATACENTERS'],
        infer_metadata=False,
        verbose=False,
    )


def test_every_key_is_counted_in_one_statement() -> None:
    connection = _FakeConnection()
    graph = _counted_graph(connection)
    connection.statements.clear()

    counts = count_keys(
        [
            (graph['RACKS'], ['RACK_ID']),
            (graph['DATACENTERS'], ['REGION', 'COUNTRY']),
        ]
    )

    # A declared key is asked about first: only a hybrid table enforces one.
    shown, counted = connection.statements
    assert shown.startswith('SHOW TABLES LIKE ? IN SCHEMA "DB"."SCH"')
    assert counted == (
        'SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT "RACK_ID") AS d, '
        'COUNT(CASE WHEN "RACK_ID" IS NULL THEN 1 END) AS z\n'
        'FROM "DB"."SCH"."RACKS"\n'
        'UNION ALL\n'
        'SELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT "REGION", "COUNTRY") '
        'AS d, COUNT(CASE WHEN "REGION" IS NULL OR "COUNTRY" IS NULL THEN 1 '
        'END) AS z\n'
        'FROM "DB"."SCH"."DATACENTERS"'
    )
    assert counts == [
        KeyCount(rows=12, distinct=12, nulls=0, enforced=False),
        KeyCount(rows=4, distinct=4, nulls=0, enforced=False),
    ]


def test_a_hybrid_table_key_is_not_counted() -> None:
    connection = _FakeConnection()
    connection.hybrid = {'RACKS'}
    graph = _counted_graph(connection)
    connection.statements.clear()

    counts = count_keys([(graph['RACKS'], ['RACK_ID'])])

    assert counts == [KeyCount(None, None, 0, enforced=True)]
    assert not [sql for sql in connection.statements if 'COUNT(' in sql]
