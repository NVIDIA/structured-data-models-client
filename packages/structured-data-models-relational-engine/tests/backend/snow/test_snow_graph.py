# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import re
import sys
import warnings

import pandas as pd
import pytest
from sdm_relational_engine.exceptions import GraphConstructionError
from sdm_relational_engine.rfm import Graph
from sdm_relational_engine.rfm.backend.snow.binding import paramstyle_attribute
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
        # The driver binds a float NaN as a FLOAT, which a timestamp refuses.
        if any(
            isinstance(v, float) and math.isnan(v) for v in parameters or ()
        ):
            raise ProgrammingError(f'NaN bound into: {sql}')
        sql = ' '.join(sql.split())

        if sql == f'SHOW TABLES IN SCHEMA "{_DATABASE}"."{_SCHEMA}"':
            assert self._connection is not None
            fields = self._connection.listing
            hybrid = self._connection.hybrid
            self.description = [(field,) for field in fields]
            self._rows = [
                (None, table, 'Y' if table in hybrid else 'N')[: len(fields)]
                for table in _COLUMN_TYPES
            ]
            return

        if ' AS k, COUNT(*) AS n, ' in sql:
            self._rows = [
                _count_key(select) for select in sql.split(' UNION ALL ')
            ]
            return

        if sql.startswith('SHOW TABLES LIKE ? '):
            # The row count is the eighth field of the listing.
            self._rows = [(None,) * 7 + (len(_DATA[parameters[0]]),)]
            return

        if ' as table_index, ' in sql:
            self._rows = [
                _min_max(select) for select in sql.split(' UNION ALL ')
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


def _min_max(select: str) -> tuple[int, object, object]:
    r"""Answer one ``SELECT`` of a time range from the tables above."""
    index = int(re.match(r'SELECT (\d+) as table_index', select).group(1))
    table = re.search(r'FROM "[^"]+"\."[^"]+"\."([^"]+)"', select).group(1)
    column = re.search(r'MIN\("([^"]+)"\)', select).group(1)
    values = _DATA[table][column]
    return index, values.min(), values.max()


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
        setattr(self, paramstyle_attribute(self), 'pyformat')
        self.statements: list[str] = []
        self.hybrid: set[str] = set()
        self.listing = ('created_on', 'name', 'is_hybrid')

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
            (graph['DATACENTERS'], ['DATACENTER_ID']),
        ]
    )

    # Declared keys are asked about first, in one listing for the schema:
    # only a hybrid table enforces one.
    shown, counted = connection.statements
    assert shown == 'SHOW TABLES IN SCHEMA "DB"."SCH"'
    assert counted == (
        'SELECT 0 AS k, COUNT(*) AS n, COUNT(DISTINCT "RACK_ID") AS d, '
        'COUNT(CASE WHEN "RACK_ID" IS NULL THEN 1 END) AS z\n'
        'FROM "DB"."SCH"."RACKS"\n'
        'UNION ALL\n'
        'SELECT 1 AS k, COUNT(*) AS n, COUNT(DISTINCT "REGION", "COUNTRY") '
        'AS d, COUNT(CASE WHEN "REGION" IS NULL OR "COUNTRY" IS NULL THEN 1 '
        'END) AS z\n'
        'FROM "DB"."SCH"."DATACENTERS"\n'
        'UNION ALL\n'
        'SELECT 2 AS k, COUNT(*) AS n, COUNT(DISTINCT "DATACENTER_ID") AS d, '
        'COUNT(CASE WHEN "DATACENTER_ID" IS NULL THEN 1 END) AS z\n'
        'FROM "DB"."SCH"."DATACENTERS"'
    )
    assert counts == [
        KeyCount(rows=12, distinct=12, nulls=0, enforced=False),
        KeyCount(rows=4, distinct=4, nulls=0, enforced=False),
        KeyCount(rows=4, distinct=4, nulls=0, enforced=False),
    ]


def test_a_hybrid_table_key_is_not_counted() -> None:
    connection = _FakeConnection()
    connection.hybrid = {'RACKS'}
    graph = _counted_graph(connection)
    connection.statements.clear()

    counts = count_keys(
        [(graph['RACKS'], ['RACK_ID']), (graph['DATACENTERS'], ['COUNTRY'])]
    )

    assert counts == [
        KeyCount(None, None, 0, enforced=True),
        KeyCount(rows=4, distinct=4, nulls=0, enforced=False),
    ]
    assert '"RACKS"' not in connection.statements[-1]


def test_a_listing_without_a_hybrid_column_counts_the_key() -> None:
    connection = _FakeConnection()
    connection.hybrid = {'RACKS'}
    connection.listing = ('created_on', 'name')
    graph = _counted_graph(connection)

    counts = count_keys([(graph['RACKS'], ['RACK_ID'])])

    assert counts == [KeyCount(rows=12, distinct=12, nulls=0, enforced=False)]


def _racks_installed_after_every_alert(monkeypatch: pytest.MonkeyPatch):
    r"""A graph whose entity table records an edit, not a creation, so every
    rack is installed after the last alert and none exists at any anchor.
    """
    from predictive_query.ast import Aggregation, Column, DateOffsetRange
    from predictive_query.query import ValidatedPredictiveQuery
    from predictive_query.schema.typing import AggregationType
    from sdm_relational_engine.rfm import KumoRelational

    racks = _DATA['RACKS'].assign(INSTALLED_AT=pd.Timestamp('2025-01-01'))
    alerts = pd.DataFrame(
        {
            'ALERT_ID': [f'alert_{i}' for i in range(6)],
            'RACK_ID': [f'rack_{i % 3}' for i in range(6)],
            'RAISED_AT': pd.date_range('2024-01-01', periods=6, freq='W'),
        }
    )
    monkeypatch.setitem(_DATA, 'RACKS', racks)
    monkeypatch.setitem(_DATA, 'ALERTS', alerts)
    monkeypatch.setitem(
        _COLUMN_TYPES,
        'RACKS',
        {**_COLUMN_TYPES['RACKS'], 'INSTALLED_AT': 'TIMESTAMP_NTZ(9)'},
    )
    monkeypatch.setitem(
        _COLUMN_TYPES,
        'ALERTS',
        {
            'ALERT_ID': 'VARCHAR(16)',
            'RACK_ID': 'VARCHAR(16)',
            'RAISED_AT': 'TIMESTAMP_NTZ(9)',
        },
    )
    monkeypatch.setitem(_PRIMARY_KEYS, 'ALERTS', 'ALERT_ID')
    connection = _FakeConnection()
    graph = Graph.from_snowflake(
        connection=connection,
        database=_DATABASE,
        schema=_SCHEMA,
        tables=[
            dict(
                name='RACKS', primary_key='RACK_ID', time_column='INSTALLED_AT'
            ),
            dict(
                name='ALERTS', primary_key='ALERT_ID', time_column='RAISED_AT'
            ),
        ],
        edges=[('ALERTS', 'RACK_ID', 'RACKS')],
        infer_metadata=False,
        verbose=False,
    )
    sampler = KumoRelational(graph, verbose=False)._sampler
    query = ValidatedPredictiveQuery(
        target_ast=Aggregation(
            aggr=AggregationType.COUNT,
            target=Column(fqn='ALERTS.*'),
            aggr_time_range=DateOffsetRange(0, 30),
        ),
        entity_ast=Column(fqn='RACKS.RACK_ID'),
    )
    connection.statements.clear()
    return sampler, query, connection


def test_no_entity_at_the_anchor_is_refused_without_a_window_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""No entity exists at any anchor, and the window over none of them was
    bound by NaN, which the warehouse refused.
    """
    sampler, query, connection = _racks_installed_after_every_alert(monkeypatch)

    with pytest.raises(ValueError, match='Failed to collect any context') as e:
        sampler.sample_target(
            query=query,
            num_train_examples=4,
            train_anchor_time=pd.Timestamp('2024-02-01'),
            num_train_trials=8,
            num_test_examples=0,
            test_anchor_time=pd.Timestamp('2024-02-01'),
            num_test_trials=0,
        )

    assert "No row of 'RACKS' existed at the anchor" in str(e.value)
    assert not [sql for sql in connection.statements if 'FLATTEN' in sql]


def test_lagged_targets_for_no_entities_ask_the_warehouse_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Lagged targets reach the window lookup without the guard context
    sampling has, so the lookup carries its own.
    """
    sampler, query, connection = _racks_installed_after_every_alert(monkeypatch)

    lagged = sampler.sample_lagged_target(
        query,
        entity_pkey=pd.Series([], dtype='object'),
        anchor_time=pd.Series([], dtype='datetime64[ns]'),
        lag_timesteps=2,
    )

    assert list(lagged.columns) == ['__kumo_arl0__', '__kumo_arl1__']
    assert len(lagged) == 0
    assert not [sql for sql in connection.statements if 'FLATTEN' in sql]


def test_semantic_view_refuses_an_edit_stamp_on_a_table_it_relates_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`RACKS` relates to `DATACENTERS`, so a datacenter is an entity edited
    in place and its `LAST_LOGIN` is no datacenter's time.
    """
    datacenters = _DATA['DATACENTERS'].assign(
        LAST_LOGIN=pd.date_range('2025-01-01', periods=4, freq='D')
    )
    monkeypatch.setitem(_DATA, 'DATACENTERS', datacenters)
    monkeypatch.setitem(
        _COLUMN_TYPES,
        'DATACENTERS',
        {**_COLUMN_TYPES['DATACENTERS'], 'LAST_LOGIN': 'TIMESTAMP_NTZ(9)'},
    )
    monkeypatch.setattr(
        sys.modules[__name__],
        '_SEMANTIC_VIEW_YAML',
        _SEMANTIC_VIEW_YAML.replace(
            '      - name: COUNTRY\n',
            '      - name: LAST_LOGIN\n'
            '        expr: datacenters.LAST_LOGIN\n'
            '        data_type: TIMESTAMP_NTZ(9)\n'
            '      - name: COUNTRY\n',
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        graph = Graph.from_snowflake_semantic_view(
            'GPU_FLEET_SV', connection=_FakeConnection(), verbose=False
        )

    assert [edge.dst_table for edge in graph.edges] == ['DATACENTERS']
    choice = graph['DATACENTERS'].time_column_choice
    assert choice is not None and choice.fallback == 'LAST_LOGIN'
    assert graph['DATACENTERS'].time_column is None
