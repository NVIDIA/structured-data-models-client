# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A Snowflake semantic view as a schema's source.

The connection is faked the way the engine's own semantic-view tests fake it:
it answers the view's YAML, the tables' descriptions, and previews. Nothing
here reaches a Snowflake account.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import pandas as pd
import pytest

pytest.importorskip('sdm_relational_engine.relationallib')
pytest.importorskip('snowflake.connector')
pa = pytest.importorskip('pyarrow')
pytest.importorskip('yaml')

from sdm_connectors import SemanticModel, SnowflakeSource  # noqa: E402
from sdm_relational_engine.rfm.backend.snow import Connection  # noqa: E402
from sdm_relational_engine.rfm.backend.snow.binding import (  # noqa: E402
    paramstyle_attribute,
)

from sdm_client import Relationship, Schema, SourceTable  # noqa: E402
from sdm_client.materializers import RelationalMaterializer  # noqa: E402


@pytest.fixture(autouse=True)
def _legacy_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the engine whose constructors and driver SQL are faked here."""
    monkeypatch.setenv('SDM_CORE_ROUTES', '')


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
    def __init__(self, error: Exception | None = None) -> None:
        self._error = error
        self._rows: list = []
        self._arrow: pa.Table | None = None

    def __enter__(self) -> _FakeCursor:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def execute(self, sql: str, parameters: object = None) -> None:
        if self._error is not None:
            raise self._error

        self._rows = []
        self._arrow = None
        sql = ' '.join(sql.split())

        if 'SYSTEM$READ_YAML_FROM_SEMANTIC_VIEW' in sql:
            # An unqualified name resolves against the session, not the
            # source, so only the name the source qualifies is found.
            if parameters != ('DB.SCH.GPU_FLEET_SV',):
                raise RuntimeError(
                    f'SQL compilation error: {parameters!r} does not exist'
                )
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
                    'Y',
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

    def fetch_arrow_all(self, force_return_table: bool = False) -> pa.Table:
        assert self._arrow is not None
        return self._arrow


class _FakeConnection(Connection):
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.closed = False
        setattr(self, paramstyle_attribute(self), 'pyformat')

    def cursor(self) -> _FakeCursor:  # type: ignore[override]
        return _FakeCursor(self.error)

    def close(self) -> None:
        self.closed = True

    def __del__(self) -> None:
        pass


def _model() -> SemanticModel:
    return SnowflakeSource(
        _FakeConnection(), database=_DATABASE, schema=_SCHEMA
    ).semantic_model('GPU_FLEET_SV')


def _schema(tables: Any = None) -> Schema:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        return Schema.from_source(_model(), tables)


def test_the_view_s_tables_keys_and_relationships() -> None:
    schema = _schema()

    assert set(schema.tables) == {'DATACENTERS', 'RACKS'}
    assert all(isinstance(t, SourceTable) for t in schema.tables.values())
    assert schema['DATACENTERS'].primary_key == 'DATACENTER_ID'
    assert schema['RACKS'].primary_key == 'RACK_ID'
    assert schema.relationships == [
        Relationship('RACKS', 'DATACENTER_ID', 'DATACENTERS', 'DATACENTER_ID')
    ]


def test_tables_select_and_rename_within_the_view() -> None:
    from sdm_relational_engine.rfm.base import DataBackend

    schema = _schema({'racks': 'RACKS', 'sites': 'DATACENTERS'})

    assert set(schema.tables) == {'racks', 'sites'}
    assert schema.relationships == [
        Relationship('racks', 'DATACENTER_ID', 'sites', 'DATACENTER_ID')
    ]
    graph = RelationalMaterializer(schema)._build_graph()
    assert graph.backend == DataBackend.SNOWFLAKE
    assert set(graph.tables) == {'racks', 'sites'}
    assert {(e.src_table, e.fkey, e.dst_table) for e in graph.edges} == {
        ('racks', 'DATACENTER_ID', 'sites')
    }


@pytest.mark.parametrize('trusted', [False, True])
def test_trusting_the_view_reaches_the_engine(
    monkeypatch: pytest.MonkeyPatch, trusted: bool
) -> None:
    from sdm_relational_engine.rfm import Graph

    seen: list[Any] = []
    read = Graph.from_snowflake_semantic_view.__func__  # type: ignore[attr-defined]

    def spy(cls: Any, *args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs.get('trust_expressions'))
        return read(cls, *args, **kwargs)

    monkeypatch.setattr(Graph, 'from_snowflake_semantic_view', classmethod(spy))
    model = SnowflakeSource(
        _FakeConnection(), database=_DATABASE, schema=_SCHEMA
    ).semantic_model('GPU_FLEET_SV', trust_expressions=trusted)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        schema = Schema.from_source(model)
    RelationalMaterializer(schema)._build_graph()

    assert seen == [trusted]


@pytest.mark.parametrize('missing', ['table', 'column'])
def test_a_view_naming_what_is_not_there_is_an_invalid_request(
    monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    r"""A base table, or a declared column, that is not there refused the
    view as an `INVALID_REQUEST` until only the engine's view refusals were
    translated, and then escaped as the engine's own error.
    """
    from sdm_client import StructuredDataError

    if missing == 'table':
        monkeypatch.delitem(_COLUMN_TYPES, 'RACKS')
        said = "RACKS' does not exist"
    else:
        monkeypatch.setitem(
            _COLUMN_TYPES,
            'RACKS',
            {k: v for k, v in _COLUMN_TYPES['RACKS'].items() if k != 'HALL'},
        )
        said = "table 'RACKS' cannot hold what it declares"

    with pytest.raises(StructuredDataError) as caught:
        _schema()

    assert caught.value.code == 'INVALID_REQUEST'
    assert said in caught.value.message
