# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public standalone objects and consistency checks use shared native policy."""

import copy

import pandas as pd
import pytest

from sdm_client import Column, Schema, SourceTable, Table
from sdm_client.errors import StructuredDataError


@pytest.fixture(autouse=True)
def forbid_legacy(monkeypatch):
    monkeypatch.delenv('SDM_CORE_ROUTES', raising=False)

    def fail(*args, **kwargs):
        raise AssertionError('legacy schema policy reached')

    for method in ('_check_stype', '_check_supports_dtype'):
        monkeypatch.setattr(Column, method, fail)
    for method in ('_check_key', '_adopt_key', '_check_time_column'):
        monkeypatch.setattr(Table, method, fail)
        monkeypatch.setattr(SourceTable, method, fail)
    monkeypatch.setattr(Schema, '_check_relationships', fail)


def frame_table():
    return Table(
        'items',
        pd.DataFrame({'id': [1, 2, 3], 'value': [10.0, 20.0, 30.0]}),
        {
            'id': Column('id', 'int64', 'numerical'),
            'value': Column('value', 'float64', 'numerical'),
        },
    )


def test_standalone_column_checks_are_atomic_and_native():
    c = Column('value', 'float64', 'numerical')
    c.stype = 'categorical'
    assert c.stype_declared
    with pytest.raises(StructuredDataError, match='not a semantic type'):
        c.stype = 'not-a-type'
    assert c.stype == 'categorical'
    with pytest.raises(StructuredDataError, match='cannot mean'):
        Column('value', 'string', 'numerical')
    c.dtype = 'string'
    c.check_consistent()
    c.dtype = 'float64'
    c.stype = 'numerical'
    c.dtype = 'string'
    with pytest.raises(StructuredDataError, match='cannot mean'):
        c.check_consistent()


def test_standalone_table_edit_retains_identity_without_schema_owner():
    table = frame_table()
    original = table['id']
    table.primary_key = 'id'
    assert table['id'] is original
    assert table['id'].stype == 'ID'
    assert getattr(table, '_schema', None) is None
    table.data.loc[2, 'value'] = 10.0
    with pytest.raises(StructuredDataError):
        table.primary_key = 'value'
    assert table.primary_key == 'id'
    assert table['value'].stype == 'numerical'
    table.primary_key = None
    assert table['id'].stype == 'numerical'


def test_standalone_table_constructor_keeps_borrowed_columns_isolated():
    original = frame_table()
    other = Table('other', original.data, original.columns, primary_key='id')
    assert other['id'] is not original['id']
    assert other['id'].stype == 'ID'
    assert original['id'].stype == 'numerical'
    clone = copy.copy(other)
    clone.primary_key = None
    assert other.primary_key == 'id'
    assert clone.primary_key is None


def test_standalone_source_table_edits_do_not_read_source():
    table = SourceTable(
        'items',
        'catalog.items',
        {
            'id': Column('id', 'int64', 'numerical'),
            'when': Column('when', 'timestamp', 'timestamp'),
        },
    )
    table.primary_key = 'id'
    table.time_column = 'when'
    assert table['id'].stype == 'ID'
    assert table.time_column == 'when'
    assert getattr(table, '_schema', None) is None
    with pytest.raises(StructuredDataError):
        table.primary_key = 'when'
    assert table.primary_key == 'id'


def test_schema_check_preserves_declarations_and_does_not_adopt_time():
    schema = Schema({'items': frame_table()})
    column = schema['items']['value']
    before = column.stype_declared
    schema.check_consistent()
    assert column.stype_declared == before
    assert schema['items'].time_column is None
    column.dtype = 'string'
    with pytest.raises(StructuredDataError, match='cannot mean'):
        schema.check_consistent()
    assert column.dtype == 'string'
    assert column.stype == 'numerical'


@pytest.mark.parametrize(
    'dtype,stype',
    [
        ('unknown', 'numerical'),
        ('float64', 'unknown'),
        (None, 'numerical'),
    ],
)
def test_column_consistency_keeps_permissive_incomplete_types(dtype, stype):
    column = Column('value', 'float64', 'numerical')
    object.__setattr__(column, 'dtype', dtype)
    object.__setattr__(column, 'stype', stype)
    column.check_consistent()


@pytest.mark.parametrize('source_kind', ['frames', 'duckdb'])
def test_bulk_binding_does_not_reenter_standalone_checks(
    monkeypatch, source_kind
):
    from sdm_client._rust import edits

    def fail(*args, **kwargs):
        raise AssertionError('bulk result reran standalone policy')

    monkeypatch.setattr(edits, 'check_column', fail)
    monkeypatch.setattr(edits, 'table', fail)
    if source_kind == 'frames':
        schema = Schema.from_data({'items': pd.DataFrame({'id': [1, 2]})})
    else:
        import duckdb
        from sdm_connectors import DuckDBSource

        with duckdb.connect() as connection:
            connection.execute('CREATE TABLE items(id INTEGER PRIMARY KEY)')
            connection.execute('INSERT INTO items VALUES (1), (2)')
            schema = Schema.from_source(
                DuckDBSource(connection), {'items': 'items'}
            )
    assert schema['items'].primary_key == 'id'


@pytest.mark.parametrize('catalog_facts', [False, True])
def test_source_schema_declarations_and_missing_dtype(
    monkeypatch, catalog_facts
):
    import duckdb
    from sdm_connectors import DuckDBSource

    from sdm_client.schema import _described_on_the_core

    def fail(*args, **kwargs):
        raise AssertionError('declared source schema reached legacy policy')

    monkeypatch.setattr(Schema, '_settle', fail)
    monkeypatch.setattr(Schema, '_mark_key_columns', fail)
    with duckdb.connect() as connection:
        source = DuckDBSource(connection)
        if catalog_facts:
            connection.execute('CREATE TABLE items(id INTEGER, value DOUBLE)')
            connection.execute('INSERT INTO items VALUES (1, 1.5), (2, 2.5)')
            schema = Schema.from_source(source, {'items': 'items'})
            schema['items']['value'].dtype = None
        else:
            schema = Schema(
                {
                    'items': SourceTable(
                        'items',
                        'items',
                        {
                            'id': Column('id', 'int64', 'numerical'),
                            'value': Column('value', None, 'numerical'),
                        },
                    )
                },
                source=source,
            )
        assert _described_on_the_core(schema) == catalog_facts
        schema.relationships = []
        schema['items'].primary_key = 'id'
        schema.check_consistent()
        assert schema['items']['value'].dtype is None
        assert _described_on_the_core(schema) == catalog_facts


def test_missing_dtype_survives_frame_constructor_edits_and_checks():
    table = Table(
        'items',
        pd.DataFrame({'id': [1, 2], 'value': [10, 20]}),
        {
            'id': Column('id', 'int64', 'numerical'),
            'value': Column('value', None, 'numerical'),
        },
    )
    schema = Schema({'items': table})
    schema['items'].primary_key = 'id'
    schema['items']['value'].stype = 'categorical'
    schema.check_consistent()
    assert schema['items']['value'].dtype is None
    assert schema['items']['value']._source is None


def test_standalone_table_constructors_preserve_end_time_role():
    columns = {
        'id': Column('id', 'int64', 'ID'),
        'closed': Column('closed', 'timestamp[ns]', 'timestamp'),
    }
    frame = pd.DataFrame({'id': [1], 'closed': pd.to_datetime(['2024-01-01'])})
    tables = [
        Table(
            'users', frame, columns, primary_key='id', end_time_column='closed'
        ),
        SourceTable(
            'users',
            'users',
            columns,
            primary_key='id',
            end_time_column='closed',
        ),
    ]
    for table in tables:
        assert table.end_time_column == 'closed'
        assert table.metadata_report().end_time_column == 'closed'
        assert table.time_column is None
