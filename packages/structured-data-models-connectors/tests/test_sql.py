# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from sdm_connectors.sql import (
    ConnectorError,
    quote_ident,
    resolve_sql,
)


@pytest.mark.parametrize(
    'ident, char, expected',
    [
        ('items', '"', '"items"'),
        ('items', '`', '`items`'),
        ('a"b', '"', '"a""b"'),
        ("a'b", "'", "'a''b'"),
    ],
)
def test_quote_ident(ident, char, expected):
    assert quote_ident(ident, char) == expected


@pytest.mark.parametrize(
    'table',
    [
        'items',
        'main.items',
        'DB.SCHEMA.TABLE_1',
        '_private$tbl',
    ],
)
def test_resolve_sql_accepts_valid_identifiers(table):
    assert resolve_sql(table=table, query=None) == f'SELECT * FROM {table}'


def test_resolve_sql_passes_query_through():
    assert resolve_sql(table=None, query='SELECT 1') == 'SELECT 1'


@pytest.mark.parametrize(
    'table',
    [
        'items; DROP TABLE users',
        'items --',
        'items OR 1=1',
        '1items',
        'items..b',
        '',
    ],
)
def test_resolve_sql_rejects_invalid_identifiers(table):
    with pytest.raises(ConnectorError) as excinfo:
        resolve_sql(table=table, query=None)
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'


@pytest.mark.parametrize(
    'table',
    [
        'café',
        'Mixed Case Tbl',
        '1items',
    ],
)
def test_resolve_sql_rejection_names_the_workaround(table):
    # These are legal quotable table names that `table=` cannot express. The
    # message has to say so, because nothing else does.
    with pytest.raises(ConnectorError) as excinfo:
        resolve_sql(table=table, query=None)
    message = excinfo.value.message
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
    assert 'query=' in message
    assert 'quote_ident' in message


def test_quote_ident_reaches_the_names_resolve_sql_refuses():
    # The documented escape hatch has to actually produce valid SQL.
    assert quote_ident('café') == '"café"'
    assert quote_ident('Mixed Case Tbl') == '"Mixed Case Tbl"'
    assert quote_ident('select') == '"select"'


@pytest.mark.parametrize('table', ['select', 'from', 'table'])
def test_resolve_sql_accepts_reserved_words_unquoted(table):
    # Documented explicitly because it is surprising: a reserved word is a
    # plain identifier by this layer's rule, so it is *not* rejected here and
    # fails at execution instead. The docstring and README must keep saying so.
    assert resolve_sql(table=table, query=None) == f'SELECT * FROM {table}'
    message = resolve_sql.__doc__
    assert 'reserved word' in message and 'not* rejected here' in message


@pytest.mark.parametrize(
    'kwargs',
    [
        {},
        {'table': 'items', 'query': 'SELECT 1'},
    ],
)
def test_resolve_sql_requires_exactly_one_source(kwargs):
    with pytest.raises(ConnectorError) as excinfo:
        resolve_sql(table=kwargs.get('table'), query=kwargs.get('query'))
    assert excinfo.value.code == 'INVALID_CONNECTOR_ARGS'
