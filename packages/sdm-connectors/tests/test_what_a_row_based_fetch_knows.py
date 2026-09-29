# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the row-based fetch reads, and what it threw away.

Everything a driver that answers in rows can say about a column beyond its
values is in ``cursor.description``, and only the name was being read. A
column of nothing but nulls was built as ``object`` and described to the model
as text, though the server had called it ``bigint``; every other backend gets
its type through Arrow, so postgres was the only one that lost it.

Three more of the same shape. Rows were indexed by position, which a mapping
row factory refuses -- and ``row_factory`` is on this connector's own
allow-list, so the connection opens and every read fails with ``KeyError: 0``
surfacing as a query failure naming ``0``. A result set of no columns kept no
rows. A null-free integer came back nullable where every other path narrows
it.

And the guard called every failure a failure to execute, including the ones
where the statement executed and the *fetch* is what failed.
"""

from __future__ import annotations

import pandas as pd
import pytest

from sdm_connectors.reader import _frame_by_position, _frame_from_rows


class Described:
    """A ``cursor.description`` entry the way psycopg builds one."""

    def __init__(self, name: str, type_display: str | None = None) -> None:
        self.name = name
        self.type_display = type_display

    def __getitem__(self, index: int) -> object:
        return (self.name, None)[index]


def test_a_column_of_nothing_but_nulls_is_typed_from_what_the_server_called_it() -> (
    None
):
    frame = _frame_from_rows(
        [(1, None, None), (2, None, None)],
        [
            Described('a', 'int8'),
            Described('b', 'int8'),
            Described('c', 'text'),
        ],
    )

    assert frame['a'].dtype == 'int64'
    assert frame['b'].dtype == 'Int64'
    assert frame['c'].dtype == object


def test_nothing_is_claimed_where_the_driver_names_no_type() -> None:
    frame = _frame_from_rows([(None,), (None,)], [('a', None, None)])
    assert frame['a'].dtype == object


def test_a_column_the_server_calls_integral_but_that_holds_values_is_left_alone() -> (
    None
):
    frame = _frame_from_rows([('x',), ('y',)], [Described('a', 'int8')])
    assert frame['a'].tolist() == ['x', 'y']


def test_rows_that_are_mappings_are_read_by_name() -> None:
    frame = _frame_by_position(
        [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}], ['a', 'b']
    )

    assert list(frame.columns) == ['a', 'b']
    assert frame['a'].tolist() == [1, 2]
    assert frame['b'].tolist() == ['x', 'y']


def test_mapping_rows_that_lost_a_duplicate_name_say_so() -> None:
    from sdm_connectors.sql import ConnectorError

    with pytest.raises(ConnectorError) as raised:
        _frame_by_position([{'a': 2}], ['a', 'a'])

    said = str(raised.value)
    assert 'row factory' in said
    assert 'AS' in said


def test_rows_of_tuples_still_keep_two_columns_of_one_name() -> None:
    frame = _frame_by_position([(1, 2)], ['a', 'a'])

    assert list(frame.columns) == ['a', 'a']
    assert frame.iloc[0].tolist() == [1, 2]


def test_a_result_of_no_columns_keeps_its_rows() -> None:
    assert _frame_by_position([(), ()], []).shape == (2, 0)
    assert _frame_by_position([], []).shape == (0, 0)


def test_a_null_free_integer_comes_back_the_way_every_other_path_gives_it() -> (
    None
):
    frame = _frame_from_rows([(1,), (2,)], [Described('a', 'int8')])
    assert frame['a'].dtype == 'int64'

    with_a_null = _frame_from_rows([(1,), (None,)], [Described('a', 'int8')])
    assert with_a_null['a'].dtype == 'Int64'


def test_a_wide_integer_survives_the_narrowing() -> None:
    frame = _frame_from_rows([(2**70,), (1,)], [Described('a', 'numeric')])
    assert frame['a'].tolist() == [2**70, 1]


class _Cursor:
    description = [Described('a', 'int8')]

    def __init__(self, fail_on: str) -> None:
        self._fail_on = fail_on

    def execute(self, sql: str) -> None:
        if self._fail_on == 'execute':
            raise RuntimeError('relation "nope" does not exist')

    def fetchall(self) -> list[tuple[int]]:
        if self._fail_on == 'fetch':
            raise RuntimeError("Can't unify schema with duplicate field names.")
        return [(1,)]

    def close(self) -> None:
        pass


class _Connection:
    def __init__(self, fail_on: str) -> None:
        self._fail_on = fail_on

    def cursor(self) -> _Cursor:
        return _Cursor(self._fail_on)


@pytest.mark.parametrize(
    ('fail_on', 'said'),
    [('execute', 'query execution failed'), ('fetch', 'reading the result')],
)
def test_a_query_that_executed_is_not_reported_as_one_that_failed_to(
    fail_on: str, said: str
) -> None:
    from sdm_connectors.reader import read_table
    from sdm_connectors.sql import ConnectorError

    with pytest.raises(ConnectorError) as raised:
        read_table(_Connection(fail_on), query='select 1')

    assert said in str(raised.value)
    if fail_on == 'fetch':
        assert 'query execution failed' not in str(raised.value)


def test_a_query_that_works_still_answers() -> None:
    from sdm_connectors.reader import read_table

    frame = read_table(_Connection('neither'), query='select 1')
    assert frame['a'].tolist() == [1]
    assert isinstance(frame, pd.DataFrame)


def test_a_databricks_connect_with_no_credential_is_refused_not_opened(
    monkeypatch,
) -> None:
    """The driver's answer to no credential is to open a browser and wait for
    a human, binding a listening socket while it does, so a script that forgot
    the token never returned."""
    pytest.importorskip('databricks.sql')
    from sdm_connectors.backends import databricks as backend
    from sdm_connectors.sql import ConnectorError

    def refuse(**kwargs: object) -> object:
        raise AssertionError('the driver was called with no credential')

    monkeypatch.setattr(backend.databricks_sql, 'connect', refuse)
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    for absent in ({}, {'access_token': None}):
        with pytest.raises(ConnectorError) as raised:
            backend.connect(server_hostname='h', http_path='p', **absent)
        assert 'needs a credential' in str(raised.value)


@pytest.mark.parametrize(
    'credential',
    [
        {'access_token': 't'},
        {'username': 'u', 'password': 'p'},
        {'credentials_provider': object()},
        {'auth_type': 'databricks-oauth'},
    ],
)
def test_a_credential_of_any_kind_is_taken_as_one(
    monkeypatch, credential: dict
) -> None:
    pytest.importorskip('databricks.sql')
    from sdm_connectors.backends import databricks as backend

    seen: dict = {}
    monkeypatch.setattr(
        backend.databricks_sql,
        'connect',
        lambda **kwargs: seen.update(kwargs) or object(),
    )
    monkeypatch.setattr(backend.os, 'getenv', lambda name: None)

    backend.connect(server_hostname='h', http_path='p', **credential)
    assert set(credential) <= set(seen)


def test_the_postgres_docstring_says_what_passing_none_does() -> None:
    pytest.importorskip('psycopg')
    from sdm_connectors.backends import postgres

    said = ' '.join((postgres.connect.__doc__ or '').split())
    assert 'leave it to the server' not in said
    assert 'libpq then reads the same variable itself' in said
