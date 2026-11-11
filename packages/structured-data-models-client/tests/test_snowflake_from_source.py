# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Snowflake public schema and materialization regressions.
A local connector stub supplies source rows and catalog metadata, with authored
key reads and table listings, spelling retries, session refresh and hybrid-key tests.
Schema and payload expectations are independent assertions over the source shop rows.
These are local connector tests, not live Snowflake/model validation.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pytest
from sdm_connectors import SnowflakeSource
from snowflake.connector import SnowflakeConnection
from snowflake.connector.errors import (
    Error,
    NotSupportedError,
    ProgrammingError,
)

from sdm_client import (
    KumoRelational,
    PredictiveQuery,
    Schema,
    SDMClient,
    StructuredDataError,
)

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'contract' / 'fixtures' / 'maintained'

DB = '{catalog}'
SCHEMA = '{schema}'

#: A second statement reading the listing's ``rows``, which the core no
#: longer sends: it reads the count from the listing, as the old client does.
ROWS_OF_THE_LISTING = (
    'SELECT "rows" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID())) LIMIT 1'
)


def _captured(case: str) -> dict[str, Any]:
    """Read only independently extracted facts; the archive is not a test dependency."""
    name = 'snowflake.json'
    cases = json.loads((FIXTURES / name).read_text(encoding='utf-8'))
    assert case in cases, f'no retained Snowflake case {case}'
    return cases[case]


def _missing(text: str) -> ProgrammingError:
    r"""The connector's error for an object Snowflake cannot resolve."""
    return ProgrammingError(
        msg=f'SQL compilation error: {text} does not exist or not authorized.',
        errno=2003,
        sqlstate='02000',
    )


class _Recording:
    """Retained source previews and catalog inputs, indexed by SQL and parameters."""

    def __init__(self, case: str) -> None:
        directory = FIXTURES / 'snowflake'
        capture = _captured(case)
        self.answers: dict[tuple[str, str], Any] = {}
        for statement in capture['statements']:
            key = (statement['text'], json.dumps(statement['parameters']))
            if statement.get('result'):
                # Read from a file, so each buffer is allocated aligned, as
                # a driver's are, never a view into the file's bytes.
                with (directory / statement['result']).open('rb') as stream:
                    self.answers.setdefault(
                        key, pa.ipc.open_stream(stream).read_all()
                    )
            else:
                self.answers.setdefault(key, _missing(statement['text']))


class _Cursor:
    def __init__(self, connection: _Warehouse) -> None:
        self._connection = connection
        self._answer: pa.Table | None = None
        self._rows_only = False
        self.description: list[tuple[Any, ...]] | None = None

    def __enter__(self) -> _Cursor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def close(self) -> None:
        return None

    def execute(self, command: str, params: Any = None, **_: Any) -> _Cursor:
        parameters = list(params or [])
        self._answer = self._connection.answer(command, parameters)
        self._rows_only = command.lstrip().startswith(('SHOW', 'DESCRIBE'))
        self.description = [
            (name, 2, None, None, None, None, True)
            for name in self._answer.column_names
        ]
        return self

    def fetch_arrow_all(
        self, force_return_table: bool = False, **_: Any
    ) -> pa.Table | None:
        assert self._answer is not None
        if self._rows_only:
            raise NotSupportedError
        if self._answer.num_rows == 0 and not force_return_table:
            return None
        return self._answer

    def fetchall(self) -> list[tuple[Any, ...]]:
        assert self._answer is not None
        columns = [column.to_pylist() for column in self._answer.columns]
        return list(zip(*columns, strict=True))

    def fetchone(self) -> tuple[Any, ...] | None:
        rows = self.fetchall()
        return rows[0] if rows else None


class _Warehouse(SnowflakeConnection):
    r"""A connector connection answering ``case``'s recorded statements;
    ``upper`` holds the tables stored in upper case (a statement
    naming one in lower case fails with errno 2003, its upper-cased form
    answered as the recording's lower-case one); the database and schema
    are stored in upper case, as the recordings' were; ``fail`` maps a
    statement's text to the error it raises."""

    def __init__(self, case: str = 'base.binary') -> None:
        self._case = case
        self._recording = _Recording(case)
        self._view_base = (
            _Recording('base.regression')
            if case == 'snowflake.rowcount.view-as-table'
            else None
        )
        self._paramstyle = 'pyformat'
        # Connector public methods route raised errors through PEP249 handlers.
        # Keep that driver behavior without opening a warehouse connection.
        self._messages = []
        self._errorhandler = Error.default_errorhandler
        self._last: pa.Table | None = None
        self.upper: set[str] = set()
        self.fail: dict[str, Exception] = {}
        #: Each statement asked: its text and parameters.
        self.calls: list[tuple[str, list[Any]]] = []

    def cursor(self) -> _Cursor:  # type: ignore[override]
        return _Cursor(self)

    def close(self, retry: bool = True) -> None:
        return None

    def answer(self, text: str, parameters: list[Any]) -> pa.Table:
        self.calls.append((text, parameters))
        if text in self.fail:
            raise self.fail[text]
        # The recordings' database and schema are stored in upper case, so
        # their upper-cased spellings name the same objects.
        for part in (DB, SCHEMA):
            text = text.replace(f'"{part.upper()}"', f'"{part}"')
        for name in self.upper:
            if f'"{name}"' in text or parameters == [name]:
                raise _missing(name)
            text = text.replace(f'"{name.upper()}"', f'"{name}"')
            parameters = [name if p == name.upper() else p for p in parameters]
        if text == 'SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()':
            assert not parameters
            return pa.table(
                {'CURRENT_DATABASE()': [DB], 'CURRENT_SCHEMA()': [SCHEMA]}
            )
        if text.lstrip().startswith('SELECT TABLE_NAME'):
            assert ' '.join(text.split()) == (
                f'SELECT TABLE_NAME FROM "{DB}".INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ?'
            )
            assert parameters == [SCHEMA]
            names = ['users', 'orders']
            if self._case not in (
                'base.binary',
                'key.past-2-53.catalog',
                'time.timestamptz',
                'snowflake.name.database-only',
            ):
                names.reverse()
            return pa.table({'TABLE_NAME': names})
        if text.startswith('SHOW TABLES'):
            assert text in (
                f'SHOW TABLES LIKE ? IN SCHEMA "{DB}"."{SCHEMA}"',
                f'SHOW TABLES IN SCHEMA "{DB}"."{SCHEMA}"',
            )
            assert len(parameters) == (1 if 'LIKE ?' in text else 0)
            return self._table_listing(parameters)
        if text.startswith('WITH TMP as ('):
            found = self._read_seeds(text, parameters)
            self._last = found
            return found
        count = re.fullmatch(
            r'SELECT COUNT\(\*\) AS "rows" FROM "[^"]+"\."[^"]+"\."([^"]+)"',
            text,
        )
        if count:
            name = count.group(1)
            found = self._table_listing([name])
            if isinstance(found, pa.Table) and found.num_rows:
                return pa.table(
                    {
                        'rows': pa.array(
                            [int(found['rows'][0].as_py())], pa.int64()
                        )
                    }
                )
            # The view case's genuine schema preview has all rows. Its
            # catalog listing is empty because SHOW TABLES excludes views.
            preview = next(
                (
                    answer
                    for (
                        query,
                        params,
                    ), answer in self._recording.answers.items()
                    if query.startswith('SELECT ')
                    and query.endswith('LIMIT 1000')
                    and f'"{SCHEMA}"."{name}"' in query
                    and params == '[]'
                ),
                None,
            )
            if isinstance(preview, pa.Table):
                return pa.table({'rows': pa.array([len(preview)], pa.int64())})
        found = self._recording.answers.get((text, json.dumps(parameters)))
        if found is None and self._view_base is not None:
            # Recorded base rows supply the reads after the old client's
            # view-count failure; only the source relation name differs.
            base_text = text.replace('"users_v"', '"users"')
            found = self._view_base.answers.get(
                (base_text, json.dumps(parameters))
            )
        if found is None:
            raise AssertionError(
                f'no recorded answer for {text!r} {parameters!r}'
            )
        if isinstance(found, Exception):
            raise found
        self._last = found
        return found

    def _table_listing(self, parameters: list[Any]) -> pa.Table:
        """Only tables appear in SHOW TABLES; catalog row counts are exact here."""
        names = list(parameters) if parameters else ['orders', 'users']
        if self._case == 'snowflake.rowcount.view-as-table':
            names = [name for name in names if name != 'users_v']
        assert all(name.lower() in ('users', 'orders') for name in names)
        return pa.table(
            {
                'name': pa.array(names, pa.string()),
                'rows': pa.array(
                    [8 if name.lower() == 'users' else 80 for name in names],
                    pa.int64(),
                ),
                'is_hybrid': pa.array(['N'] * len(names), pa.string()),
                'is_dynamic': pa.array(['N'] * len(names), pa.string()),
                'is_external': pa.array(['N'] * len(names), pa.string()),
            }
        )

    def _read_seeds(self, text: str, parameters: list[Any]) -> pa.Table:
        """Answer the shop's two supported key-read shapes from source rows.

        SQL spelling/binding is checked independently by the Rust dialect tests.
        Public tests below check labels, history and row identity from these inputs.
        """
        join = re.search(
            r'JOIN "[^" ]+"\."[^" ]+"\."([^" ]+)"\s+ON "([^" ]+)" = TMP.__KUMO_ID__',
            text,
        )
        assert join is not None, text
        name, key = join.groups()
        previews = [
            answer
            for (query, _), answer in self._recording.answers.items()
            if query.startswith('SELECT ')
            and query.endswith('LIMIT 1000')
            and f'."{name}" ' in query
            and isinstance(answer, pa.Table)
        ]
        if not previews and name == 'users_v' and self._view_base is not None:
            previews = [
                answer
                for (query, _), answer in self._view_base.answers.items()
                if query.startswith('SELECT ')
                and query.endswith('LIMIT 1000')
                and '."users" ' in query
                and isinstance(answer, pa.Table)
            ]
        assert len(previews) == 1, (name, len(previews))
        source = previews[0]
        projection = re.search(
            r'SELECT TMP.__KUMO_BATCH__ as __KUMO_BATCH__, (.*?)\nFROM TMP',
            text,
        )
        assert projection is not None, text
        columns = re.findall(r'"([^" ]+)"', projection[1].replace(_COUNTED, ''))
        seeds = json.loads(parameters[0])
        window = '__KUMO_END_TIME__' in text
        capped = re.search(r'\) <= (\d+)\s+ORDER BY', text)
        sql = ' '.join(text.split())
        assert f'JOIN "{DB}"."{SCHEMA}"."{name}"' in sql
        assert 'f.index as __KUMO_BATCH__' in sql
        assert 'FLATTEN(INPUT => PARSE_JSON(?))' in sql
        assert sql.count('?') == len(parameters)
        if window:
            assert '"ts" <= TMP.__KUMO_END_TIME__' in sql
            assert '"ts" > TMP.__KUMO_START_TIME__' in sql
            assert all(len(seed) == 3 for seed in seeds)
            assert all(
                seed[1] is not None and seed[2] is not None for seed in seeds
            )
            assert list(parameters[1:]) == [
                max(seed[1] for seed in seeds),
                min(seed[2] for seed in seeds),
            ]
        else:
            assert 'f.value::NUMBER as __KUMO_ID__' in sql
            assert len(parameters) == 1
        if capped:
            order_key = 'oid' if 'oid' in source.column_names else 'order_id'
            assert f'ORDER BY "ts" DESC, "{order_key}"' in sql
        source_rows = source.to_pylist()
        selected: list[int] = []
        batches: list[int] = []
        for batch, seed in enumerate(seeds):
            value = seed[0] if window else seed
            matches = [
                at for at, row in enumerate(source_rows) if row[key] == value
            ]
            if window:
                end = pd.Timestamp(seed[1]) if seed[1] is not None else None
                start = (
                    pd.Timestamp(seed[2])
                    if len(seed) > 2 and seed[2] is not None
                    else None
                )
                matches = [
                    at
                    for at in matches
                    if end is not None
                    and source_rows[at]['ts'] is not None
                    and pd.Timestamp(source_rows[at]['ts']) <= end
                    and (
                        start is None
                        or pd.Timestamp(source_rows[at]['ts']) > start
                    )
                ]
            if capped:
                order_key = (
                    'oid' if 'oid' in source.column_names else 'order_id'
                )
                matches.sort(
                    key=lambda at: (
                        -pd.Timestamp(source_rows[at]['ts']).value,
                        source_rows[at][order_key],
                    )
                )
                matches = matches[: int(capped[1])]
            selected.extend(matches)
            batches.extend([batch] * len(matches))
        result = source.select(columns).take(pa.array(selected, pa.int64()))
        result = result.add_column(
            0, '__KUMO_BATCH__', pa.array(batches, pa.int64())
        )
        if 'QUALIFY ROW_NUMBER() OVER (PARTITION BY' in text:
            result = _first_of_each_seed(result, _COUNTED in text)
        return result

    def asked(self, since: int = 0) -> list[str]:
        return [text for text, _ in self.calls[since:]]


#: The core's count of the rows holding each seed's key, in a lookup.
_COUNTED = (
    ', COUNT(*) OVER (PARTITION BY TMP.__KUMO_BATCH__) AS "__kumo_matches__"'
)


def _first_of_each_seed(table: pa.Table, counted: bool) -> pa.Table:
    r"""``table``, a lookup's rows, its first row of each seed, with how
    many rows each seed's key holds where ``counted``."""
    batches = table.column('__KUMO_BATCH__').to_pylist()
    seen: dict[Any, int] = {}
    held: dict[Any, int] = {}
    for at, batch in enumerate(batches):
        seen.setdefault(batch, at)
        held[batch] = held.get(batch, 0) + 1
    first = table.take(sorted(seen.values()))
    if counted:
        first = first.append_column(
            '__kumo_matches__',
            pa.array(
                [held[b] for b in first.column('__KUMO_BATCH__').to_pylist()],
                pa.int64(),
            ),
        )
    return first


class Session:
    r"""A Snowpark session, recognised by its class's name and module."""

    __module__ = 'snowflake.snowpark.session'

    def __init__(self, connection: _Warehouse) -> None:
        self.connection = connection


def _assert_schema_facts(
    schema: Schema, case: str, *, selected: set[str] | None = None
) -> None:
    """Assert the declared shop schema and variant-specific key/type facts."""
    renamed = case == 'snowflake.forecast.catalog-keys'
    composite = case == 'rule.composite-foreign-key-declared'
    user_key, order_key, foreign_key = (
        ('uid', 'oid', 'buyer')
        if renamed
        else ('user_id', 'order_id', 'user_id')
    )
    columns = {
        'users': [user_key, 'region', 'age']
        if composite
        else [user_key, 'age', 'country'],
        'orders': [order_key, foreign_key]
        + (['region'] if composite else [])
        + ['price', 'ts'],
    }
    if case == 'decimal.feature-38-0-wide':
        columns['users'].append('f')
    selected = {'users', 'orders'} if selected is None else selected
    assert set(schema.tables) == selected
    for name in selected:
        table = schema[name]
        assert table.source_name == name
        assert list(table.columns) == columns[name]
        assert table.primary_key == (user_key if name == 'users' else order_key)
        assert table.time_column == ('ts' if name == 'orders' else None)
        for column, facts in table.columns.items():
            if column in (user_key, order_key, foreign_key):
                dtype, stype = 'int64', 'ID'
            elif column in ('country', 'region'):
                dtype, stype = 'string', 'categorical'
            elif column == 'ts':
                dtype, stype = 'timestamp[us]', 'timestamp'
            elif column == 'f':
                dtype, stype = 'int64', 'categorical'
            else:
                assert column in ('age', 'price')
                dtype, stype = (
                    ('int64' if column == 'age' else 'float64'),
                    'numerical',
                )
            assert (facts.dtype, facts.stype) == (dtype, stype), (name, column)
    links = sorted(
        (r.source_table, r.source_column, r.target_table, r.target_column)
        for r in schema.relationships
    )
    assert links == (
        [('orders', foreign_key, 'users', user_key)]
        if selected == {'users', 'orders'}
        else []
    )


def _described(
    warehouse: _Warehouse,
    **given: Any,
) -> Schema:
    tables = given.pop('tables', None)
    source = SnowflakeSource(warehouse, **given)
    if tables is None:
        return Schema.from_source(source)
    return Schema.from_source(source, tables=tables)


_NAMED = {'database': DB, 'schema': SCHEMA}
_TABLES = {'users': 'users', 'orders': 'orders'}


@pytest.mark.parametrize(
    ('case', 'given'),
    [
        ('base.binary', _NAMED),
        ('snowflake.name.database-only', {'database': DB}),
        ('rule.composite-foreign-key-declared', _NAMED),
        ('snowflake.forecast.catalog-keys', _NAMED),
        ('key.past-2-53.catalog', _NAMED),
    ],
    ids=lambda value: value if isinstance(value, str) else '',
)
def test_a_source_preserves_declared_schema_facts(
    case: str, given: dict[str, Any]
) -> None:
    _assert_schema_facts(_described(_Warehouse(case), **given), case)


def test_a_source_naming_its_database_alone_reads_the_session_once() -> None:
    warehouse = _Warehouse('snowflake.name.database-only')
    _described(warehouse, database=DB)
    sessions = [
        text
        for text in warehouse.asked()
        if text == 'SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()'
    ]
    assert len(sessions) == 1
    # The tables are listed in the database named and the session's schema.
    assert warehouse.calls[1][1] == [SCHEMA]
    assert f'FROM "{DB}".INFORMATION_SCHEMA.TABLES' in warehouse.calls[1][0]


def test_a_snowpark_session_is_described_through_its_own_connection() -> None:
    warehouse = _Warehouse('base.binary')
    schema = Schema.from_source(SnowflakeSource(Session(warehouse), **_NAMED))
    _assert_schema_facts(schema, 'base.binary')
    assert warehouse.calls


#: The database and schema, each upper-cased, as a statement names them.
_UPPER = f'"{DB.upper()}"."{SCHEMA.upper()}"'


def _describes(warehouse: _Warehouse, since: int = 0) -> list[str]:
    return [
        text for text in warehouse.asked(since) if text.startswith('DESCRIBE')
    ]


def test_a_table_stored_upper_cased_is_found_upper_cased_and_read_so() -> None:
    upper = _Warehouse('snowflake.name.upper-case-folded')
    on_the_core = _described(upper, **_NAMED, tables=_TABLES)
    _assert_schema_facts(on_the_core, 'snowflake.name.upper-case-folded')
    # Each table as given, then all three parts upper-cased.
    describes = _describes(upper)
    assert describes.index(f'DESCRIBE TABLE "{DB}"."{SCHEMA}"."users"') < (
        describes.index(f'DESCRIBE TABLE {_UPPER}."USERS"')
    )
    # Every later statement names the spelling that resolved.
    after = upper.asked()[
        upper.asked().index(f'DESCRIBE TABLE {_UPPER}."ORDERS"') :
    ]
    assert not [
        text for text in after if '"users"' in text or '"orders"' in text
    ]
    assert any('"USERS"' in text for text in after)


@pytest.mark.parametrize(
    ('case', 'given'),
    [
        ('base.binary', _NAMED),
        ('key.past-2-53.catalog', _NAMED),
        ('snowflake.name.upper-case-folded', {**_NAMED, 'tables': _TABLES}),
    ],
    ids=['base.binary', 'declared-key', 'upper-case-folded'],
)
def test_each_spelling_is_described_once(
    case: str, given: dict[str, Any]
) -> None:
    warehouse = _Warehouse(case)
    _described(warehouse, **dict(given))
    asked = _describes(warehouse)
    expected = [
        f'DESCRIBE TABLE "{DB}"."{SCHEMA}"."{name}"'
        for name in ('users', 'orders')
    ]
    if case == 'snowflake.name.upper-case-folded':
        expected += [
            f'DESCRIBE TABLE {_UPPER}."{name}"' for name in ('USERS', 'ORDERS')
        ]
    assert sorted(asked) == sorted(expected)


def test_an_all_upper_name_is_tried_once() -> None:
    r"""Database, schema and table named in upper case, as the recording's
    were: the one spelling, as given."""
    warehouse = _Warehouse('snowflake.name.all-upper-once')
    with pytest.raises(StructuredDataError) as refused:
        _described(
            warehouse,
            database=DB.upper(),
            schema=SCHEMA.upper(),
            tables={'users': 'USERS', 'orders': 'orders'},
        )
    assert set(_describes(warehouse)) == {f'DESCRIBE TABLE {_UPPER}."USERS"'}
    assert refused.value.code == 'INVALID_REQUEST'
    assert (
        f"Table '{DB.upper()}.{SCHEMA.upper()}.USERS' does not exist in the "
        'remote data backend'
    ) in str(refused.value)


@pytest.mark.parametrize(
    'failure',
    [
        ProgrammingError(msg='Authentication token has expired.', errno=390114),
        ProgrammingError(
            msg='Insufficient privileges to operate on table.',
            errno=3001,
            sqlstate='42501',
        ),
    ],
    ids=['token-expired', 'no-privilege'],
)
def test_a_failure_that_is_not_a_missing_object_tries_no_other_spelling(
    failure: Exception,
) -> None:
    warehouse = _Warehouse('base.binary')
    warehouse.fail[f'DESCRIBE TABLE "{DB}"."{SCHEMA}"."users"'] = failure
    with pytest.raises(Exception):  # noqa: B017, PT011 - connector failure
        _described(warehouse, **_NAMED, tables=_TABLES)
    assert '"USERS"' not in ' '.join(_describes(warehouse))


def _missing_schema_once() -> _Warehouse:
    r"""A warehouse whose tables are stored upper-cased, each named in
    lower case failing with errno 2043 and a message saying "does not
    exist", as a missing schema fails."""
    warehouse = _Warehouse('base.binary')
    warehouse.upper = {'users', 'orders'}
    for name in warehouse.upper:
        warehouse.fail[f'DESCRIBE TABLE "{DB}"."{SCHEMA}"."{name}"'] = (
            ProgrammingError(
                msg='Schema does not exist.', errno=2043, sqlstate='02000'
            )
        )
    return warehouse


def test_a_failure_whose_message_says_does_not_exist_tries_the_upper_spelling() -> (
    None
):
    r"""The old client retries on errno 2003 or a message saying "does not
    exist" (``SNOW/table.py:34-45``), and the current client preserves that behavior."""
    warehouse = _missing_schema_once()
    schema = _described(warehouse, **_NAMED, tables=_TABLES)
    _assert_schema_facts(schema, 'base.binary')
    assert f'DESCRIBE TABLE {_UPPER}."USERS"' in _describes(warehouse)


class _MaskedHybrid(_Warehouse):
    r"""snowflake.forecast.catalog-keys' warehouse whose ``users`` is a
    hybrid table, its enforced key ``uid`` masked for the reading role to
    twelve zeroes: counted, it would repeat."""

    def __init__(self) -> None:
        super().__init__('snowflake.forecast.catalog-keys')

    def answer(self, text: str, parameters: list[Any]) -> pa.Table:
        if text == f'SHOW TABLES IN SCHEMA "{DB}"."{SCHEMA}"':
            self.calls.append((text, parameters))
            return pa.table(
                {'name': ['users', 'orders'], 'is_hybrid': ['Y', 'N']}
            )
        if 'COUNT(DISTINCT "uid")' in text:
            self.calls.append((text, parameters))
            return pa.table({'K': [0], 'N': [12], 'D': [1], 'Z': [0]})
        if 'HAVING COUNT(*) > 1' in text:
            self.calls.append((text, parameters))
            return pa.table({'uid': [0]})
        found = super().answer(text, parameters)
        if text.endswith('LIMIT 1000') and '"users"' in text:
            at = found.column_names.index('uid')
            return found.set_column(
                at,
                found.schema.field(at),
                pa.array([0] * len(found), type=found.column(at).type),
            )
        return found


def test_a_hybrid_tables_enforced_key_is_not_counted() -> None:
    r"""The old client lists the schema's hybrid tables once before it
    counts a key holding a declared primary key, and counts no key a hybrid
    table enforces (``SNOW/table.py:219-285``): a masked enforced key keeps
    its table's key, never refused as repeating."""
    warehouse = _MaskedHybrid()
    schema = _described(warehouse, **_NAMED, tables={'users': 'users'})
    _assert_schema_facts(
        schema, 'snowflake.forecast.catalog-keys', selected={'users'}
    )
    assert schema['users'].primary_key == 'uid'
    asked = warehouse.asked()
    assert asked.count(f'SHOW TABLES IN SCHEMA "{DB}"."{SCHEMA}"') == 1
    assert not [text for text in asked if 'COUNT(DISTINCT "uid")' in text]


class _NoSession(_Warehouse):
    r"""base.binary's warehouse on a connection opened with no database and
    no schema: the session answers two nulls."""

    def answer(self, text: str, parameters: list[Any]) -> pa.Table:
        if text == 'SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()':
            self.calls.append((text, parameters))
            return pa.table(
                {
                    'CURRENT_DATABASE()': pa.array([None], pa.string()),
                    'CURRENT_SCHEMA()': pa.array([None], pa.string()),
                }
            )
        return super().answer(text, parameters)


@pytest.mark.parametrize(
    ('tables', 'words'),
    [
        (
            None,
            "No current 'schema' set. Please specify the Snowflake schema "
            'manually',
        ),
        (
            _TABLES,
            f"Unspecified 'schema' for table 'users' in database '{DB}'",
        ),
    ],
    ids=['listed', 'named'],
)
def test_a_database_named_on_a_session_with_none_is_kept_and_the_schema_refused(
    tables: dict[str, str] | None,
    words: str,
) -> None:
    r"""The old client keeps the database the source names
    (``database or result[0]``, ``RE/graph.py:1030-1031``) and refuses the
    missing schema in its words (``:1035-1040``, ``SNOW/table.py:88-93``):
    so does the core, as a StructuredDataError (P3D-5)."""
    given: dict[str, Any] = {'database': DB}
    if tables is not None:
        given['tables'] = tables
    with pytest.raises(Exception) as refused:  # noqa: PT011
        _described(_NoSession(), **given)
    assert words in str(refused.value)


class _TypedEmpty(_Warehouse):
    r"""empty.fact-table's warehouse, its empty preview answered as the
    connector builds it for ``force_return_table=True``: the columns typed
    from the result's metadata, no row (``result_set.py:288``); and none for
    ``force_return_table=False``, as the old client asks."""

    def __init__(self) -> None:
        super().__init__('empty.fact-table')
        self._typed = _Warehouse('base.binary')

    def answer(self, text: str, parameters: list[Any]) -> pa.Table:
        found = super().answer(text, parameters)
        if text.endswith('LIMIT 1000') and found.num_columns == 0:
            return self._typed.answer(text, parameters).slice(0, 0)
        return found


def test_an_empty_table_is_refused_at_describe() -> None:
    r"""An empty table is refused with a structured, actionable error."""
    with pytest.raises(Exception) as refused:  # noqa: PT011
        _described(_TypedEmpty(), **_NAMED)
    assert isinstance(refused.value, StructuredDataError)
    assert refused.value.code == 'INVALID_REQUEST'
    assert 'orders' in str(refused.value)
    assert 'is empty' in str(refused.value)


def _materialize(schema: Schema, query: str | None = None) -> Any:
    return PredictiveQuery(
        query
        or 'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
        schema,
    ).materialize(indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01'))


def _materialize_reads(schema: Schema) -> None:
    r"""``materialize``, for what it reads: past the reads held here it
    may ask for a statement no recording holds."""
    try:
        _materialize(schema)
    except Exception:  # noqa: BLE001, S110 - the reads are what is held
        pass


def test_each_materialize_resolves_each_tables_spelling_again() -> None:
    r"""Plan B: a table described as given and stored upper-cased since is
    found upper-cased at the next run, and the run after reads it as given
    again once it is back; no spelling outlives its run."""
    warehouse = _Warehouse('base.binary')
    schema = _described(warehouse, **_NAMED, tables=_TABLES)
    assert '"USERS"' not in ' '.join(warehouse.asked())
    warehouse.upper = {'users'}
    mark = len(warehouse.calls)
    _materialize_reads(schema)
    describes = _describes(warehouse, mark)
    assert describes[:1] == [f'DESCRIBE TABLE "{DB}"."{SCHEMA}"."users"']
    assert f'DESCRIBE TABLE {_UPPER}."USERS"' in describes
    reads = [
        text
        for text in warehouse.asked(mark)
        if not text.startswith('DESCRIBE')
    ]
    assert not any(text.endswith('LIMIT 1000') for text in reads), reads
    assert any('"USERS"' in text for text in reads), reads
    warehouse.upper = set()
    mark = len(warehouse.calls)
    _materialize_reads(schema)
    assert '"USERS"' not in ' '.join(warehouse.asked(mark))


def test_each_materialize_reads_the_session_again() -> None:
    warehouse = _Warehouse('snowflake.name.database-only')
    schema = _described(warehouse, database=DB)
    for _ in range(2):
        mark = len(warehouse.calls)
        _materialize_reads(schema)
        assert warehouse.asked(mark)[:1] == [
            'SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()'
        ]
        assert (
            warehouse.asked(mark).count(
                'SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()'
            )
            == 1
        )


def test_row_counts_name_the_exact_object() -> None:
    warehouse = _Warehouse('base.binary')
    schema = _described(warehouse, **_NAMED)
    mark = len(warehouse.calls)
    _materialize_reads(schema)
    asked = warehouse.calls[mark:]
    for name in ('users', 'orders'):
        assert (
            f'SELECT COUNT(*) AS "rows" FROM "{DB}"."{SCHEMA}"."{name}"',
            [],
        ) in asked
    assert not any(text.startswith('SHOW TABLES LIKE') for text, _ in asked)


def test_a_view_materializes_when_counted_exactly() -> None:
    warehouse = _Warehouse('snowflake.rowcount.view-as-table')
    schema = _described(
        warehouse,
        **_NAMED,
        tables={'users': 'users_v', 'orders': 'orders'},
    )
    request = _materialize(
        schema, 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
    )
    assert request is not None
    assert any(
        'COUNT(*)' in text and '"users_v"' in text
        for text, _ in warehouse.calls
    )


def _assert_payload_facts(request: Any, case: str) -> None:
    """Check labels and sampled history against the independently defined shop rows."""
    binary = case in ('base.binary', 'snowflake.name.database-only')
    wide = case == 'decimal.feature-38-0-wide'
    assert request.payloads is not None and len(request.payloads) == 1
    payload = request.payloads[0]
    assert payload['task']['kind'] == (
        'binary_classification' if binary else 'regression'
    )
    assert payload['task']['entity_table_names'] == ['users']
    if binary:
        assert payload['task']['target']['classes'] == ['false', 'true']
        assert payload['task']['target']['positive_class'] == 'true'
    origin = pd.Timestamp('2023-01-01', tz='UTC')
    orders = [
        (i % 8, 10 + i * 1.5, origin + pd.Timedelta(days=i * 7))
        for i in range(80)
    ]
    prediction_time = pd.Timestamp('2024-03-01', tz='UTC')
    expected_context = {
        (entity, prediction_time - pd.Timedelta(days=30 * step))
        for step in range(1, 15)
        for entity in range(8)
    }
    all_instance_ids: set[int] = set()
    all_node_ids: set[int] = set()
    for stage in ('context', 'predict'):
        data = payload[stage]
        instances = data['instance_table']
        columns = ['instance_id', '__kumo_entity_ref_0', '__kumo_anchor_time']
        assert instances['columns'] == columns + (
            ['TARGET'] if stage == 'context' else []
        )
        by_instance = {
            row[0]: (row[1], pd.Timestamp(row[2])) for row in instances['rows']
        }
        assert len(by_instance) == len(instances['rows'])
        assert not all_instance_ids.intersection(by_instance)
        all_instance_ids.update(by_instance)
        if stage == 'context':
            assert len(by_instance) == len(expected_context)
            assert set(by_instance.values()) == expected_context
            for row in instances['rows']:
                entity, anchor = by_instance[row[0]]
                future = [
                    price
                    for user, price, time in orders
                    if user == entity
                    and anchor < time <= anchor + pd.Timedelta(days=30)
                ]
                assert row[3] == (bool(future) if binary else sum(future))
                assert type(row[3]) is (bool if binary else float)
        else:
            assert list(by_instance.values()) == [
                (0, prediction_time),
                (1, prediction_time),
            ]
        assert set(data['related_tables']) == {'users', 'orders'}
        users = data['related_tables']['users']
        assert users['columns'] == [
            'instance_id',
            'user_id',
            'age',
            'country',
        ] + (['f'] if wide else [])
        assert len(users['rows']) == len(by_instance)
        assert {row[0] for row in users['rows']} == set(by_instance)
        for row in users['rows']:
            entity, _ = by_instance[row[0]]
            assert row[1:4] == [
                entity,
                20 + entity,
                ['US', 'DE', 'IN'][entity % 3],
            ]
            if wide:
                assert row[4] == str(12345678901234567890123456789 + entity)
        history = data['related_tables']['orders']
        assert history['columns'] == [
            'instance_id',
            '__node_id',
            'price',
            'ts',
            'user_id',
        ]
        observed = {instance: [] for instance in by_instance}
        for instance, node, price, time, entity in history['rows']:
            assert node not in all_node_ids
            all_node_ids.add(node)
            assert entity == by_instance[instance][0]
            observed[instance].append((price, pd.Timestamp(time)))
        for instance, (entity, anchor) in by_instance.items():
            expected = [
                (price, time)
                for user, price, time in orders
                if user == entity and time <= anchor
            ]
            assert sorted(observed[instance]) == sorted(expected)
    schema = payload['schema']
    assert set(schema['related_tables']) == {'users', 'orders'}
    assert schema['instance_table']['primary_key'] == 'instance_id'
    assert payload['task']['target']['dtype'] == (
        'bool' if binary else 'float32'
    )
    assert payload['task']['anchor_time_column'] == '__kumo_anchor_time'
    for name, primary in [
        ('users', ['instance_id', 'user_id']),
        ('orders', ['instance_id', '__node_id']),
    ]:
        table = schema['related_tables'][name]
        assert table['primary_key'] == primary
        expected = {'instance_id': 'int64', 'user_id': 'int64'}
        expected.update(
            {'age': 'int64', 'country': 'string'}
            if name == 'users'
            else {
                '__node_id': 'int64',
                'price': 'float64',
                'ts': 'timestamp[us]',
            }
        )
        if name == 'users' and wide:
            expected['f'] = 'string'
        assert {
            column: facts['dtype'] for column, facts in table['columns'].items()
        } == expected
        assert 'TARGET' not in table['columns']
        stypes = {
            'instance_id': 'ID',
            'user_id': None if name == 'users' else 'ID',
        }
        stypes.update(
            {'age': 'numerical', 'country': 'categorical'}
            if name == 'users'
            else {'__node_id': 'ID', 'price': 'numerical', 'ts': 'timestamp'}
        )
        if name == 'users' and wide:
            stypes['f'] = 'categorical'
        assert {
            column: facts.get('stype')
            for column, facts in table['columns'].items()
        } == stypes
    for column, dtype, stype in [
        ('instance_id', 'int64', 'ID'),
        ('__kumo_entity_ref_0', 'int64', 'ID'),
        ('__kumo_anchor_time', 'timestamp[us]', 'timestamp'),
    ]:
        facts = schema['instance_table']['columns'][column]
        assert (facts['dtype'], facts['stype']) == (dtype, stype)

    assert schema['instance_table']['columns']['TARGET'] == {
        'dtype': 'bool' if binary else 'float32',
        'stype': 'categorical' if binary else 'numerical',
    }
    assert schema['relationships'] == [
        {
            'source_columns': ['__kumo_entity_ref_0'],
            'target_table': 'users',
            'target_columns': ['user_id'],
        },
        {
            'source_table': 'orders',
            'source_columns': ['user_id'],
            'target_table': 'users',
            'target_columns': ['user_id'],
        },
    ]


@pytest.mark.parametrize(
    ('case', 'given', 'query'),
    [
        (
            'base.binary',
            _NAMED,
            'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
        ),
        (
            'snowflake.name.upper-case-folded',
            {**_NAMED, 'tables': _TABLES},
            'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        ),
        (
            'snowflake.name.database-only',
            {'database': DB},
            'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
        ),
    ],
    ids=['base.binary', 'upper-case-folded', 'database-only'],
)
def test_materialize_preserves_source_labels_and_history(
    case: str, given: dict[str, Any], query: str
) -> None:
    schema = _described(_Warehouse(case), **dict(given))
    with warnings.catch_warnings(record=True) as shown:
        warnings.simplefilter('always')
        request = _materialize(schema, query)
    _assert_payload_facts(request, case)
    assert not [w for w in shown if 'seeded random sampling' in str(w.message)]


def test_every_statement_names_the_database_and_schema_in_double_quotes() -> (
    None
):
    warehouse = _Warehouse('base.binary')
    _described(warehouse, **_NAMED)
    for text, _ in warehouse.calls:
        for found in re.findall(r'"([^"]*)"\."([^"]*)"\."([^"]*)"', text):
            assert found[:2] == (DB, SCHEMA), text


def test_no_current_database_is_a_named_error() -> None:
    with pytest.raises(StructuredDataError) as refused:
        _described(_NoSession())
    assert refused.value.code == 'INVALID_REQUEST'
    assert "No current 'database' set" in str(refused.value)


@pytest.mark.parametrize('listing', ['failed', 'malformed', 'other-case'])
def test_hybrid_enforcement_requires_exact_optional_evidence(
    listing: str,
) -> None:
    class Warehouse(_MaskedHybrid):
        def answer(self, text: str, parameters: list[Any]) -> pa.Table:
            if text.startswith('SHOW TABLES IN SCHEMA'):
                self.calls.append((text, parameters))
                if listing == 'failed':
                    raise ProgrammingError(msg='not authorized', errno=3001)
                if listing == 'malformed':
                    return pa.table({'name': ['users']})
                return pa.table({'NAME': ['USERS'], 'IS_HYBRID': ['Y']})
            return super().answer(text, parameters)

    warehouse = Warehouse()
    with pytest.raises(Exception):  # noqa: B017, PT011 - connector failure
        _described(
            warehouse,
            **_NAMED,
            tables={'users': 'users'},
        )
    assert any('COUNT(DISTINCT "uid")' in text for text, _ in warehouse.calls)


@pytest.mark.parametrize(
    ('case', 'kind'),
    [('base.binary', 'TIMESTAMP_NTZ'), ('time.timestamptz', 'TIMESTAMP_LTZ')],
)
def test_described_timestamp_facts_keep_the_current_catalog_type(
    case: str, kind: str
) -> None:
    schema = _described(_Warehouse(case), **_NAMED)
    facts = schema['orders']['ts']._source
    assert facts is not None
    assert facts.declared_type is not None
    assert facts.declared_type.startswith(kind)


def test_a_missing_anchor_keeps_its_entity_and_other_entities_history() -> None:
    class Warehouse(_Warehouse):
        def __init__(self) -> None:
            super().__init__('sample.anchor-entity.null-time')
            preview = f'SELECT "order_id", "user_id", "price", "ts" FROM "{DB}"."{SCHEMA}"."orders" LIMIT 1000'
            self.orders = self._recording.answers[(preview, '[]')]
            self.missing_batches: set[int] = set()
            self.present_batches: set[int] = set()

        def answer(self, text: str, parameters: list[Any]) -> pa.Table:
            if '\nJOIN ' in text and f'"{SCHEMA}"."orders"' in text:
                self.calls.append((text, parameters))
                seeds = json.loads(parameters[0])
                assert not any(
                    isinstance(v, float) for row in seeds for v in row[1:]
                )
                missing = [at for at, row in enumerate(seeds) if row[1] is None]
                valid = [
                    at
                    for at, row in enumerate(seeds)
                    if row[0] == 1 and row[1] is not None
                ]
                self.missing_batches.update(missing)
                self.present_batches.update(valid)
                # Fixed fixture response for the valid user, drawn from
                # the recorded orders preview. NULL windows have no row.
                recorded = self.orders.filter(
                    pc.equal(self.orders['user_id'], 1)
                ).slice(0, 1)
                rows = (
                    pa.concat_tables([recorded] * len(valid))
                    if valid
                    else recorded.slice(0, 0)
                )
                return rows.add_column(
                    0, '__KUMO_BATCH__', pa.array(valid, pa.int64())
                )
            found = super().answer(text, parameters)
            if (
                'signup' in found.column_names
                and 'user_id' in found.column_names
            ):
                values = found['signup'].to_pylist()
                for at, user in enumerate(found['user_id'].to_pylist()):
                    if user == 1:
                        values[at] = pd.Timestamp('2024-03-01').to_pydatetime()
                field = found.schema.field('signup')
                found = found.set_column(
                    found.column_names.index('signup'),
                    field,
                    pa.array(values, field.type),
                )
            return found

    warehouse = Warehouse()
    schema = _described(warehouse, **_NAMED)
    request = PredictiveQuery(
        'PREDICT users.country FOR EACH users.user_id', schema
    ).materialize(indices=[0, 1], anchor_time='entity')
    prediction = request.payloads[0]['predict']
    assert len(prediction['instance_table']['rows']) == 2
    users = prediction['related_tables']['users']
    by_entity = {
        row[users['columns'].index('user_id')]: dict(
            zip(users['columns'], row, strict=True)
        )
        for row in users['rows']
    }
    assert set(by_entity) == {0, 1}
    assert by_entity[0]['age'] == 20
    assert by_entity[0]['signup'] is None
    assert by_entity[1]['age'] == 21
    assert by_entity[1]['signup'] == '2024-03-01T00:00:00.000000Z'
    instances = prediction['instance_table']
    anchors = {
        row[instances['columns'].index('instance_id')]: row[
            instances['columns'].index('__kumo_anchor_time')
        ]
        for row in instances['rows']
    }
    assert anchors[by_entity[0]['instance_id']] is None
    assert anchors[by_entity[1]['instance_id']] == '2024-03-01T00:00:00.000000Z'
    orders = prediction['related_tables']['orders']
    assert {
        row[orders['columns'].index('instance_id')] for row in orders['rows']
    } == {by_entity[1]['instance_id']}
    assert {
        row[orders['columns'].index('user_id')] for row in orders['rows']
    } == {1}
    assert warehouse.missing_batches
    assert warehouse.present_batches


def _capture_runner() -> Any:
    """The real functional recorder around this module's optional driver."""
    name = 'test_snowflake_from_source_public_snowflake_capture'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/runners/run_python.py'
        )
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    return sys.modules[name]


def test_public_schema_records_successful_rows_only_metadata() -> None:
    run = _capture_runner()
    warehouse = _Warehouse('base.binary')
    log = run.DriverLog()
    log.stage = 'from_source'
    captured = run._CapturedSnowflake(warehouse, log)
    schema = _described(captured, **_NAMED)
    assert set(schema.tables) == {'users', 'orders'}
    metadata = [
        statement
        for statement in log.statements
        if statement.get('fetch_errors')
    ]
    assert metadata
    for statement in metadata:
        assert statement['sql'].lstrip().startswith(('SHOW', 'DESCRIBE'))
        assert statement['error'] is None
        assert statement['fetch_errors'] == [
            {'method': 'fetch_arrow_all', 'error': 'NotSupportedError'}
        ]
        assert statement['results'][0]['method'] == 'fetchall'
    assert log.incomplete == []
    # Driver parameter style and result-column metadata are property reads,
    # not unrecorded SQL or fetch methods. No other access passes through.
    assert set(log.passed) == {'_paramstyle', 'description'}
    assert any(event['event'] == 'close' for event in log.events)


def test_public_materialize_keeps_the_reference_large_key_refusal() -> None:
    from decimal import Decimal

    warehouse = _Warehouse('snowflake.key.number-38-0.past-int64')
    schema = _described(warehouse, **_NAMED)
    query = PredictiveQuery(
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
        schema,
    )
    with pytest.raises(StructuredDataError) as raised:
        query.materialize(
            indices=[Decimal(10**25), Decimal(10**25 + 1)],
            anchor_time=pd.Timestamp('2024-03-01'),
        )
    assert raised.value.code == 'INVALID_REQUEST'
    assert raised.value._sentence == 'key_set_integer_range'
    assert 'SQL key binding' in str(raised.value)
    assert 'signed 64-bit range' in str(raised.value)


@pytest.mark.parametrize('misaligned', [False, True])
def test_public_wide_decimal_feature_keeps_the_reference_journey(
    misaligned: bool,
) -> None:
    class Warehouse(_Warehouse):
        def __init__(self) -> None:
            super().__init__('decimal.feature-38-0-wide')
            self.decimal_answers = 0

        def answer(self, text: str, parameters: list[Any]) -> pa.Table:
            found = super().answer(text, parameters)
            if 'f' not in found.column_names:
                return found
            column = found['f'].combine_chunks()
            assert column.type == pa.decimal128(38, 0)
            original = column.buffers()[1]
            storage = pa.allocate_buffer(original.size + 16)
            offset = 8 if misaligned else 0
            memoryview(storage).cast('B')[offset : offset + original.size] = (
                memoryview(original).cast('B')
            )
            copied = storage.slice(offset, original.size)
            assert copied.address % 16 == (8 if misaligned else 0)
            replacement = pa.Array.from_buffers(
                column.type,
                len(column),
                [column.buffers()[0], copied],
                null_count=column.null_count,
                offset=column.offset,
            )
            assert replacement.equals(column)
            self.decimal_answers += 1
            return found.set_column(
                found.schema.get_field_index('f'),
                found.schema.field('f'),
                replacement,
            )

    endpoint = _capture_runner()._sibling('scripted_endpoint')
    warehouse = Warehouse()
    schema = _described(warehouse, **_NAMED)
    request = _materialize(
        schema,
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
    )
    _assert_schema_facts(schema, 'decimal.feature-38-0-wide')
    _assert_payload_facts(request, 'decimal.feature-38-0-wide')
    assert request.payloads[0]['schema']['related_tables']['users']['columns'][
        'f'
    ] == {'dtype': 'string', 'stype': 'categorical'}
    assert warehouse.decimal_answers
    users = request.payloads[0]['predict']['related_tables']['users']
    feature = users['columns'].index('f')
    key = users['columns'].index('user_id')
    assert {row[key]: row[feature] for row in users['rows']} == {
        0: '12345678901234567890123456789',
        1: '12345678901234567890123456790',
    }
    with endpoint.serve() as (url, received):
        with SDMClient(url, max_retries=0) as client:
            result = client.predict(request, model=KumoRelational())
        assert result.to_pandas()['id'].tolist() == [0, 1]
        assert len(received.requests) == 1
