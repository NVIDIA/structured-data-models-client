# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""A Snowflake source through the public API on both routes (T3.6), the
warehouse a connector connection answering the old client's recorded
statements (T3.20's ``contract/captures``, Snowflake leg): each statement
by its text and parameters, ``SHOW`` and ``DESCRIBE`` as rows only, as the
connector gives them, a recorded failure as the connector raises it.

- ``Schema.from_source`` describes the source on the core as on the old
  route: both parts named, the database alone (the session read once,
  ``SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()``), a Snowpark session.
- The spelling retry (``SNOW/table.py:123-158``): a table named in lower
  case and stored in upper case is found upper-cased, and every later
  statement of the run names it so; an all-upper name is tried once; a
  failure that is not errno 2003 (a login, a permission) tries no second
  spelling.
- Plan B (the owner, 2026-10-24): each run reads the session and resolves
  each table's spelling again; nothing outlives the run.
- ``materialize`` on the core sends the old route's payloads.
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
from snowflake.connector.errors import NotSupportedError, ProgrammingError

from sdm_client import PredictiveQuery, Schema, StructuredDataError, _routing

ROOT = Path(__file__).resolve().parents[3]
CAPTURES = ROOT / 'contract' / 'captures'

DB = '{catalog}'
SCHEMA = '{schema}'

#: A second statement reading the listing's ``rows``, which the core no
#: longer sends: it reads the count from the listing, as the old client does.
ROWS_OF_THE_LISTING = (
    'SELECT "rows" FROM TABLE(RESULT_SCAN(LAST_QUERY_ID())) LIMIT 1'
)


def _capture_dir(case: str) -> Path:
    for root in (CAPTURES / 'snowflake', CAPTURES / 's2' / 'snowflake'):
        if (root / case / 'capture.json').exists():
            return root / case
    raise AssertionError(f'no capture of {case}')


def _missing(text: str) -> ProgrammingError:
    r"""The connector's error for an object Snowflake cannot resolve."""
    return ProgrammingError(
        msg=f'SQL compilation error: {text} does not exist or not authorized.',
        errno=2003,
        sqlstate='02000',
    )


class _Recording:
    r"""One case's recorded statements: each answer by text and parameters,
    and the old client's request payloads."""

    def __init__(self, case: str) -> None:
        directory = _capture_dir(case)
        capture = json.loads(
            (directory / 'capture.json').read_text(encoding='utf-8')
        )
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
        self._recording = _Recording(case)
        self._view_base = (
            _Recording('base.regression')
            if case == 'snowflake.rowcount.view-as-table'
            else None
        )
        self._paramstyle = 'pyformat'
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
        count = re.fullmatch(
            r'SELECT COUNT\(\*\) AS "rows" FROM "[^"]+"\."[^"]+"\."([^"]+)"',
            text,
        )
        if count:
            name = count.group(1)
            listing = f'SHOW TABLES LIKE ? IN SCHEMA "{DB}"."{SCHEMA}"'
            found = self._recording.answers.get((listing, json.dumps([name])))
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
        # The core keeps a seed's first row in the statement, as the old
        # client keeps it after its fetch; answered as Snowflake does, from
        # the old client's plain join recorded.
        first = re.search(
            r'\nQUALIFY ROW_NUMBER\(\) OVER \(PARTITION BY TMP\.__KUMO_BATCH__ '
            r'ORDER BY [^\n]*\) = 1$',
            text,
        )
        counted = _COUNTED in text
        if first is not None:
            text = text[: first.start()].replace(_COUNTED, '')
        found = self._recording.answers.get((text, json.dumps(parameters)))
        if first is not None and isinstance(found, pa.Table):
            found = _first_of_each_seed(found, counted)
        if found is None and self._view_base is not None:
            # Recorded base rows supply the reads after the old client's
            # view-count failure; only the source relation name differs.
            base_text = text.replace('"users_v"', '"users"')
            found = self._view_base.answers.get(
                (base_text, json.dumps(parameters))
            )
            if first is not None and isinstance(found, pa.Table):
                found = _first_of_each_seed(found, counted)
        if found is None:
            raise AssertionError(
                f'no recorded answer for {text!r} {parameters!r}'
            )
        if isinstance(found, Exception):
            raise found
        self._last = found
        return found

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


def _core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        _routing.ENVIRONMENT,
        ','.join(
            _routing.FROM_SOURCE_SNOWFLAKE
            + _routing.FROM_SOURCE_SNOWFLAKE_TABLES
            + _routing.MATERIALIZE_SNOWFLAKE
        ),
    )


def _old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, '')


def _projection(schema: Schema) -> dict[str, Any]:
    return {
        'tables': {
            name: {
                'source_name': table.source_name,
                'columns': [
                    [c, col.dtype, col.stype]
                    for c, col in table.columns.items()
                ],
                'primary_key': table.primary_key,
                'time_column': table.time_column,
            }
            for name, table in schema.tables.items()
        },
        'links': sorted(
            [r.source_table, r.source_column, r.target_table, r.target_column]
            for r in schema.relationships
        ),
    }


def _described(
    route: str,
    monkeypatch: pytest.MonkeyPatch,
    warehouse: _Warehouse,
    **given: Any,
) -> Schema:
    (_core if route == 'core' else _old)(monkeypatch)
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
def test_a_source_is_described_on_the_core_as_on_the_old_route(
    case: str, given: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    on_the_core = _described('core', monkeypatch, _Warehouse(case), **given)
    on_the_old_route = _described('old', monkeypatch, _Warehouse(case), **given)
    assert _projection(on_the_core) == _projection(on_the_old_route)


def test_a_source_naming_its_database_alone_reads_the_session_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warehouse = _Warehouse('snowflake.name.database-only')
    _described('core', monkeypatch, warehouse, database=DB)
    sessions = [
        text
        for text in warehouse.asked()
        if text == 'SELECT CURRENT_DATABASE(), CURRENT_SCHEMA()'
    ]
    assert len(sessions) == 1
    # The tables are listed in the database named and the session's schema.
    assert warehouse.calls[1][1] == [SCHEMA]
    assert f'FROM "{DB}".INFORMATION_SCHEMA.TABLES' in warehouse.calls[1][0]


def test_a_snowpark_session_is_described_through_its_own_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warehouse = _Warehouse('base.binary')
    _core(monkeypatch)
    on_the_core = Schema.from_source(
        SnowflakeSource(Session(warehouse), **_NAMED)
    )
    _old(monkeypatch)
    on_the_old_route = Schema.from_source(
        SnowflakeSource(Session(_Warehouse('base.binary')), **_NAMED)
    )
    assert _projection(on_the_core) == _projection(on_the_old_route)
    assert warehouse.calls


#: The database and schema, each upper-cased, as a statement names them.
_UPPER = f'"{DB.upper()}"."{SCHEMA.upper()}"'


def _describes(warehouse: _Warehouse, since: int = 0) -> list[str]:
    return [
        text for text in warehouse.asked(since) if text.startswith('DESCRIBE')
    ]


def test_a_table_stored_upper_cased_is_found_upper_cased_and_read_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upper = _Warehouse('snowflake.name.upper-case-folded')
    on_the_core = _described(
        'core', monkeypatch, upper, **_NAMED, tables=_TABLES
    )
    on_the_old_route = _described(
        'old',
        monkeypatch,
        _Warehouse('snowflake.name.upper-case-folded'),
        **_NAMED,
        tables=_TABLES,
    )
    assert _projection(on_the_core) == _projection(on_the_old_route)
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
def test_each_spelling_is_described_once_as_on_the_old_route(
    case: str, given: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    r"""One DESCRIBE TABLE per spelling tried, its columns and its keys
    read from that one answer (``SNOW/table.py:123-170``)."""
    asked = {}
    for route in ('core', 'old'):
        warehouse = _Warehouse(case)
        _described(route, monkeypatch, warehouse, **dict(given))
        asked[route] = _describes(warehouse)
    assert asked['core'] == asked['old']


def test_an_all_upper_name_is_tried_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Database, schema and table named in upper case, as the recording's
    were: the one spelling, as given."""
    refusals = {}
    for route in ('core', 'old'):
        warehouse = _Warehouse('snowflake.name.all-upper-once')
        with pytest.raises(StructuredDataError) as refused:
            _described(
                route,
                monkeypatch,
                warehouse,
                database=DB.upper(),
                schema=SCHEMA.upper(),
                tables={'users': 'USERS', 'orders': 'orders'},
            )
        refusals[route] = (refused.value.code, str(refused.value))
        assert set(_describes(warehouse)) == {
            f'DESCRIBE TABLE {_UPPER}."USERS"'
        }
    assert refusals['core'] == refusals['old']
    assert refusals['core'][0] == 'INVALID_REQUEST'
    assert (
        f"Table '{DB.upper()}.{SCHEMA.upper()}.USERS' does not exist in the "
        'remote data backend'
    ) in refusals['core'][1]


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
@pytest.mark.parametrize('route', ['core', 'old'])
def test_a_failure_that_is_not_a_missing_object_tries_no_other_spelling(
    route: str, failure: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    warehouse = _Warehouse('base.binary')
    warehouse.fail[f'DESCRIBE TABLE "{DB}"."{SCHEMA}"."users"'] = failure
    with pytest.raises(Exception):  # noqa: B017, PT011 - each route's own
        _described(route, monkeypatch, warehouse, **_NAMED, tables=_TABLES)
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


def test_a_failure_whose_message_says_does_not_exist_tries_the_upper_spelling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""The old client retries on errno 2003 or a message saying "does not
    exist" (``SNOW/table.py:34-45``), and so does the core: both routes
    find the tables upper-cased and describe them alike."""
    described = {}
    for route in ('core', 'old'):
        warehouse = _missing_schema_once()
        described[route] = _projection(
            _described(route, monkeypatch, warehouse, **_NAMED, tables=_TABLES)
        )
        assert f'DESCRIBE TABLE {_UPPER}."USERS"' in _describes(warehouse)
    assert described['core'] == described['old']


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


def test_a_hybrid_tables_enforced_key_is_not_counted_on_either_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""The old client lists the schema's hybrid tables once before it
    counts a key holding a declared primary key, and counts no key a hybrid
    table enforces (``SNOW/table.py:219-285``): a masked enforced key keeps
    its table's key on both routes, never refused as repeating."""
    described = {}
    for route in ('core', 'old'):
        warehouse = _MaskedHybrid()
        schema = _described(
            route, monkeypatch, warehouse, **_NAMED, tables={'users': 'users'}
        )
        described[route] = _projection(schema)
        assert schema['users'].primary_key == 'uid'
        asked = warehouse.asked()
        assert asked.count(f'SHOW TABLES IN SCHEMA "{DB}"."{SCHEMA}"') == 1
        assert not [text for text in asked if 'COUNT(DISTINCT "uid")' in text]
    assert described['core'] == described['old']


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
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""The old client keeps the database the source names
    (``database or result[0]``, ``RE/graph.py:1030-1031``) and refuses the
    missing schema in its words (``:1035-1040``, ``SNOW/table.py:88-93``):
    so does the core, as a StructuredDataError (P3D-5)."""
    said = {}
    for route in ('core', 'old'):
        given: dict[str, Any] = {'database': DB}
        if tables is not None:
            given['tables'] = tables
        with pytest.raises(Exception) as refused:  # noqa: PT011
            _described(route, monkeypatch, _NoSession(), **given)
        said[route] = str(refused.value)
    assert words in said['core']
    assert words in said['old']


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


def test_an_empty_table_is_refused_at_describe_on_both_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Parity row 26: an empty table is refused at describe on both routes,
    its words the old client's; the class and code differ (P3D-15)."""
    raised = {}
    for route in ('core', 'old'):
        with pytest.raises(Exception) as refused:  # noqa: PT011
            _described(route, monkeypatch, _TypedEmpty(), **_NAMED)
        raised[route] = refused.value
    assert f"Table '{DB}.{SCHEMA}.orders' is empty" in str(raised['old'])
    assert type(raised['old']).__name__ == 'RuntimeError'
    assert isinstance(raised['core'], StructuredDataError)
    assert raised['core'].code == 'INVALID_REQUEST'
    assert 'orders' in str(raised['core'])
    assert 'is empty' in str(raised['core'])


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


def test_each_materialize_resolves_each_tables_spelling_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""Plan B: a table described as given and stored upper-cased since is
    found upper-cased at the next run, and the run after reads it as given
    again once it is back; no spelling outlives its run."""
    warehouse = _Warehouse('base.binary')
    schema = _described(
        'core', monkeypatch, warehouse, **_NAMED, tables=_TABLES
    )
    assert '"USERS"' not in ' '.join(warehouse.asked())
    warehouse.upper = {'users'}
    mark = len(warehouse.calls)
    _materialize_reads(schema)
    describes = _describes(warehouse, mark)
    assert describes[:1] == [f'DESCRIBE TABLE "{DB}"."{SCHEMA}"."users"']
    assert f'DESCRIBE TABLE {_UPPER}."USERS"' in describes
    previews = [
        text for text in warehouse.asked(mark) if text.endswith('LIMIT 1000')
    ]
    assert any('"USERS"' in text for text in previews), previews
    warehouse.upper = set()
    mark = len(warehouse.calls)
    _materialize_reads(schema)
    assert '"USERS"' not in ' '.join(warehouse.asked(mark))


def test_each_materialize_reads_the_session_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warehouse = _Warehouse('snowflake.name.database-only')
    schema = _described('core', monkeypatch, warehouse, database=DB)
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


def test_row_counts_name_the_exact_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warehouse = _Warehouse('base.binary')
    schema = _described('core', monkeypatch, warehouse, **_NAMED)
    mark = len(warehouse.calls)
    _materialize_reads(schema)
    asked = warehouse.calls[mark:]
    for name in ('users', 'orders'):
        assert (
            f'SELECT COUNT(*) AS "rows" FROM "{DB}"."{SCHEMA}"."{name}"',
            [],
        ) in asked
    assert not any(text.startswith('SHOW TABLES LIKE') for text, _ in asked)


def test_a_view_materializes_when_counted_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warehouse = _Warehouse('snowflake.rowcount.view-as-table')
    schema = _described(
        'core',
        monkeypatch,
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


def _payloads(request: Any) -> str:
    return json.dumps(request.payloads, sort_keys=True, default=str)


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
def test_materialize_on_the_core_sends_the_old_routes_payloads(
    case: str,
    given: dict[str, Any],
    query: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent = {}
    for route in ('core', 'old'):
        schema = _described(route, monkeypatch, _Warehouse(case), **dict(given))
        with warnings.catch_warnings(record=True) as shown:
            warnings.simplefilter('always')
            sent[route] = (
                _payloads(_materialize(schema, query)),
                # The core's warnings carry their code first.
                sorted(
                    re.sub(r'^\[[A-Z_]+\] ', '', str(w.message))
                    for w in shown
                    if 'seeded random sampling' in str(w.message)
                ),
            )
    assert sent['core'][0] == sent['old'][0]
    assert sent['core'][1] == []
    # ``SAMPLE ROW`` takes no seed, and the old client says so once.
    assert sent['old'][1] == [
        "The 'snowflake' backend does not support seeded random sampling, "
        "so 'random_seed' is ignored when drawing in-context examples. "
        'Repeated calls with the same seed may return different predictions.'
    ]


def test_every_statement_names_the_database_and_schema_in_double_quotes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    warehouse = _Warehouse('base.binary')
    _described('core', monkeypatch, warehouse, **_NAMED)
    for text, _ in warehouse.calls:
        for found in re.findall(r'"([^"]*)"\."([^"]*)"\."([^"]*)"', text):
            assert found[:2] == (DB, SCHEMA), text


def test_no_current_database_is_a_named_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(StructuredDataError) as refused:
        _described('core', monkeypatch, _NoSession())
    assert refused.value.code == 'INVALID_REQUEST'
    assert "No current 'database' set" in str(refused.value)


@pytest.mark.parametrize('listing', ['failed', 'malformed', 'other-case'])
def test_hybrid_enforcement_requires_exact_optional_evidence(
    listing: str, monkeypatch: pytest.MonkeyPatch
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

    for route in ('core', 'old'):
        warehouse = Warehouse()
        with pytest.raises(Exception):  # noqa: B017, PT011 - each route's error
            _described(
                route,
                monkeypatch,
                warehouse,
                **_NAMED,
                tables={'users': 'users'},
            )
        assert any(
            'COUNT(DISTINCT "uid")' in text for text, _ in warehouse.calls
        )


@pytest.mark.parametrize(
    ('case', 'kind'),
    [('base.binary', 'TIMESTAMP_NTZ'), ('time.timestamptz', 'TIMESTAMP_LTZ')],
)
def test_described_timestamp_facts_keep_the_current_catalog_type(
    case: str, kind: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    schema = _described('core', monkeypatch, _Warehouse(case), **_NAMED)
    facts = schema['orders']['ts']._source
    assert facts is not None
    assert facts.declared_type is not None
    assert facts.declared_type.startswith(kind)


def test_a_missing_anchor_keeps_its_entity_and_other_entities_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Warehouse(_Warehouse):
        def __init__(self) -> None:
            super().__init__('sample.anchor-entity.null-time')
            preview = f'SELECT "order_id", "user_id", "price", "ts" FROM "{DB}"."{SCHEMA}"."orders" LIMIT 1000'
            self.orders = self._recording.answers[(preview, '[]')]
            # Moving user 1's signup to March below moves its context seed
            # first. The recorded lookup result is ordered by source ID,
            # not by batch, so retain its rows and remap only the batch IDs.
            for (lookup, parameters), answer in list(
                self._recording.answers.items()
            ):
                if (
                    'f.value::NUMBER as __KUMO_ID__' not in lookup
                    or '\nJOIN ' not in lookup
                ):
                    continue
                assert json.loads(json.loads(parameters)[0]) == [
                    7,
                    6,
                    5,
                    4,
                    3,
                    2,
                    1,
                    0,
                    1,
                ]
                positions = [1, 2, 3, 4, 5, 6, 0, 7, 8]
                batches = [
                    positions[int(at)] for at in answer.column(0).to_pylist()
                ]
                remapped = answer.set_column(
                    0,
                    answer.schema.field(0),
                    pa.array(batches, answer.column(0).type),
                )
                reordered = json.dumps(
                    [json.dumps([1, 7, 6, 5, 4, 3, 2, 0, 1])]
                )
                self._recording.answers[(lookup, reordered)] = remapped
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
    schema = _described('core', monkeypatch, warehouse, **_NAMED)
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
    name = 't310_public_snowflake_capture'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/runners/run_python.py'
        )
        assert spec is not None and spec.loader is not None
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
    return sys.modules[name]


def test_public_schema_records_successful_rows_only_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = _capture_runner()
    warehouse = _Warehouse('base.binary')
    log = run.DriverLog()
    log.stage = 'from_source'
    captured = run._CapturedSnowflake(warehouse, log)
    schema = _described('core', monkeypatch, captured, **_NAMED)
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


def test_public_materialize_keeps_the_reference_large_key_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from decimal import Decimal

    warehouse = _Warehouse('snowflake.key.number-38-0.past-int64')
    schema = _described('core', monkeypatch, warehouse, **_NAMED)
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
