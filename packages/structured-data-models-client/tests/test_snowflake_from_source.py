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

import json
import re
import warnings
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
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
                data = (directory / statement['result']).read_bytes()
                self.answers.setdefault(
                    key, pa.ipc.open_stream(pa.py_buffer(data)).read_all()
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
        found = self._recording.answers.get((text, json.dumps(parameters)))
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
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


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


def test_only_errno_2003_tries_another_spelling_on_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""The old client also retries a failure whose text says "does not
    exist" (``SNOW/table.py:34-45``); the core keys on errno 2003 alone
    (registered, proposed: design S3)."""
    warehouse = _Warehouse('base.binary')
    warehouse.fail[f'DESCRIBE TABLE "{DB}"."{SCHEMA}"."users"'] = (
        ProgrammingError(msg='Schema does not exist.', errno=2043)
    )
    with pytest.raises(StructuredDataError):
        _described('core', monkeypatch, warehouse, **_NAMED, tables=_TABLES)
    assert '"USERS"' not in ' '.join(_describes(warehouse))


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


def test_a_row_count_is_read_from_the_old_clients_listing_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""``SHOW TABLES LIKE ?``, the table's name bound, and nothing more: the
    count is its first row's ``rows`` (``SNOW/table.py:287-297``)."""
    warehouse = _Warehouse('base.binary')
    schema = _described('core', monkeypatch, warehouse, **_NAMED)
    mark = len(warehouse.calls)
    _materialize_reads(schema)
    asked = warehouse.calls[mark:]
    listing = f'SHOW TABLES LIKE ? IN SCHEMA "{DB}"."{SCHEMA}"'
    for name in ('users', 'orders'):
        assert (listing, [name]) in asked
    assert ROWS_OF_THE_LISTING not in [text for text, _ in asked]


def test_a_view_listed_as_no_table_is_refused_where_its_rows_are_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""``SHOW TABLES LIKE`` lists no view, so the count reads no row: an
    internal error on both routes, as the old client's bare assert is
    (``SNOW/table.py:295-296``); the words differ (P3D-3)."""
    codes = {}
    for route in ('core', 'old'):
        warehouse = _Warehouse('snowflake.rowcount.view-as-table')
        schema = _described(
            route,
            monkeypatch,
            warehouse,
            **_NAMED,
            tables={'users': 'users_v', 'orders': 'orders'},
        )
        with pytest.raises(StructuredDataError) as refused:
            _materialize(
                schema,
                'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id',
            )
        codes[route] = refused.value.code
    assert codes == {'core': 'INTERNAL_ERROR', 'old': 'INTERNAL_ERROR'}


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
    assert sent['core'] == sent['old']
    # ``SAMPLE ROW`` takes no seed, and the old client says so once.
    assert sent['core'][1] == [
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
