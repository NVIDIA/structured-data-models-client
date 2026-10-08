# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The live backends' calls go through the parity tape, offline.

The Databricks and Snowflake cells are recorded only with credentials, so a
call the tape cannot take, such as ``execute(sql, parameters=...)`` or
``fetch_arrow_all(force_return_table=True)``, would otherwise surface on the
first credentialed run. Here the engine's own Databricks and Snowflake tables
and samplers run over stub drivers, whose ``execute`` and fetch methods have
the real drivers' signatures and answer from a small scripted shop, three
ways:

* directly over the stub, as production would;
* through a recording facade over the stub;
* through a replaying facade, from what was recorded, with no stub at all.

All three must give the same answers, the stub must see exactly the calls it
saw directly, and the replay must take every recorded step.
"""

from __future__ import annotations

import inspect
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts' / 'parity'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from _parity.drivers import facade  # noqa: E402
from _parity.tape import RecordingTape, ReplayingTape  # noqa: E402
from _parity.transcript import load, save  # noqa: E402

# The shop the stubs answer from.

USERS = pa.table(
    {
        'user_id': pa.array([1, 2, 3, 4], pa.int64()),
        'country': ['DE', 'FR', 'DE', 'US'],
    }
)
ORDERS = pa.table(
    {
        'order_id': pa.array([10, 11, 12, 13, 14], pa.int64()),
        'user_id': pa.array([1, 1, 2, 3, 3], pa.int64()),
        'amount': [5.0, 7.5, 2.0, 9.0, 1.5],
        'ordered_at': pa.array(
            pd.to_datetime(
                [
                    '2024-01-01 08:00',
                    '2024-01-03 09:00',
                    '2024-01-03 09:00',
                    '2024-02-01 10:00',
                    '2024-02-11 11:00',
                ]
            ),
            pa.timestamp('us'),
        ),
    }
)
TABLES = {'users': USERS, 'orders': ORDERS}
KEYS = {'users': 'user_id', 'orders': 'order_id'}
TYPES = {
    pa.int64(): ('bigint', 'NUMBER(38,0)'),
    pa.float64(): ('double', 'FLOAT'),
    pa.string(): ('string', 'VARCHAR(16777216)'),
    pa.timestamp('us'): ('timestamp_ntz', 'TIMESTAMP_NTZ(9)'),
}


class Refused(Exception):
    r"""What a stub raises for SQL it has no answer for."""


def _bind(method: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
    r"""The arguments of a call, bound as the real driver's method binds
    them: a call the real driver would refuse fails here too."""
    return dict(inspect.signature(method).bind(None, *args, **kwargs).arguments)


class StubCursor:
    r"""A cursor that answers from the shop, and logs every call as made."""

    #: The real driver's cursor class, whose signatures calls are bound to.
    real: Any

    def __init__(self, connection: StubConnection) -> None:
        self._connection = connection
        self._table: pa.Table | None = None
        self._error: Exception | None = None
        self.description: list[tuple[Any, ...]] | None = None

    def __enter__(self) -> StubCursor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        pass

    def _log(self, name: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self._connection.calls.append((name, repr(args), repr(kwargs)))
        return _bind(getattr(self.real, name), *args, **kwargs)

    def _run(self, sql: str, params: Any) -> None:
        self._table = None
        self.description = None
        self._table = self._connection.answer(sql, params)
        self.description = [
            (field.name, self._connection.type_code(field.type)) + (None,) * 5
            for field in self._table.schema
        ]

    def _rows(self) -> list[tuple[Any, ...]]:
        assert self._table is not None
        columns = [column.to_pylist() for column in self._table.columns]
        return list(zip(*columns, strict=True))

    def fetchall(self, *args: Any, **kwargs: Any) -> list[tuple[Any, ...]]:
        self._log('fetchall', *args, **kwargs)
        return self._rows()

    def fetchone(self, *args: Any, **kwargs: Any) -> tuple[Any, ...] | None:
        self._log('fetchone', *args, **kwargs)
        rows = self._rows()
        return rows[0] if rows else None


class StubDatabricksCursor(StubCursor):
    @property
    def real(self) -> Any:
        from databricks.sql.client import Cursor

        return Cursor

    def execute(self, *args: Any, **kwargs: Any) -> StubDatabricksCursor:
        bound = self._log('execute', *args, **kwargs)
        self._run(bound['operation'], bound.get('parameters'))
        return self

    def fetchall_arrow(self, *args: Any, **kwargs: Any) -> pa.Table:
        self._log('fetchall_arrow', *args, **kwargs)
        assert self._table is not None
        return self._table


class StubSnowflakeCursor(StubCursor):
    @property
    def real(self) -> Any:
        from snowflake.connector.cursor import SnowflakeCursor

        return SnowflakeCursor

    def execute(self, *args: Any, **kwargs: Any) -> StubSnowflakeCursor:
        bound = self._log('execute', *args, **kwargs)
        self._statement = ' '.join(bound['command'].split())
        self._run(bound['command'], bound.get('params'))
        return self

    def fetch_arrow_all(self, *args: Any, **kwargs: Any) -> pa.Table | None:
        bound = self._log('fetch_arrow_all', *args, **kwargs)
        if self._statement.startswith(('SHOW', 'DESC')):
            # As the connector does: a metadata result has no Arrow form.
            from snowflake.connector.errors import NotSupportedError

            raise NotSupportedError(
                msg='Not supported for metadata statements', errno=255001
            )
        assert self._table is not None
        if self._table.num_rows == 0 and not bound.get('force_return_table'):
            return None
        return self._table


# Answers.


def _source(sql: str) -> str:
    r"""The shop table a statement reads ``FROM``."""
    match = re.search(r'FROM\s+[^\s]*?\b(users|orders)\b', sql, re.IGNORECASE)
    if match is None:
        raise Refused(sql)
    return match.group(1).lower()


def _projected(
    select_list: str, table: str, tables: dict[str, pa.Table] = TABLES
) -> list[str]:
    r"""The shop columns a ``SELECT`` list names, in order."""
    names = re.findall(r'[`"](\w+)[`"](?![.\w`"])', select_list)
    known = {name.lower(): name for name in tables[table].column_names}
    return [known[name.lower()] for name in names if name.lower() in known]


def _columns(
    table: str, names: list[str], tables: dict[str, pa.Table] = TABLES
) -> dict[str, pa.Array]:
    data = tables[table]
    return {name: data.column(name).combine_chunks() for name in names}


def _sample(sql: str, tables: dict[str, pa.Table] = TABLES) -> pa.Table:
    r"""An inference sample: the order terms first, then the columns."""
    table = _source(sql)
    head = sql.split(' FROM ', 1)[0]
    terms = re.findall(r'AS (__kumo_order_\d+__)', head)
    rest = head.rsplit(terms[-1], 1)[1] if terms else head
    names = _projected(rest, table, tables)
    rows = tables[table].num_rows
    out = {term: pa.array([0] * rows, pa.int64()) for term in terms}
    out.update(_columns(table, names, tables))
    return pa.table(out)


def _by_ids(
    table: str,
    ids: list[Any],
    names: list[str],
    tables: dict[str, pa.Table] = TABLES,
) -> pa.Table:
    r"""The rows holding ``ids``, under each id's position as its batch."""
    data = tables[table]
    key = data.column(0).to_pylist()
    picked = [(pos, key.index(i)) for pos, i in enumerate(ids) if i in key]
    out = {'__KUMO_BATCH__': pa.array([p for p, _ in picked], pa.int64())}
    for name, column in _columns(table, names, tables).items():
        out[name] = column.take(pa.array([r for _, r in picked], pa.int64()))
    return pa.table(out)


def _min_max(sql: str, tables: dict[str, pa.Table] = TABLES) -> pa.Table:
    rows = []
    for select in sql.split('UNION ALL'):
        found = re.search(r'(\d+) as table_index', select)
        assert found is not None, sql
        values = tables[_source(select)].column(3).to_pylist()
        rows.append((int(found.group(1)), min(values), max(values)))
    return pa.table(
        {
            'table_index': pa.array([r[0] for r in rows], pa.int64()),
            'min_date': pa.array([r[1] for r in rows], pa.timestamp('us')),
            'max_date': pa.array([r[2] for r in rows], pa.timestamp('us')),
        }
    )


def databricks_answer(sql: str, params: Any) -> pa.Table:
    flat = ' '.join(sql.split())
    if 'information_schema.columns' in flat:
        schema, table = (name.lower() for name in params)
        data = TABLES[table]
        n = data.num_columns
        return pa.table(
            {
                'column_name': data.column_names,
                'full_data_type': [TYPES[f.type][0] for f in data.schema],
                'is_nullable': [
                    'NO' if name == KEYS[table] else 'YES'
                    for name in data.column_names
                ],
                'table_catalog': ['main'] * n,
                'table_schema': [schema] * n,
                'table_name': [table] * n,
            }
        )
    if "constraint_type = 'FOREIGN KEY'" in flat:
        found = params[1].lower() == 'orders'
        return pa.table(
            {
                'constraint_name': ['orders_users_fk'] * found,
                'column_name': ['user_id'] * found,
                'table_catalog': ['main'] * found,
                'table_schema': ['shop'] * found,
                'table_name': ['users'] * found,
                'ref_column_name': ['user_id'] * found,
            }
        )
    if 'information_schema.table_constraints' in flat:
        table = params[1].lower()
        return pa.table(
            {
                'constraint_type': ['PRIMARY KEY'],
                'constraint_name': [f'{table}_pk'],
                'column_name': [KEYS[table]],
            }
        )
    if flat.startswith('SELECT COUNT(*) FROM'):
        return pa.table({'n': [TABLES[_source(flat)].num_rows]})
    if '__kumo_order_' in flat:
        return _sample(flat)
    if 'as table_index' in flat:
        return _min_max(flat)
    if 'WITH TMP AS' in flat:
        joined, selected = (
            flat.split(' JOIN ', 1)[1],
            flat.split(' FROM TMP')[0],
        )
        table = _source('FROM ' + joined)
        names = _projected(selected.split('__KUMO_BATCH__,', 1)[1], table)
        return _by_ids(table, json.loads(params[0]), names)
    if 'posexplode' in flat:
        table = _source(flat)
        names = _projected(flat.split(' FROM ', 1)[0], table)
        picked = _by_ids(table, json.loads(params[0]), names)
        return picked.drop_columns(['__KUMO_BATCH__'])
    if 'TABLESAMPLE' in flat:
        table = _source(flat)
        names = _projected(flat.split(' FROM ', 1)[0], table)
        return pa.table(_columns(table, names))
    raise Refused(sql)


SNOW_TABLES = {
    name: table.rename_columns([c.upper() for c in table.column_names])
    for name, table in TABLES.items()
}


def snowflake_answer(sql: str, params: Any) -> pa.Table:
    from snowflake.connector.errors import ProgrammingError

    flat = ' '.join(sql.split())
    if flat.startswith('DESCRIBE TABLE'):
        found = re.fullmatch(r'DESCRIBE TABLE "(\w+)"\."(\w+)"\."(\w+)"', flat)
        assert found is not None, flat
        database, schema, table = found.groups()
        if (database, schema) != ('DB', 'SHOP') or table.lower() not in TABLES:
            raise ProgrammingError(
                msg=f"SQL compilation error:\nTable '{table}' does not exist "
                f'or not authorized.',
                errno=2003,
                sqlstate='42S02',
                sfqid='01b2c3d4-0000-1111-0000-000000000001',
            )
        data = SNOW_TABLES[table.lower()]
        key = KEYS[table.lower()].upper()
        n = data.num_columns
        return pa.table(
            {
                'name': data.column_names,
                'type': [TYPES[f.type][1] for f in data.schema],
                'kind': ['COLUMN'] * n,
                'null?': ['N' if c == key else 'Y' for c in data.column_names],
                'default': pa.array([None] * n, pa.string()),
                'primary key': [
                    'Y' if c == key else 'N' for c in data.column_names
                ],
                'unique key': ['N'] * n,
            }
        )
    if flat.startswith('SHOW IMPORTED KEYS'):
        return _empty(14)
    if flat.startswith('SHOW TABLES IN SCHEMA'):
        return pa.table(
            {
                'created_on': ['2024-01-01', '2024-01-01'],
                'name': ['USERS', 'ORDERS'],
                'is_hybrid': ['N', 'N'],
            }
        )
    if flat.startswith('SHOW TABLES LIKE'):
        table = params[0].upper()
        return pa.table(
            {
                'created_on': ['2024-01-01'],
                'name': [table],
                'kind': ['TABLE'],
                'database_name': ['DB'],
                'schema_name': ['SHOP'],
                'comment': [''],
                'cluster_by': [''],
                'rows': [SNOW_TABLES[table.lower()].num_rows],
            }
        )
    if '__kumo_order_' in flat:
        return _sample(flat, SNOW_TABLES)
    if 'as table_index' in flat:
        return _min_max(flat, SNOW_TABLES)
    if 'FLATTEN' in flat:
        joined, selected = (
            flat.split(' JOIN ', 1)[1],
            flat.split(' FROM TMP')[0],
        )
        table = _source('FROM ' + joined)
        names = _projected(
            selected.split('__KUMO_BATCH__,', 1)[1], table, SNOW_TABLES
        )
        return _by_ids(table, json.loads(params[0]), names, SNOW_TABLES)
    if ' IN (' in flat:
        table = _source(flat)
        names = _projected(flat.split(' FROM ', 1)[0], table, SNOW_TABLES)
        picked = _by_ids(table, list(params), names, SNOW_TABLES)
        return picked.drop_columns(['__KUMO_BATCH__'])
    if re.fullmatch(r'SELECT [^()]+ FROM "DB"\."SHOP"\."\w+"', flat):
        table = _source(flat)
        names = _projected(flat.split(' FROM ', 1)[0], table, SNOW_TABLES)
        return pa.table(_columns(table, names, SNOW_TABLES))
    raise Refused(sql)


def _empty(width: int) -> pa.Table:
    return pa.table({f'c{i}': pa.array([], pa.string()) for i in range(width)})


# Connections.


class StubConnection:
    r"""What both stub connections hold: the answers, and the calls."""

    calls: list[tuple[str, str, str]]
    answer: Callable[[str, Any], pa.Table]

    def type_code(self, data_type: pa.DataType) -> Any:
        raise NotImplementedError


def _databricks_connection() -> Any:
    from databricks.sql.client import Connection

    class StubDatabricksConnection(StubConnection, Connection):
        def __init__(self) -> None:
            self.calls = []
            self.answer = databricks_answer

        def type_code(self, data_type: pa.DataType) -> Any:
            return TYPES[data_type][0]

        def cursor(self, *args: Any, **kwargs: Any) -> StubDatabricksCursor:
            return StubDatabricksCursor(self)

        def close(self) -> None:
            pass

        def __del__(self) -> None:
            pass

    return StubDatabricksConnection()


def _snowflake_connection() -> Any:
    from snowflake.connector import SnowflakeConnection

    class StubSnowflakeConnection(StubConnection, SnowflakeConnection):
        def __init__(self) -> None:
            self.calls = []
            self.answer = snowflake_answer
            self._paramstyle = 'pyformat'

        def type_code(self, data_type: pa.DataType) -> Any:
            # The connector's codes: FIXED, REAL, TEXT, TIMESTAMP_NTZ.
            return {
                pa.int64(): 0,
                pa.float64(): 1,
                pa.string(): 2,
                pa.timestamp('us'): 8,
            }.get(data_type, 2)

        def cursor(self, *args: Any, **kwargs: Any) -> StubSnowflakeCursor:
            return StubSnowflakeCursor(self)

        def close(self) -> None:
            pass

        def __del__(self) -> None:
            pass

    return StubSnowflakeConnection()


# What each backend is asked.


def _databricks_exercise(connection: Any) -> list[Any]:
    from sdm_relational_engine.rfm import Graph
    from sdm_relational_engine.rfm.backend.databricks.sampler import (
        DatabricksSampler,
    )

    graph = Graph.from_databricks(
        connection=connection,
        catalog='main',
        schema='SHOP',
        tables=[
            dict(name='users', primary_key='user_id'),
            dict(
                name='orders', primary_key='order_id', time_column='ordered_at'
            ),
        ],
        verbose=False,
    )
    sampler = DatabricksSampler(graph, verbose=False)
    columns = {'user_id', 'country'}
    return [
        [(e.src_table, e.fkey, e.dst_table) for e in graph.edges],
        {name: graph[name].source_name for name in ('users', 'orders')},
        sampler._get_min_max_time_dict(['orders']),
        sampler._by_pkey('users', pd.Series([3, 1, 9]), columns),
        sampler._sample_entity_table('users', columns, 10, entity_ids=[2, 4]),
        sampler._sample_entity_table('users', columns, 2, random_seed=7),
    ]


def _snowflake_exercise(connection: Any) -> list[Any]:
    from sdm_connectors.reader import read_table
    from sdm_relational_engine.rfm import Graph
    from sdm_relational_engine.rfm.backend.snow.sampler import SnowSampler

    graph = Graph.from_snowflake(
        connection=connection,
        database='db',
        schema='shop',
        tables=[
            dict(name='users', primary_key='USER_ID'),
            dict(
                name='orders', primary_key='ORDER_ID', time_column='ORDERED_AT'
            ),
        ],
        edges=[('orders', 'USER_ID', 'users')],
        verbose=False,
    )
    sampler = SnowSampler(graph, verbose=False)
    columns = {'USER_ID', 'COUNTRY'}
    return [
        {name: graph[name].source_name for name in ('users', 'orders')},
        sampler._get_min_max_time_dict(['orders']),
        sampler._by_pkey('users', pd.Series([3, 1, 9]), columns),
        # No row holds the id: Arrow forced to a table, with no rows.
        sampler._by_pkey('users', pd.Series([9]), columns),
        sampler._sample_entity_table('users', columns, 10, entity_ids=[2, 4]),
        # A metadata statement: its Arrow fetch raises, and rows are read.
        read_table(connection, query='SHOW TABLES IN SCHEMA "DB"."SHOP"'),
    ]


def _plain(value: Any) -> Any:
    r"""A value as nested lists and text, for comparing runs."""
    if isinstance(value, pd.DataFrame):
        return [
            list(value.columns),
            [str(dtype) for dtype in value.dtypes],
            value.astype(object).where(value.notna(), None).values.tolist(),
        ]
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if hasattr(value, 'tolist'):
        return value.tolist()
    return repr(value)


def _shapes(calls: list[tuple[str, str, str]]) -> list[tuple[str, ...]]:
    r"""The calls a driver saw, with row fetches as one.

    The recorder reads a result's rows with one ``fetchall`` however the run
    reads them, and serves ``fetchone`` and ``fetchmany`` from those.
    """
    out: list[tuple[str, ...]] = []
    for call in calls:
        if call[0] in ('fetchall', 'fetchone', 'fetchmany'):
            call = ('rows',)
            if out and out[-1] == call:
                continue
        out.append(call)
    return out


def _schema_problems(step: dict[str, Any]) -> list[str]:
    r"""What the transcript schema says is wrong with ``step``.

    In full where ``jsonschema`` is installed. Where it is not, as in some CI
    jobs, the step's keys are still checked against the ones the schema
    declares for its ``op``, so a key the recorder writes and the schema
    lacks is caught everywhere.
    """
    schema = json.loads(
        (
            Path(__file__).parent / 'schema' / 'transcript.schema.json'
        ).read_text()
    )
    declared = {
        variant['properties']['op']['const']: set(variant['properties'])
        for variant in schema['$defs']['step']['oneOf']
    }
    problems = [
        f'{key!r} is not declared for {step["op"]!r} steps'
        for key in sorted(set(step) - declared.get(step['op'], set()))
    ]
    try:
        import jsonschema
    except ImportError:
        return problems
    validator = jsonschema.Draft202012Validator(
        {'$ref': '#/$defs/step', '$defs': schema['$defs']}
    )
    return problems + [error.message for error in validator.iter_errors(step)]


def _three_ways(
    driver: str,
    connect: Callable[[], Any],
    exercise: Callable[[Any], list[Any]],
    directory: Path,
) -> list[dict[str, Any]]:
    r"""Run ``exercise`` directly, recording and replaying; the stored steps."""
    from _parity.drivers import facts_of

    direct = connect()
    expected = _plain(exercise(direct))

    real = connect()
    facts = facts_of(driver, real)
    tape = RecordingTape()
    recorded = _plain(exercise(facade(driver, tape, real, facts)))
    assert _shapes(real.calls) == _shapes(direct.calls), (
        'the facade changed what the driver saw'
    )
    assert recorded == expected

    save(directory, {'steps': tape.steps}, tape.results, None)
    steps = load(directory / 'transcript.json')['steps']
    for index, step in enumerate(steps):
        problems = _schema_problems(step)
        assert not problems, f'step {index}: {problems}'

    replaying = ReplayingTape(steps, directory)
    replayed = _plain(exercise(facade(driver, replaying, None, facts)))
    assert replaying.divergence is None, replaying.divergence
    assert replaying.leftover() is None
    assert replayed == expected
    return steps


def test_databricks_calls_record_and_replay(tmp_path: Path) -> None:
    pytest.importorskip('databricks.sql')
    pytest.importorskip('sdm_relational_engine')
    steps = _three_ways(
        'databricks', _databricks_connection, _databricks_exercise, tmp_path
    )
    executed = [step for step in steps if step['op'] == 'execute']
    # `execute(sql, parameters=[...])`, stored as the parameters alone:
    assert ['shop', 'users'] in [step['params'] for step in executed]
    assert {'method': 'fetchall_arrow', 'args': [], 'kwargs': {}} in [
        step.get('fetch_call') for step in executed
    ]
    assert executed[0]['description'][1] == {
        'name': 'full_data_type',
        'type_code': 'string',
    }


def test_snowflake_calls_record_and_replay(tmp_path: Path) -> None:
    pytest.importorskip('snowflake.connector')
    pytest.importorskip('sdm_relational_engine')
    steps = _three_ways(
        'snowflake', _snowflake_connection, _snowflake_exercise, tmp_path
    )
    executed = [step for step in steps if step['op'] == 'execute']
    missing = executed[0]['error']
    assert missing['type'] == 'snowflake.connector.errors.ProgrammingError'
    assert missing['attributes']['errno'] == 2003
    calls = [step.get('fetch_call') for step in executed]
    # The connector always asks for a table, an empty one included:
    arrow = [call for call in calls if call is not None]
    assert arrow
    assert all(
        call
        == {
            'method': 'fetch_arrow_all',
            'args': [],
            'kwargs': {'force_return_table': True},
        }
        for call in arrow
    )
    show = executed[-1]
    assert show['fetch_errors'][0]['error']['type'].endswith(
        'NotSupportedError'
    )
    assert show['fetch'] == 'rows'
    assert {step['op'] for step in steps} >= {'execute', 'paramstyle'}


def test_a_rebuilt_error_keeps_its_class_text_and_attributes() -> None:
    errors = pytest.importorskip('snowflake.connector.errors')
    from _parity.tape import describe_error, rebuild_error

    error = errors.ProgrammingError(
        msg='Object does not exist, or operation cannot be performed.',
        errno=2043,
        sqlstate='02000',
    )
    rebuilt = rebuild_error(json.loads(json.dumps(describe_error(error))))
    assert type(rebuilt) is type(error)
    assert str(rebuilt) == str(error)
    assert (rebuilt.errno, rebuilt.sqlstate) == (2043, '02000')


def test_a_fetch_differing_from_the_recording_diverges(tmp_path: Path) -> None:
    pytest.importorskip('snowflake.connector')
    from _parity.tape import Divergence

    real = _snowflake_connection()
    tape = RecordingTape()
    cursor = facade(
        'snowflake', tape, real, {'paramstyle': 'pyformat'}
    ).cursor()
    cursor.execute('SELECT "USER_ID" FROM "DB"."SHOP"."USERS"')
    cursor.fetch_arrow_all(force_return_table=True)
    save(tmp_path, {'steps': tape.steps}, tape.results, None)

    replaying = ReplayingTape(
        load(tmp_path / 'transcript.json')['steps'], tmp_path
    )
    cursor = facade('snowflake', replaying, None, {}).cursor()
    cursor.execute('SELECT "USER_ID" FROM "DB"."SHOP"."USERS"')
    with pytest.raises(Divergence, match='step 0: the fetches differ'):
        cursor.fetch_arrow_all(force_return_table=False)


@pytest.mark.parametrize('keyword', ['parameters', 'params'])
def test_parameters_by_keyword_are_stored_as_by_position(keyword: str) -> None:
    class Cursor:
        description = None

        def __init__(self) -> None:
            self.calls: list[Any] = []

        def execute(self, *args: Any, **kwargs: Any) -> None:
            self.calls.append((args, kwargs))

    from _parity.tape import ParityCursor

    positional, named = Cursor(), Cursor()
    tape = RecordingTape()
    ParityCursor(tape, positional).execute('SELECT ?', [1])
    ParityCursor(tape, named).execute('SELECT ?', **{keyword: [1]})
    assert tape.steps[0]['params'] == tape.steps[1]['params'] == [1]
    assert positional.calls == [(('SELECT ?', [1]), {})]
    assert named.calls == [(('SELECT ?',), {keyword: [1]})]
