# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Sampling branches public calls reach offline and the parity corpus
cannot, held to today's Python through the core.

``scripts/parity/sampling_coverage_gaps.json`` lists the sampling branches
no transcript, fixture case or PQL case takes. Some of them a public call
takes offline, by a route the corpus vocabulary has no words for: a frame
changed in place after ``Schema.from_data`` read it, a list column a caller
puts back into a schema ``Schema.from_source`` left it out of, entity times
at the last instant pandas holds, and ``random_seed=None``, whose entity
draws come from the system's entropy. Each such entry says it is "reachable
only" by the tests here, which make the call through today's Python and
hold the core to it:

- through ``scripts/parity/request_host.py``, the core's own host, the same
  call must give the same payloads, or the same refusal, and the same
  warnings;
- or, where the host cannot make the call, today's
  ``KumoRelational._get_task_table`` and ``_get_context`` are watched as
  ``test_rfm_runs.py`` watches them, and the core's run of each, driven from
  the call's inputs, must ask exactly the statements today's ran and give
  what it gave, or raise what it raised.

Without the core or the engine installed, as in CI's test job, it skips.
"""

from __future__ import annotations

import contextlib
import importlib.util
import itertools
import json
import os
import sys
import tempfile
import warnings
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip('sdm_core')
pytest.importorskip('sdm_relational_engine')

HERE = Path(__file__).resolve().parent
for _path in (HERE, HERE.parent / 'sampling' / 'driver'):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from _driver import Mismatch, encode_params, native, outcome_of  # noqa: E402
from test_rfm_runs import (  # noqa: E402
    REPO,
    Boundary,
    _compare,
    _model_describer,
    context_got,
    context_want,
    task_table_got,
    task_table_want,
)

SCRIPTS = REPO / 'scripts' / 'parity'
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

COUNT = 'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id'
GATHER = {'indices': [1, 2], 'num_context_samples': 5, 'random_seed': 7}
#: ``pd.Timestamp.max`` as text, which SQLite holds a time as.
TIMESTAMP_MAX = '2262-04-11 23:47:16.854775807'
_SCHEMAS = itertools.count()


# --- Data and sources -------------------------------------------------------------


def _frames(name: str = 'shop') -> dict[str, Any]:
    r"""The parity corpus's generated tables."""
    from _parity.data import generate

    return generate(name, 0)


def _sqlite(tables: dict[str, Any], *statements: str) -> Any:
    import adbc_driver_sqlite.dbapi as dbapi
    from sdm_connectors import SQLiteSource

    path = Path(tempfile.mkdtemp(prefix='sdm-gaps-sqlite-')) / 'data.sqlite'
    connection = dbapi.connect(str(path))
    with connection.cursor() as cursor:
        for name, frame in tables.items():
            cursor.adbc_ingest(name, frame, mode='create')
        for statement in statements:
            cursor.execute(statement)
    connection.commit()
    return SQLiteSource(connection)


def _duckdb(tables: dict[str, Any], *statements: str) -> Any:
    from _parity.runner import _duckdb as connect
    from sdm_connectors import DuckDBSource

    connection = connect(tables, {'column_types': {}, 'session': {}})
    for statement in statements:
        connection.execute(statement)
    return DuckDBSource(connection)


def _postgres_dsn() -> str:
    dsn = os.environ.get('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        if os.environ.get('SDM_REQUIRE_POSTGRES') == '1':
            pytest.fail('SDM_REQUIRE_POSTGRES is set and no DSN is given')
        pytest.skip('needs SDM_POSTGRES_TEST_DSN')
    return dsn


@contextlib.contextmanager
def _postgres(tables: dict[str, Any], *statements: str) -> Iterator[Any]:
    r"""A PostgreSQL source over ``tables``, in a schema of its own that is
    dropped afterwards; ``{schema}`` in a statement is that schema."""
    import psycopg
    from _parity.runner import load_postgres
    from sdm_connectors import PostgresSource

    schema = f'sdm_gaps_{os.getpid()}_{next(_SCHEMAS)}'
    connection = psycopg.connect(_postgres_dsn(), autocommit=True)
    try:
        connection.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
        load_postgres(connection, schema, tables, {})
        for statement in statements:
            connection.execute(statement.format(schema=schema))
        yield PostgresSource(connection, schema=schema)
    finally:
        with contextlib.suppress(Exception):
            connection.execute(f'DROP SCHEMA IF EXISTS {schema} CASCADE')
        connection.close()


def _ask(schema: Any, **gather: Any) -> Any:
    r"""The COUNT question's ``materialize`` on a fresh model."""
    from sdm_client import KumoRelational, PredictiveQuery

    return PredictiveQuery(COUNT, schema).materialize(
        model=KumoRelational(), **gather
    )


# --- Through the host -------------------------------------------------------------


def _host() -> Any:
    native('materialize_run')
    if 'request_host' in sys.modules:
        return sys.modules['request_host']
    spec = importlib.util.spec_from_file_location(
        'request_host', SCRIPTS / 'request_host.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['request_host'] = module
    spec.loader.exec_module(module)
    return module


def _outcome(make: Callable[[], Any]) -> tuple[Any, list[Any]]:
    r"""What ``make`` gave, its payloads as JSON, or what it raised as
    class, code, message and details; with the warnings given on the way."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            got: Any = ('value', json.loads(json.dumps(make())))
        except Exception as error:  # noqa: BLE001 - compared
            cls = type(error)
            got = (
                'error',
                (
                    f'{cls.__module__}.{cls.__qualname__}',
                    getattr(error, 'code', None),
                    getattr(error, 'message', str(error)),
                    dict(getattr(error, 'details', None) or {}),
                ),
            )
    said = [
        (f'{w.category.__module__}.{w.category.__qualname__}', str(w.message))
        for w in caught
    ]
    return got, said


def _text_of_tuples(tables: dict[str, Any]) -> None:
    users = tables['users']
    users['country'] = ['x y'] + [('a', 'b')] * (len(users) - 1)


def test_a_text_column_of_tuples() -> None:
    r"""``users.country`` changed in place after ``Schema.from_data`` to one
    text and then tuples, and typed ``text``: the engine types the column by
    its first value, so it stays a text column, and the sampled rows, which
    start at a tuple, are kept as they are rather than split into words
    (``base/sampler.py`` ``_normalize_text``). The core, through the host,
    gives today's payloads and warnings."""
    from sdm_client import PredictiveQuery, Schema

    gather = {'indices': [30, 31], 'num_context_samples': 3, 'random_seed': 1}

    def today() -> Any:
        tables = _frames()
        schema = Schema.from_data(tables)
        _text_of_tuples(tables)
        schema['users']['country'].stype = 'text'
        return list(
            PredictiveQuery(COUNT, schema).materialize(**gather).payloads
        )

    host = _host()

    def core() -> Any:
        tables = _frames()
        schema = host.schema_from_data(
            tables,
            edits=[
                {
                    'table': 'users',
                    'column': 'country',
                    'set': 'stype',
                    'value': 'text',
                }
            ],
        )
        _text_of_tuples(tables)
        query = host.predictive_query(COUNT, schema)
        return host.materialize(query, **gather).state['payloads']

    want = _outcome(today)
    assert want[0][0] == 'value', want
    users = [
        p['schema']['related_tables']['users']['columns']['country']
        for p in want[0][1]
    ]
    assert users and all(c['stype'] == 'text' for c in users), users
    assert _outcome(core) == want


@pytest.mark.parametrize(
    ('tokens', 'null_key'),
    [
        (('A', 'B'), None),
        (('New York', ' B '), None),
        (('a.b', 'c!'), None),
        (('A', 'B'), 30),
        (('A', 'B'), 31),
    ],
    ids=['case', 'whitespace', 'punctuation', 'null-30', 'null-31'],
)
def test_text_tuples_preserve_cells(
    monkeypatch: pytest.MonkeyPatch,
    tokens: tuple[str, str],
    null_key: int | None,
) -> None:
    def changed(tables: dict[str, Any]) -> None:
        users = tables['users']
        users['country'] = ['x y'] + [
            None if key == null_key else tokens
            for key in users['user_id'].iloc[1:]
        ]

    monkeypatch.setattr(sys.modules[__name__], '_text_of_tuples', changed)
    test_a_text_column_of_tuples()


@pytest.mark.parametrize(
    ('fill', 'refused'),
    [
        ('ARRAY[1.25,2.50]', False),
        ('ARRAY[0.001,0.000]', False),
        (
            'CASE WHEN user_id % 2 = 0 THEN ARRAY[0.00001,12.3400] '
            'ELSE ARRAY[1234.0,-0.025] END',
            False,
        ),
        ('ARRAY[0.000000000000000000000000000000000000001,2]', False),
        ('ARRAY[1,1.0,1.00,0.000,-0.0100]', False),
        (
            'CASE WHEN user_id % 2 = 0 THEN ARRAY[1,1.00] END',
            False,
        ),
        (
            'ARRAY[1.25,NULL,2.50]',
            'stringlist columns must not contain null items',
        ),
        ('CASE WHEN user_id % 2 = 0 THEN ARRAY[1.25,2.50] END', False),
        ('ARRAY[12345678901234567890.123456789012345678,2]', True),
        ('ARRAY[12345678901234567890.1234567890123456789,2]', True),
        ('ARRAY[-12345678901234567890.1234567890123456789,2]', True),
        ('ARRAY[' + '9' * 76 + ',2]', False),
    ],
    ids=[
        'ordinary',
        'subunit',
        'mixed-scale',
        'fraction39',
        'trailing-zeros',
        'mixed-scale-null-row',
        'null-item',
        'null-row',
        '38',
        '39',
        'negative',
        '76',
    ],
)
def test_postgres_numeric_array_boundary_and_payload(
    fill: str, refused: bool | str
) -> None:
    _compare_postgres_numeric_array(fill, refused, compare_boundary=True)


@pytest.mark.parametrize(
    ('fill', 'refused'),
    [
        ('ARRAY[' + '9' * 77 + ',2]', False),
        (
            "ARRAY['NaN'::numeric,2]",
            'stringlist columns must not contain null items',
        ),
        (
            "ARRAY['Infinity'::numeric,2]",
            'stringlist columns must not contain null items',
        ),
    ],
    ids=['77', 'nan', 'infinity'],
)
def test_postgres_numeric_array_outside_arrow_decimal(
    fill: str, refused: bool | str
) -> None:
    _compare_postgres_numeric_array(fill, refused, compare_boundary=False)


@pytest.mark.parametrize('first_wide', [False, True], ids=['later', 'first'])
@pytest.mark.parametrize(
    ('ordinary', 'exceptional', 'compare_boundary', 'refused'),
    [
        ('ARRAY[1.0,2]', 'ARRAY[1.0,2]', True, False),
        ('ARRAY[1.0,2]', 'ARRAY[123.0,2]', True, False),
        ('ARRAY[1.0,2]', 'ARRAY[0.00001,2]', True, False),
        ('ARRAY[1.0,2]', 'ARRAY[1E-39,2]', True, False),
        ('ARRAY[1E-39,2]', 'ARRAY[1E-40,2]', True, False),
        ('ARRAY[]::numeric[]', 'ARRAY[1E-39,2]', True, False),
        ('NULL::numeric[]', 'ARRAY[1E-39,2]', True, False),
        (
            'ARRAY[NULL]::numeric[]',
            'ARRAY[1E-39,2]',
            True,
            'stringlist columns must not contain null items',
        ),
        ('ARRAY[1.0,2]', 'ARRAY[' + '9' * 76 + ',2]', False, False),
        ('ARRAY[1.0,2]', 'ARRAY[' + '9' * 77 + ',2]', False, False),
        (
            'ARRAY[1.0,2]',
            "ARRAY['NaN'::numeric,2]",
            False,
            'stringlist columns must not contain null items',
        ),
    ],
    ids=[
        'uniform',
        'precision128',
        'scale128',
        'width256',
        'scale256',
        'empty-lists',
        'null-rows',
        'null-items',
        'union-overflow',
        'precision77',
        'nonfinite',
    ],
)
def test_postgres_numeric_batches_preserve_cells(
    monkeypatch: pytest.MonkeyPatch,
    first_wide: bool,
    ordinary: str,
    exceptional: str,
    compare_boundary: bool,
    refused: bool | str,
) -> None:
    from sdm_relational_engine.rfm.backend.postgres.sampler import (
        PostgresSampler,
    )

    user_reads: list[list[int]] = []
    original = PostgresSampler._by_pkey

    def observed(
        sampler: Any, table_name: str, index: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if table_name == 'users':
            user_reads.append([int(value) for value in index])
        return original(sampler, table_name, index, *args, **kwargs)

    monkeypatch.setattr(PostgresSampler, '_by_pkey', observed)
    key = 20 if first_wide else 34
    _compare_postgres_numeric_array(
        f'CASE WHEN user_id = {key} THEN {exceptional} ELSE {ordinary} END',
        refused,
        compare_boundary=compare_boundary,
        gather={**GATHER, 'num_neighbors': [2, 2, 2, 2]},
    )
    assert len(user_reads) == 2, user_reads
    assert (key in user_reads[0]) == first_wide, user_reads
    assert (key in user_reads[1]) != first_wide, user_reads


def _compare_postgres_numeric_array(
    fill: str,
    refused: bool | str,
    *,
    compare_boundary: bool,
    gather: dict[str, Any] | None = None,
) -> None:
    from sdm_client import Column, PredictiveQuery, Schema

    host = _host()
    gather = GATHER if gather is None else gather
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with _postgres(
            _frames(),
            'ALTER TABLE {schema}.users ADD COLUMN tags numeric[]',
            f'UPDATE {{schema}}.users SET tags = {fill}',
        ) as source:
            python_schema = Schema.from_source(source)
            python_schema['users'].columns['tags'] = Column(
                'tags', 'string', 'multicategorical'
            )
            core_schema = host.schema_from_source(source)
            users = next(
                table
                for table in core_schema.state['schema']['tables']
                if table['name'] == 'users'
            )
            users['columns'].append(
                {
                    'name': 'tags',
                    'dtype': 'string',
                    'stype': 'multicategorical',
                    'untyped': False,
                    'stype_declared': True,
                    'before_key': None,
                }
            )
            boundaries = (
                _boundaries('_get_task_table', '_get_context')
                if compare_boundary
                else []
            )
            with _installed(boundaries):
                want = _outcome(
                    lambda: list(
                        PredictiveQuery(COUNT, python_schema)
                        .materialize(**gather)
                        .payloads
                    )
                )
            got = _outcome(
                lambda: host.materialize(
                    host.predictive_query(COUNT, core_schema), **gather
                ).state['payloads']
            )
    if refused:
        assert want[0][0] == 'error', want
        assert want[0][1][1] == 'INVALID_REQUEST', want
        message = (
            refused
            if isinstance(refused, str)
            else 'too large for float64 to represent'
        )
        assert message in want[0][1][2], want
    else:
        assert want[0][0] == 'value', want
    assert got == want
    if compare_boundary:
        _the_core_agrees(boundaries)


@pytest.mark.parametrize(
    ('table', 'key'),
    [('products', 'product_id'), ('users', 'user_id')],
    ids=['products', 'users'],
)
def test_a_key_set_to_nan_after_the_schema(table: str, key: str) -> None:
    r"""A key set to NaN in place after ``Schema.from_data``, in a table
    without a time column: the graph store finds no valid row in it
    (``backend/local/graph_store.py``) and today's refuses. The core,
    through the host, refuses alike, with the same warnings."""
    import numpy as np
    from sdm_client import PredictiveQuery, Schema

    def today() -> Any:
        tables = _frames()
        schema = Schema.from_data(tables)
        tables[table][key] = np.nan
        return list(
            PredictiveQuery(COUNT, schema).materialize(**GATHER).payloads
        )

    host = _host()

    def core() -> Any:
        tables = _frames()
        schema = host.schema_from_data(tables)
        tables[table][key] = np.nan
        query = host.predictive_query(COUNT, schema)
        return host.materialize(query, **GATHER).state['payloads']

    want = _outcome(today)
    kind, (_, code, message, _) = want[0]
    assert kind == 'error' and code == 'INVALID_REQUEST', want
    assert f"Found no valid rows in table '{table}'" in message, message
    assert _outcome(core) == want


# --- Through the boundary ---------------------------------------------------------


def _boundaries(*methods: str) -> list[Boundary]:
    from sdm_relational_engine.rfm.rfm import KumoRelational

    return [
        Boundary(KumoRelational, method, lambda model: model._sampler)
        for method in methods
    ]


@contextlib.contextmanager
def _installed(boundaries: list[Boundary]) -> Iterator[None]:
    with contextlib.ExitStack() as stack:
        for boundary in boundaries:
            stack.enter_context(
                boundary.installed(_model_describer(boundary.method))
            )
        yield


def _the_core_agrees(boundaries: list[Boundary]) -> None:
    r"""The core's run of each watched call, driven from its inputs, asks
    exactly today's statements and gives today's outputs or error."""
    compared = {
        '_get_task_table': ('task_table', task_table_want, task_table_got),
        '_get_context': ('context', context_want, context_got),
    }
    for boundary in boundaries:
        method, want, got = compared[boundary.method]
        count, problems = _compare(method, boundary, want, got)
        assert count >= 1, f'today made no {boundary.method} call'
        assert not problems, f'{method}: ' + '\n'.join(problems[:10])


def _quiet(make: Callable[[], Any]) -> tuple[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            return 'value', make()
        except Exception as error:  # noqa: BLE001 - kept, checked
            return 'error', error


#: List columns ``Schema.from_source`` leaves out, by backend: the column's
#: SQL type and what fills it.
LIST_COLUMNS = {
    'duckdb-VARCHAR[]': ('duckdb', 'VARCHAR[]', "['a', 'b']"),
    'duckdb-INTEGER[]': ('duckdb', 'INTEGER[]', '[1, 2]'),
    'duckdb-DOUBLE[]': ('duckdb', 'DOUBLE[]', '[1.5, 2]'),
    'postgres-text[]': ('postgres', 'text[]', "ARRAY['a', 'b']"),
    'postgres-int[]': ('postgres', 'int[]', 'ARRAY[1, 2]'),
    'postgres-float8[]': ('postgres', 'float8[]', 'ARRAY[1.5, 2]'),
    'postgres-text[]-some-null': (
        'postgres',
        'text[]',
        "CASE WHEN user_id % 2 = 0 THEN ARRAY['a', 'b'] END",
    ),
    'postgres-text[]-empty': ('postgres', 'text[]', 'ARRAY[]::text[]'),
}


def _a_list_column_put_back(backend: str, sql_type: str, fill: str) -> Any:
    from sdm_client import Column, Schema

    def ask(source: Any) -> Any:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            schema = Schema.from_source(source)
        assert 'tags' not in schema['users'].columns
        assert any('tags' in str(w.message) for w in caught), [
            str(w.message) for w in caught
        ]
        schema['users'].columns['tags'] = Column(
            'tags', 'string', 'multicategorical'
        )
        return _ask(schema, **GATHER)

    if backend == 'duckdb':
        return ask(
            _duckdb(
                _frames(),
                f'ALTER TABLE users ADD COLUMN tags {sql_type}',
                f'UPDATE users SET tags = {fill}',
            )
        )
    with _postgres(
        _frames(),
        f'ALTER TABLE {{schema}}.users ADD COLUMN tags {sql_type}',
        f'UPDATE {{schema}}.users SET tags = {fill}',
    ) as source:
        return ask(source)


@pytest.mark.parametrize('case', list(LIST_COLUMNS))
def test_a_list_column_put_back(case: str) -> None:
    r"""A list column ``Schema.from_source`` leaves out with a warning, put
    back as ``multicategorical``: the SQL sampler reads it, and leaves it out
    of the order it breaks ties by (``base/sql_sampler.py``
    ``_tie_break_terms``). The core's task table and context runs ask
    today's statements and give today's outputs."""
    if LIST_COLUMNS[case][0] == 'postgres':
        _postgres_dsn()
    boundaries = _boundaries('_get_task_table', '_get_context')
    with _installed(boundaries):
        kind, got = _quiet(lambda: _a_list_column_put_back(*LIST_COLUMNS[case]))
    assert kind == 'value', f'today refused: {got!r}'
    _the_core_agrees(boundaries)


def _entity_times_at_max(backend: str) -> Any:
    import pandas as pd
    from sdm_client import Schema

    tables = _frames('shop-referrals')
    if backend == 'memory':
        users = tables['users']
        users['joined_at'] = pd.Series([pd.Timestamp.max] * len(users))
        schema = Schema.from_data(tables)
    else:
        schema = Schema.from_source(
            _sqlite(tables, f"UPDATE users SET joined_at = '{TIMESTAMP_MAX}'")
        )
    schema['users'].time_column = 'joined_at'
    return _ask(schema, **GATHER)


@pytest.mark.parametrize('backend', ['memory', 'sqlite'])
def test_entity_times_at_the_last_instant(backend: str) -> None:
    r"""Every entity time ``pd.Timestamp.max``, the time column set on the
    entity table: the entity table's earliest time is the sentinel for no
    time (``base/sampler.py`` ``_no_entity_at_anchor``), no context example
    is found, and today's task table refuses. The core's task table run
    raises what today's raised, with the warnings before it."""
    (boundary,) = _boundaries('_get_task_table')
    with _installed([boundary]):
        kind, got = _quiet(lambda: _entity_times_at_max(backend))
    assert kind == 'error', f'today did not refuse: {got!r}'
    assert 'Failed to collect any context examples' in str(got), got
    assert any(call['error'] is not None for call in boundary.calls)
    _the_core_agrees([boundary])


def _sampler_class(backend: str) -> type:
    if backend == 'sqlite':
        from sdm_relational_engine.rfm.backend.sqlite.sampler import (
            SQLiteSampler,
        )

        return SQLiteSampler
    from sdm_relational_engine.rfm.backend.postgres.sampler import (
        PostgresSampler,
    )

    return PostgresSampler


def _unseeded(backend: str) -> Any:
    from sdm_client import Schema

    gather = {**GATHER, 'num_context_samples': 30, 'random_seed': None}
    if backend == 'sqlite':
        return _ask(Schema.from_source(_sqlite(_frames())), **gather)
    with _postgres(_frames()) as source:
        return _ask(Schema.from_source(source), **gather)


def _drive_through(
    core: Any, run: Any, steps: list[dict[str, Any]], upto: int
) -> list[str]:
    r"""Drive ``run``, answering its queries with ``steps``' results, until it
    has asked for the first ``upto`` of them, each the next step, its SQL
    byte for byte and its parameters encoded alike; the statements asked."""
    asked: list[str] = []
    while len(asked) < upto:
        step = run.next()
        kind = getattr(step, 'kind', None)
        if kind == 'done':
            raise Mismatch(
                f'the run finished after {len(asked)} statements, before '
                f'asking for statement {upto - 1}'
            )
        assert kind == 'run_queries', f'a step of kind {kind!r}'
        for query in list(step.queries):
            if len(asked) >= upto:
                break
            position = len(asked)
            want = steps[position]
            assert query.sql == want['sql'], (
                f'statement {position}: {query.sql!r} instead of '
                f'{want["sql"]!r}'
            )
            params = encode_params(query.params)
            assert json.dumps(params) == json.dumps(want['params']), (
                f'statement {position}: parameters {params} instead of '
                f'{want["params"]}'
            )
            run.submit(query.id, outcome_of(core, want, want['result']))
            asked.append(query.sql)
    return asked


@pytest.mark.parametrize('backend', ['sqlite', 'postgres'])
def test_the_first_unseeded_entity_read(backend: str) -> None:
    r"""With ``random_seed=None``, the entity draw reads its rows without a
    seed: SQLite in a random order (``backend/sqlite/sampler.py``), and
    PostgreSQL without setting the seed first
    (``backend/postgres/sampler.py``). Those statements are deterministic.
    The core's task table run, driven from today's call and answered with
    today's results, asks exactly today's statements up to the end of
    today's first entity draw (``_sample_entity_table``). What follows
    depends on draws from the system's entropy, on both sides, and is not
    compared."""
    if backend == 'postgres':
        _postgres_dsn()
    (boundary,) = _boundaries('_get_task_table')
    cls = _sampler_class(backend)
    original = vars(cls)['_sample_entity_table']
    spans: list[tuple[int, int]] = []

    def noted(self: Any, *args: Any, **kwargs: Any) -> Any:
        steps = boundary._steps
        start = len(steps) if steps is not None else -1
        try:
            return original(self, *args, **kwargs)
        finally:
            steps = boundary._steps
            spans.append((start, len(steps) if steps is not None else -1))

    setattr(cls, '_sample_entity_table', noted)  # noqa: B010 - a method
    try:
        with _installed([boundary]):
            kind, got = _quiet(lambda: _unseeded(backend))
    finally:
        setattr(cls, '_sample_entity_table', original)  # noqa: B010
    assert kind == 'value', f'today refused: {got!r}'
    assert boundary.calls, 'today made no _get_task_table call'
    call = boundary.calls[0]
    assert call['error'] is None, call['error']
    assert json.loads(call['described'][0])['arguments']['random_seed'] is None
    assert spans and spans[0][0] >= 0, 'no entity draw inside _get_task_table'
    start, end = spans[0]
    assert end > start, 'the first entity draw ran no statement'
    drawn = [step['sql'] for step in call['steps'][start:end]]
    if backend == 'sqlite':
        assert any('RANDOM()' in sql for sql in drawn), drawn
    assert not any('setseed' in sql for sql in drawn), drawn
    core = native('task_table', 'Outcome')
    inputs_json, tables = call['described']
    asked = _drive_through(
        core, core.task_table(inputs_json, tables), call['steps'], end
    )
    assert asked[start:end] == drawn
