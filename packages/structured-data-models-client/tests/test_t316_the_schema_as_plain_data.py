# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The schema reaches the core as plain data, and the core's results come
back typed (T3.16, the Phase 3 structure design's fixes A and D).

Fix A: the schema a host hands ``edit_schema`` and ``materialize`` is the
schema as it stands, each column carrying what the source said of it
(``Column._source``, out of repr and equality, copied and pickled with the
column); nothing the core answered earlier is merged in, and a column the
source was never asked of is read from the source at materialize, as the
old client reads every table it samples in place. So a schema built by
hand, unpickled, copied, edited or described on the old route materializes
as the old client materializes it, and a described schema, however it was
copied, asks the source nothing more.

Fix D: the host reads the core's results through its own typed structures
(``sdm_client._rust.results``): a request materialized on the core keeps
``Request._core``, a ``CoreRequest``, and one with none is refused by name,
never a ``RouteError``.

Each public call runs on both routes, the old client's (the reference) and
the core's with every core route forced, and the core's answer is held to
the old route's.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import io
import json
import os
import pickle
import uuid
import warnings
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource, PostgresSource

from sdm_client import (
    Column,
    PredictiveQuery,
    Relationship,
    Schema,
    SourceTable,
    StructuredDataError,
    _routing,
)
from sdm_client.errors import MissingExtraError

#: Whether the old route, which gathers with the relational engine, can
#: run: CI's ``test`` job installs no engine, its ``engine`` job does.
ENGINE = importlib.util.find_spec('sdm_relational_engine') is not None

#: Every path this build can run on the core, forced.
CORE = ','.join(sorted(_routing.CORE_ROUTED))
ROUTES = ('old', 'core')
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
ANCHOR = pd.Timestamp('2024-03-01')
#: Two users of the NUMERIC-keyed shop, past a double's exact integers.
NUMERIC_IDS = (90000000000000001, 90000000000000002)

SHOP = (
    'CREATE TABLE users (user_id BIGINT PRIMARY KEY, age BIGINT, '
    'country VARCHAR)',
    "INSERT INTO users SELECT i, 20 + i % 50, ['US', 'DE', 'IN'][1 + i % 3] "
    'FROM range(40) t(i)',
    'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id BIGINT, '
    'price DOUBLE, ts TIMESTAMP)',
    'INSERT INTO orders SELECT i, i % 40, 10 + i * 1.5, TIMESTAMP '
    "'2023-06-01' + to_days(CAST(i AS INTEGER)) FROM range(400) t(i)",
    # The same orders, every price doubled: a table to point the schema at.
    'CREATE TABLE other_orders AS SELECT order_id, user_id, price * 2 AS '
    'price, ts FROM orders',
)


@pytest.fixture(autouse=True)
def _old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


def _on(route: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if route == 'core':
        monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    else:
        monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)


def _shop() -> DuckDBSource:
    connection = duckdb.connect()
    for statement in SHOP:
        connection.execute(statement)
    return DuckDBSource(connection)


def _described(source: Any, **arguments: Any) -> Schema:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema.from_source(
            source, {'users': 'users', 'orders': 'orders'}, **arguments
        )


def _payload(
    schema: Schema, query: str = QUERY, indices: Sequence[Any] = (0, 1, 2)
) -> Any:
    r"""The call's outcome: the request's payloads as JSON, or the refusal's
    class and code."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            request = PredictiveQuery(query, schema).materialize(
                indices=list(indices), anchor_time=ANCHOR
            )
    except StructuredDataError as refused:
        return (type(refused).__name__, refused.code)
    return json.loads(json.dumps(list(request._payloads), default=str))


def _both(
    monkeypatch: pytest.MonkeyPatch, act: Callable[[], Any]
) -> dict[str, Any]:
    r"""``act``'s outcome on each route, the old route's first; where no
    engine is installed the old route refuses as the client's missing extra,
    which is held, and the core's outcome stands for both."""
    outcomes = {}
    for route in ROUTES:
        _on(route, monkeypatch)
        if route == 'old' and not ENGINE:
            with pytest.raises(MissingExtraError):
                act()
            continue
        outcomes[route] = act()
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    outcomes.setdefault('old', outcomes['core'])
    return outcomes


def _described_on_the_old_route(
    source: Any, monkeypatch: pytest.MonkeyPatch
) -> Schema:
    r"""``source`` described on the old route; with no engine installed,
    which the old route needs, on the core with what the source said of
    each column taken off, as the old route's schema holds none."""
    if ENGINE:
        _on('old', monkeypatch)
        described = _described(source)
    else:
        _on('core', monkeypatch)
        described = _described(source)
        for table in described.tables.values():
            for column in table.columns.values():
                object.__setattr__(column, '_source', None)
    _on('core', monkeypatch)
    return described


def _hand_built(source: Any, **extra: Column) -> Schema:
    r"""The shop written out by hand, column by column, as a user would
    write it, with ``extra`` columns on ``users``."""
    return Schema(
        {
            'users': SourceTable(
                'users',
                'users',
                {
                    'user_id': Column('user_id', 'int64', 'ID'),
                    'age': Column('age', 'int64', 'numerical'),
                    **extra,
                },
                primary_key='user_id',
            ),
            'orders': SourceTable(
                'orders',
                'orders',
                {
                    'order_id': Column('order_id', 'int64', 'ID'),
                    'user_id': Column('user_id', 'int64', 'ID'),
                    'price': Column('price', 'float64', 'numerical'),
                    'ts': Column('ts', 'timestamp[us]', 'timestamp'),
                },
                primary_key='order_id',
                time_column='ts',
            ),
        },
        [Relationship('orders', 'user_id', 'users', 'user_id')],
        source=source,
    )


def test_a_hand_built_column_its_source_lacks_is_left_unread_as_the_old_client_leaves_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The old client reads each table in place, as its source holds it, and
    # a column the schema names that the source lacks is never read.
    source = _shop()
    outcomes = _both(
        monkeypatch,
        lambda: _payload(
            _hand_built(source, ghost=Column('ghost', 'float64', 'numerical'))
        ),
    )
    assert isinstance(outcomes['old'], list), outcomes['old']
    assert outcomes['core'] == outcomes['old']


def test_a_described_column_retyped_by_hand_is_sampled_as_its_source_holds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _shop()

    def act() -> Any:
        schema = _described(source)
        schema['users']['age'].dtype = 'string'
        schema['users']['age'].stype = 'categorical'
        return _payload(schema)

    outcomes = _both(monkeypatch, act)
    assert isinstance(outcomes['old'], list), outcomes['old']
    assert outcomes['core'] == outcomes['old']


def test_a_table_pointed_at_another_source_table_samples_that_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _shop()

    def act() -> Any:
        schema = _described(source)
        schema['orders'].source_name = 'other_orders'
        return _payload(schema)

    outcomes = _both(monkeypatch, act)
    assert isinstance(outcomes['old'], list), outcomes['old']
    assert outcomes['core'] == outcomes['old']
    # The doubled prices were read, not the described table's.
    _on('core', monkeypatch)
    assert outcomes['core'] != _payload(_described(source))


def test_a_column_added_by_hand_to_a_described_schema_is_edited_and_sampled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _shop()

    def act() -> Any:
        # A column left out of the described schema, then written back by
        # hand: it holds nothing the source said of it, and the caller
        # edits it before sampling.
        schema = _described(source)
        del schema['users']['country']
        schema['users'].columns['country'] = Column('country', 'string', 'text')
        schema['users']['country'].stype = 'categorical'
        return (
            schema['users']['country'].stype,
            schema['users']['country'].stype_declared,
            _payload(schema),
        )

    outcomes = _both(monkeypatch, act)
    assert isinstance(outcomes['old'][2], list), outcomes['old']
    assert outcomes['core'] == outcomes['old']


def _unpickled(schema: Schema) -> Schema:
    r"""``schema`` pickled and loaded, then bound to its source again in a
    schema of its tables: a source holds an open connection, which no
    client pickles, so a schema is pickled without it, as one written
    before it could hold one."""
    state = {
        key: value
        for key, value in vars(schema).items()
        if key not in ('_source', '_engine_tables')
    }

    class _WithoutTheSource(pickle.Pickler):
        def reducer_override(self, obj: Any) -> Any:
            if obj is schema:
                return object.__new__, (Schema,), state
            return NotImplemented

    written = io.BytesIO()
    _WithoutTheSource(written).dump(schema)
    loaded = pickle.loads(written.getvalue())
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return Schema(
            dict(loaded.tables),
            list(loaded.relationships),
            source=schema.source,
        )


def _copies() -> dict[str, Callable[[Schema], Schema]]:
    r"""A described schema as the caller may come to hold it: as described,
    copied, snapshotted, its tables adopted by a schema of their own, or
    pickled and loaded."""

    def rebuilt(schema: Schema) -> Schema:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return Schema(
                dict(schema.tables),
                list(schema.relationships),
                source=schema.source,
            )

    return {
        'as described': lambda schema: schema,
        'copy.copy': copy.copy,
        'snapshot': lambda schema: schema.snapshot(),
        'rebuilt': rebuilt,
        'pickled': _unpickled,
    }


@pytest.mark.parametrize('made', sorted(_copies()))
def test_a_described_schema_however_copied_asks_the_source_nothing_more(
    made: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sdm_client._rust import providers

    source = _shop()
    asked: list[str] = []
    real = providers.for_source

    def watched(given: Any) -> Any:
        choice, provider = real(given)

        def answering(request: dict[str, Any], keys: Any) -> Any:
            asked.append(request['request']['request'])
            return provider(request, keys)

        return choice, answering

    monkeypatch.setattr(providers, 'for_source', watched)

    def act() -> Any:
        described = _described(source)
        asked.clear()
        once = _payload(described)
        first = list(asked)
        asked.clear()
        copied = _payload(_copies()[made](described))
        if os.environ.get(_routing.ENVIRONMENT):
            # On the core, the copy's materialize asks what the schema's own
            # asked: no catalog question or preview of a table already
            # described.
            assert asked == first
            assert 'preview' not in asked
        return once, copied

    outcomes = _both(monkeypatch, act)
    assert isinstance(outcomes['old'][0], list), outcomes['old']
    assert outcomes['core'] == outcomes['old']
    assert outcomes['core'][0] == outcomes['core'][1]


def test_the_source_s_facts_ride_on_each_column_out_of_repr_and_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sdm_client._rust import results

    source = _shop()
    _on('core', monkeypatch)
    core = _described(source)
    old = _described_on_the_old_route(source, monkeypatch)
    assert repr(core) == repr(old)
    for name, table in core.tables.items():
        for column_name, column in table.columns.items():
            facts = column._source
            assert isinstance(facts, results.SourceFacts), (name, column_name)
            assert dataclasses.is_dataclass(facts)
            with pytest.raises(dataclasses.FrozenInstanceError):
                facts.nullable = not facts.nullable  # type: ignore[misc]
            # Out of equality and repr: the old route's column, which holds
            # none, is equal to it.
            assert old.tables[name].columns[column_name]._source is None
            assert column == old.tables[name].columns[column_name]
            assert 'source' not in repr(column)
    assert core['users']['user_id']._source.unique
    assert not core['users']['user_id']._source.nullable
    assert core['orders']['ts']._source.dtype == 'date'
    for made, copied in _copies().items():
        held = copied(core)
        assert held['users']['age']._source == core['users']['age']._source, (
            made
        )
    # A column copied on its own keeps its facts too.
    assert (
        copy.copy(core['users']['age'])._source == core['users']['age']._source
    )


def test_a_request_materialized_on_the_core_keeps_a_typed_core_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sdm_client._rust import results

    _on('core', monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(QUERY, _described(_shop())).materialize(
            indices=[0, 1], anchor_time=ANCHOR
        )
    assert isinstance(request._core, results.CoreRequest)
    assert request._core.version == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        request._core.version = 3  # type: ignore[misc]
    assert not hasattr(request, '_core_state')


def test_predict_of_a_request_the_core_never_materialized_asks_to_materialize_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # An old route's request, as an old pickle holds it, predicted with the
    # core's predict forced: refused by name before any call is sent, as a
    # StructuredDataError the caller can act on, never a RouteError.
    from sdm_client import KumoRelational, SDMClient

    _on('old' if ENGINE else 'core', monkeypatch)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(QUERY, _described(_shop())).materialize(
            indices=[0, 1], anchor_time=ANCHOR
        )
    if not ENGINE:
        # No old route to build it: the core's request, its state dropped,
        # as an old pickle holds none.
        request._core = None
    assert request._core is None
    _on('core', monkeypatch)
    with SDMClient('http://127.0.0.1:9', max_retries=0) as client:
        with pytest.raises(StructuredDataError) as refused:
            client.predict(request, model=KumoRelational())
    assert not isinstance(refused.value, _routing.RouteError)
    assert refused.value.code == 'INVALID_REQUEST'
    assert refused.value.details['rule'] == 'materialize_again'
    assert 'materialize' in str(refused.value)


# --- Postgres: a NUMERIC key, through copies and edits ---------------------------


@pytest.fixture
def numeric_source() -> Iterator[Any]:
    r"""A Postgres schema of its own whose accounts are keyed by a NUMERIC,
    which the provider hands over as exact text; dropped afterwards."""
    dsn = os.getenv('SDM_POSTGRES_TEST_DSN')
    if not dsn:
        # No server: the test returns, as the skip ban asks; CI's Postgres
        # jobs require one.
        if os.getenv('SDM_REQUIRE_POSTGRES'):
            pytest.fail(
                'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
            )
        yield None
        return
    import psycopg

    namespace = f'sdm_{uuid.uuid4().hex[:8]}'
    with psycopg.connect(dsn, autocommit=True) as connection:
        connection.execute(f'CREATE SCHEMA {namespace}')
        try:
            connection.execute(f'SET search_path TO {namespace}')
            connection.execute(
                'CREATE TABLE users (user_id NUMERIC PRIMARY KEY, age BIGINT)'
            )
            connection.execute(
                'INSERT INTO users SELECT i + 90000000000000001, 20 + i % 50 '
                'FROM generate_series(0, 39) AS i'
            )
            connection.execute(
                'CREATE TABLE orders (order_id BIGINT PRIMARY KEY, user_id '
                'NUMERIC REFERENCES users (user_id), price DOUBLE PRECISION, '
                'ts TIMESTAMP)'
            )
            connection.execute(
                'INSERT INTO orders SELECT i, i % 40 + 90000000000000001, '
                "10 + i * 1.5, TIMESTAMP '2023-06-01' + i * INTERVAL '1 day' "
                'FROM generate_series(0, 399) AS i'
            )
            yield PostgresSource(connection, schema=namespace)
        finally:
            connection.execute(f'DROP SCHEMA {namespace} CASCADE')


@pytest.mark.parametrize('made', sorted(_copies()))
def test_a_numeric_key_keeps_the_old_client_s_dtype_through_copies_and_edits(
    made: str, numeric_source: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    if numeric_source is None:
        return

    def act() -> Any:
        schema = _copies()[made](_described(numeric_source))
        seen: list[Any] = []

        def step(edit: Callable[[], Any]) -> None:
            r"""One edit or call, and what the schema then says of the two
            key columns: their data types and meanings, or the refusal."""
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    done = edit()
            except StructuredDataError as refused:
                seen.append((type(refused).__name__, refused.code))
                return
            seen.append(
                (
                    schema['users']['user_id'].dtype,
                    schema['orders']['user_id'].dtype,
                    schema['orders']['user_id'].stype,
                    done,
                )
            )

        link = Relationship('orders', 'user_id', 'users', 'user_id')
        step(lambda: None)
        step(lambda: _payload(schema, QUERY, NUMERIC_IDS))
        step(lambda: setattr(schema, 'relationships', []))
        step(lambda: _payload(schema, QUERY, NUMERIC_IDS))
        step(lambda: setattr(schema, 'relationships', [link]))
        step(lambda: setattr(schema['users'], 'primary_key', None))
        step(lambda: setattr(schema['users'], 'primary_key', 'user_id'))
        step(lambda: _payload(schema, QUERY, NUMERIC_IDS))
        return seen

    outcomes = _both(monkeypatch, act)
    assert outcomes['core'] == outcomes['old']
    # The key keeps its digits: read as text, as the reference reads it,
    # and the request built over it.
    assert outcomes['old'][0][0] == 'string'
    assert isinstance(outcomes['old'][1][3], list), outcomes['old'][1]


# --- rule 11: the core route runs no code of the old engine or old PQL --------------


def test_the_core_route_runs_no_code_of_the_old_engine_or_the_old_pql(
    tmp_path: Any,
) -> None:
    # The structure design's rule 11, checked here (T3.16): with every core
    # route forced, describing, editing, materializing and predicting call
    # no function of `sdm_relational_engine` or `predictive_query`, where
    # both are installed. That neither is imported at all, from before the
    # client's first import, is test_t316_the_schema_as_it_stands.py's
    # test_the_core_route_needs_nothing_of_the_old_pql_or_engine. The
    # profile hook is this test's own, in a process of its own.
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    script = f"""
    import importlib.util, os, sys, warnings
    from pathlib import Path
    import duckdb, pandas as pd
    from sdm_client import KumoRelational, PredictiveQuery, SDMClient, Schema
    from sdm_connectors import DuckDBSource

    spec = importlib.util.spec_from_file_location(
        'scripted_endpoint',
        {str(root / 'contract/runners/scripted_endpoint.py')!r},
    )
    endpoint = importlib.util.module_from_spec(spec)
    sys.modules['scripted_endpoint'] = endpoint
    spec.loader.exec_module(endpoint)

    OLD = []
    for name in ('predictive_query', 'sdm_relational_engine'):
        found = importlib.util.find_spec(name)
        if found is not None:
            OLD.append(str(Path(found.origin).parent) + os.sep)
    ran = set()

    def watch(frame, event, arg):
        if event == 'call':
            path = frame.f_code.co_filename
            if any(path.startswith(old) for old in OLD):
                ran.add(f'{{path}}:{{frame.f_code.co_name}}')

    connection = duckdb.connect()
    for statement in {list(SHOP)!r}:
        connection.execute(statement)
    with endpoint.serve() as (url, received):
        sys.setprofile(watch)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                schema = Schema.from_source(DuckDBSource(connection))
                schema['users']['age'].stype = 'categorical'
                request = PredictiveQuery({QUERY!r}, schema).materialize(
                    indices=[0, 1], anchor_time=pd.Timestamp('2024-03-01')
                )
                with SDMClient(url, max_retries=0) as client:
                    result = client.predict(request, model=KumoRelational())
        finally:
            sys.setprofile(None)
    print(len(result))
    for each in sorted(ran):
        print(each)
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if key != _routing.ENVIRONMENT
    }
    done = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=300,
        env={**environment, _routing.ENVIRONMENT: CORE},
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split('\n')[0] == '2'
    assert done.stdout.split('\n')[1:] == [''], done.stdout
