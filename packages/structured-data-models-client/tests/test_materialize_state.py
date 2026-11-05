# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public warehouse materialization, provider boundaries and state reuse."""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Request, Schema, StructuredDataError
from sdm_client._rust import codec
from sdm_client._rust import materialize as _rust_materialize
from sdm_client._rust import run as _rust_run
from sdm_client._rust import schema as _rust_schema
from sdm_client._rust.run import run
from sdm_client.schema import _described_on_the_core

QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'

ANCHOR = pd.Timestamp('2024-03-01')


def _shop() -> list[str]:
    return json.loads((Path(__file__).parent / 'data/shop.json').read_text())[
        'statements'
    ]


def _connection(*setup: str) -> Any:
    connection = duckdb.connect()
    for statement in [*_shop(), *setup]:
        connection.execute(statement)
    return connection


def _materialize(
    *setup: str,
    query: str = QUERY,
    indices: list[Any] | None = None,
    after: tuple[str, ...] = (),
    **options: Any,
) -> Request:
    """``query`` materialized over the ``shop`` base and ``setup``; the
    statements ``after`` run once the schema is described."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        connection = _connection(*setup)
        schema = Schema.from_source(DuckDBSource(connection))
        for statement in after:
            connection.execute(statement)
        return PredictiveQuery(query, schema).materialize(
            indices=[0, 1] if indices is None else indices,
            anchor_time=ANCHOR,
            **options,
        )


@pytest.mark.parametrize('per_call', [1, 2])
def test_each_call_of_a_multi_call_request_takes_its_own_rows_lags(
    per_call: int,
) -> None:
    indices = [0, 1, 3, 5, 6]
    one = _materialize(indices=indices, lag_timesteps=2)._payloads
    several = _materialize(
        indices=indices, lag_timesteps=2, rows_per_call=per_call
    )._payloads
    assert len(one) == 1
    assert len(several) == -(-len(indices) // per_call)

    def lags(table: dict[str, Any]) -> list[list[Any]]:
        at = [
            i
            for i, c in enumerate(table['columns'])
            if c.startswith('__kumo_arl')
        ]
        assert len(at) == 2
        return [[row[i] for i in at] for row in table['rows']]

    assert [
        cells
        for payload in several
        for cells in lags(payload['predict']['instance_table'])
    ] == lags(one[0]['predict']['instance_table'])
    for payload in several:
        assert lags(payload['context']['instance_table']) == lags(
            one[0]['context']['instance_table']
        )


def test_an_unseeded_sample_is_a_new_sample_of_the_entities_asked() -> None:
    for _ in range(3):
        payload = _materialize(random_seed=None)._payloads[0]
        assert payload['metadata']['num_prediction_examples'] == 2
        rows = payload['context']['related_tables']
        assert set(rows) <= {'users', 'orders'}


def test_a_schema_not_described_on_the_core_is_described_and_materialized() -> (
    None
):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        described = Schema.from_source(DuckDBSource(_connection()))
        schema = described.snapshot()
        for table in schema.tables.values():
            for column in table.columns.values():
                object.__setattr__(column, '_source', None)
        asked = [
            PredictiveQuery(QUERY, held)
            .materialize(indices=[0, 1], anchor_time=ANCHOR, random_seed=7)
            ._payloads
            for held in (schema, described)
        ]
    assert asked[0] == asked[1]
    assert not _described_on_the_core(schema)


def test_the_largest_declared_seed_reaches_the_core_exactly() -> None:
    seed = 2**64 - 1
    assert codec.seed(seed) == '0xffffffffffffffff'
    assert float(seed) != seed and seed > 2**63 - 1
    description = {
        'tables': [
            {
                'name': 'users',
                'source': {'source': ['users']},
                'columns': [
                    {
                        'name': 'user_id',
                        'received': {'type': 'int64'},
                        'dtype': 'int',
                        'stype': 'ID',
                        'nullable': False,
                        'unique': True,
                    }
                ],
                'primary_key': ['user_id'],
                'time_column': None,
                'duplicate_keys': None,
            },
            {
                'name': 'orders',
                'source': {'source': ['orders']},
                'columns': [
                    {
                        'name': 'order_id',
                        'received': {'type': 'int64'},
                        'dtype': 'int',
                        'stype': 'ID',
                        'nullable': False,
                        'unique': True,
                    },
                    {
                        'name': 'user_id',
                        'received': {'type': 'int64'},
                        'dtype': 'int',
                        'stype': 'ID',
                        'nullable': False,
                        'unique': True,
                    },
                    {
                        'name': 'price',
                        'received': {'type': 'float64'},
                        'dtype': 'float',
                        'stype': 'numerical',
                        'nullable': True,
                        'unique': False,
                    },
                    {
                        'name': 'ts',
                        'received': {
                            'type': 'timestamp',
                            'unit': 'us',
                            'zone': None,
                        },
                        'dtype': 'date',
                        'stype': 'timestamp',
                        'nullable': True,
                        'unique': False,
                    },
                ],
                'primary_key': ['order_id'],
                'time_column': 'ts',
                'duplicate_keys': None,
            },
        ],
        'links': [
            [
                {'table': 'orders', 'column': 'user_id'},
                {'table': 'users', 'column': 'user_id'},
            ]
        ],
    }
    wire = {'int': 'int64', 'float': 'float64', 'date': 'timestamp[us]'}
    held = {
        'tables': [
            {
                **{
                    key: table[key]
                    for key in ('name', 'source', 'primary_key', 'time_column')
                },
                'columns': [
                    {
                        'name': column['name'],
                        'dtype': wire[column['dtype']],
                        'stype': column['stype'],
                        'stype_before_key': None,
                        'source': {
                            'relation': table['source'],
                            **{
                                key: column[key]
                                for key in (
                                    'dtype',
                                    'received',
                                    'nullable',
                                    'unique',
                                )
                            },
                        },
                    }
                    for column in table['columns']
                ],
            }
            for table in description['tables']
        ],
        'links': description['links'],
    }
    import pyarrow as pa

    asked: list[dict[str, Any]] = []

    class Stop(Exception):
        pass

    def provider(step: dict[str, Any], keys: Any) -> Any:
        request = step['request']
        asked.append(request)
        if request['request'] == 'time_bounds':
            return pa.table(
                {
                    'table_index': pa.array([0], pa.int32()),
                    'min_date': pa.array([0], pa.timestamp('us')),
                    'max_date': pa.array([2**50], pa.timestamp('us')),
                }
            )
        raise Stop

    operation = {
        'op': 'materialize',
        'provider': {
            'provider': 'host',
            'capabilities': {
                'optional': [
                    'time_bounds',
                    'sample_entities',
                    'neighbors',
                    'lookup',
                ],
                'max_keys_per_request': None,
                'sampling': 'seeded_stable',
                'max_seed': seed,
                'null_order': 'last',
            },
        },
        'schema': 1,
        'query': 2,
        'indices': 3,
        'anchor_time': codec.timestamp(int(ANCHOR.value), 'ns', None),
        'seed': codec.seed(seed),
    }
    with pytest.raises(StructuredDataError) as stopped:
        run(
            operation,
            {
                1: codec.attachment(held),
                2: codec.attachment(QUERY),
                3: pa.table({'indices': pa.array([0, 1], pa.int64())}),
            },
            provider=provider,
        )
    assert stopped.value.code == 'PROVIDER_ERROR'
    sample = asked[-1]
    assert sample['request'] == 'sample_entities'
    assert codec.decode_scalar(sample['seed']) == seed


def test_duckdb_refuses_a_seed_its_parser_refuses_by_name() -> None:
    for seed in (2**31, 2**70):
        with pytest.raises(StructuredDataError) as refused:
            _materialize(random_seed=seed)
        assert refused.value.code == 'INVALID_REQUEST'
        assert refused.value.details['field'] == 'seed'
        assert refused.value.details['max_seed'] == 2**31 - 1
    _materialize(random_seed=2**31 - 1)


def test_missing_indices_are_warned_of_once_per_query() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
    query = PredictiveQuery(QUERY, schema)
    shown = []
    for _ in range(3):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            query.materialize(indices=[0, 1], anchor_time=ANCHOR)
        shown.append(
            sum('MISSING_INDICES' in str(warning.message) for warning in caught)
        )
    assert shown == [1, 0, 0]


def test_the_provider_refuses_two_key_sets_bound_to_one_name() -> None:
    import pyarrow as pa

    from sdm_client._rust.providers.duckdb import DuckDBProvider, PlanNotRun

    def key_set(attachment: int) -> dict[str, Any]:
        return {
            'param': 'key_set',
            'keys': {'attachment': attachment, 'columns': []},
            'relation': 'seeds',
        }

    step = {
        'plan': {
            'plan': 'sql',
            'statements': [
                {
                    'dialect': 'duckdb',
                    'text': 'SELECT 1',
                    'params': [key_set(1), key_set(2)],
                }
            ],
        }
    }
    keys = {1: pa.table({'key': [1]}), 2: pa.table({'key': [2]})}
    with pytest.raises(PlanNotRun):
        DuckDBProvider(duckdb.connect())(step, keys)


def test_an_entity_that_does_not_cast_is_refused_after_the_index_warning() -> (
    None
):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
    query = PredictiveQuery(QUERY, schema)
    calls = [{'indices': ['abc']}, {'indices': [0, 1]}, {'indices': [0, 1]}]
    shown = []
    for index, arguments in enumerate(calls):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if index == 0:
                with pytest.raises(StructuredDataError) as refused:
                    query.materialize(anchor_time=ANCHOR, **arguments)
                assert refused.value.code == 'INVALID_REQUEST'
                assert refused.value.details['field'] == 'indices'
            else:
                request = query.materialize(anchor_time=ANCHOR, **arguments)
                assert request.indices == [0, 1]
                assert len(request._payloads) == 1
        shown.append(sum('Missing' in str(w.message) for w in caught))
    assert shown == [1, 0, 0]


def test_a_source_of_5000_columns_is_described_and_materialized_on_the_core(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features = ', '.join(f'(user_id * {i}) % 7 AS f{i}' for i in range(3, 5000))
    taken: list[str] = []
    real = _rust_run.run

    def counted(operation: dict[str, Any], *rest: Any, **named: Any) -> Any:
        finished = real(operation, *rest, **named)
        taken.append(operation['op'])
        return finished

    for module in (_rust_schema, _rust_materialize):
        monkeypatch.setattr(module, 'run', counted)
    request = _materialize(
        f'CREATE OR REPLACE TABLE users AS SELECT *, {features} FROM users'
    )
    assert taken == ['describe_source', 'materialize']
    users = request._payloads[0]['context']['related_tables']['users']
    assert len(users['columns']) == 5001


def _logged_texts(connection: Any) -> list[str]:
    """The statements DuckDB logged since its query log was turned on, in
    order, the log's own reads left out."""
    rows = connection.execute(
        "SELECT message FROM duckdb_logs WHERE type = 'QueryLog'"
    ).fetchall()
    return [text for (text,) in rows if text and 'duckdb_logs' not in text]


def _logged(connection: Any) -> int:
    """How many statements DuckDB logged since its query log was turned
    on, the log's own reads left out."""
    return len(_logged_texts(connection))


class _Located:
    """Observe engine fetch plans without replacing provider execution.

    Locate reads remain separately counted; raw driver logs must contain
    exactly the planned SQL after bounded driver bookkeeping is reconciled.
    """

    def __init__(self) -> None:
        self.statements = 0
        self.requests: list[dict[str, Any]] = []

    def observed_statements(self, logged: list[str]) -> int:
        """Count actual SQL, allowing only bounded driver bookkeeping.

        The allowed registrations come from each observed plan's key sets.
        A materialized-relation read is allowed only immediately after an
        observed table-info plan's own PRAGMA query-log entry.
        Extra data queries and unrequested bookkeeping remain counted and fail.
        """
        registrations: Counter[str] = Counter()
        pragmas: Counter[str] = Counter()
        planned = 0
        for request in self.requests:
            statements = request['plan'].get('statements', [])
            planned += len(statements)
            relations = {
                param['relation']
                for statement in statements
                for param in statement.get('params', [])
                if param.get('param') == 'key_set'
            }
            for relation in relations:
                name = relation.replace('"', '""')
                registrations[
                    f'CREATE TEMPORARY VIEW "{name}" AS SELECT 1'
                ] += 1
            for statement in statements:
                match = re.fullmatch(
                    r'PRAGMA table_info\("((?:[^"]|"")*)"\)', statement['text']
                )
                if match:
                    name = match[1].replace('""', '"').replace("'", "''")
                    pragmas[f"SELECT * FROM pragma_table_info('{name}');"] += 1
        materialized = re.compile(
            r'SELECT \* FROM "__duckdb_source_[0-9a-f]+_[0-9]+" AS "unnamed_relation_[0-9a-f]+"'
        )
        actual = 0
        previous = None
        for text in logged:
            if registrations[text]:
                registrations[text] -= 1
            elif (
                previous is not None
                and pragmas[previous]
                and materialized.fullmatch(text)
            ):
                pragmas[previous] -= 1
            else:
                actual += 1
            previous = text
        assert actual == planned, (
            f'unplanned driver statements: observed {actual}, planned {planned}'
        )
        return actual - self.statements

    def __enter__(self) -> _Located:
        from sdm_client._rust import codec, run

        located = self
        self._native = run.native

        class _Handle:
            def __init__(self, handle: Any) -> None:
                self._handle = handle

            def next(self) -> Any:
                control, sent = self._handle.next()
                step = codec.read(control, sent)
                if step.get('step') == 'fetch':
                    located.requests.extend(step['requests'])
                    for asked in step['requests']:
                        if asked.get('purpose') == 'locate':
                            located.statements += len(
                                asked['plan'].get('statements', [])
                            )
                return (control, sent)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._handle, name)

        class _Run:
            def __init__(self, real: Any) -> None:
                self._real = real

            def start(self, *arguments: Any) -> Any:
                return _Handle(self._real.start(*arguments))

            def __getattr__(self, name: str) -> Any:
                return getattr(self._real, name)

        class _Native:
            def __init__(self, real: Any) -> None:
                self._real = real
                self.Run = _Run(real.Run)

            def __getattr__(self, name: str) -> Any:
                return getattr(self._real, name)

        imported = self._native
        run.native = lambda: _Native(imported())
        return self

    def __exit__(self, *_: Any) -> None:
        from sdm_client._rust import run

        run.native = self._native


def _calls(
    queries: list[PredictiveQuery],
    connection: Any,
    *,
    expected_error: str | None = None,
    **arguments: Any,
) -> list[tuple[int, int]]:
    """Each query materialized in turn: its actual planned SQL statements
    recorded by the driver (excluding locate reads), and index warnings."""
    made = []
    for query in queries:
        before = len(_logged_texts(connection))
        with (
            warnings.catch_warnings(record=True) as caught,
            _Located() as located,
        ):
            warnings.simplefilter('always')
            if expected_error is not None:
                with pytest.raises(StructuredDataError) as refused:
                    query.materialize(anchor_time=ANCHOR, **arguments)
                assert refused.value.code == expected_error
            else:
                request = query.materialize(anchor_time=ANCHOR, **arguments)
                assert request._payloads
        warned = sum('Missing' in str(w.message) for w in caught)
        actual = located.observed_statements(_logged_texts(connection)[before:])
        made.append((actual, warned))
    return made


def _shop_logged() -> tuple[Any, Schema]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        connection = _connection()
        connection.execute("CALL enable_logging('QueryLog')")
        schema = Schema.from_source(DuckDBSource(connection))
    return (connection, schema)


def test_source_refresh_reads_time_bounds_without_repeating_index_checks() -> (
    None
):
    connection, schema = _shop_logged()
    query = PredictiveQuery(QUERY, schema)
    for call in range(3):
        with warnings.catch_warnings(), _Located() as located:
            warnings.simplefilter('ignore')
            query.materialize(anchor_time=ANCHOR, indices=[0, 1])
        bounds = [
            asked
            for asked in located.requests
            if asked['request'].get('request') == 'time_bounds'
        ]
        assert len(bounds) == 1
        assert bounds[0].get('purpose') is None
        indexes = [
            asked
            for asked in located.requests
            if asked['request'].get('request') == 'describe'
            and asked['request'].get('part') == 'indexes'
        ]
        assert len(indexes) == int(call == 0)


def test_a_call_the_core_refuses_before_its_check_leaves_it_to_the_next() -> (
    None
):
    connection, schema = _shop_logged()
    query = PredictiveQuery(QUERY, schema)
    assert _calls(
        [query],
        connection,
        indices=[0, 1],
        num_hops=7,
        expected_error='UNSUPPORTED_REQUEST',
    ) == [(0, 0)]
    assert _calls([query], connection, indices=[0, 1]) == [(6, 1)]
    other = PredictiveQuery(QUERY, schema)
    assert _calls(
        [other], connection, indices=[None, 1], expected_error='INVALID_REQUEST'
    ) == [(1, 1)]
    assert _calls([other], connection, indices=[0, 1]) == [(5, 0)]


@pytest.mark.parametrize(
    'held',
    [{'version': 99, 'memo': b'{}'}, {'version': 1, 'memo': b'not a memo'}],
)
def test_a_state_the_core_refuses_is_not_handed_back_again(
    held: dict[str, Any],
) -> None:
    connection, schema = _shop_logged()
    query = PredictiveQuery(QUERY, schema)
    query._relational_materializer._core_sampler_state['held'] = held
    with pytest.raises(StructuredDataError) as refused:
        query.materialize(indices=[0, 1], anchor_time=ANCHOR)
    assert refused.value.code in ('INVALID_REQUEST', 'INVALID_MESSAGE')
    assert 'held' not in query._relational_materializer._core_sampler_state
    assert _calls([query, query], connection, indices=[0, 1]) == [
        (6, 1),
        (5, 0),
    ]


@pytest.mark.parametrize('said', [True, False])
def test_the_host_drops_a_state_on_the_core_s_word_not_the_field_it_names(
    monkeypatch: pytest.MonkeyPatch, said: bool
) -> None:
    read = codec.read

    def apart(control: bytes, attachments: dict[int, Any]) -> dict[str, Any]:
        message = read(control, attachments)
        error = message.get('error')
        if message['message'] == 'error' and error is not None:
            assert error.get('drop_held') == ['sampler_state']
            assert error['fields']['field']['v'] == 'sampler_state.version'
            if said:
                del error['fields']['field']
            else:
                del error['drop_held']
        return message

    monkeypatch.setattr(codec, 'read', apart)
    _, schema = _shop_logged()
    query = PredictiveQuery(QUERY, schema)
    held = {'version': 99, 'memo': b'{}'}
    query._relational_materializer._core_sampler_state['held'] = held
    with pytest.raises(StructuredDataError):
        query.materialize(indices=[0, 1], anchor_time=ANCHOR)
    kept = query._relational_materializer._core_sampler_state
    assert ('held' in kept) is not said


def test_a_zoned_time_column_one_answer_holds_no_value_of_still_joins() -> None:
    """fix 6: a TIMESTAMPTZ column that one
    of a table's answers holds no value of (the referrers' last logins,
    read by their own lookup) keeps its unit there as in the others, so
    the answers join and ``materialize`` succeeds, as on the old route; a
    column holding no value was read to nanoseconds before its zone was
    looked at, and the join failed INTERNAL_ERROR."""
    connection = duckdb.connect()
    for statement in (
        'CREATE TABLE users (user_id INTEGER PRIMARY KEY, last_login TIMESTAMPTZ, country VARCHAR)',
        "INSERT INTO users SELECT i, TIMESTAMPTZ '2024-01-01 00:00:00+00' + i * INTERVAL 1 HOUR, 'c' || (i % 3) FROM range(40) t(i)",
        "INSERT INTO users SELECT i, NULL, 'r' FROM range(100, 105) t(i)",
        'CREATE TABLE orders (order_id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(user_id), referrer_id INTEGER REFERENCES users(user_id), ts TIMESTAMP, price DOUBLE)',
        "INSERT INTO orders SELECT i, i % 40, 100 + i % 5, TIMESTAMP '2024-01-10' + (i % 50) * INTERVAL 1 DAY, i + 0.5 FROM range(200) t(i)",
        'CREATE TABLE visits (visit_id INTEGER PRIMARY KEY, user_id INTEGER REFERENCES users(user_id), ts TIMESTAMP)',
        "INSERT INTO visits SELECT i, i % 40, TIMESTAMP '2024-01-10' + (i % 50) * INTERVAL 1 DAY FROM range(300) t(i)",
    ):
        connection.execute(statement)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
        request = PredictiveQuery(
            'PREDICT COUNT(visits.*, 0, 30, days) FOR EACH users.user_id',
            schema,
        ).materialize(indices=[0, 1], anchor_time=ANCHOR)
    [payload] = request._payloads
    assert payload['context']['instance_table']['rows']


_EDITABLE = (
    'ALTER TABLE orders ADD COLUMN ts2 TIMESTAMP',
    "UPDATE orders SET ts2 = ts + INTERVAL '10 days'",
    'ALTER TABLE orders ADD COLUMN alt BIGINT',
    'UPDATE orders SET alt = order_id + 1000',
    'CREATE TABLE extra AS SELECT i::BIGINT AS extra_id, (i % 8)::BIGINT AS user_id, i::DOUBLE AS w FROM range(20) t(i)',
)


def _edit(schema: Schema, edit: str) -> None:
    if edit == 'time_column':
        schema['orders'].time_column = 'ts2'
    elif edit == 'key':
        schema['orders'].primary_key = 'alt'
    else:
        schema.relationships = [
            link
            for link in schema.relationships
            if link.source_table != 'extra'
        ]


def _editable() -> tuple[Any, Schema]:
    from sdm_client import Relationship

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        connection = _connection(*_EDITABLE)
        connection.execute("CALL enable_logging('QueryLog')")
        schema = Schema.from_source(
            DuckDBSource(connection),
            primary_keys={
                'users': 'user_id',
                'orders': 'order_id',
                'extra': 'extra_id',
            },
            time_columns={'orders': 'ts'},
            relationships=[
                Relationship('orders', 'user_id', 'users', 'user_id'),
                Relationship('extra', 'user_id', 'users', 'user_id'),
            ],
        )
    return (connection, schema)


def _edited_between_calls(edit: str) -> list[tuple[int, int, str]]:
    """Two calls of one query, ``edit`` made to its schema between them:
    each call's statements, missing-index warnings and payload."""
    connection, schema = _editable()
    query = PredictiveQuery(QUERY, schema)
    made = []
    for call in range(2):
        if call == 1:
            _edit(schema, edit)
        before = len(_logged_texts(connection))
        with (
            warnings.catch_warnings(record=True) as caught,
            _Located() as located,
        ):
            warnings.simplefilter('always')
            payload = query.materialize(
                indices=[0, 1], anchor_time=ANCHOR
            )._payloads[0]
        actual = located.observed_statements(_logged_texts(connection)[before:])
        made.append(
            (
                actual,
                sum('Missing' in str(w.message) for w in caught),
                json.dumps(payload, sort_keys=True, default=str),
            )
        )
    return made


@pytest.mark.parametrize('edit', ['time_column', 'key', 'link'])
def test_a_schema_edited_between_two_calls_is_sampled_as_edited(
    edit: str,
) -> None:
    core = _edited_between_calls(edit)
    assert [made[:2] for made in core] == [
        (7, 1),
        (6, 1) if edit == 'link' else (7, 1),
    ]
    assert core[0][2] != core[1][2]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, schema = _editable()
        _edit(schema, edit)
        fresh = (
            PredictiveQuery(QUERY, schema)
            .materialize(indices=[0, 1], anchor_time=ANCHOR)
            ._payloads[0]
        )
    assert json.dumps(fresh, sort_keys=True, default=str) == core[1][2]


def test_a_second_call_reuses_the_index_check_but_refreshes_data() -> None:

    def made() -> list[tuple[int, int]]:
        connection, schema = _shop_logged()
        query = PredictiveQuery(QUERY, schema)
        return _calls([query] * 3, connection, indices=[0, 1])

    found = made()
    assert found == [(6, 1), (5, 0), (5, 0)]


def test_two_materializers_each_check_their_source_once() -> None:

    def made() -> list[tuple[int, int]]:
        connection, schema = _shop_logged()
        one = PredictiveQuery(QUERY, schema)
        two = PredictiveQuery(QUERY, schema)
        return _calls([one, two, one, two], connection, indices=[0, 1])

    found = made()
    assert found == [(6, 1), (6, 1), (5, 0), (5, 0)]


_DEEPER = (
    "CREATE TABLE items AS SELECT i::BIGINT AS item_id, 'Item ' || i || ' (the BEST<br>one)!' AS title, (i * 2.5)::DOUBLE AS cost FROM range(5) t(i)",
    'ALTER TABLE orders ADD COLUMN item_id BIGINT',
    'UPDATE orders SET item_id = order_id % 5',
)


def _deeper(*setup: str, edit: bool = True) -> Request:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        connection = _connection(*_DEEPER, *setup)
        schema = Schema.from_source(
            DuckDBSource(connection),
            primary_keys={
                'users': 'user_id',
                'orders': 'order_id',
                'items': 'item_id',
            },
            check_keys=False,
        )
        if edit:
            schema['items']['title'].stype = 'text'
        return PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1], anchor_time=ANCHOR
        )


def test_a_second_hop_and_a_text_feature_are_sampled_as_the_old_route_samples_them() -> (
    None
):
    core = _deeper()._payloads[0]
    items = core['context']['related_tables']['items']
    title = items['columns'].index('title')
    assert items['rows'][0][title][:3] == ['item', '0', 'the']
    for half in ('context', 'predict'):
        tables = core[half]['related_tables']
        items = tables['items']
        assert items['columns'] == ['instance_id', '__node_id', 'title', 'cost']
        lookup = {}
        for instance, node, words, cost in items['rows']:
            item_id = int(cost / 2.5)
            assert item_id in range(5)
            assert cost == item_id * 2.5
            assert words == ['item', str(item_id), 'the', 'best', 'one']
            assert (instance, node) not in lookup
            lookup[instance, node] = item_id
        orders = tables['orders']
        price_at = orders['columns'].index('price')
        item_at = orders['columns'].index('item_id')
        assert orders['rows'] and items['rows']
        for row in orders['rows']:
            original_order_id = int((row[price_at] - 10) / 1.5)
            assert lookup[row[0], row[item_at]] == original_order_id % 5
    assert _crossing(core) == 0
    assert _unreferenced(core) == 0

    assert core['schema']['related_tables']['items']['columns']['title'] == {
        'dtype': 'stringlist',
        'stype': 'text',
    }


def _crossing(payload: dict[str, Any]) -> int:
    """How many foreign keys of ``payload`` name no row of their own
    example in the table they refer to."""
    crossing = 0
    for half in ('context', 'predict'):
        tables = payload[half]['related_tables']
        held = set()
        for name, table in tables.items():
            columns = table['columns']
            key = '__node_id' if '__node_id' in columns else columns[1]
            at = columns.index(key)
            held.update((name, row[at], row[0]) for row in table['rows'])
        for link in payload['schema']['relationships']:
            if 'source_table' not in link or link['source_table'] not in tables:
                continue
            table = tables[link['source_table']]
            at = table['columns'].index(link['source_columns'][0])
            for row in table['rows']:
                if row[at] is None:
                    continue
                value = int(row[at]) if isinstance(row[at], float) else row[at]
                crossing += (link['target_table'], value, row[0]) not in held
    return crossing


def _unreferenced(payload: dict[str, Any]) -> int:
    """How many rows of ``payload``'s items no order of their own example
    refers to."""
    unreferenced = 0
    for half in ('context', 'predict'):
        tables = payload[half]['related_tables']
        if 'items' not in tables:
            continue
        orders = tables['orders']
        at = orders['columns'].index('item_id')
        referred = {
            (row[0], int(row[at]))
            for row in orders['rows']
            if row[at] is not None
        }
        items = tables['items']
        key = items['columns'].index('__node_id')
        unreferenced += sum(
            (row[0], row[key]) not in referred for row in items['rows']
        )
    return unreferenced


@pytest.mark.parametrize(
    'setup',
    [
        ('UPDATE orders SET item_id = NULL WHERE order_id % 3 = 0',),
        ('UPDATE orders SET order_id = NULL WHERE order_id % 13 = 6',),
    ],
)
def test_a_null_key_the_walk_reaches_links_nothing(
    setup: tuple[str, ...],
) -> None:
    core = _deeper(*setup, edit=False)._payloads[0]
    assert _crossing(core) == 0
    assert _unreferenced(core) == 0


def test_every_foreign_key_looked_up_null_is_the_reference_s_request() -> None:

    def made() -> list[tuple[int, int]]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            connection = _connection(
                *_DEEPER, 'UPDATE orders SET item_id = NULL'
            )
            connection.execute("CALL enable_logging('QueryLog')")
            schema = Schema.from_source(
                DuckDBSource(connection),
                primary_keys={
                    'users': 'user_id',
                    'orders': 'order_id',
                    'items': 'item_id',
                },
            )
            query = PredictiveQuery(QUERY, schema)
        return _calls([query, query], connection, indices=[0, 1])

    found = made()
    assert found == [(7, 1), (6, 0)]
    core = _deeper('UPDATE orders SET item_id = NULL', edit=False)._payloads[0]
    for half in ('context', 'predict'):
        tables = core[half]['related_tables']
        assert 'items' not in tables
        orders = tables['orders']
        assert orders['columns'] == [
            'instance_id',
            '__node_id',
            'price',
            'ts',
            'user_id',
        ]
        assert orders['rows']
    assert _crossing(core) == 0
    assert _unreferenced(core) == 0
    # An entirely unreachable table and its foreign-key column are omitted.
    assert core == _materialize()._payloads[0]


def test_query_accounting_rejects_an_extra_provider_data_scan(monkeypatch):
    from sdm_client._rust.providers.duckdb import DuckDBProvider

    connection, schema = _shop_logged()
    query = PredictiveQuery(QUERY, schema)
    original = DuckDBProvider.__call__

    def duplicate_scan(provider, *args, **kwargs):
        answer = original(provider, *args, **kwargs)
        provider._connection.execute('SELECT * FROM users').fetchall()
        return answer

    monkeypatch.setattr(DuckDBProvider, '__call__', duplicate_scan)
    with pytest.raises(AssertionError, match='unplanned driver statements'):
        _calls([query], connection, indices=[0, 1])


@pytest.mark.parametrize(
    'extra',
    [
        'SELECT * FROM users',
        'CREATE TEMPORARY VIEW "__sdm_seeds__" AS SELECT 1',
        'SELECT * FROM "__duckdb_source_aabb_1" AS "unnamed_relation_ccdd"',
    ],
)
def test_query_accounting_rejects_unrequested_driver_bookkeeping(extra):
    observed = _Located()
    observed.requests = [
        {'plan': {'statements': [{'text': 'SELECT 1', 'params': []}]}}
    ]
    with pytest.raises(AssertionError, match='unplanned driver statements'):
        observed.observed_statements(['SELECT 1', extra])
