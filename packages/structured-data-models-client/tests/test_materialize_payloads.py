# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warehouse payload labels, lag windows and relation joins."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource
from test_materialize_options import ANCHOR, QUERY, _connection, _materialize

from sdm_client import PredictiveQuery, Request, Schema
from sdm_client._rust.spec import held_schema
from sdm_client.schema import _described_on_the_core


@pytest.mark.parametrize(
    ('query', 'anchor', 'lag', 'days', 'kind', 'context_count'),
    [
        (QUERY, '2024-03-01', 0, 30, 'regression', 112),
        (
            'PREDICT SUM(orders.price, 0, 7, days) FOR EACH users.user_id',
            '2023-03-01',
            2,
            7,
            'regression',
            64,
        ),
        (
            'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id',
            '2023-03-01',
            2,
            30,
            'classification',
            8,
        ),
        (
            'PREDICT users.age FOR EACH users.user_id',
            '2023-03-01',
            2,
            0,
            'regression',
            6,
        ),
    ],
    ids=['regression', 'regression-lags', 'binary-lags', 'static-lags-ignored'],
)
def test_materialized_payload_labels_lags_and_links(
    query, anchor, lag, days, kind, context_count
):
    connection = _connection()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(connection))
        request = PredictiveQuery(query, schema).materialize(
            indices=[0, 1], anchor_time=pd.Timestamp(anchor), lag_timesteps=lag
        )
    [payload] = request._payloads
    assert request.entity_table == 'users'
    assert request.indices == [0, 1]
    assert payload['client_generated_keys'] == {'orders': '__node_id'}
    assert 'client_analysis_delta' not in payload
    config = payload['inference']['inference_config']
    expected = dict(
        kind=kind,
        num_estimators=1,
        column_shuffle=False,
        category_shuffle=False,
        hop_shuffle=False,
    )
    if kind == 'classification':
        expected['class_shuffle'] = False
    else:
        expected.update(target_transforms=['quantile'], output_type='median')
    assert config == expected

    def value(entity, start):
        if not days:
            return connection.execute(
                'SELECT age FROM users WHERE user_id = ?', [entity]
            ).fetchone()[0]
        aggregate = (
            'COUNT(*) > 0'
            if kind == 'classification'
            else 'coalesce(sum(price), 0)'
        )
        result = connection.execute(
            f'SELECT {aggregate} FROM orders WHERE user_id = ? AND ts > ? AND ts <= ?',
            [
                entity,
                start.tz_localize(None),
                (start + pd.Timedelta(days=days)).tz_localize(None),
            ],
        ).fetchone()[0]
        return result if kind == 'classification' else float(np.float32(result))

    for side in ['context', 'predict']:
        table = payload[side]['instance_table']
        lag_columns = [f'__kumo_arl{i}__' for i in range(lag)] if days else []
        assert table['columns'] == [
            'instance_id',
            '__kumo_entity_ref_0',
            '__kumo_anchor_time',
            *(['TARGET'] if side == 'context' else []),
            *lag_columns,
        ]
        assert len(table['rows']) == (context_count if side == 'context' else 2)
        instances = {row[0]: row[1] for row in table['rows']}
        anchors = {row[0]: pd.Timestamp(row[2]) for row in table['rows']}
        assert len(instances) == len(table['rows'])
        if side == 'predict':
            assert list(instances.values()) == [0, 1]
            assert set(anchors.values()) == {pd.Timestamp(anchor, tz='UTC')}
        else:
            assert set(instances.values()) == set(
                range(8) if days else range(2, 8)
            )
        for row in table['rows']:
            cells = dict(zip(table['columns'], row, strict=True))
            start = pd.Timestamp(cells['__kumo_anchor_time'])
            if side == 'context':
                if days:
                    assert start + pd.Timedelta(days=days) <= pd.Timestamp(
                        anchor, tz='UTC'
                    )
                assert cells['TARGET'] == value(row[1], start)
                if kind == 'classification':
                    assert type(cells['TARGET']) is bool
            for i, column in enumerate(lag_columns):
                assert cells[column] == value(
                    row[1], start - pd.Timedelta(days=days * (i + 1))
                )
                if kind == 'classification':
                    assert type(cells[column]) is bool
        related = payload[side]['related_tables']
        assert set(related) == {'users', 'orders'}
        users = related['users']
        assert len(users['rows']) == len(instances)
        assert {row[0] for row in users['rows']} == set(instances)
        if not days:
            assert 'age' not in users['columns']  # Withheld prediction target.
        for row in users['rows']:
            cells = dict(zip(users['columns'], row, strict=True))
            assert cells['user_id'] == instances[cells['instance_id']]
            expected_age, expected_country = connection.execute(
                'SELECT age, country FROM users WHERE user_id = ?',
                [cells['user_id']],
            ).fetchone()
            assert cells['country'] == expected_country
            if days:
                assert cells['age'] == expected_age
        orders = related['orders']
        # Compare the complete ordered history, so omissions, duplicates and
        # permutations cannot pass merely by containing valid source rows.
        expected_orders = []
        for instance, entity in instances.items():
            source_rows = connection.execute(
                'SELECT user_id, price, ts FROM orders '
                'WHERE user_id = ? AND ts <= ? '
                'ORDER BY ts DESC, order_id ASC',
                [entity, anchors[instance].tz_localize(None)],
            ).fetchall()
            expected_orders.extend(
                (instance, user_id, price, pd.Timestamp(ts, tz='UTC'))
                for user_id, price, ts in source_rows
            )
        actual_orders = []
        for row in orders['rows']:
            cells = dict(zip(orders['columns'], row, strict=True))
            actual_orders.append(
                (
                    cells['instance_id'],
                    cells['user_id'],
                    cells['price'],
                    pd.Timestamp(cells['ts']),
                )
            )
        assert actual_orders == expected_orders


def test_seed_controls_materialization_reproducibly():
    request = _materialize()
    assert request._payloads == _materialize(random_seed=42)._payloads
    assert request._payloads != _materialize(random_seed=7)._payloads


def test_regression_statistic_is_recorded():
    request = _materialize(query=QUERY, prediction_statistic='mean')
    assert request._payloads[0]['output'] == {
        'fields': ['prediction'],
        'prediction_statistic': 'mean',
    }


NAMED = {'buyers': 'users', 'purchases': 'orders'}


def _named(**options: Any) -> tuple[Request, Schema]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(
            DuckDBSource(_connection()),
            NAMED,
            primary_keys={'buyers': 'user_id', 'purchases': 'order_id'},
        )
        request = PredictiveQuery(
            'PREDICT SUM(purchases.price, 0, 30, days) FOR EACH buyers.user_id',
            schema,
        ).materialize(indices=[0, 1], anchor_time=ANCHOR, **options)
    return (request, schema)


def test_tables_named_are_sampled_as_the_old_route_samples_them() -> None:
    core, schema = _named()
    assert _described_on_the_core(schema)
    assert {t['name']: t['source'] for t in held_schema(schema)['tables']} == {
        name: {'source': [source]} for name, source in NAMED.items()
    }
    payload = core._payloads[0]
    assert set(payload['context']['related_tables']) == {'buyers', 'purchases'}
    assert core.entity_table == 'buyers'

    def original_names(value):
        if isinstance(value, dict):
            return {
                NAMED.get(key, key): original_names(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [original_names(item) for item in value]
        if isinstance(value, str):
            return NAMED.get(value, value)
        return value

    restored = original_names(payload)
    restored['metadata']['source_query'] = QUERY
    assert restored == _materialize()._payloads[0]


def test_a_table_mapped_to_another_source_table_samples_that_table() -> None:

    def sampled() -> Any:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            connection = _connection()
            schema = Schema.from_source(DuckDBSource(connection))
            connection.execute(
                'CREATE TABLE other_orders AS SELECT order_id, user_id, price * 2 AS price, ts FROM orders'
            )
            schema.tables['orders'].source_name = 'other_orders'
            return (
                PredictiveQuery(QUERY, schema)
                .materialize(indices=[0, 1], anchor_time=ANCHOR)
                ._payloads[0]
            )

    core = sampled()
    expected = _materialize(
        after=('UPDATE orders SET price = price * 2',)
    )._payloads[0]
    assert core == expected
    original = _materialize()._payloads[0]['context']['instance_table']['rows']
    actual = core['context']['instance_table']['rows']
    assert len(actual) == len(original)
    for got, before in zip(actual, original, strict=True):
        assert got[:3] == before[:3]
        assert got[3] == before[3] * 2


def _context_row(payload: dict[str, Any], instance: int) -> list[Any]:
    rows = payload['context']['instance_table']['rows']
    [row] = [row for row in rows if row[0] == instance]
    return row


NAN = "UPDATE orders SET price = 'NaN'::DOUBLE WHERE order_id = 50"

FINITE = (
    'INSERT INTO orders SELECT 1000, 2, 5.0, ts FROM orders WHERE order_id = 50'
)


def test_a_nan_in_a_label_s_window_is_skipped_as_the_reference_skips_it() -> (
    None
):
    only = _materialize(after=(NAN,))._payloads[0]
    row = _context_row(only, 22)
    assert row[1:3] == [2, '2023-12-02T00:00:00.000000Z']
    assert row[3] == 0.0
    mixed = _materialize(after=(NAN, FINITE))._payloads[0]
    assert _context_row(mixed, 22)[3] == 5.0
    for payload, changes in [(only, (NAN,)), (mixed, (NAN, FINITE))]:
        connection = _connection(*changes)
        table = payload['context']['instance_table']
        assert len(table['rows']) == 112
        for row in table['rows']:
            anchor = pd.Timestamp(row[2])
            [(expected,)] = connection.execute(
                'SELECT coalesce(sum(price) FILTER (WHERE NOT isnan(price)), 0) FROM orders WHERE user_id = ? AND ts > ? AND ts <= ?',
                [
                    row[1],
                    anchor.tz_localize(None),
                    (anchor + pd.Timedelta(days=30)).tz_localize(None),
                ],
            ).fetchall()
            assert row[3] == float(np.float32(expected))


def test_a_table_named_as_the_core_s_seeds_is_read() -> None:
    payload = _materialize(
        'CREATE TABLE __sdm_seeds__ (order_id BIGINT PRIMARY KEY, user_id BIGINT, price DOUBLE, ts TIMESTAMP)',
        'INSERT INTO __sdm_seeds__ SELECT * FROM orders',
        'DROP TABLE orders',
        query=QUERY.replace('orders.', '__sdm_seeds__.'),
    )._payloads[0]
    assert len(payload['context']['instance_table']['rows']) == 112
    assert '__sdm_seeds__' in payload['context']['related_tables']

    def original_names(value):
        if isinstance(value, dict):
            return {
                'orders' if key == '__sdm_seeds__' else key: original_names(
                    item
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [original_names(item) for item in value]
        return 'orders' if value == '__sdm_seeds__' else value

    restored = original_names(payload)
    restored['metadata']['source_query'] = QUERY
    assert restored == _materialize()._payloads[0]


@pytest.mark.parametrize(
    ('setup', 'indices', 'cast'),
    [
        ((), ['0', '1'], [0, 1]),
        ((), [True, False], [1, 0]),
        (
            (
                'ALTER TABLE orders ALTER user_id TYPE VARCHAR',
                'ALTER TABLE users ALTER user_id TYPE VARCHAR',
            ),
            [0, 1],
            ['0', '1'],
        ),
    ],
)
def test_entities_are_cast_to_the_key_s_type_as_the_reference_casts_them(
    setup: tuple[str, ...], indices: list[Any], cast: list[Any]
) -> None:
    payload = _materialize(*setup, indices=indices)._payloads[0]
    predicted = payload['predict']['instance_table']['rows']
    assert [row[1] for row in predicted] == cast
    assert payload == _materialize(*setup, indices=cast)._payloads[0]


@pytest.mark.parametrize('aggregate', ['SUM', 'COUNT', 'AVG', 'static'])
def test_lags_use_each_instance_own_past_windows(aggregate):
    if aggregate == 'static':
        query = 'PREDICT users.age FOR EACH users.user_id'
    elif aggregate == 'COUNT':
        query = (
            'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'
        )
    else:
        query = f'PREDICT {aggregate}(orders.price, 0, 30, days) FOR EACH users.user_id'
    payload = _materialize(query=query, lag_timesteps=2)._payloads[0]
    baseline = _materialize(query=query)._payloads[0]
    connection = _connection()
    for half in ('context', 'predict'):
        table = payload[half]['instance_table']
        lagged = [
            name for name in table['columns'] if name.startswith('__kumo_arl')
        ]
        assert lagged == (
            [] if aggregate == 'static' else ['__kumo_arl0__', '__kumo_arl1__']
        )
        for row in table['rows']:
            for lag, name in enumerate(lagged):
                end = pd.Timestamp(row[2]).tz_localize(None) - pd.Timedelta(
                    days=30 * lag
                )
                start = end - pd.Timedelta(days=30)
                expression = (
                    'COUNT(*) > 0'
                    if aggregate == 'COUNT'
                    else f'{aggregate}(price)'
                )
                [(expected,)] = connection.execute(
                    f'SELECT {expression} FROM orders WHERE user_id = ? AND ts > ? AND ts <= ?',
                    [row[1], start, end],
                ).fetchall()
                if expected is None and aggregate == 'SUM':
                    expected = 0.0
                if expected is not None and aggregate != 'COUNT':
                    expected = float(np.float32(expected))
                assert row[table['columns'].index(name)] == expected
        keep = [
            i for i, name in enumerate(table['columns']) if name not in lagged
        ]
        table['columns'] = [table['columns'][i] for i in keep]
        table['rows'] = [[row[i] for i in keep] for row in table['rows']]
    columns = payload['schema']['instance_table']['columns']
    for name in list(columns):
        if name.startswith('__kumo_arl'):
            del columns[name]
    assert payload == baseline


TIME_TYPES = {
    'date': ['ALTER TABLE orders ALTER ts TYPE DATE'],
    'timestamptz': [
        "SET TimeZone = 'America/New_York'",
        'ALTER TABLE orders ALTER ts TYPE TIMESTAMPTZ',
    ],
    'text': [
        "ALTER TABLE orders ALTER ts TYPE VARCHAR USING strftime(ts, '%Y-%m-%d')"
    ],
    'text feature': [
        "CREATE OR REPLACE TABLE users AS SELECT *, strftime(DATE '2020-01-01' + user_id::INTEGER, '%Y-%m-%d') AS signup_date FROM users"
    ],
}


@pytest.mark.parametrize('kind', sorted(TIME_TYPES))
def test_source_time_representations_match_canonical_timestamps(kind):
    payload = _materialize(*TIME_TYPES[kind])._payloads[0]
    canonical = ()
    if kind == 'timestamptz':
        # Normalize each source instant to UTC, including daylight-saving changes.
        canonical = (
            "UPDATE orders SET ts = (ts AT TIME ZONE 'America/New_York') "
            "AT TIME ZONE 'UTC'",
        )
    elif kind == 'text feature':
        canonical = (
            'CREATE OR REPLACE TABLE users AS SELECT *, '
            "CAST(DATE '2020-01-01' + user_id::INTEGER AS TIMESTAMP) "
            'AS signup_date FROM users',
        )
    assert payload['context']['instance_table']['rows']
    assert payload == _materialize(*canonical)._payloads[0]


@pytest.mark.parametrize(
    'options',
    [
        {'lag_timesteps': float('nan')},
        {'lag_timesteps': np.longdouble('nan')},
        {'num_neighbors': [-1]},
        {'num_neighbors': [2, -1]},
        {'num_neighbors': ['3 OR TRUE']},
    ],
    ids=[
        'nan-lag',
        'numpy-wide-nan-lag',
        'negative-neighbor',
        'negative-second-hop',
        'neighbor-sql-expression',
    ],
)
def test_nan_lags_and_neighbor_expressions_have_their_declared_meaning(
    options: dict[str, Any],
) -> None:
    payload = _materialize(**options)._payloads[0]
    if 'lag_timesteps' in options:
        assert payload == _materialize(lag_timesteps=0)._payloads[0]
    else:
        neighbors = options['num_neighbors']
        normalized = (
            [1000]
            if neighbors == ['3 OR TRUE']
            else [max(0, n) for n in neighbors]
        )
        assert payload == _materialize(num_neighbors=normalized)._payloads[0]
        for half in ('context', 'predict'):
            tables = payload[half]['related_tables']
            if neighbors == [-1]:
                assert set(tables) == {'users'}
            else:
                assert tables['orders']['rows']
                counts = {}
                for row in tables['orders']['rows']:
                    counts[row[0]] = counts.get(row[0], 0) + 1
                if neighbors == [2, -1]:
                    assert max(counts.values()) == 2
                else:
                    assert max(counts.values()) > 3


SIGNED_UP = (
    'ALTER TABLE users ADD COLUMN signup TIMESTAMP',
    "UPDATE users SET signup = TIMESTAMP '2023-01-01' + to_days((user_id * 11)::INTEGER)",
)

STATIC = 'PREDICT users.age FOR EACH users.user_id'


@pytest.mark.parametrize(
    ('setup', 'options'),
    [
        ((), {'num_hops': 3}),
        ((), {'num_hops': 1}),
        ((), {'num_neighbors': [3, 2, 1]}),
        ((), {'num_neighbors': [3, 0, 5]}),
        ((), {'num_neighbors': [2, 0.5, 4]}),
        ((), {'num_neighbors': range(2, 4)}),
        ((), {'max_pq_iterations': 2}),
        ((), {'use_prediction_time': True}),
        ((), {'return_embeddings': True}),
        ((), {'exclude_cols_dict': {'orders': ['price'], 'users': ['age']}}),
        ((), {'context_anchor_time': pd.Timestamp('2023-12-01')}),
        (SIGNED_UP, {'query': STATIC, 'anchor_time': 'entity'}),
        (SIGNED_UP, {'query': STATIC, 'max_pq_iterations': 1.5}),
    ],
    ids=lambda value: (
        'signed-up'
        if value == SIGNED_UP
        else '-'.join(sorted(value))
        if isinstance(value, dict)
        else 'shop'
    ),
)
def test_sampling_options_change_only_the_requested_behavior(
    setup: tuple[str, ...], options: dict[str, Any]
) -> None:

    def make(arguments):
        arguments = dict(arguments)
        query = arguments.pop('query', QUERY)
        anchor = arguments.pop('anchor_time', ANCHOR)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(_connection(*setup)))
            return (
                PredictiveQuery(query, schema)
                .materialize(indices=[0, 1], anchor_time=anchor, **arguments)
                ._payloads[0]
            )

    payload = make(options)
    if options.get('anchor_time') == 'entity':
        for half in ('context', 'predict'):
            table = payload[half]['instance_table']
            at = table['columns'].index('__kumo_anchor_time')
            assert table['rows']
            for row in table['rows']:
                expected = pd.Timestamp('2023-01-01', tz='UTC') + pd.Timedelta(
                    days=row[1] * 11
                )
                assert pd.Timestamp(row[at]) == expected
        assert len(payload['predict']['instance_table']['rows']) == 2
    elif 'context_anchor_time' in options:
        # Default context starts one label window before the prediction anchor.
        expected = make(
            {
                'anchor_time': options['context_anchor_time']
                + pd.Timedelta(days=30)
            }
        )
        assert payload['context'] == expected['context']
        assert payload['schema'] == expected['schema']
        predicted = payload['predict']['instance_table']
        at = predicted['columns'].index('__kumo_anchor_time')
        assert [row[1] for row in predicted['rows']] == [0, 1]
        assert all(
            pd.Timestamp(row[at]) == ANCHOR.tz_localize('UTC')
            for row in predicted['rows']
        )
    else:
        baseline_options = {'query': options.get('query', QUERY)}
        if 'num_neighbors' in options:
            baseline_options['num_neighbors'] = [
                list(options['num_neighbors'])[0]
            ]
        expected = make(baseline_options)
        if options.get('use_prediction_time'):
            expected['inference']['use_prediction_time'] = True
        if options.get('return_embeddings'):
            expected['output']['fields'].append('embeddings')
        for table_name, excluded in options.get(
            'exclude_cols_dict', {}
        ).items():
            for half in ('context', 'predict'):
                table = expected[half]['related_tables'][table_name]
                keep = [
                    i
                    for i, name in enumerate(table['columns'])
                    if name not in excluded
                ]
                table['columns'] = [table['columns'][i] for i in keep]
                table['rows'] = [
                    [row[i] for i in keep] for row in table['rows']
                ]
            for column in excluded:
                del expected['schema']['related_tables'][table_name]['columns'][
                    column
                ]
        assert payload == expected
