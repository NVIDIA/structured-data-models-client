# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Materialization option validation through the public DuckDB API."""

from __future__ import annotations

import datetime
import json
import re
import warnings
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Request, Schema, StructuredDataError
from sdm_client._rust import codec

QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
BINARY = 'PREDICT COUNT(orders.*, 0, 30, days) > 0 FOR EACH users.user_id'
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


@pytest.mark.parametrize(
    ('value', 'reason'),
    [
        (-1, 'negative'),
        (1.5, 'not_a_whole_number'),
        (2.0, 'not_a_whole_number'),
        (True, 'not_a_whole_number'),
        ('2', 'not_a_number'),
    ],
)
def test_a_refused_lag_is_said_plainly_under_the_reference_s_code(
    value: Any, reason: str
) -> None:
    with pytest.raises(StructuredDataError) as refused:
        _materialize(lag_timesteps=value)
    assert refused.value.code == 'INVALID_REQUEST'
    assert refused.value.details['field'] == 'lag_timesteps'
    assert refused.value.details['reason'] == reason
    assert 'a whole number of zero or more' in str(refused.value)
    assert '(' not in str(refused.value)
    assert 'not_a_' not in str(refused.value)


@pytest.mark.parametrize(
    ('query', 'value', 'lags'),
    [
        ('PREDICT users.age FOR EACH users.user_id', Decimal(2), 0),
        ('PREDICT users.age FOR EACH users.user_id', Fraction(2, 1), 0),
        ('PREDICT users.age FOR EACH users.user_id', Decimal('0.1'), 0),
        ('PREDICT users.age FOR EACH users.user_id', Decimal('1e-400'), 0),
        ('PREDICT users.age FOR EACH users.user_id', Decimal('1e400'), 0),
        (QUERY, Decimal('1e-400'), None),
        (QUERY, Decimal('1e400'), None),
        (QUERY, Decimal(0), 0),
        (QUERY, Fraction(0), 0),
        (QUERY, Decimal(2), None),
        (QUERY, Fraction(1, 3), None),
        (QUERY, Decimal(-1), None),
    ],
)
def test_a_decimal_or_fraction_lag_is_met_as_the_reference_meets_it(
    query: str, value: Any, lags: int | None
) -> None:
    if lags is None:
        with pytest.raises(StructuredDataError) as refused:
            _materialize(query=query, lag_timesteps=value)
        assert refused.value.code == 'INVALID_REQUEST'
        assert refused.value.details['field'] == 'lag_timesteps'
        return
    columns = _materialize(query=query, lag_timesteps=value)._payloads[0][
        'context'
    ]['instance_table']['columns']
    assert len([c for c in columns if c.startswith('__kumo_arl')]) == lags


def test_a_decimal_seed_is_said_as_the_reference_says_it() -> None:
    with pytest.raises(StructuredDataError) as refused:
        _materialize(random_seed=Decimal(3))
    assert refused.value.code == 'INVALID_REQUEST'
    assert 'random_seed must be a whole number' in str(refused.value)


@pytest.mark.parametrize(
    'scalar', [float, np.float32, np.float64, np.longdouble]
)
def test_numeric_nan_is_only_encoded_on_the_option_attachment(scalar) -> None:
    value = scalar('nan')
    assert (
        codec.attachment(
            {'lag_timesteps': value}, named=True, option_values=True
        )
        == b'{"lag_timesteps": NaN}'
    )
    with pytest.raises(StructuredDataError) as refused:
        codec.attachment({'lag_timesteps': value}, named=True)
    assert refused.value.code == 'INVALID_REQUEST'
    assert refused.value.details == {'field': 'lag_timesteps'}


@pytest.mark.parametrize(
    ('options', 'field'),
    [
        ({'num_hops': None}, 'num_hops'),
        ({'max_pq_iterations': 0}, 'max_pq_iterations'),
        ({'use_prediction_time': None}, 'use_prediction_time'),
        ({'exclude_cols_dict': {'orders': ['order_id']}}, 'exclude_cols_dict'),
        ({'exclude_cols_dict': {'nowhere': []}}, 'exclude_cols_dict'),
        ({'context_anchor_time': 'entity'}, 'context_anchor_time'),
    ],
    ids=lambda value: (
        '-'.join(map(str, value)) if isinstance(value, dict) else value
    ),
)
def test_a_sampling_option_the_engine_refuses_is_the_caller_s_error(
    options: dict[str, Any], field: str
) -> None:
    with pytest.raises(StructuredDataError) as refused:
        _materialize(**options)
    assert refused.value.code == 'INVALID_REQUEST'
    assert refused.value.details.get('field') == field


@pytest.mark.parametrize('expression', ['x', 'nonsense'])
def test_an_invalid_neighbor_expression_is_refused_by_its_provider(
    expression: str,
) -> None:
    with pytest.raises(StructuredDataError) as refused:
        _materialize(num_neighbors=[expression])
    assert refused.value.code == 'PROVIDER_ERROR'
    assert refused.value.details == {
        'request': 'neighbors',
        'kind': 'data_source',
        'retryable': False,
        'driver_class': '_duckdb.BinderException',
    }


@pytest.mark.parametrize(
    'given', [['num_estimators'], object()], ids=['a-list', 'not-json']
)
def test_inference_options_not_a_mapping_are_refused(given: Any) -> None:
    with pytest.raises(StructuredDataError) as refused:
        _materialize(inference_config=given)
    assert refused.value.code == 'INVALID_REQUEST'


@pytest.mark.parametrize(
    'query, title',
    [
        (QUERY, 'RegressionInferenceConfig'),
        (BINARY, 'ClassificationInferenceConfig'),
    ],
)
@pytest.mark.parametrize(
    'given',
    [
        float('inf'),
        float('-inf'),
        float('nan'),
        np.float32('inf'),
        np.float64('inf'),
        np.longdouble('inf'),
    ],
)
def test_nonfinite_estimators_keep_the_task_validators_public_message(
    query: str, title: str, given: float
) -> None:
    from pydantic import ValidationError

    expected = ValidationError.from_exception_data(
        title,
        [{'type': 'finite_number', 'loc': ('num_estimators',), 'input': given}],
    )
    with pytest.raises(StructuredDataError) as refused:
        _materialize(query=query, inference_config={'num_estimators': given})
    assert refused.value.code == 'INVALID_REQUEST'
    assert str(refused.value) == f'[INVALID_REQUEST] {expected}'
    assert refused.value.details['field'] == 'inference_config'
    assert refused.value.details['key'] == 'num_estimators'


def test_invalid_statistic_keeps_its_approved_early_precedence() -> None:
    with pytest.raises(StructuredDataError) as refused:
        _materialize(
            prediction_statistic='unknown',
            inference_config={'num_estimators': float('inf')},
        )
    assert refused.value.code == 'INVALID_REQUEST'
    assert refused.value.details['field'] == 'prediction_statistic'


@pytest.mark.parametrize('given', [float('inf'), float('nan')])
def test_other_nonfinite_inference_fields_are_still_refused(
    given: float,
) -> None:
    for config, message in [
        ({'column_shuffle': given}, 'is true or false'),
        ({'target_transforms': [given]}, 'target_transforms'),
        ({'class_shuffle': given}, 'not an inference option'),
    ]:
        with pytest.raises(StructuredDataError) as refused:
            _materialize(inference_config=config)
        assert refused.value.code == 'INVALID_REQUEST'
        assert message in str(refused.value)


@pytest.mark.parametrize(
    'depth', [500, 100000], ids=['past-the-core', 'past-json']
)
def test_inference_options_nested_deep_are_refused_never_a_crash(
    depth: int,
) -> None:
    nested: Any = 1
    for _ in range(depth):
        nested = [nested]
    with pytest.raises(StructuredDataError) as refused:
        _materialize(inference_config={'num_estimators': nested})
    assert refused.value.code == 'INVALID_REQUEST'
    if depth == 500:
        assert refused.value.details['field'] == 'values'
        assert refused.value.details['category'] == 'depth'


@pytest.mark.parametrize('value', [float('inf'), -float('inf')])
def test_an_infinite_lag_is_refused_on_the_core_route(value: float) -> None:
    with pytest.raises(StructuredDataError) as refused:
        _materialize(lag_timesteps=value)
    assert refused.value.code == 'INVALID_REQUEST'
    assert refused.value.details == {
        'field': 'lag_timesteps',
        'kind': 'invalid_request',
        'retryable': False,
    }
    assert str(refused.value).startswith(
        '[INVALID_REQUEST] lag_timesteps carries a value'
    )


@pytest.mark.parametrize(
    'indices',
    [[2**100], [2**63, 0], [2**64, 0], [datetime.date(2024, 1, 1), 0]],
)
def test_entities_that_do_not_cast_are_the_caller_s_error_on_the_core(
    indices: list[Any],
) -> None:
    with pytest.raises(StructuredDataError) as refused:
        _materialize(indices=indices)
    assert refused.value.code == 'INVALID_REQUEST'
    assert refused.value.details['field'] == 'indices'
    assert refused.value.details['position'] == 0


@pytest.mark.parametrize(
    ('tables', 'key'),
    [
        (None, ['user_id']),
        ({'users': 'users', 'orders': 'orders'}, ('user_id',)),
    ],
    ids=['every-table-list', 'tables-named-tuple'],
)
def test_a_key_of_one_column_as_a_sequence_is_pinned_on_both_routes(
    tables: Any, key: Any
) -> None:

    def built() -> tuple[Any, Any]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(
                DuckDBSource(_connection()), tables, primary_keys={'users': key}
            )
            try:
                PredictiveQuery(QUERY, schema).materialize(
                    indices=[0, 1], anchor_time=ANCHOR
                )
            except StructuredDataError as refused:
                return (schema, refused.code)
            return (schema, 'success')

    schema, outcome = built()
    assert schema['users'].primary_key == 'user_id'
    assert len(schema.relationships) == 1
    assert outcome == 'success'


def _anchored(anchor: pd.Timestamp) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
        request = PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1], anchor_time=anchor
        )
    return json.loads(json.dumps(request._payloads[0], default=str))


def test_a_zoned_anchor_time_is_read_at_its_utc_instant() -> None:
    zoned = pd.Timestamp('2024-03-01 05:00', tz='America/New_York')
    core = _anchored(zoned)
    assert core == _anchored(pd.Timestamp('2024-03-01 10:00'))
    assert codec.timestamp(1, 'ns', 'UTC')['zone'] == 'UTC'


OUT_OF_RANGE = {
    'date 0001': "CREATE OR REPLACE TABLE orders AS SELECT order_id, user_id, price, DATE '0001-01-01' + (order_id * 7)::INTEGER AS ts FROM orders",
    'date 3000': "CREATE OR REPLACE TABLE orders AS SELECT order_id, user_id, price, DATE '3000-01-01' + (order_id * 7)::INTEGER AS ts FROM orders",
    'timestamptz 3000': "CREATE OR REPLACE TABLE orders AS SELECT order_id, user_id, price, CASE WHEN order_id = 3 THEN TIMESTAMPTZ '3000-01-01 00:00:00+00' ELSE ts::TIMESTAMPTZ END AS ts FROM orders",
    'text 0001': "CREATE OR REPLACE TABLE orders AS SELECT order_id, user_id, price, strftime(DATE '0001-01-01' + (order_id * 7)::INTEGER, '%Y-%m-%d') AS ts FROM orders",
}


@pytest.mark.parametrize('kind', sorted(OUT_OF_RANGE))
def test_a_time_past_nanoseconds_is_refused_on_both_routes_under_its_code(
    kind: str,
) -> None:
    setup = OUT_OF_RANGE[kind]
    with pytest.raises(StructuredDataError) as core:
        _materialize(setup)
    assert core.value.code == 'VALUE_OUT_OF_RANGE'


def _refusal_of(**options: Any) -> StructuredDataError:
    with pytest.raises(StructuredDataError) as refused:
        _materialize(**options)
    return refused.value


def test_a_limit_is_raised_as_the_subclass_on_the_core_register_p1b_20() -> (
    None
):
    for options in ({'num_context_samples': 10001},):
        core = _refusal_of(**options)
        assert core.code == 'UNSUPPORTED_REQUEST', options
        assert type(core).__name__ == 'UnsupportedRequestError', options
        assert 'rule' not in core.details, options
    for options, rule in (
        ({'indices': list(range(100001))}, 'most_entities'),
        ({'indices': list(range(101)), 'rows_per_call': 1}, 'most_calls'),
    ):
        core = _refusal_of(**options)
        assert (type(core), core.code) == (
            StructuredDataError,
            'UNSUPPORTED_REQUEST',
        ), options
        assert core.details.get('rule') == rule, options
    core = _refusal_of(indices=list(range(1001)))
    assert (type(core), core.code) == (StructuredDataError, 'INVALID_REQUEST')


def test_an_average_compared_with_null_is_refused_by_name_register_p1b_21() -> (
    None
):
    query = (
        'PREDICT AVG(orders.price, 0, 30, days) != NULL FOR EACH users.user_id'
    )
    core = _refusal_of(query=query)
    assert (core.code, core.details['rule']) == (
        'UNSUPPORTED_REQUEST',
        'average_null',
    )


def test_a_time_of_day_as_the_time_column_is_the_source_s_failure_register_dt_18() -> (
    None
):

    def refused() -> StructuredDataError:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            connection = _connection(
                'CREATE OR REPLACE TABLE users AS SELECT *, make_time(user_id::BIGINT, 30, 0) AS tod FROM users'
            )
            schema = Schema.from_source(DuckDBSource(connection))
            schema['users'].time_column = 'tod'
            with pytest.raises(StructuredDataError) as error:
                PredictiveQuery(
                    'PREDICT users.age > 30 FOR EACH users.user_id', schema
                ).materialize(indices=[0, 1])
        return error.value

    assert refused().code == 'PROVIDER_ERROR'


@pytest.mark.parametrize(
    'first',
    [{'indices': [None, 0]}, {'indices': [0, 1], 'random_seed': 10**40}],
)
def test_a_refused_call_does_not_lose_the_index_warning(
    first: dict[str, Any],
) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(DuckDBSource(_connection()))
    query = PredictiveQuery(QUERY, schema)
    shown = []
    for index, arguments in enumerate(
        [first, {'indices': [0, 1]}, {'indices': [0, 1]}]
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            if index == 0:
                with pytest.raises(StructuredDataError) as refused:
                    query.materialize(anchor_time=ANCHOR, **arguments)
                assert refused.value.code == 'INVALID_REQUEST'
            else:
                request = query.materialize(anchor_time=ANCHOR, **arguments)
                assert request.indices == [0, 1]
                assert len(request._payloads) == 1
        shown.append(sum('Missing' in str(w.message) for w in caught))
    assert shown == [1, 0, 0]


@pytest.mark.parametrize(
    'count',
    [np.int64(4), np.uint64(4), np.int8(4)],
    ids=['int64', 'uint64', 'int8'],
)
def test_a_numpy_context_size_reaches_the_core(count: Any) -> None:
    core = _materialize(num_context_samples=count)
    assert core._payloads[0]['metadata']['num_context_examples'] == 4
    assert (
        core._payloads == _materialize(num_context_samples=int(count))._payloads
    )


@pytest.mark.parametrize(
    ('query', 'config'),
    [
        (QUERY, {'num_estimators': np.float32(2)}),
        (
            QUERY,
            {'num_estimators': np.int64(3), 'column_shuffle': np.bool_(True)},
        ),
        (
            QUERY,
            {
                'hop_shuffle': np.bool_(False),
                'category_shuffle': np.float64(1.0),
            },
        ),
        (BINARY, {'class_shuffle': np.bool_(False)}),
    ],
    ids=['float32', 'int64-bool', 'bool-float64', 'binary-class-shuffle'],
)
def test_numpy_inference_options_normalize_to_builtin_scalars(
    query: str, config: dict[str, Any]
) -> None:
    request = _materialize(query=query, inference_config=config)
    plain = {key: value.item() for key, value in config.items()}
    assert (
        request._payloads
        == _materialize(query=query, inference_config=plain)._payloads
    )
    expected = {
        key: int(value) if key == 'num_estimators' else bool(value)
        for key, value in plain.items()
    }
    received = request._payloads[0]['inference']['inference_config']
    for key, value in expected.items():
        assert received[key] == value
        assert type(received[key]) is type(value)


def _notices(*setup: str, text: str | None = None) -> list[str]:
    """The client's notices (``SDMWarning``) a materialize of ``QUERY``
    over the shop base and ``setup`` gives, each without the core's code;
    ``text`` names a column of ``users`` typed text."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        connection = _connection(*setup)
        schema = Schema.from_source(DuckDBSource(connection))
        if text is not None:
            schema['users'][text].stype = 'text'
        PredictiveQuery(QUERY, schema).materialize(
            indices=[0, 1], anchor_time=ANCHOR
        )
    return sorted(
        re.sub('^\\[[A-Z_]+\\] ', '', str(w.message))
        for w in caught
        if type(w.message).__name__ == 'SDMWarning'
    )


@pytest.mark.parametrize(
    'setup',
    [
        (
            'CREATE TABLE notes (note_id BIGINT PRIMARY KEY, body VARCHAR)',
            "INSERT INTO notes VALUES (1, 'a')",
        ),
        (
            "CREATE TABLE tags AS SELECT i::BIGINT AS tag_id, (100 + i)::BIGINT AS user_id, 'x' AS tag FROM range(3) t(i)",
        ),
        (
            'CREATE TABLE notes (note_id BIGINT PRIMARY KEY, body VARCHAR)',
            "INSERT INTO notes VALUES (1, 'a')",
            'CREATE TABLE more AS SELECT 1::BIGINT AS more_id, 2.0 AS x',
            "CREATE OR REPLACE TABLE users AS SELECT *, repeat('a long piece of free text that says something different each time ', 4) || user_id AS bio FROM users",
        ),
    ],
    ids=['unlinked', 'unreached', 'several-and-text'],
)
def test_the_notices_of_what_the_request_leaves_out_are_the_old_route_s(
    setup: tuple[str, ...],
) -> None:
    text = 'bio' if any(' AS bio ' in s for s in setup) else None
    core = _notices(*setup, text=text)
    assert core, 'a notice was given'
    if text is not None:
        assert any('users.bio is typed text' in n for n in core), core
    if any('CREATE TABLE tags' in statement for statement in setup):
        matching = [
            notice for notice in core if 'sampler reached no row' in notice
        ]
        assert len(matching) == 1
        assert 'tags' in matching[0]
    else:
        matching = [
            notice for notice in core if 'no relationship connects' in notice
        ]
        assert len(matching) == 1
        assert 'notes' in matching[0]
        assert ('more' in matching[0]) == (text is not None)


def test_the_default_mode_s_cap_is_noticed_as_on_the_old_route() -> None:
    statements = [
        'CREATE TABLE users AS SELECT i::BIGINT AS user_id, (20 + i % 50)::INTEGER AS age FROM range(1001) t(i)',
        "CREATE TABLE orders AS SELECT i::BIGINT AS order_id, (i % 1001)::BIGINT AS user_id, (10 + (i % 97) * 1.5)::DOUBLE AS price, TIMESTAMP '2023-06-01' + to_days((i % 270)::INTEGER) AS ts FROM range(900) t(i)",
    ]

    def notices(**options: Any) -> list[str]:
        connection = duckdb.connect()
        for statement in statements:
            connection.execute(statement)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            schema = Schema.from_source(DuckDBSource(connection))
            PredictiveQuery(QUERY, schema).materialize(
                indices=[0, 1], anchor_time=ANCHOR, **options
            )
        return [
            re.sub('^\\[[A-Z_]+\\] ', '', str(w.message))
            for w in caught
            if type(w.message).__name__ == 'SDMWarning'
        ]

    core = notices()
    assert any('context examples were gathered' in n for n in core), core
    assert not any(
        'context examples were gathered' in n
        for n in notices(num_context_samples=1000)
    )
