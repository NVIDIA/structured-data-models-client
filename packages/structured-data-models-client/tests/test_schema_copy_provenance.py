# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schema copies preserve source facts and refresh metadata when sampled."""

from __future__ import annotations

import copy
import dataclasses
import io
import json
import pickle
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema, StructuredDataError

QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'


ANCHOR = pd.Timestamp('2024-03-01')


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


def _watched(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    r"""The kind of each request a source is asked from now on."""
    from sdm_client._rust import providers

    asked: list[str] = []
    real = providers.for_source

    def watched(given):
        choice, provider = real(given)

        def answering(request, keys):
            asked.append(request['request']['request'])
            return provider(request, keys)

        return choice, answering

    monkeypatch.setattr(providers, 'for_source', watched)
    return asked


@pytest.mark.parametrize('made', sorted(_copies()))
def test_copies_refresh_the_same_source_metadata(made, monkeypatch):
    source = _shop()
    asked = _watched(monkeypatch)
    described = _described(source)
    asked.clear()
    copied = _copies()[made](described)
    assert asked == []
    once = _payload(described)
    first = list(asked)
    asked.clear()
    again = _payload(copied)
    assert isinstance(once, list)
    assert again == once
    assert asked == first
    assert 'describe' in asked and 'preview' not in asked


@pytest.mark.parametrize(
    'change',
    [
        'ALTER TABLE users ALTER age SET NOT NULL',
        'ALTER TABLE users ALTER age TYPE DOUBLE',
    ],
)
def test_a_column_declared_otherwise_since_is_read_again_in_full(
    change, monkeypatch
):
    r"""A run reads a table's catalog alone while it declares every column
    as describe read it, and the table in full once one is declared
    otherwise, its nullability as much as its type."""
    source = _shop()
    asked = _watched(monkeypatch)
    schema = _described(source)
    asked.clear()
    _payload(schema)
    assert 'describe' in asked and 'preview' not in asked
    source.connection.execute(change)
    asked.clear()
    _payload(schema)
    assert 'preview' in asked


def test_source_facts_are_immutable_and_excluded_from_column_equality():
    from sdm_client._rust import results

    schema = _described(_shop())
    without_schema_facts = schema.snapshot()
    for table in without_schema_facts.tables.values():
        for column in table.columns.values():
            object.__setattr__(column, '_source', None)
    assert repr(schema) == repr(without_schema_facts)
    for table in schema.tables.values():
        for column in table.columns.values():
            facts = column._source
            assert isinstance(facts, results.SourceFacts)
            with pytest.raises(dataclasses.FrozenInstanceError):
                facts.nullable = not facts.nullable
            without_facts = copy.copy(column)
            object.__setattr__(without_facts, '_source', None)
            assert column == without_facts
            assert repr(column) == repr(without_facts)
    assert schema['users']['user_id']._source.unique
    assert not schema['users']['user_id']._source.nullable
    assert schema['orders']['ts']._source.dtype == 'date'
    for copied in _copies().values():
        held = copied(schema)
        for name, table in schema.tables.items():
            for key, column in table.columns.items():
                assert held[name][key]._source == column._source
    assert (
        copy.copy(schema['users']['age'])._source
        == schema['users']['age']._source
    )


def test_materialized_request_keeps_immutable_typed_state():
    from sdm_client._rust import results

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(QUERY, _described(_shop())).materialize(
            indices=[0, 1], anchor_time=ANCHOR
        )
    assert isinstance(request._core, results.CoreRequest)
    assert request._core.version == 2
    with pytest.raises(dataclasses.FrozenInstanceError):
        request._core.version = 3
    assert not hasattr(request, '_core_state')
