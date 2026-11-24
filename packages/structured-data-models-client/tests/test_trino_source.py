# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The Trino example of your own source (``trino_source.py``), against a
Trino server where ``SDM_TRINO_TEST_URL`` names one, such as
``docker run -p 8080:8080 trinodb/trino``; CI starts one. Each test makes
a schema of its own in Trino's memory catalog and drops it after.

As Java's ``TrinoProviderTest``: the source builds a schema, materializes
the compatibility manifest's context scenarios keeping exactly the examples
the fixture holds, splits a large key set into lookups of its key limit,
and tells the client what Trino's errors mean.
"""

from __future__ import annotations

import dataclasses
import json
import os
import uuid
import warnings
from collections.abc import Iterator
from typing import Any

import pyarrow as pa
import pytest
from context_scenarios import (
    CASES,
    FIXTURE,
    TYPES,
    drawn,
    expected,
    insert_sql,
    kinds,
    materialize,
    scenarios,
)
from trino_source import TrinoError, TrinoSource

from sdm_client import (
    PredictiveQuery,
    Relationship,
    Schema,
    StructuredDataError,
)
from sdm_client import provider as p

pytestmark = pytest.mark.skipif(
    not os.environ.get('SDM_TRINO_TEST_URL'),
    reason='SDM_TRINO_TEST_URL names no Trino server',
)

CATALOG = 'memory'


class _Recording(TrinoSource):
    r"""The example source, keeping each request it answers, and the seeds
    of each lookup."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.asked: list[p.DataRequest] = []
        self.lookup_keys: list[int] = []

    def answer(
        self, request: p.DataRequest, expected: pa.Schema | None
    ) -> pa.Table:
        self.asked.append(request)
        if isinstance(request, p.Lookup):
            # A key set is read once: read it here, and hand on its rows.
            seeds = request.seeds.reader.read_all()
            self.lookup_keys.append(seeds.num_rows)
            request = dataclasses.replace(
                request, seeds=p.KeySet(request.seeds.schema, seeds.to_reader())
            )
        return super().answer(request, expected)


@pytest.fixture(scope='module')
def url() -> str:
    return os.environ['SDM_TRINO_TEST_URL']


@pytest.fixture(scope='module')
def shop(url: str) -> Iterator[str]:
    r"""A Trino schema holding the context fixture's tables, by name."""
    schema = 'sdm_' + uuid.uuid4().hex
    trino = TrinoSource(url, CATALOG, schema)
    trino._run(f'CREATE SCHEMA {CATALOG}.{schema}')
    fixture = json.loads(FIXTURE.read_text())
    try:
        for table, columns in TYPES.items():
            name = f'{CATALOG}.{schema}."{table}"'
            trino._run(f'CREATE TABLE {name} ({columns})')
            for statement in insert_sql(fixture, table, name):
                trino._run(statement)
        yield schema
    finally:
        for table in [*TYPES, 'gone']:
            trino._run(f'DROP TABLE IF EXISTS {CATALOG}.{schema}."{table}"')
        trino._run(f'DROP SCHEMA IF EXISTS {CATALOG}.{schema}')


def test_the_schema_is_described_from_trino_s_catalog(
    url: str, shop: str
) -> None:
    source = _Recording(url, CATALOG, shop)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(
            source,
            # Trino declares no keys: the caller names them.
            primary_keys={'users': 'id', 'events': 'event', 'items': 'item'},
            time_columns={'events': 'event_time'},
            relationships=[Relationship('events', 'id', 'users', 'id')],
        )
    assert sorted(schema.tables) == ['events', 'items', 'users']
    events = schema['events']
    assert events.primary_key == 'event'
    assert events.time_column == 'event_time'
    assert list(events) == [
        'event',
        'id',
        'feature',
        'event_time',
        'item',
    ]
    assert schema.source is source
    parts = {r.part for r in source.asked if isinstance(r, p.Describe)}
    assert {p.DescribePart.TABLES, p.DescribePart.COLUMNS} <= parts


@pytest.mark.parametrize(('scenario', 'declared'), CASES)
def test_each_query_keeps_exactly_the_examples_the_source_holds(
    url: str, shop: str, scenario: dict[str, object], declared: str
) -> None:
    source = TrinoSource(url, CATALOG, shop, filters=kinds(declared))
    assert drawn(materialize(source, scenario)) == expected(scenario, declared)


def test_a_key_set_past_the_key_limit_reaches_the_source_in_lookups_of_it(
    url: str, shop: str
) -> None:
    scenario = next(
        s for s in scenarios() if s['id'] == 'context.static-exhausted'
    )
    parts = _Recording(url, CATALOG, shop, max_keys_per_request=2)
    whole = _Recording(url, CATALOG, shop)
    split, single = materialize(parts, scenario), materialize(whole, scenario)
    assert parts.lookup_keys, 'the source declares lookup, so seeds reach it'
    assert max(parts.lookup_keys) <= 2
    assert len(parts.lookup_keys) > len(whole.lookup_keys)
    # Answered in parts, a lookup is answered as one.
    assert split.payloads == single.payloads


def test_a_count_and_a_preview_answer_as_a_scan_does(
    url: str, shop: str
) -> None:
    source = TrinoSource(url, CATALOG, shop)
    users = p.Relation(('users',))
    over = (p.Compare('value', p.CompareOp.GT, p.Value(40.0, pa.float64())),)
    scanned = source.answer(p.Scan(users, ('id',), over, None), None)
    counted = source.answer(p.CountRows(users, over), None)
    assert counted.column(0).to_pylist() == [scanned.num_rows]
    preview = source.answer(p.Preview(users, ('id', 'class'), 3), None)
    assert preview.column_names == ['id', 'class'] and preview.num_rows == 3


def test_trino_s_errors_tell_the_client_what_they_mean(
    url: str, shop: str
) -> None:
    source = TrinoSource(url, CATALOG, shop)
    scan = p.Scan(p.Relation(('gone',)), ('id',), (), None)
    with pytest.raises(TrinoError) as raised:
        source.answer(scan, None)
    told = source.failure(raised.value)
    assert told is not None
    assert told.error_class == 'TABLE_NOT_FOUND'
    assert told.facts == p.FailureFacts(
        not_found=True,
        unusable=p.Unusable.MISSING,
        reason=p.FailureReason.TABLE_UNRESOLVED,
    )
    # Any other failure is read by its message alone.
    assert source.failure(RuntimeError('down')) is None


def test_a_table_dropped_after_the_schema_fails_the_operation_by_name(
    url: str, shop: str
) -> None:
    source = TrinoSource(url, CATALOG, shop)
    source._run(
        f'CREATE TABLE {CATALOG}.{shop}."gone" AS SELECT * FROM {CATALOG}.{shop}."users"'
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(
            source, {'gone': 'gone'}, primary_keys={'gone': 'id'}
        )
    source._run(f'DROP TABLE {CATALOG}.{shop}."gone"')
    with pytest.raises(StructuredDataError) as raised:
        PredictiveQuery(
            'PREDICT gone.value FOR EACH gone.id', schema
        ).materialize(indices=[1])
    assert raised.value.code == 'PROVIDER_ERROR', raised.value
