# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""The compatibility manifest's context scenarios, as a caller's own source
is tested with them: the fixture's tables loaded into an engine, a query
materialized over the source, and the examples it keeps compared with the
ones the fixture holds for it.

Each scenario runs twice: with the source declaring every value filter
kind, and declaring only those of an engine without ``OR`` or subqueries.
Either way each query keeps exactly the examples the fixture holds for it:
the core checks every row by the query's own rules, and a filter the
source does not declare is never sent to it.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Relationship, Schema, Source
from sdm_client import provider as p

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / 'tests/compatibility/fixtures/context.json'
TYPES = {
    'users': 'id BIGINT, value DOUBLE, class VARCHAR, flag BOOLEAN',
    'events': 'event BIGINT, id BIGINT, feature DOUBLE, event_time TIMESTAMP(6), item BIGINT',
    'items': 'item BIGINT, price DOUBLE',
}
NO_OR_NO_SUBQUERY = frozenset(
    {
        p.FilterKind.COMPARE,
        p.FilterKind.IN,
        p.FilterKind.IS_NULL,
        p.FilterKind.ALL_OF,
    }
)


def scenarios() -> list[dict[str, Any]]:
    manifest = json.loads(
        (ROOT / 'tests/compatibility/manifest.json').read_text()
    )
    return [s for s in manifest['scenarios'] if s['id'].startswith('context.')]


def insert_sql(fixture: dict[str, Any], table: str, name: str) -> Iterator[str]:
    rows = fixture['tables'][table]['rows']
    for start in range(0, len(rows), 500):
        values = ', '.join(
            '(' + ', '.join(fixture_literal(v) for v in row) + ')'
            for row in rows[start : start + 500]
        )
        yield f'INSERT INTO {name} VALUES {values}'


def fixture_literal(value: Any) -> str:
    if (
        isinstance(value, str)
        and len(value) == 19
        and value[4] == '-'
        and value[10] == 'T'
    ):
        return f"TIMESTAMP '{value.replace('T', ' ')}'"
    if isinstance(value, bool):
        return 'TRUE' if value else 'FALSE'
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    return repr(value)


def materialize(provider: Source, scenario: dict[str, Any]) -> Any:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        schema = Schema.from_source(
            provider,
            # The engine declares no keys: the caller names them.
            primary_keys={'users': 'id', 'events': 'event', 'items': 'item'},
            time_columns={'events': 'event_time'},
            relationships=[
                Relationship('events', 'id', 'users', 'id'),
                Relationship('events', 'item', 'items', 'item'),
            ],
        )
        schema['users']['value'].stype = 'numerical'
        options = dict(scenario['options'])
        if 'anchor_time' in options:
            options['anchor_time'] = pd.Timestamp(options['anchor_time'])
        return PredictiveQuery(scenario['pql'], schema).materialize(**options)


def expected(scenario: dict[str, Any], declared: str) -> list[tuple[Any, ...]]:
    # A filter reaches the provider only where it declares the kinds the
    # scenario's filter is written with (``sent_with``).
    context = scenario['context']
    declared_kinds = {kind.value for kind in kinds(declared)}
    sent = (
        context['filtered_at_source']
        and set(context['sent_with']) <= declared_kinds
    )
    return [
        (kept, context['asked'], context['stopped'], sent)
        for kept in context['kept']
    ]


def drawn(request: Any) -> list[tuple[Any, ...]]:
    report = request.context_report
    assert report.status == 'available'
    return [
        (d.kept, d.asked, d.stopped, d.filtered_at_source) for d in report.parts
    ]


CASES = [
    pytest.param(s, declared, id=f'{s["id"]} ({declared})')
    for s in scenarios()
    for declared in ('every kind', 'no OR or subquery')
]


def kinds(declared: str) -> frozenset[p.FilterKind]:
    return (
        frozenset(p.FilterKind)
        if declared == 'every kind'
        else NO_OR_NO_SUBQUERY
    )
