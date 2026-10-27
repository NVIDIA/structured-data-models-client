# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The rules every SQL provider applies to a plan before running it are
written once, in ``sdm_connectors.providers`` (design fix F; T3.4 fix 1,
round 1's Codex P1-7 and Claude P2-3): each driver's provider refuses a
plan with the one ``PlanNotRun`` and reads a key set with the one reader.
"""

from __future__ import annotations

from typing import Any

import pytest
from sdm_connectors import providers as shared

from sdm_client._rust.providers import databricks, duckdb, postgres, sqlite

MODULES = {
    'duckdb': (duckdb, duckdb.DuckDBProvider),
    'sqlite': (sqlite, sqlite.SqliteProvider),
    'postgres': (postgres, postgres.PostgresProvider),
    'databricks': (databricks, databricks.DatabricksSqlProvider),
}


@pytest.mark.parametrize('dialect', sorted(MODULES))
def test_each_provider_refuses_with_the_one_shared_rule(dialect: str) -> None:
    module, provider = MODULES[dialect]
    assert module.PlanNotRun is shared.PlanNotRun
    # A key set's Arrow is read with the one reader; a provider binding the
    # text the core wrote of one (T3.28) reads it with the one text reader.
    assert getattr(module, 'arrow_table', shared.arrow_table) is (
        shared.arrow_table
    )
    host: dict[str, Any] = {'plan': {'plan': 'host', 'expect': {}}}
    if dialect == 'postgres':
        host['plan']['result_conversions'] = [{'conversion': 'zoned_text_utc'}]
    with pytest.raises(shared.PlanNotRun, match='host plan'):
        provider(None)(host, {})
    other = {
        'plan': {
            'plan': 'sql',
            'statements': [{'dialect': 'elsewhere', 'text': '', 'params': []}],
            'expect': {},
        }
    }
    if dialect == 'postgres':
        other['plan']['result_conversions'] = [{'conversion': 'zoned_text_utc'}]
    with pytest.raises(shared.PlanNotRun, match='another dialect'):
        provider(None)(other, {})


@pytest.mark.parametrize(
    'conversions', [None, [], [{'conversion': 'timestamp_micros'}]]
)
def test_postgres_refuses_an_unmarked_plan_before_shared_plan_rules(
    conversions: Any,
) -> None:
    plan = {'plan': 'host', 'expect': {}}
    if conversions is not None:
        plan['result_conversions'] = conversions
    with pytest.raises(shared.PlanNotRun, match='exact UTC text policy'):
        postgres.PostgresProvider(None)({'plan': plan}, {})


def test_a_key_set_not_handed_over_is_refused_by_name() -> None:
    param = {'param': 'key_set', 'keys': {'attachment': 3}, 'relation': 'k'}
    with pytest.raises(shared.PlanNotRun, match='not handed over'):
        shared.key_set(param, {})
    handed = object()
    assert shared.key_set(param, {3: handed}) == (3, handed)
    with pytest.raises(shared.PlanNotRun, match='scalar'):
        shared.key_set({'param': 'value', 'value': {}}, {})


def test_a_key_set_name_binds_one_key_set_however_often_it_is_named() -> None:
    r"""The relation each key set is bound under, one rule for every
    provider binding key sets by name (round 2's Codex P1-3)."""

    def param(attachment: int) -> dict[str, Any]:
        return {
            'param': 'key_set',
            'keys': {'attachment': attachment},
            'relation': 'keys',
        }

    keys = {1: object(), 2: object()}
    same = [{'params': [param(1)]}, {'params': [param(1)]}]
    assert shared.bound_relations(same, keys) == {'keys': 1}
    two = [{'params': [param(1)]}, {'params': [param(2)]}]
    with pytest.raises(shared.PlanNotRun, match='two key sets to one name'):
        shared.bound_relations(two, keys)


def test_a_key_set_s_text_is_read_by_the_one_shared_reader() -> None:
    r"""A key set the dialect writes as text (T3.28) is handed over as a
    one-cell text attachment; every provider binding one reads it here."""
    import pyarrow as pa

    param = {'param': 'text', 'attachment': 4}
    with pytest.raises(shared.PlanNotRun, match='not handed over'):
        shared.text(param, {})
    cell = pa.table({'keys': pa.array(['[1,2]'], pa.string())})
    assert shared.text(param, {4: cell}) == (4, '[1,2]')
    for bad in (
        pa.table({'keys': pa.array(['[1]', '[2]'], pa.string())}),
        pa.table({'keys': pa.array([None], pa.string())}),
        pa.table({'keys': pa.array([1], pa.int64())}),
        pa.table({'a': ['[1]'], 'b': ['[2]']}),
    ):
        with pytest.raises(shared.PlanNotRun, match='not one text'):
            shared.text(param, {4: bad})
    with pytest.raises(shared.PlanNotRun, match='not a text'):
        shared.text({'param': 'value', 'value': {}}, {})
