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
    assert module.arrow_table is shared.arrow_table
    host: dict[str, Any] = {'plan': {'plan': 'host', 'expect': {}}}
    with pytest.raises(shared.PlanNotRun, match='host plan'):
        provider(None)(host, {})
    other = {
        'plan': {
            'plan': 'sql',
            'statements': [{'dialect': 'elsewhere', 'text': '', 'params': []}],
            'expect': {},
        }
    }
    with pytest.raises(shared.PlanNotRun, match='another dialect'):
        provider(None)(other, {})


def test_a_key_set_not_handed_over_is_refused_by_name() -> None:
    param = {'param': 'key_set', 'keys': {'attachment': 3}, 'relation': 'k'}
    with pytest.raises(shared.PlanNotRun, match='not handed over'):
        shared.key_set(param, {})
    assert shared.key_set(param, {3: object()}) == 3
    with pytest.raises(shared.PlanNotRun, match='scalar'):
        shared.key_set({'param': 'value', 'value': {}}, {})
