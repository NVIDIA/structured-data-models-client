# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Query diagnostics stay consistent for ordinary and unusual column names.

The same public scenarios run on DuckDB, SQLite and optional PostgreSQL.
Assertions name the refusal cause and remedy independently of client output.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
DATABASES = ('duckdb', 'sqlite', 'postgres')


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


R = _module(
    'column_name_scenarios',
    ROOT / 'contract/support/column_name_scenarios.py',
)

MESSAGES = {
    'task': (
        "[INVALID_REQUEST] task='regression' settles what a single-table "
        'question is asking, which is the only route that has to decide. A '
        'relational query resolves to its own kind, so state it in the '
        'query rather than beside it'
    ),
    'typo': (
        "[INVALID_REQUEST] '<pql>' is not a query a relational model can "
        "answer: row 1, column 12: Column 'pricee' does not exist in table "
        "'orders'. Please be mindful of case sensitivity and ensure that the "
        'name is spelled correctly.'
    ),
    'parts': (
        '[INVALID_REQUEST] 2 forecast steps means 2 calls, one for each, '
        'because the model has no task that answers them together. That is '
        'past the 1 this will do without being told to; ask about fewer, or '
        'raise sdm_client.decompose.MOST_PARTS if the wait is what you want.'
    ),
}


def reachable(database: str) -> bool:
    if database != 'postgres' or os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    return False


@pytest.mark.parametrize('vector', R.vectors(), ids=lambda vector: vector['id'])
def test_column_names_preserve_actionable_diagnostics(vector) -> None:
    """Real connector calls retain refusal stage, cause and useful remedy."""
    if not reachable(vector['database']):
        pytest.skip('SDM_POSTGRES_TEST_DSN is not configured')
    got = R.run(vector)
    assert (got['result'], got['stage'], got['error'], got['code']) == (
        'refusal',
        'materialize',
        'StructuredDataError',
        'INVALID_REQUEST',
    )
    assert got['message'] == MESSAGES[vector['kind']].replace(
        '<pql>', vector['pql']
    )
