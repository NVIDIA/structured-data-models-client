# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The core selects UTC text policy; PostgreSQL hosts require its instruction."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sdm_connectors import PostgresSource
from sdm_connectors.providers import PlanNotRun

from sdm_client import Schema, StructuredDataError, _routing
from sdm_client._rust import providers
from sdm_client._rust.providers.postgres import PostgresProvider

UTC_TEXT = [{'conversion': 'zoned_text_utc'}]


def test_public_postgres_schema_fetches_carry_the_core_utc_text_instruction(
    monkeypatch,
):
    seen = []

    class Spy:
        def __init__(self, connection):
            pass

        def __call__(self, step, keys=None):
            seen.append(step['plan'])
            raise RuntimeError('stop before database I/O')

    connection = type('Connection', (), {'__module__': 'psycopg'})()
    monkeypatch.delenv(_routing.ENVIRONMENT, raising=False)
    monkeypatch.setattr(providers, 'PostgresProvider', Spy)
    with pytest.raises(StructuredDataError):
        Schema.from_source(PostgresSource(connection, schema='public'))
    assert seen
    assert all(plan.get('result_conversions') == UTC_TEXT for plan in seen)


@pytest.mark.parametrize(
    'conversions',
    [
        None,
        [],
        [{'conversion': 'unknown'}],
        [{'conversion': 'zoned_text_utc', 'column': 0}],
        UTC_TEXT * 2,
    ],
)
def test_unmarked_or_invalid_postgres_policy_is_refused_before_io(conversions):
    def io():
        raise AssertionError('invalid policy reached database I/O')

    plan = {
        'plan': 'sql',
        'statements': [
            {'dialect': 'postgres', 'text': 'SELECT 1', 'params': []}
        ],
    }
    if conversions is not None:
        plan['result_conversions'] = conversions
    with pytest.raises(PlanNotRun):
        PostgresProvider(SimpleNamespace(transaction=io, cursor=io))(
            {'plan': plan}
        )


def test_utc_marker_does_not_admit_another_dialect():
    plan = {
        'plan': 'sql',
        'statements': [{'dialect': 'sqlite', 'text': 'SELECT 1', 'params': []}],
        'result_conversions': UTC_TEXT,
    }
    with pytest.raises(PlanNotRun):
        PostgresProvider(object())({'plan': plan})
