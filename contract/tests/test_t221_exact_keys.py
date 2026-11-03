# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact numeric keys survive catalog, supplied-key and edited-relationship paths."""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import sys
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]

RUNNER = ROOT / 'contract/runners/run_python.py'

CASES = ROOT / 'contract/cases/s1'

VECTORS = ROOT / 'contract/exact-key-vectors.json'

DATABASES = ('duckdb', 'sqlite', 'postgres')

FIRST = 2**53

GIVEN_KEY = 'key.past-2-53.given-key'

GIVEN_LINK = 'key.past-2-53.given-link'

EDITED_LINK = 'key.past-2-53.edited-link'

CATALOG = 'key.past-2-53.catalog'

CASE_IDS = (GIVEN_KEY, GIVEN_LINK, EDITED_LINK, CATALOG)

LINK = {
    'source_table': 'orders',
    'source_column': 'user_id',
    'target_table': 'users',
    'target_column': 'user_id',
}


def _module(name: str, path: Path) -> ModuleType:
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def runner() -> ModuleType:
    return _module('t221_run_python', RUNNER)


R = _module('exact_keys_record', ROOT / 'contract/tools/exact_keys_record.py')

GIVEN_VECTORS = tuple(R.CONTROL)


def _server() -> bool:
    r"""Whether a Postgres server is named; where none is, one must not be
    required (H26)."""
    if os.environ.get('SDM_POSTGRES_TEST_DSN'):
        return True
    assert not os.environ.get('SDM_REQUIRE_POSTGRES'), (
        'SDM_REQUIRE_POSTGRES is set but SDM_POSTGRES_TEST_DSN is not'
    )
    pytest.skip('SDM_POSTGRES_TEST_DSN is not configured')


RECORDED: dict[str, Any] = json.loads(VECTORS.read_text(encoding='utf-8'))

BY_ID = {vector['id']: vector for vector in RECORDED['vectors']}


def _projection(answer: dict[str, Any]) -> dict[str, Any]:
    r"""An answer as a given key is held to its control's: its outcome,
    public schema and request."""
    return {k: answer[k] for k in ('outcome', 'schema', 'payloads')}


def _differs(got: dict[str, Any], want: dict[str, Any]) -> list[str]:
    return sorted(k for k in want if got.get(k) != want[k])


@pytest.mark.parametrize('vector_id', sorted(set(R.CONTROL.values())))
def test_python_core_answers_a_catalog_key_as_the_reference(
    vector_id: str,
) -> None:
    r"""Exit "as with a catalog key", the controls: python-core answers the
    tables with the key and link declared as python-ref does (outcome,
    public schema, request, each column's type), the key its text."""
    if not _server():
        return

    def check() -> None:
        want = BY_ID[vector_id]['reference']
        got = R.observe(BY_ID[vector_id], os.environ['SDM_POSTGRES_TEST_DSN'])
        found = _differs(got, want)
        assert not found, (
            f'{vector_id}: {found} differ: got {got["outcome"]}, the '
            f'reference {want["outcome"]}'
        )

    check()


@pytest.mark.parametrize('vector_id', GIVEN_VECTORS)
def test_python_core_answers_a_given_or_edited_key_as_a_catalog_key(
    vector_id: str,
) -> None:
    r"""Exit "the reproduction and its supplied-foreign-key and
    edited-relationship variants work on Postgres as with a catalog key":
    python-core answers the key given (``pk.given``), the keys and link
    given (``fk.given``) or the link set by an edit (``fk.edited``), nothing
    declared, exactly as python-ref answers its control with the key and
    link declared: the outcome, the public schema (each ``user_id`` its
    text) and the request."""
    if not _server():
        return

    def check() -> None:
        control = R.CONTROL[vector_id]
        want = _projection(BY_ID[control]['reference'])
        got = _projection(
            R.observe(BY_ID[vector_id], os.environ['SDM_POSTGRES_TEST_DSN'])
        )
        found = _differs(got, want)
        assert not found, (
            f'{vector_id}: {found} differ from python-ref on {control}: got '
            f'{got["outcome"]}, schema {got["schema"]}'
        )

    check()


@functools.cache
def cases() -> dict[str, tuple[dict[str, Any], str]]:
    r"""The T2.21 cases by id, each with its digest, as the runner loads
    them."""
    loaded = runner().load_cases([CASES / f'{c}.json' for c in CASE_IDS])
    return {case['id']: (case, digest) for case, digest in loaded}


def _built(case: dict[str, Any]) -> Any:
    r"""The case's DuckDB database, its base and setup run, as the runner
    builds it."""
    import duckdb

    connection = duckdb.connect()
    for statement in [
        *runner().load_bases()[case['input']['base']],
        *case['input'].get('setup', []),
    ]:
        connection.execute(statement)
    return connection


@pytest.mark.parametrize('case_id', CASE_IDS)
def test_each_case_holds_an_exact_key_past_2_53(case_id: str) -> None:
    r"""Exit "cases in the corpus on all three databases": each case runs
    through predict on DuckDB, SQLite and Postgres over ``user_id`` a
    ``DECIMAL(38, 0)`` in both tables, every id past 2**53, two of them
    one double (so a key read as doubles merges them), asked for 2**53 + 1
    and 2**53 + 2; the key and link declared only in ``catalog``, given in
    ``given-key`` (the key) and ``given-link`` (the keys and link), and the
    link set by an edit in ``edited-link`` (given no link)."""
    case, _ = cases()[case_id]
    given = case['input']
    assert case['status'] == 'acceptance' and case['slice'] == 'S1'
    assert case['until'] == 'predict'
    assert runner().case_databases(case) == DATABASES
    assert given['pql'] == (
        'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
    )
    assert given['materialize']['indices'] == [FIRST + 1, FIRST + 2]
    connection = _built(case)
    try:
        types = dict(
            connection.execute(
                'SELECT table_name, data_type FROM duckdb_columns() '
                "WHERE column_name = 'user_id'"
            ).fetchall()
        )
        ids = [
            Decimal(row[0])
            for row in connection.execute(
                'SELECT user_id FROM users UNION SELECT user_id FROM orders'
            ).fetchall()
        ]
        constraints = sorted(
            row[0]
            for row in connection.execute(
                'SELECT constraint_type FROM duckdb_constraints() '
                "WHERE constraint_type IN ('PRIMARY KEY', 'FOREIGN KEY')"
            ).fetchall()
        )
    finally:
        connection.close()
    assert types == {'users': 'DECIMAL(38,0)', 'orders': 'DECIMAL(38,0)'}
    assert min(ids) >= FIRST
    assert len({float(i) for i in ids}) < len(set(ids))
    if case_id == CATALOG:
        assert 'from_source' not in given and 'edits' not in given
        assert constraints == ['FOREIGN KEY', 'PRIMARY KEY', 'PRIMARY KEY']
        return
    assert constraints == []
    keys = {'users': 'user_id', 'orders': 'order_id'}
    expected = {
        GIVEN_KEY: ({'primary_keys': {'users': 'user_id'}}, None),
        GIVEN_LINK: ({'primary_keys': keys, 'relationships': [LINK]}, None),
        EDITED_LINK: (
            {'primary_keys': keys, 'relationships': []},
            [{'op': 'set_relationships', 'relationships': [LINK]}],
        ),
    }[case_id]
    assert (given['from_source'], given.get('edits')) == expected
