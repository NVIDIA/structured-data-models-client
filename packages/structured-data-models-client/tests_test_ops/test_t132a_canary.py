# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The canary test over the keys family's error paths (T1.32a's exit "the
canary test covers the family's error paths"; decision H17).

Written by the package's independent test author, before the
implementation. It drives each error path of the entities' cast to the
key's type and of the key's own range through the canary of
``test_the_canary.py`` (its sentinels, its watch over the core's error,
``str``, ``repr``, the details, the traceback, every warning, every log
record at DEBUG, stderr and stdout, and its check that none of them holds
a sentinel), on the core route:

- cast: a text entity no integer key holds, alone and among integers;
- cast: an integer past an unsigned key's width, and a negative one (the
  reference's INVALID_REQUEST and INTERNAL_ERROR);
- range: a ``UBIGINT`` key past int64, planted in the data, the request
  refused as the reference refuses it (``materialize/int64_range``);
- lookup: a text entity no row of a text key holds;
- round 2, each naming its reason: a text entity and a fraction for a
  DECIMAL(18, 0) key (not_a_number, inexact), an integer a FLOAT key cannot
  hold exactly and a text entity for a DOUBLE key (inexact,
  not_a_number), and bytes and a timestamp for a BIGINT key (type).

The sentinels are the canary's data sentinel and a number of our own,
``NUMBER``, both planted in the data the paths read: two users' countries
and an order's price (:data:`PLANTED`), the text keys a lookup misses and
the keys past int64. From KO2-86 (the owner's narrowing of H17 and E79) an
error may name a value the caller passed in the same call, as the reference
does (``indices[0] is ...``, the missing keys), so the entities hold no
sentinel; a value read from the data stays out, so no sentinel may be
written anywhere.

Each error must be the core's: raised from the core's error (the canary
watches ``errors.error_from_core``), so a refusal the bridge still makes
itself, in pandas, fails here until the cast runs in the core. Its code is
INVALID_REQUEST or INTERNAL_ERROR, never a rule not taken yet.

Gating (``contract/tests/t132a_gate.py``): a scenario named in
``t132a-pending.json``'s ``pending["exits"]`` (as ``canary:<scenario>``)
failed when written. It still runs and must still fail; once it passes it
fails, naming itself, until it is taken out of the list.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema, _routing

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def _canary() -> Any:
    name = 'test_the_canary'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, HERE / 'test_the_canary.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _gate() -> Any:
    name = 't132a_gate'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, ROOT / 'contract/tests/t132a_gate.py'
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


C = _canary()
G = _gate()
DATA = C.DATA
#: Our number sentinel: digits no statement, code or field holds by chance.
NUMBER = '73400917'
#: The UBIGINT keys past int64: user 0's ``PAST_INT64``, the entity asked
#: about, which holds no sentinel; every other user's holds ``NUMBER`` in
#: its digits (``PAST_INT64 + user_id * 10**9 + NUMBER``), keys only the
#: data holds.
PAST_INT64 = 9300000000000000000
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
#: The sentinels planted in every scenario's data: two users' countries and
#: an order's price.
PLANTED = [
    f"UPDATE users SET country = '{DATA}' WHERE user_id IN (3, 5)",
    f'UPDATE orders SET price = {NUMBER} WHERE order_id = 11',
]
#: An entity no key holds, holding no sentinel.
NOBODY = 'nobody'
CORE = ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB)


def _alter(sql_type: str, using: str) -> list[str]:
    return [
        f'ALTER TABLE {table} ALTER user_id TYPE {sql_type} USING {using}'
        for table in ('users', 'orders')
    ]


def _materialize(setup: list[str], indices: list[Any]) -> Callable[[], Any]:
    def call() -> Any:
        connection = duckdb.connect()
        for statement in C._shop() + PLANTED + setup:
            connection.execute(statement)
        schema = Schema.from_source(DuckDBSource(connection))
        return PredictiveQuery(QUERY, schema).materialize(
            indices=indices, anchor_time=pd.Timestamp('2024-03-01')
        )

    return call


UINT8 = _alter('UTINYINT', 'user_id::UTINYINT')
#: Text keys, each holding the data sentinel, which a lookup that misses
#: must not list.
TEXT = _alter('VARCHAR', f"user_id::VARCHAR || '-{DATA}'")
DECIMAL = _alter('DECIMAL(18, 0)', 'user_id::DECIMAL(18, 0)')
FLOAT = _alter('FLOAT', 'user_id::FLOAT')
DOUBLE = _alter('DOUBLE', 'user_id::DOUBLE')
UBIGINT_PAST_INT64 = _alter(
    'UBIGINT',
    f'{PAST_INT64}::UBIGINT + user_id::UBIGINT * 1000000000 + CASE WHEN '
    f'user_id = 0 THEN 0 ELSE {NUMBER} END',
)

#: The reason each round-2 scenario's refusal names (the core's
#: ``details['reason']``), so the canary covers each reason's path.
REASONS = {
    'decimal: a text entity a decimal key cannot read': 'not_a_number',
    'decimal: a fraction a whole decimal key cannot hold': 'inexact',
    'float: an integer a float32 key cannot hold exactly': 'inexact',
    'float: a text entity a double key cannot read': 'not_a_number',
    'type: bytes for an integer key': 'type',
    'type: a timestamp for an integer key': 'type',
}

SCENARIOS: dict[str, Callable[[], Any]] = {
    'cast: a text entity an integer key cannot hold': _materialize(
        [], [NOBODY]
    ),
    'cast: a text entity among integers': _materialize([], [1, NOBODY]),
    'cast: an integer past an unsigned key s width': _materialize(UINT8, [300]),
    'cast: a negative integer for an unsigned key': _materialize(UINT8, [-3]),
    'range: a key past int64 planted in the data': _materialize(
        UBIGINT_PAST_INT64, [PAST_INT64]
    ),
    'lookup: a text entity no row holds': _materialize(TEXT, [NOBODY]),
    # Round 2: the decimal, float and type reasons (each named below).
    'decimal: a text entity a decimal key cannot read': _materialize(
        DECIMAL, [NOBODY]
    ),
    'decimal: a fraction a whole decimal key cannot hold': _materialize(
        DECIMAL, [Decimal('12.5')]
    ),
    'float: an integer a float32 key cannot hold exactly': _materialize(
        FLOAT, [16_777_217]
    ),
    'float: a text entity a double key cannot read': _materialize(
        DOUBLE, [NOBODY]
    ),
    'type: bytes for an integer key': _materialize([], [NOBODY.encode()]),
    'type: a timestamp for an integer key': _materialize(
        [], [pd.Timestamp(5, unit='us')]
    ),
}


@pytest.fixture(autouse=True)
def _core_route(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    # Our number is a sentinel too, for this file's scenarios.
    monkeypatch.setattr(C, 'SENTINELS', (*C.SENTINELS, NUMBER))


@pytest.mark.parametrize('name', list(SCENARIOS))
def test_no_sentinel_of_the_keys_family_reaches_anything_written(
    name: str,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    def check() -> None:
        with C._watching(monkeypatch, capfd) as seen:
            error = C._raised(seen, SCENARIOS[name])
        assert seen.leaks() == [], name
        # The core's error, raised with no cause chained: never the bridge's.
        assert any(where == 'core error' for where, _ in seen.texts), (
            f'{name}: refused outside the core ({error.code})'
        )
        assert error.__cause__ is None and error.__suppress_context__, name
        assert error.code in {'INVALID_REQUEST', 'INTERNAL_ERROR'}, error.code
        if name in REASONS:
            assert error.code == 'INVALID_REQUEST', error.code
            assert (error.details or {}).get('reason') == REASONS[name], (
                name,
                error.details,
            )

    G.held('exits', f'canary:{name}', check)


def test_each_sentinel_is_on_the_path_that_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The data the paths read holds both sentinels: the planted countries
    # and price, the text keys and the keys past int64; no entity a
    # scenario passes holds one (KO2-86: an error may name those).
    connection = duckdb.connect()
    for statement in C._shop() + PLANTED + UBIGINT_PAST_INT64:
        connection.execute(statement)
    rows = connection.execute('SELECT user_id FROM users').fetchall()
    assert all(
        (NUMBER in str(key)) == (key != PAST_INT64) for (key,) in rows
    ), rows
    countries = {
        c for (c,) in connection.execute('SELECT country FROM users').fetchall()
    }
    prices = {
        str(int(p))
        for (p,) in connection.execute('SELECT price FROM orders').fetchall()
    }
    assert DATA in countries and NUMBER in prices
    texts = duckdb.connect()
    for statement in C._shop() + TEXT:
        texts.execute(statement)
    keys = texts.execute('SELECT user_id FROM users').fetchall()
    assert all(DATA in key for (key,) in keys)
    for given in (NOBODY, 300, -3, PAST_INT64, Decimal('12.5'), 16_777_217):
        assert DATA not in str(given) and NUMBER not in str(given)
    # The watch reports either sentinel wherever it is written.
    monkeypatch.setattr(C, 'SENTINELS', (*C.SENTINELS, NUMBER))
    seen = C._Seen()
    seen.add('str', f'the keys {NUMBER} are past int64')
    seen.add('details', f'{{"keys": ["0-{DATA}"]}}')
    assert {sentinel for _, sentinel in seen.leaks()} == {NUMBER, DATA}
