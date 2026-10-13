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

The sentinels are the canary's data sentinel, carried by the entities, and
a number of our own, ``NUMBER``, carried by entities and planted as key
values. The reference quotes each in its message (``indices[0] is ...``,
the value past int64, the missing keys); the core may not.

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
#: A UBIGINT key past int64 whose digits hold ``NUMBER``.
PAST_INT64 = 9273400917000000000
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
CORE = ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB)


def _alter(sql_type: str, using: str) -> list[str]:
    return [
        f'ALTER TABLE {table} ALTER user_id TYPE {sql_type} USING {using}'
        for table in ('users', 'orders')
    ]


def _materialize(setup: list[str], indices: list[Any]) -> Callable[[], Any]:
    def call() -> Any:
        connection = duckdb.connect()
        for statement in C._shop() + setup:
            connection.execute(statement)
        schema = Schema.from_source(DuckDBSource(connection))
        return PredictiveQuery(QUERY, schema).materialize(
            indices=indices, anchor_time=pd.Timestamp('2024-03-01')
        )

    return call


UINT8 = _alter('UTINYINT', 'user_id::UTINYINT')
TEXT = _alter('VARCHAR', 'user_id::VARCHAR')
DECIMAL = _alter('DECIMAL(18, 0)', 'user_id::DECIMAL(18, 0)')
FLOAT = _alter('FLOAT', 'user_id::FLOAT')
DOUBLE = _alter('DOUBLE', 'user_id::DOUBLE')
UBIGINT_PAST_INT64 = _alter(
    'UBIGINT', f'user_id::UBIGINT + {PAST_INT64}::UBIGINT'
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
    'cast: a text entity an integer key cannot hold': _materialize([], [DATA]),
    'cast: a text entity among integers': _materialize([], [1, DATA]),
    'cast: an integer past an unsigned key s width': _materialize(
        UINT8, [int(NUMBER)]
    ),
    'cast: a negative integer for an unsigned key': _materialize(
        UINT8, [-int(NUMBER)]
    ),
    'range: a key past int64 planted in the data': _materialize(
        UBIGINT_PAST_INT64, [PAST_INT64 + 1]
    ),
    'lookup: a text entity no row holds': _materialize(TEXT, [DATA]),
    # Round 2: the decimal, float and type reasons (each named below).
    'decimal: a text entity a decimal key cannot read': _materialize(
        DECIMAL, [DATA]
    ),
    'decimal: a fraction a whole decimal key cannot hold': _materialize(
        DECIMAL, [Decimal(f'{NUMBER}.5')]
    ),
    'float: an integer a float32 key cannot hold exactly': _materialize(
        FLOAT, [int(NUMBER)]
    ),
    'float: a text entity a double key cannot read': _materialize(
        DOUBLE, [DATA]
    ),
    'type: bytes for an integer key': _materialize([], [DATA.encode()]),
    'type: a timestamp for an integer key': _materialize(
        [], [pd.Timestamp(int(NUMBER), unit='us')]
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
    # The planted keys hold the number where the source hands them over;
    # the other scenarios carry a sentinel in their entities.
    connection = duckdb.connect()
    for statement in C._shop() + UBIGINT_PAST_INT64:
        connection.execute(statement)
    rows = connection.execute('SELECT user_id FROM users').fetchall()
    keys = [str(key) for (key,) in rows]
    assert all(NUMBER in key for key in keys)
    assert NUMBER in str(int(NUMBER)) and NUMBER in str(-int(NUMBER))
    # The watch reports either sentinel wherever it is written.
    monkeypatch.setattr(C, 'SENTINELS', (*C.SENTINELS, NUMBER))
    seen = C._Seen()
    seen.add('str', f'indices[0] is {int(NUMBER)}')
    seen.add('details', f'{{"value": "{DATA}"}}')
    assert {sentinel for _, sentinel in seen.leaks()} == {NUMBER, DATA}
