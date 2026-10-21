# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T1.28w exit 2 on python-core, held from outside the implementation (the
package's independent test author): the core says when a held sampler state
is dropped, and the Python host follows it.

At the public route (``PredictiveQuery.materialize`` forced onto the core
over a DuckDB source), a materializer holding a sampler state the core
refuses (of another version, a memo that is not JSON, a memo no run left)
fails that call, drops the state, and its next call runs as a fresh
materializer's first call: the same statements, the same payload. An
error that is not about the state (a call refused by the core, a run that
fails after it started) leaves the held state in place, and the next call
runs as a second call does, skipping what the state lets it skip. A memo
made under another source is no refusal: the run reads afresh.

The number of statements a call runs tells the two apart (decision E52,
H9's counts): a materializer's first call over the ``shop`` base runs more
statements than its second, which skips the index check and the time
bounds its held state carries.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from sdm_connectors import DuckDBSource

from sdm_client import PredictiveQuery, Schema, StructuredDataError, _routing

ROOT = Path(__file__).resolve().parents[3]
CORE = ','.join(_routing.FROM_SOURCE_DUCKDB + _routing.MATERIALIZE_DUCKDB)
QUERY = 'PREDICT SUM(orders.price, 0, 30, days) FOR EACH users.user_id'
ANCHOR = pd.Timestamp('2024-03-01')


@pytest.fixture(autouse=True)
def _core(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)


def _shop() -> list[str]:
    lines = (
        (ROOT / 'contract/bases.yaml').read_text(encoding='utf-8').splitlines()
    )
    start = lines.index('  shop:') + 1
    statements = []
    for line in lines[start:]:
        if not line.startswith('  - '):
            break
        statements.append(line[len('  - ') :])
    return statements


class _Shop:
    r"""One materializer over the ``shop`` base, its statements counted."""

    def __init__(self, query: str = QUERY) -> None:
        self.connection = duckdb.connect()
        for statement in _shop():
            self.connection.execute(statement)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            schema = Schema.from_source(DuckDBSource(self.connection))
        self.query = PredictiveQuery(query, schema)
        self.connection.execute("CALL enable_logging('QueryLog')")

    @property
    def state(self) -> dict[str, Any]:
        return self.query._relational_materializer._core_sampler_state

    def _logged(self) -> int:
        rows = self.connection.execute(
            "SELECT message FROM duckdb_logs WHERE type = 'QueryLog'"
        ).fetchall()
        return sum(1 for (text,) in rows if text and 'duckdb_logs' not in text)

    def call(self, **options: Any) -> tuple[int, Any]:
        r"""The statements one call ran, and its payload or its error."""
        before = self._logged()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                made: Any = self.query.materialize(
                    indices=[0, 1], anchor_time=ANCHOR, **options
                )._payloads[0]
            except StructuredDataError as error:
                made = error
        return self._logged() - before, made


def _fresh() -> tuple[tuple[int, Any], tuple[int, Any]]:
    shop = _Shop()
    first = shop.call(random_seed=7)
    second = shop.call(random_seed=7)
    return first, second


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(
        b, sort_keys=True, default=str
    )


def test_the_held_state_is_seen_in_the_statements_a_call_runs() -> None:
    # The control the tests below rest on: a held state changes what the
    # next call runs, so a dropped state can be told from a kept one.
    (first, payload), (second, again) = _fresh()
    assert not isinstance(payload, Exception), payload
    assert second < first, (first, second)
    assert _same(payload, again)


def _unsound(held: dict[str, Any]) -> bytes:
    memo = json.loads(bytes(held['memo']))
    memo['bounds'] = [
        {'table': 'orders', 'column': 'x', 'min': 2, 'max': 1},
    ]
    return json.dumps(memo).encode()


@pytest.mark.parametrize(
    'corrupt',
    [
        pytest.param(lambda h: {**h, 'version': h['version'] + 1}, id='newer'),
        pytest.param(lambda h: {**h, 'version': 0}, id='version-0'),
        pytest.param(lambda h: {**h, 'memo': b'\x00not json'}, id='not-json'),
        pytest.param(lambda h: {**h, 'memo': b'{"bounds": 1}'}, id='shape'),
        pytest.param(lambda h: {**h, 'memo': _unsound(h)}, id='unsound'),
    ],
)
def test_a_held_state_the_core_refuses_is_dropped_and_the_next_call_is_fresh(
    corrupt: Any,
) -> None:
    (first, payload), (second, _) = _fresh()
    shop = _Shop()
    shop.call(random_seed=7)
    assert 'held' in shop.state
    shop.state['held'] = corrupt(shop.state['held'])
    ran, refused = shop.call(random_seed=7)
    assert isinstance(refused, StructuredDataError), refused
    assert ran == 0, ran
    # Dropped on the core's word: nothing is held after the refusal.
    assert 'held' not in shop.state, shop.state
    # The next call is a fresh materializer's first call ...
    ran, made = shop.call(random_seed=7)
    assert (ran, _same(made, payload)) == (first, True)
    # ... and leaves a state the one after it uses.
    assert 'held' in shop.state
    assert shop.call(random_seed=7)[0] == second


def test_a_memo_of_another_source_is_read_afresh_not_refused() -> None:
    (first, payload), (second, _) = _fresh()
    shop = _Shop()
    # A memo of a materializer over another source: one more table, so a
    # description whose tables differ (a column added would not do: the
    # memo's fingerprint is of what its reads depend on, keys, links and
    # time columns, decision E52).
    elsewhere = _Shop()
    elsewhere.connection.execute('CREATE TABLE extra AS SELECT 1 AS n')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        changed = Schema.from_source(DuckDBSource(elsewhere.connection))
    elsewhere.query = PredictiveQuery(QUERY, changed)
    elsewhere.call(random_seed=7)
    assert 'held' in elsewhere.state
    shop.state['held'] = dict(elsewhere.state['held'])
    ran, made = shop.call(random_seed=7)
    assert not isinstance(made, Exception), made
    assert (ran, _same(made, payload)) == (first, True)
    assert shop.call(random_seed=7)[0] == second


@pytest.mark.parametrize(
    'unrelated',
    [
        pytest.param({'random_seed': -1}, id='negative-seed'),
        pytest.param({'random_seed': 2**70}, id='seed-past-the-provider'),
        pytest.param({'num_context_samples': 0}, id='no-context'),
    ],
)
def test_an_error_not_about_the_state_keeps_it(
    unrelated: dict[str, Any],
) -> None:
    (first, payload), (second, again) = _fresh()
    shop = _Shop()
    shop.call(random_seed=7)
    held = dict(shop.state['held'])
    _, failed = shop.call(**unrelated)
    assert isinstance(failed, StructuredDataError), failed
    # Kept: the same state, and the next call skips what it lets it skip.
    assert shop.state.get('held') is not None
    assert shop.state['held']['version'] == held['version']
    ran, made = shop.call(random_seed=7)
    assert not isinstance(made, Exception), made
    assert (ran, _same(made, again)) == (second, True)


def test_the_core_s_own_refusal_of_a_start_not_about_the_state_keeps_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The errors above are refused by the client before a run, or fail a
    # run once it started; this one is the core's refusal of the start
    # itself, with a sound state handed to it: a negative seed, which the
    # core refuses before it reads the state. The client's own check of
    # the seed is set aside so the core sees it. A host dropping the state
    # on any refusal of a start, not on the core's word, fails here.
    from sdm_client.materializers.relational import RelationalMaterializer

    (_, _), (second, again) = _fresh()
    shop = _Shop()
    shop.call(random_seed=7)
    held = dict(shop.state['held'])
    monkeypatch.setattr(RelationalMaterializer, '_check_shape', lambda *a: None)
    ran, refused = shop.call(random_seed=-1)
    monkeypatch.undo()
    monkeypatch.setenv(_routing.ENVIRONMENT, CORE)
    assert isinstance(refused, StructuredDataError), refused
    # The core's refusal, before anything was read, worded as the
    # reference words the same seed (python-ref at 6298913c; T2.14b moved
    # the core's refusals into the reference's words).
    assert (refused.code, ran) == ('INVALID_REQUEST', 0)
    assert str(refused) == (
        '[INVALID_REQUEST] random_seed is -1, and a seed is not negative; '
        'pass a number from 0 upwards, or leave it out to let the sampler '
        'choose'
    ), refused
    assert shop.state.get('held') == held
    ran, made = shop.call(random_seed=7)
    assert (ran, _same(made, again)) == (second, True)
