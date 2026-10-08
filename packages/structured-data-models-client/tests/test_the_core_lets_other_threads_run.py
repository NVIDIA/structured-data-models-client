# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The core works without holding the GIL, so other Python threads run
while a step of it does, and the language's validation runs on a thread
with the default stack."""

from __future__ import annotations

import threading
import time
import warnings
from typing import Any

import pandas as pd
import pytest

from sdm_client import PredictiveQuery, Schema, _core, _impl

pytestmark = pytest.mark.skipif(
    _impl.OLD, reason='the old implementation runs in Python'
)


class _Timed:
    r"""A run whose steps are timed: each ``next`` and ``submit`` call's
    start and end."""

    def __init__(self, run: Any, spans: list[tuple[str, float, float]]) -> None:
        self._run = run
        self._spans = spans

    def next(self) -> Any:
        start = time.perf_counter()
        try:
            return self._run.next()
        finally:
            self._spans.append(('next', start, time.perf_counter()))

    def submit(self, *args: Any) -> Any:
        start = time.perf_counter()
        try:
            return self._run.submit(*args)
        finally:
            self._spans.append(('submit', start, time.perf_counter()))


def _large_shop() -> dict[str, pd.DataFrame]:
    users = pd.DataFrame({'user_id': range(5_000), 'age': range(5_000)})
    orders = pd.DataFrame(
        {
            'order_id': range(400_000),
            'user_id': [i % 5_000 for i in range(400_000)],
            'quantity': [i % 7 for i in range(400_000)],
            'ordered_at': pd.date_range(
                '2020-01-01', periods=400_000, freq='min'
            ),
        }
    )
    return {'users': users, 'orders': orders}


def test_another_thread_runs_while_the_core_steps(monkeypatch):
    spans: list[tuple[str, float, float]] = []
    drive = _core.Driver.drive
    monkeypatch.setattr(
        _core.Driver,
        'drive',
        lambda self, run: drive(self, _Timed(run, spans)),
    )
    schema = Schema.from_data(_large_shop())
    query = PredictiveQuery(
        'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id', schema
    )

    beats: list[float] = []
    stop = threading.Event()

    def beat() -> None:
        while not stop.is_set():
            beats.append(time.perf_counter())
            time.sleep(0.001)

    beating = threading.Thread(target=beat)
    beating.start()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            query.materialize(indices=[1, 2], random_seed=7)
    finally:
        stop.set()
        beating.join()

    next_spans = [
        (start, end) for method, start, end in spans if method == 'next'
    ]
    all_spans = [(start, end) for _, start, end in spans]
    for eligible in (all_spans, next_spans):
        start, end = max(eligible, key=lambda span: span[1] - span[0])
        assert end - start > 0.2, f'the longest step took {end - start:.3f}s'
        during = [stamp for stamp in beats if start + 0.05 < stamp < end - 0.05]
        assert len(during) > 10, (
            f'{len(during)} beats in a step of {end - start:.3f}s: the core held '
            'the GIL'
        )


def test_submit_releases_the_gil_while_discarding_an_unclaimed_outcome():
    """Rejecting an unknown query drops the cloned batches inside submit."""
    import pyarrow as pa
    from sdm_core import _native

    batch = pa.record_batch([pa.array([1], type=pa.int64())], names=['count'])
    table = pa.Table.from_batches([batch] * 1_000_000)
    assert table.column(0).num_chunks == 1_000_000
    outcome = _native.Outcome.data(table)
    run = _native.count_rows('proof')
    assert run.next().kind == 'run_queries'
    spans: list[tuple[str, float, float]] = []
    beats: list[float] = []
    stop = threading.Event()

    def beat() -> None:
        while not stop.is_set():
            beats.append(time.perf_counter())
            time.sleep(0.001)

    beating = threading.Thread(target=beat)
    beating.start()
    try:
        with pytest.raises(_native.CoreError) as error:
            _Timed(run, spans).submit(2**63, outcome)
        assert error.value.code == 'UNKNOWN_QUERY'
    finally:
        stop.set()
        beating.join()
    assert len(spans) == 1
    method, start, end = spans[0]
    assert method == 'submit'
    assert end - start > 0.02, spans
    during = [stamp for stamp in beats if start + 0.002 < stamp < end - 0.002]
    assert len(during) >= 3, (spans, len(during))
    assert during[-1] - during[0] >= 0.002, during


def test_a_long_question_validates_on_a_thread_of_the_default_stack():
    r"""The language's parser and validator recurse; on a thread with the
    default stack (512 KiB on macOS), a deeply nested question is refused
    or answered, never a crash of the process."""
    users = pd.DataFrame({'user_id': range(20), 'age': range(20)})
    orders = pd.DataFrame(
        {
            'order_id': range(200),
            'user_id': [i % 20 for i in range(200)],
            'quantity': [i % 5 for i in range(200)],
            'ordered_at': pd.date_range('2024-01-01', periods=200, freq='D'),
        }
    )
    schema = Schema.from_data({'users': users, 'orders': orders})
    nested = ' AND '.join(
        f'(COUNT(orders.*, 0, {d}, days) > {d % 3})' for d in range(1, 400)
    )
    deep = '(' * 2_000 + 'users.age > 3' + ')' * 2_000
    found: dict[str, Any] = {}

    def ask() -> None:
        for name, question in (
            ('long', f'PREDICT {nested} FOR EACH users.user_id'),
            (
                'deep',
                f'PREDICT COUNT(orders.*, 0, 30, days) FOR EACH users.user_id WHERE {deep}',
            ),
        ):
            try:
                PredictiveQuery(question, schema)
                found[name] = 'answered'
            except Exception as error:  # noqa: BLE001 - either is fine
                found[name] = type(error).__name__

    asking = threading.Thread(target=ask)
    asking.start()
    asking.join()
    assert set(found) == {'long', 'deep'}, found
