# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Two callers configuring one model at the same time.

``batch_mode`` and ``retry`` used to hold their setting on the instance, so a
model shared by concurrent callers served whichever value was set last. A caller
inside its own ``batch_mode(100)`` could be batched at 500 because another thread
had entered meanwhile, and the restore on exit put back a value that was already
someone else's.

The setting is per execution context now, which is what a caching client needs:
one model, many concurrent predictions, each with the configuration it asked for.
"""

import threading

import pytest
from sdm_relational_engine.rfm.rfm import KumoRelational


def _model() -> KumoRelational:
    r"""A model with only the configuration state these tests touch."""
    from contextvars import ContextVar

    model = KumoRelational.__new__(KumoRelational)
    model._batch_size = ContextVar('batch_size', default=None)
    model._num_retries = ContextVar('num_retries', default=0)
    return model


def test_a_second_caller_does_not_change_the_first_ones_batch_size() -> None:
    r"""The interleaving that corrupted it: both inside at once."""
    model = _model()
    seen: dict[str, object] = {}
    second_is_inside = threading.Event()
    first_has_read = threading.Event()

    def first() -> None:
        with model.batch_mode(batch_size=100):
            second_is_inside.wait(5)
            seen['first'] = model._batch_size.get()
            first_has_read.set()

    def second() -> None:
        with model.batch_mode(batch_size=500):
            second_is_inside.set()
            first_has_read.wait(5)
            seen['second'] = model._batch_size.get()

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert seen['first'] == 100
    assert seen['second'] == 500


def test_a_second_caller_does_not_change_the_first_ones_retries() -> None:
    model = _model()
    seen: dict[str, object] = {}
    second_is_inside = threading.Event()
    first_has_read = threading.Event()

    def first() -> None:
        with model.retry(num_retries=1):
            second_is_inside.wait(5)
            seen['first'] = model._num_retries.get()
            first_has_read.set()

    def second() -> None:
        with model.retry(num_retries=9):
            second_is_inside.set()
            first_has_read.wait(5)

    threads = [threading.Thread(target=first), threading.Thread(target=second)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)

    assert seen['first'] == 1


def test_the_configuration_is_put_back_when_the_block_ends() -> None:
    model = _model()

    with model.batch_mode(batch_size=100, num_retries=3):
        assert model._batch_size.get() == 100
        assert model._num_retries.get() == 3

    assert model._batch_size.get() is None
    assert model._num_retries.get() == 0


def test_nesting_restores_the_outer_setting() -> None:
    model = _model()

    with model.batch_mode(batch_size=100):
        with model.batch_mode(batch_size=500):
            assert model._batch_size.get() == 500
        assert model._batch_size.get() == 100


def test_two_async_tasks_on_one_thread_keep_their_own_settings() -> None:
    r"""The case a per-thread model cache cannot cover.

    The client hands each THREAD its own engine model precisely because the
    batch size and retry count used to live on the instance. Two asyncio tasks
    on one thread share that model, so the workaround does not reach them:
    holding the setting on the instance, the second task's block overwrites the
    first's and the first's exit restores a value the second is still inside of.

    Context variables are copied per task, so each block sees only its own.
    """
    import asyncio

    model = _model()
    seen: dict[str, object] = {}

    async def predict_with(name: str, batch_size: int) -> None:
        with model.batch_mode(batch_size):
            await asyncio.sleep(0.05)
            seen[name] = model._batch_size.get()

    async def both() -> None:
        await asyncio.gather(predict_with('a', 100), predict_with('b', 500))

    asyncio.run(both())

    assert seen == {'a': 100, 'b': 500}


def test_two_async_tasks_keep_their_own_retry_counts() -> None:
    import asyncio

    model = _model()
    seen: dict[str, object] = {}

    async def predict_with(name: str, retries: int) -> None:
        with model.retry(retries):
            await asyncio.sleep(0.05)
            seen[name] = model._num_retries.get()

    async def both() -> None:
        await asyncio.gather(predict_with('a', 2), predict_with('b', 7))

    asyncio.run(both())

    assert seen == {'a': 2, 'b': 7}


def test_a_real_predict_uses_the_batch_size_of_its_own_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    r"""These tests drive the context managers; this drives predict() itself.

    A regression in how predict resolves ``self._batch_size.get()`` into a
    batch would not be caught by exercising ``batch_mode`` on its own.
    """
    import pandas as pd
    import sdm_relational_engine.rfm as rfm
    from sdm_relational_engine.rfm.rfm import KumoRelational

    customers = pd.DataFrame({'customer_id': range(40), 'seg': ['a', 'b'] * 20})
    orders = pd.DataFrame(
        {
            'order_id': range(200),
            'customer_id': [i % 40 for i in range(200)],
            'placed_at': pd.to_datetime(
                [
                    f'2025-{1 + (i // 17) % 12:02d}-{1 + i % 28:02d}'
                    for i in range(200)
                ]
            ),
        }
    )
    graph = rfm.Graph.from_data(
        {'customers': customers, 'orders': orders},
        infer_metadata=True,
        verbose=False,
    )
    graph['customers'].primary_key = 'customer_id'
    graph['orders'].primary_key = 'order_id'
    if not graph.edges:
        graph.link('orders', 'customer_id', 'customers')
    model = KumoRelational(graph, verbose=False)

    seen: dict[str, object] = {}

    def capture(self: KumoRelational, requests: object, **kw: object) -> tuple:
        seen[threading.current_thread().name] = (
            self._resolve_batch_size.__self__._batch_size.get()
        )
        return (
            [pd.DataFrame({'ENTITY': [1], 'TARGET_PRED': [1.0]})],
            None,
            None,
            None,
        )

    monkeypatch.setattr(KumoRelational, '_predict_batches', capture)
    query = (
        'PREDICT COUNT(orders.*, 0, 30) FOR customers.customer_id IN (1, 2, 3)'
    )
    gate = threading.Barrier(2, timeout=10)

    def run(size: int) -> None:
        with model.batch_mode(size):
            gate.wait()
            model.predict(query, verbose=False)

    threads = [
        threading.Thread(target=run, args=(size,), name=str(size))
        for size in (100, 500)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)

    assert seen == {'100': 100, '500': 500}
