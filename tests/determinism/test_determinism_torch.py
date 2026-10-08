# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Payloads do not depend on whether ``torch`` is installed.

The in-memory graph store sorted each neighborhood with numpy when ``torch``
was missing and with ``torch`` when it was present. The two unstable sorts
break ties differently, and the tie order decides which neighbor the sampler
picks. A real ``torch`` is a large install that may break ties like numpy by
coincidence, so these checks use a fake whose default sort puts ties in
descending position, which numpy's does not. Code that still sorts with
whatever ``torch`` provides fails here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from determinism_payloads import FAKE_TORCH, describe_differences, run_payloads


def _load_fake_torch() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        'torch', FAKE_TORCH / 'torch' / '__init__.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- the fake is a fair test -----------------------------------------------


def test_the_fake_breaks_ties_unlike_numpy() -> None:
    torch = _load_fake_torch()
    keys = np.repeat(np.arange(40, dtype=np.int64), 5)[::-1].copy()
    fake = torch.from_numpy(keys).argsort().numpy()
    assert np.array_equal(keys[fake], np.sort(keys))
    assert not np.array_equal(fake, np.argsort(keys))
    assert not np.array_equal(fake, np.argsort(keys, kind='stable'))


def test_the_fake_s_stable_sort_is_stable() -> None:
    torch = _load_fake_torch()
    keys = np.repeat(np.arange(40, dtype=np.int64), 5)[::-1].copy()
    stable = torch.from_numpy(keys).argsort(stable=True).numpy()
    assert np.array_equal(stable, np.argsort(keys, kind='stable'))


# --- the payloads ------------------------------------------------------------


@pytest.fixture(scope='module')
def by_torch(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    tmp_path = tmp_path_factory.mktemp('torch')
    return {
        mode: run_payloads(tmp_path, mode, 'snapshot', '--torch', mode)
        for mode in ('absent', 'fake', 'ambient')
    }


def test_each_run_had_the_torch_it_asked_for(by_torch: dict) -> None:
    assert by_torch['absent']['torch'] == 'absent'
    fake_origin = Path(by_torch['fake']['torch']).resolve()
    assert fake_origin == (FAKE_TORCH / 'torch' / '__init__.py').resolve()


def test_payloads_are_the_same_with_a_fake_torch_and_without_one(
    by_torch: dict,
) -> None:
    differences = describe_differences(
        by_torch['absent']['payloads'], by_torch['fake']['payloads']
    )
    assert not differences, (
        'payloads change when a torch can be imported:\n  '
        + '\n  '.join(differences)
    )


def test_payloads_are_the_same_with_this_environment_s_torch(
    by_torch: dict,
) -> None:
    r"""Whatever this environment has, real ``torch`` included."""
    differences = describe_differences(
        by_torch['absent']['payloads'], by_torch['ambient']['payloads']
    )
    assert not differences, '\n  '.join(differences)


# --- the neighborhood order itself ------------------------------------------


def _graph_with_ties(timed: bool) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    pytest.importorskip('sdm_relational_engine.relationallib')
    from determinism_payloads import tie_frames
    from sdm_relational_engine.rfm import Graph

    users, orders = tie_frames()
    # Shuffled rows, so that source position and time disagree and a sort
    # that ignores either one is visible.
    orders = orders.sample(frac=1.0, random_state=np.random.RandomState(3))
    orders = orders.reset_index(drop=True)
    if not timed:
        orders = orders.drop(columns=['ordered_at'])
    graph = Graph.from_data(
        {'users': users, 'orders': orders}, edges=[], verbose=False
    )
    graph.link('orders', 'user_id', 'users')
    return graph, users, orders


@pytest.mark.parametrize('timed', [True, False], ids=['by-time', 'untimed'])
@pytest.mark.parametrize('torch_mode', ['absent', 'fake'])
def test_a_neighborhood_is_ordered_by_time_then_source_position(
    monkeypatch: pytest.MonkeyPatch, timed: bool, torch_mode: str
) -> None:
    r"""Ties are broken by the neighbor's row position, ascending.

    That is the order the overflow-safe route, a stable lexicographic sort on
    (parent, time), already produces, so both routes agree. It is also the
    only order that does not depend on which sort algorithm a platform's
    numpy dispatches to.
    """
    graph, users, orders = _graph_with_ties(timed)
    from sdm_relational_engine.rfm.backend.local.graph_store import (
        LocalGraphStore,
    )

    if torch_mode == 'absent':
        monkeypatch.setitem(sys.modules, 'torch', None)
    else:
        monkeypatch.setitem(sys.modules, 'torch', _load_fake_torch())

    store = LocalGraphStore(graph, verbose=False)
    row = store.row_dict[('orders', 'user_id', 'users')]
    colptr = store.colptr_dict[('orders', 'user_id', 'users')]

    parent = orders['user_id'].to_numpy()
    keys: list[np.ndarray] = [np.arange(len(orders))]
    if timed:
        keys.append(orders['ordered_at'].to_numpy().astype('int64'))
    keys.append(parent)
    expected = np.lexsort(keys)

    assert np.array_equal(row, expected)
    assert np.array_equal(
        colptr, np.concatenate([[0], np.cumsum(np.bincount(parent))])
    )
