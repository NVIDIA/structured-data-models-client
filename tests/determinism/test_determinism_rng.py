# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""All client-side randomness draws from numpy's legacy MT19937 streams.

``np.random.RandomState`` is frozen by NEP 19: the same seed gives the same
stream in every numpy release, and the same generator family as the C++
sampler's ``std::mt19937``. numpy's ``Generator`` API and Python's ``random``
module carry no such promise, so neither may appear in shipped code.

Each client-side stream is pinned below to an explicit ``RandomState``
reference. These are the documented seeds:

* the tabular context sample: ``RandomState(42)``;
* the relational entity-table sample (in-memory backend):
  ``RandomState(random_seed)``;
* the relational shuffle of candidate examples: one
  ``RandomState(random_seed)`` stream per ``sample_target`` call, shuffling
  the test candidates first and then the training candidates when they are
  drawn separately.
"""

from __future__ import annotations

import ast
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent.parent

GENERATOR_API = frozenset(
    {
        'default_rng',
        'Generator',
        'BitGenerator',
        'SeedSequence',
        'PCG64',
        'PCG64DXSM',
        'Philox',
        'SFC64',
    }
)


# --- no other generator in shipped code -------------------------------------


def _shipped_sources() -> list[Path]:
    return sorted(REPO.glob('packages/*/src/**/*.py'))


def _dotted(node: ast.expr) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return '.'.join(reversed(parts))


def _violations(path: Path) -> Iterator[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    numpy_random_aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == 'random' or alias.name.startswith('random.'):
                    yield f'{node.lineno}: import {alias.name}'
                if alias.name == 'numpy.random' and alias.asname:
                    numpy_random_aliases.add(alias.asname)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            if node.level == 0 and (
                module == 'random' or module.startswith('random.')
            ):
                yield f'{node.lineno}: from {module} import ...'
            if module == 'numpy':
                for alias in node.names:
                    if alias.name == 'random':
                        numpy_random_aliases.add(alias.asname or 'random')
            if module.startswith('numpy.random'):
                for alias in node.names:
                    if alias.name in GENERATOR_API or alias.name == '*':
                        yield f'{node.lineno}: from {module} import {alias.name}'
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in GENERATOR_API:
            owner = _dotted(node.value)
            if owner is None:
                continue
            if owner.endswith('.random') or owner in numpy_random_aliases:
                yield f'{node.lineno}: {owner}.{node.attr}'


def test_the_scan_sees_the_shipped_sources() -> None:
    sources = _shipped_sources()
    assert len(sources) > 100
    assert any('sdm_relational_engine' in str(p) for p in sources)
    assert any('sdm_client' in str(p) for p in sources)


@pytest.mark.parametrize(
    'snippet',
    [
        'import numpy as np\nnp.random.default_rng(1)',
        'import numpy\nnumpy.random.Generator(numpy.random.PCG64(1))',
        'from numpy.random import default_rng',
        'from numpy import random as npr\nnpr.SeedSequence(1)',
        'import random',
        'from random import shuffle',
    ],
)
def test_the_scan_catches_each_spelling(snippet: str, tmp_path: Path) -> None:
    path = tmp_path / 'snippet.py'
    path.write_text(snippet)
    assert list(_violations(path))


def test_the_scan_allows_the_legacy_stream_and_unrelated_names(
    tmp_path: Path,
) -> None:
    path = tmp_path / 'snippet.py'
    path.write_text(
        'import numpy as np\n'
        'from collections.abc import Generator\n'
        'def f() -> Generator[int, None, None]: ...\n'
        'np.random.RandomState(1).shuffle([1])\n'
    )
    assert not list(_violations(path))


def test_no_shipped_code_uses_another_generator() -> None:
    found = [
        f'{path.relative_to(REPO)}:{violation}'
        for path in _shipped_sources()
        for violation in _violations(path)
    ]
    assert not found, (
        'Client-side randomness must use np.random.RandomState only:\n  '
        + '\n  '.join(found)
    )


# --- the tabular context sample ---------------------------------------------


def _people(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            'user_id': range(n),
            'age': [20.0 + i % 50 for i in range(n)],
            'churn': ['yes' if i % 2 else 'no' for i in range(n)],
        }
    )


@pytest.mark.parametrize('wanted', [100, 250])
def test_the_tabular_context_sample_is_random_state_42(wanted: int) -> None:
    from sdm_client import PredictiveQuery

    people = _people(4_000)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        request = PredictiveQuery(
            'PREDICT people.churn FOR EACH people.user_id', {'people': people}
        ).materialize(indices=[0], num_context_samples=wanted)

    candidates = people[people['user_id'] != 0]
    drawn = np.random.RandomState(42).choice(
        len(candidates), size=wanted, replace=False
    )
    expected = candidates.iloc[np.sort(drawn)]['user_id'].tolist()
    assert request.context.instance_table['user_id'].tolist() == expected


# --- the relational streams ---------------------------------------------------

_NUM_USERS = 40


def _local_sampler() -> tuple[Any, Any, np.ndarray]:
    pytest.importorskip('sdm_relational_engine.relationallib')
    from sdm_relational_engine.rfm import Graph, KumoRelational

    users = pd.DataFrame(
        {
            'user_id': np.arange(100, 100 + _NUM_USERS),
            'age': np.arange(_NUM_USERS) % 7,
            'churn': np.arange(_NUM_USERS) % 2,
        }
    )
    graph = Graph.from_data({'users': users}, verbose=False)
    model = KumoRelational(graph, verbose=False)
    query = model._parse_query('PREDICT users.churn FOR EACH users.user_id')
    return model._sampler, query, users['user_id'].to_numpy()


def _sample(
    sampler: Any,
    query: Any,
    seed: int,
    *,
    trials: int,
    train_anchor: pd.Timestamp,
    test_anchor: pd.Timestamp,
) -> tuple[Any, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return sampler.sample_target(
            query=query,
            num_train_examples=10,
            train_anchor_time=train_anchor,
            num_train_trials=trials,
            num_test_examples=5,
            test_anchor_time=test_anchor,
            num_test_trials=trials,
            random_seed=seed,
        )


@pytest.mark.parametrize('seed', [0, 7, 42, 12_345, 2**32 - 1])
def test_the_example_shuffle_is_random_state_of_the_seed(seed: int) -> None:
    r"""Every entity is a candidate, so only the shuffle orders them."""
    sampler, query, keys = _local_sampler()
    anchor = pd.Timestamp('2026-01-01')
    train, test = _sample(
        sampler,
        query,
        seed,
        trials=_NUM_USERS,
        train_anchor=anchor,
        test_anchor=anchor,
    )

    index = np.arange(_NUM_USERS)
    np.random.RandomState(seed).shuffle(index)
    expected = keys[index][:15].tolist()
    got = test.entity_pkey.tolist() + train.entity_pkey.tolist()
    assert got == expected


@pytest.mark.parametrize('seed', [0, 7, 12_345])
def test_test_then_training_candidates_share_one_stream(seed: int) -> None:
    r"""Two anchors: the test candidates are shuffled, then the training."""
    sampler, query, keys = _local_sampler()
    train, test = _sample(
        sampler,
        query,
        seed,
        trials=_NUM_USERS,
        train_anchor=pd.Timestamp('2025-12-01'),
        test_anchor=pd.Timestamp('2026-01-01'),
    )

    stream = np.random.RandomState(seed)
    test_index = np.arange(_NUM_USERS)
    stream.shuffle(test_index)
    train_index = np.arange(_NUM_USERS)
    stream.shuffle(train_index)
    assert test.entity_pkey.tolist() == keys[test_index][:5].tolist()
    assert train.entity_pkey.tolist() == keys[train_index][:10].tolist()


@pytest.mark.parametrize('seed', [0, 7, 12_345])
def test_the_entity_sample_is_random_state_of_the_seed(seed: int) -> None:
    r"""Fewer candidates than entities: the table is sampled, then shuffled."""
    sampler, query, keys = _local_sampler()
    anchor = pd.Timestamp('2026-01-01')
    trials = 8  # 16 of 40 entities are drawn for the shared train/test set.
    train, test = _sample(
        sampler,
        query,
        seed,
        trials=trials,
        train_anchor=anchor,
        test_anchor=anchor,
    )

    drawn = keys[
        np.random.RandomState(seed).choice(
            _NUM_USERS, size=2 * trials, replace=False
        )
    ]
    index = np.arange(2 * trials)
    np.random.RandomState(seed).shuffle(index)
    expected = drawn[index][:15].tolist()
    got = test.entity_pkey.tolist() + train.entity_pkey.tolist()
    assert got == expected
