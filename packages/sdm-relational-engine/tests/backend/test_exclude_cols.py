# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``exclude_cols_dict`` is a public ``predict_task`` argument.

a typo in it used to
delete straight out of the sampler's stype map and raise a bare ``KeyError``,
which the client could only classify as ``INTERNAL_ERROR`` -- telling the user to
file a bug against the client for their own typo. Raising ``ValueError`` here puts
it in the ``INVALID_REQUEST`` branch and names what is wrong.
"""

from __future__ import annotations

import pandas as pd
import pytest
from sdm_relational_engine.rfm import Graph
from sdm_relational_engine.rfm.backend.local import LocalSampler


@pytest.fixture()
def sampler() -> LocalSampler:
    graph = Graph.from_data(
        {
            'users': pd.DataFrame(
                {
                    'user_id': range(8),
                    'age': [20 + i for i in range(8)],
                }
            ),
            'orders': pd.DataFrame(
                {
                    'order_id': range(16),
                    'user_id': [i % 8 for i in range(16)],
                    'amount': [float(i) for i in range(16)],
                }
            ),
        },
        verbose=False,
    )
    return LocalSampler(graph, verbose=False)


def _sample(sampler: LocalSampler, exclude: dict[str, list[str]]) -> None:
    sampler.sample_subgraph(
        entity_table_names=('users',),
        entity_pkey=pd.Series([0, 1]),
        anchor_time=pd.Series(pd.to_datetime(['2025-01-01'] * 2)),
        num_neighbors=[4, 4],
        exclude_cols_dict=exclude,
    )


def test_an_unknown_table_is_named(sampler: LocalSampler) -> None:
    with pytest.raises(ValueError) as excinfo:
        _sample(sampler, {'nope': ['age']})
    message = str(excinfo.value)
    assert "'nope'" in message
    assert 'not in the graph' in message
    assert 'users' in message and 'orders' in message


def test_an_unknown_column_is_named(sampler: LocalSampler) -> None:
    with pytest.raises(ValueError) as excinfo:
        _sample(sampler, {'users': ['nope']})
    message = str(excinfo.value)
    assert "'nope'" in message
    assert "'users'" in message
    assert "'age'" in message


def test_a_primary_key_is_reported_as_not_excludable(
    sampler: LocalSampler,
) -> None:
    r"""Primary and foreign keys never travel as features, so excluding one is
    a no-op the caller almost certainly did not intend -- and it raised the
    same bare ``KeyError``.
    """
    with pytest.raises(ValueError, match='feature columns'):
        _sample(sampler, {'users': ['user_id']})


@pytest.mark.parametrize(
    'exclude',
    [
        None,
        {},
        {'users': []},
        {'users': ['age']},
        {'orders': ['amount']},
        {'users': ['age'], 'orders': ['amount']},
    ],
)
def test_valid_exclusions_are_still_accepted(
    sampler: LocalSampler, exclude
) -> None:
    r"""The guard must not reject what already worked."""
    _sample(sampler, exclude)
