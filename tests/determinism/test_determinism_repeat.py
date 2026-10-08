# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The same request builds the same payload, every time and everywhere.

Every scenario the wire snapshot records, and in-memory relational scenarios
whose orders share a parent and a timestamp, are built ten times in one process
and once in each of three fresh processes. Each fresh process gets its own
string-hash seed, so nothing that leaks set or dict order can agree by luck.
Payloads are compared parsed, with ``metadata.request_id`` normalized as the
snapshot normalizes it.
"""

from __future__ import annotations

import pytest
from determinism_payloads import (
    TIE_SCENARIOS,
    describe_differences,
    relational_available,
    run_payloads,
    snapshot_module,
    tie_frames,
)

HASH_SEEDS = ('0', '1', '2')
RUNS_IN_ONE_PROCESS = 10


@pytest.fixture(scope='module')
def fresh_runs(tmp_path_factory: pytest.TempPathFactory) -> list[dict]:
    tmp_path = tmp_path_factory.mktemp('repeat')
    return [
        run_payloads(
            tmp_path,
            f'seed-{seed}',
            'snapshot',
            '--repeat',
            str(RUNS_IN_ONE_PROCESS),
            hash_seed=seed,
        )
        for seed in HASH_SEEDS
    ]


def test_every_snapshot_scenario_is_covered(fresh_runs: list[dict]) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected = set(snapshot_module().collect())
    built = set(fresh_runs[0]['payloads'])
    assert expected <= built, sorted(expected - built)
    if relational_available():
        assert any(name.startswith('relational/') for name in built)
        for scenario in TIE_SCENARIOS:
            assert any(name.startswith(scenario) for name in built), scenario


def test_the_tie_fixture_has_ties_the_sampler_must_choose_among() -> None:
    _, orders = tie_frames()
    per_instant = orders.groupby(['user_id', 'ordered_at']).size()
    for _, options in TIE_SCENARIOS.values():
        assert min(options['num_neighbors']) < per_instant.min()
    assert orders['price'].is_unique


def test_ten_builds_in_one_process_are_identical(
    fresh_runs: list[dict],
) -> None:
    # The runner itself compares its ten builds and fails otherwise; reaching
    # here with three results means each process agreed with itself.
    assert len(fresh_runs) == len(HASH_SEEDS)
    assert all(run['payloads'] for run in fresh_runs)


def test_three_fresh_processes_build_identical_payloads(
    fresh_runs: list[dict],
) -> None:
    first = fresh_runs[0]['payloads']
    for seed, run in zip(HASH_SEEDS[1:], fresh_runs[1:], strict=True):
        differences = describe_differences(first, run['payloads'])
        assert not differences, (
            f'PYTHONHASHSEED={seed} built different payloads from '
            f'PYTHONHASHSEED={HASH_SEEDS[0]}:\n  ' + '\n  '.join(differences)
        )
