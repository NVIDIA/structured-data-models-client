# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A seeded request over tables left in SQLite or DuckDB repeats exactly.

The tables are built in each process from the same in-repo frames, then
sampled where they are, with SQL. Three fresh processes, each with its own
string-hash seed, must build parsed-equal payloads. SQLite drew its context
with ``ORDER BY RANDOM()``, which takes no seed, and a query without
``ORDER BY`` returns rows in whatever order the plan produces; either one
fails here.
"""

from __future__ import annotations

import pytest
from determinism_payloads import (
    PUSHDOWN_SCENARIOS,
    describe_differences,
    run_payloads,
)

HASH_SEEDS = ('0', '1', '2')

_DRIVERS = {'sqlite': 'adbc_driver_sqlite', 'duckdb': 'duckdb'}


@pytest.mark.parametrize('backend', ['sqlite', 'duckdb'])
def test_three_fresh_processes_build_identical_payloads(
    backend: str, tmp_path_factory: pytest.TempPathFactory
) -> None:
    pytest.importorskip('sdm_relational_engine')
    pytest.importorskip(_DRIVERS[backend])
    tmp_path = tmp_path_factory.mktemp(f'pushdown-{backend}')

    runs = [
        run_payloads(
            tmp_path,
            f'{backend}-{seed}',
            'pushdown',
            '--backend',
            backend,
            '--workdir',
            str(tmp_path / f'process-{seed}'),
            hash_seed=seed,
        )
        for seed in HASH_SEEDS
    ]

    first = runs[0]['payloads']
    for scenario in PUSHDOWN_SCENARIOS:
        assert any(name.startswith(scenario) for name in first), scenario
    for name, payloads in first.items():
        rows = payloads[0]['context']['instance_table']['rows']
        assert rows, f'{name} gathered no context; nothing was compared'
    for seed, run in zip(HASH_SEEDS[1:], runs[1:], strict=True):
        differences = describe_differences(first, run['payloads'])
        assert not differences, (
            f'{backend}: PYTHONHASHSEED={seed} built different payloads from '
            f'PYTHONHASHSEED={HASH_SEEDS[0]}:\n  ' + '\n  '.join(differences)
        )
