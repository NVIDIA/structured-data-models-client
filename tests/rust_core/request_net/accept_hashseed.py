# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P8.hashseed: the corpus records identically under two ``PYTHONHASHSEED``
values.

- ``run_corpus.py record`` (every offline scenario, PostgreSQL through
  ``SDM_POSTGRES_TEST_DSN``) writes the same files, byte for byte, under
  ``PYTHONHASHSEED`` 1 and 4242, and they are the stored transcripts: every
  stored transcript of an offline backend is recorded, file for file, and
  nothing recorded is unstored.
- ``make_fixtures.py --check`` passes under both, so the generator writes
  the stored fixtures whatever the seed.
- The corpus replay and the fixture runner pass under both.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from _net import (
    LIVE_BACKENDS,
    MAKE_FIXTURES,
    PARITY,
    REPO,
    RUN_CORPUS,
    TRANSCRIPTS,
    transcripts,
)

SEEDS = ('1', '4242')


def _env(seed: str) -> dict[str, str]:
    env = dict(os.environ)
    env['PYTHONHASHSEED'] = seed
    return env


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob('*'))
        if p.is_file()
    }


def test_the_corpus_records_identically(tmp_path: Path) -> None:
    recorded: list[dict[str, bytes]] = []
    for seed in SEEDS:
        corpus = tmp_path / f'seed-{seed}'
        corpus.mkdir()
        proc = subprocess.run(
            [
                sys.executable,
                str(RUN_CORPUS),
                'record',
                '--corpus',
                str(corpus),
            ],
            cwd=REPO,
            env=_env(seed),
            capture_output=True,
            text=True,
            timeout=7200,
        )
        assert proc.returncode == 0, (
            f'record failed under PYTHONHASHSEED={seed}:\n'
            f'{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
        )
        recorded.append(_files(corpus))
    first, second = recorded
    assert len(first) > 100, f'only {len(first)} files recorded'
    differ = sorted(
        name
        for name in set(first) | set(second)
        if first.get(name) != second.get(name)
    )
    assert not differ, (
        f'{len(differ)} files differ between PYTHONHASHSEED 1 and 4242: '
        f'{differ[:20]}'
    )
    offline = {
        directory.relative_to(TRANSCRIPTS).as_posix()
        for directory, data in transcripts().values()
        if data.get('backend') not in LIVE_BACKENDS
    }
    stored = {
        name: data
        for name, data in _files(TRANSCRIPTS).items()
        if name.rsplit('/', 1)[0] in offline
    }
    unrecorded = sorted(set(stored) - set(first))
    assert not unrecorded, (
        f'stored offline files record does not write: {unrecorded[:20]}'
    )
    unstored = sorted(set(first) - set(stored))
    assert not unstored, f'recorded but not stored: {unstored[:20]}'
    stale = sorted(n for n in stored if stored[n] != first[n])
    assert not stale, (
        f'{len(stale)} stored files are not what record writes: {stale[:20]}'
    )


@pytest.mark.parametrize('seed', SEEDS)
def test_the_fixtures_generate_identically(seed: str) -> None:
    proc = subprocess.run(
        [sys.executable, str(MAKE_FIXTURES), '--check'],
        cwd=REPO,
        env=_env(seed),
        capture_output=True,
        text=True,
        timeout=3600,
    )
    assert proc.returncode == 0, (
        f'make_fixtures.py --check fails under PYTHONHASHSEED={seed}:\n'
        f'{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
    )


@pytest.mark.parametrize('seed', SEEDS)
def test_the_corpus_and_fixtures_replay(seed: str) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            str(PARITY / 'test_corpus.py'),
            str(PARITY / 'test_fixtures.py'),
            '-q',
            '-p',
            'no:cacheprovider',
            '-m',
            'not live_nim and not live_warehouse',
        ],
        cwd=REPO,
        env=_env(seed),
        capture_output=True,
        text=True,
        timeout=7200,
    )
    assert proc.returncode == 0, (
        f'the replay fails under PYTHONHASHSEED={seed}:\n'
        f'{proc.stdout[-4000:]}\n{proc.stderr[-2000:]}'
    )
