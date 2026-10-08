# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P3.leak: the no-write check finds the leak it was written for.

``test_no_writes.py`` must fail on the code phase 3 started from: a SQLite
push-down prediction there leaves ``tmp_*`` tables in the user's file. The
packages are taken from ``BASE`` (with this checkout's compiled engine
extension, which phase 3 does not change), put ahead of the installed ones,
and the SQLite checks run against them.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from _seam import BASE, REPO, SEAM, output

PACKAGES = (
    'sdm-api',
    'sdm-pql',
    'sdm-connectors',
    'sdm-client',
    'sdm-relational-engine',
)


def _base_sources(destination: Path) -> list[Path]:
    paths = [f'packages/{p}/src' for p in PACKAGES]
    archive = subprocess.run(
        ['git', 'archive', BASE, *paths],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ['tar', '-x', '-C', str(destination)], input=archive, check=True
    )
    engine = REPO / 'packages' / 'sdm-relational-engine' / 'src'
    for built in engine.rglob('*.so'):
        target = destination / built.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, target)
    for built in engine.rglob('*.pyd'):
        target = destination / built.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(built, target)
    return [destination / p for p in paths]


def test_the_sqlite_check_fails_on_base(tmp_path: Path) -> None:
    sources = _base_sources(tmp_path / 'base')
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(
        [str(p) for p in sources] + [env.get('PYTHONPATH', '')]
    ).rstrip(os.pathsep)

    where = subprocess.run(
        [
            sys.executable,
            '-c',
            'import sdm_relational_engine.relationallib, sdm_client, '
            'sdm_connectors; print(sdm_relational_engine.__file__); '
            'print(sdm_client.__file__); print(sdm_connectors.__file__)',
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert where.returncode == 0, output(where)
    for line in where.stdout.split():
        assert line.startswith(str(tmp_path / 'base')), (
            f'BASE code was not the code imported: {where.stdout}'
        )

    proc = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            str(SEAM / 'test_no_writes.py'),
            '-k',
            'sqlite_prediction_writes_nothing',
            '-p',
            'no:cacheprovider',
            '-rA',
            '--color=no',
            '-q',
        ],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode != 0, (
        f'the no-write check passed on BASE, which leaks:\n{output(proc)}'
    )
    for variant in ('autocommit', 'committed-afterwards'):
        assert (
            f'FAILED tests/seam/test_no_writes.py::'
            f'test_sqlite_prediction_writes_nothing[{variant}]' in proc.stdout
        ), f'{variant} did not fail on BASE:\n{output(proc)}'
    assert 'tmp_* objects left in the database' in proc.stdout, output(proc)
