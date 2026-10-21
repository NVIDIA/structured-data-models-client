# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.16a fix 1 (Codex round 1, P2): ``heldout_run.sh`` runs each case on
every database ``--databases`` names, so an environment that cannot import
a named database's driver would fail its leg after the one-shot stamp.
The run refuses such an environment before the stamp: an ``--execute``
whose ``--python`` lacks Postgres's driver runs nothing and leaves no
stamp, and naming only the databases it can import clears that problem.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'scripts/rust_core/heldout_run.sh'
HELDOUT = ROOT / 'scripts/rust_core/heldout.py'
PROBLEM = 'cannot import a database driver (missing psycopg)'


def _python_without_psycopg(tmp_path: Path) -> Path:
    r"""A Python that answers as this one but cannot import psycopg."""
    shim = tmp_path / 'bin' / 'python'
    shim.parent.mkdir()
    shim.write_text(
        '#!/bin/sh\n'
        'for a in "$@"; do\n'
        '  case "$a" in psycopg) echo "missing psycopg" >&2; exit 1 ;; esac\n'
        'done\n'
        f'exec {sys.executable} "$@"\n'
    )
    shim.chmod(0o755)
    return shim


def _run(tmp_path: Path, python: Path, databases: str) -> tuple[str, Path]:
    bundle = tmp_path / 'bundle'
    bundle.mkdir(exist_ok=True)
    (bundle / 'h9.one.json').write_text('{"id": "h9.one"}\n')
    sys.path.insert(0, str(HELDOUT.parent))
    try:
        import heldout
    finally:
        sys.path.pop(0)
    stamp = tmp_path / f'stamp-{databases}'
    done = subprocess.run(
        [
            'bash',
            str(RUNNER),
            '--bundle',
            str(bundle),
            '--prefix',
            'h9.',
            '--bundle-sha',
            heldout.bundle_digest(bundle),
            '--inputs-sha',
            heldout.inputs_digest(bundle, 'h9.'),
            '--candidate',
            str(tmp_path / 'candidate'),
            '--reference',
            str(tmp_path / 'reference'),
            '--reference-sha',
            '0' * 40,
            '--reference-python',
            str(python),
            '--python',
            str(python),
            '--out',
            str(tmp_path / f'out-{databases}'),
            '--threshold',
            '4',
            '--databases',
            databases,
            '--stamp',
            str(stamp),
            '--execute',
        ],  # fmt: skip
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        timeout=120,
        env={
            'PATH': '/usr/bin:/bin',
            'HELDOUT_TOOL_PYTHON': sys.executable,
            'SDM_POSTGRES_TEST_DSN': 'set',
        },
        cwd=tmp_path,
    )
    return done.stdout + done.stderr, stamp


def test_an_environment_without_a_database_driver_is_refused_before_the_stamp(
    tmp_path,
):
    if shutil.which('bash') is None:
        pytest.skip('no bash')
    python = _python_without_psycopg(tmp_path)
    text, stamp = _run(tmp_path, python, 'duckdb,sqlite,postgres')
    assert f'the reference python {PROBLEM}' in text, text
    assert f'the candidate python {PROBLEM}' in text, text
    assert 'nothing run' in text
    assert '== 7.' not in text
    assert not stamp.exists()
    # Without Postgres named, its driver is not asked for.
    text, stamp = _run(tmp_path, python, 'duckdb')
    assert 'cannot import a database driver' not in text, text
    assert not stamp.exists()
