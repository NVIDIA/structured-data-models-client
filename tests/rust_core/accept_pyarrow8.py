# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P4.pyarrow8: the smoke test passes on Python 3.10 with pyarrow 8.0.0.

That is Databricks Runtime 14.3 LTS and serverless environment 1: Python
3.10, pyarrow 8.0.0 and numpy 1.23.5, with no PyCapsule interface, so Arrow
crosses through ``_export_to_c`` and ``_import_from_c``. The manylinux wheel
of each architecture is installed into a ``python:3.10-slim`` container with
those versions and runs ``smoke.py``.
"""

from __future__ import annotations

import subprocess

import pytest
from _rc import SMOKE, linux_wheels, sha256, smoke_lines, wheel_by_arch

IMAGE = 'python:3.10-slim'
PLATFORMS = {'x86_64': 'linux/amd64', 'aarch64': 'linux/arm64'}


@pytest.fixture(scope='module')
def wheels():
    built, _ = linux_wheels()
    return wheel_by_arch(built)


@pytest.mark.parametrize('arch', sorted(PLATFORMS))
def test_the_smoke_test_passes_with_pyarrow_8(wheels, arch):
    assert arch in wheels, f'no {arch} wheel was built'
    wheel = wheels[arch]
    script = (
        'set -e; '
        'pip install -q --disable-pip-version-check --no-cache-dir '
        f"'numpy==1.23.5' 'pyarrow==8.0.0' /wheels/{wheel.name}; "
        'python -c "import pyarrow; assert pyarrow.__version__ == \'8.0.0\'"; '
        'python /smoke/smoke.py'
    )
    proc = subprocess.run(
        [
            'docker',
            'run',
            '--rm',
            '--platform',
            PLATFORMS[arch],
            '-v',
            f'{wheel.parent}:/wheels:ro',
            '-v',
            f'{SMOKE.parent}:/smoke:ro',
            IMAGE,
            'sh',
            '-c',
            script,
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-6000:]
    lines = smoke_lines(proc.stdout)
    assert len(lines) == 1, proc.stdout
    line = lines[0]
    assert line['sha256'] == sha256(SMOKE)
    assert line['machine'] == arch
    assert line['pyarrow'] == '8.0.0'
    assert line['python'].startswith('3.10.')
