# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P4.wheel: the locally built abi3 wheel installs into the phase venv and
passes the smoke test.

The wheel is built with maturin from this checkout, must be tagged for the
stable ABI from Python 3.10, and is installed as a wheel, not in editable
mode; ``smoke.py`` then runs in a fresh interpreter outside the checkout and
must report the installed module.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from _rc import (
    SMOKE,
    install_wheel,
    sha256,
    site_packages,
    smoke_lines,
    wheel_tags,
)
from _rc import build_local_wheel as build


def test_the_local_wheel_is_abi3_installs_and_passes_the_smoke_test():
    with tempfile.TemporaryDirectory(prefix='sdm-core-wheel-') as tmp:
        wheel = build(Path(tmp) / 'out')
        dist, py, abi, plats = wheel_tags(wheel.name)
        assert dist == 'sdm_core', wheel.name
        assert (py, abi) == ('cp310', 'abi3'), (
            f'{wheel.name} is not an abi3 wheel for Python 3.10 and later'
        )
        if sys.platform == 'darwin':
            assert all(p.startswith('macosx_') for p in plats), wheel.name
        install_wheel(wheel)

        # A directory with nothing importable in it, so only the installed
        # wheel can answer `import sdm_core`.
        proc = subprocess.run(
            [sys.executable, '-I', str(SMOKE)],
            cwd=tmp,
            capture_output=True,
            text=True,
        )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    lines = smoke_lines(proc.stdout)
    assert len(lines) == 1, proc.stdout
    line = lines[0]
    assert line['sha256'] == sha256(SMOKE)
    native = Path(line['native'])
    assert any(native.is_relative_to(p) for p in site_packages()), (
        f'sdm_core._native was imported from {native}, not from the wheel'
    )
    assert '.abi3.' in native.name, f'{native.name} is not an abi3 module'
