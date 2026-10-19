# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.0h: the floor cell at its new pins.

T2.0c moves the floor CI cell to pandas 2.2.2 and NumPy 1.23.5 (KO2-37).
The runner names an environment's cell from its versions; one it does not
recognise records no cell, and compare then holds it to the pandas-3
reference. Both the new pins and the earlier ones (which the phase
branch's ci.yml installs until T2.0c lands) are the floor cell.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parents[2] / 'contract/runners/run_python.py'
spec = importlib.util.spec_from_file_location('run_python', RUNNER)
assert spec is not None and spec.loader is not None
run_python = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_python)


@pytest.mark.parametrize(
    ('pandas', 'numpy'), [('2.2.2', '1.23.5'), ('2.2.0', '1.26.4')]
)
def test_the_floor_pins_are_the_floor_cell(pandas: str, numpy: str) -> None:
    versions = {
        'python': '3.10.14',
        'pandas': pandas,
        'pyarrow': '14.0.1',
        'numpy': numpy,
    }
    assert run_python._cell(versions) == 'floor-pandas2.2'
    assert run_python._cell({**versions, 'numpy': '1.24.0'}) is None
