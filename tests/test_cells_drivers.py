# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``scripts/rust_core/cells.sh`` installs the connectors with the database
drivers CI's "Install the cell" step installs (T2.16): a local cell without
the SQLite and Postgres drivers cannot run the gate's database legs."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _extras(text: str) -> set[str]:
    found = re.findall(
        r'structured-data-models-connectors\[([a-z0-9,]+)\]', text
    )
    assert len(found) == 1, found
    return set(found[0].split(','))


def test_cells_install_the_drivers_ci_s_cells_install():
    workflow = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())
    steps = [
        step['run']
        for job in workflow['jobs'].values()
        for step in job.get('steps', [])
        if isinstance(step, dict) and step.get('name') == 'Install the cell'
    ]
    assert steps, 'no "Install the cell" step in ci.yml'
    ci = {frozenset(_extras(run)) for run in steps}
    assert len(ci) == 1, ci
    cells = _extras((ROOT / 'scripts/rust_core/cells.sh').read_text())
    assert cells == set(next(iter(ci)))
    assert {'duckdb', 'sqlite', 'postgres'} <= cells
