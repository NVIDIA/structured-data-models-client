# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime CI cells install the database drivers their tests exercise."""

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


def test_runtime_cells_install_the_required_database_drivers():
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
    assert {'duckdb', 'sqlite', 'postgres'} <= set(next(iter(ci)))
