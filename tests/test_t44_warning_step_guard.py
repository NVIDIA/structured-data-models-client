# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The step guard keeps immutable history and the consumed warning boundary."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCAN = Path(__file__).resolve().parents[1] / 'scripts/rust_core/l0_scan.py'
LEGACY = ['fetch', 'call_model', 'done']
CURRENT = ['warnings', *LEGACY]


def schema(steps):
    return {
        'oneOf': [
            {
                'properties': {'message': {'const': 'step'}},
                'oneOf': [
                    {'properties': {'step': {'const': step}}} for step in steps
                ],
            }
        ]
    }


@pytest.fixture
def tree(tmp_path):
    contract = tmp_path / 'rust/crates/sdm-contract'
    (contract / 'src').mkdir(parents=True)
    (contract / 'schema').mkdir()
    (contract / 'src/run.rs').write_text(
        'pub enum Step { Warnings, Fetch(FetchStep), '
        'CallModel(ModelCall), Done(OperationResult), }\n'
    )
    for version, steps in [(25, LEGACY), (32, LEGACY), (33, CURRENT)]:
        (contract / f'schema/sdm-contract-{version}.json').write_text(
            json.dumps(schema(steps))
        )
    return tmp_path


def scan(tree):
    return subprocess.run(
        [sys.executable, str(SCAN), 'steps', '--root', str(tree)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_warning_boundary_and_immutable_step_history_pass_together(tree):
    result = scan(tree)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('version', [25, 32, 33])
def test_a_schema_cannot_add_a_language_behavior_step(tree, version):
    path = tree / f'rust/crates/sdm-contract/schema/sdm-contract-{version}.json'
    steps = CURRENT if version == 33 else LEGACY
    path.write_text(json.dumps(schema([*steps, 'truthiness'])))
    result = scan(tree)
    assert result.returncode == 1
    assert path.name in result.stdout and 'truthiness' in result.stdout


@pytest.mark.parametrize('version', [25, 32])
def test_historical_schema_cannot_gain_the_warning_boundary(tree, version):
    path = tree / f'rust/crates/sdm-contract/schema/sdm-contract-{version}.json'
    path.write_text(json.dumps(schema(CURRENT)))
    result = scan(tree)
    assert result.returncode == 1
    assert path.name in result.stdout and 'warnings' in result.stdout


def test_current_schema_cannot_lose_the_consumed_warning_boundary(tree):
    path = tree / 'rust/crates/sdm-contract/schema/sdm-contract-33.json'
    path.write_text(json.dumps(schema(LEGACY)))
    result = scan(tree)
    assert result.returncode == 1
    assert path.name in result.stdout


def test_source_cannot_add_a_language_behavior_step(tree):
    path = tree / 'rust/crates/sdm-contract/src/run.rs'
    path.write_text(
        path.read_text().replace('Warnings,', 'Warnings, SortRows,')
    )
    result = scan(tree)
    assert result.returncode == 1
    assert 'sort_rows' in result.stdout
