# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The client suites keep their candidate dependencies and bounded fixtures."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_the_general_client_matrix_has_the_legacy_route_dependency() -> None:
    jobs = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())[
        'jobs'
    ]
    job = jobs['test']
    commands = '\n'.join(step.get('run', '') for step in job['steps'])
    assert '[test,relational]' in commands
    assert 'packages/structured-data-models-relational-engine' in commands


@pytest.mark.parametrize('name', ['installed-wheel', 'forced-routes'])
def test_legacy_client_suites_install_the_candidate_engine(name: str) -> None:
    jobs = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())[
        'jobs'
    ]
    job = jobs[name]
    commands = '\n'.join(step.get('run', '') for step in job['steps'])
    assert 'packages/structured-data-models-relational-engine' in commands
    extras = re.findall(r'structured-data-models-client\[([^]]+)\]', commands)
    assert any('relational' in value.split(',') for value in extras)


def test_runtime_client_tests_install_the_schema_validator() -> None:
    jobs = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text())[
        'jobs'
    ]
    step = next(
        step
        for step in jobs['runtime-install']['steps']
        if step.get('name')
        == "The binding's and the client's tests in the runtime"
    )
    assert re.search(r'\bjsonschema\b', step['run'])


@pytest.mark.tooling
def test_passing_temp_trees_are_removed_and_failed_ones_preserved(
    tmp_path: Path,
) -> None:
    test = tmp_path / 'test_retention.py'
    test.write_text(
        'from pathlib import Path\n'
        f'MARKER = Path({str(tmp_path / "passed.txt")!r})\n'
        f'FAILED = Path({str(tmp_path / "failed.txt")!r})\n'
        'def test_pass(tmp_path):\n'
        '    MARKER.write_text(str(tmp_path))\n'
        'def test_passed_tree_was_removed():\n'
        '    assert not Path(MARKER.read_text()).exists()\n'
        'def test_failure_retains_its_tree(tmp_path):\n'
        '    FAILED.write_text(str(tmp_path))\n'
        '    assert False, "intentional retention control"\n'
    )
    done = subprocess.run(
        [
            sys.executable,
            '-m',
            'pytest',
            '-c',
            str(ROOT / 'pyproject.toml'),
            '-p',
            'no:cacheprovider',
            '--basetemp',
            str(tmp_path / 'run'),
            '-q',
            str(test),
        ],
        env={
            **{
                name: value
                for name, value in os.environ.items()
                if not name.startswith('PYTEST_')
            },
            'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert '1 failed, 2 passed' in done.stdout, done.stdout + done.stderr
    assert Path((tmp_path / 'failed.txt').read_text()).is_dir()
