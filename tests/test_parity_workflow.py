# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live credentials and release acceptance cannot be bypassed by partial CI."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def workflow(name):
    return yaml.load(
        (ROOT / '.github/workflows' / name).read_text(), Loader=yaml.BaseLoader
    )


def test_live_credentials_only_on_trusted_refs():
    data = workflow('parity-live.yml')
    assert set(data['on']) == {'schedule', 'workflow_dispatch'}
    job = data['jobs']['parity']
    assert job['environment'] == 'parity-live'
    assert job['if'] == (
        "github.ref == 'refs/heads/main' || "
        "github.ref == 'refs/heads/rust-core-phase-4'"
    )
    assert data['permissions'] == {'contents': 'read'}
    evidence = next(
        s
        for s in job['steps']
        if s.get('with', {}).get('name', '').startswith('live-parity-')
    )
    assert evidence['with']['path'].endswith('/parity/evidence/')
    assert 'config.json' not in evidence['with']['path']


@pytest.mark.parametrize(
    'conclusion', ['success', 'failure', 'cancelled', 'missing', 'skipped']
)
def test_release_requires_success_for_exact_commit(tmp_path, conclusion):
    step = next(
        step
        for step in workflow('publish.yml')['jobs']['build']['steps']
        if step.get('name')
        == 'Require complete live parity for the release commit'
    )
    fake = tmp_path / 'gh'
    fake.write_text(
        '#!/bin/sh\n'
        'printf "%s\\n" "$@" > "$PARITY_ARGS"\n'
        'printf "%s\\n" "$PARITY_CONCLUSION"\n'
    )
    fake.chmod(0o755)
    args = tmp_path / 'args'
    result = subprocess.run(
        ['bash', '-c', step['run']],
        env={
            **os.environ,
            'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
            'GITHUB_SHA': 'exact-release-commit',
            'PARITY_ARGS': str(args),
            'PARITY_CONCLUSION': conclusion,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) == (conclusion == 'success')
    actual = args.read_text().splitlines()
    assert actual[actual.index('--workflow') + 1] == 'parity-live.yml'
    assert actual[actual.index('--commit') + 1] == 'exact-release-commit'
    assert actual[actual.index('--branch') + 1] == 'main'
    assert actual[actual.index('--status') + 1] == 'completed'
