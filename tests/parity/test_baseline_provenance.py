# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Historical comparison identifies the actual clean source being imported."""

import subprocess
import sys
import types

import pytest
from python_runner import Incomplete, artifact


def test_baseline_uses_requested_clean_checkout(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.check_output(
            ['git', '-C', str(tmp_path), *args], text=True
        ).strip()

    git('init', '-q')
    package = tmp_path / 'packages/structured-data-models-client/src/sdm_client'
    package.mkdir(parents=True)
    module = package / '__init__.py'
    module.write_text('')
    dependency = tmp_path / 'dependency.py'
    dependency.write_text('VALUE = 1\n')
    git('add', '.')
    git(
        '-c',
        'user.name=Test',
        '-c',
        'user.email=test@example.com',
        'commit',
        '-q',
        '--signoff',
        '-m',
        'Baseline fixture',
    )
    commit = git('rev-parse', 'HEAD')
    imported = types.ModuleType('sdm_client')
    imported.__file__ = str(module)
    monkeypatch.setitem(sys.modules, 'sdm_client', imported)
    artifact.cache_clear()
    try:
        observed = artifact(commit, 'baseline')
        assert observed['commit'] == commit
        assert observed['package_hash']
        with pytest.raises(Incomplete, match='requested clean commit'):
            artifact('0' * 40, 'baseline')
        # A changed dependency outside sdm_client also invalidates evidence.
        dependency.write_text('VALUE = 2\n')
        artifact.cache_clear()
        with pytest.raises(Incomplete, match='requested clean commit'):
            artifact(commit, 'baseline')
    finally:
        artifact.cache_clear()
