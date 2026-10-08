# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Brief task 5 (packaging): CI installs ``sdm-core`` wherever it tests
``sdm-client``, publishing builds it, the release contract knows it, and the
prerequisites say Windows is not built (DV29).

A job "installs ``sdm-core``" when its definition, with what it extends
resolved, names the binding (``sdm-core``, ``sdm_core`` or
``rust/bindings/python``) or the jobs and directories that build its wheels
(``rust-wheels``, ``dist-rust``). The wheels themselves, and a clean install
of ``sdm-client`` with them, are P9.wheels'.
"""

from __future__ import annotations

import json
import re

import pytest
import yaml
from _switch import REPO

NAMES = (
    'sdm-core',
    'sdm_core',
    'rust/bindings/python',
    'rust-wheels',
    'dist-rust',
)
CLIENT_TESTS = 'packages/structured-data-models-client/tests'


def _names_the_core(job) -> bool:
    text = json.dumps(job)
    return any(name in text for name in NAMES)


def _gitlab_jobs():
    data = yaml.safe_load((REPO / '.gitlab-ci.yml').read_text())

    def resolved(job):
        parents = job.get('extends') or []
        parents = [parents] if isinstance(parents, str) else parents
        merged = {}
        for parent in parents:
            merged.update(resolved(data.get(parent) or {}))
        merged.update(job)
        return merged

    return {
        name: resolved(job)
        for name, job in data.items()
        if isinstance(job, dict)
        and not name.startswith('.')
        and 'script' in json.dumps(job)
    }


def test_every_gitlab_job_testing_sdm_client_installs_the_core():
    jobs = {
        n: j for n, j in _gitlab_jobs().items() if CLIENT_TESTS in json.dumps(j)
    }
    assert jobs, 'no GitLab job runs the sdm-client tests'
    missing = sorted(n for n, j in jobs.items() if not _names_the_core(j))
    assert not missing, (
        f'GitLab jobs testing sdm-client without sdm-core: {missing}'
    )


def test_every_github_job_testing_sdm_client_installs_the_core():
    workflow = yaml.safe_load(
        (REPO / '.github' / 'workflows' / 'ci.yml').read_text()
    )
    jobs = {
        n: j
        for n, j in workflow['jobs'].items()
        if CLIENT_TESTS in json.dumps(j)
    }
    assert jobs, 'no GitHub job runs the sdm-client tests'
    missing = sorted(n for n, j in jobs.items() if not _names_the_core(j))
    assert not missing, (
        f'GitHub jobs testing sdm-client without sdm-core: {missing}'
    )


def test_publishing_builds_the_core_for_linux_and_macos():
    text = (REPO / '.github' / 'workflows' / 'publish.yml').read_text()
    assert 'rust/bindings/python' in text, 'publish.yml does not build sdm-core'
    assert re.search(r'macos', text), 'publish.yml builds nothing on macOS'


def test_the_release_contract_knows_the_core():
    text = (REPO / 'tests' / 'test_release_contract.py').read_text()
    assert any(
        name in text
        for name in ('sdm-core', 'sdm_core', 'rust/bindings/python')
    ), 'tests/test_release_contract.py never names sdm-core'


@pytest.mark.parametrize('page', ['docs/get-started/prerequisites.md'])
def test_the_prerequisites_say_windows_is_not_built(page):
    text = (REPO / page).read_text()
    assert re.search(r'\bWindows\b', text), f'{page} does not mention Windows'
