# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks on the release workflow, which no other suite can reach.

The publish path runs once per release, against a service this repository
cannot call from CI, so the parts that can be read statically are read here:
the order projects publish in, the states the poller accepts, and the two
properties of the workflow job that a reviewer had to notice by eye.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / '.github' / 'workflows' / 'publish.yml'
SCRIPT = ROOT / '.github' / 'workflows' / 'kitmaker_release.py'


@pytest.fixture(scope='module')
def kitmaker():
    """The publish script, which reads its credentials at import."""
    previous = {k: os.environ.get(k) for k in ('KITMAKER_API_TOKEN', 'PIC')}
    os.environ['KITMAKER_API_TOKEN'] = 'test-token'
    os.environ['PIC'] = 'nobody@nvidia.com'
    try:
        spec = importlib.util.spec_from_file_location(
            'kitmaker_release', SCRIPT
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules['kitmaker_release'] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        sys.modules.pop('kitmaker_release', None)


@pytest.fixture(scope='module')
def release_job():
    return yaml.safe_load(WORKFLOW.read_text())['jobs']['release']


def test_the_job_that_publishes_checks_out_the_script_it_runs(release_job):
    """It runs `kitmaker_release.py`, which ships in the repository and not in
    the build artifact, so without a checkout the step fails on a missing
    file."""
    steps = release_job['steps']
    checkout = next(
        (
            i
            for i, s in enumerate(steps)
            if 'actions/checkout' in s.get('uses', '')
        ),
        None,
    )
    assert checkout is not None, (
        'the publish job never checks out the repository'
    )
    download = next(
        i
        for i, s in enumerate(steps)
        if 'download-artifact' in s.get('uses', '')
    )
    assert checkout < download, (
        'checkout cleans the workspace, so it cannot follow the download'
    )


def test_nothing_is_published_or_released_outside_the_protected_environment(
    release_job,
):
    """The GitHub Release cannot be a draft, so creating it in an ungated job
    would publish it before anyone approved the upload."""
    assert release_job.get('environment') == 'pypi'
    run = ' '.join(s.get('run', '') for s in release_job['steps'])
    assert 'gh release create' in run
    assert 'kitmaker_release.py' in run


def test_an_attached_asset_is_never_replaced(release_job):
    """These builds are not byte reproducible, so clobbering would leave the
    GitHub asset and the PyPI file disagreeing."""
    commands = [
        line
        for step in release_job['steps']
        for line in step.get('run', '').splitlines()
        if not line.strip().startswith('#')
    ]
    assert not [line for line in commands if '--clobber' in line]


def test_the_client_publishes_after_everything_it_pins(kitmaker):
    """It requires api and pql exactly and reaches connectors and the engine
    through extras, so publishing it first would leave it live and
    uninstallable if a later project failed."""
    order = list(kitmaker.PROJECTS.values())
    assert order[-1] == 'structured-data-models-client'


def test_every_distribution_has_a_portal_project(kitmaker):
    packaged = {
        p.parent.name for p in (ROOT / 'packages').glob('*/pyproject.toml')
    }
    assert set(kitmaker.PROJECTS.values()) == packaged


@pytest.mark.parametrize(
    ('filename', 'distribution', 'version'),
    [
        (
            'structured_data_models_api-1.0.0-py3-none-any.whl',
            'structured_data_models_api',
            '1.0.0',
        ),
        (
            'structured_data_models_api-1.0.0.tar.gz',
            'structured_data_models_api',
            '1.0.0',
        ),
        (
            'structured_data_models_relational_engine-1.0.0-cp312-cp312-manylinux_2_28_x86_64.whl',
            'structured_data_models_relational_engine',
            '1.0.0',
        ),
    ],
)
def test_a_built_file_names_its_distribution_and_version(
    kitmaker, filename, distribution, version
):
    assert kitmaker.distribution_of(filename) == distribution
    assert kitmaker.version_of(filename) == version


def test_an_unrecognised_status_fails_instead_of_waiting(kitmaker, monkeypatch):
    """Polling an unexpected state to exhaustion reports a timeout, which
    describes the wrong problem and spends half an hour doing it."""
    monkeypatch.setattr(
        kitmaker, 'call', lambda *a, **k: {'status': 'rejected-by-policy'}
    )
    monkeypatch.setattr(kitmaker.time, 'sleep', lambda _: None)
    with pytest.raises(SystemExit, match='unrecognised status'):
        kitmaker.wait_for('some-uuid', 'publish api')


def test_a_known_failure_is_reported_as_itself(kitmaker, monkeypatch):
    monkeypatch.setattr(
        kitmaker, 'call', lambda *a, **k: {'status': 'rejected'}
    )
    monkeypatch.setattr(kitmaker.time, 'sleep', lambda _: None)
    with pytest.raises(SystemExit, match='finished as rejected'):
        kitmaker.wait_for('some-uuid', 'publish api')


EXPECTED = {'a-1.0.0-py3-none-any.whl': 'aaa', 'a-1.0.0.tar.gz': 'bbb'}


def test_a_version_already_published_is_skipped(kitmaker, monkeypatch):
    """What makes a rerun after a partial publish finish the rest."""
    monkeypatch.setattr(kitmaker, 'on_pypi', lambda *a: dict(EXPECTED))
    assert kitmaker.settled('a', '1.0.0', EXPECTED) is True


def test_a_version_not_on_pypi_is_work_to_do(kitmaker, monkeypatch):
    monkeypatch.setattr(kitmaker, 'on_pypi', lambda *a: None)
    assert kitmaker.settled('a', '1.0.0', EXPECTED) is False


def test_a_partly_published_version_is_refused(kitmaker, monkeypatch):
    """PyPI will not accept the rest of the files under a version it already
    has, so publishing over it cannot repair this."""
    monkeypatch.setattr(
        kitmaker, 'on_pypi', lambda *a: {'a-1.0.0-py3-none-any.whl': 'aaa'}
    )
    with pytest.raises(SystemExit, match='partly on PyPI'):
        kitmaker.settled('a', '1.0.0', EXPECTED)


def test_a_version_published_from_other_bytes_is_refused(kitmaker, monkeypatch):
    monkeypatch.setattr(
        kitmaker,
        'on_pypi',
        lambda *a: {
            'a-1.0.0-py3-none-any.whl': 'aaa',
            'a-1.0.0.tar.gz': 'different',
        },
    )
    with pytest.raises(SystemExit, match='different bytes'):
        kitmaker.settled('a', '1.0.0', EXPECTED)


def test_publishing_is_not_complete_until_pypi_holds_the_built_bytes(
    kitmaker, monkeypatch
):
    monkeypatch.setattr(kitmaker, 'on_pypi', lambda *a: None)
    monkeypatch.setattr(kitmaker.time, 'sleep', lambda _: None)
    monkeypatch.setattr(kitmaker, 'VERIFY_ATTEMPTS', 2)
    with pytest.raises(SystemExit, match='did not appear in full'):
        kitmaker.verify('a', '1.0.0', EXPECTED)
