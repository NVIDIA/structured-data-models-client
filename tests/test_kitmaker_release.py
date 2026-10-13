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

import hashlib
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
    assert kitmaker.outstanding('a', '1.0.0', EXPECTED) == {}


def test_a_version_not_on_pypi_is_published_whole(kitmaker, monkeypatch):
    monkeypatch.setattr(kitmaker, 'on_pypi', lambda *a: None)
    assert kitmaker.outstanding('a', '1.0.0', EXPECTED) == EXPECTED


def test_a_partly_published_version_is_finished(kitmaker, monkeypatch):
    """PyPI takes files into a version one at a time, so the rest can still go."""
    monkeypatch.setattr(
        kitmaker, 'on_pypi', lambda *a: {'a-1.0.0-py3-none-any.whl': 'aaa'}
    )
    assert kitmaker.outstanding('a', '1.0.0', EXPECTED) == {
        'a-1.0.0.tar.gz': 'bbb'
    }


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
        kitmaker.outstanding('a', '1.0.0', EXPECTED)


def test_a_version_holding_files_we_did_not_build_is_refused(
    kitmaker, monkeypatch
):
    held = dict(EXPECTED) | {'a-1.0.0-cp312-cp312-linux_x86_64.whl': 'ccc'}
    monkeypatch.setattr(kitmaker, 'on_pypi', lambda *a: held)
    with pytest.raises(SystemExit, match='did not build'):
        kitmaker.outstanding('a', '1.0.0', EXPECTED)
    with pytest.raises(SystemExit, match='did not build'):
        kitmaker.verify('a', '1.0.0', EXPECTED)


def test_publishing_is_not_complete_until_pypi_holds_the_built_bytes(
    kitmaker, monkeypatch
):
    monkeypatch.setattr(kitmaker, 'on_pypi', lambda *a: None)
    monkeypatch.setattr(kitmaker.time, 'sleep', lambda _: None)
    monkeypatch.setattr(kitmaker, 'VERIFY_ATTEMPTS', 2)
    with pytest.raises(SystemExit, match='did not appear in full'):
        kitmaker.verify('a', '1.0.0', EXPECTED)


@pytest.fixture
def release(kitmaker, tmp_path, monkeypatch):
    """A whole release laid out the way the workflow hands it over, with every
    call to Portal and PyPI answered locally."""

    def arrange(*, already_on_pypi=()):
        dist = tmp_path / 'dist'
        dist.mkdir(exist_ok=True)
        lines = []
        for name in kitmaker.PROJECTS.values():
            filename = f'{name.replace("-", "_")}-1.0.0-py3-none-any.whl'
            (dist / filename).write_bytes(name.encode())
            lines.append(f'{filename}\thttps://example.invalid/{filename}')
        (tmp_path / 'assets.tsv').write_text('\n'.join(lines) + '\n')

        live = set(already_on_pypi)

        def on_pypi(name, version):
            if name not in live:
                return None
            filename = f'{name.replace("-", "_")}-1.0.0-py3-none-any.whl'
            return {filename: hashlib.sha256(name.encode()).hexdigest()}

        def submit(project_id, name, urls, upload):
            if upload:
                live.add(name)
            return 'uuid'

        monkeypatch.setattr(kitmaker, 'on_pypi', on_pypi)
        monkeypatch.setattr(kitmaker, 'submit', submit)
        monkeypatch.setattr(kitmaker, 'wait_for', lambda *a: None)
        monkeypatch.setattr(
            kitmaker,
            'call',
            lambda method, path, body=None: {'name': _name_for(kitmaker, path)},
        )
        monkeypatch.chdir(tmp_path)
        return tmp_path / 'kitmaker-summary.txt'

    return arrange


def _name_for(kitmaker, path):
    return kitmaker.PROJECTS[int(path.rsplit('/', 1)[1])]


def test_a_clean_release_reports_every_project_once(kitmaker, release):
    """Each project belongs in exactly one line of the summary: the record of
    what is live is what a half-finished release is read for."""
    summary = release()
    kitmaker.main()
    assert states(summary, kitmaker) == {
        name: 'published' for name in kitmaker.PROJECTS.values()
    }


def test_a_release_already_published_still_writes_a_summary(kitmaker, release):
    summary = release(already_on_pypi=set(kitmaker.PROJECTS.values()))
    kitmaker.main()
    assert states(summary, kitmaker) == {
        name: 'already published before this run'
        for name in kitmaker.PROJECTS.values()
    }


def test_a_project_live_but_unverified_is_not_reported_as_unpublished(
    kitmaker, release, monkeypatch
):
    """Portal reported success, so the files are on PyPI whatever verification
    finds. Saying otherwise would send the next person to republish something
    that is already there."""
    summary = release()
    monkeypatch.setattr(
        kitmaker,
        'verify',
        lambda *a: (_ for _ in ()).throw(SystemExit('bad bytes')),
    )
    with pytest.raises(SystemExit, match='bad bytes'):
        kitmaker.main()
    first = next(iter(kitmaker.PROJECTS.values()))
    assert states(summary, kitmaker)[first] == 'published'


def states(summary, kitmaker):
    """The summary as ``{project: state}``, asserting on the way that it names
    each project once and names nothing else. A count of lines does not say
    that: five lines naming one project would satisfy it."""
    lines = summary.read_text().strip().splitlines()
    parsed = dict(line.split(': ', 1) for line in lines)
    assert len(parsed) == len(lines), f'a project is named twice: {lines}'
    assert set(parsed) == set(kitmaker.PROJECTS.values())
    return parsed


def test_a_failure_in_pass_one_still_records_what_happened(
    kitmaker, release, monkeypatch
):
    """A release that stopped before publishing is exactly when the summary is
    read, and nothing was asked of PyPI, so nothing is claimed about it."""
    summary = release()
    names = list(kitmaker.PROJECTS.values())

    def mismatched(method, path, body=None):
        project_id = int(path.rsplit('/', 1)[1])
        if kitmaker.PROJECTS[project_id] == names[1]:
            return {'name': 'something-else'}
        return {'name': kitmaker.PROJECTS[project_id]}

    monkeypatch.setattr(kitmaker, 'call', mismatched)
    with pytest.raises(SystemExit, match='not'):
        kitmaker.main()

    assert states(summary, kitmaker) == {name: 'not reached' for name in names}


def test_a_preflight_failure_still_writes_a_summary(
    kitmaker, release, tmp_path
):
    """The checks that run before anything is submitted used to exit without
    one, so a release that stopped there said nothing at all."""
    summary = release()
    (tmp_path / 'dist' / 'structured_data_models_api-9.9.9.tar.gz').write_bytes(
        b'x'
    )
    with pytest.raises(SystemExit, match='disagree on the version'):
        kitmaker.main()
    assert states(summary, kitmaker) == {
        name: 'not reached' for name in kitmaker.PROJECTS.values()
    }


def test_every_project_is_accounted_for_exactly_once(kitmaker, release):
    """The one property the summary exists to have, across a mixed run."""
    names = list(kitmaker.PROJECTS.values())
    summary = release(already_on_pypi={names[0], names[2]})
    kitmaker.main()
    assert states(summary, kitmaker) == {
        names[0]: 'already published before this run',
        names[2]: 'already published before this run',
        **{name: 'published' for name in (names[1], names[3], names[4])},
    }


@pytest.mark.parametrize('status', ['failed', 'cancelled', 'error'])
def test_a_portal_failure_after_the_files_landed_says_they_are_live(
    kitmaker, release, monkeypatch, status
):
    """Portal's batches are not atomic, so a failure can follow files that
    already published. The summary asks PyPI rather than repeating the status,
    which is what stops the next person republishing something that is there."""
    names = list(kitmaker.PROJECTS.values())
    summary = release()
    real = kitmaker.wait_for

    def fail_after_publishing(uuid, what):
        if what == names[1]:
            raise SystemExit(f'{what} finished as {status} (uuid)')
        return real(uuid, what)

    monkeypatch.setattr(kitmaker, 'wait_for', fail_after_publishing)
    with pytest.raises(SystemExit, match=status):
        kitmaker.main()

    assert states(summary, kitmaker)[names[1]] == 'published'


def test_an_upload_that_published_nothing_says_so(
    kitmaker, release, monkeypatch
):
    """The other direction: Portal refused, PyPI has nothing, and the summary
    should not leave it ambiguous."""
    names = list(kitmaker.PROJECTS.values())
    summary = release()

    real = kitmaker.submit

    def refuse(project_id, name, urls, upload):
        if upload and name == names[1]:
            raise SystemExit('POST /releases returned 400: rejected')
        return real(project_id, name, urls, upload)

    monkeypatch.setattr(kitmaker, 'submit', refuse)
    with pytest.raises(SystemExit, match='400'):
        kitmaker.main()

    assert states(summary, kitmaker)[names[1]] == 'not on PyPI'


def test_a_partly_published_project_is_reported_as_partial(
    kitmaker, release, monkeypatch
):
    names = list(kitmaker.PROJECTS.values())
    summary = release()
    monkeypatch.setattr(kitmaker, 'verify', lambda *a: None)
    original = kitmaker.on_pypi

    def half(name, version):
        held = original(name, version)
        if held and name == names[1]:
            return {}
        return held

    monkeypatch.setattr(kitmaker, 'on_pypi', half)
    kitmaker.main()
    state = states(summary, kitmaker)[names[1]]
    assert state.startswith('PARTLY on PyPI, missing ')


def test_distributions_that_disagree_on_the_version_are_refused(
    kitmaker, release, tmp_path
):
    """The workflow gate reads the wheels and the sdists; this is what submits,
    so it reads every file too."""
    release()
    (tmp_path / 'dist' / 'structured_data_models_api-9.9.9.tar.gz').write_bytes(
        b'x'
    )
    with pytest.raises(SystemExit, match='disagree on the version'):
        kitmaker.main()


def test_the_summary_survives_pypi_answering_with_nonsense(
    kitmaker, release, monkeypatch
):
    """It runs from a finally while an exception is already on its way out, so
    raising here would replace the reason the release stopped and leave no
    summary either."""
    names = list(kitmaker.PROJECTS.values())
    summary = release()

    def rubbish(name, version):
        raise ValueError('Expecting value: line 1 column 1 (char 0)')

    def stop(uuid, what):
        # Only the publish, so pass one completes and the project is one the
        # summary has to describe rather than one nothing reached.
        if not what.startswith('dry run'):
            raise SystemExit('Portal fell over')

    monkeypatch.setattr(kitmaker, 'outstanding', lambda n, v, e: dict(e))
    monkeypatch.setattr(kitmaker, 'wait_for', stop)
    monkeypatch.setattr(kitmaker, 'on_pypi', rubbish)

    with pytest.raises(SystemExit, match='Portal fell over'):
        kitmaker.main()

    assert states(summary, kitmaker)[names[0]].startswith(
        'UNKNOWN, could not ask PyPI'
    )


def test_every_state_the_summary_can_print_is_documented(kitmaker):
    """RELEASING.md tells a releaser what to do per state, which only works if
    it lists them all."""
    releasing = (ROOT / 'RELEASING.md').read_text()
    source = SCRIPT.read_text()
    phrases = [
        'published',
        'already published before this run',
        'not reached',
        'not on PyPI',
        'PARTLY on PyPI, missing ',
        'on PyPI with UNEXPECTED bytes for ',
        'on PyPI with files this run did not build',
        'UNKNOWN, could not ask PyPI',
    ]
    for phrase in phrases:
        assert phrase in source, f'{phrase!r} is not a state the script emits'
        assert phrase.rstrip() in releasing, f'{phrase!r} is undocumented'
