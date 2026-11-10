# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks on the release workflow, which no other suite can reach.

The publish path runs once per release, against a service only the release
workflow can reach through its tunnel, so the parts that can be read
statically are read here:
the order projects publish in, the states the poller accepts, and the two
properties of the workflow job that a reviewer had to notice by eye.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import subprocess
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
def workflow():
    return yaml.safe_load(WORKFLOW.read_text())


@pytest.fixture(scope='module')
def release_job(workflow):
    return workflow['jobs']['release']


@pytest.fixture(scope='module')
def kitmaker_job(workflow):
    return workflow['jobs']['kitmaker']


def test_the_job_that_publishes_checks_out_the_script_it_runs(kitmaker_job):
    """It runs `kitmaker_release.py`, which ships in the repository and not in
    the build artifact, so without a checkout the step fails on a missing
    file."""
    steps = kitmaker_job['steps']
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
    release_job, kitmaker_job
):
    """The GitHub Release cannot be a draft, so creating it in an ungated job
    would make it public before anyone approved, and the Portal token is an
    environment secret, so only an approved job can read it."""
    assert release_job.get('environment') == 'pypi'
    assert 'gh release create' in ' '.join(
        s.get('run', '') for s in release_job['steps']
    )
    assert kitmaker_job.get('environment') == 'kitmaker'
    assert kitmaker_job['needs'] == 'release'
    assert 'kitmaker_release.py' in ' '.join(
        s.get('run', '') for s in kitmaker_job['steps']
    )


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
    """It requires the api exactly, so publishing it first would leave it live
    and uninstallable if a later project failed."""
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
            'structured_data_models_client-2.0.0rc1-cp310-abi3-manylinux_2_28_x86_64.whl',
            'structured_data_models_client',
            '2.0.0rc1',
        ),
    ],
)
def test_a_built_file_names_its_distribution_and_version(
    kitmaker, filename, distribution, version
):
    assert kitmaker.distribution_of(filename) == distribution
    assert kitmaker.version_of(filename) == version


def test_the_states_are_the_ones_portal_documents(kitmaker):
    """Guessed state names were the original defect here. Portal documents
    exactly these five, and nothing is invented alongside them."""
    assert kitmaker.PENDING == {'pending', 'processing', 'in_progress'}
    assert kitmaker.TERMINAL_OK == {'completed'}
    assert kitmaker.TERMINAL_BAD == {'failed'}


def test_an_unrecognised_status_keeps_waiting_and_is_reported(
    kitmaker, monkeypatch, capsys
):
    """Portal's docs say to continue bounded polling on a state the client does
    not know, record it, and never read an unfamiliar nonterminal value as
    success. Failing fast on it would stop a release that was still running."""
    monkeypatch.setattr(
        kitmaker, 'call', lambda *a, **k: {'status': 'awaiting_approval'}
    )
    monkeypatch.setattr(kitmaker.time, 'sleep', lambda _: None)
    monkeypatch.setattr(kitmaker, 'POLL_ATTEMPTS', 3)
    with pytest.raises(SystemExit, match='never reached a terminal state'):
        kitmaker.wait_for('some-uuid', 'publish api')
    out = capsys.readouterr().out
    assert "unrecognised status 'awaiting_approval'" in out
    assert out.count('unrecognised status') == 1, 'reported once, not per poll'


def test_a_documented_failure_is_reported_as_itself(kitmaker, monkeypatch):
    monkeypatch.setattr(kitmaker, 'call', lambda *a, **k: {'status': 'failed'})
    monkeypatch.setattr(kitmaker.time, 'sleep', lambda _: None)
    with pytest.raises(SystemExit, match='finished as failed'):
        kitmaker.wait_for('some-uuid', 'publish api')


def test_an_sdist_goes_as_a_wheel_release_job_with_an_explicit_strategy(
    kitmaker, monkeypatch
):
    """`wheel-release-job` is the only job type Portal supports and it takes
    sdists. The strategy is sent explicitly because omitting it chooses by
    size, and the `wheel-stub` picked at or above 1 GiB cannot process one."""
    sent = {}

    def record(method, path, body=None):
        sent.update(body or {})
        return {'release_uuid': 'u'}

    monkeypatch.setattr(kitmaker, 'call', record)
    kitmaker.submit(1, 'a', ['https://example.invalid/a-1.0.0.tar.gz'], True)
    entry = sent['payload'][0]
    assert entry['job_type'] == 'wheel-release-job'
    assert entry['mirroring_strategy'] == 'upload-both'


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

    def files_of(name):
        """A wheel and an sdist each, as a real release has: with one file per
        project a part-published version could not be represented at all."""
        stem = name.replace('-', '_')
        return {
            f'{stem}-1.0.0-py3-none-any.whl': hashlib.sha256(
                f'{name}-whl'.encode()
            ).hexdigest(),
            f'{stem}-1.0.0.tar.gz': hashlib.sha256(
                f'{name}-sdist'.encode()
            ).hexdigest(),
        }

    def arrange(*, already_on_pypi=(), partly_on_pypi=()):
        dist = tmp_path / 'dist'
        dist.mkdir(exist_ok=True)
        lines = []
        for name in kitmaker.PROJECTS.values():
            for filename, digest in files_of(name).items():
                (dist / filename).write_bytes(
                    f'{name}-{"whl" if filename.endswith(".whl") else "sdist"}'.encode()
                )
                lines.append(
                    f'{filename}\thttps://example.invalid/{filename}\tsha256:{digest}'
                )
        (tmp_path / 'assets.tsv').write_text('\n'.join(lines) + '\n')

        live = set(already_on_pypi)
        half = set(partly_on_pypi)

        def on_pypi(name, version):
            if name in half:
                # The wheel landed, the sdist did not.
                return {
                    f: d
                    for f, d in files_of(name).items()
                    if f.endswith('.whl')
                }
            if name not in live:
                return None
            return files_of(name)

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
    # Pinned to this failure: `match='not'` also matched `not attached to the
    # release` and `dist/ holds no distributions`, so the test passed whatever
    # went wrong first.
    with pytest.raises(SystemExit, match="is 'something-else', not"):
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
        names[1]: 'published',
        names[2]: 'already published before this run',
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


@pytest.mark.parametrize(
    'failure',
    [
        ValueError('Expecting value: line 1 column 1 (char 0)'),
        # `SystemExit` is a `BaseException`, so `except Exception` misses it,
        # and it is what every read in this module raises when it gives up.
        SystemExit('PyPI structured-data-models-api 1.0.0: timed out'),
    ],
    ids=['malformed answer', 'read gave up'],
)
def test_the_summary_survives_pypi_failing(
    kitmaker, release, monkeypatch, failure
):
    """It runs from a finally while an exception is already on its way out, so
    raising here would replace the reason the release stopped and leave no
    summary either."""
    names = list(kitmaker.PROJECTS.values())
    summary = release()

    def rubbish(name, version):
        raise failure

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


def emitted_states():
    """Every state the summary can print, read off the source rather than
    listed here, so a new one cannot be added without this noticing."""
    tree = ast.parse(SCRIPT.read_text())
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }

    def literal(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            head = node.values[0]
            if isinstance(head, ast.Constant):
                return head.value
        return None

    states = set()
    for node in ast.walk(functions['pypi_state']):
        if isinstance(node, ast.Return) and node.value is not None:
            phrase = literal(node.value)
            assert phrase, (
                f'pypi_state returns something unreadable: {node.lineno}'
            )
            states.add(phrase)
    for node in ast.walk(functions['write_summary']):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == 'state' for t in node.targets
        ):
            phrase = literal(node.value)
            if phrase:
                states.add(phrase)
    return states


def documented_states():
    """The first column of the state table in RELEASING.md."""
    rows = re.findall(
        r'^\| `([^`]+)` \|', (ROOT / 'RELEASING.md').read_text(), re.MULTILINE
    )
    return set(rows)


def test_every_state_the_summary_can_print_is_documented():
    """RELEASING.md tells a releaser what to do per state, which only works if
    it lists them all, and misleads if it lists one that cannot happen."""
    emitted = emitted_states()
    documented = documented_states()
    assert len(emitted) >= 8, f'only found {emitted}: the reader is broken'

    def normalise(phrases):
        """The fixed part of each state, with the punctuation that introduces
        its variable tail removed. Compared as sets rather than by substring:
        `published` is inside `already published before this run`, which let a
        missing row pass."""
        return {
            phrase.rstrip()
            .removesuffix('[...]')
            .rstrip()
            .rstrip(':(,')
            .rstrip()
            .removesuffix(' for')
            .rstrip()
            for phrase in phrases
        }

    undocumented = normalise(emitted) - normalise(documented)
    assert not undocumented, (
        f'states RELEASING.md does not explain: {undocumented}'
    )

    unreachable = normalise(documented) - normalise(emitted)
    assert not unreachable, (
        f'RELEASING.md documents states that cannot happen: {unreachable}'
    )


def test_an_asset_that_is_not_what_this_run_built_is_refused(
    kitmaker, release, tmp_path
):
    """Kitmaker fetches the asset, not the local file, and a published file
    cannot be recalled. An asset left by an earlier run that built different
    bytes would be published as this release and only then fail verification,
    which is after the part that cannot be undone."""
    release()
    tsv = tmp_path / 'assets.tsv'
    rows = [line.split('\t') for line in tsv.read_text().strip().splitlines()]
    rows[1][2] = 'sha256:' + '0' * 64
    tsv.write_text('\n'.join('\t'.join(row) for row in rows) + '\n')

    with pytest.raises(SystemExit, match='not the bytes this run built'):
        kitmaker.main()


def test_an_asset_with_no_digest_is_refused(kitmaker, release, tmp_path):
    """An asset the API cannot vouch for is the dangerous case, not the safe
    one."""
    release()
    tsv = tmp_path / 'assets.tsv'
    rows = [line.split('\t') for line in tsv.read_text().strip().splitlines()]
    rows[1][2] = ''
    tsv.write_text('\n'.join('\t'.join(row) for row in rows) + '\n')

    with pytest.raises(SystemExit, match='no digest'):
        kitmaker.main()


def test_kitmaker_is_given_a_url_that_serves_the_file(kitmaker_job):
    """In the REST response `url` is the API metadata endpoint, which answers
    with JSON about the asset unless asked for octet-stream. Kitmaker is handed
    only the URL, so it has to be the download one."""
    commands = ' '.join(
        line
        for step in kitmaker_job['steps']
        for line in step.get('run', '').splitlines()
        if not line.strip().startswith('#')
    )
    assert 'assets.tsv' in commands
    assert '.browser_download_url' in commands
    assert '\\(.url)' not in commands


def test_the_release_is_not_latest_until_it_has_published(
    release_job, kitmaker_job
):
    """Kitmaker needs public asset URLs, so the release cannot be a draft and
    has to exist before the dry runs. A dry run that then failed would leave it
    standing as the current release for a version with nothing on PyPI."""
    create = next(
        s['run']
        for s in release_job['steps']
        if 'gh release create' in s.get('run', '')
    )
    assert '--latest=false' in create
    steps = kitmaker_job['steps']
    promote = next(
        i
        for i, s in enumerate(steps)
        if '--latest' in s.get('run', '')
        and 'gh release edit' in s.get('run', '')
    )
    publish = next(
        i
        for i, s in enumerate(steps)
        if s.get('run', '').rstrip().endswith('kitmaker_release.py')
    )
    assert publish < promote, 'Latest must be set after publishing'


def test_only_a_tag_publishes_and_main_only_checks(kitmaker_job):
    """Dispatched from `main`, the job asks Portal for each project and must
    not reach the step that submits a release."""
    assert "github.ref == 'refs/heads/main'" in kitmaker_job['if']
    by_name = {s.get('name'): s for s in kitmaker_job['steps']}
    check = by_name['Ask Portal for each project, publishing nothing']
    publish = by_name['Publish each distribution through Kitmaker']
    assert check['if'] == "github.ref_type != 'tag'"
    assert check['run'].endswith('kitmaker_release.py check')
    assert publish['if'] == "github.ref_type == 'tag'"


def test_two_runs_for_one_tag_cannot_race(release_job, workflow):
    """They would upload assets from two different builds to one release, and
    each would then refuse the other's files as stale."""
    group = workflow['concurrency']['group']
    assert 'github.ref' in group
    assert workflow['concurrency']['cancel-in-progress'] is False


def test_a_partial_project_is_described_even_if_the_run_stops_first(
    kitmaker, release, monkeypatch
):
    """PyPI is asked about every project before any dry run. Asking as each dry
    run came up meant a project an earlier run had left part-way went
    unexamined whenever an earlier project failed, and the summary then called
    it untouched."""
    names = list(kitmaker.PROJECTS.values())
    last = names[-1]
    summary = release(partly_on_pypi={last})

    monkeypatch.setattr(
        kitmaker,
        'call',
        lambda method, path, body=None: {'name': 'something-else'},
    )
    with pytest.raises(SystemExit, match="is 'something-else', not"):
        kitmaker.main()

    reported = states(summary, kitmaker)
    assert reported[names[0]] == 'not reached'
    assert reported[last].startswith('PARTLY on PyPI, missing '), (
        f'a version already partly on PyPI was reported as {reported[last]!r}'
    )


def test_a_portal_request_names_this_run_to_charon(kitmaker, monkeypatch):
    """Charon admits a request only with a GitHub OIDC token minted for its own
    audience, and the Portal token has to pass through it untouched."""
    monkeypatch.setenv(
        'ACTIONS_ID_TOKEN_REQUEST_URL', 'https://oidc.test/token?x=1'
    )
    monkeypatch.setenv('ACTIONS_ID_TOKEN_REQUEST_TOKEN', 'runner-token')
    sent = []

    class Answer(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def urlopen(request, timeout):
        sent.append(request)
        if request.full_url.startswith('https://oidc.test/'):
            return Answer(json.dumps({'value': 'run-identity'}).encode())
        return Answer(b'{"name": "structured-data-models-api"}')

    monkeypatch.setattr(kitmaker.urllib.request, 'urlopen', urlopen)
    kitmaker.call('GET', '/projects/6899')

    mint, portal = sent
    assert mint.full_url.endswith('&audience=charon.nvidia.com')
    assert mint.get_header('Authorization') == 'Bearer runner-token'
    assert portal.full_url == f'{kitmaker.API}/projects/6899'
    assert portal.full_url.startswith('http://127.0.0.1:8888/')
    assert portal.get_header('X-charon-gha-token') == 'run-identity'
    assert portal.get_header('Authorization') == 'Bearer test-token'


def test_a_portal_request_without_an_identity_is_refused(kitmaker, monkeypatch):
    monkeypatch.delenv('ACTIONS_ID_TOKEN_REQUEST_URL', raising=False)
    with pytest.raises(SystemExit, match='id-token: write'):
        kitmaker.charon_identity()


@pytest.mark.parametrize(
    ('records', 'refused'),
    [
        ({}, None),
        (
            {6901: {'name': 'structured-data-models-pql'}},
            "6901 is 'structured-data-models-pql'",
        ),
        (
            {6898: {'owner_email': 'someone@nvidia.com'}},
            'owned by someone other',
        ),
    ],
)
def test_the_check_refuses_a_project_unfit_to_publish_to(
    kitmaker, monkeypatch, records, refused
):
    def call(method, path):
        project_id = int(path.rsplit('/', 1)[1])
        record = {
            'name': kitmaker.PROJECTS[project_id],
            'owner_email': 'Nobody@nvidia.com',
        }
        return {**record, **records.get(project_id, {})}

    monkeypatch.setattr(kitmaker, 'call', call)
    if refused is None:
        kitmaker.check()
    else:
        with pytest.raises(SystemExit, match=re.escape(refused)):
            kitmaker.check()


@pytest.mark.parametrize(
    ('tag', 'releases', 'published', 'latest'),
    [
        ('v2.0.0', ['v2.0.0', 'v1.0.0'], {'2.0.0', '1.0.0'}, True),
        ('v1.0.1', ['v2.0.0', 'v1.0.1'], {'2.0.0', '1.0.1'}, False),
        ('v1.0.1', ['v2.0.0', 'v1.0.1'], {'1.0.1'}, True),
        ('v2.0.10', ['v2.0.10', 'v2.0.9'], {'2.0.10', '2.0.9'}, True),
        ('v2.0.0', ['v2.1.0rc1', 'v2.0.0'], {'2.1.0rc1', '2.0.0'}, True),
    ],
)
def test_latest_goes_to_the_highest_release_on_pypi(
    kitmaker, monkeypatch, tag, releases, published, latest
):
    """An older release finishing after a newer one must not take Latest back,
    and a newer one that is only on GitHub, waiting on its approval or failed
    in it, must not hold Latest back from one that published."""
    monkeypatch.setattr(
        kitmaker,
        'on_pypi',
        lambda name, version: {} if version in published else None,
    )
    assert kitmaker.stays_latest(tag, releases) is latest


@pytest.mark.skipif(shutil.which('bash') is None, reason='the step is bash')
@pytest.mark.parametrize(
    ('tag', 'verdict', 'marked'),
    [
        ('v2.0.0', 'latest', True),
        ('v1.0.1', 'v1.0.1 stays unmarked', False),
        ('v2.1.0rc1', 'latest', False),
    ],
)
def test_the_latest_step_marks_only_what_the_script_says(
    kitmaker_job, tmp_path, tag, verdict, marked
):
    step = next(
        s
        for s in kitmaker_job['steps']
        if s.get('name') == 'Mark the release Latest, now that it is published'
    )
    for name, body in {
        'gh': 'if [ "$2" = edit ]; then echo "$@" >> edited.txt; fi\n',
        'python3': 'echo "$VERDICT"\n',
    }.items():
        (tmp_path / name).write_text(f'#!/bin/sh\n{body}')
        (tmp_path / name).chmod(0o755)
    subprocess.run(
        ['bash', '-c', step['run']],
        cwd=tmp_path,
        check=True,
        env={
            'PATH': f'{tmp_path}{os.pathsep}{os.environ["PATH"]}',
            'GITHUB_REF_NAME': tag,
            'GITHUB_REPOSITORY': 'NVIDIA/structured-data-models-client',
            'VERDICT': verdict,
        },
    )
    edited = tmp_path / 'edited.txt'
    assert edited.exists() is marked
    if marked:
        assert '--latest' in edited.read_text()
