# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CI's cross-host comparison (T1.29; design, "Observations and
comparator"), its step's text run as CI runs it (``bash -e``).

``cross-host`` takes python-core's observations from contract-python
(ref-pandas3) and rust-host's from rust-test and runs ``compare.py`` over
both, so the ``core`` rule gates a difference between the two hosts on
every acceptance case in CI. It compares two executions of one core, so a
change both hosts make alike passes it; ``contract/pinned/base.regression.core.json``,
the fixed expectation both hosts are held to, catches that and stays
beside it (E81(c)). Each claim and what holds it:

- The job needs both producing jobs and downloads the artifacts they
  upload, by name.
- Positive control: python-core's observations of every case and a
  rust-host stand-in made from them (what ``contract/availability.yaml``
  makes unavailable to rust-host left out, each statement read as Arrow, as
  ``sdm-testkit`` reads it) pass the step.
- One-sided faults on either host's side fail the step by the ``core``
  rule on an acceptance case: the operations started, their inputs and the
  acceptance, on the first acceptance case both hosts run on the core (an
  S0 case); and the operations, inputs, fetches (the step trace),
  description, request state, call, result and the request id's type of
  ``base.regression``, and the operations of the other three baseline
  cases, gated where the case is an acceptance case and reported while it
  is diagnostic (T1.29's part 1 keeps them diagnostic; part 2 admits
  them). The request id's value alone (E35) passes, and a diagnostic
  case's difference is reported, not gated.
- Fail-closed: either file missing, empty, not JSON, nested past any
  parser, naming no runner or another runner's fails the step before any
  comparison, and so does ``ratchet.py where`` or ``effective`` failing.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.tooling

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / '.github/workflows/ci.yml'
JOB = 'cross-host'


def _jobs() -> dict[str, Any]:
    jobs: dict[str, Any] = yaml.safe_load(CI.read_text(encoding='utf-8'))[
        'jobs'
    ]
    return jobs


def _step() -> dict[str, Any]:
    [step] = [
        each
        for each in _jobs()[JOB]['steps']
        if str(each.get('name', '')).startswith('compare (')
    ]
    return step


def _uploads(job: dict[str, Any]) -> dict[str, str]:
    return {
        each['with']['name']: each['with']['path']
        for each in job['steps']
        if str(each.get('uses', '')).startswith('actions/upload-artifact@')
    }


def test_the_job_takes_both_hosts_observations_by_name() -> None:
    jobs = _jobs()
    job = jobs[JOB]
    # The landing job, for its own condition on a landing commit (E131).
    assert set(job['needs']) == {
        'landing',
        'rust-test',
        'contract-python-ref-pandas3',
    }
    downloads = {
        each['with']['name']: each['with']['path']
        for each in job['steps']
        if str(each.get('uses', '')).startswith('actions/download-artifact@')
    }
    assert downloads == {
        'observations-python-core': 'cross-host',
        'observations-rust-host': 'cross-host',
    }
    assert (
        _uploads(jobs['rust-test'])['observations-rust-host']
        == 'rust-host.jsonl'
    )
    assert (
        _uploads(jobs['contract-python-ref-pandas3'])[
            'observations-python-core'
        ]
        == 'cross-host/python-core.jsonl'
    )
    for name in ('rust-test', 'contract-python-ref-pandas3'):
        for each in jobs[name]['steps']:
            if str(each.get('uses', '')).startswith('actions/upload-artifact@'):
                assert each['with']['if-no-files-found'] == 'error'


@pytest.fixture(scope='module')
def observed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, list[Any]]:
    r"""python-core's observations of every case, and the rust-host
    stand-in made from them."""
    out = tmp_path_factory.mktemp('observed') / 'python-core.jsonl'
    done = subprocess.run(
        [
            sys.executable,
            '-I',
            str(ROOT / 'contract/runners/run_python.py'),
            '--mode',
            'python-core',
            '--out',
            str(out),
            str(ROOT / 'contract/cases'),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    core = [json.loads(line) for line in out.read_text().splitlines()]
    return {'python-core': core, 'rust-host': _rust_host(core)}


def _rust_host(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    available = yaml.safe_load(
        (ROOT / 'contract/availability.yaml').read_text(encoding='utf-8')
    )['runners']['rust-host']
    made = []
    for line in copy.deepcopy(lines):
        identity = line['identity']
        identity.update(
            runner='rust-host', cell=None, native=None, native_loaded=None
        )
        identity['versions'] = {'duckdb': identity['versions']['duckdb']}
        # The endpoint's refusal of a call (python-core's code-less
        # NimRequestError at predict, its core's INVALID_RESPONSE): the
        # pure-Rust host has no transport or client class, so it reports
        # its core's code for that call (T2.1g, register P2G-1).
        outcome = line['outcome']
        error = (line.get('core') or {}).get('error') or {}
        if (
            outcome['result'] == 'refusal'
            and outcome['stage'] == 'predict'
            and outcome['error'] == 'NimRequestError'
            and outcome['code'] is None
            and error.get('code') == 'INVALID_RESPONSE'
        ):
            outcome['code'] = 'INVALID_RESPONSE'
        for field, state in available.items():
            if state != 'unavailable':
                continue
            if field == 'public':
                line.pop('public', None)
            elif '.' in field:
                section, key = field.split('.')
                line[section][key] = None
            else:
                line[field] = None
        capture = (line.get('driver') or {}).get('capture') or {}
        for statement in capture.get('statements', []):
            for read in statement.get('results', []):
                read['method'] = 'query_arrow'
                read['values']['types'] = None
                read.pop('digest', None)
        made.append(line)
    return made


def _executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


#: What ``ratchet.py where`` prints on a lane of the phase branch.
LANE = '--base origin/rust-core-phase-1'


def _run(
    tmp_path: Path,
    files: dict[str, str | None],
    where: str | None = '',
    lane: bool = False,
    cases: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    r"""The step's text in a scratch directory holding the repository's
    contract and scripts, with ``files`` as the downloaded artifacts (None:
    not downloaded). ``ratchet.py where`` and ``effective`` are stubbed:
    ``where`` prints ``where`` (None: it fails), empty as off the phase
    branch, and ``effective`` records its call and fails, so a run with a
    base shows the step asks it and stops when it fails. With ``lane``, the
    step runs as on a lane: ``where`` prints :data:`LANE` and the real
    ``effective`` writes the gate's effective expectations from the phase
    branch's checkpoint (CI checks out every branch), so an admitted case's
    approved outcome is held as the gate holds it."""
    if lane:
        where = LANE
    work = tmp_path / 'work'
    work.mkdir(parents=True)
    for name in ('scripts', 'docs', 'rust'):
        (work / name).symlink_to(ROOT / name)
    if cases is None:
        (work / 'contract').symlink_to(ROOT / 'contract')
    else:
        # The repository's contract with `cases` in place of its cases.
        (work / 'contract').mkdir()
        for child in (ROOT / 'contract').iterdir():
            target = cases if child.name == 'cases' else child
            (work / 'contract' / child.name).symlink_to(target)
    (work / 'cross-host').mkdir()
    for name, text in files.items():
        if text is not None:
            (work / 'cross-host' / name).write_text(text, encoding='utf-8')
    bin_dir = tmp_path / 'bin'
    _executable(bin_dir / 'pip', '#!/usr/bin/env bash\nexit 0\n')
    answer = 'exit 1' if where is None else f'printf "%s\\n" {where!r}; exit 0'
    ratchet = '[ "$1" = scripts/rust_core/ratchet.py ]'
    called = tmp_path / 'effective-called'
    # On a lane, the real effective, on this checkout's history (the step
    # runs at the checkout's root in CI; the scratch directory is no
    # repository).
    effective = (
        f'if {ratchet} && [ "$2" = effective ]; then '
        f'exec {sys.executable} "$@" --repo {str(ROOT)!r}; fi\n'
        if lane
        else f'if {ratchet} && [ "$2" = effective ]; then '
        f'echo "$@" > {str(called)!r}; exit 1; fi\n'
    )
    _executable(
        bin_dir / 'python',
        '#!/usr/bin/env bash\n'
        f'if {ratchet} && [ "$2" = where ]; then {answer}; fi\n'
        f'{effective}'
        f'exec {sys.executable} "$@"\n',
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith('GITHUB_')
    }
    env['PATH'] = f'{bin_dir}{os.pathsep}{env["PATH"]}'
    return subprocess.run(
        ['bash', '--noprofile', '--norc', '-e', '-c', _step()['run']],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _text(lines: list[dict[str, Any]]) -> str:
    return ''.join(json.dumps(line) + '\n' for line in lines)


def _files(observed: dict[str, list[Any]]) -> dict[str, str | None]:
    return {
        'python-core.jsonl': _text(observed['python-core']),
        'rust-host.jsonl': _text(observed['rust-host']),
    }


def test_both_hosts_agreeing_pass(
    tmp_path: Path, observed: dict[str, list[Any]]
) -> None:
    done = _run(tmp_path, _files(observed), lane=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'cross-host: python-core and rust-host observed' in done.stdout
    assert 'compare: no differences' in done.stdout


def _predicted(line: dict[str, Any]) -> dict[str, Any]:
    predicted: dict[str, Any] = line['core']['predicted']
    return predicted


def _first_cell(rows: list[Any]) -> None:
    cell = rows[0][0]
    key = next(iter(cell))
    cell[key] = '7' if cell[key] != '7' else '8'


#: The failures the pin caught, each a change to one host's core fields.
SEEDS: dict[str, Callable[[dict[str, Any]], None]] = {
    'operations': lambda o: o['identity']['operations'].pop(),
    'inputs': lambda o: o['core']['inputs'][1].update(seed='0x1'),
    'fetches': lambda o: o['core']['fetches'].pop(),
    'described': lambda o: o['core']['described']['tables'][0]['columns'].pop(),
    'materialized': lambda o: o['core']['materialized'].update(version=99),
    'call': lambda o: _predicted(o)['body'].update(model='another'),
    'result': lambda o: _first_cell(_predicted(o)['result']['rows']),
    # A number where the call holds text: payloads are compared parsed,
    # each number as {"number": text}.
    'request id type': lambda o: _predicted(o)['body']['metadata'].update(
        request_id={'number': '1'}
    ),
}


def _seeded(
    observed: dict[str, list[Any]], side: str, case: str, seed: str
) -> dict[str, str | None]:
    changed = copy.deepcopy(observed)
    [line] = [each for each in changed[side] if each['case'] == case]
    SEEDS[seed](line)
    return _files(changed)


def _status(case: str) -> str:
    for path in ROOT.glob('contract/cases/*/*.json'):
        found = json.loads(path.read_text(encoding='utf-8'))
        if found['id'] == case:
            status: str = found['status']
            return status
    raise AssertionError(case)


def _held(done: subprocess.CompletedProcess[str], case: str) -> None:
    r"""The step gates the seeded difference where the case is an
    acceptance case, and reports it where it is diagnostic (T1.29's part 1
    keeps the baseline cases diagnostic; part 2 admits them)."""
    if _status(case) == 'acceptance':
        assert done.returncode != 0, done.stdout
        assert f'DIFFERS core: {case}: python-core against rust-host' in (
            done.stdout
        )
    else:
        assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize('seed', sorted(SEEDS))
@pytest.mark.parametrize('side', ['python-core', 'rust-host'])
def test_each_one_sided_fault_of_base_regression_is_held(
    tmp_path: Path, observed: dict[str, list[Any]], side: str, seed: str
) -> None:
    _held(
        _run(tmp_path, _seeded(observed, side, 'base.regression', seed)),
        'base.regression',
    )


#: One-sided faults of a case whose core operation is validate_query.
VALIDATION_SEEDS: dict[str, Callable[[dict[str, Any]], None]] = {
    'operations': lambda o: o['identity']['operations'].append('materialize'),
    'inputs': lambda o: o['core']['inputs'][0].update(
        query=o['core']['inputs'][0]['query'] + ' '
    ),
    'accepted': lambda o: o['core'].update(accepted=not o['core']['accepted']),
}


@pytest.mark.parametrize('seed', sorted(VALIDATION_SEEDS))
@pytest.mark.parametrize('side', ['python-core', 'rust-host'])
def test_an_acceptance_case_s_one_sided_fault_fails_the_step(
    tmp_path: Path, observed: dict[str, list[Any]], side: str, seed: str
) -> None:
    # The gate's own path, on an acceptance case in every part: the first
    # acceptance case both hosts ran on the core.
    [case, *_] = [
        each['case']
        for each in observed['python-core']
        if _status(each['case']) == 'acceptance'
        and each['core'] is not None
        and each['core'].get('accepted') is not None
    ]
    changed = copy.deepcopy(observed)
    [line] = [each for each in changed[side] if each['case'] == case]
    VALIDATION_SEEDS[seed](line)
    done = _run(tmp_path, _files(changed))
    assert done.returncode != 0, done.stdout
    assert f'DIFFERS core: {case}: python-core against rust-host' in done.stdout


def test_the_request_id_s_value_alone_passes(
    tmp_path: Path, observed: dict[str, list[Any]]
) -> None:
    changed = copy.deepcopy(observed)
    [line] = [
        each
        for each in changed['rust-host']
        if each['case'] == 'base.regression'
    ]
    _predicted(line)['body']['metadata']['request_id'] = 'another-id'
    done = _run(tmp_path, _files(changed), lane=True)
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_other_baseline_cases_are_held_by_their_status(
    tmp_path: Path, observed: dict[str, list[Any]]
) -> None:
    for case in ('base.binary', 'base.multiclass', 'base.adventureworks'):
        changed = copy.deepcopy(observed)
        [line] = [each for each in changed['rust-host'] if each['case'] == case]
        VALIDATION_SEEDS['operations'](line)
        _held(_run(tmp_path / case, _files(changed)), case)


def test_a_diagnostic_case_s_difference_is_reported_not_gated(
    tmp_path: Path, observed: dict[str, list[Any]]
) -> None:
    # A diagnostic case on whose fetches both hosts ran the core: from T2.12
    # no S1 case is diagnostic, so an acceptance case with fetches stands in,
    # its file marked diagnostic in a copy of the cases and both hosts'
    # lines naming that file's digest.
    statuses = {
        case['id']: case['status']
        for case in (
            json.loads(path.read_text(encoding='utf-8'))
            for path in ROOT.glob('contract/cases/*/*.json')
        )
    }
    [case, *_] = [
        each['case']
        for each in observed['python-core']
        if statuses[each['case']] in ('diagnostic', 'acceptance')
        and (each['core'] or {}).get('fetches')
    ]
    cases = tmp_path / 'cases'
    shutil.copytree(ROOT / 'contract/cases', cases)
    [path] = [
        each
        for each in cases.glob('*/*.json')
        if json.loads(each.read_text(encoding='utf-8'))['id'] == case
    ]
    marked = json.loads(path.read_text(encoding='utf-8'))
    marked['status'] = 'diagnostic'
    path.write_text(json.dumps(marked, indent=2) + '\n', encoding='utf-8')
    case_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    changed = copy.deepcopy(observed)
    for side in ('python-core', 'rust-host'):
        for line in changed[side]:
            if line['case'] == case:
                line['identity']['inputs']['case'] = case_digest
    done = _run(
        tmp_path,
        _seeded(changed, 'rust-host', case, 'fetches'),
        lane=True,
        cases=cases,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert f'diagnostic: core: {case}: python-core against rust-host' in (
        done.stdout
    )


@pytest.mark.parametrize('side', ['python-core.jsonl', 'rust-host.jsonl'])
@pytest.mark.parametrize(
    'fault',
    [
        'missing',
        'empty',
        'another runner',
        'not json',
        'nested deep',
        'wide',
        'no identity',
    ],
)
def test_either_side_missing_or_retyped_fails_before_comparing(
    tmp_path: Path, observed: dict[str, list[Any]], side: str, fault: str
) -> None:
    files = _files(observed)
    other = (
        'rust-host.jsonl'
        if side == 'python-core.jsonl'
        else 'python-core.jsonl'
    )
    files[side] = {
        'missing': None,
        'empty': '',
        'another runner': files[other],
        'not json': '{\n',
        # A line nested past any parser's depth, and one very wide object
        # naming no runner: each refused by name, never a crash.
        'nested deep': '[' * 100_000 + ']' * 100_000 + '\n',
        'wide': json.dumps({f'k{i}': i for i in range(200_000)}) + '\n',
        'no identity': json.dumps({'identity': None}) + '\n',
    }[fault]
    done = _run(tmp_path, files)
    assert done.returncode != 0, done.stdout
    assert 'cross-host:' in done.stdout + done.stderr
    assert 'compare:' not in done.stdout


def test_with_a_base_the_effective_expectations_are_asked_and_needed(
    tmp_path: Path, observed: dict[str, list[Any]]
) -> None:
    done = _run(tmp_path, _files(observed), '--base origin/rust-core-phase-1')
    assert done.returncode != 0, done.stdout
    called = (tmp_path / 'effective-called').read_text()
    assert called.startswith(
        'scripts/rust_core/ratchet.py effective --base origin/rust-core-phase-1 '
    )
    assert 'compare:' not in done.stdout


def test_a_failing_where_fails_the_step(
    tmp_path: Path, observed: dict[str, list[Any]]
) -> None:
    done = _run(tmp_path, _files(observed), None)
    assert done.returncode != 0, done.stdout
    assert 'cross-host: ratchet.py where failed' in done.stdout
