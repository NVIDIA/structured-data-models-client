# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gate's local modes, ``--only`` and ``--locked`` (T1.19, merged into
T1.15; decisions E53(b) and H26), and its new default check, ``coverage``,
through the real ``scripts/rust_core/gate.sh`` on a temporary repository
whose checkpoint is a landing holding this tree's harness (as
``tests/test_contract_ratchet.py``'s ``build()`` makes it), with the Python
runners' real observations.

- ``--only CHECK`` runs that one check: the summary lists exactly it, the
  exit is its result, and the last line says it was one check, not the
  gate; another check's seeded failure does not reach it; ``--only
  compare`` judges the observations given against the checkpoint's
  effective expectations without the ratchet's check; an unknown check, or
  ``--only`` with ``--locked``, is a usage error.
- ``coverage`` runs in the default gate, and fails it on a seeded fault.
- ``--locked 1a`` and ``--locked 1b`` print the table of the half's locked
  checks, write the evidence to ``DIR/gate/<sha>.json`` and exit non-zero
  when a locked check fails. Each locked check of 1a passes on stubs and
  fails on a seeded defect: a ``cargo`` first on ``PATH`` (``l1``, and
  ``--run rust-host``, whose observations are ``python-core``'s, keeping
  what ``contract/availability.yaml`` makes available to it), and, in the
  checkpoint, stubs for ``wheel_smoke.py``, the seeded-fault suite,
  ``selftest.py`` and the L0 scripts (the temporary copy has no Cargo
  workspace). 1b's checks 4 and 5 hold the ``inventory`` step, which no
  stub can pass while the inventory's cases and ``base.adventureworks``
  are diagnostic: they are shown failing on it, with every other step of
  theirs passing (H26: the owner's).
- Evidence inside the candidate's worktree is refused; a run of the other
  half adds its half and keeps the first; the default directory is under
  the repository's common git directory.
- Check 3 gives the checkpoint's ``wheel_smoke.py`` the candidate's
  worktree as its contract inputs and the candidate's HEAD as the wheels'
  commit (T1.20); with ``--wheel-run ID`` (only with ``--locked``) its
  evidence records each wheel's name, SHA-256 and identity's commit, and
  the run.
- The post-landing state: this tree's harness, judging a candidate equal to
  its own landing, passes ``--only coverage`` and the default gate.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
if 'ratchet_tests' in sys.modules:
    helpers = sys.modules['ratchet_tests']
else:
    spec = importlib.util.spec_from_file_location(
        'ratchet_tests', ROOT / 'tests/test_contract_ratchet.py'
    )
    assert spec is not None and spec.loader is not None
    helpers = importlib.util.module_from_spec(spec)
    sys.modules['ratchet_tests'] = helpers
    spec.loader.exec_module(helpers)
git, commit, land = helpers.git, helpers.commit, helpers.land

CASE = 's0.accept.sum-window'
MANIFEST = 'docs/design/rust-core/phase-0/scope-python.yaml'
#: An S1 path no case names: dropping it fails coverage's rule 8 only.
PATH = 'source.open/explicit/duckdb'
#: The runners whose observations a locked run is given.
CHECKED = ('python-core', 'python-old', 'rust-host')
#: The checks ``--only`` takes.
CHECKS = ('observations', 'ratchet', 'compare', 'l0', 'coverage')
#: The locked checks of each half, and the steps each holds.
LOCKED = {
    '1a': {
        '1': {
            'runners',
            'observations',
            'ratchet',
            'compare',
            'coverage',
            'l1',
        },
        '2': {'seeded'},
        '3': {'wheel-smoke'},
    },
    '1b': {
        '4': {
            'runners',
            'observations',
            'ratchet',
            'compare',
            'l1',
            'inventory',
        },
        '5': {'observations', 'ratchet', 'compare', 'inventory'},
    },
}

#: The seeded-fault suite's files, as stubs in the stubbed checkpoint:
#: every contract test and every root contract test, so whichever the
#: gate's fixed list names runs a stub.
SUITES = sorted(
    {
        *(
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / 'contract/tests').glob('test_*.py')
        ),
        *(
            p.relative_to(ROOT).as_posix()
            for p in (ROOT / 'tests').glob('test_contract_*.py')
        ),
        'tests/test_contract_coverage.py',
    }
)
PASSING_SUITE = 'def test_stub() -> None:\n    pass\n'
FAILING_SUITE = "def test_stub() -> None:\n    raise AssertionError('seeded')\n"
#: The seeded file: one the gate's list names.
SEEDED_SUITE = 'contract/tests/test_capture.py'
#: Stubs for the checkpoint's scripts the locked steps run: each passes,
#: recording how it was called.
LOGGED = """import json
import os
import sys


def log(tool):
    if os.environ.get('STUB_LOG'):
        with open(os.environ['STUB_LOG'], 'a') as handle:
            handle.write(json.dumps(
                {'tool': tool, 'cwd': os.getcwd(), 'argv': sys.argv[1:]}
            ) + '\\n')
"""
STUB_SCRIPT = LOGGED + '\nlog(os.path.basename(sys.argv[0]))\n'
STUB_WHEEL_SMOKE = (
    LOGGED
    + """
import argparse
from pathlib import Path

log('wheel_smoke.py')
parser = argparse.ArgumentParser()
parser.add_argument('--lane', choices=['floor', 'newest'], required=True)
parser.add_argument('--wheel', required=True)
parser.add_argument('--test-ops-wheel', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--container', action='store_true')
arguments, _ = parser.parse_known_args()
for wheel in (arguments.wheel, arguments.test_ops_wheel):
    if 'broken' in Path(wheel).read_text():
        print(f'stub: {wheel}: does not install')
        sys.exit(1)
"""
)
#: The ``cargo`` first on PATH: ``test`` fails where the candidate's
#: ``rust/`` holds ``l1-defect``; ``run`` (``--run rust-host``) writes the
#: stubbed rust-host observations.
STUB_CARGO = (
    LOGGED
    + """
import shutil
from pathlib import Path

log('cargo')
arguments = sys.argv[1:]
if arguments[:1] == ['test']:
    sys.exit(101 if Path('l1-defect').exists() else 0)
if arguments[:1] == ['run']:
    shutil.copy(
        os.environ['STUB_RUST_HOST'], arguments[arguments.index('--out') + 1]
    )
    sys.exit(0)
sys.exit(2)
"""
)


def stub_the_checkpoint(repo: Path) -> None:
    for name in SUITES:
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(PASSING_SUITE)
    for name in ('selftest.py', 'l0_deps.py', 'l0_scan.py'):
        (repo / 'scripts/rust_core' / name).write_text(STUB_SCRIPT)
    (repo / 'scripts/rust_core/wheel_smoke.py').write_text(STUB_WHEEL_SMOKE)


@pytest.fixture(scope='module')
def origin(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    r"""The temporary repository, its landing holding this tree's harness,
    and a later landing whose harness has the locked steps' stubs."""
    repo, landed = helpers.build(tmp_path_factory.mktemp('origin'))
    stubbed = land(
        repo, 'T1.15', stub_the_checkpoint, protected='scripts/rust_core/**'
    )
    return {'repo': repo, 'landed': landed, 'stubbed': stubbed}


@pytest.fixture(scope='module')
def observed(
    origin: dict[str, Any], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, list[dict[str, Any]]]:
    r"""Each Python runner's observations of the repository's cases, and
    rust-host's stubbed from python-core's."""
    repo = origin['repo']
    out = tmp_path_factory.mktemp('observed')
    found = {}
    for runner in ('python-core', 'python-old'):
        path = out / f'{runner}.jsonl'
        done = subprocess.run(
            [
                sys.executable,
                '-I',
                str(repo / 'contract/runners/run_python.py'),
                '--mode',
                runner,
                '--out',
                str(path),
                str(repo / 'contract/cases'),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        found[runner] = [json.loads(x) for x in path.read_text().splitlines()]
    found['rust-host'] = rust_host(found['python-core'])
    return found


def rust_host(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    r"""python-core's observations as rust-host's stub: what
    ``contract/availability.yaml`` makes unavailable to rust-host left out,
    and its identity a Rust host's."""
    available = yaml.safe_load(
        (ROOT / 'contract/availability.yaml').read_text()
    )['runners']['rust-host']
    made = []
    for line in copy.deepcopy(lines):
        identity = line['identity']
        identity.update(
            runner='rust-host', cell=None, native=None, native_loaded=None
        )
        identity['versions'] = {'duckdb': identity['versions']['duckdb']}
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
        made.append(line)
    return made


def worktree(origin: dict[str, Any], tmp_path: Path, at: str) -> Path:
    tree = tmp_path / 'tree'
    git(origin['repo'], 'worktree', 'add', '-q', '--detach', str(tree), at)
    return tree


@pytest.fixture
def repo(origin: dict[str, Any], tmp_path: Path) -> Path:
    r"""A worktree at the landing holding this tree's harness."""
    return worktree(origin, tmp_path, origin['landed'])


@pytest.fixture
def stubbed(origin: dict[str, Any], tmp_path: Path) -> Path:
    r"""A worktree at the landing holding the stubs."""
    return worktree(origin, tmp_path, origin['stubbed'])


@pytest.fixture
def stubs(
    tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    r"""The environment for the locked steps: the stub ``cargo`` first on
    PATH, its rust-host observations, the log every stub writes, and two
    wheels that install."""
    where = tmp_path / 'stubs'
    where.mkdir()
    cargo = where / 'bin/cargo'
    cargo.parent.mkdir()
    cargo.write_text(f'#!{sys.executable}\n' + STUB_CARGO)
    cargo.chmod(0o755)
    rust = where / 'rust-host.jsonl'
    rust.write_text(
        ''.join(json.dumps(x) + '\n' for x in observed['rust-host'])
    )
    wheels = {}
    for name in ('shipped', 'test-ops'):
        wheels[name] = where / f'{name}.whl'
        wheels[name].write_text('stub')
    return {
        'env': {
            'PATH': f'{cargo.parent}{os.pathsep}{os.environ["PATH"]}',
            'STUB_LOG': str(where / 'log.jsonl'),
            'STUB_RUST_HOST': str(rust),
        },
        'log': where / 'log.jsonl',
        'wheels': wheels,
        'evidence': tmp_path / 'evidence',
    }


def files(
    observed: dict[str, list[dict[str, Any]]],
    tmp_path: Path,
    runners: tuple[str, ...] = ('python-core', 'python-old'),
    change: Callable[[str, list[dict[str, Any]]], None] | None = None,
) -> list[str]:
    arguments = []
    where = tmp_path / 'given'
    where.mkdir(exist_ok=True)
    for runner in runners:
        lines = json.loads(json.dumps(observed[runner]))
        if change is not None:
            change(runner, lines)
        path = where / f'{runner}.jsonl'
        path.write_text(''.join(json.dumps(x) + '\n' for x in lines))
        arguments += ['--observations', str(path)]
    return arguments


def gate(
    repo: Path, *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(repo / 'scripts/rust_core/gate.sh'), *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo,
        env={**os.environ, 'PYTHON': sys.executable, **(env or {})},
    )


def summary(done: subprocess.CompletedProcess[str]) -> dict[str, str]:
    r"""The gate summary's lines: each check's result, by name."""
    lines = done.stdout.splitlines()
    starts = [i for i, x in enumerate(lines) if x.startswith('gate summary')]
    assert starts, done.stdout + done.stderr
    found = {}
    for line in lines[starts[-1] + 1 :]:
        matched = re.match(r'^  (ok|FAIL) +(\S+)\s*$', line)
        if matched is None:
            break
        found[matched[2]] = matched[1]
    return found


def table(done: subprocess.CompletedProcess[str], half: str) -> dict[str, str]:
    r"""The locked checks' table: each check's result, by number."""
    lines = done.stdout.splitlines()
    header = f'locked checks ({half}):'
    starts = [i for i, x in enumerate(lines) if x.strip() == header]
    assert len(starts) == 1, done.stdout + done.stderr
    found = {}
    for line in lines[starts[0] + 1 :]:
        if line and not line.startswith((' ', '\t')):
            break
        matched = re.match(r'^  (ok|FAIL) +check (\d+)\b', line)
        if matched is not None:
            assert matched[2] not in found, done.stdout
            found[matched[2]] = matched[1]
    return found


def last(done: subprocess.CompletedProcess[str]) -> str:
    lines = [x for x in done.stdout.splitlines() if x.strip()]
    assert lines, done.stderr
    return lines[-1]


def locked(
    repo: Path,
    stubs: dict[str, Any],
    half: str,
    *arguments: str,
    wheels: bool = True,
    evidence: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    given = list(arguments)
    if wheels:
        given += [
            '--wheel',
            str(stubs['wheels']['shipped']),
            '--test-ops-wheel',
            str(stubs['wheels']['test-ops']),
        ]
    return gate(
        repo,
        '--base',
        git(repo, 'rev-parse', 'HEAD~1'),
        '--locked',
        half,
        '--evidence',
        str(evidence or stubs['evidence']),
        *given,
        env=stubs['env'],
    )


def evidence_of(
    repo: Path, stubs: dict[str, Any], where: Path | None = None
) -> dict[str, Any]:
    sha = git(repo, 'rev-parse', 'HEAD')
    path = (where or stubs['evidence']) / 'gate' / f'{sha}.json'
    assert path.is_file(), sorted(
        str(p) for p in (where or stubs['evidence']).rglob('*')
    )
    found: dict[str, Any] = json.loads(path.read_text())
    assert found['sha'] == sha
    return found


def logged(stubs: dict[str, Any], tool: str) -> list[dict[str, Any]]:
    if not stubs['log'].exists():
        return []
    return [
        x
        for x in map(json.loads, stubs['log'].read_text().splitlines())
        if x['tool'] == tool
    ]


def drop_the_path(r: Path) -> None:
    document = yaml.safe_load((r / MANIFEST).read_text())
    document['paths'] = [x for x in document['paths'] if x['id'] != PATH]
    (r / MANIFEST).write_text(yaml.safe_dump(document, sort_keys=False))


def corrupt(r: Path) -> None:
    path = r / f'contract/cases/s0/{CASE}.json'
    case = json.loads(path.read_text())
    case['expect']['outcome'] = 'refusal'
    path.write_text(json.dumps(case, indent=1))


def without_the_case(runner: str, lines: list[dict[str, Any]]) -> None:
    if runner == 'python-old':
        lines[:] = [x for x in lines if x['case'] != CASE]


# --- --only: one check ---------------------------------------------------------


def test_only_compare_judges_the_observations_given(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)
    done = gate(
        repo,
        '--no-l0',
        '--base',
        landed,
        '--only',
        'compare',
        *files(observed, tmp_path),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert summary(done) == {'compare': 'ok'}, done.stdout
    assert (
        last(done) == 'gate: passed (--only compare: one check, not the gate)'
    )
    # Its own seeded fault fails it.
    done = gate(
        repo,
        '--no-l0',
        '--base',
        landed,
        '--only',
        'compare',
        *files(observed, tmp_path, change=without_the_case),
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert summary(done) == {'compare': 'FAIL'}, done.stdout
    assert last(done) == 'gate: FAILED'
    assert f'{CASE}: python-old: runner failure: no observation' in done.stdout


def test_only_compare_without_observations_fails(repo: Path) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)
    done = gate(repo, '--no-l0', '--base', landed, '--only', 'compare')
    assert done.returncode != 0, done.stdout + done.stderr
    assert 'gate: passed' not in done.stdout
    assert 'observations' in done.stdout + done.stderr


def test_only_compare_holds_the_effective_expectations_without_the_ratchet(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    r"""A fault only the ratchet's check sees (a protected change no
    trailer names) leaves ``--only compare`` passing; the effective
    expectations still come from the checkpoint's ratchet, so an outcome
    against its baseline fails it."""
    landed = git(repo, 'rev-parse', 'HEAD')

    def touch(r: Path) -> None:
        with (r / 'scripts/rust_core/l0_scan.py').open('a') as handle:
            handle.write('\n# touched\n')

    commit(repo, edit=touch)
    # Before landing, the protected paths are named on the command line.
    given = [*files(observed, tmp_path), '--protected-paths', 'nothing']
    done = gate(repo, '--no-l0', '--base', landed, '--only', 'ratchet', *given)
    assert summary(done) == {'ratchet': 'FAIL'}, done.stdout
    assert done.returncode == 1
    done = gate(repo, '--no-l0', '--base', landed, '--only', 'compare', *given)
    assert done.returncode == 0, done.stdout + done.stderr
    assert summary(done) == {'compare': 'ok'}, done.stdout

    def moved(runner: str, lines: list[dict[str, Any]]) -> None:
        for line in lines:
            if runner == 'python-old' and line['case'] == CASE:
                line['outcome']['stage'] = 'predict'

    done = gate(
        repo,
        '--no-l0',
        '--base',
        landed,
        '--only',
        'compare',
        *files(observed, tmp_path, change=moved),
    )
    assert summary(done) == {'compare': 'FAIL'}, done.stdout
    assert 'against its effective expectation: outcome.stage' in done.stdout


def test_another_check_s_failure_does_not_reach_only(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo, edit=corrupt)
    # The whole gate fails the corrupted expectation...
    done = gate(repo, '--no-l0', '--base', landed, *files(observed, tmp_path))
    assert done.returncode == 1
    assert {k for k, v in summary(done).items() if v == 'FAIL'} == {
        'ratchet',
        'compare',
    }, done.stdout
    # ...and --only coverage does not see it.
    done = gate(repo, '--no-l0', '--base', landed, '--only', 'coverage')
    assert done.returncode == 0, done.stdout + done.stderr
    assert summary(done) == {'coverage': 'ok'}, done.stdout


def test_only_coverage_fails_its_own_fault_and_only_ratchet_does_not(
    repo: Path,
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo, edit=drop_the_path)
    done = gate(repo, '--no-l0', '--base', landed, '--only', 'coverage')
    assert done.returncode == 1, done.stdout + done.stderr
    assert summary(done) == {'coverage': 'FAIL'}, done.stdout
    assert last(done) == 'gate: FAILED'
    assert any(
        'rule 8:' in line and PATH in line for line in done.stdout.splitlines()
    ), done.stdout
    done = gate(repo, '--no-l0', '--base', landed, '--only', 'ratchet')
    assert done.returncode == 0, done.stdout + done.stderr
    assert summary(done) == {'ratchet': 'ok'}, done.stdout


def test_a_mode_given_wrong_is_a_usage_error(repo: Path) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)
    # Not vacuous: the gate takes the mode itself.
    done = gate(repo, '--no-l0', '--base', landed, '--only', 'coverage')
    assert done.returncode != 2, done.stdout + done.stderr
    for arguments in (
        ['--only', 'everything'],
        ['--only', 'seeded'],
        ['--only'],
        ['--only', 'l0', '--locked', '1a'],
        ['--locked', '1a', '--only', 'coverage'],
        ['--locked', '1c'],
        ['--locked'],
    ):
        done = gate(repo, '--no-l0', '--base', landed, *arguments)
        assert done.returncode == 2, (arguments, done.stdout + done.stderr)
        assert 'gate: passed' not in done.stdout, arguments


# --- coverage, a default check ----------------------------------------------------


def test_the_default_gate_runs_coverage(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    given = files(observed, tmp_path)
    commit(repo)
    done = gate(repo, '--no-l0', '--base', landed, *given)
    assert done.returncode == 0, done.stdout + done.stderr
    assert summary(done) == {'ratchet': 'ok', 'compare': 'ok', 'coverage': 'ok'}
    assert last(done) == 'gate: passed'
    git(repo, 'reset', '-q', '--hard', landed)
    commit(repo, edit=drop_the_path)
    done = gate(repo, '--no-l0', '--base', landed, *given)
    assert done.returncode == 1
    assert summary(done) == {
        'ratchet': 'ok',
        'compare': 'ok',
        'coverage': 'FAIL',
    }, done.stdout


# --- the post-landing state ------------------------------------------------------


def test_the_post_landing_state_passes(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    r"""The phase branch at a landing whose tree is this harness's own
    landing's, judged as CI judges a phase push, from its previous tip:
    ``--only coverage`` and the default gate pass."""
    before = git(repo, 'rev-parse', 'HEAD')
    land(repo, 'T1.15')
    done = gate(repo, '--no-l0', '--phase-push', before, '--only', 'coverage')
    assert done.returncode == 0, done.stdout + done.stderr
    assert summary(done) == {'coverage': 'ok'}, done.stdout
    done = gate(
        repo, '--no-l0', '--phase-push', before, *files(observed, tmp_path)
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert summary(done)['coverage'] == 'ok', done.stdout


# --- --locked 1a ----------------------------------------------------------------


def test_locked_1a_passes_on_stubs_and_writes_the_evidence(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
) -> None:
    landed = git(stubbed, 'rev-parse', 'HEAD')
    commit(stubbed)
    done = locked(
        stubbed, stubs, '1a', '--run', 'rust-host', *files(observed, tmp_path)
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert table(done, '1a') == {'1': 'ok', '2': 'ok', '3': 'ok', '6': 'ok'}
    assert {'observations', 'ratchet', 'compare', 'coverage', 'l0'} <= set(
        summary(done)
    ), done.stdout
    assert set(summary(done).values()) == {'ok'}, done.stdout
    # The evidence.
    found = evidence_of(stubbed, stubs)
    assert set(found['halves']) == {'1a'}
    half = found['halves']['1a']
    assert half['checkpoint'] == landed
    assert half['base'] == landed
    assert half['passed'] is True
    assert set(half['checks']) == {'1', '2', '3', '6'}
    for number, steps in LOCKED['1a'].items():
        assert half['checks'][number]['result'] == 'ok'
        assert set(half['checks'][number]['steps']) == steps, number
    assert set(half['checks']['6']['steps']) == set(half['steps'])
    assert half['checks']['6']['result'] == 'ok'
    every = {
        'runners',
        'observations',
        'ratchet',
        'compare',
        'coverage',
        'l0',
        'l1',
        'seeded',
        'wheel-smoke',
    }
    assert every <= set(half['steps']), half['steps']
    assert set(half['steps'].values()) == {'ok'}
    # The stubs were run as the steps say.
    rust = Path(git(stubbed, 'rev-parse', '--show-toplevel')).resolve() / 'rust'
    tests = [x for x in logged(stubs, 'cargo') if x['argv'][:1] == ['test']]
    assert [x['argv'] for x in tests] == [['test', '--workspace', '--locked']]
    assert Path(tests[0]['cwd']).resolve() == rust
    runs = [x for x in logged(stubs, 'cargo') if x['argv'][:1] == ['run']]
    assert len(runs) == 1
    smoke = logged(stubs, 'wheel_smoke.py')
    lanes = [x['argv'][x['argv'].index('--lane') + 1] for x in smoke]
    assert lanes == ['floor', 'newest']
    for each in smoke:
        assert '--container' in each['argv']
        assert '--out' in each['argv']
        argv = each['argv']
        assert argv[argv.index('--wheel') + 1] == str(
            stubs['wheels']['shipped']
        )
        assert argv[argv.index('--test-ops-wheel') + 1] == str(
            stubs['wheels']['test-ops']
        )
    assert logged(stubs, 'selftest.py'), 'the L0 self-tests did not run'
    assert '--no-clippy' in logged(stubs, 'selftest.py')[0]['argv']


def _option(argv: list[str], name: str) -> str:
    r"""The value of `name` in `argv`, given as ``NAME VALUE`` or
    ``NAME=VALUE``; exactly once."""
    found = [
        argv[i + 1] for i, word in enumerate(argv[:-1]) if word == name
    ] + [word.split('=', 1)[1] for word in argv if word.startswith(name + '=')]
    assert len(found) == 1, (name, argv)
    return found[0]


def test_locked_1a_gives_the_smoke_the_candidate_s_inputs_and_commit(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
) -> None:
    r"""T1.20: check 3 runs the checkpoint's ``wheel_smoke.py`` with the
    candidate's worktree as its contract inputs (``--inputs``) and the
    candidate's HEAD as the wheels' commit (``--commit``), in both lanes."""
    commit(stubbed)
    done = locked(stubbed, stubs, '1a', *files(observed, tmp_path, CHECKED))
    assert done.returncode == 0, done.stdout + done.stderr
    assert table(done, '1a')['3'] == 'ok', done.stdout
    head = git(stubbed, 'rev-parse', 'HEAD')
    assert len(head) == 40
    candidate = Path(git(stubbed, 'rev-parse', '--show-toplevel')).resolve()
    smoke = logged(stubs, 'wheel_smoke.py')
    lanes = [_option(x['argv'], '--lane') for x in smoke]
    assert lanes == ['floor', 'newest']
    for each in smoke:
        argv = each['argv']
        assert Path(_option(argv, '--inputs')).resolve() == candidate, argv
        assert _option(argv, '--commit') == head, argv


#: The commits the fake wheels' identity files name.
WHEEL_COMMITS = {
    'shipped': '0123456789abcdef0123456789abcdef01234567',
    'test-ops': 'fedcba9876543210fedcba9876543210fedcba98',
}


def _identified(stubs: dict[str, Any], which: tuple[str, ...]) -> None:
    r"""The fake wheels made distinct (so their hashes are), each of
    `which` with an identity file beside it as ``native_wheel.py`` writes
    it, naming its commit."""
    for name, wheel in stubs['wheels'].items():
        wheel.write_text(f'stub wheel {name}')
        if name in which:
            identity = {
                'wheel': wheel.name,
                'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
                'commit': WHEEL_COMMITS[name],
                'tree_clean': True,
                'features': ['test-ops'] if name == 'test-ops' else [],
            }
            Path(f'{wheel}.json').write_text(json.dumps(identity, indent=2))


def _recorded(
    stubs: dict[str, Any], name: str, commit: str | None
) -> dict[str, Any]:
    wheel = stubs['wheels'][name]
    return {
        'name': wheel.name,
        'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest(),
        'commit': commit,
    }


def test_locked_1a_records_the_wheels_and_their_run_in_check_3(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
) -> None:
    r"""T1.20: ``--wheel-run ID`` names the CI run the wheels came from;
    the evidence's check 3 records each wheel's file name, its SHA-256 and
    the commit its identity file names, and the run."""
    commit(stubbed)
    _identified(stubs, ('shipped', 'test-ops'))
    done = locked(
        stubbed,
        stubs,
        '1a',
        '--wheel-run',
        '24680135',
        *files(observed, tmp_path, CHECKED),
    )
    assert done.returncode == 0, done.stdout + done.stderr
    check = evidence_of(stubbed, stubs)['halves']['1a']['checks']['3']
    assert check['result'] == 'ok'
    assert set(check['steps']) == {'wheel-smoke'}
    wheels = check['wheels']
    assert set(wheels) == {'production', 'test-ops', 'run'}, wheels
    assert wheels['production'] == _recorded(
        stubs, 'shipped', WHEEL_COMMITS['shipped']
    )
    assert wheels['test-ops'] == _recorded(
        stubs, 'test-ops', WHEEL_COMMITS['test-ops']
    )
    assert wheels['production']['sha256'] != wheels['test-ops']['sha256']
    assert wheels['run'] in ('24680135', 24680135), wheels


def test_locked_1a_without_a_wheel_run_records_none(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
) -> None:
    r"""Without ``--wheel-run`` the run is null; a wheel with no identity
    file beside it is recorded with a null commit."""
    commit(stubbed)
    _identified(stubs, ('shipped',))
    done = locked(stubbed, stubs, '1a', *files(observed, tmp_path, CHECKED))
    assert done.returncode == 0, done.stdout + done.stderr
    check = evidence_of(stubbed, stubs)['halves']['1a']['checks']['3']
    wheels = check['wheels']
    assert set(wheels) == {'production', 'test-ops', 'run'}, wheels
    assert wheels['run'] is None
    assert wheels['production'] == _recorded(
        stubs, 'shipped', WHEEL_COMMITS['shipped']
    )
    assert wheels['test-ops'] == _recorded(stubs, 'test-ops', None)


def test_a_wheel_run_without_locked_is_a_usage_error(repo: Path) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)
    for arguments in (
        ['--wheel-run', '24680135'],
        ['--only', 'coverage', '--wheel-run', '24680135'],
        ['--locked', '1a', '--wheel-run'],
    ):
        done = gate(repo, '--no-l0', '--base', landed, *arguments)
        assert done.returncode == 2, (arguments, done.stdout + done.stderr)
        assert 'gate: passed' not in done.stdout, arguments
        # Refused as --locked's, not as an argument the gate does not know.
        if arguments[-1] != '--wheel-run':
            assert '--locked' in done.stderr, (arguments, done.stderr)


def _l1_defect(r: Path) -> None:
    (r / 'rust/l1-defect').write_text('seeded\n')


def _failing_suite(r: Path) -> None:
    (r / SEEDED_SUITE).write_text(FAILING_SUITE)


#: Each seeded defect: the candidate's change, the runners whose
#: observations are given, an observation change, the wheels, the
#: protected paths named, the check it fails, and the step.
DEFECTS: dict[str, dict[str, Any]] = {
    'runners': {
        'runners': ('python-core', 'python-old'),
        'fails': ('1', 'runners'),
    },
    'l1': {'edit': _l1_defect, 'fails': ('1', 'l1')},
    'coverage': {'edit': drop_the_path, 'fails': ('1', 'coverage')},
    'compare': {'change': without_the_case, 'fails': ('1', 'compare')},
    'seeded': {
        'edit': _failing_suite,
        'protected': f'{SEEDED_SUITE} (seeded)',
        'fails': ('2', 'seeded'),
    },
    'broken wheel': {'broken': True, 'fails': ('3', 'wheel-smoke')},
    'no wheels': {'wheels': False, 'fails': ('3', 'wheel-smoke')},
}


@pytest.mark.parametrize('defect', list(DEFECTS))
def test_locked_1a_fails_each_check_on_its_seeded_defect(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
    defect: str,
) -> None:
    seed = DEFECTS[defect]
    commit(stubbed, edit=seed.get('edit'))
    runners = seed.get('runners', ('python-core', 'python-old', 'rust-host'))
    arguments = files(observed, tmp_path, runners, seed.get('change'))
    if 'protected' in seed:
        arguments += ['--protected-paths', seed['protected']]
    if seed.get('broken'):
        stubs['wheels']['test-ops'].write_text('broken')
    done = locked(
        stubbed, stubs, '1a', *arguments, wheels=seed.get('wheels', True)
    )
    number, step = seed['fails']
    assert done.returncode != 0, done.stdout + done.stderr
    wanted = {'1': 'ok', '2': 'ok', '3': 'ok', '6': 'FAIL', number: 'FAIL'}
    assert table(done, '1a') == wanted, done.stdout
    half = evidence_of(stubbed, stubs)['halves']['1a']
    assert half['passed'] is False
    assert half['steps'][step] == 'FAIL', half['steps']
    assert {n: c['result'] for n, c in half['checks'].items()} == wanted
    if defect == 'no wheels':
        assert '--wheel' in done.stdout and '--test-ops-wheel' in done.stdout


# --- --locked 1b ----------------------------------------------------------------


def test_locked_1b_fails_on_the_diagnostic_inventory_and_writes_the_evidence(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
) -> None:
    r"""Today the inventory's cases and ``base.adventureworks`` are
    diagnostic, so 1b's checks 4 and 5 fail on the ``inventory`` step, and
    on it alone: their other steps pass on the stubs."""
    commit(stubbed)
    done = locked(
        stubbed, stubs, '1b', '--run', 'rust-host', *files(observed, tmp_path)
    )
    assert done.returncode != 0, done.stdout + done.stderr
    assert table(done, '1b') == {'4': 'FAIL', '5': 'FAIL', '6': 'FAIL'}
    assert 'base.adventureworks' in done.stdout
    half = evidence_of(stubbed, stubs)['halves']['1b']
    assert half['passed'] is False
    assert set(half['checks']) == {'4', '5', '6'}
    for number, steps in LOCKED['1b'].items():
        assert half['checks'][number]['result'] == 'FAIL'
        assert set(half['checks'][number]['steps']) == steps, number
    assert half['steps']['inventory'] == 'FAIL'
    assert {
        step: result
        for step, result in half['steps'].items()
        if result == 'FAIL'
    } == {'inventory': 'FAIL'}, half['steps']


def test_a_second_half_adds_to_the_evidence_and_keeps_the_first(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
) -> None:
    commit(stubbed)
    given = files(observed, tmp_path, CHECKED)
    done = locked(stubbed, stubs, '1a', *given)
    assert done.returncode == 0, done.stdout + done.stderr
    first = evidence_of(stubbed, stubs)
    done = locked(stubbed, stubs, '1b', *given)
    assert done.returncode != 0
    both = evidence_of(stubbed, stubs)
    assert set(both['halves']) == {'1a', '1b'}
    assert both['halves']['1a'] == first['halves']['1a']
    assert both['halves']['1b']['passed'] is False


def test_the_evidence_defaults_to_the_common_git_directory(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
) -> None:
    # A candidate of its own, so no other test's run shares its SHA.
    commit(stubbed, f'T1.16: the default evidence, {tmp_path.name}')
    common = Path(git(stubbed, 'rev-parse', '--git-common-dir'))
    if not common.is_absolute():
        common = stubbed / common
    where = common.resolve() / 'sdm-gate-evidence'
    done = gate(
        stubbed,
        '--base',
        git(stubbed, 'rev-parse', 'HEAD~1'),
        '--locked',
        '1a',
        '--wheel',
        str(stubs['wheels']['shipped']),
        '--test-ops-wheel',
        str(stubs['wheels']['test-ops']),
        *files(observed, tmp_path, CHECKED),
        env=stubs['env'],
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert evidence_of(stubbed, stubs, where)['halves']['1a']['passed'] is True


@pytest.mark.parametrize('relative', [False, True])
def test_evidence_inside_the_candidate_s_worktree_is_refused(
    stubbed: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    stubs: dict[str, Any],
    relative: bool,
) -> None:
    commit(stubbed)
    inside = Path('evidence') if relative else stubbed / 'evidence'
    done = locked(
        stubbed,
        stubs,
        '1a',
        *files(observed, tmp_path, CHECKED),
        evidence=inside,
    )
    assert done.returncode == 2, done.stdout + done.stderr
    assert 'evidence' in (done.stdout + done.stderr).lower()
    assert not (stubbed / 'evidence').exists()


@pytest.mark.parametrize('check', CHECKS)
def test_only_runs_one_check_and_says_so(
    repo: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    check: str,
) -> None:
    r"""Each name ``--only`` takes runs that check and no other, and the
    last line says it was one check, not the gate. Here ``observations``
    has no evidence and ``l0`` no Cargo workspace, so those two fail, as
    checks, not as usage errors; ``compare`` is given the observations."""
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)
    given = files(observed, tmp_path) if check == 'compare' else []
    done = gate(repo, '--base', landed, '--only', check, *given)
    assert summary(done) == {
        check: 'FAIL' if check in ('observations', 'l0') else 'ok'
    }, done.stdout + done.stderr
    if check in ('observations', 'l0'):
        assert done.returncode == 1, done.stdout + done.stderr
        assert last(done) == 'gate: FAILED'
    else:
        assert done.returncode == 0, done.stdout + done.stderr
        assert last(done) == (
            f'gate: passed (--only {check}: one check, not the gate)'
        )
