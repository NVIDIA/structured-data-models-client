# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The gate's fail-first controls on the candidate (T1.14's exit), through
the real ``scripts/rust_core/gate.sh`` on a temporary repository whose
checkpoint is a landing holding the harness, with the Python runners'
real observations (``python-core`` and ``python-old``, run here as CI's
``engine`` job can):

- the unseeded candidate passes, and the output names the landing whose
  harness judged; before T1.14's landing the bootstrap says so;
- a corrupted expectation, a missing case, a skipped call, a missing
  native library and an old route that loaded the core each fail;
- a protected change fails unless named, and the checkpoint's harness
  refuses to judge from any other commit.

The ratchet's own seeded checks are ``tests/test_contract_ratchet.py``'s.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

# Gate tooling (T3.0): CI runs it in the tooling jobs, not in every matrix job.
pytestmark = pytest.mark.gate_tooling

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'ratchet_tests', ROOT / 'tests/test_contract_ratchet.py'
)
assert spec is not None and spec.loader is not None
helpers = importlib.util.module_from_spec(spec)
sys.modules['ratchet_tests'] = helpers
spec.loader.exec_module(helpers)
git, commit = helpers.git, helpers.commit
CASE = 's0.accept.sum-window'


@pytest.fixture(scope='module')
def origin(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, str]:
    built: tuple[Path, str] = helpers.build(tmp_path_factory.mktemp('origin'))
    return built


@pytest.fixture(scope='module')
def observed(
    origin: tuple[Path, str], tmp_path_factory: pytest.TempPathFactory
) -> dict[str, list[dict[str, Any]]]:
    r"""Each Python runner's observations of the repository's cases."""
    repo, _ = origin
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
    return found


@pytest.fixture
def repo(origin: tuple[Path, str], tmp_path: Path) -> Path:
    source, landed = origin
    tree = tmp_path / 'tree'
    git(source, 'worktree', 'add', '-q', '--detach', str(tree), landed)
    return tree


def files(
    observed: dict[str, list[dict[str, Any]]],
    tmp_path: Path,
    change: Callable[[str, list[dict[str, Any]]], None] | None = None,
) -> list[str]:
    arguments = []
    for runner, lines in observed.items():
        lines = json.loads(json.dumps(lines))
        if change is not None:
            change(runner, lines)
        path = tmp_path / f'{runner}.jsonl'
        path.write_text(''.join(json.dumps(x) + '\n' for x in lines))
        arguments += ['--observations', str(path)]
    return arguments


def gate(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(repo / 'scripts/rust_core/gate.sh'), '--no-l0', *arguments],
        capture_output=True,
        text=True,
        check=False,
        cwd=repo,
        env={**os.environ, 'PYTHON': sys.executable},
    )


def failed(done: subprocess.CompletedProcess[str]) -> set[str]:
    return {
        line.split()[1]
        for line in done.stdout.splitlines()
        if line.startswith('  FAIL ')
    }


def test_the_unseeded_candidate_passes_judged_by_the_checkpoint_s_harness(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)
    done = gate(repo, '--base', landed, *files(observed, tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    short = git(repo, 'rev-parse', '--short', landed)
    assert (
        f"gate: judged by the checkpoint's harness, at landing {short} "
        'Land T1.14: the package'
    ) in done.stdout


def test_before_the_harness_lands_the_bootstrap_says_so(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    # The landing itself, judged against the base it recorded: the first
    # landing, which has no ratchet.py.
    done = gate(repo, '--base', 'HEAD', *files(observed, tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr
    assert 'gate: judged by bootstrap' in done.stdout


def test_a_corrupted_expectation_fails(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')

    def corrupt(r: Path) -> None:
        path = r / f'contract/cases/s0/{CASE}.json'
        case = json.loads(path.read_text())
        case['expect']['outcome'] = 'refusal'
        path.write_text(json.dumps(case, indent=1))

    commit(repo, edit=corrupt)
    done = gate(repo, '--base', landed, *files(observed, tmp_path))
    assert done.returncode == 1
    assert failed(done) == {'ratchet', 'compare'}, done.stdout
    assert f'cases: {CASE}: its input changed' in done.stdout


def test_a_missing_case_fails(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)

    def drop(runner: str, lines: list[dict[str, Any]]) -> None:
        if runner == 'python-old':
            lines[:] = [x for x in lines if x['case'] != CASE]

    done = gate(repo, '--base', landed, *files(observed, tmp_path, drop))
    assert failed(done) == {'ratchet', 'compare'}, done.stdout
    assert f'{CASE}: python-old: baselined, and not observed' in done.stdout
    assert f'{CASE}: python-old: runner failure: no observation' in done.stdout


def test_a_skipped_call_fails(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)

    def without_the_call(runner: str, lines: list[dict[str, Any]]) -> None:
        for line in lines:
            if runner == 'python-core' and line['case'] == CASE:
                line['identity']['operations'] = []

    done = gate(
        repo, '--base', landed, *files(observed, tmp_path, without_the_call)
    )
    assert failed(done) == {'compare'}, done.stdout
    assert "['validate_query'] did not run on the core" in done.stdout


def test_a_missing_native_library_fails(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)

    def unload(runner: str, lines: list[dict[str, Any]]) -> None:
        for line in lines:
            if runner == 'python-core' and line['case'] == CASE:
                line['identity']['native'] = None
                line['identity']['native_loaded'] = False

    done = gate(repo, '--base', landed, *files(observed, tmp_path, unload))
    assert failed(done) == {'compare'}, done.stdout
    assert 'sdm_client._native was not loaded' in done.stdout


def test_an_old_route_that_loaded_the_core_fails(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    r"""The route proof's other half (T1.15, T1.19 merged into it): a
    ``python-old`` observation with ``sdm_client._native`` loaded fails;
    ``contract/tests/test_route_proof_seeds.py`` makes one with a real run."""
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)

    def load(runner: str, lines: list[dict[str, Any]]) -> None:
        for line in lines:
            if runner == 'python-old' and line['case'] == CASE:
                line['identity']['native_loaded'] = True

    done = gate(repo, '--base', landed, *files(observed, tmp_path, load))
    assert failed(done) == {'compare'}, done.stdout
    assert f'{CASE}: python-old: route: the old route loaded the core' in (
        done.stdout
    )


def test_a_protected_change_fails_unless_named(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    path = 'scripts/rust_core/l0_scan.py'

    def touch(r: Path) -> None:
        with (r / path).open('a') as handle:
            handle.write('\n# touched\n')

    commit(repo, edit=touch)
    given = files(observed, tmp_path)
    done = gate(repo, '--base', landed, '--protected-paths', 'nothing', *given)
    assert failed(done) == {'ratchet'}, done.stdout
    done = gate(
        repo,
        '--base',
        landed,
        '--protected-paths',
        f'{path} (a comment)',
        *given,
    )
    assert done.returncode == 0, done.stdout


def test_the_checkpoint_s_harness_refuses_to_judge_from_another_commit(
    repo: Path,
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)
    # The candidate's own copy, called as if it were the checkpoint's.
    done = subprocess.run(
        [
            str(repo / 'scripts/rust_core/gate.sh'),
            '--harness-for',
            str(repo),
            '--base',
            landed,
        ],
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ, 'PYTHON': sys.executable},
    )
    assert done.returncode == 1
    assert "this harness is not the checkpoint's" in done.stderr


def test_the_gate_fails_without_behavioural_evidence(repo: Path) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    commit(repo)
    done = gate(repo, '--base', landed)
    assert done.returncode == 1
    assert 'gate: passed' not in done.stdout
    assert failed(done) == {'observations'}, done.stdout
    assert 'no observations: give --run RUNNERS or --observations FILE' in (
        done.stdout
    )


def _moved(runner: str, lines: list[dict[str, Any]]) -> None:
    r"""python-old's outcome of the case moves to predict."""
    for line in lines:
        if runner == 'python-old' and line['case'] == CASE:
            line['outcome']['stage'] = 'predict'


@pytest.mark.parametrize('transition', ['approved', 'wrong', 'unapproved'])
def test_an_approved_outcome_change_passes_the_whole_gate_and_no_other(
    repo: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    transition: str,
) -> None:
    landed = git(repo, 'rev-parse', 'HEAD')
    arguments = files(observed, tmp_path, _moved)
    old = {'approved': 'success at materialize', 'wrong': 'success'}
    if transition in old:
        item = {
            'case': CASE,
            'cell': 'ref-pandas3',
            'runner': 'python-old',
            'old': old[transition],
            'new': 'success at predict',
            'summary': 'seeded',
        }
        commit(
            repo,
            edit=lambda r: helpers.append_entry(
                r, {'id': 'SEED-1', 'cases': [CASE], 'case_outcomes': [item]}
            ),
        )
    # The successor baselines, as ratchet.py accept writes them; an
    # unapproved change is written into python-old's by hand.
    paths = [Path(a) for a in arguments if a != '--observations']
    helpers.accepted(repo, landed, tuple(paths))
    baseline = repo / 'contract/baselines/ref-pandas3/python-old.jsonl'
    if transition != 'approved':
        lines = [json.loads(x) for x in baseline.read_text().splitlines()]
        if not any(x['case'] == CASE for x in lines):
            projection = {
                'outcome.code': None,
                'outcome.error': None,
                'outcome.result': 'success',
                'outcome.stage': 'predict',
            }
            lines.append(
                {
                    'case': CASE,
                    'cell': 'ref-pandas3',
                    'runner': 'python-old',
                    'projection': projection,
                    'digest': helpers.ratchet.digest(projection),
                }
            )
            lines.sort(key=lambda x: x['case'])
        baseline.write_text(
            ''.join(json.dumps(x, sort_keys=True) + '\n' for x in lines)
        )
    commit(repo)
    done = gate(
        repo,
        '--base',
        landed,
        '--protected-paths',
        'contract/baselines/** docs/design/rust-core/register.yaml',
        *arguments,
    )
    if transition == 'approved':
        assert done.returncode == 0, done.stdout
        assert (
            f'{CASE}: python-old: success at materialize to success at predict, approved'
            in done.stdout
        )
    else:
        assert failed(done) == {'ratchet', 'compare'}, done.stdout
        assert 'against its effective expectation: outcome.stage' in done.stdout


def _runners_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        'runners_for_gate', ROOT / 'contract/tests/test_runners.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('transition', ['approved', 'unapproved'])
def test_an_approved_outcome_change_passes_contract_tests_comparison(
    repo: Path,
    tmp_path: Path,
    observed: dict[str, list[dict[str, Any]]],
    transition: str,
) -> None:
    r"""The comparison ``contract/tests`` makes (``test_runners``'s, CI's
    contract-python job) holds outcomes to the same effective expectations
    as the gate (T1.18): python-old's moved stage passes under an approved
    entry stating the exact transition, and fails without one."""
    landed = git(repo, 'rev-parse', 'HEAD')
    arguments = files(observed, tmp_path, _moved)
    if transition == 'approved':
        item = {
            'case': CASE,
            'cell': 'ref-pandas3',
            'runner': 'python-old',
            'old': 'success at materialize',
            'new': 'success at predict',
            'summary': 'seeded',
        }
        commit(
            repo,
            edit=lambda r: helpers.append_entry(
                r, {'id': 'SEED-1', 'cases': [CASE], 'case_outcomes': [item]}
            ),
        )
    else:
        commit(repo)
    runners = _runners_module()
    expect = runners.expectations(tmp_path, repo, ['--base', landed])
    assert expect[0] == '--expect'
    done = subprocess.run(
        [
            sys.executable,
            str(repo / 'scripts/rust_core/compare.py'),
            '--cases',
            str(repo / 'contract/cases'),
            '--reference',
            str(next((repo / 'contract/reference').glob('*.jsonl'))),
            *expect,
            *[a for a in arguments if a != '--observations'],
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if transition == 'approved':
        assert done.returncode == 0, done.stdout
    else:
        assert done.returncode == 1, done.stdout
        assert 'against its effective expectation: outcome.stage' in done.stdout


def test_a_bootstrap_the_checkpoint_s_tree_contradicts_is_refused(
    repo: Path, tmp_path: Path, observed: dict[str, list[dict[str, Any]]]
) -> None:
    r"""A candidate whose ratchet.py claims the bootstrap for a checkpoint
    holding ratchet.py does not get to run its own harness: the outer
    gate.sh reads the checkpoint's tree and refuses (T1.18)."""
    landed = git(repo, 'rev-parse', 'HEAD')

    def claim(r: Path) -> None:
        path = r / 'scripts/rust_core/ratchet.py'
        text = path.read_text()
        old = "'bootstrap' if found.bootstrap else 'harness'"
        assert old in text
        path.write_text(text.replace(old, "'bootstrap'"))

    commit(repo, edit=claim)
    done = gate(repo, '--base', landed, *files(observed, tmp_path))
    assert done.returncode == 1, done.stdout + done.stderr
    assert 'holds ratchet.py, so its harness judges, not the bootstrap' in (
        done.stderr
    )
    assert 'gate: judged by bootstrap' not in done.stdout
