# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The handoff in minutes, and no push under a running one (the owner's
"2026-10-22 evening: the five speed changes" and "2026-10-23 cut the
tooling back"):

- lint, then the package's own tests, each printed as it starts and ends
  with its time, the run stopping at the first failing step;
- every command the run starts reads the lane's interpreter as
  ``SDM_PYTHON``;
- a stamp or a failure is an event ``handoff.py wait`` returns on;
- the pre-push hook refuses a push while the package's handoff runs.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/rust_core'))


def _module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HANDOFF = _module(ROOT / 'scripts/rust_core/handoff.py', 't32_handoff_speed')
AFFECTED = HANDOFF.affected
ORDER = ['lint', 'own tests']
GIT_ENV = {
    'GIT_AUTHOR_NAME': 't',
    'GIT_AUTHOR_EMAIL': 't@example.invalid',
    'GIT_COMMITTER_NAME': 't',
    'GIT_COMMITTER_EMAIL': 't@example.invalid',
    'GIT_CONFIG_GLOBAL': os.devnull,
    'GIT_CONFIG_NOSYSTEM': '1',
}


def _git(where: Path, *args: str) -> str:
    done = subprocess.run(
        ['git', '-C', str(where), *args],
        env={**os.environ, **GIT_ENV},
        capture_output=True,
        text=True,
        check=True,
    )
    return done.stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str, str]:
    repo = tmp_path / 'repo'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'topic')
    _git(repo, 'commit', '-q', '--allow-empty', '-m', 'base')
    base = _git(repo, 'rev-parse', 'HEAD')
    (repo / 'notes.txt').write_text('a change\n')
    _git(repo, 'add', 'notes.txt')
    _git(repo, 'commit', '-q', '-m', 'T9.9: a change')
    return repo, base, _git(repo, 'rev-parse', 'HEAD')


def _fake_steps(
    monkeypatch: pytest.MonkeyPatch, failing: str | None
) -> tuple[list[str], dict[str, str | None]]:
    ran: list[str] = []
    seen: dict[str, str | None] = {}
    names = {'step_lint': 'lint', 'step_own_tests': 'own tests'}
    for function, name in names.items():

        def fake(*args: object, _name: str = name, **kw: object) -> object:
            ran.append(_name)
            seen[_name] = os.environ.get('SDM_PYTHON')
            step = HANDOFF.Step(_name)
            if _name == failing:
                step.problems.append(f'{_name} is seeded to fail')
            return step

        monkeypatch.setattr(HANDOFF, function, fake)
    return ran, seen


def _events(repo: Path) -> list[str]:
    path = repo / '.git' / 'sdm-handoff' / 'events.jsonl'
    return [json.loads(line)['event'] for line in path.read_text().splitlines()]


def test_the_cheap_steps_run_first_and_the_first_failure_ends_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, base, sha = _repo(tmp_path)
    monkeypatch.chdir(repo)
    monkeypatch.delenv('SDM_PYTHON', raising=False)
    ran, seen = _fake_steps(monkeypatch, failing='lint')
    args = ['run', '--package', 'T9.9', '--base', base, '--python', '/lane/py']
    assert HANDOFF.main(args) == 1
    said = capsys.readouterr().out
    assert ran == ['lint'], said
    assert 'start lint at ' in said and 'FAIL lint (' in said, said
    assert 'NO STAMP' in said and 'lint failed' in said, said
    assert not (repo / '.git/sdm-handoff' / f'{sha}.json').exists()
    assert not (repo / '.git/sdm-handoff/running/T9.9.json').exists()
    assert HANDOFF.main(['wait', sha, '--timeout', '1']) == 1

    ran, seen = _fake_steps(monkeypatch, failing=None)
    assert HANDOFF.main(args) == 0
    said = capsys.readouterr().out
    assert ran == ORDER, said
    assert all(seen[name] == '/lane/py' for name in ORDER), seen
    stamp = json.loads((repo / '.git/sdm-handoff' / f'{sha}.json').read_text())
    assert list(stamp['seconds']) == ORDER, stamp
    assert _events(repo) == ['start', 'failed', 'start', 'stamped']
    assert HANDOFF.main(['wait', sha, '--timeout', '1']) == 0


def test_a_push_while_the_package_s_handoff_runs_is_refused(
    tmp_path: Path,
) -> None:
    repo, _, sha = _repo(tmp_path)
    (repo / 'scripts').mkdir()
    (repo / 'scripts/rust_core').symlink_to(ROOT / 'scripts/rust_core')
    _git(repo, 'config', 'extensions.worktreeConfig', 'true')
    _git(repo, 'config', '--worktree', 'sdm.package', 'T9.9')
    run = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(120)', 'handoff.py']
    )
    try:
        held = repo / '.git/sdm-handoff/running/T9.9.json'
        held.parent.mkdir(parents=True)
        held.write_text(
            json.dumps(
                {
                    'pid': run.pid,
                    'sha': sha,
                    'package': 'T9.9',
                    'started': 'now',
                }
            )
        )
        done = subprocess.run(
            [
                'bash',
                str(ROOT / 'scripts/rust_core/prepush.sh'),
                '--pushed-refs',
            ],
            cwd=repo,
            input=f'refs/heads/topic {sha} refs/heads/topic {"0" * 40}\n',
            env={**os.environ, **GIT_ENV},
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert done.returncode == 1, done.stdout + done.stderr
        assert 'REFUSED: a handoff of T9.9 runs on' in done.stderr, done.stderr
        assert 'prepush summary' not in done.stdout, done.stdout
    finally:
        run.kill()
        run.wait()
    # A run that is gone holds nothing.
    done = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/rust_core/handoff.py'), 'running',
         '--package', 'T9.9'],
        cwd=repo, capture_output=True, text=True, timeout=60,
    )  # fmt: skip
    assert done.returncode == 0, done.stdout + done.stderr


def test_the_suites_run_with_a_short_temporary_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A lane's long TMPDIR put a test's socket path past AF_UNIX's limit.
    seen: list[str] = []
    argvs: list[list[str]] = []

    def fake(
        argv: list[str], cwd: Path, env: dict[str, str] | None = None
    ) -> object:
        if argv[1:3] == ['-c', 'import xdist']:
            return subprocess.CompletedProcess(argv, 1, '', '')
        assert env is not None
        seen.append(env['TMPDIR'])
        argvs.append(argv)
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(HANDOFF, 'run', fake)
    monkeypatch.setenv('TMPDIR', '/mnt/localssd/tmp/a-long-lane-directory/t32')
    step = HANDOFF.run_suites(
        HANDOFF.Step('own tests'),
        ROOT,
        [(*AFFECTED.PYTEST, 'tests/test_x.py'), ('python3', 'x_selftest.py')],
        'py',
    )
    assert step.ok and len(seen) == 2
    assert seen[0].startswith('/tmp/sdmh-') and len(seen[0]) < 30, seen
    assert not Path(seen[0]).exists()
    # pytest's base directly in it, not under pytest-of-<user>/pytest-<n>.
    base = argvs[0][argvs[0].index('--basetemp') + 1]
    assert base.startswith(f'{seen[0]}/p'), argvs


def test_two_packages_same_named_test_files_run_in_sessions_of_their_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # In one session they fail collection, "import file mismatch" (T3.8).
    ran: list[list[str]] = []

    def fake(
        argv: list[str], cwd: Path, env: dict[str, str] | None = None
    ) -> object:
        if argv[1:3] != ['-c', 'import xdist']:
            ran.append(argv)
        return subprocess.CompletedProcess(argv, 0, '', '')

    monkeypatch.setattr(HANDOFF, 'run', fake)
    client = 'packages/structured-data-models-client/tests/test_x.py'
    engine = 'packages/structured-data-models-relational-engine/tests/client/test_x.py'
    suites = [
        (*AFFECTED.PYTEST, client),
        (*AFFECTED.PYTEST, engine),
        (*AFFECTED.PYTEST, 'tests/test_y.py'),
    ]
    step = HANDOFF.run_suites(HANDOFF.Step('affected'), ROOT, suites, 'py')
    assert step.ok
    runs = [{a for a in argv if a.endswith('.py')} for argv in ran]
    assert {client} in runs and {engine} in runs and {'tests/test_y.py'} in runs
    assert len(ran) == 3, ran
