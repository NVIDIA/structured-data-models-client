#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replays CI's Python jobs locally, each in a clean virtual environment.

``prepush.sh --full`` runs this (decision E43). It reads
``.github/workflows/ci.yml`` as HEAD has it and, for each job it can
replay, makes a fresh virtual environment at the job's Python, checks HEAD
out into a temporary worktree, and runs the job's ``run`` steps there in order, with the job's
and the step's environment: its install list exactly as CI has it, then its
tests. A test that imports something the install list lacks fails here as
it would in CI (T1.9's ``yaml`` in the installed-wheel job, T1.10's
``jsonschema`` in the engine job), which a developer's own environment,
holding everything, never shows.

It replays HEAD and nothing else, so it refuses a worktree with uncommitted
changes or untracked files: a replay of HEAD would not have tested them.
Commit (or stash) first.

Which jobs: by default those whose block in ci.yml differs from the base,
that read a workflow-level variable (``$BINDING_FLOOR``, ``env.X``) whose
value differs from the base's, or whose steps name a path a file changed
since the base is under; ``--job`` names them instead and ``--all`` takes every
job it can replay. A job it can replay is one whose actions are only
checkout, setup-python, the artifact actions and the caches: the Rust jobs
and the docker wheel builds are left to CI. Where a job downloads the
``native-wheel`` artifact, the client wheel is built once for this machine
with ``native_wheel.py`` (or taken from ``--native-wheel DIR``); where it
downloads ``wheels``, the build job is replayed first in the same checkout.
Paths under ``/tmp`` in a step are moved under the replay's own directory.

The PostgreSQL service is not started: ``SDM_REQUIRE_POSTGRES`` and the DSN
are left out, so those tests skip, unless ``--postgres`` keeps the job's
values for a server the caller runs on localhost:5432 (user and password
``postgres``, database ``sdm_test``).

    replay_jobs.py [--base REF] [--job NAME ...] [--all] [--python 3.12]
                   [--native-wheel DIR] [--postgres] [--list] [--keep]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
CI = '.github/workflows/ci.yml'
PHASE = 'origin/rust-core-phase-1'
#: The actions a replayable job may use; anything else is left to CI.
REPLAYABLE_USES = (
    'actions/checkout@',
    'actions/setup-python@',
    'actions/download-artifact@',
    'actions/upload-artifact@',
    'actions/cache@',
    'actions/cache/restore@',
    'actions/cache/save@',
)
POSTGRES_ENV = ('SDM_REQUIRE_POSTGRES', 'SDM_POSTGRES_TEST_DSN')
EXPRESSION = re.compile(r'\$\{\{\s*(.*?)\s*\}\}')


class Unreplayable(Exception):
    pass


def git(*args: str, cwd: Path | None = None) -> str:
    return subprocess.run(
        ['git', *args],
        cwd=cwd or REPO,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def workflow(text: str) -> dict[str, Any]:
    return yaml.safe_load(text)['jobs']


def why_unreplayable(job: dict[str, Any]) -> str | None:
    steps = job.get('steps', [])
    if not any(
        str(s.get('uses', '')).startswith('actions/setup-python@')
        for s in steps
    ):
        return 'sets up no Python'
    for step in steps:
        uses = str(step.get('uses', ''))
        if uses and not uses.startswith(REPLAYABLE_USES):
            return f'uses {uses.split("@")[0]}'
        if 'docker ' in str(step.get('run', '')):
            return 'runs docker'
    return None


def named_paths(job: dict[str, Any]) -> set[str]:
    """The repository paths a job's steps name."""
    found: set[str] = set()
    for step in job.get('steps', []):
        text = str(step.get('run', ''))
        for token in re.findall(r'[A-Za-z0-9_./*-]+', text):
            token = token.strip('./').split('[')[0].rstrip('/')
            if token and '*' not in token and (REPO / token).exists():
                found.add(token)
    return found


def changed_files(base: str) -> list[str]:
    return sorted(set(git('diff', '--name-only', f'{base}...HEAD').split()))


def reads_variable(job: dict[str, Any], name: str) -> bool:
    """Whether a job's steps or environment read a workflow variable."""
    text = yaml.safe_dump(job)
    return bool(
        re.search(
            rf'\$\{{?{re.escape(name)}\b|\benv\.{re.escape(name)}\b', text
        )
    )


def select(
    jobs: dict[str, Any],
    base_jobs: dict[str, Any],
    changed: list[str],
    env: dict[str, Any] | None = None,
    base_env: dict[str, Any] | None = None,
) -> dict[str, str]:
    """The replayable jobs a change touches, each with the reason."""
    env, base_env = env or {}, base_env or {}
    moved = sorted(
        k for k in {*env, *base_env} if env.get(k) != base_env.get(k)
    )
    chosen: dict[str, str] = {}
    for name, job in jobs.items():
        if why_unreplayable(job):
            continue
        if base_jobs.get(name) != job:
            chosen[name] = 'its block in ci.yml changed'
            continue
        read = [k for k in moved if reads_variable(job, k)]
        if read:
            chosen[name] = f'it reads {", ".join(read)}, which changed'
            continue
        for path in sorted(named_paths(job)):
            hit = next(
                (f for f in changed if f == path or f.startswith(path + '/')),
                None,
            )
            if hit:
                chosen[name] = f'it names {path}, and {hit} changed'
                break
    return chosen


def with_dependencies(jobs: dict[str, Any], names: list[str]) -> list[str]:
    """The jobs in an order that runs a job producing ``wheels`` first."""
    order: list[str] = []

    def visit(name: str) -> None:
        if name in order:
            return
        job = jobs[name]
        for step in job.get('steps', []):
            if str(step.get('uses', '')).startswith(
                'actions/download-artifact@'
            ):
                artifact = step.get('with', {}).get('name')
                if artifact == 'wheels':
                    needs = job.get('needs', [])
                    for need in [needs] if isinstance(needs, str) else needs:
                        visit(need)
        order.append(name)

    for name in names:
        visit(name)
    return order


def substitute(text: str, values: dict[str, str]) -> str:
    def one(match: re.Match[str]) -> str:
        expression = match.group(1)
        if expression in values:
            return values[expression]
        raise Unreplayable(f'cannot replay the expression {match.group(0)}')

    return EXPRESSION.sub(one, text)


class Replay:
    def __init__(
        self, args: argparse.Namespace, root: Path, env: dict[str, Any]
    ) -> None:
        self.args = args
        self.workflow_env = env
        self.root = root
        self.checkout = root / 'checkout'
        self.tmp = root / 'tmp'
        self.logs = root / 'logs'
        self.native: Path | None = (
            Path(args.native_wheel).resolve() if args.native_wheel else None
        )
        self.tmp.mkdir(parents=True)
        self.logs.mkdir()

    def native_wheel(self) -> Path:
        if self.native is None:
            out = self.root / 'native'
            print(f'  building the client wheel for this machine into {out}')
            subprocess.run(
                [
                    'uv', 'run', '--no-project', '--python', '3.12',
                    '--with', 'maturin==1.15.0', 'python',
                    'scripts/rust_core/native_wheel.py', '--out', str(out),
                ],
                cwd=self.checkout,
                check=True,
            )  # fmt: skip
            self.native = out
        return self.native

    def python_of(self, job: dict[str, Any]) -> str:
        for step in job['steps']:
            if str(step.get('uses', '')).startswith('actions/setup-python@'):
                version = str(step.get('with', {}).get('python-version', ''))
                return self.args.python if '${{' in version else version
        raise Unreplayable('no setup-python step')

    def run_job(self, name: str, job: dict[str, Any]) -> tuple[bool, str]:
        python = self.python_of(job)
        venv = self.root / 'venvs' / name
        subprocess.run(
            ['uv', 'venv', '--quiet', '--seed', '--python', python, str(venv)],
            check=True,
        )
        values = {
            'matrix.python-version': python,
            'runner.os': 'Linux',
            'github.workspace': str(self.checkout),
        }
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ('VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME')
            and not k.startswith('UV_')
        }
        env.update(
            {
                'PATH': f'{venv / "bin"}{os.pathsep}{os.environ["PATH"]}',
                'VIRTUAL_ENV': str(venv),
                'GITHUB_WORKSPACE': str(self.checkout),
                'RUNNER_TEMP': str(self.tmp),
                # Files a step may write its outputs and summary to; nothing
                # here reads them back.
                'GITHUB_OUTPUT': str(self.tmp / 'github-output'),
                'GITHUB_ENV': str(self.tmp / 'github-env'),
                'GITHUB_PATH': str(self.tmp / 'github-path'),
                'GITHUB_STEP_SUMMARY': str(self.tmp / 'github-summary'),
                'PIP_DISABLE_PIP_VERSION_CHECK': '1',
            }
        )
        for key, value in self.workflow_env.items():
            # A workflow variable only CI can compute (SAVE_CACHES, which
            # only the cache actions read) is left out.
            try:
                env[key] = substitute(str(value), values)
            except Unreplayable:
                env.pop(key, None)
        for key, value in (job.get('env') or {}).items():
            env[key] = substitute(str(value), values)
        if not self.args.postgres:
            for key in POSTGRES_ENV:
                env.pop(key, None)
        values.update({f'env.{k}': v for k, v in env.items()})
        default_dir = (
            job.get('defaults', {}).get('run', {}).get('working-directory', '.')
        )
        log = self.logs / f'{name}.log'
        with log.open('w') as out:
            for index, step in enumerate(job['steps'], 1):
                uses = str(step.get('uses', ''))
                if uses.startswith('actions/download-artifact@'):
                    self.download(step.get('with', {}))
                    continue
                if uses or 'run' not in step:
                    continue
                if 'if' in step:
                    raise Unreplayable(f'step {index} has a condition')
                script = substitute(str(step['run']), values)
                if script.lstrip().startswith('sudo '):
                    print(f'  step {index}: left out (sudo): {script[:60]}')
                    continue
                script = script.replace('/tmp/', f'{self.tmp}/')
                where = str(step.get('working-directory', default_dir))
                cwd = (
                    self.tmp / where.removeprefix('/tmp').lstrip('/')
                    if where.startswith('/tmp')
                    else self.checkout / where
                )
                cwd.mkdir(parents=True, exist_ok=True)
                step_env = dict(env)
                for key, value in (step.get('env') or {}).items():
                    step_env[key] = substitute(str(value), values)
                title = step.get('name') or script.strip().splitlines()[0]
                out.write(f'\n=== step {index}: {title}\n')
                out.flush()
                started = time.monotonic()
                done = subprocess.run(
                    ['bash', '--noprofile', '--norc', '-e', '-c', script],
                    cwd=cwd,
                    env=step_env,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                )
                seconds = time.monotonic() - started
                print(f'  step {index} ({seconds:.0f}s): {title[:70]}')
                if done.returncode != 0:
                    out.flush()
                    tail = log.read_text().splitlines()[-40:]
                    print('\n'.join(f'    | {line}' for line in tail))
                    return False, f'step {index} failed: {title[:70]}'
        return True, f'{len(job["steps"])} steps'

    def download(self, spec: dict[str, Any]) -> None:
        name, path = (
            spec.get('name'),
            self.checkout / str(spec.get('path', '.')),
        )
        if name == 'native-wheel':
            if path.exists():
                shutil.rmtree(path)
            shutil.copytree(self.native_wheel(), path)
        elif name == 'wheels':
            if not path.is_dir():
                raise Unreplayable(
                    'needs the wheels artifact, which the build job makes; '
                    'replay build first (--job build)'
                )
        else:
            raise Unreplayable(f'cannot provide the artifact {name}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--base', help=f'default: the merge base with {PHASE}')
    parser.add_argument('--job', action='append', default=[])
    parser.add_argument('--all', action='store_true')
    parser.add_argument(
        '--python', default='3.12', help='for a job with a Python matrix'
    )
    parser.add_argument('--native-wheel', metavar='DIR')
    parser.add_argument('--postgres', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--keep', action='store_true')
    parser.add_argument(
        '--root', help="the repository (default: this script's)", type=Path
    )
    args = parser.parse_args(argv)
    # Progress as it happens, also when the output is a pipe or a file.
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    global REPO
    if args.root:
        REPO = args.root.resolve()

    dirty = git('status', '--porcelain', '--untracked-files=all')
    if dirty.strip():
        print(
            'replay: refused: the worktree has uncommitted changes or untracked '
            'files, and a replay runs HEAD only, so it would not test them. '
            'Commit or stash them first:\n' + dirty.rstrip()
        )
        return 2
    # The jobs as HEAD defines them: the tree the replay checks out.
    text = git('show', f'HEAD:{CI}')
    jobs = workflow(text)
    if args.job or args.all:
        names = list(jobs) if args.all else args.job
        unknown = [n for n in names if n not in jobs]
        if unknown:
            parser.error(f'no such job: {", ".join(unknown)}')
        chosen = {
            n: 'named'
            for n in names
            if not (args.all and why_unreplayable(jobs[n]))
        }
    else:
        base = args.base or git('merge-base', 'HEAD', PHASE).strip()
        try:
            base_text = git('show', f'{base}:{CI}')
        except subprocess.CalledProcessError:
            base_text = 'jobs: {}'
        chosen = select(
            jobs,
            workflow(base_text),
            changed_files(base),
            yaml.safe_load(text).get('env') or {},
            yaml.safe_load(base_text).get('env') or {},
        )
    for name, job in jobs.items():
        reason = why_unreplayable(job)
        if name in chosen and reason:
            print(f'cannot replay {name}: it {reason}')
            return 2
    order = with_dependencies(jobs, list(chosen))
    if not order:
        print('replay: no job to replay for this change')
        return 0
    for name in order:
        print(f'replay {name}: {chosen.get(name, "its wheels are needed")}')
    if args.list:
        return 0

    # Resolved: on macOS the temporary directory is reached through a symlink,
    # and a test comparing a module's file with a warning's would see both.
    root = Path(tempfile.mkdtemp(prefix='sdm-replay-')).resolve()
    replay = Replay(args, root, yaml.safe_load(text).get('env') or {})
    git('worktree', 'add', '--quiet', '--detach', str(replay.checkout), 'HEAD')
    results: list[tuple[str, bool, str]] = []
    try:
        for name in order:
            print(f'== {name}')
            try:
                ok, detail = replay.run_job(name, jobs[name])
            except (Unreplayable, subprocess.CalledProcessError) as error:
                ok, detail = False, str(error)
            results.append((name, ok, detail))
            if not ok and name not in chosen:
                break
    finally:
        if not args.keep:
            git('worktree', 'remove', '--force', str(replay.checkout))
            shutil.rmtree(root, ignore_errors=True)
    for name, ok, detail in results:
        print(f'{"ok  " if ok else "FAIL"} {name}: {detail}')
    if args.keep:
        print(f'kept {root}; remove the worktree with `git worktree remove '
              f'--force {replay.checkout}`')  # fmt: skip
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == '__main__':
    sys.exit(main())
