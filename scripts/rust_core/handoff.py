#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The handoff check: lint and the package's own tests, and a stamp for
the exact SHA that passed (T3.0, item 4; cut back by the owner,
"2026-10-23 cut the tooling back").

    handoff.py run --package ID [--previous SHA] [--python PY] [--base REV]
    handoff.py check SHA
    handoff.py wait SHA [--timeout SECONDS]
    handoff.py running --package ID

``run`` checks the committed ``HEAD`` of a clean worktree and, only when
both steps pass and HEAD and the tree are unchanged after them, writes
``<git common dir>/sdm-handoff/<sha>.json`` (``{"sha", "package", "ok":
true, "time", "previous", "steps", "seconds"}``); a failed step exits 1 and
writes no stamp. It runs once, on the SHA both reviewers approve; reviews
start at push and need no stamp (the owner's "2026-10-23 stamp only at
landing"). The stamp is what ``land.py``, its ``stack-land`` and
``auto_land.py`` need, beside the two APPROVE reports; the broad suites are the nightly CI run's and the phase
end's. Run it with the tools' interpreter; ``--python`` is the lane's
(default the same), and every command the run starts sees it as
``SDM_PYTHON``. The steps, each printed as it starts and ends with its
time, stopping at the first that fails:

- ``lint``: ``ruff check`` and ``ruff format --check`` on the changed Python
  files, ``cargo fmt --check`` on the changed crates, ``mypy`` on each
  changed package's ``src``, ``l0_scan.py``, the hygiene check, and
  ``cargo deny --locked check`` when ``rust/`` changed;
- ``own tests``: the test files the package adds or changes (pytest files,
  one session per test root as CI runs them; self-tests; each crate test by
  its own ``--test``), each run in a short temporary directory it removes.

In a fix round whose ``--previous`` SHA has a stamp, ``own tests`` takes
the files changed since it.

While it runs, ``run`` holds ``<git common dir>/sdm-handoff/running/<ID>.json``,
which ``running`` reads: it exits 1 naming a live run of the package.
The pre-push hook refuses a push from the package's worktree while one
runs, and a second ``run`` of the package is refused. A signal (SIGTERM,
SIGHUP) stops the run through its clean-up. Each start, stamp, failure and
stop is a line of ``<git common dir>/sdm-handoff/events.jsonl``; ``wait
SHA`` blocks until SHA is stamped (exit 0) or its run fails or stops (exit
1), so the landing follows the moment the stamp is written (exit 3 after
``--timeout``).

``check SHA`` exits 0 when SHA has a passing stamp, else 1, naming why.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import affected  # noqa: E402
import judging  # noqa: E402
from phases import PACKAGE_ID, PhaseError, phase_branch  # noqa: E402

STAMPS = 'sdm-handoff'
RUNNING = 'running'
#: Where the suites' temporary files go: short, as CI's are.
SHORT_TMP = '/tmp'
EVENTS = 'events.jsonl'
CRATE_TEST = re.compile(r'rust/crates/([^/]+)/tests/([^/]+)\.rs$')
#: A crate's or the binding's Rust source, whose ``#[cfg(test)]`` tests
#: run with its crate's ``--lib``.
INLINE = re.compile(r'rust/(?:crates|bindings)/([^/]+)/src/.+\.rs$')


class Failure(Exception):
    """A usage problem: nothing was checked."""


# --- stamps ----------------------------------------------------------------------


def stamp_path(common: Path, sha: str) -> Path:
    return Path(common) / STAMPS / f'{sha}.json'


def stamp_problem(common: Path, sha: str) -> str | None:
    """None when ``sha`` has a passing stamp in the clone whose common git
    directory is ``common``; else why not."""
    if not re.fullmatch(r'[0-9a-f]{40}', sha):
        return f'{sha!r} is not a full 40-character SHA'
    path = stamp_path(common, sha)
    try:
        stamp = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return f'no handoff stamp for {sha[:12]} (handoff.py run on that SHA)'
    except (OSError, ValueError) as error:
        return f'the stamp {path} is unreadable: {error}'
    if not isinstance(stamp, dict) or stamp.get('sha') != sha:
        return f'the stamp {path} names another SHA'
    if stamp.get('ok') is not True:
        return f'the stamp {path} does not say ok'
    return None


# --- the running file and the events ---------------------------------------------


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec='seconds'
    )


def running_path(common: Path, package: str) -> Path:
    return Path(common) / STAMPS / RUNNING / f'{package}.json'


def alive(pid: object) -> bool:
    """Whether ``pid`` is a running handoff, not a process that reused it."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        command = Path(f'/proc/{pid}/cmdline').read_bytes()
    except OSError:
        return False
    return b'handoff.py' in command


def running_problem(common: Path, package: str) -> str | None:
    """None unless a handoff of ``package`` runs in this clone; else a line
    naming it. A file left by a run that ended without its clean-up (a
    SIGKILL) names a dead process and counts for nothing."""
    path = running_path(common, package)
    try:
        held = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if not isinstance(held, dict) or not alive(held.get('pid')):
        return None
    return (
        f'a handoff of {package} runs on {str(held.get("sha"))[:12]} '
        f'(process {held.get("pid")}, since {held.get("started")}, in '
        f'{held.get("worktree")}); a push now would start a new round under it: '
        'wait for its stamp (handoff.py wait <sha>) or stop it first'
    )


def hold_running(common: Path, package: str, sha: str, root: Path) -> Path:
    problem = running_problem(common, package)
    if problem:
        raise Failure(problem)
    path = running_path(common, package)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                'pid': os.getpid(),
                'sha': sha,
                'package': package,
                'started': now(),
                'worktree': str(root),
            }
        )
        + '\n',
        encoding='utf-8',
    )
    return path


def event(
    common: Path, kind: str, sha: str, package: str, **more: object
) -> None:
    """One line of ``events.jsonl``: what a waiter reads."""
    path = Path(common) / STAMPS / EVENTS
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {'time': now(), 'event': kind, 'sha': sha, 'package': package, **more}
    )
    with open(path, 'a', encoding='utf-8') as handle:
        handle.write(line + '\n')


def last_event(common: Path, sha: str) -> dict[str, object] | None:
    path = Path(common) / STAMPS / EVENTS
    try:
        lines = path.read_text(encoding='utf-8').splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            seen = json.loads(line)
        except ValueError:
            continue
        if isinstance(seen, dict) and seen.get('sha') == sha:
            return seen
    return None


# --- git --------------------------------------------------------------------------


def run(
    argv: Sequence[str], cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )


def git(root: Path, *args: str) -> str:
    done = run(['git', *args], root)
    if done.returncode != 0:
        raise Failure(f'git {" ".join(args)}: {done.stderr.strip()}')
    return done.stdout


def common_dir(root: Path) -> Path:
    return Path(
        git(
            root, 'rev-parse', '--path-format=absolute', '--git-common-dir'
        ).strip()
    )


def phase_base(root: Path, given: str | None) -> str:
    if given:
        return git(root, 'rev-parse', f'{given}^{{commit}}').strip()
    branch = run(
        ['git', 'symbolic-ref', '--short', '-q', 'HEAD'], root
    ).stdout.strip()
    try:
        phase = phase_branch(branch=branch or None)
    except PhaseError as error:
        raise Failure(str(error)) from None
    for ref in (f'refs/remotes/origin/{phase}', f'refs/heads/{phase}'):
        done = run(['git', 'merge-base', 'HEAD', ref], root)
        if done.returncode == 0:
            return stack_floor(root, done.stdout.strip())
    raise Failure(f'no {phase} here to diff against; fetch it or give --base')


def stack_floor(root: Path, base: str) -> str:
    """The newest landing commit between ``base`` and HEAD, else ``base``:
    a package stacked on others' landings (``land.py --stack``) is its own
    commits above them (Claude r1 P2-5), not the stack below."""
    out = git(
        root, 'log', '--first-parent', '--format=%H%x00%s', f'{base}..HEAD'
    )
    for line in out.splitlines():
        sha, _, subject = line.partition('\0')
        if not subject.startswith('Land '):
            continue
        tree = git(root, 'rev-parse', f'{sha}^{{tree}}').strip()
        parent = run(
            ['git', 'rev-parse', f'{sha}^^{{tree}}'], root
        ).stdout.strip()
        if tree == parent:
            return sha
    return base


# --- the steps --------------------------------------------------------------------


class Step:
    def __init__(self, name: str) -> None:
        self.name = name
        self.problems: list[str] = []
        self.notes: list[str] = []
        self.seconds = 0.0

    @property
    def ok(self) -> bool:
        return not self.problems


def tail(text: str, lines: int = 25) -> str:
    return '\n'.join(text.strip().splitlines()[-lines:])


def is_test(path: str) -> bool:
    return bool(
        judging.matches(affected.IS_TEST, path)
        or path.endswith('selftest.py')
        or CRATE_TEST.match(path)
    )


def own_suites(root: Path, touched: list[str]) -> list[affected.Suite]:
    """The test files among ``touched`` as suites: a pytest file (the
    binding's under ``affected.BINDING``), a self-test, a crate test by its
    own ``--test``, and a crate's or the binding's source file holding
    ``#[cfg(test)]`` unit tests by its crate's ``--lib`` (Codex r3 P1-2)."""
    found: list[affected.Suite] = []
    for path in touched:
        if not (root / path).is_file():
            continue
        inline = INLINE.match(path)
        if inline and '#[cfg(test)]' in (root / path).read_text(
            encoding='utf-8', errors='replace'
        ):
            suite: affected.Suite = (
                'cargo', 'test', '--manifest-path', 'rust/Cargo.toml',
                '-p', inline.group(1), '--lib',
            )  # fmt: skip
            if suite not in found:
                found.append(suite)
            continue
        if not is_test(path):
            continue
        crate = CRATE_TEST.match(path)
        if crate:
            suite = (
                'cargo', 'test', '--manifest-path', 'rust/Cargo.toml',
                '-p', crate.group(1), '--test', crate.group(2),
            )  # fmt: skip
        elif path.endswith('selftest.py'):
            suite = ('python3', path)
        elif path.startswith('rust/bindings/'):
            suite = affected.BINDING
        else:
            suite = (*affected.PYTEST, path)
        if suite not in found:
            found.append(suite)
    return found


def step_own_tests(root: Path, touched: list[str], python: str) -> Step:
    step = Step('own tests')
    suites = own_suites(root, touched)
    if not suites:
        step.notes.append('the package adds or changes no test')
        return step
    return run_suites(step, root, suites, python)


def sessions(files: list[str]) -> list[tuple[str, list[str]]]:
    """``files`` by the test root each runs from in CI: a package's
    (``packages/<name>/``), ``contract/``, ``tests/``, else the root."""
    grouped: dict[str, list[str]] = {}
    for path in files:
        parts = path.split('/')
        if parts[0] == 'packages' and len(parts) > 2:
            where = '/'.join(parts[:2]) + '/'
        elif len(parts) > 1 and parts[0] in ('contract', 'tests'):
            where = parts[0] + '/'
        else:
            where = './'
        grouped.setdefault(where, []).append(path)
    return sorted(grouped.items())


def run_suites(
    step: Step, root: Path, suites: list[affected.Suite], python: str
) -> Step:
    """The pytest files in one run, two workers a file each when
    pytest-xdist is there (the brief's ``-n 2``), then each other suite in
    turn, stopping at the first that fails."""
    prefix = affected.PYTEST
    files = [s[len(prefix) :] for s in suites if s[: len(prefix)] == prefix]
    others = [s for s in suites if s[: len(prefix)] != prefix]
    batches: list[tuple[str, list[str]]] = []
    # A short temporary directory, as CI's /tmp is, with pytest's base
    # right in it: under a lane's long TMPDIR, and under pytest's own
    # pytest-of-<user>/pytest-<n>, a test's socket path passed AF_UNIX's
    # 108 bytes (T3.2's clean_tmp tests failed so in the handoff and pass in
    # CI).
    short = Path(tempfile.mkdtemp(prefix='sdmh-', dir=SHORT_TMP))
    if files:
        flat = [f for group in files for f in group]
        workers = (
            ['-n', '2', '--dist', 'loadfile']
            if run([python, '-c', 'import xdist'], root).returncode == 0
            else []
        )
        # One session per test root, as CI runs them: two packages' test
        # files of one name fail collection together ("import file
        # mismatch", the client's and the engine's
        # test_credential_redaction.py; T3.8).
        for n, (where, group) in enumerate(sessions(flat)):
            batches.append(
                (
                    f'pytest on {len(group)} test file(s) under {where}',
                    [
                        python,
                        *prefix[1:],
                        *workers,
                        '--basetemp',
                        str(short / f'p{n}'),
                        *group,
                    ],
                )  # fmt: skip
            )
    kit = os.environ.get('SDM_KIT_PYTHON', '')
    for suite in others:
        if suite[0] == affected.KIT and not kit:
            step.problems.append(
                f"{' '.join(suite)}: set SDM_KIT_PYTHON to the lane's test-ops "
                'interpreter (cells.sh prints it)'
            )
            continue
        argv = [
            {'python3': python, affected.KIT: kit}.get(part, part)
            if n == 0
            else part
            for n, part in enumerate(suite)
        ]
        batches.append((' '.join(suite), argv))
    env = {**os.environ, 'TMPDIR': str(short)}
    try:
        for shown, argv in batches:
            if step.problems:
                break
            begun = time.monotonic()
            done = run(argv, root, env)
            took = f'{time.monotonic() - begun:.0f}s'
            if done.returncode != 0:
                step.problems.append(
                    f'{shown} failed ({took}):\n'
                    + tail(done.stdout + done.stderr)
                )
            else:
                step.notes.append(f'{shown}: ok ({took})')
    finally:
        shutil.rmtree(short, ignore_errors=True)
    return step


def ruff() -> list[str] | None:
    beside = Path(sys.executable).parent / 'ruff'
    if beside.is_file():
        return [str(beside)]
    found = shutil.which('ruff')
    if found:
        return [found]
    uvx = shutil.which('uvx')
    if uvx:
        return [uvx, '--quiet', '--from', 'ruff==0.16.*', 'ruff']
    return None


def mypy(python: str, root: Path) -> list[str]:
    """mypy judging the lane's installed packages, as CI's typecheck job
    does: whichever interpreter has mypy (the tools', the lane's, else
    uvx's) runs it with ``--python-executable`` the lane's interpreter, so
    numpy, pandas and the client's other dependencies are the lane's
    (Claude r2 P2-1)."""
    target = ['--python-executable', python]
    for interpreter in (sys.executable, python):
        if run([interpreter, '-c', 'import mypy'], root).returncode == 0:
            return [interpreter, '-m', 'mypy', *target]
    uvx = shutil.which('uvx')
    if uvx:
        return [uvx, '--quiet', 'mypy', *target]
    return [sys.executable, '-m', 'mypy', *target]


def step_lint(root: Path, touched: list[str], python: str) -> Step:
    step = Step('lint')
    present = [p for p in touched if (root / p).is_file()]
    python_files = [p for p in present if p.endswith('.py')]
    if python_files:
        tool = ruff()
        if tool is None:
            step.problems.append('no ruff here (nor uvx to fetch it)')
        else:
            for args in (['check'], ['format', '--check']):
                done = run([*tool, *args, *python_files], root)
                if done.returncode != 0:
                    step.problems.append(
                        f'ruff {" ".join(args)}:\n'
                        + tail(done.stdout + done.stderr)
                    )
    crates = sorted(
        {
            m.group(1)
            for p in present
            if (m := re.match(r'rust/crates/([^/]+)/', p))
        }
    )
    for crate in crates:
        done = run(
            [
                'cargo',
                'fmt',
                '--manifest-path',
                'rust/Cargo.toml',
                '-p',
                crate,
                '--',
                '--check',
            ],
            root,
        )
        if done.returncode != 0:
            step.problems.append(
                f'cargo fmt -p {crate}:\n' + tail(done.stdout + done.stderr)
            )
    packages = sorted(
        {
            m.group(1)
            for p in python_files
            if (m := re.match(r'(packages/[^/]+/src)/', p))
        }
    )
    for source in packages:
        done = run([*mypy(python, root), source], root)
        if done.returncode != 0:
            step.problems.append(
                f'mypy {source}:\n' + tail(done.stdout + done.stderr)
            )
    for script, args in (
        ('scripts/rust_core/l0_scan.py', ()),
        ('scripts/check_repo_hygiene.py', ('all',)),
    ):
        if (root / script).is_file():
            done = run([sys.executable, str(root / script), *args], root)
            if done.returncode != 0:
                step.problems.append(
                    f'{Path(script).name}:\n' + tail(done.stdout + done.stderr)
                )
    if any(p.startswith('rust/') for p in touched) and shutil.which(
        'cargo-deny'
    ):
        done = run(['cargo', 'deny', '--locked', 'check'], root / 'rust')
        if done.returncode != 0:
            step.problems.append(
                'cargo deny --locked check:\n' + tail(done.stdout + done.stderr)
            )
    elif any(p.startswith('rust/') for p in touched):
        step.problems.append(
            'the package changes rust/, and cargo-deny is not here'
        )
    if not (python_files or crates):
        step.notes.append('no Python or Rust file changed')
    return step


# --- the commands -----------------------------------------------------------------


def changed(root: Path, since: str) -> list[str]:
    return [
        p
        for p in git(root, 'diff', '--name-only', '-z', since, 'HEAD').split(
            '\0'
        )
        if p
    ]


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(git(Path.cwd(), 'rev-parse', '--show-toplevel').strip())
    status = git(root, 'status', '--porcelain', '--untracked-files=all')
    if status.strip():
        raise Failure(
            'the worktree has uncommitted or untracked files; a stamp is for '
            'a commit: commit them first'
        )
    if not PACKAGE_ID.fullmatch(args.package):
        raise Failure(f'{args.package!r} is not a package ID')
    sha = git(root, 'rev-parse', 'HEAD').strip()
    base = phase_base(root, args.base)
    touched = changed(root, base)
    common = common_dir(root)
    # A fix round on a stamped previous round tests what changed since it;
    # lint still takes the whole package.
    since, recent = base, touched
    if args.previous:
        previous = git(root, 'rev-parse', f'{args.previous}^{{commit}}').strip()
        if stamp_problem(common, previous) is None:
            since, recent = previous, changed(root, previous)
    python = args.python or sys.executable
    # Every command the run starts reads the lane's interpreter (the Rust
    # tests that load cases run SDM_PYTHON, else a bare python3 that lacks
    # the loader's packages; T3.3's fourth handoff).
    if args.python:
        os.environ['SDM_PYTHON'] = python
    else:
        os.environ.setdefault('SDM_PYTHON', python)
    held = hold_running(common, args.package, sha, root)
    started = time.monotonic()
    steps: list[Step] = []
    outcome = 'stopped'
    try:
        finish_on_signals()
        event(common, 'start', sha, args.package, pid=os.getpid())
        print(
            f'handoff of {args.package} at {sha} ({len(touched)} file(s) since '
            f'{base[:12]}; own tests on {len(recent)} since {since[:12]})',
            flush=True,
        )
        plan = [
            ('lint', lambda: step_lint(root, touched, python)),
            ('own tests', lambda: step_own_tests(root, recent, python)),
        ]
        for name, step_of in plan:
            print(f'start {name} at {now()}', flush=True)
            begun = time.monotonic()
            step = step_of()
            step.seconds = time.monotonic() - begun
            steps.append(step)
            show(step)
            if not step.ok:
                break
        took = round(time.monotonic() - started)
        if not all(step.ok for step in steps):
            failed = next(s for s in steps if not s.ok)
            outcome = 'failed'
            event(
                common,
                'failed',
                sha,
                args.package,
                step=failed.name,
                seconds=took,
            )
            print(
                f'NO STAMP ({took}s): {failed.name} failed; fix it and run '
                'handoff.py again on the new SHA',
                flush=True,
            )
            return 1
        # The steps read the live worktree: a stamp is for the SHA they ran
        # on, so HEAD must still be it and the tree still clean (Claude r1
        # P2-8).
        moved = changed_under(root, sha)
        if moved:
            outcome = 'failed'
            event(
                common,
                'failed',
                sha,
                args.package,
                step='unchanged',
                seconds=took,
            )
            print(
                f'NO STAMP ({took}s): {moved} while the steps ran, so they did '
                f'not test {sha[:12]} alone; run handoff.py again on a clean '
                'worktree',
                flush=True,
            )
            return 1
        path = stamp_path(common, sha)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    'sha': sha,
                    'package': args.package,
                    'ok': True,
                    'time': now(),
                    'previous': args.previous,
                    'steps': {s.name: s.notes for s in steps},
                    'seconds': {s.name: round(s.seconds) for s in steps},
                },
                indent=1,
            )
            + '\n',
            encoding='utf-8',
        )
        outcome = 'stamped'
        event(common, 'stamped', sha, args.package, seconds=took)
        print(f'STAMPED {sha} ({took}s): {path}', flush=True)
        return 0
    finally:
        if outcome == 'stopped':
            event(common, 'stopped', sha, args.package)
        held.unlink(missing_ok=True)


def changed_under(root: Path, sha: str) -> str | None:
    """What changed in the worktree at ``root`` since a run on ``sha``
    began: HEAD moved, or the tree is no longer clean; None when neither."""
    head = git(root, 'rev-parse', 'HEAD').strip()
    if head != sha:
        return f'HEAD moved to {head[:12]}'
    if git(root, 'status', '--porcelain', '--untracked-files=all').strip():
        return 'the worktree changed'
    return None


def show(step: Step) -> None:
    print(
        f'{"ok  " if step.ok else "FAIL"} {step.name} ({step.seconds:.0f}s)',
        flush=True,
    )
    for note in step.notes:
        print(f'     {note}', flush=True)
    for problem in step.problems:
        print('     ' + problem.replace('\n', '\n     | '), flush=True)


class Stopped(BaseException):
    r"""The run was stopped by a signal: raised where it runs, so every
    ``finally`` removes what the run made (its scratch, the previous
    round's worktree) before the process ends (T3.2: a stopped run left its
    ``previous`` worktree registered and dirty, which only a forced removal
    could take away)."""


def finish_on_signals() -> None:
    r"""SIGTERM and SIGHUP stop the run as Ctrl-C does, through its
    ``finally`` blocks, rather than ending the process where it stands."""

    def stop(signum: int, frame: object) -> None:
        raise Stopped(f'stopped by signal {signum}')

    for name in ('SIGTERM', 'SIGHUP'):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), stop)


def cmd_check(args: argparse.Namespace) -> int:
    root = Path(git(Path.cwd(), 'rev-parse', '--show-toplevel').strip())
    done = run(
        ['git', 'rev-parse', '--verify', '--quiet', f'{args.sha}^{{commit}}'],
        root,
    )
    sha = done.stdout.strip() if done.returncode == 0 else args.sha
    why = stamp_problem(common_dir(root), sha)
    if why:
        print(f'NO STAMP: {why}')
        return 1
    print(f'STAMPED {sha}')
    return 0


def cmd_wait(args: argparse.Namespace) -> int:
    """Blocks until SHA is stamped (0), or its last run failed or stopped
    (1), or ``--timeout`` seconds pass (3)."""
    root = Path(git(Path.cwd(), 'rev-parse', '--show-toplevel').strip())
    sha = git(root, 'rev-parse', f'{args.sha}^{{commit}}').strip()
    common = common_dir(root)
    deadline = time.monotonic() + args.timeout
    while True:
        if stamp_problem(common, sha) is None:
            print(f'STAMPED {sha}', flush=True)
            return 0
        seen = last_event(common, sha)
        if seen is not None and seen.get('event') in ('failed', 'stopped'):
            print(
                f'NO STAMP {sha}: the run {seen.get("event")}'
                + (f' at {seen["step"]}' if seen.get('step') else ''),
                flush=True,
            )
            return 1
        if time.monotonic() >= deadline:
            print(f'NO STAMP {sha} after {args.timeout:.0f}s', flush=True)
            return 3
        time.sleep(min(args.poll, max(deadline - time.monotonic(), 0.1)))


def cmd_running(args: argparse.Namespace) -> int:
    root = Path(git(Path.cwd(), 'rev-parse', '--show-toplevel').strip())
    problem = running_problem(common_dir(root), args.package)
    if problem:
        print(problem)
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n', 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser(
        'run', help='check HEAD and stamp it when every step passes'
    )
    p.add_argument('--package', required=True)
    p.add_argument('--previous', help="a fix round's previous SHA")
    p.add_argument(
        '--python', help="the interpreter holding the candidate's client"
    )
    p.add_argument(
        '--base', help='diff from here (default: the phase head merge base)'
    )
    p = sub.add_parser('check', help='exit 0 when SHA has a passing stamp')
    p.add_argument('sha')
    p = sub.add_parser(
        'wait', help='block until SHA is stamped, or its run failed'
    )
    p.add_argument('sha')
    p.add_argument('--timeout', type=float, default=4 * 3600)
    p.add_argument('--poll', type=float, default=10, help=argparse.SUPPRESS)
    p = sub.add_parser(
        'running', help='exit 1 naming a live handoff of the package'
    )
    p.add_argument('--package', required=True)
    args = parser.parse_args(argv)
    commands = {
        'run': cmd_run,
        'check': cmd_check,
        'wait': cmd_wait,
        'running': cmd_running,
    }
    try:
        return commands[args.command](args)
    except Failure as error:
        print(f'REFUSED: {error}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
