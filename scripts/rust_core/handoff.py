#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The handoff check before review (T3.0, item 4): problems found before a
reviewer looks, and a stamp for the exact SHA that passed.

    handoff.py run --package ID [--previous SHA] [--case ID ...]
        [--python PY] [--base REV]
    handoff.py check SHA
    handoff.py wait SHA [--timeout SECONDS]
    handoff.py running --package ID

``run`` checks the committed ``HEAD`` of a clean worktree and, only when
every step passes, writes ``<git common dir>/sdm-handoff/<sha>.json``
(``{"sha", "package", "ok": true, "time", "previous", "steps", "seconds"}``);
a failed step exits 1 and writes no stamp. Implementers and test authors
run it before they report "ready"; ``codex_pane.py review`` and the Claude
reviewer launch (``guard_cmd.py``) refuse a round on a SHA without a stamp.
Run it with the tools' interpreter (the one that runs ``compare.py``);
``--python`` is the interpreter holding the candidate's client (the lane's
environment), default the same, and every command the run starts sees it
as ``SDM_PYTHON`` (the Rust tests that load cases run it, never a bare
``python3``).

The steps run cheapest first, each printed as it starts and ends with its
time, and the run stops at the first failing step (the owner's
"2026-10-22 evening: the five speed changes", change 1): the handoff is
the package's own evidence, in about 15 minutes; the broad suites are CI's,
and ``land.py`` already refuses a landing without a green CI run.

- ``lint``: ``ruff check`` and ``ruff format --check`` on the changed Python
  files, ``cargo fmt --check`` on the changed crates, ``mypy`` on each
  changed package's ``src``, ``l0_scan.py``, the hygiene check, and
  ``cargo deny --locked check`` when ``rust/`` changed;
- ``reference``: the package's cases (each case file it adds or changes,
  and each ``--case``) run on python-core, on each database the case names
  (Postgres with ``SDM_POSTGRES_TEST_DSN``, which a case naming Postgres
  needs, or no stamp), and on rust-host (DuckDB and SQLite) when the package changes
  ``rust/``, then ``compare.py`` holds them to the recorded old-client
  answers in ``contract/reference/`` (with the ratchet's effective
  expectations). No live old-client run, except for a case with no
  recorded answer on a database: that one runs python-ref live when
  ``SDM_REFERENCE_PYTHON`` (an interpreter holding the old client) and
  ``SDM_REFERENCE_REPO`` (its checkout) are set, and fails naming the case
  otherwise;
- ``own tests``: the test files the package adds or changes (pytest files,
  self-tests, and each crate test by its own ``--test``);
- ``fails-before``: in a fix round (a commit ``<ID> fix <n>: ...`` of this
  package since its base) ``--previous`` is the previous round's SHA, and
  the round's new or changed tests run on the previous tree, where at least
  one must fail. A fix to Rust production code needs a Rust test (a crate
  test or a ``#[cfg(test)]`` block, spliced into the previous source) that
  fails there with cargo, and is refused with Python tests only; any other
  fix's Python tests run on the previous tree's package sources. The
  previous tree's Rust builds in a target directory of its own;
- ``affected``: the suites ``affected.py``'s map rows name for the changed
  files, not every test that mentions a file's name (``affected.suites``
  with ``naming=False``), less the own tests already run;
- ``wheel``: the Linux x86 wheel, built as CI's native-wheel job builds it,
  when ``rust/`` or the client's package changed.

The stamp is the package's local check, and with the two reviews what a
landing needs (the owner's "2026-10-22 evening: CI off the landing path");
the full CI matrix runs nightly on the phase head and at the phase's end.

In a fix round whose ``--previous`` has a passing stamp, ``own tests`` and
``affected`` take the files changed since that stamp; the other steps take
the package's whole change since its base.

While it runs, ``run`` holds ``<git common dir>/sdm-handoff/running/<ID>.json``
(its process, SHA and start), which ``running`` reads: it exits 1 naming a
live run of the package, else 0. ``prepush.sh`` refuses a push from the
package's worktree while one runs (change 2), and a second ``run`` of the
package is refused. A signal (SIGTERM, SIGHUP) stops the run through its
clean-up: the scratch, the previous round's worktree and the running file
go. Each start, stamp, failure and stop is a line of
``<git common dir>/sdm-handoff/events.jsonl``; ``wait SHA`` blocks until SHA
is stamped (exit 0) or its run fails or stops (exit 1), so the reviewers
start the moment the stamp is written (exit 3 after ``--timeout``).

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
EVENTS = 'events.jsonl'
DATABASES = ('duckdb', 'sqlite', 'postgres')
FIX = re.compile(rf'({PACKAGE_ID.pattern}) fix [0-9]+: ')
CRATE_TEST = re.compile(r'rust/crates/([^/]+)/tests/([^/]+)\.rs$')


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


def case_index(root: Path) -> dict[str, Path]:
    found = {}
    for path in sorted((root / 'contract/cases').rglob('*.json')):
        try:
            case = json.loads(path.read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue
        if isinstance(case, dict) and isinstance(case.get('id'), str):
            found[case['id']] = path
    return found


def reference_lines(root: Path) -> tuple[Path | None, dict[str, str]]:
    files = sorted((root / 'contract/reference').glob('*.jsonl'))
    if len(files) != 1:
        return None, {}
    lines = {}
    for line in files[0].read_text(encoding='utf-8').splitlines():
        if line.strip():
            lines[json.loads(line)['case']] = line
    return files[0], lines


def key(case: str, database: str) -> str:
    return case if database == 'duckdb' else f'{case}@{database}'


def step_reference(
    root: Path,
    touched: list[str],
    named: list[str],
    python: str,
    base: str,
    scratch: Path,
) -> Step:
    step = Step('reference')
    index = case_index(root)
    chosen: dict[str, Path] = {}
    for path in touched:
        if path.startswith('contract/cases/') and path.endswith('.json'):
            full = root / path
            if full.is_file():
                case = json.loads(full.read_text(encoding='utf-8'))
                chosen[str(case.get('id'))] = full
    for name in named:
        if name not in index:
            step.problems.append(f'--case {name}: no case with that id')
        else:
            chosen[name] = index[name]
    if not chosen:
        step.notes.append('no case changed or named')
        return step
    reference, recorded = reference_lines(root)
    if reference is None:
        step.problems.append(
            'contract/reference/ holds no single reference file'
        )
        return step
    effective = scratch / 'effective.json'
    done = run(
        [sys.executable, str(root / 'scripts/rust_core/ratchet.py'), 'effective',
         '--repo', str(root), '--base', base, '--out', str(effective)],
        root,
    )  # fmt: skip
    expect = ['--expect', str(effective)] if done.returncode == 0 else []
    if done.returncode != 0:
        step.notes.append(
            "the ratchet's effective expectations are unreadable here, so the "
            'committed answers stand alone: '
            + tail(done.stdout + done.stderr, 2)
        )
    rust = any(p.startswith('rust/') for p in touched)
    for database in DATABASES:
        cases = {
            case: path
            for case, path in chosen.items()
            if database
            in (json.loads(path.read_text()).get('databases') or DATABASES)
        }
        if not cases:
            continue
        if database == 'postgres' and not os.environ.get(
            'SDM_POSTGRES_TEST_DSN'
        ):
            # Every database a case names is judged here (Codex r2): no
            # server, no stamp.
            step.problems.append(
                f'{", ".join(sorted(cases))} run(s) on Postgres, and '
                'SDM_POSTGRES_TEST_DSN is not set: point it at a server (a '
                'throwaway postgres:17 container, as CI runs), or no stamp'
            )
            continue
        files = [str(p) for p in cases.values()]
        observed = []
        out = scratch / f'{database}-python-core.jsonl'
        done = run(
            [python, '-I', str(root / 'contract/runners/run_python.py'), '--mode',
             'python-core', '--database', database, '--out', str(out), *files],
            root,
        )  # fmt: skip
        if done.returncode != 0:
            step.problems.append(
                f'python-core on {database} failed for {", ".join(sorted(cases))}:\n'
                + tail(done.stdout + done.stderr)
            )
            continue
        observed.append(out)
        if rust and database in ('duckdb', 'sqlite'):
            host = scratch / f'{database}-rust-host.jsonl'
            flags = ['--database', 'sqlite'] if database == 'sqlite' else []
            done = run(
                ['cargo', 'run', '-q', '-p', 'sdm-testkit', '--bin', 'sdm-contract-run',
                 '--locked', '--', *flags, '--out', str(host), *files],
                root / 'rust',
            )  # fmt: skip
            if done.returncode != 0:
                step.problems.append(
                    f'rust-host on {database} failed:\n'
                    + tail(done.stdout + done.stderr)
                )
            else:
                observed.append(host)
        # The recorded answers of these cases on this database, and a live
        # old-client answer where none is recorded.
        lines = []
        for case in sorted(cases):
            line = recorded.get(key(case, database))
            if line is None:
                live = live_reference(
                    root, case, cases[case], database, scratch
                )
                if isinstance(live, str):
                    step.problems.append(live)
                    continue
                lines.extend(live)
                step.notes.append(
                    f'{key(case, database)}: no recorded answer; ran the old client live'
                )
            else:
                lines.append(line)
        held = scratch / f'{database}-reference.jsonl'
        held.write_text(
            ''.join(line + '\n' for line in lines), encoding='utf-8'
        )
        done = run(
            [sys.executable, str(root / 'scripts/rust_core/compare.py'),
             *(arg for f in files for arg in ('--cases', f)),
             '--database', database, '--reference', str(held), *expect,
             '--register', str(root / 'docs/design/rust-core/register.yaml'),
             *map(str, observed)],
            root,
        )  # fmt: skip
        if done.returncode != 0:
            differs = [
                line
                for line in done.stdout.splitlines()
                if line.startswith('DIFFERS')
            ]
            step.problems.append(
                f'{database}: the cases differ from the old client:\n'
                + '\n'.join(differs[:40] or [tail(done.stdout + done.stderr)])
            )
        else:
            step.notes.append(
                f'{database}: {", ".join(sorted(cases))} match the old client'
            )
    return step


def live_reference(
    root: Path, case: str, path: Path, database: str, scratch: Path
) -> list[str] | str:
    python = os.environ.get('SDM_REFERENCE_PYTHON')
    checkout = os.environ.get('SDM_REFERENCE_REPO')
    if not (python and checkout):
        return (
            f'{key(case, database)} has no recorded old-client answer; set '
            'SDM_REFERENCE_PYTHON and SDM_REFERENCE_REPO to run the old client '
            'live, or record its answer first'
        )
    out = scratch / f'{database}-{case}-python-ref.jsonl'
    done = run(
        [python, '-I', str(root / 'contract/runners/run_python.py'), '--mode',
         'python-ref', '--reference', checkout, '--database', database,
         '--out', str(out), str(path)],
        root,
    )  # fmt: skip
    if done.returncode != 0:
        return (
            f'{key(case, database)}: the live old-client run failed:\n'
            + tail(done.stdout + done.stderr)
        )
    return [
        line
        for line in out.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


INLINE = re.compile(r'rust/(crates|bindings)/([^/]+)/src/.+\.rs$')
TEST_BLOCK = re.compile(
    r'^[ \t]*#\[cfg\(test\)\][ \t]*\n[ \t]*(pub(\([^)]*\))?[ \t]+)?mod[ \t]+\w+[ \t]*\{',
    re.MULTILINE,
)


def test_block(text: str) -> tuple[int, int] | None:
    """Where a Rust file's ``#[cfg(test)] mod ... { ... }`` block is, as
    (start, end), braces counted outside strings, chars and comments."""
    found = TEST_BLOCK.search(text)
    if found is None:
        return None
    depth, n = 0, found.end() - 1
    while n < len(text):
        ch = text[n]
        if text.startswith('//', n):
            n = text.find('\n', n)
            n = len(text) if n < 0 else n
            continue
        if text.startswith('/*', n):
            end = text.find('*/', n + 2)
            n = len(text) if end < 0 else end + 2
            continue
        if ch == '"':
            n += 1
            while n < len(text) and text[n] != '"':
                n += 2 if text[n] == '\\' else 1
        elif ch == "'" and re.match(r"'(\\.|[^\\'])'", text[n : n + 4]):
            n = text.index("'", n + 1 + (text[n + 1] == '\\'))
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return found.start(), n + 1
        n += 1
    return None


def inline_tests(root: Path, previous: str, path: str) -> str | None:
    """For a Rust source file whose ``#[cfg(test)]`` block changed since
    ``previous``, the previous file with HEAD's test block in place of its
    own (the previous production code under the new tests); else None."""
    if not INLINE.match(path) or not (root / path).is_file():
        return None
    new = (root / path).read_text(encoding='utf-8')
    new_block = test_block(new)
    if new_block is None:
        return None
    done = run(['git', 'show', f'{previous}:{path}'], root)
    old = done.stdout if done.returncode == 0 else ''
    old_block = test_block(old)
    tests = new[new_block[0] : new_block[1]]
    if old_block is not None:
        if old[old_block[0] : old_block[1]] == tests:
            return None
        return old[: old_block[0]] + tests + old[old_block[1] :]
    return old + ('\n' if old and not old.endswith('\n') else '') + tests + '\n'


def is_test(path: str) -> bool:
    return bool(
        judging.matches(affected.IS_TEST, path)
        or path.endswith('selftest.py')
        or CRATE_TEST.match(path)
    )


def source_path(tree: Path, python: str) -> str:
    """``PYTHONPATH`` for running tests on the previous tree's Python
    sources: each package's ``src``, with the compiled parts the installed
    build has and the sources lack (the core's ``_native``) linked in, since
    the previous tree is not built, each batch linking its own
    interpreter's. A Rust fix's own test is a Rust test, which runs on the
    previous tree's code with cargo."""
    paths = []
    for src in sorted(tree.glob('packages/*/src')):
        paths.append(str(src))
        for package in sorted(p for p in src.iterdir() if p.is_dir()):
            done = run(
                [python, '-c', 'import importlib.util, sys; '
                 's = importlib.util.find_spec(sys.argv[1]); '
                 'print(s.submodule_search_locations[0] if s and '
                 's.submodule_search_locations else "")', package.name],
                tree,
            )  # fmt: skip
            installed = (
                Path(done.stdout.strip()) if done.stdout.strip() else None
            )
            if installed is None or not installed.is_dir():
                continue
            for compiled in installed.glob('*.so'):
                target = package / compiled.name
                # Each interpreter's own build: a link an earlier batch made
                # for another interpreter (production against test-ops) is
                # replaced (Codex r4).
                if target.is_symlink():
                    target.unlink()
                if not target.exists():
                    target.symlink_to(compiled)
    return os.pathsep.join(paths)


def without_tests(text: str) -> str:
    """A Rust file's text without its ``#[cfg(test)]`` block."""
    block = test_block(text)
    return text if block is None else text[: block[0]] + text[block[1] :]


#: Rust production code: a crate's or the binding's Rust source, its
#: manifest and build script; never a test, a test helper, a vector or a
#: Python file under ``rust/`` (Codex r3, Claude r3 P3-2).
PRODUCTION = re.compile(
    r'rust/(crates|bindings)/[^/]+/(src/.+\.rs|build\.rs|Cargo\.toml)$'
)


def code_lines(text: str) -> list[str]:
    """The lines of Rust source that carry code: each stripped at its ends,
    blank ones dropped, so a test block's surrounding blank lines change
    nothing while a space inside a string literal does (Codex r3)."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def rust_production(root: Path, previous: str, diffed: list[str]) -> list[str]:
    """The Rust files whose production code changed since ``previous``: a
    source file (its ``#[cfg(test)]`` block aside), a manifest or a build
    script under a crate or the binding."""
    found = []
    for path in diffed:
        if not PRODUCTION.match(path):
            continue
        if not path.endswith('.rs'):
            found.append(path)
            continue
        done = run(['git', 'show', f'{previous}:{path}'], root)
        old = done.stdout if done.returncode == 0 else ''
        new = (
            (root / path).read_text(encoding='utf-8')
            if (root / path).is_file()
            else ''
        )
        if code_lines(without_tests(old)) != code_lines(without_tests(new)):
            found.append(path)
    return found


def previous_cargo_env(
    environ: dict[str, str] | os._Environ[str], root: Path
) -> dict[str, str]:
    r"""The environment the previous round's Rust tests build in: the
    caller's, with a target directory of its own beside the caller's. Cargo
    keys a workspace member's build by its name, not its directory, so the
    previous tree built into the caller's target leaves artifacts newer
    than the candidate's sources, which the next build takes as fresh (a
    module the candidate adds then cannot be found)."""
    target = environ.get('CARGO_TARGET_DIR') or str(root / 'rust' / 'target')
    return {**environ, 'CARGO_TARGET_DIR': target.rstrip('/') + '-previous'}


def step_fails_before(
    root: Path,
    base: str,
    previous: str | None,
    python: str,
    scratch: Path,
    package: str | None = None,
) -> Step:
    """A fix round's new or changed tests, on the previous round's tree; at
    least one must fail there. Two rules, by what the fix changes (the
    owner's stop-and-redesign rule after two rounds of patches):

    - a fix to Rust production code comes with a Rust test, a crate test
      (``tests/*.rs``) or a ``#[cfg(test)]`` block spliced into the
      previous source, that fails there with cargo; Python tests cannot
      show it, since the previous tree's core is not built, and a Rust fix
      with Python tests only is refused, asking for a Rust test;
    - any other fix: its Python tests run on the previous tree's package
      sources (``source_path``) and its self-tests on the previous tree,
      pytest stopping at the first failure."""
    step = Step('fails-before')
    subjects = git(root, 'log', '--format=%s', f'{base}..HEAD').splitlines()
    fixes = [
        s
        for s in subjects
        if (m := FIX.match(s)) and (package is None or m.group(1) == package)
    ]
    if previous is None:
        if fixes:
            step.problems.append(
                f"a fix round ({fixes[0]}) names the previous round's SHA: "
                '--previous SHA'
            )
        else:
            step.notes.append('not a fix round')
        return step
    previous = git(root, 'rev-parse', f'{previous}^{{commit}}').strip()
    diffed = [
        p
        for p in git(root, 'diff', '--name-only', '-z', previous, 'HEAD').split(
            '\0'
        )
        if p
    ]
    production = rust_production(root, previous, diffed)
    changed = [p for p in diffed if is_test(p) and (root / p).is_file()]
    spliced = {
        p: text
        for p in diffed
        if (text := inline_tests(root, previous, p)) is not None
    }
    crate_tests = [p for p in changed if CRATE_TEST.match(p)]
    if production and not (crate_tests or spliced):
        step.problems.append(
            'the fix changes Rust production code ('
            + ', '.join(production[:5])
            + ') and adds no Rust test: add a #[cfg(test)] test or a crate test '
            '(rust/crates/<crate>/tests/*.rs) that fails without the fix'
        )
        return step
    if not production and not changed and not spliced:
        step.problems.append(
            f'the fix round adds or changes no test since {previous[:12]}; a fix '
            'comes with a test that fails without it'
        )
        return step
    tree = scratch / 'previous'
    git(root, 'worktree', 'add', '--quiet', '--detach', str(tree), previous)
    failed: list[str] = []
    passed: list[str] = []
    try:
        for path in changed:
            target = tree / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / path, target)
        for path, text in spliced.items():
            (tree / path).parent.mkdir(parents=True, exist_ok=True)
            (tree / path).write_text(text, encoding='utf-8')
        # Rust tests run with cargo whenever there are any; for a Rust
        # production fix they are the only judges.
        if production or crate_tests or spliced:
            for path in crate_tests:
                match = CRATE_TEST.match(path)
                assert match is not None
                done = run(
                    ['cargo', 'test', '-q', '--manifest-path', 'rust/Cargo.toml',
                     '-p', match.group(1), '--test', match.group(2)],
                    tree, previous_cargo_env(os.environ, root),
                )  # fmt: skip
                (failed if done.returncode else passed).append(path)
            for path in spliced:
                crate = INLINE.match(path)
                assert crate is not None
                # One workspace holds the crates and the binding.
                done = run(
                    ['cargo', 'test', '-q', '--manifest-path', 'rust/Cargo.toml',
                     '-p', crate.group(2), '--lib'],
                    tree, previous_cargo_env(os.environ, root),
                )  # fmt: skip
                (failed if done.returncode else passed).append(
                    f'{path} (#[cfg(test)])'
                )
        if not production:
            # New test files first: with -x the first failure ends the run,
            # and a new test is the likeliest to fail on the previous tree.
            pytest_files = sorted(
                (p for p in changed if judging.matches(affected.IS_TEST, p)),
                key=lambda p: (
                    run(
                        ['git', 'cat-file', '-e', f'{previous}:{p}'], root
                    ).returncode
                    == 0
                ),
            )
            # The binding's tests need the test-ops build and its option
            # (as affected.py's BINDING runs them); the rest plain pytest.
            kit = os.environ.get('SDM_KIT_PYTHON', '')
            binding = [
                p for p in pytest_files if p.startswith('rust/bindings/')
            ]
            plain = [p for p in pytest_files if p not in binding]
            batches = [(python, plain, [])]
            if binding and not kit:
                step.problems.append(
                    "the binding tests need SDM_KIT_PYTHON, the lane's test-ops "
                    'interpreter: ' + ', '.join(binding)
                )
            elif binding:
                batches.append((kit, binding, ['--native-build', 'test-ops']))
            for interpreter, files, extra in batches:
                if not files:
                    continue
                env = {
                    **os.environ,
                    'PYTHONPATH': source_path(tree, interpreter),
                }
                done = run(
                    [interpreter, '-m', 'pytest', '-q', '-x', '-p',
                     'no:cacheprovider', *files, *extra],
                    tree,
                    env,
                )  # fmt: skip
                (failed if done.returncode in (1, 2) else passed).extend(files)
                if done.returncode not in (0, 1, 2, 5):
                    step.problems.append(
                        f'pytest could not run the tests on {previous[:12]}:\n'
                        + tail(done.stdout + done.stderr)
                    )
            for path in changed:
                if path.endswith('selftest.py'):
                    done = run([python, path], tree)
                    (failed if done.returncode else passed).append(path)
    finally:
        shutil.rmtree(tree, ignore_errors=True)
        run(['git', 'worktree', 'prune'], root)
    if failed:
        step.notes.append(f'fails on {previous[:12]}: ' + ', '.join(failed))
    elif not step.problems:
        kind = 'Rust test' if production else 'test'
        step.problems.append(
            f'every new or changed {kind} passes on the previous round '
            f'{previous[:12]}, so none shows the fix: ' + ', '.join(passed)
        )
    return step


def own_suites(root: Path, touched: list[str]) -> list[affected.Suite]:
    """The test files among ``touched`` as suites: a pytest file (the
    binding's under ``affected.BINDING``), a self-test, a crate test by its
    own ``--test``."""
    found: list[affected.Suite] = []
    for path in touched:
        if not (root / path).is_file() or not is_test(path):
            continue
        crate = CRATE_TEST.match(path)
        if crate:
            suite: affected.Suite = (
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


def step_affected(
    root: Path, touched: list[str], python: str, own: list[str] | None = None
) -> Step:
    """The suites ``affected.py``'s map rows name for ``touched`` (not every
    test naming a file: the broad suites are CI's), less the package's own
    tests (``own``, the touched files those came from), already run."""
    step = Step('affected')
    ran = set(own_suites(root, own if own is not None else touched))
    suites = [s for s in affected.suites(touched, naming=False) if s not in ran]
    if not suites:
        step.notes.append('no mapped suite beyond the own tests')
        return step
    return run_suites(step, root, suites, python)


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
    if files:
        flat = [f for group in files for f in group]
        workers = (
            ['-n', '2', '--dist', 'loadfile']
            if run([python, '-c', 'import xdist'], root).returncode == 0
            else []
        )
        batches.append(
            (
                f'pytest on {len(flat)} test file(s)',
                [python, *prefix[1:], *workers, *flat],
            )
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
    for shown, argv in batches:
        if step.problems:
            break
        begun = time.monotonic()
        done = run(argv, root)
        took = f'{time.monotonic() - begun:.0f}s'
        if done.returncode != 0:
            step.problems.append(
                f'{shown} failed ({took}):\n' + tail(done.stdout + done.stderr)
            )
        else:
            step.notes.append(f'{shown}: ok ({took})')
    return step


#: What the shipped wheel is built from: a change here builds it.
WHEEL_SOURCES = ('rust/', 'packages/structured-data-models-client/')


def step_wheel(root: Path, touched: list[str], scratch: Path) -> Step:
    r"""The Linux x86 wheel, built as CI's native-wheel job builds it
    (``native_wheel.py`` with its pinned maturin), when the package changes
    what it is built from."""
    step = Step('wheel')
    if not any(p.startswith(WHEEL_SOURCES) for p in touched):
        step.notes.append('nothing the wheel is built from changed')
        return step
    script = root / 'scripts/rust_core/native_wheel.py'
    pinned = re.search(
        r"^MATURIN = '([^']+)'", script.read_text(encoding='utf-8'), re.M
    )
    uv = shutil.which('uv')
    if pinned is None or uv is None:
        step.problems.append('no uv, or no MATURIN pin in native_wheel.py')
        return step
    done = run(
        [uv, 'run', '--quiet', '--no-project', '--python', '3.12', '--with',
         f'maturin=={pinned.group(1)}', 'python3', str(script),
         '--out', str(scratch / 'wheel')],
        root,
    )  # fmt: skip
    if done.returncode != 0:
        step.problems.append(
            'native_wheel.py:\n' + tail(done.stdout + done.stderr)
        )
    else:
        step.notes.append('the Linux x86 wheel builds')
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
    # the reference and lint steps still take the whole package.
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
            f'{base[:12]}; own tests and affected on {len(recent)} since '
            f'{since[:12]})',
            flush=True,
        )
        scratch = Path(tempfile.mkdtemp(prefix='sdm-handoff-'))
        plan = [
            ('lint', lambda: step_lint(root, touched, python)),
            (
                'reference',
                lambda: step_reference(
                    root, touched, args.case, python, base, scratch
                ),
            ),
            ('own tests', lambda: step_own_tests(root, recent, python)),
            (
                'fails-before',
                lambda: step_fails_before(
                    root, base, args.previous, python, scratch, args.package
                ),
            ),
            ('affected', lambda: step_affected(root, recent, python)),
            ('wheel', lambda: step_wheel(root, touched, scratch)),
        ]
        try:
            for name, step_of in plan:
                print(f'start {name} at {now()}', flush=True)
                begun = time.monotonic()
                step = step_of()
                step.seconds = time.monotonic() - begun
                steps.append(step)
                show(step)
                if not step.ok:
                    break
        finally:
            shutil.rmtree(scratch, ignore_errors=True)
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
                f'NO STAMP ({took}s): {failed.name} failed, and the steps after '
                'it did not run; fix it and run handoff.py again on the new SHA',
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
        '--case', action='append', default=[], help='a case id to run'
    )
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
