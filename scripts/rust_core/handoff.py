#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The handoff check before review (T3.0, item 4): problems found before a
reviewer looks, and a stamp for the exact SHA that passed.

    handoff.py run --package ID [--previous SHA] [--case ID ...]
        [--python PY] [--base REV]
    handoff.py check SHA

``run`` checks the committed ``HEAD`` of a clean worktree and, only when
every step passes, writes ``<git common dir>/sdm-handoff/<sha>.json``
(``{"sha", "package", "ok": true, "time", "previous", "steps"}``); a failed
step exits 1 and writes no stamp. Implementers and test authors run it
before they report "ready"; ``codex_pane.py review`` and the Claude reviewer
launch (``guard_cmd.py``) refuse a round on a SHA without a stamp. Run it
with the tools' interpreter (the one that runs ``compare.py``); ``--python``
is the interpreter holding the candidate's client (the lane's environment),
default the same. The steps:

- ``reference``: the package's cases (each case file it adds or changes,
  and each ``--case``) run on python-core, on each database the case names
  (Postgres with ``SDM_POSTGRES_TEST_DSN``, which a case naming Postgres
  needs, or no stamp), and on rust-host (DuckDB and SQLite) when the package changes
  ``rust/``, and on their Java legs (T3.11: the cases ``run_java.py`` runs on
  the Java host, S0's today) when ``SDM_JAVA_JAR`` names a JAR, or the
  package changes the Java library (:data:`JAVA_PATHS`), on the JAR
  ``java_build.py jar`` builds (a Java package with no JDK or Maven gets no
  stamp), then ``compare.py`` holds them to the recorded old-client
  answers in ``contract/reference/`` (with the ratchet's effective
  expectations). No live old-client run, except for a case with no
  recorded answer on a database: that one runs python-ref live when
  ``SDM_REFERENCE_PYTHON`` (an interpreter holding the old client) and
  ``SDM_REFERENCE_REPO`` (its checkout) are set, and fails naming the case
  otherwise;
- ``fails-before``: in a fix round (a commit ``<ID> fix <n>: ...`` of this
  package since its base) ``--previous`` is the previous round's SHA, and
  the round's new or changed tests run on the previous tree, where at least
  one must fail. A fix to Rust production code needs a Rust test (a crate
  test or a ``#[cfg(test)]`` block, spliced into the previous source) that
  fails there with cargo, and is refused with Python tests only; any other
  fix's Python tests run on the previous tree's package sources;
- ``affected``: every suite ``affected.py`` names for the changed files;
- ``lint``: ``ruff check`` and ``ruff format --check`` on the changed Python
  files, ``cargo fmt --check`` on the changed crates, and ``mypy`` on each
  changed package's ``src``.

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
from collections.abc import Sequence
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import affected  # noqa: E402
import judging  # noqa: E402
from phases import PACKAGE_ID, PhaseError, phase_branch  # noqa: E402

STAMPS = 'sdm-handoff'
DATABASES = ('duckdb', 'sqlite', 'postgres')
FIX = re.compile(rf'({PACKAGE_ID.pattern}) fix [0-9]+: ')
CRATE_TEST = re.compile(r'rust/crates/([^/]+)/tests/([^/]+)\.rs$')
#: What makes a package a Java one, whose cases' Java legs its handoff runs
#: (T3.11); any other runs them only with ``SDM_JAVA_JAR`` set, so Java
#: never holds a Python package (plan section 6).
JAVA_PATHS = (
    'java/',
    'rust/bindings/sdm-jni/',
    'contract/runners/run_java.py',
    'contract/runners/java/',
)


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
    java = bool(os.environ.get('SDM_JAVA_JAR')) or any(
        p.startswith(JAVA_PATHS) for p in touched
    )
    built: list[str] = []
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
        legs: Path | None = None
        if java and database == 'duckdb':
            found = java_legs(root, python, list(cases.values()), scratch, built)
            if isinstance(found, str):
                step.problems.append(found)
            else:
                legs = found
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
        if legs is not None:
            # The Java legs judged on their own, so a difference only Java
            # shows is named as Java's (T3.11, exit 2).
            ran = sorted(
                json.loads(line)['case']
                for line in legs.read_text(encoding='utf-8').splitlines()
                if line.strip()
            )
            done = run(
                [sys.executable, str(root / 'scripts/rust_core/compare.py'),
                 *(arg for f in files for arg in ('--cases', f)),
                 '--reference', str(held), *expect,
                 '--register', str(root / 'docs/design/rust-core/register.yaml'),
                 str(legs)],
                root,
            )  # fmt: skip
            if done.returncode != 0:
                differs = [
                    line
                    for line in done.stdout.splitlines()
                    if line.startswith('DIFFERS')
                ]
                step.problems.append(
                    f'java: {", ".join(ran)} on the Java host differ from the '
                    'old client:\n'
                    + '\n'.join(
                        differs[:40] or [tail(done.stdout + done.stderr)]
                    )
                )
            else:
                step.notes.append(
                    f'java: {", ".join(ran)} on the Java host match the old client'
                )
    return step


def java_legs(
    root: Path, python: str, files: list[Path], scratch: Path, built: list[str]
) -> Path | str | None:
    """The Java legs of the cases in ``files``, observed on the JAR
    ``SDM_JAVA_JAR`` names, else the one ``java_build.py jar`` builds (once
    per handoff, kept in ``built``): their observations' file, None when no
    case has a Java leg, or why they could not run."""
    given = os.environ.get('SDM_JAVA_JAR')
    if given and not built:
        built.extend([given, str(Path(given).parent / 'lib'), ''])
    if not built:
        done = run(
            [sys.executable, str(root / 'scripts/rust_core/java_build.py'), 'jar'],
            root,
        )
        if done.returncode != 0:
            return 'the Java library did not build:\n' + tail(
                done.stdout + done.stderr
            )
        paths = dict(
            line.split(' ', 1)
            for line in done.stdout.splitlines()
            if line.startswith(('JAR ', 'LIB ', 'NATIVE '))
        )
        built.extend([paths['JAR'], paths['LIB'], paths['NATIVE']])
    jar, lib, native = built
    out = scratch / 'java.jsonl'
    java_home = os.environ.get('SDM_JAVA_HOME') or os.environ.get('JAVA_HOME')
    done = run(
        [python, '-I', str(root / 'contract/runners/run_java.py'), '--jar', jar,
         '--classpath', lib, *(['--native', native] if native else []),
         '--out', str(out),
         *(['--java', str(Path(java_home) / 'bin' / 'java')] if java_home else []),
         *map(str, files)],
        root,
    )  # fmt: skip
    if done.returncode != 0:
        return 'java failed:\n' + tail(done.stdout + done.stderr)
    if not out.read_text(encoding='utf-8').strip():
        return None
    return out


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
                    tree,
                )  # fmt: skip
                (failed if done.returncode else passed).append(path)
            for path in spliced:
                crate = INLINE.match(path)
                assert crate is not None
                # One workspace holds the crates and the binding.
                done = run(
                    ['cargo', 'test', '-q', '--manifest-path', 'rust/Cargo.toml',
                     '-p', crate.group(2), '--lib'],
                    tree,
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


def step_affected(root: Path, touched: list[str], python: str) -> Step:
    """Every suite ``affected.py`` names: the pytest files in one run, two
    workers a file each when pytest-xdist is there (the brief's ``-n 2``),
    then each other suite in turn."""
    step = Step('affected')
    suites = affected.suites(touched)
    if not suites:
        step.notes.append('no suite reads the changed files')
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
        done = run(argv, root)
        if done.returncode != 0:
            step.problems.append(
                f'{shown} failed:\n' + tail(done.stdout + done.stderr)
            )
        else:
            step.notes.append(f'{shown}: ok')
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
    if not (python_files or crates):
        step.notes.append('no Python or Rust file changed')
    return step


# --- the commands -----------------------------------------------------------------


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
    touched = [
        p
        for p in git(root, 'diff', '--name-only', '-z', base, 'HEAD').split(
            '\0'
        )
        if p
    ]
    python = args.python or sys.executable
    finish_on_signals()
    scratch = Path(tempfile.mkdtemp(prefix='sdm-handoff-'))
    print(
        f'handoff of {args.package} at {sha} ({len(touched)} file(s) since {base[:12]})'
    )
    try:
        steps = [
            step_reference(root, touched, args.case, python, base, scratch),
            step_fails_before(
                root, base, args.previous, python, scratch, args.package
            ),
            step_affected(root, touched, python),
            step_lint(root, touched, python),
        ]
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    for step in steps:
        print(f'{"ok  " if step.ok else "FAIL"} {step.name}')
        for note in step.notes:
            print(f'     {note}')
        for problem in step.problems:
            print('     ' + problem.replace('\n', '\n     | '))
    if not all(step.ok for step in steps):
        print(
            'NO STAMP: fix the failing steps and run handoff.py again on the new SHA'
        )
        return 1
    path = stamp_path(common_dir(root), sha)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                'sha': sha,
                'package': args.package,
                'ok': True,
                'time': datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec='seconds'
                ),
                'previous': args.previous,
                'steps': {s.name: s.notes for s in steps},
            },
            indent=1,
        )
        + '\n',
        encoding='utf-8',
    )
    print(f'STAMPED {sha}: {path}')
    return 0


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
    args = parser.parse_args(argv)
    try:
        return (cmd_run if args.command == 'run' else cmd_check)(args)
    except Failure as error:
        print(f'REFUSED: {error}')
        return 2


if __name__ == '__main__':
    sys.exit(main())
