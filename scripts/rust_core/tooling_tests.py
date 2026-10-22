#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The tests of the checking tools, found by what they do, and when CI
runs them (T3.0, item 3).

The tests of our own checking and orchestration scripts carry the pytest
marker ``tooling`` (``pytestmark = pytest.mark.tooling``). CI's
client matrix, ``engine`` and ``contract-python`` deselect them, and the two
``tooling`` jobs run only them, only when a change can affect them
(:func:`must_run`).

A test file is *gate-copying* (:func:`gate_copying`) when it, or a module
it loads from ``tests/`` or ``contract/tests/`` (by import, or by a path to
the file, followed transitively), holds a *builder*: source that copies
files out of the checkout (``shutil.copytree``, ``copy2``, ``copyfile``,
or ``git ls-files``) into a temporary git repository (``'init'``) or copies
the contract (a ``'contract'`` path). Those files rerun the gate, the
ratchet or the comparator over a copy of the whole corpus, minutes a test,
and every one of them must carry the marker. A file may carry it without
being gate-copying (``contract/tests/test_runners.py``): what the marker
selects is the tooling job's, whatever the detector finds.

    tooling_tests.py check [--root DIR]
        lists the gate-copying test files without the marker; exit 1 if any
    tooling_tests.py list [--root DIR]
        the marked files, then the gate-copying ones, each with its builders
    tooling_tests.py must-run --base SHA [--head REV] [--root DIR]
        prints ``true`` or ``false``, then why, on two lines: whether the
        tooling jobs run

``must-run`` is ``true`` when a file changed since ``git merge-base BASE
HEAD`` matches ``scripts/rust_core/judging-paths.txt`` (protected-paths
syntax, the one list of judging paths), is under ``scripts/rust_core/``
(the scripts those tests run), is the workflow, the marker's
registration (``pyproject.toml``) or a suite's ``conftest.py``, or is a
marked file or a module one loads; and when the list cannot be read or the
base is no commit (fail closed: the jobs run).
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ratchet import patterns, protects  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
#: The two suites the marker splits.
SUITES = ('tests', 'contract/tests')
MARKER = 'tooling'
#: The marker as a test file carries it.
MARKED = re.compile(rf'\bpytest\.mark\.{MARKER}\b')
#: The one list of judging paths (owned by the tooling train, T3.0 item 1).
JUDGING_PATHS = 'scripts/rust_core/judging-paths.txt'
#: Paths that decide the split itself, beyond the judging paths.
ALWAYS = (
    '.github/workflows/ci.yml',
    'pyproject.toml',
    'tests/conftest.py',
    'contract/tests/conftest.py',
)

#: A module of a suite loaded by its path (``ROOT / 'tests/x.py'``,
#: ``'contract/tests/x.py'``).
LOAD_PATH = re.compile(r"""['"/]((?:contract/)?tests/[A-Za-z0-9_]+\.py)['"]""")
#: A module of the same suite loaded by name (``import ci4_gate``,
#: ``from t132g_gate import x``, ``HERE / 'x.py'``).
LOAD_NAME = re.compile(
    r"""^\s*(?:from|import)\s+(?:contract\.tests\.|tests\.)?([A-Za-z0-9_]+)"""
    r"""|(?:HERE|TESTS|parent)\s*/\s*['"]([A-Za-z0-9_]+)\.py['"]""",
    re.MULTILINE,
)
COPY = re.compile(r"""\b(?:copytree|copy2|copyfile)\(|['"]ls-files['"]""")
INIT = re.compile(r"""['"]init['"]""")
CONTRACT = re.compile(r"""['"]contract['"/]""")


def sources(root: Path) -> dict[str, str]:
    r"""Every Python file of the two suites, by its path from ``root``."""
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding='utf-8')
        for suite in SUITES
        for path in sorted((root / suite).glob('*.py'))
    }


def is_builder(text: str) -> bool:
    r"""``text`` copies out of the checkout into a temporary git repository,
    or copies the contract."""
    return bool(
        COPY.search(text) and (INIT.search(text) or CONTRACT.search(text))
    )


def loads(name: str, files: dict[str, str]) -> set[str]:
    r"""The suites' modules ``name`` loads directly."""
    text = files[name]
    here = name.rpartition('/')[0]
    found = {
        m.group(1) for m in LOAD_PATH.finditer(text) if m.group(1) in files
    }
    for match in LOAD_NAME.finditer(text):
        module = match.group(1) or match.group(2)
        candidate = f'{here}/{module}.py'
        if candidate in files:
            found.add(candidate)
    return found - {name}


def closure(name: str, files: dict[str, str]) -> set[str]:
    r"""``name`` and every module of the suites it loads, transitively."""
    seen: set[str] = set()
    pending = [name]
    while pending:
        current = pending.pop()
        if current not in seen:
            seen.add(current)
            pending.extend(loads(current, files))
    return seen


def test_files(files: dict[str, str]) -> list[str]:
    return [
        name for name in files if name.rpartition('/')[2].startswith('test_')
    ]


def gate_copying(root: Path = ROOT) -> dict[str, list[str]]:
    r"""Each gate-copying test file, with the builders it reaches."""
    files = sources(root)
    builders = {name for name, text in files.items() if is_builder(text)}
    found = {}
    for name in test_files(files):
        reached = sorted(closure(name, files) & builders)
        if reached:
            found[name] = reached
    return found


def marked(root: Path = ROOT) -> list[str]:
    r"""The test files carrying the marker."""
    files = sources(root)
    return [name for name in test_files(files) if MARKED.search(files[name])]


def unmarked(root: Path = ROOT) -> dict[str, list[str]]:
    r"""The gate-copying test files without the marker."""
    have = set(marked(root))
    return {
        name: builders
        for name, builders in gate_copying(root).items()
        if name not in have
    }


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(root), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def must_run(root: Path, base: str, head: str = 'HEAD') -> tuple[bool, str]:
    r"""Whether the tooling jobs run for ``head`` against ``base``, and why."""
    try:
        fork = _git(root, 'merge-base', base, head).strip()
        changed = [
            line
            for line in _git(
                root, 'diff', '--name-only', '--no-renames', fork, head
            ).splitlines()
            if line
        ]
    except subprocess.CalledProcessError as error:
        return (
            True,
            f'the change cannot be read ({error.stderr.strip()}): they run',
        )
    listed = root / JUDGING_PATHS
    if not listed.is_file():
        return True, f'{JUDGING_PATHS} is missing: they run'
    rules = patterns(listed.read_text(encoding='utf-8'))
    files = sources(root)
    loaded: set[str] = set()
    for name in marked(root):
        loaded |= closure(name, files)
    for path in changed:
        if path in ALWAYS:
            return True, f'{path} changed'
        if protects(rules, path):
            return True, f'{path} changed, a judging path ({JUDGING_PATHS})'
        if path.startswith('scripts/rust_core/'):
            # The brief's trigger: any of our checking and orchestration
            # scripts, judging path or not (land.py, lane.py, handoff.py).
            return True, f'{path} changed, one of the scripts the tests run'
        if path in loaded:
            return (
                True,
                f'{path} changed, a {MARKER} test or a module one loads',
            )
    return (
        False,
        f'none of the {len(changed)} changed files is a judging path or a {MARKER} test',
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--root', type=Path, default=ROOT)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('check')
    commands.add_parser('list')
    run = commands.add_parser('must-run')
    run.add_argument('--base', required=True)
    run.add_argument('--head', default='HEAD')
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    if arguments.command == 'check':
        missing = unmarked(root)
        for name, builders in missing.items():
            print(
                f'{name}: copies the checkout for the gate (through '
                f'{", ".join(builders)}) but lacks pytest.mark.{MARKER}; '
                f'add "pytestmark = pytest.mark.{MARKER}" so CI runs it in '
                'the tooling jobs, not in every matrix job'
            )
        return 1 if missing else 0
    if arguments.command == 'list':
        for name in marked(root):
            print(f'marked {name}')
        for name, builders in gate_copying(root).items():
            print(f'gate-copying {name}: {", ".join(builders)}')
        return 0
    run_them, why = must_run(root, arguments.base, arguments.head)
    print('true' if run_them else 'false')
    print(why)
    return 0


if __name__ == '__main__':
    sys.exit(main())
