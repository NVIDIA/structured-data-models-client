#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The test suites that read a touched file (T3.0, item 4).

    affected.py [--base REV] [FILES...]
    affected.py --unmapped

Prints one command a line: every suite that reads one of ``FILES`` (or,
with none given, the files changed since ``--base``, by default the merge
base with the phase branch). ``handoff.py`` runs them before a package is
reported ready. ``python3`` in a printed command stands for the interpreter
the caller runs them with, and ``python3-kit`` the lane's test-ops interpreter
(``SDM_KIT_PYTHON``).

A path's suites come from :data:`MAP` (a path pattern, in
``contract/protected-paths.txt``'s syntax, and its suites), plus every test
file that names the path's file name (``tests/``, ``contract/tests/``, the
binding's tests and the self-tests under ``scripts/rust_core/``), plus the
file itself when it is a test (a path under ``docs/`` names none: people
read docs, and a test naming one only cites it; a test file is its own
suite). A path no pattern of the map covers gets
only those, and ``--unmapped`` lists each tracked path under ``contract/``,
``scripts/rust_core/`` and ``rust/crates/`` that the map does not cover,
exiting 1 when there is one: a new kind of file there needs its row.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judging  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
#: The roots every tracked path of which the map covers.
COVERED = ('contract/', 'scripts/rust_core/', 'rust/crates/', 'rust/bindings/')
#: A suite: an argv, ``python3`` standing for the caller's interpreter, run
#: from the repository's root.
Suite = tuple[str, ...]

PYTEST = ('python3', '-m', 'pytest', '-q', '-p', 'no:cacheprovider')
LAND = ('python3', 'scripts/rust_core/land_selftest.py')
LANE = ('python3', 'scripts/rust_core/lane_selftest.py')
CODEX = ('python3', 'scripts/rust_core/orchestration/codex_pane_selftest.py')
AUTO_LAND = ('python3', 'scripts/rust_core/orchestration/auto_land_selftest.py')
REPLAY = ('python3', 'scripts/rust_core/replay_selftest.py')
T30 = (*PYTEST, 'tests/test_t30_process.py')
RATCHET = (*PYTEST, 'tests/test_contract_ratchet.py')
CI = (*PYTEST, 'tests/test_ci_workflow.py')
LOADER = (*PYTEST, 'contract/tests/test_contract_loader.py')
RUNNERS = (*PYTEST, 'contract/tests/test_runners.py')
#: ``python3-kit`` stands for the lane's test-ops interpreter
#: (``SDM_KIT_PYTHON``, cells.sh's ``kit``), which the binding's tests need.
KIT = 'python3-kit'
BINDING = (
    KIT, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
    'rust/bindings/sdm-py/tests', '--native-build', 'test-ops',
)  # fmt: skip


def cargo(crate: str) -> Suite:
    return ('cargo', 'test', '--manifest-path', 'rust/Cargo.toml', '-p', crate)


#: Each pattern with its suites; ``{crate}`` takes the crate a path is in.
#: Order does not matter: every matching row adds its suites.
MAP: tuple[tuple[str, tuple[Suite, ...]], ...] = (
    # The landing tools and what they share.
    ('scripts/rust_core/land.py', (LAND, LANE, T30)),
    ('scripts/rust_core/land_selftest*', (LAND,)),
    ('scripts/rust_core/lane.py', (LANE, LAND)),
    ('scripts/rust_core/lane_selftest.py', (LANE,)),
    ('scripts/rust_core/phases.py', (LAND, LANE, RATCHET)),
    ('scripts/rust_core/judging.py', (LAND, T30)),
    ('scripts/rust_core/judging-paths.txt', (LAND, T30)),
    ('scripts/rust_core/commit_msg.py', (LAND, T30)),
    ('scripts/rust_core/handoff.py', (T30,)),
    ('scripts/rust_core/affected.py', (T30,)),
    ('scripts/rust_core/prepush.sh', (LANE,)),
    ('scripts/rust_core/replay_*', (REPLAY,)),
    ('scripts/rust_core/ratchet.py', (RATCHET, CI)),
    ('scripts/rust_core/orchestration/codex_pane*', (CODEX, T30)),
    ('scripts/rust_core/orchestration/auto_land*', (AUTO_LAND,)),
    ('scripts/rust_core/orchestration/guard_cmd.py', (T30,)),
    ('scripts/rust_core/orchestration/**', ()),
    # Every other script: the tests naming it, else its syntax.
    ('scripts/rust_core/**', ()),
    # The contract: its cases and data go through the reference step and
    # the loader; the runners and tools through their own tests.
    ('contract/cases/**', (LOADER,)),
    ('contract/reference/**', (LOADER,)),
    ('contract/baselines/**', (LOADER,)),
    ('contract/runners/**', (RUNNERS, LOADER)),
    ('contract/tools/**', ()),
    ('contract/tests/conftest.py', ((*PYTEST, 'contract/tests'),)),
    ('contract/tests/**', ()),
    ('contract/*.schema.json', (LOADER,)),
    ('contract/**', (LOADER,)),
    # The crates: each one's own tests.
    ('rust/crates/*/**', (cargo('{crate}'),)),
    # The binding: its Python tests on the test-ops build, as CI's binding
    # job runs them (Codex r2).
    ('rust/bindings/sdm-py/**', (BINDING,)),
    # Each Python package: its own tests, and the client's also the root
    # suites that drive it through the public routes.
    ('packages/*/**', ((*PYTEST, 'packages/{pkg}/tests'),)),
    (
        'packages/structured-data-models-client/src/**',
        (
            (*PYTEST, 'tests/test_release_contract.py'),
            (*PYTEST, 'tests/test_skills.py'),
            RUNNERS,
        ),
    ),
    ('.github/workflows/ci.yml', (CI,)),
)
#: Where the test files are that may name a touched file.
TEST_FILES = (
    'tests/test_*.py',
    'tests/**/test_*.py',
    'contract/tests/test_*.py',
    'contract/tests/**/test_*.py',
    'packages/*/tests/test_*.py',
    'packages/*/tests/**/test_*.py',
    'rust/bindings/*/tests/test_*.py',
    'rust/bindings/*/tests/**/test_*.py',
)
SELFTESTS = 'scripts/rust_core/*selftest.py\nscripts/rust_core/**/*selftest.py'
IS_TEST = judging.patterns('\n'.join(TEST_FILES))


def git(*args: str) -> str:
    done = subprocess.run(
        ['git', '-C', str(ROOT), *args],
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    if done.returncode != 0:
        raise SystemExit(
            f'affected.py: git {" ".join(args)}: {done.stderr.strip()}'
        )
    return done.stdout


def tracked() -> list[str]:
    return [p for p in git('ls-files', '-z').split('\0') if p]


def rows(path: str) -> list[tuple[Suite, ...]]:
    return [
        suites
        for pattern, suites in MAP
        if judging.matches(judging.patterns(pattern), path)
    ]


def mapped(path: str) -> bool:
    return bool(rows(path))


def naming_tests(path: str, files: Sequence[str]) -> list[Suite]:
    """The test files and self-tests that name ``path``'s file name."""
    name = Path(path).name
    stem = Path(path).stem
    # The file's name, or, for a module, an import of it.
    wanted = re.compile(
        rf'(?<![\w.-]){re.escape(name)}(?![\w-])'
        + (
            rf'|\b(import|from)\s+{re.escape(stem)}\b'
            if path.endswith('.py')
            else ''
        )
    )
    tests = judging.patterns('\n'.join(TEST_FILES))
    selftests = judging.patterns(SELFTESTS)
    found: list[Suite] = []
    for candidate in files:
        is_test = judging.matches(tests, candidate)
        if candidate == path or not (
            is_test or judging.matches(selftests, candidate)
        ):
            continue
        try:
            text = (ROOT / candidate).read_text(
                encoding='utf-8', errors='replace'
            )
        except OSError:
            continue
        if wanted.search(text):
            found.append(
                (*PYTEST, candidate) if is_test else ('python3', candidate)
            )
    return found


def syntax(path: str) -> list[Suite]:
    if path.endswith('.py'):
        return [('python3', '-m', 'py_compile', path)]
    if path.endswith('.sh'):
        return [('bash', '-n', path)]
    return []


def suites(
    paths: Iterable[str],
    files: Sequence[str] | None = None,
    naming: bool = True,
) -> list[Suite]:
    """The suites for ``paths``, each once, in a stable order. With
    ``naming`` False, only the map's rows and the paths that are tests
    themselves: no test is added for naming a path (``handoff.py``'s
    affected step, the owner's 2026-10-22 speed change 1; the broad suites
    are CI's)."""
    files = (tracked() if naming else []) if files is None else files
    out: list[Suite] = []
    for path in paths:
        found: list[Suite] = []
        crate = re.match(r'rust/crates/([^/]+)/', path)
        package = re.match(r'packages/([^/]+)/', path)
        for row in rows(path):
            for suite in row:
                found.append(
                    tuple(
                        part.replace(
                            '{crate}', crate.group(1) if crate else ''
                        ).replace('{pkg}', package.group(1) if package else '')
                        for part in suite
                    )
                )
        if judging.matches(IS_TEST, path) and (ROOT / path).is_file():
            found.append((*PYTEST, path))
        if path.endswith('selftest.py') and (ROOT / path).is_file():
            found.append(('python3', path))
        # Docs are read by people; a test naming one is an example, not a
        # reader. The crates' suites are cargo's.
        # A test is its own suite.
        if (
            naming
            and (ROOT / path).is_file()
            and not path.startswith(('rust/', 'docs/'))
            and not judging.matches(IS_TEST, path)
        ):
            found.extend(naming_tests(path, files))
        if not found and (ROOT / path).is_file():
            found.extend(syntax(path))
        for suite in found:
            # The binding's tests run only under BINDING, which supplies
            # --native-build and the test-ops interpreter (Codex r3).
            if suite[: len(PYTEST)] == PYTEST and any(
                part.startswith('rust/bindings/')
                for part in suite[len(PYTEST) :]
            ):
                suite = BINDING
            if suite not in out:
                out.append(suite)
    return out


def changed_since(base: str) -> list[str]:
    out = git('diff', '--name-only', '-z', f'{base}...HEAD')
    return [p for p in out.split('\0') if p]


def default_base() -> str:
    from phases import phase_branch  # noqa: PLC0415

    phase = phase_branch()
    for ref in (f'refs/remotes/origin/{phase}', phase):
        done = subprocess.run(
            ['git', '-C', str(ROOT), 'rev-parse', '--verify', '--quiet', ref],
            capture_output=True,
            text=True,
        )
        if done.returncode == 0:
            return ref
    raise SystemExit(f'affected.py: no {phase} to diff against; give --base')


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split('\n\n', 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('files', nargs='*')
    parser.add_argument('--base')
    parser.add_argument('--unmapped', action='store_true')
    args = parser.parse_args(argv)
    if args.unmapped:
        missing = [
            p for p in tracked() if p.startswith(COVERED) and not mapped(p)
        ]
        for path in missing:
            print(f'UNMAPPED {path}')
        if missing:
            print(f'{len(missing)} path(s) map to no suite: add a row to MAP')
            return 1
        print('every path under ' + ', '.join(COVERED) + ' maps to a suite')
        return 0
    paths = args.files or changed_since(args.base or default_base())
    for suite in suites(paths):
        print(' '.join(suite))
    return 0


if __name__ == '__main__':
    sys.exit(main())
