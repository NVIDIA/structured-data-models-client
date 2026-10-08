# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P75.coverage: at least 90% branch coverage of each ported module, or a
gap list a reviewer checks.

The corpus replay and the fixture and PQL runners run under branch coverage
(``scripts/parity/coverage.py``'s ``measure``). A module's ported branches
are those whose source line belongs, innermost symbol deciding, to a symbol
the porting ledger's rules send to ``sampling`` (``CONTRACT.md``,
"Coverage"). For each module with ported branches, at least 90% are taken,
or every one not taken is in ``scripts/parity/sampling_coverage_gaps.json``
with a reason; every entry there is a ported branch not taken.
"""

from __future__ import annotations

import ast
import fnmatch
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from _driver import GAPS, REPO

THRESHOLD = 0.9
CRATE = 'sampling'


def harness() -> Path:
    return Path(
        os.environ.get('SDM_RC_HARNESS', '~/.local/share/sdm-dbx/rust-core')
    ).expanduser()


def rules() -> list[tuple[str, str, str]]:
    r"""``(path glob, symbol glob, destination)``, in order, as
    ``tools/ledger.py`` reads them."""
    out = []
    path = harness() / 'ledger' / 'rules.tsv'
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        pattern, dest = line.split('\t')[:2]
        path_glob, _, symbol_glob = pattern.partition('::')
        out.append((path_glob, symbol_glob or '*', dest))
    return out


def destination(
    table: list[tuple[str, str, str]], path: str, symbol: str
) -> str:
    for path_glob, symbol_glob, dest in table:
        if fnmatch.fnmatch(path, path_glob) and fnmatch.fnmatch(
            symbol, symbol_glob
        ):
            return dest
    return 'UNMAPPED'


def symbols(path: Path) -> list[tuple[str, int, int]]:
    r"""Every function and class of a module, nested ones by their dotted
    name as the ledger names them, with their first and last lines."""
    found: list[tuple[str, int, int]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        if isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            name = f'{prefix}{node.name}'
            found.append((name, node.lineno, node.end_lineno or node.lineno))
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    walk(child, f'{name}.')

    for node in ast.parse(path.read_text(), filename=str(path)).body:
        walk(node, '')
    return found


def owner(spans: list[tuple[str, int, int]], line: int) -> str:
    inside = [s for s in spans if s[1] <= line <= s[2]]
    if not inside:
        return '<module>'
    return min(inside, key=lambda s: s[2] - s[1])[0]


def measured() -> dict[str, Any]:
    script = REPO / 'scripts' / 'parity' / 'coverage.py'
    spec = importlib.util.spec_from_file_location('parity_coverage', script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='sdm-p75-coverage-') as tmp:
        report, runs = module.measure(Path(tmp))
    assert 'corpus replay' in runs, f'coverage ran {runs}'
    files = {}
    for path, body in report['files'].items():
        absolute = Path(path)
        if not absolute.is_absolute():
            absolute = REPO / absolute
        try:
            relative = absolute.resolve().relative_to(REPO.resolve())
        except ValueError:
            continue
        files[str(relative)] = body
    return files


def ported_branches(
    files: dict[str, Any],
) -> dict[str, tuple[set[tuple[int, int]], set[tuple[int, int]]]]:
    r"""By module: its ported branches taken, and not taken."""
    table = rules()
    out = {}
    for module, body in sorted(files.items()):
        if not module.endswith('.py'):
            continue
        spans = symbols(REPO / module)
        cache: dict[int, bool] = {}

        def ported(
            line: int,
            spans: list = spans,
            module: str = module,
            cache: dict = cache,
        ) -> bool:
            if line not in cache:
                dest = destination(table, module, owner(spans, line))
                cache[line] = dest.removeprefix('triage:') == CRATE
            return cache[line]

        taken = {
            (a, b) for a, b in body.get('executed_branches', []) if ported(a)
        }
        missing = {
            (a, b) for a, b in body.get('missing_branches', []) if ported(a)
        }
        if taken or missing:
            out[module] = (taken, missing)
    return out


def test_ported_branches_are_covered_or_listed() -> None:
    branches = ported_branches(measured())
    assert branches, 'no ported branch was measured'
    gaps: dict[str, Any] = {}
    if GAPS.is_file():
        gaps = json.loads(GAPS.read_text('utf-8'))
        assert isinstance(gaps, dict), f'{GAPS.name} is not an object'
    listed: dict[str, dict[tuple[int, int], str]] = {}
    problems = []
    for module, entries in gaps.items():
        if not isinstance(entries, list):
            problems.append(f'{GAPS.name}: {module} is not a list')
            continue
        for entry in entries:
            branch = entry.get('branch') if isinstance(entry, dict) else None
            reason = entry.get('reason') if isinstance(entry, dict) else None
            if not (
                isinstance(branch, list)
                and len(branch) == 2
                and all(isinstance(x, int) for x in branch)
            ):
                problems.append(f'{GAPS.name}: {module}: entry {entry!r}')
                continue
            if not (isinstance(reason, str) and reason.strip()):
                problems.append(f'{GAPS.name}: {module} {branch}: no reason')
            listed.setdefault(module, {})[(branch[0], branch[1])] = reason or ''
    for module, found in listed.items():
        missing = branches.get(module, (set(), set()))[1]
        stale = sorted(set(found) - missing)
        if stale:
            problems.append(
                f'{GAPS.name}: {module} lists {stale[:5]}, which are not '
                f'ported branches left untaken'
            )
    summary = []
    for module, (taken, missing) in branches.items():
        total = len(taken) + len(missing)
        share = len(taken) / total
        summary.append(f'{module}: {len(taken)}/{total} ({share:.1%})')
        if share >= THRESHOLD:
            continue
        unlisted = sorted(missing - set(listed.get(module, {})))
        if unlisted:
            problems.append(
                f'{module}: {len(taken)} of {total} ported branches taken '
                f'({share:.1%}), and {len(unlisted)} untaken are not in '
                f'{GAPS.name}, such as {unlisted[:8]}'
            )
    assert not problems, '\n'.join(problems[:40] + ['', *summary])
