# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P85.coverage: at least 90% branch coverage of the code the ledger sends
to ``request`` or ``core``, module by module, or a gap list a reviewer
checks (``CONTRACT.md``, "Coverage").

As Phase 7.5's check: the corpus replay and the fixture and PQL runners run
under branch coverage (``scripts/parity/coverage.py``'s ``measure``), and a
branch is ported when its source line belongs, innermost symbol deciding, to
a symbol the harness's ledger rules send to ``request`` or ``core``. Each
module is at least 90% taken, or every untaken branch is in
``scripts/parity/request_coverage_gaps.json`` with a reason; every entry
there is a ported branch left untaken.
"""

from __future__ import annotations

import json
from typing import Any

from _request import GAPS, REPO, RUST_CORE

THRESHOLD = 0.9
CRATES = frozenset({'request', 'core'})


def _phase_75() -> Any:
    import importlib.util

    path = RUST_CORE / 'sampling' / 'driver' / 'accept_driver_coverage.py'
    spec = importlib.util.spec_from_file_location('p75_coverage', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ported_branches(
    files: dict[str, Any], p75: Any
) -> dict[str, tuple[set[tuple[int, int]], set[tuple[int, int]]]]:
    r"""By module: its ported branches taken, and not taken."""
    table = p75.rules()
    out = {}
    for module, body in sorted(files.items()):
        if not module.endswith('.py'):
            continue
        spans = p75.symbols(REPO / module)
        cache: dict[int, bool] = {}

        def ported(
            line: int,
            spans: list[Any] = spans,
            module: str = module,
            cache: dict[int, bool] = cache,
        ) -> bool:
            if line not in cache:
                dest = p75.destination(table, module, p75.owner(spans, line))
                cache[line] = dest.removeprefix('triage:') in CRATES
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


def _listed(problems: list[str]) -> dict[str, dict[tuple[int, int], str]]:
    listed: dict[str, dict[tuple[int, int], str]] = {}
    if not GAPS.is_file():
        return listed
    gaps = json.loads(GAPS.read_text('utf-8'))
    if not isinstance(gaps, dict):
        problems.append(f'{GAPS.name} is not an object')
        return listed
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
            if not (isinstance(reason, str) and len(reason.split()) >= 3):
                problems.append(
                    f'{GAPS.name}: {module} {branch}: the reason is under '
                    f'three words'
                )
            key = (branch[0], branch[1])
            if key in listed.setdefault(module, {}):
                problems.append(f'{GAPS.name}: {module} {branch}: listed twice')
            listed[module][key] = reason or ''
    return listed


def test_ported_branches_are_covered_or_listed() -> None:
    p75 = _phase_75()
    branches = ported_branches(p75.measured(), p75)
    assert len(branches) >= 20, f'only {len(branches)} modules were measured'
    problems: list[str] = []
    listed = _listed(problems)
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
    assert not problems, '\n'.join(problems[:60] + ['', *summary])
