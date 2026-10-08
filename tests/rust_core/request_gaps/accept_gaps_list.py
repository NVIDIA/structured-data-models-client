# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P86.gaps: no entry of ``scripts/parity/request_coverage_gaps.json`` that
Phase 8.5's verifier called wrong stays as it was, and
``tests/parity/coverage.json`` shows the public branches taken
(``CONTRACT.md``, "The gap list" and "Host tests").

- No public branch is listed.
- Every named entry still listed has a reason other than its base reason,
  without the clauses the verifier showed false.
- A listed route or narrow entry says "Reachable only" and cites, by pytest
  node id, a host test for each route the verifier found.
- Every host test a named "Reachable only" entry cites passes, calls into
  ``sdm_core._native``, and takes the entry's branch in today's Python.
- ``tests/parity/coverage.json`` is what ``scripts/parity/coverage.py``
  writes, and no public branch is among its missing branches.
- A public branch's module is in ``tests/parity/coverage.json``, unless the
  ledger keeps the module in Python (its first whole-module rule, read as
  ``coverage.py`` reads it, is ``python``), so ``coverage.py`` does not
  report it; then a held case (``_held.py``) must take the branch.
"""

from __future__ import annotations

import fnmatch
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from functools import cache
from pathlib import Path
from typing import Any

from _reqgaps import (
    COVERAGE_JSON,
    ENTRIES,
    NARROW,
    PUBLIC,
    REACHABLE_ONLY,
    REPO,
    ROUTE,
    SCRIPTS,
    Entry,
    cited,
    held_measure,
    host_measure,
    listed,
    listed_at_base,
)

RUST_CORE = REPO / 'tests' / 'rust_core'


def _still_listed() -> list[tuple[Entry, str]]:
    found = listed()
    return [
        (entry, found[(entry.module, entry.branch)])
        for entry in ENTRIES
        if (entry.module, entry.branch) in found
    ]


def test_every_named_entry_was_listed_at_the_base() -> None:
    at_base = listed_at_base()
    missing = [
        entry.name
        for entry in ENTRIES
        if (entry.module, entry.branch) not in at_base
    ]
    assert not missing, missing


def test_no_public_branch_is_listed() -> None:
    still = [
        f'{entry.name} ({entry.what})'
        for entry, _ in _still_listed()
        if entry.group == PUBLIC
    ]
    assert not still, (
        'ordinary public calls reach these branches, so a recorded case must '
        'take them and the gap list must not list them:\n' + '\n'.join(still)
    )


def test_no_named_entry_stays_as_it_was() -> None:
    at_base = listed_at_base()
    problems = []
    for entry, reason in _still_listed():
        if ' '.join(reason.split()) == ' '.join(
            at_base[(entry.module, entry.branch)].split()
        ):
            problems.append(f'{entry.name}: the reason is the base reason')
        for clause in entry.false:
            if clause.lower() in reason.lower():
                problems.append(
                    f'{entry.name}: the reason still says {clause!r}, which '
                    f'the verifier showed false ({entry.what})'
                )
    assert not problems, '\n'.join(problems)


def test_route_and_narrow_entries_cite_a_host_test_per_route() -> None:
    problems = []
    for entry, reason in _still_listed():
        if entry.group not in (ROUTE, NARROW):
            continue
        if not reason.lstrip().startswith(REACHABLE_ONLY):
            problems.append(
                f'{entry.name}: public calls reach it ({entry.what}), so its '
                f'reason must start {REACHABLE_ONLY!r}'
            )
        tests = cited(reason)
        if len(tests) < entry.routes:
            problems.append(
                f'{entry.name}: cites {len(tests)} host tests by node id, '
                f'and the verifier found {entry.routes} routes ({entry.what})'
            )
    assert not problems, '\n'.join(problems)


def _reachable_only() -> list[tuple[Entry, list[str]]]:
    return [
        (entry, cited(reason))
        for entry, reason in _still_listed()
        if reason.lstrip().startswith(REACHABLE_ONLY)
    ]


@cache
def _hosts() -> dict[str, Any]:
    sys.path.insert(0, str(RUST_CORE / 'request'))
    sys.path.insert(0, str(RUST_CORE / 'sampling'))
    sys.path.insert(0, str(RUST_CORE))
    from _request import native

    # Built from this checkout and installed once; the run below imports it.
    native()
    node_ids = sorted(
        {node for _, nodes in _reachable_only() for node in nodes}
    )
    return host_measure(node_ids, {'SDM_P7_WHEEL_INSTALLED': '1'})


def test_cited_host_tests_hold_their_branches() -> None:
    entries = _reachable_only()
    problems = []
    for entry, nodes in entries:
        if not nodes:
            problems.append(
                f'{entry.name}: says {REACHABLE_ONLY!r} and cites no host test '
                f'by node id'
            )
    found = _hosts() if any(nodes for _, nodes in entries) else {}
    for entry, nodes in entries:
        for node in nodes:
            body = found.get(node)
            if body is None:
                problems.append(f'{entry.name}: {node} did not run')
                continue
            outcomes = dict(map(tuple, body['outcomes']))
            if outcomes.get('call') != 'passed' or any(
                outcome != 'passed' for outcome in outcomes.values()
            ):
                problems.append(f'{entry.name}: {node} gave {outcomes}')
            if body['native_calls'] < 1:
                problems.append(
                    f'{entry.name}: {node} made no call into sdm_core._native'
                )
            if entry.branch not in body['executed'].get(entry.module, set()):
                problems.append(
                    f'{entry.name}: {node} does not take the branch in '
                    f"today's Python"
                )
    assert not problems, '\n'.join(problems)


@cache
def _fresh_report() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix='sdm-p86-coverage-') as tmp:
        out = Path(tmp) / 'coverage.json'
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / 'coverage.py'), '--out', str(out)],
            cwd=REPO,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            timeout=7200,
        )
        assert proc.returncode == 0, (
            f'coverage.py failed ({proc.returncode}):\n'
            f'{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}'
        )
        return json.loads(out.read_text('utf-8'))


def _modules(report: dict[str, Any]) -> dict[str, Any]:
    return {
        module: body
        for entry in report['crates'].values()
        for module, body in entry['modules'].items()
    }


def test_coverage_json_is_what_coverage_py_writes() -> None:
    stored = json.loads(COVERAGE_JSON.read_text('utf-8'))
    fresh = _fresh_report()
    old, new = _modules(stored), _modules(fresh)
    stale = sorted(
        module
        for module in set(old) | set(new)
        if old.get(module) != new.get(module)
    )
    assert not stale, (
        f'tests/parity/coverage.json is stale for {len(stale)} modules, such '
        f'as {stale[:5]}: regenerate it with scripts/parity/coverage.py'
    )
    assert stored == fresh


@cache
def _coverage_py() -> Any:
    spec = importlib.util.spec_from_file_location(
        '_p86_parity_coverage', SCRIPTS / 'coverage.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def _module_rules() -> tuple[tuple[str, str], ...]:
    r"""The ledger's whole-module rules, from the path and by the reading
    ``scripts/parity/coverage.py`` uses."""
    measure = _coverage_py()
    rules = Path(
        os.environ.get('SDM_PARITY_LEDGER_RULES', measure.DEFAULT_RULES)
    ).expanduser()
    assert rules.is_file(), f'no ledger rules at {rules}'
    return tuple(measure.module_rules(rules))


def _first_rule(module: str) -> str | None:
    r"""The destination of the first whole-module rule matching
    ``module``, as ``coverage.py``'s ``core_modules`` picks it."""
    for pattern, crate in _module_rules():
        if fnmatch.fnmatch(module, pattern):
            return crate
    return None


@cache
def _held() -> dict[str, Any]:
    return held_measure()


def test_coverage_json_shows_the_public_branches_taken() -> None:
    modules = _modules(json.loads(COVERAGE_JSON.read_text('utf-8')))
    problems = []
    for entry in ENTRIES:
        if entry.group != PUBLIC:
            continue
        body = modules.get(entry.module)
        if body is None:
            rule = _first_rule(entry.module)
            if rule != 'python':
                problems.append(
                    f'{entry.name}: its module is not in the report, and its '
                    f'first whole-module ledger rule is {rule!r}, not '
                    f"'python'"
                )
            elif entry.branch not in _held()['executed'].get(
                entry.module, set()
            ):
                problems.append(
                    f'{entry.name} ({entry.what}): the ledger keeps its module '
                    f'in Python, so the report leaves it out, and no held '
                    f'case takes the branch'
                )
            continue
        missing = {tuple(b) for b in body.get('missing_branches', [])}
        if entry.branch in missing:
            problems.append(f'{entry.name} ({entry.what}) is still untaken')
    assert not problems, '\n'.join(problems)
