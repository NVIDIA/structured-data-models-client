# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P8.coverage: the parity corpus reaches at least 90% of the branches of
each Phase 8.5 module, or every branch it misses there is listed with a
reason.

Coverage is measured here, not read from ``tests/parity/coverage.json``: the
corpus replay (``scripts/parity/run_corpus.py replay``), the fixture runner
(``tests/parity/test_fixtures.py``) and the PQL corpus runner
(``tests/parity/test_pql_corpus.py``) run under branch coverage of
``sdm_client`` and ``sdm_relational_engine``, as ``scripts/parity/coverage.py``
runs them. Those are what a port is compared against, so only they count.

The modules are those ``_net.coverage_modules`` names (CONTRACT.md,
Coverage). A module under 90% passes only if each of its missing branches,
as coverage names it (``from->to``, a negative ``to`` being an exit), is a
row of ``tests/parity/COVERAGE_GAPS.md`` with a reason; a listed branch must
be one the corpus misses. Whether each reason holds is the reviewer's check.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _net import (
    COVERAGE_GAPS,
    PARITY,
    REPO,
    RUN_CORPUS,
    check_import_location,
    coverage_modules,
    table_rows,
    unquote,
)

THRESHOLD = 0.90
BRANCH = re.compile(r'(-?\d+)\s*->\s*(-?\d+)')

_RC = """\
[run]
branch = True
parallel = True
data_file = {data}
source_pkgs = sdm_client, sdm_relational_engine
"""


def _measure(scratch: Path) -> dict[str, Any]:
    rc = scratch / 'coveragerc'
    rc.write_text(_RC.format(data=scratch / 'data'))
    runs = [
        [str(RUN_CORPUS), 'replay'],
        [
            '-m',
            'pytest',
            str(PARITY / 'test_fixtures.py'),
            '-q',
            '-p',
            'no:cacheprovider',
        ],
        [
            '-m',
            'pytest',
            str(PARITY / 'test_pql_corpus.py'),
            '-q',
            '-p',
            'no:cacheprovider',
        ],
    ]
    for args in runs:
        proc = subprocess.run(
            [sys.executable, '-m', 'coverage', 'run', f'--rcfile={rc}', *args],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=7200,
        )
        assert proc.returncode == 0, (
            f'{args} failed under coverage:\n{proc.stdout[-4000:]}\n'
            f'{proc.stderr[-2000:]}'
        )
    for args in (['combine'], ['json', '-o', str(scratch / 'report.json')]):
        proc = subprocess.run(
            [sys.executable, '-m', 'coverage', *args, f'--rcfile={rc}'],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads((scratch / 'report.json').read_text())
    files: dict[str, Any] = {}
    for path, body in report['files'].items():
        absolute = Path(path)
        if not absolute.is_absolute():
            absolute = REPO / absolute
        try:
            rel = absolute.resolve().relative_to(REPO.resolve()).as_posix()
        except ValueError:
            continue
        files[rel] = body
    return files


@pytest.fixture(scope='module')
def measured(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    check_import_location()
    return _measure(tmp_path_factory.mktemp('p8-coverage'))


def _gaps() -> tuple[dict[tuple[str, int, int], str], list[str]]:
    listed: dict[tuple[str, int, int], str] = {}
    problems: list[str] = []
    if not COVERAGE_GAPS.is_file():
        return listed, problems
    for cells in table_rows(COVERAGE_GAPS, 3):
        module = unquote(cells[0])
        found = BRANCH.fullmatch(unquote(cells[1]))
        if not module.endswith('.py') or found is None:
            continue  # a header row
        key = (module, int(found[1]), int(found[2]))
        if len(cells[2].split()) < 3:
            problems.append(f'{module} {cells[1]}: the reason says nothing')
        if key in listed:
            problems.append(f'{module} {cells[1]}: listed twice')
        listed[key] = cells[2]
    return listed, problems


def _missing(body: dict[str, Any]) -> set[tuple[int, int]]:
    return {(int(a), int(b)) for a, b in body.get('missing_branches', [])}


def test_every_module_is_measured(measured: dict[str, Any]) -> None:
    absent = [m for m in coverage_modules() if m not in measured]
    assert not absent, f'coverage did not measure {absent}'


def test_each_module_reaches_90_percent_or_lists_its_gaps(
    measured: dict[str, Any],
) -> None:
    listed, _ = _gaps()
    short: list[str] = []
    for module in coverage_modules():
        body = measured.get(module)
        if body is None:
            continue
        total = body['summary']['num_branches']
        covered = body['summary']['covered_branches']
        if not total or covered / total >= THRESHOLD:
            continue
        unlisted = sorted(
            (a, b) for a, b in _missing(body) if (module, a, b) not in listed
        )
        if unlisted:
            short.append(
                f'{module}: {covered}/{total} branches '
                f'({100 * covered / total:.1f}%), {len(unlisted)} missing '
                f'branches unlisted, such as '
                + ', '.join(f'{a}->{b}' for a, b in unlisted[:6])
            )
    assert not short, (
        f'{len(short)} module(s) under {THRESHOLD:.0%} with unlisted gaps:\n'
        + '\n'.join(short)
    )


def test_the_gap_list_holds(measured: dict[str, Any]) -> None:
    listed, problems = _gaps()
    modules = set(coverage_modules())
    for module, a, b in listed:
        if module not in modules:
            problems.append(f'{module}: not a module this check measures')
            continue
        body = measured.get(module)
        if body is None or (a, b) not in _missing(body):
            problems.append(
                f'{module} {a}->{b}: listed as a gap, but the corpus reaches '
                f'it or it is no branch'
            )
    assert not problems, '\n'.join(problems[:60])
