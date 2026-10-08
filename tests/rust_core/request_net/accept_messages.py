# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P8.messages: every warning and refusal of the Phase 8.5 modules has a
fixture that records it, or a reviewed reason why none can.

The sites are found here by parsing the source (``_net.required_sites``),
together with every catalog entry in those modules: each ``StructuredDataError``
raise and each warning in ``decompose.py``, ``request.py``, ``client.py``,
``result.py``, ``base.py``, ``models.py``, ``query.py``, ``errors.py``,
``materializers/``, ``adapters/`` and ``wire/``, each warning in
``rfm/inference/`` and ``rfm/rfm.py``, and the engine raises there that reach
users (P8.catalog).

Every case of every fixture file is run through the Python runner
(``tests/parity/_fixture_ops.py``) and must give what it records. A case is
credited with a site when the case passes and:

- for a raise, the case records ``expected_error`` with ``class``, ``code``,
  ``message`` and ``details_keys``, and the site is where the error the case
  raises, or one it was raised from (``__cause__``, or a shown
  ``__context__``), was raised;
- for a warning, the case records ``expected_warnings`` and the site's
  ``warnings.warn`` gave one of the warnings the case's caller saw.

Each site must be credited to a case or listed in
``tests/parity/UNREACHABLE_DIAGNOSTICS.md`` with a reason; a listed site must
be credited to no case. Whether each reason holds is the reviewer's check.
See ``CONTRACT.md``.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections import defaultdict

import pytest
from _net import (
    BASE,
    REPO,
    UNREACHABLE,
    Outcome,
    Site,
    all_sites,
    catalog_entries,
    check_import_location,
    credits,
    entry_matches,
    fixture_cases,
    fixture_ops,
    p85_modules,
    required_sites,
    run_case,
    table_rows,
    trees,
    unquote,
)

#: About 177 client and engine-warning sites, and about 68 engine raises.
MIN_REQUIRED = 200
SITE = re.compile(r'(?P<file>[\w./-]+\.py):(?P<line>\d+)')


class Run:
    def __init__(self) -> None:
        check_import_location()
        parsed = trees()
        sites = required_sites(parsed)
        every, _ = all_sites(parsed)
        modules = set(p85_modules())
        for entry in catalog_entries():
            if entry.get('file') not in modules:
                continue
            if any(entry_matches(entry, s) for s in sites):
                continue
            sites += [
                s
                for s in every
                if entry_matches(entry, s)
                and s.cls == str(entry.get('class', '')).rsplit('.', 1)[-1]
            ]
        self.sites: list[Site] = sorted(
            set(sites), key=lambda s: (s.file, s.line, s.kind)
        )
        ops = fixture_ops()
        self.cases = fixture_cases()
        self.outcomes: list[Outcome] = [run_case(ops, c) for c in self.cases]
        self.credited: dict[Site, list[str]] = defaultdict(list)
        for outcome in self.outcomes:
            for site in credits(outcome, self.sites):
                self.credited[site].append(str(outcome.case.get('id')))


@pytest.fixture(scope='module')
def run() -> Run:
    return Run()


def _base_case_ids() -> set[str]:
    listed = subprocess.run(
        [
            'git',
            '-C',
            str(REPO),
            'ls-tree',
            '-r',
            '--name-only',
            BASE,
            '--',
            'tests/parity/fixtures',
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    ids: set[str] = set()
    for rel in listed:
        if rel.endswith('.json'):
            text = subprocess.run(
                ['git', '-C', str(REPO), 'show', f'{BASE}:{rel}'],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            ids |= {str(c['id']) for c in json.loads(text)['cases']}
    return ids


def test_every_fixture_passes_against_todays_python(run: Run) -> None:
    ids = [str(c.get('id')) for c in run.cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates, f'case ids repeat: {duplicates[:20]}'
    gone = sorted(_base_case_ids() - set(ids))
    assert not gone, f'{len(gone)} fixture cases of BASE are gone: {gone[:20]}'
    problems = [p for o in run.outcomes for p in o.problems]
    assert not problems, f'{len(problems)} case(s) fail:\n' + '\n'.join(
        problems[:40]
    )


def _unreachable(run: Run) -> tuple[dict[Site, str], list[str]]:
    listed: dict[Site, str] = {}
    problems: list[str] = []
    if not UNREACHABLE.is_file():
        return listed, problems
    rows = table_rows(UNREACHABLE, 4)
    for cells in rows:
        found = SITE.fullmatch(unquote(cells[0]))
        if found is None:
            continue  # a header row
        file, line = found['file'], int(found['line'])
        kind, cls, reason = unquote(cells[1]), unquote(cells[2]), cells[3]
        matching = [
            s for s in run.sites if s.holds(file, line) and s.kind == kind
        ]
        if not matching:
            problems.append(
                f'{file}:{line}: listed as unreachable, but no {kind} of the '
                f'Phase 8.5 modules is there'
            )
            continue
        site = matching[0]
        if cls.rsplit('.', 1)[-1] != site.cls:
            problems.append(
                f'{site.where}: listed as {cls!r}, not {site.cls!r}'
            )
        if len(reason.split()) < 3:
            problems.append(f'{site.where}: the reason {reason!r} says nothing')
        if site in listed:
            problems.append(f'{site.where}: listed twice')
        listed[site] = reason
    return listed, problems


def test_the_scan_still_sees_the_code(run: Run) -> None:
    assert len(run.sites) >= MIN_REQUIRED, (
        f'only {len(run.sites)} sites in the Phase 8.5 modules'
    )


def test_every_site_has_a_fixture_or_a_reason(run: Run) -> None:
    listed, _ = _unreachable(run)
    missing = [
        f'{s.where}: {s.kind} {s.cls}'
        for s in run.sites
        if s not in run.credited and s not in listed
    ]
    total = len(run.sites)
    assert not missing, (
        f'{len(missing)} of {total} sites have neither a fixture nor a '
        f'reason in {UNREACHABLE.relative_to(REPO)} '
        f'({len(run.credited)} credited, {len(listed)} listed):\n'
        + '\n'.join(missing[:80])
    )


def test_the_unreachable_list_holds(run: Run) -> None:
    listed, problems = _unreachable(run)
    for site in listed:
        if site in run.credited:
            problems.append(
                f'{site.where}: listed as unreachable, but fixture '
                f'{run.credited[site][0]} reaches it'
            )
    assert not problems, '\n'.join(problems[:60])


def test_the_unreachable_list_is_a_table(run: Run) -> None:
    if not UNREACHABLE.is_file():
        pytest.skip('no unreachable list; every site has a fixture')
    text = UNREACHABLE.read_text(encoding='utf-8')
    assert re.search(
        r'^\|\s*Site\s*\|\s*Kind\s*\|\s*Class\s*\|\s*Reason\s*\|', text, re.M
    ), 'the list has no | Site | Kind | Class | Reason | table'
