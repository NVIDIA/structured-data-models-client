# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P8.catalog: ``diagnostics.json`` lists the engine raises that reach users,
and is what ``scripts/diagnostics.py`` writes.

The engine raises are found here independently, by parsing the source: every
``raise`` of a bare ``ValueError``, ``KeyError`` or ``TypeError``, or of the
engine's ``KumoRelationalError`` family, in
``sdm_relational_engine/rfm/inference`` and ``sdm_relational_engine/rfm/rfm.py``.
``materializers/relational.py::_engine_errors`` turns each of these into an
``INVALID_REQUEST`` carrying its text. The catalog must list each one with its
kind (``raise``), its class, every literal part of its message in the
template, and a line of the statement.

It must still list every ``StructuredDataError`` raise and every warning of
both packages, list nothing that is not a real site, and be byte for byte what
the generator writes; ``--check`` must agree. The Phase 3 regeneration test
(``tests/seam/test_diagnostics_catalog.py``) runs beside this file in the
check.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from _net import (
    CATALOG,
    GENERATOR,
    REPO,
    ROOT_CLASS,
    Site,
    all_sites,
    catalog_entries,
    engine_sites,
    entry_matches,
    family,
    template_lacks,
    trees,
)

#: The bare raises research counted in ``rfm/inference`` and ``rfm/rfm.py``
#: (ValueError 64, KeyError 3, TypeError 1), less a margin.
MIN_ENGINE_SITES = 60


@pytest.fixture(scope='module')
def parsed() -> dict:
    return trees()


@pytest.fixture(scope='module')
def scanned(parsed: dict) -> tuple[list[Site], dict]:
    return all_sites(parsed)


def _listed(entries: list[dict], site: Site) -> list[str]:
    r"""What is wrong with how ``entries`` list ``site``, empty when right."""
    found = [e for e in entries if entry_matches(e, site)]
    if not found:
        return [f'{site.where}: {site.kind} {site.cls} is not listed']
    entry = found[0]
    problems = []
    if str(entry.get('class', '')).rsplit('.', 1)[-1] != site.cls:
        problems.append(
            f'{site.where}: class {entry.get("class")!r}, not {site.cls!r}'
        )
    lacks = template_lacks(entry, site)
    if lacks is not None:
        problems.append(
            f'{site.where}: template {entry.get("template")!r:.120} lacks '
            f'{lacks!r:.80}'
        )
    return problems


def test_the_engine_raises_that_reach_users_are_listed(
    parsed: dict, scanned: tuple[list[Site], dict]
) -> None:
    sites = engine_sites(parsed, scanned[0])
    assert len(sites) >= MIN_ENGINE_SITES, (
        f'the scan found only {len(sites)} engine raises; it no longer sees '
        f'the code'
    )
    entries = catalog_entries()
    problems = [p for site in sites for p in _listed(entries, site)]
    assert not problems, f'{len(problems)} problem(s):\n' + '\n'.join(
        problems[:60]
    )


def test_every_refusal_and_warning_is_still_listed(
    parsed: dict, scanned: tuple[list[Site], dict]
) -> None:
    sde = family(parsed, ROOT_CLASS)
    sites = [
        s
        for s in scanned[0]
        if s.kind == 'warning' or (s.kind == 'raise' and s.cls in sde)
    ]
    assert len(sites) >= 274, f'only {len(sites)} sites found'
    entries = catalog_entries()
    problems = [p for site in sites for p in _listed(entries, site)]
    assert not problems, f'{len(problems)} problem(s):\n' + '\n'.join(
        problems[:60]
    )


def test_nothing_else_is_listed(scanned: tuple[list[Site], dict]) -> None:
    sites, builds = scanned
    problems = []
    for entry in catalog_entries():
        matching = [s for s in sites if entry_matches(entry, s)]
        if any(
            str(entry.get('class', '')).rsplit('.', 1)[-1] == s.cls
            for s in matching
        ):
            continue
        spans = builds.get(str(entry.get('file')), [])
        line = entry.get('line')
        if (
            entry.get('kind') == 'raise'
            and isinstance(line, int)
            and any(a <= line <= b for a, b in spans)
        ):
            continue
        problems.append(
            f'{entry.get("file")}:{line}: no {entry.get("kind")} of '
            f'{entry.get("class")} there'
        )
    assert not problems, f'{len(problems)} stale entries:\n' + '\n'.join(
        problems[:60]
    )


def test_the_generator_writes_the_stored_catalog(tmp_path: Path) -> None:
    fresh = tmp_path / 'diagnostics.json'
    proc = subprocess.run(
        [sys.executable, str(GENERATOR), '--out', str(fresh)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert fresh.read_bytes() == CATALOG.read_bytes(), (
        'the stored catalog is not what scripts/diagnostics.py writes'
    )
    proc = subprocess.run(
        [sys.executable, str(GENERATOR), '--check'],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
