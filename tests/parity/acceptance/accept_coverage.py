# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P2.coverage: ``tests/parity/coverage.json`` exists and covers every module
the ledger rules map to a core crate.

The rules are read from the harness, as the brief says. Set
``SDM_PARITY_LEDGER_RULES`` to read them from elsewhere.
"""

from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path
from typing import Any

import pytest
from _support import COVERAGE, PARITY, REPO, output, run

RULES = Path(
    os.environ.get(
        'SDM_PARITY_LEDGER_RULES',
        '~/.local/share/sdm-dbx/rust-core/ledger/rules.tsv',
    )
).expanduser()
CORE_CRATES = (
    'wire',
    'core',
    'pql',
    'schema',
    'dialects',
    'sampling',
    'labels',
    'request',
)
# The ledger's own source roots and exclusions (tools/ledger.py).
SOURCES = (
    'packages/sdm-pql/src',
    'packages/sdm-relational-engine/src',
    'packages/sdm-client/src',
    'packages/sdm-connectors/src',
    'packages/sdm-api/src',
)
SKIP = ('*/generated/*', '*/grammar/PQLGrammar*.py', '*/__pycache__/*')
# The corpus exercises each of these; zero covered branches means coverage
# was not measured.
EXERCISED = ('pql', 'schema', 'dialects', 'sampling', 'labels', 'request')


def _module_rules() -> list[tuple[str, str]]:
    out = []
    for line in RULES.read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        pattern, dest = line.split('\t')[:2]
        if '::' in pattern:
            # A rule for some symbols, not for the module.
            continue
        out.append((pattern, dest))
    return out


def expected_modules() -> dict[str, str]:
    rules = _module_rules()
    out = {}
    for source in SOURCES:
        for path in sorted((REPO / source).rglob('*.py')):
            rel = str(path.relative_to(REPO))
            if any(fnmatch.fnmatch(rel, g) for g in SKIP):
                continue
            for pattern, dest in rules:
                if fnmatch.fnmatch(rel, pattern):
                    crate = dest.removeprefix('triage:')
                    if crate in CORE_CRATES:
                        out[rel] = crate
                    break
    return out


def _check_shape(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    r"""``{module: entry}``, after checking every entry's shape."""
    crates = report['crates']
    modules = {}
    for crate, body in crates.items():
        for module, entry in body['modules'].items():
            assert module not in modules, f'{module} is in two crates'
            total = entry['num_branches']
            covered = entry['covered_branches']
            missing = entry['missing_branches']
            assert isinstance(total, int) and isinstance(covered, int)
            assert 0 <= covered <= total, (module, covered, total)
            assert isinstance(missing, list)
            assert len(missing) == total - covered, (
                f'{module}: {len(missing)} listed, {total - covered} uncovered'
            )
            modules[module] = {**entry, 'crate': crate}
    return modules


@pytest.fixture(scope='module')
def stored() -> dict[str, dict[str, Any]]:
    return _check_shape(json.loads((PARITY / 'coverage.json').read_text()))


def test_the_ledger_rules_map_modules_to_core_crates() -> None:
    expected = expected_modules()
    assert len(expected) > 100, 'the ledger rules matched too few modules'
    assert set(expected.values()) >= set(EXERCISED)


def test_every_core_module_is_in_the_report_under_its_crate(
    stored: dict[str, dict[str, Any]],
) -> None:
    expected = expected_modules()
    missing = sorted(set(expected) - set(stored))
    assert not missing, f'{len(missing)} modules missing: {missing[:20]}'
    misplaced = sorted(
        m for m, crate in expected.items() if stored[m]['crate'] != crate
    )
    assert not misplaced, [
        (m, stored[m]['crate'], expected[m]) for m in misplaced
    ]


def test_coverage_was_measured(stored: dict[str, dict[str, Any]]) -> None:
    for crate in EXERCISED:
        covered = sum(
            e['covered_branches']
            for e in stored.values()
            if e['crate'] == crate
        )
        assert covered > 0, f'no branch of {crate} covered'


def test_the_report_is_reproduced_by_the_script(
    stored: dict[str, dict[str, Any]], tmp_path: Path
) -> None:
    out = tmp_path / 'coverage.json'
    proc = run(COVERAGE, '--out', out, timeout=1800)
    assert proc.returncode == 0, output(proc)
    fresh = _check_shape(json.loads(out.read_text()))
    assert set(fresh) == set(stored)
    for module, entry in fresh.items():
        assert entry['crate'] == stored[module]['crate'], module
        assert entry['num_branches'] == stored[module]['num_branches'], module
