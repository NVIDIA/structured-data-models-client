# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixed structural and preserved-gate execution, independent of other criteria."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import name_signatures
from support import HERE, checked_path, require

STRUCTURAL = {
    'rust/crates/request/src/layout.rs': 'encode',
    'rust/crates/request/src/request.rs': 'encode',
    'rust/crates/labels/src/executor.rs': 'parse_timestamp',
}


def amendment_rows(root: Path):
    def parse(raw):
        records = {}
        for line in raw.splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            fields = line.split('\t')
            require(
                len(fields) == 2 and fields[0] not in records,
                'amendment.duplicate_or_malformed_row',
            )
            records[fields[0]] = fields[1]
        return records

    required = parse((HERE / 'required-changed-test-rows.tsv').read_text())
    require(len(required) == 3, 'amendment.required_rows')
    actual = parse(
        (root / 'tests/rust_core/shadow/changed_tests.tsv').read_text()
    )
    require(
        all(actual.get(path) == text for path, text in required.items()),
        'amendment.required_row_changed',
    )


def tracked_sources(root: Path) -> dict:
    paths = subprocess.check_output(
        ['git', 'ls-files', '-z', 'rust'], cwd=root
    ).split(b'\0')
    return {
        name.decode(): checked_path(root, name.decode()).read_text(
            errors='replace'
        )
        for name in paths
        if name
    }


def names(root: Path, exemptions: list, accepted_dvs: list):
    amendment_rows(root)
    sources = tracked_sources(root)
    name_signatures.validate(root, sources, exemptions, accepted_dvs)
    for path, function in STRUCTURAL.items():
        text = sources[path]
        require(
            not re.search(r'\bfixture\s*::', text), 'structure.fixture_import'
        )
        require(
            not re.search(r'\bfixture\s*::\s*' + function, text),
            'structure.fixture_call',
        )


def dialect_keys(root: Path, expected: dict):
    paths = subprocess.check_output(
        ['git', 'ls-files', '-z', 'tests/parity/dialects'], cwd=root
    ).split(b'\0')
    observed = {}
    for raw in filter(None, paths):
        name = raw.decode()
        text = checked_path(root, name).read_text()
        count = text.count('_Keys')
        if count:
            observed[name] = count
    require(observed == expected, 'names.dialect_disposition')


def phase_harness() -> Path:
    value = os.environ.get('P97_PHASE_HARNESS')
    require(type(value) is str and bool(value), 'preserved.harness_required')
    return Path(value)


def preserved_rows(harness: Path | None = None) -> list:
    harness = phase_harness() if harness is None else harness

    def rows(path):
        result = {}
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            fields = line.split('\t')
            require(
                len(fields) == 4 and fields[0] not in result, 'preserved.row'
            )
            result[fields[0]] = fields
        return result

    original = rows(harness / 'phases/phase-9/checks.tsv')
    require(len(original) == 32, 'preserved.original_count')
    destination = harness / 'phases/phase-9.7'
    require(
        (destination / 'BASE').is_file() and (destination / 'LOCK').is_file(),
        'preserved.not_locked',
    )
    current = rows(destination / 'checks.tsv')
    rebound = {key: list(value) for key, value in original.items()}
    for key in ('P9.ledger', 'P9.corpus'):
        require(
            rebound[key][2].count('--phase 9') == 1, 'preserved.phase_argument'
        )
        rebound[key][2] = rebound[key][2].replace('--phase 9', '--phase 9.7')
    old_base = str(harness / 'phases/phase-9/BASE')
    require(
        rebound['P9.public'][2].count(old_base) == 1, 'preserved.public_base'
    )
    rebound['P9.public'][2] = rebound['P9.public'][2].replace(
        old_base, str(destination / 'BASE')
    )
    for row in rebound.values():
        for package in (
            'client',
            'pql',
            'connectors',
            'api',
            'relational-engine',
        ):
            row[2] = re.sub(
                r'(?<!\S)'
                + re.escape(f'packages/sdm-{package}')
                + r'(?=/|\s|$)',
                f'packages/structured-data-models-{package}',
                row[2],
            )
        row[2] = row[2].replace(str(harness), '"$P97_HARNESS"')
    require(
        all(current.get(key) == value for key, value in rebound.items()),
        'preserved.changed_check',
    )
    return list(original)


def run_preserved(root: Path, selected: list[str]) -> int:
    amendment_rows(root)
    names = preserved_rows()
    require(
        bool(selected)
        and len(set(selected)) == len(selected)
        and set(selected) <= set(names),
        'preserved.selection',
    )
    return subprocess.run(
        [
            sys.executable,
            str(phase_harness() / 'tools/gate.py'),
            '9.7',
            '--worktree',
            str(root),
            '--only',
            ','.join(selected),
        ],
        check=False,
    ).returncode
