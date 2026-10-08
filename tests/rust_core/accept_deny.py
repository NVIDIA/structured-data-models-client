# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P4.deny: the licence audit allows only what Apache-2.0 can ship.

``cargo deny check licenses bans`` passing says little on its own: a broad
allow-list, an exception or a clarification passes anything. So the list is
held to the Apache-2.0-compatible set below, and a copy of the workspace
with a GPL dependency added must fail the same audit.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import tomllib
from _rc import RUST, cargo_metadata, find_crate, run

# Licences that a binary distributed under Apache-2.0 can include without
# imposing further terms: the ASF's category A.
COMPATIBLE = {
    '0BSD',
    'Apache-2.0',
    'Apache-2.0 WITH LLVM-exception',
    'BSD-2-Clause',
    'BSD-3-Clause',
    'BSL-1.0',
    'CC0-1.0',
    'ISC',
    'MIT',
    'MIT-0',
    'NCSA',
    'Unicode-3.0',
    'Unicode-DFS-2016',
    'Unlicense',
    'Zlib',
}


def deny_config() -> dict:
    return tomllib.loads((RUST / 'deny.toml').read_text())


def test_only_apache_compatible_licences_are_allowed():
    licenses = deny_config().get('licenses', {})
    allowed = licenses.get('allow', [])
    assert allowed, 'deny.toml allows no licence at all'
    assert set(allowed) <= COMPATIBLE, (
        f'deny.toml allows {sorted(set(allowed) - COMPATIBLE)}'
    )
    for exception in licenses.get('exceptions', []):
        extra = set(exception.get('allow', [])) - COMPATIBLE
        assert not extra, f'an exception for {exception} allows {extra}'
    assert not licenses.get('clarify'), (
        'deny.toml clarifies a licence; clarifications need a review of their '
        'own, not a pass from this check'
    )


def test_a_gpl_dependency_fails_the_audit():
    r"""The audit is real: a GPL path dependency in a copy is refused."""
    meta = cargo_metadata()
    wire = find_crate('wire', meta)
    with tempfile.TemporaryDirectory(prefix='sdm-rc-deny-') as tmp:
        copy = Path(tmp) / 'rust'
        shutil.copytree(
            RUST,
            copy,
            ignore=shutil.ignore_patterns('target'),
        )
        probe = Path(tmp) / 'gpl-probe'
        (probe / 'src').mkdir(parents=True)
        (probe / 'Cargo.toml').write_text(
            '[package]\nname = "gpl-probe"\nversion = "0.1.0"\n'
            'edition = "2021"\nlicense = "GPL-3.0-only"\n'
            'description = "acceptance probe"\n'
            'repository = "https://example.invalid/gpl-probe"\n'
        )
        (probe / 'src' / 'lib.rs').write_text('//! Probe.\n')
        rel = Path(wire['manifest_path']).resolve().relative_to(RUST.resolve())
        manifest = copy / rel
        text = manifest.read_text()
        dep = f'gpl-probe = {{ path = "{probe}" }}\n'
        if '\n[dependencies]\n' in text:
            text = text.replace(
                '\n[dependencies]\n', '\n[dependencies]\n' + dep
            )
        else:
            text += '\n[dependencies]\n' + dep
        manifest.write_text(text)
        audit = run(
            ['cargo', 'deny', 'check', 'licenses', 'bans'],
            cwd=copy,
            check=False,
        )
        assert 'gpl-probe' in audit.stdout + audit.stderr, (
            'the audit did not look at the added dependency:\n'
            + audit.stderr[-4000:]
        )
        assert audit.returncode != 0, (
            'cargo deny passed a GPL-3.0-only dependency'
        )
