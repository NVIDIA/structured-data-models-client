# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The binding's dependency edges (decision E21).

Among first-party crates, ``sdm-py`` depends only on ``sdm-host`` and, for
types, ``sdm-contract``, never on ``sdm-core``; every third-party dependency
it has is one ``dependencies.md`` lists for it. From T1.10c,
``scripts/rust_core/l0_deps.py`` checks the whole first-party graph, the
binding's edges included; this test keeps the third-party half.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
BINDING = ROOT / 'rust/bindings/sdm-py/Cargo.toml'
DEPENDENCIES = ROOT / 'docs/design/rust-core/phase-1/dependencies.md'
FIRST_PARTY = ROOT / 'rust/crates'


def _manifest() -> dict[str, object]:
    with BINDING.open('rb') as file:
        return tomllib.load(file)


def test_the_binding_depends_on_sdm_host_and_sdm_contract_alone() -> None:
    manifest = _manifest()
    first_party = {path.name for path in FIRST_PARTY.iterdir()}
    sections = [
        manifest.get('dependencies', {}),
        manifest.get('build-dependencies', {}),
    ]
    named = {name for section in sections for name in section}
    assert named & first_party == {'sdm-host', 'sdm-contract'}
    features = manifest['features']
    assert isinstance(features, dict)
    assert features['test-ops'] == [
        'sdm-host/test-ops',
        'sdm-contract/test-ops',
    ]


def test_every_third_party_dependency_is_listed_for_the_binding() -> None:
    manifest = _manifest()
    dependencies = manifest['dependencies']
    assert isinstance(dependencies, dict)
    third_party = {name for name in dependencies if not name.startswith('sdm-')}
    rows = [
        line.split('|')
        for line in DEPENDENCIES.read_text(encoding='utf-8').splitlines()
        if line.startswith('| `')
    ]
    for name in sorted(third_party):
        listed = [
            row
            for row in rows
            if f'`{name}`' in row[1] and re.search(r'`sdm-py`', row[3])
        ]
        assert listed, f'{name} has no dependencies.md row naming sdm-py'
