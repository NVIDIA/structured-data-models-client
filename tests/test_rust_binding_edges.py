# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the Python binding thin: host/contract edges and Arrow/PyO3 adapters."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
BINDING = ROOT / 'rust/bindings/sdm-py/Cargo.toml'
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


def test_binding_dependencies_are_only_arrow_and_python_adapters() -> None:
    manifest = _manifest()
    dependencies = manifest['dependencies']
    assert isinstance(dependencies, dict)
    third_party = {name for name in dependencies if not name.startswith('sdm-')}
    assert third_party == {
        'arrow-array',
        'arrow-data',
        'arrow-schema',
        'pyo3',
        'pyo3-log',
    }
