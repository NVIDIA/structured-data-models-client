# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two distributions installed together must not pin one package differently.

The five are released as a set and a caller installs several at once, so a pin
only one of them carries makes the set unsatisfiable. `pip` reports it as
`ResolutionImpossible` at install time, which is late and reads as the user's
problem. It happened: the query language moved to ANTLR 4.13.2 and the engine's
`setup.py` kept 4.9.3, so `structured-data-models-client[relational]` could not be installed at
all.
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:  # These packages support 3.10, where tomllib is not in the stdlib.
    tomllib = pytest.importorskip('tomli')

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGES = (
    'structured-data-models-api',
    'structured-data-models-pql',
    'structured-data-models-client',
    'structured-data-models-connectors',
    'structured-data-models-relational-engine',
)
_REQUIREMENT = re.compile(r'^([A-Za-z0-9._-]+)\s*(?:\[[^\]]*\])?\s*(.*)$')


def _requirements(package: str) -> list[str]:
    directory = _ROOT / 'packages' / package
    config = tomllib.loads((directory / 'pyproject.toml').read_text())
    project = config.get('project', {})
    found = list(project.get('dependencies', []))
    for extra in project.get('optional-dependencies', {}).values():
        found.extend(extra)

    # The engine declares its runtime dependencies in setup.py, which is where
    # the pin that broke this went unnoticed.
    setup = directory / 'setup.py'
    if setup.exists():
        text = setup.read_text()
        block = re.search(r'install_requires=\[(.*?)\]', text, re.DOTALL)
        if block:
            found.extend(re.findall(r"['\"]([^'\"]+)['\"]", block.group(1)))
    return found


def test_no_two_packages_pin_the_same_dependency_differently() -> None:
    constraints: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for package in _PACKAGES:
        for requirement in _requirements(package):
            match = _REQUIREMENT.match(requirement.strip())
            if not match:
                continue
            name, specifier = match.group(1).lower(), match.group(2).strip()
            if specifier.startswith('=='):
                constraints[name][specifier].add(package)

    disagreements = {
        name: {pin: sorted(who) for pin, who in pins.items()}
        for name, pins in constraints.items()
        if len(pins) > 1
    }
    assert not disagreements, (
        f'these dependencies are pinned to different versions by different '
        f'packages, so installing them together is impossible: {disagreements}'
    )
