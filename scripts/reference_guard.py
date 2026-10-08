#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Refuse to run as the reference if the Rust core can be loaded.

The ``rust-core-reference`` branch is today's Python with only the
determinism fixes, and other environments compare their answers with it.
That comparison means nothing if the reference itself reached the Rust core,
so this exits non-zero when any module the core would ship is already
imported, or could be imported from this interpreter's path.

Run it at the start of every reference session, with the reference's own
interpreter:

    .venv/bin/python scripts/reference_guard.py
"""

from __future__ import annotations

import importlib.util
import sys

# Top-level names the Rust core's Python bindings may take.
_TOP_LEVEL = ('sdm_core', 'sdm_py', '_native', 'sdm_core_py')

# The five import roots, each of which could grow a native submodule.
_IMPORT_ROOTS = (
    'sdm_api',
    'sdm_client',
    'sdm_connectors',
    'sdm_relational_engine',
    'predictive_query',
)
_SUBMODULES = ('_native', '_core', 'sdm_core', 'sdm_py')


def _candidates() -> list[str]:
    names = list(_TOP_LEVEL)
    for root in _IMPORT_ROOTS:
        names.extend(f'{root}.{sub}' for sub in _SUBMODULES)
    return names


def _importable(name: str) -> bool:
    parent, _, _ = name.rpartition('.')
    if parent and importlib.util.find_spec(parent) is None:
        return False
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def problems() -> list[str]:
    r"""Every Rust core module that is imported or importable here."""
    found = []
    for name in _candidates():
        if name in sys.modules:
            found.append(f'{name} is imported')
        elif _importable(name):
            spec = importlib.util.find_spec(name)
            origin = spec.origin if spec is not None else None
            found.append(f'{name} is importable from {origin}')
    return found


def main() -> int:
    found = problems()
    if found:
        print('the reference environment can load the Rust core:')
        for line in found:
            print(f'  {line}')
        return 1
    print(f'reference guard: none of {len(_candidates())} names load')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
