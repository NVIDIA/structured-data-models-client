# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1e: a register entry's ``cases`` may name a case's line on another
database, ``<case>@sqlite`` or ``<case>@postgres``, so an entry for that
database names its lines only and never the DuckDB case a family's guard
keeps unexcused; the case part must be a corpus case, and a
``more_exact`` item may name the line as listed.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CASE = 'key.decimal-38-10'


def _register() -> ModuleType:
    name = 't21e_register'
    if name not in sys.modules:
        path = ROOT / 'scripts/rust_core/register.py'
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _problems(cases: list[str], more_exact: str | None = None) -> list[str]:
    r"""The register's problems with its first entry naming :data:`CASE`
    given ``cases``, and a ``more_exact`` item for ``more_exact``."""
    register = _register()
    changed: dict[str, Any] = copy.deepcopy(
        register.load(ROOT / 'docs/design/rust-core/register.yaml')
    )
    entry = next(e for e in changed['entries'] if CASE in e['cases'])
    entry['cases'] = cases
    entry.pop('more_exact', None)
    if more_exact is not None:
        entry['more_exact'] = [
            {
                'case': more_exact,
                'runners': ['python-core'],
                'fields': ['public.result/digest'],
            }
        ]
    found = register.problems(changed, ROOT)
    # A schema problem is named before any entry's.
    assert not any(p.startswith('schema') for p in found), found
    return [p for p in found if p.startswith(f'{entry["id"]}:')]


def test_an_entry_names_a_case_s_line_on_another_database() -> None:
    line = f'{CASE}@postgres'
    assert _problems([line], more_exact=line) == []
    assert _problems([f'{CASE}@sqlite']) == []
    # Another database, and a case outside the corpus, are refused.
    assert _problems([f'{CASE}@mysql'])
    assert _problems(['nonesuch.case@postgres'])
