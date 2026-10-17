# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""T2.1d: a ``more_exact`` item names a line's key, ``<case>@sqlite`` or
``<case>@postgres`` on another database, as a case outcome does (T2.0f);
the part before ``@`` one of its entry's cases, and no other database.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def _module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


register = _module('register', ROOT / 'scripts/rust_core/register.py')
REGISTER = register.load(ROOT / 'docs/design/rust-core/register.yaml')


def _problems(case: str) -> list[str]:
    r"""The register's problems with its first entry naming
    ``key.decimal-38-10`` given a ``more_exact`` item for ``case``."""
    changed = copy.deepcopy(REGISTER)
    entry = next(
        e for e in changed['entries'] if 'key.decimal-38-10' in e['cases']
    )
    entry.setdefault('more_exact', []).append(
        {
            'case': case,
            'runners': ['python-core'],
            'fields': ['public.result/digest'],
        }
    )
    return [p for p in register.problems(changed, ROOT) if entry['id'] in p] + [
        p for p in register.problems(changed, ROOT) if 'schema' in p
    ]


def test_a_more_exact_item_names_a_database_s_line() -> None:
    assert _problems('key.decimal-38-10@postgres') == []
    assert _problems('key.decimal-38-10@sqlite') == []
    # Another database, and a case not the entry's, are refused.
    assert _problems('key.decimal-38-10@mysql')
    assert any(
        'not one of its cases' in p for p in _problems('nonesuch.case@postgres')
    )
