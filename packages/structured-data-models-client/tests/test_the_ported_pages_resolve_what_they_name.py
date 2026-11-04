# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every name the ported attribution reads is a name it can reach.

Moving a module into a package rewrites where its imports point, and a name
that was resolved by the layout it came from is resolved by nothing here. The
failure is a `NameError` at the moment the line runs, which for a page drawn
once in a long explanation is a long way from the move that caused it. These
walk every function of the ported modules and check up front.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import pkgutil
import symtable
from pathlib import Path

import pytest

pytest.importorskip('torch', reason='the ported attribution runs on torch')

PORTED = 'sdm_client.explain._attribution'
NAMES = sorted(
    found.name
    for found in pkgutil.iter_modules(importlib.import_module(PORTED).__path__)
)
MODULES = [f'{PORTED}.{name}' for name in NAMES]


def read_as_globals(source: str, filename: str) -> set[str]:
    """Every name the module's code reads and expects to find around it.

    Asked of Python's own scope analysis rather than worked out from the
    tree. Collecting bindings by walking merges every scope into one, and a
    parameter named `logger` in one function then resolves a `logger` that
    another function reads and nothing defines.
    """
    found: set[str] = set()
    pending = [symtable.symtable(source, filename, 'exec')]
    while pending:
        scope = pending.pop()
        pending.extend(scope.get_children())
        found.update(
            symbol.get_name()
            for symbol in scope.get_symbols()
            if symbol.is_referenced() and symbol.is_global()
        )
    return found


@pytest.mark.parametrize('module_name', MODULES)
def test_the_ported_module_resolves_every_name_it_reads(module_name):
    module = importlib.import_module(module_name)
    source = Path(module.__file__).read_text(encoding='utf-8')
    unresolved = {
        name
        for name in read_as_globals(source, module.__file__)
        if not hasattr(module, name) and not hasattr(builtins, name)
    }

    assert not unresolved


@pytest.mark.parametrize('module_name', MODULES)
def test_the_ported_module_imports_nothing_from_where_it_came_from(
    module_name,
):
    """An import of a bare top-level module is the shape the move leaves
    behind: upstream ran these as scripts beside one another. Both spellings
    of it, because `import channel_flow` leaves the same module missing as
    `from channel_flow import ...` does."""
    module = importlib.import_module(module_name)
    source = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    sibling = {
        named
        for node in ast.walk(source)
        for named in (
            [node.module]
            if isinstance(node, ast.ImportFrom)
            and node.level == 0
            and node.module is not None
            else [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else []
        )
        if named.split('.')[0] in NAMES
    }

    assert not sibling
