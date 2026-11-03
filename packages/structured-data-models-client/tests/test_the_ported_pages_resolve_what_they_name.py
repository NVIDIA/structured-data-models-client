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
from pathlib import Path

import pytest

pytest.importorskip('torch', reason='the ported attribution runs on torch')

PORTED = 'sdm_client.explain._attribution'
NAMES = sorted(
    found.name
    for found in pkgutil.iter_modules(importlib.import_module(PORTED).__path__)
)
MODULES = [f'{PORTED}.{name}' for name in NAMES]


def named(tree: ast.AST) -> set[str]:
    """Every global name the module's code reads, bound names aside."""
    bound: set[str] = set()
    read: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            (bound if isinstance(node.ctx, ast.Store) else read).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bound.add(node.name)
            arguments = node.args
            for argument in (
                arguments.args
                + arguments.posonlyargs
                + arguments.kwonlyargs
                + [a for a in (arguments.vararg, arguments.kwarg) if a]
            ):
                bound.add(argument.arg)
        elif isinstance(node, ast.ClassDef):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Global):
            bound.update(node.names)
        elif isinstance(node, (ast.comprehension,)):
            for target in ast.walk(node.target):
                if isinstance(target, ast.Name):
                    bound.add(target.id)
    return read - bound


@pytest.mark.parametrize('module_name', MODULES)
def test_the_ported_module_resolves_every_name_it_reads(module_name):
    module = importlib.import_module(module_name)
    source = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    unresolved = {
        name
        for name in named(source)
        if not hasattr(module, name) and not hasattr(builtins, name)
    }

    assert not unresolved


@pytest.mark.parametrize('module_name', MODULES)
def test_the_ported_module_imports_nothing_from_where_it_came_from(
    module_name,
):
    """An import of a bare top-level module is the shape the move leaves
    behind: upstream ran these as scripts beside one another."""
    module = importlib.import_module(module_name)
    source = ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    sibling = {
        node.module
        for node in ast.walk(source)
        if isinstance(node, ast.ImportFrom)
        and node.level == 0
        and node.module is not None
        and node.module.split('.')[0] in NAMES
    }

    assert not sibling
