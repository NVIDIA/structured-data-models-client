#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write the catalog of every warning and refusal a user can meet.

``packages/structured-data-models-client/diagnostics.json`` lists each ``warnings.warn`` and each
raise of ``StructuredDataError`` or a subclass in ``sdm_client`` and
``sdm_relational_engine``, read from the source, not run. It also lists the
engine raises that reach users: each raise of a bare ``ValueError``,
``KeyError`` or ``TypeError``, or of the engine's ``KumoRelationalError``
family, in ``sdm_relational_engine/rfm/inference/`` and
``sdm_relational_engine/rfm/rfm.py``. The client's ``_engine_errors``
(``sdm_client/materializers/relational.py``) re-raises each of those as an
``INVALID_REQUEST`` carrying its text. Each entry holds:

* ``kind``: ``raise`` or ``warning``;
* ``class``: the exception or warning class, as the source names it;
* ``code``: the ``code=`` the raise passes, or ``null`` where there is none or
  it is not a literal;
* ``template``: the message, each interpolated part written as ``{...}``
  holding its source, and every literal part as it is, a literal brace
  doubled;
* ``file`` and ``line``: where the ``raise`` or the ``warnings.warn`` starts.

Phase 8 of the Rust-core migration ports them from this list, so a new or
changed warning or refusal shows up in review as a change to the catalog.

    diagnostics.py --out PATH       write the catalog to PATH
    diagnostics.py --check [PATH]   exit 1 unless PATH, by default the stored
                                    catalog, is what the code gives now
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

#: The repository this script is in.
REPO = Path(__file__).resolve().parent.parent

CATALOG = Path('packages/structured-data-models-client/diagnostics.json')

PACKAGES = (
    'packages/structured-data-models-client/src/sdm_client',
    'packages/structured-data-models-relational-engine/src/sdm_relational_engine',
)

#: The class every refusal derives from.
ROOT_CLASS = 'StructuredDataError'

ENGINE = 'packages/structured-data-models-relational-engine/src/sdm_relational_engine'
#: Where the engine raises that reach users are: the inference code the
#: client's materializer calls, which ``_engine_errors`` wraps.
ENGINE_REACHING = (f'{ENGINE}/rfm/inference/', f'{ENGINE}/rfm/rfm.py')
#: The engine's own error family, which ``_engine_errors`` also re-raises.
ENGINE_ROOT_CLASS = 'KumoRelationalError'
#: The built-in errors ``_engine_errors`` re-raises as ``INVALID_REQUEST``.
ENGINE_BUILTINS = frozenset({'ValueError', 'KeyError', 'TypeError'})


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _escape(text: str) -> str:
    return text.replace('{', '{{').replace('}', '}}')


def _placeholder(node: ast.FormattedValue) -> str:
    text = ast.unparse(node.value)
    if node.conversion != -1:
        text += f'!{chr(node.conversion)}'
    if node.format_spec is not None:
        text += f':{_template(node.format_spec, nested=True)}'
    return f'{{{text}}}'


def _template(node: ast.AST | None, *, nested: bool = False) -> str:
    r"""A message expression's text: literal parts as they are, each
    interpolated part as ``{...}`` holding its source."""
    if node is None:
        return ''
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value if nested else _escape(node.value)
    if isinstance(node, ast.JoinedStr):
        return ''.join(
            _placeholder(value)
            if isinstance(value, ast.FormattedValue)
            else _template(value, nested=nested)
            for value in node.values
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _template(node.left) + _template(node.right)
    return f'{{{ast.unparse(node)}}}'


def _trees(root: Path) -> dict[str, ast.Module]:
    trees: dict[str, ast.Module] = {}
    for package in PACKAGES:
        for path in sorted((root / package).rglob('*.py')):
            rel = path.relative_to(root).as_posix()
            trees[rel] = ast.parse(path.read_text(encoding='utf-8'), rel)
    return trees


def _classes(trees: dict[str, ast.Module], root: str = ROOT_CLASS) -> set[str]:
    r"""``root`` and every class deriving from it, by name."""
    classes = {root}
    grew = True
    while grew:
        grew = False
        for tree in trees.values():
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name not in classes:
                    if any(_name(base) in classes for base in node.bases):
                        classes.add(node.name)
                        grew = True
    return classes


def _imports_warn(tree: ast.Module) -> bool:
    r"""Whether the module calls ``warnings.warn`` as ``warn``."""
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == 'warnings'
        and any(a.name == 'warn' and a.asname is None for a in node.names)
        for node in ast.walk(tree)
    )


def _raise(
    rel: str, node: ast.Raise, classes: set[str]
) -> dict[str, Any] | None:
    exc = node.exc
    if exc is None:
        return None
    call = exc if isinstance(exc, ast.Call) else None
    cls = _name(call.func if call else exc)
    if cls not in classes:
        return None
    code = None
    message: ast.AST | None = None
    if call is not None:
        for keyword in call.keywords:
            if (
                keyword.arg == 'code'
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                code = keyword.value.value
            if keyword.arg == 'message':
                message = keyword.value
        if call.args:
            message = call.args[0]
    return {
        'kind': 'raise',
        'class': cls,
        'code': code,
        'template': _template(message),
        'file': rel,
        'line': node.lineno,
    }


def _warning(
    rel: str, node: ast.Call, bare_warn: bool
) -> dict[str, Any] | None:
    func = node.func
    is_warn = (
        isinstance(func, ast.Attribute)
        and func.attr == 'warn'
        and isinstance(func.value, ast.Name)
        and func.value.id == 'warnings'
    ) or (bare_warn and isinstance(func, ast.Name) and func.id == 'warn')
    if not is_warn:
        return None
    category: ast.AST | None = node.args[1] if len(node.args) > 1 else None
    message: ast.AST | None = node.args[0] if node.args else None
    for keyword in node.keywords:
        if keyword.arg == 'category':
            category = keyword.value
        if keyword.arg == 'message':
            message = keyword.value
    if category is None:
        cls = 'UserWarning'
    else:
        cls = _name(category) or ast.unparse(category)
    return {
        'kind': 'warning',
        'class': cls,
        'code': None,
        'template': _template(message),
        'file': rel,
        'line': node.lineno,
    }


def catalog(root: Path) -> dict[str, Any]:
    r"""Every warning and refusal in the packages under ``root``."""
    trees = _trees(root)
    classes = _classes(trees)
    reaching = classes | ENGINE_BUILTINS | _classes(trees, ENGINE_ROOT_CLASS)
    entries: list[dict[str, Any]] = []
    for rel, tree in trees.items():
        bare_warn = _imports_warn(tree)
        raised = reaching if rel.startswith(ENGINE_REACHING) else classes
        for node in ast.walk(tree):
            entry = None
            if isinstance(node, ast.Raise):
                entry = _raise(rel, node, raised)
            elif isinstance(node, ast.Call):
                entry = _warning(rel, node, bare_warn)
            if entry is not None:
                entries.append(entry)
    entries.sort(key=lambda e: (e['file'], e['line'], e['kind']))
    return {
        'about': (
            'Every warning and refusal sdm_client and sdm_relational_engine '
            'give, with the engine raises that reach users as '
            'INVALID_REQUEST, generated by scripts/diagnostics.py. Do not edit.'
        ),
        'diagnostics': entries,
    }


def render(root: Path) -> str:
    return json.dumps(catalog(root), indent=2, ensure_ascii=False) + '\n'


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path, help='write the catalog here')
    group.add_argument(
        '--check',
        type=Path,
        nargs='?',
        const=REPO / CATALOG,
        help='compare this catalog, the stored one by default, with the code',
    )
    args = parser.parse_args(argv)
    text = render(REPO)
    if args.out is not None:
        args.out.write_text(text, encoding='utf-8')
        return 0
    path: Path = args.check
    if not path.is_file():
        print(f'{path}: no catalog; run with --out', file=sys.stderr)
        return 1
    if path.read_text(encoding='utf-8') != text:
        print(
            f'{path} is not what the code gives now; regenerate it with '
            f'scripts/diagnostics.py --out {CATALOG}',
            file=sys.stderr,
        )
        return 1
    print(f'{path} matches the code')
    return 0


if __name__ == '__main__':
    sys.exit(main())
