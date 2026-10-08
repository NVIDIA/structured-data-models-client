# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""P3.diagnostics: a catalog of every warning and refusal a user can meet.

``packages/sdm-client/diagnostics.json`` lists every ``warnings.warn`` and
every raise of ``StructuredDataError`` or one of its subclasses in
``sdm_client`` and ``sdm_relational_engine``, generated from the code by
``scripts/diagnostics.py``. Phase 8 ports the warnings and refusals from it.

The sites are found here independently, by parsing both packages, and the
catalog must list each one: its kind, class, ``code=`` where the source
spells it, every literal part of its message, and a line of the statement.
It must list nothing that is not such a site, hold at least the 221 raises
and 53 warnings the code had when phase 3 started, and be exactly what the
generator writes, including for a copy of the repository with a warning
added.

Since phase 8 it also lists the engine raises that reach users: each raise of
a bare ``ValueError``, ``KeyError`` or ``TypeError``, or of the engine's
``KumoRelationalError`` family, in ``sdm_relational_engine/rfm/inference/``
and ``sdm_relational_engine/rfm/rfm.py``, which the client's
``_engine_errors`` re-raises as ``INVALID_REQUEST``.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from _seam import REPO, output, run_python

CATALOG = REPO / 'packages' / 'sdm-client' / 'diagnostics.json'
GENERATOR = REPO / 'scripts' / 'diagnostics.py'
PACKAGES = (
    'packages/sdm-client/src/sdm_client',
    'packages/sdm-relational-engine/src/sdm_relational_engine',
)
ROOT_CLASS = 'StructuredDataError'
ENGINE = 'packages/sdm-relational-engine/src/sdm_relational_engine'
ENGINE_REACHING = (f'{ENGINE}/rfm/inference/', f'{ENGINE}/rfm/rfm.py')
ENGINE_ROOT_CLASS = 'KumoRelationalError'
ENGINE_BUILTINS = frozenset({'ValueError', 'KeyError', 'TypeError'})
#: The engine raises that reach users when phase 8 started.
MIN_ENGINE_RAISES = 68
MIN_RAISES = 221
MIN_WARNINGS = 53


@dataclass
class Site:
    kind: str
    file: str
    line: int
    end_line: int
    cls: str
    code: str | None
    fragments: list[str] = field(default_factory=list)

    def where(self) -> str:
        return f'{self.file}:{self.line}'


def _name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _fragments(node: ast.AST | None) -> list[str]:
    r"""The literal text of a message expression, part by part."""
    if node is None:
        return []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.JoinedStr):
        return [
            v.value
            for v in node.values
            if isinstance(v, ast.Constant) and isinstance(v.value, str)
        ]
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _fragments(node.left) + _fragments(node.right)
    return []


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
                    if any(_name(b) in classes for b in node.bases):
                        classes.add(node.name)
                        grew = True
    return classes


def _warns_name(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == 'warnings'
        and any(a.name == 'warn' and a.asname is None for a in node.names)
        for node in ast.walk(tree)
    )


def scan(root: Path) -> tuple[list[Site], dict[str, list[tuple[int, int]]]]:
    r"""Every raise and warning site, and where each file builds one of the
    classes, as ``(line, end_line)`` spans."""
    trees = _trees(root)
    classes = _classes(trees)
    reaching = classes | ENGINE_BUILTINS | _classes(trees, ENGINE_ROOT_CLASS)
    sites: list[Site] = []
    builds: dict[str, list[tuple[int, int]]] = {}
    for rel, tree in trees.items():
        bare_warn = _warns_name(tree)
        raised = reaching if rel.startswith(ENGINE_REACHING) else classes
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _name(node.func) in classes:
                builds.setdefault(rel, []).append(
                    (node.lineno, node.end_lineno or node.lineno)
                )
            if isinstance(node, ast.Raise) and node.exc is not None:
                exc = node.exc
                call = exc if isinstance(exc, ast.Call) else None
                cls = _name(call.func if call else exc)
                if cls not in raised:
                    continue
                code = None
                message = None
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
                sites.append(
                    Site(
                        'raise',
                        rel,
                        node.lineno,
                        node.end_lineno or node.lineno,
                        cls,
                        code,
                        _fragments(message),
                    )
                )
            if isinstance(node, ast.Call):
                func = node.func
                is_warn = (
                    isinstance(func, ast.Attribute)
                    and func.attr == 'warn'
                    and isinstance(func.value, ast.Name)
                    and func.value.id == 'warnings'
                ) or (
                    bare_warn
                    and isinstance(func, ast.Name)
                    and func.id == 'warn'
                )
                if not is_warn:
                    continue
                category: ast.AST | None = (
                    node.args[1] if len(node.args) > 1 else None
                )
                message = node.args[0] if node.args else None
                for keyword in node.keywords:
                    if keyword.arg == 'category':
                        category = keyword.value
                    if keyword.arg == 'message':
                        message = keyword.value
                cls = _name(category) if category is not None else 'UserWarning'
                sites.append(
                    Site(
                        'warning',
                        rel,
                        node.lineno,
                        node.end_lineno or node.lineno,
                        cls or '',
                        None,
                        _fragments(message),
                    )
                )
    return sites, builds


@pytest.fixture(scope='module')
def catalog() -> list[dict[str, Any]]:
    assert CATALOG.is_file(), f'{CATALOG.relative_to(REPO)} does not exist'
    data = json.loads(CATALOG.read_text())
    assert isinstance(data, dict) and isinstance(
        data.get('diagnostics'), list
    ), 'the catalog is not an object with a diagnostics array'
    return data['diagnostics']


@pytest.fixture(scope='module')
def sites() -> tuple[list[Site], dict[str, list[tuple[int, int]]]]:
    return scan(REPO)


def test_the_scan_still_sees_the_code(
    sites: tuple[list[Site], dict[str, Any]],
) -> None:
    found, _ = sites
    raises = sum(s.kind == 'raise' for s in found)
    warns = sum(s.kind == 'warning' for s in found)
    assert raises >= MIN_RAISES and warns >= MIN_WARNINGS, (raises, warns)


def test_the_scan_sees_the_engine_raises(
    sites: tuple[list[Site], dict[str, Any]],
) -> None:
    found, _ = sites
    engine = [
        s
        for s in found
        if s.kind == 'raise' and s.file.startswith(ENGINE_REACHING)
    ]
    assert len(engine) >= MIN_ENGINE_RAISES, len(engine)
    assert {s.cls for s in engine} <= ENGINE_BUILTINS | _classes(
        _trees(REPO), ENGINE_ROOT_CLASS
    )


def test_every_entry_has_its_fields(catalog: list[dict[str, Any]]) -> None:
    problems = []
    for index, entry in enumerate(catalog):
        ok = (
            isinstance(entry, dict)
            and entry.get('kind') in ('raise', 'warning')
            and isinstance(entry.get('class'), str)
            and entry['class']
            and (entry.get('code') is None or isinstance(entry['code'], str))
            and 'code' in entry
            and isinstance(entry.get('template'), str)
            and isinstance(entry.get('file'), str)
            and isinstance(entry.get('line'), int)
        )
        if not ok:
            problems.append(f'entry {index}: {entry!r:.300}')
    assert not problems, '\n'.join(problems[:40])
    raises = sum(e['kind'] == 'raise' for e in catalog)
    warns = sum(e['kind'] == 'warning' for e in catalog)
    assert raises >= MIN_RAISES, f'only {raises} raises listed'
    assert warns >= MIN_WARNINGS, f'only {warns} warnings listed'


def _matches(entry: dict[str, Any], site: Site) -> bool:
    return (
        entry.get('kind') == site.kind
        and entry.get('file') == site.file
        and site.line <= entry.get('line', -1) <= site.end_line
    )


def test_every_site_is_listed(
    catalog: list[dict[str, Any]],
    sites: tuple[list[Site], dict[str, Any]],
) -> None:
    found, _ = sites
    problems: list[str] = []
    for site in found:
        entries = [e for e in catalog if _matches(e, site)]
        if not entries:
            problems.append(
                f'{site.where()}: {site.kind} {site.cls} not listed'
            )
            continue
        entry = entries[0]
        if str(entry['class']).rsplit('.', 1)[-1] != site.cls:
            problems.append(
                f'{site.where()}: class {entry["class"]!r}, not {site.cls!r}'
            )
        if site.code is not None and entry.get('code') != site.code:
            problems.append(
                f'{site.where()}: code {entry.get("code")!r}, not {site.code!r}'
            )
        template = str(entry['template'])
        unescaped = template.replace('{{', '{').replace('}}', '}')
        for fragment in site.fragments:
            if fragment.strip() and not (
                fragment in template or fragment in unescaped
            ):
                problems.append(
                    f'{site.where()}: template {template!r:.120} '
                    f'lacks {fragment!r:.80}'
                )
                break
    assert not problems, f'{len(problems)} problem(s):\n' + '\n'.join(
        problems[:60]
    )


def test_nothing_else_is_listed(
    catalog: list[dict[str, Any]],
    sites: tuple[list[Site], dict[str, list[tuple[int, int]]]],
) -> None:
    found, builds = sites
    problems: list[str] = []
    for entry in catalog:
        if any(_matches(entry, site) for site in found):
            continue
        spans = builds.get(entry.get('file', ''), [])
        if entry.get('kind') == 'raise' and any(
            a <= entry.get('line', -1) <= b for a, b in spans
        ):
            continue  # builds a StructuredDataError that a caller raises
        problems.append(
            f'{entry.get("file")}:{entry.get("line")}: no {entry.get("kind")} '
            f'there'
        )
    assert not problems, f'{len(problems)} stale entries:\n' + '\n'.join(
        problems[:60]
    )


def test_the_generator_writes_the_stored_catalog(tmp_path: Path) -> None:
    assert GENERATOR.is_file(), f'{GENERATOR.relative_to(REPO)} is missing'
    fresh = tmp_path / 'diagnostics.json'
    proc = run_python(GENERATOR, '--out', fresh, timeout=300)
    assert proc.returncode == 0, output(proc)
    assert fresh.read_bytes() == CATALOG.read_bytes(), (
        'the stored catalog is not what the generator writes'
    )
    proc = run_python(GENERATOR, '--check', timeout=300)
    assert proc.returncode == 0, output(proc)


def test_the_check_fails_on_any_difference(
    catalog: list[dict[str, Any]], tmp_path: Path
) -> None:
    assert GENERATOR.is_file(), f'{GENERATOR.relative_to(REPO)} is missing'
    data = json.loads(CATALOG.read_text())
    variants = {
        'an entry dropped': {**data, 'diagnostics': data['diagnostics'][1:]},
        'a line moved': {
            **data,
            'diagnostics': [
                {
                    **data['diagnostics'][0],
                    'line': data['diagnostics'][0]['line'] + 1,
                }
            ]
            + data['diagnostics'][1:],
        },
        'a template changed': {
            **data,
            'diagnostics': [
                {
                    **data['diagnostics'][-1],
                    'template': data['diagnostics'][-1]['template'] + '!',
                }
            ]
            + data['diagnostics'][:-1],
        },
    }
    for label, variant in variants.items():
        path = tmp_path / f'{label.replace(" ", "-")}.json'
        path.write_text(json.dumps(variant, indent=2) + '\n')
        proc = run_python(GENERATOR, '--check', path, timeout=300)
        assert proc.returncode != 0, f'--check passed with {label}'


def _copy_repository(destination: Path) -> Path:
    files = subprocess.run(
        ['git', 'ls-files', '-z'],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout.split(b'\0')
    for raw in files:
        if not raw:
            continue
        rel = raw.decode()
        source = REPO / rel
        if not source.is_file():
            continue
        target = destination / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for extra in (CATALOG, GENERATOR):
        target = destination / extra.relative_to(REPO)
        if extra.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(extra, target)
    return destination


def test_the_catalog_follows_the_code(tmp_path: Path) -> None:
    assert GENERATOR.is_file(), f'{GENERATOR.relative_to(REPO)} is missing'
    copy = _copy_repository(tmp_path / 'repo')
    module = (
        copy / 'packages' / 'sdm-client' / 'src' / 'sdm_client' / 'query.py'
    )
    lines = module.read_text().splitlines(keepends=True)
    lines += [
        '\n',
        '\n',
        'def _seam_probe() -> None:\n',
        '    import warnings\n',
        '\n',
        "    warnings.warn(f'seam probe {1 + 1} added', UserWarning)\n",
    ]
    module.write_text(''.join(lines))
    added_line = len(lines)

    generator = copy / 'scripts' / 'diagnostics.py'
    proc = subprocess.run(
        [sys.executable, str(generator), '--check'],
        cwd=copy,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode != 0, (
        '--check passed though the code gained a warning'
    )
    out = tmp_path / 'fresh.json'
    proc = subprocess.run(
        [sys.executable, str(generator), '--out', str(out)],
        cwd=copy,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, output(proc)
    entries = json.loads(out.read_text())['diagnostics']
    rel = 'packages/sdm-client/src/sdm_client/query.py'
    added = [
        e
        for e in entries
        if e.get('file') == rel
        and e.get('line') == added_line
        and e.get('kind') == 'warning'
    ]
    assert added, 'the added warning is not in the regenerated catalog'
    assert 'seam probe ' in added[0]['template']
    assert ' added' in added[0]['template']
