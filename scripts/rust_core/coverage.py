#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The coverage check (T1.15; design, "Coverage generator"): the scope
manifests held to the rules of ``scope-format.md`` and to the L2 corpus.

    scripts/rust_core/coverage.py --check [--repo DIR] [--rev REV]
        --checkpoint REV

The candidate is ``--rev`` (``HEAD``) of the git repository ``--repo``, read
from git, so a file left uncommitted or untracked counts for nothing:
``docs/design/rust-core/phase-0/scope-python.yaml`` and ``scope-java.yaml``,
validated against ``scope.schema.json`` beside them, and the cases under
``contract/cases``. Each case says what it covers, in its ``paths`` and
``cells``; the coverage of a manifest's path or cell is generated from those,
counting acceptance cases only (a diagnostic case counts toward no coverage,
design "Contract corpus"). The rules, each problem printed naming its rule:

1. backend, operation, path and cell IDs are unique within a manifest;
2. each path is ``<operation>/<mode>/<backend>``, its operation declared,
   its mode one of that operation's and its backend declared;
3. every declared operation, each of its modes and every declared backend
   appear in at least one path;
4. every ID resolves: each case ID a manifest lists is a case of the corpus,
   and each path and route a case names is a path of the Python manifest and
   each cell a cell of it; a manifest's ``cases`` lists are generated, never
   written by hand, so each is empty (``--write`` is not built, E66(f)) or
   exactly its generated list, and Java's generated lists are empty until a
   Java runner observes cases;
5. the active slice gates: in Python, every S0 path has an acceptance case;
   S0 requires no cell, since every cell applies from S1, and Java is checked
   structurally only (design: Java runtime coverage starts with Java);
8. against the manifests at ``--checkpoint``: no path or cell is removed, no
   path moves to a later slice and no cell's ``expect`` turns ``unknown``
   (each would take it out of rule 5), unless a row of the checkpoint's
   ``docs/design/rust-core/phase-1/decisions.md``, its status starting
   "approved by owner" and a date with nothing after but a parenthesis or
   a note, none withdrawing it (``owner_approved``), names the ID in
   backticks; a row the candidate writes never authorizes its own change.

Rules 6 and 7 are Phase 2's (``--gate S1``); ``--gate S0`` was cut by the
owner (E66(f)), its rule 5 run by ``--check`` for the active slice. A file it
cannot read (missing, not UTF-8, not YAML or JSON, nested past the parser,
or not of its shape) is refused by name. Exit status 0 when every rule holds,
1 on any problem, 2 on a usage error; input values are never printed.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
PHASE_0 = 'docs/design/rust-core/phase-0'
MANIFESTS = {
    'python': f'{PHASE_0}/scope-python.yaml',
    'java': f'{PHASE_0}/scope-java.yaml',
}
SCHEMA = f'{PHASE_0}/scope.schema.json'
CASES = 'contract/cases'
DECISIONS = 'docs/design/rust-core/phase-1/decisions.md'
SLICES = ('S0', 'S1', 'S2', 'S3')
#: The language and slice whose coverage gates (design, "Coverage
#: generator"): S0 in Python.
ACTIVE = ('python', 'S0')
#: The deepest nesting a case or the schema may hold (they hold 8 and 7):
#: deeper is refused by name on every Python, whether its parser recurses
#: (a RecursionError through 3.13) or not (3.14 reads 100,000 levels).
MAX_DEPTH = 64
CASE_ID = re.compile(r'[a-z0-9][a-z0-9._-]*')
#: An owner's approval, as a decisions row's status (its last column) reads
#: it: these words at its start and a date, followed by nothing, a
#: parenthesis, or a note after "; " or ", " (the forms decisions.md uses).
OWNER = re.compile(r'approved by owner \d{4}-\d{2}-\d{2}(?:(?: \(|[;,] ).*)?')
#: A note that withdraws or qualifies the approval voids it: fail closed.
WITHDRAWN = re.compile(
    r'not approved|unapproved|reject|revoke|withdraw|supersede|to confirm',
    re.IGNORECASE,
)
ROW = re.compile(r'\| [A-Z][A-Z0-9-]* \|')


def owner_approved(status: str) -> bool:
    r"""The status is the owner's approval, read from its start: never a
    substring of another status ("not approved by owner ...")."""
    status = status.strip()
    return bool(OWNER.fullmatch(status)) and not WITHDRAWN.search(status)


class Unreadable(Exception):
    r"""A file this check cannot read, named."""


@functools.cache
def _module(name: str) -> ModuleType:
    r"""A module of this harness: ``ratchet.py``'s git reading and
    ``register.py``'s YAML reading are shared, not copied."""
    spec = importlib.util.spec_from_file_location(
        f'coverage_{name}', HERE / f'{name}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: its dataclasses look their module up.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Report:
    def __init__(self) -> None:
        self.problems: list[str] = []

    def fail(self, rule: str, text: str) -> None:
        self.problems.append(f'{rule}: {text}')


# --- reading -------------------------------------------------------------------


def _text(tree: Any, path: str) -> str:
    try:
        text = tree.read(path)
    except _module('ratchet').Unreadable as error:
        raise Unreadable(str(error)) from None
    if text is None:
        raise Unreadable(f'{path}: missing at {tree.sha[:12]}')
    return str(text)


def _yaml(text: str, name: str) -> Any:
    register = _module('register')
    try:
        return register.read_yaml(text, name)
    except register.Unreadable as error:
        raise Unreadable(str(error)) from None


def _strings(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def _deeper(value: Any, limit: int) -> bool:
    r"""``value`` nests past ``limit`` levels, walked without recursion."""
    stack = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        if isinstance(item, (dict, list)):
            if depth > limit:
                return True
            children = item.values() if isinstance(item, dict) else item
            stack.extend((child, depth + 1) for child in children)
    return False


def _json(text: str, name: str) -> Any:
    r"""A JSON document, or :class:`Unreadable` naming why not: not JSON,
    or nested past :data:`MAX_DEPTH`, the same refusal on every Python."""
    try:
        value = json.loads(text)
    except RecursionError:
        raise Unreadable(f'{name}: nested past {MAX_DEPTH} levels') from None
    except ValueError:
        raise Unreadable(f'{name}: not JSON it reads') from None
    if _deeper(value, MAX_DEPTH):
        raise Unreadable(f'{name}: nested past {MAX_DEPTH} levels')
    return value


def read_schema(tree: Any, report: Report) -> Any:
    r"""The manifests' validator, or None, the problem reported once."""
    import jsonschema

    try:
        schema = _json(_text(tree, SCHEMA), SCHEMA)
        if not isinstance(schema, dict):
            raise Unreadable(f'{SCHEMA}: not a schema it reads')
        cls = jsonschema.validators.validator_for(schema)
        cls.check_schema(schema)
    except Unreadable as error:
        report.fail('read', str(error))
        return None
    except (jsonschema.SchemaError, RecursionError):
        report.fail('read', f'{SCHEMA}: not a schema it reads')
        return None
    return cls(schema)


def read_manifest(
    tree: Any, language: str, validator: Any, report: Report
) -> Any:
    r"""A manifest, read and held to the schema, or None, the problem
    reported: the rules need its shape."""
    path = MANIFESTS[language]
    try:
        manifest = _yaml(_text(tree, path), path)
    except Unreadable as error:
        report.fail('read', str(error))
        return None
    found = []
    try:
        for error in validator.iter_errors(manifest):
            where = '/'.join(str(part) for part in error.absolute_path)
            found.append(
                f'{path}: {where or "(root)"} fails {error.validator!r}'
            )
    except RecursionError:
        found.append(f'{path}: nested past what the validator reads')
    if isinstance(manifest, dict) and manifest.get('language') != language:
        found.append(f'{path}: its language is not {language}')
    for problem in sorted(found):
        report.fail('schema', problem)
    return None if found else manifest


def read_cases(tree: Any, report: Report) -> dict[str, dict[str, Any]] | None:
    r"""The corpus by ID: each case's status and what it names, or None, a
    case unread reported (an unread case could hide what it covers)."""
    found: dict[str, dict[str, Any]] = {}
    unread = False
    try:
        names = [name for name in tree.files(CASES) if name.endswith('.json')]
    except _module('ratchet').Unreadable as error:
        report.fail('read', str(error))
        return None
    if not names:
        report.fail('read', f'{CASES}: no cases at {tree.sha[:12]}')
        return None
    for name in names:
        try:
            case = _json(_text(tree, name), name)
        except Unreadable as error:
            report.fail('read', str(error))
            unread = True
            continue
        problem = None
        if not isinstance(case, dict):
            problem = 'not an object'
        elif not isinstance(case.get('id'), str) or not CASE_ID.fullmatch(
            case['id']
        ):
            problem = 'its id is not a case ID'
        elif case.get('status') not in ('acceptance', 'diagnostic'):
            problem = 'its status is neither acceptance nor diagnostic'
        elif not all(
            _strings(case.get(key)) for key in ('paths', 'routes', 'cells')
        ):
            problem = 'its paths, routes or cells are not lists of IDs'
        elif case['id'] in found:
            problem = f'a second case {case["id"]}'
        if problem is not None:
            report.fail('read', f'{name}: {problem}')
            unread = True
            continue
        found[case['id']] = case
    return None if unread else found


# --- the rules -----------------------------------------------------------------


def _ids(manifest: dict[str, Any], key: str) -> list[str]:
    return [str(item['id']) for item in manifest[key]]


def rule_1(language: str, manifest: dict[str, Any], report: Report) -> None:
    for key in ('backends', 'operations', 'paths', 'cells'):
        for name, count in sorted(Counter(_ids(manifest, key)).items()):
            if count > 1:
                report.fail(
                    'rule 1',
                    f'{language}: {key} {name} is listed {count} times',
                )


def rule_2_3(language: str, manifest: dict[str, Any], report: Report) -> None:
    operations = {
        str(item['id']): [str(mode) for mode in item['modes']]
        for item in manifest['operations']
    }
    backends = set(_ids(manifest, 'backends'))
    used_modes: set[tuple[str, str]] = set()
    used_backends: set[str] = set()
    for path in _ids(manifest, 'paths'):
        parts = path.split('/')
        if len(parts) != 3:
            report.fail(
                'rule 2',
                f'{language}: path {path} is not operation/mode/backend',
            )
            continue
        operation, mode, backend = parts
        if operation not in operations:
            report.fail(
                'rule 2', f'{language}: path {path}: no operation {operation}'
            )
        elif mode not in operations[operation]:
            report.fail(
                'rule 2',
                f'{language}: path {path}: {operation} has no mode {mode}',
            )
        if backend not in backends:
            report.fail(
                'rule 2', f'{language}: path {path}: no backend {backend}'
            )
        used_modes.add((operation, mode))
        used_backends.add(backend)
    for operation, modes in operations.items():
        for mode in modes:
            if (operation, mode) not in used_modes:
                report.fail(
                    'rule 3',
                    f'{language}: {operation} mode {mode} is in no path',
                )
    for backend in sorted(backends - used_backends):
        report.fail('rule 3', f'{language}: backend {backend} is in no path')


def generated(
    language: str, manifest: dict[str, Any], corpus: dict[str, dict[str, Any]]
) -> dict[str, dict[str, list[str]]]:
    r"""Each path's and cell's cases as the corpus says: the acceptance
    cases naming it, by ID; none in Java until a Java runner observes."""
    out: dict[str, dict[str, list[str]]] = {'paths': {}, 'cells': {}}
    for key in out:
        for name in _ids(manifest, key):
            out[key][name] = sorted(
                case_id
                for case_id, case in corpus.items()
                if language == 'python'
                and case['status'] == 'acceptance'
                and name in case[key]
            )
    return out


def rule_4(
    manifests: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    report: Report,
) -> None:
    python = manifests.get('python')
    if python is not None:
        paths, cells = set(_ids(python, 'paths')), set(_ids(python, 'cells'))
        for case_id, case in sorted(corpus.items()):
            for key, known in (
                ('paths', paths),
                ('routes', paths),
                ('cells', cells),
            ):
                for name in case[key]:
                    if name not in known:
                        report.fail(
                            'rule 4',
                            f'case {case_id}: its {key} name {name}, not a '
                            f'{"cell" if key == "cells" else "path"} of the '
                            'Python manifest',
                        )
    for language, manifest in sorted(manifests.items()):
        made = generated(language, manifest, corpus)
        for key in ('paths', 'cells'):
            for item in manifest[key]:
                listed = [str(x) for x in item['cases']]
                for case_id in listed:
                    if case_id not in corpus:
                        report.fail(
                            'rule 4',
                            f'{language}: {key[:-1]} {item["id"]} lists '
                            f'{case_id}, no case of the corpus',
                        )
                if listed and listed != made[key][str(item['id'])]:
                    report.fail(
                        'rule 4',
                        f'{language}: {key[:-1]} {item["id"]}: its cases are '
                        'not the generated list (written by hand)',
                    )


def rule_5(
    manifests: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    report: Report,
) -> list[str]:
    r"""The active slice's paths, each with an acceptance case; the paths
    covered are returned for the summary."""
    language, active = ACTIVE
    manifest = manifests.get(language)
    if manifest is None:
        return []
    made = generated(language, manifest, corpus)
    gated = SLICES[: SLICES.index(active) + 1]
    covered = []
    for item in manifest['paths']:
        if item['slice'] not in gated:
            continue
        if made['paths'][str(item['id'])]:
            covered.append(str(item['id']))
        else:
            report.fail(
                'rule 5',
                f'{language}: {active} path {item["id"]} has no acceptance case',
            )
    if not covered and not report.problems:
        report.fail('rule 5', f'{language}: {active} has no path to gate')
    return covered


def rule_8(
    language: str,
    old: dict[str, Any],
    new: dict[str, Any],
    decisions: str,
    report: Report,
) -> None:
    rows = [
        line
        for line in decisions.splitlines()
        if ROW.match(line)
        and owner_approved(line.strip().strip('|').split('|')[-1])
    ]

    def approved(name: str) -> bool:
        return any(f'`{name}`' in row for row in rows)

    def drop(what: str, name: str, how: str) -> None:
        if not approved(name):
            report.fail(
                'rule 8',
                f'{language}: {what} {name} {how}, with no row of the '
                "checkpoint's decisions.md approved by the owner naming it",
            )

    now = {
        key: {str(item['id']): item for item in new[key]}
        for key in ('paths', 'cells')
    }
    for item in old['paths']:
        name = str(item['id'])
        if name not in now['paths']:
            drop('path', name, 'is removed')
        elif (
            now['paths'][name]['slice'] in SLICES
            and item['slice'] in SLICES
            and SLICES.index(now['paths'][name]['slice'])
            > SLICES.index(item['slice'])
        ):
            drop('path', name, 'moves to a later slice')
    for item in old['cells']:
        name = str(item['id'])
        if name not in now['cells']:
            drop('cell', name, 'is removed')
        elif (
            item['expect'] != 'unknown'
            and now['cells'][name]['expect'] == 'unknown'
        ):
            drop('cell', name, 'turns unknown')


def check(repo: Path, rev: str, checkpoint: str) -> tuple[Report, list[str]]:
    ratchet = _module('ratchet')
    report = Report()
    summary: list[str] = []
    try:
        candidate = ratchet.Tree(
            repo, ratchet.git(repo, 'rev-parse', f'{rev}^{{commit}}').strip()
        )
        before = ratchet.Tree(
            repo,
            ratchet.git(repo, 'rev-parse', f'{checkpoint}^{{commit}}').strip(),
        )
    except ratchet.Unreadable as error:
        report.fail('read', str(error))
        return report, summary
    manifests = {}
    validator = read_schema(candidate, report)
    for language in MANIFESTS if validator is not None else ():
        manifest = read_manifest(candidate, language, validator, report)
        if manifest is not None:
            manifests[language] = manifest
            rule_1(language, manifest, report)
            rule_2_3(language, manifest, report)
    corpus = read_cases(candidate, report)
    if corpus is not None and len(manifests) == len(MANIFESTS):
        rule_4(manifests, corpus, report)
        covered = rule_5(manifests, corpus, report)
        summary.append(
            f'{ACTIVE[0]} {ACTIVE[1]}: {len(covered)} paths, each with an '
            f'acceptance case; {len(corpus)} cases read'
        )
    for language, path in MANIFESTS.items():
        try:
            old_text = before.read(path)
        except ratchet.Unreadable as error:
            report.fail('read', f"the checkpoint's {error}")
            continue
        if old_text is None:
            # Every checkpoint holds both manifests (Phase 0 wrote them):
            # with none, rule 8 has nothing to hold the candidate to.
            report.fail('read', f"the checkpoint's {path}: missing")
            continue
        try:
            old = _yaml(old_text, f"the checkpoint's {path}")
        except Unreadable as error:
            report.fail('read', str(error))
            continue
        try:
            decisions = _text(before, DECISIONS)
        except Unreadable as error:
            # No row authorizes anything: rule 8 still runs.
            report.fail('read', f"the checkpoint's {error}")
            decisions = ''
        if language not in manifests:
            continue  # its read or schema failure is reported
        if not (
            isinstance(old, dict)
            and all(
                isinstance(old.get(key), list)
                and all(isinstance(x, dict) and 'id' in x for x in old[key])
                for key in ('paths', 'cells')
            )
        ):
            report.fail('read', f"the checkpoint's {path}: not a manifest")
            continue
        rule_8(language, old, manifests[language], decisions, report)
    summary.append(f'rule 8 against {before.sha[:12]}')
    return report, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--check', action='store_true', required=True)
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--rev', default='HEAD')
    parser.add_argument('--checkpoint', required=True)
    arguments = parser.parse_args(argv)
    repo = arguments.repo
    if repo is None:
        try:
            repo = Path(
                _module('ratchet')
                .git(Path('.'), 'rev-parse', '--show-toplevel')
                .strip()
            )
        except _module('ratchet').Unreadable as error:
            print(f'coverage: {error}', file=sys.stderr)
            return 2
    report, summary = check(repo, arguments.rev, arguments.checkpoint)
    for line in summary:
        print(f'coverage: {line}')
    for problem in report.problems:
        print(problem)
    print(f'coverage: {"FAIL" if report.problems else "ok"}')
    return 1 if report.problems else 0


if __name__ == '__main__':
    sys.exit(main())
