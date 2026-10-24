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
6. a contradicted cell claim fails (T1.32g; design, "Coverage generator"):
   each cell a case claims, acceptance or diagnostic alike, must be held by
   a column of the case's source in the case's ``python-ref`` observation
   (``contract/reference/*.jsonl``), at a stage the case reached. A claim
   ``<type>/<role>`` holds when some column the reference read is admitted
   as ``<type>`` (:func:`cell_type`: the Arrow type the driver returned, in
   the reference's sample of the table at ``from_source``, through the
   core's admission table, so ``timestamp[s]`` is ``timestamp-us``,
   scope-format.md: "a cell names the type the core receives") and plays
   ``<role>`` there (:func:`observed`): the key, time column, links and
   other columns of the schema the reference described, with the case's
   edits; the query's target from ``materialize`` on; and, in a refusal at
   ``from_source`` before any schema, only what the case's options declare
   and ``feature`` for a column of a table holding no declared constraint.
   A claim with no observation of the case, or an observation that does
   not show it (no capture, no sample, a shape it does not read), fails;
   it is never taken on trust;
8. against the manifests at ``--checkpoint``: no path or cell is removed, no
   path moves to a later slice and no cell's ``expect`` turns ``unknown``
   (each would take it out of rule 5), unless a row of the checkpoint's
   decisions, every phase's ``docs/design/rust-core/phase-N/decisions.md``
   (T2.0; a file missing, unreadable or malformed refused by name, and no
   row then authorizes anything), its status starting
   "approved by owner" and a date with nothing after but a parenthesis or
   a note, none withdrawing it (``ratchet.owner_approved``), names the ID in
   backticks; a row the candidate writes never authorizes its own change.

Rule 7 is Phase 2's (``--gate S1``); ``--gate S0`` was cut by the
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
#: The reference observations rule 6 holds the cells' claims to (T1.13).
REFERENCE = 'contract/reference'
#: Where every phase's ``decisions.md`` is (T2.0).
DECISIONS_DIR = 'docs/design/rust-core'
SLICES = ('S0', 'S1', 'S2', 'S3')
#: The language and slice whose coverage gates (design, "Coverage
#: generator"): S0 in Python.
ACTIVE = ('python', 'S0')
#: The deepest nesting a case or the schema may hold (they hold 8 and 7):
#: deeper is refused by name on every Python, whether its parser recurses
#: (a RecursionError through 3.13) or not (3.14 reads 100,000 levels).
MAX_DEPTH = 64
CASE_ID = re.compile(r'[a-z0-9][a-z0-9._-]*')
ROW = re.compile(r'\| [A-Z][A-Z0-9-]* \|')


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


def read_reference(tree: Any, report: Report) -> dict[str, Any] | None:
    r"""The reference observations by case, or None, a file or line unread
    reported (rule 6 would otherwise judge on less than is committed)."""
    found: dict[str, Any] = {}
    unread = False
    try:
        names = [n for n in tree.files(REFERENCE) if n.endswith('.jsonl')]
    except _module('ratchet').Unreadable as error:
        report.fail('read', str(error))
        return None
    for name in names:
        try:
            text = _text(tree, name)
        except Unreadable as error:
            report.fail('read', str(error))
            unread = True
            continue
        for number, line in enumerate(text.splitlines(), 1):
            where = f'{name}:{number}'
            try:
                observation = _json(line, where)
            except Unreadable as error:
                report.fail('read', str(error))
                unread = True
                continue
            if not isinstance(observation, dict) or not isinstance(
                observation.get('case'), str
            ):
                report.fail('read', f'{where}: not an observation of a case')
                unread = True
                continue
            if observation['case'] in found:
                report.fail(
                    'read', f'{where}: a second line of {observation["case"]}'
                )
                unread = True
                continue
            found[observation['case']] = observation
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


def on_leg(path: str, leg: str) -> str:
    r"""A case's DuckDB path as the same operation and mode on ``leg``'s
    backend (``run_python.py``'s ``on_database``); any other is itself."""
    head, _, backend = path.rpartition('/')
    return f'{head}/{leg}' if backend == 'duckdb' else path


def claims(case: dict[str, Any]) -> dict[str, set[str]]:
    r"""What a case covers (T3.2; carried row P2-T2.16-3): its own paths and
    cells, and each of its DuckDB paths on every other leg it runs on
    (``run_python.py``'s ``case_legs``: SQLite, Postgres and the warehouse
    legs), so a database or warehouse leg claims its own path."""
    legs = _module('register').runner().case_legs(case)
    # How the case's source is opened on each leg (the orchestrator's
    # decision, T3.2): with the leg's source class (DatabricksSource,
    # SnowflakeSource, PostgresSource, DuckDBSource, SQLiteSource),
    # source.open/explicit; a connect case's raw connection or session,
    # found by detection, source.open/auto.
    mode = 'auto' if 'connect' in (case.get('input') or {}) else 'explicit'
    opened = {f'source.open/{mode}/{leg}' for leg in legs}
    return {
        'paths': {on_leg(path, leg) for path in case['paths'] for leg in legs}
        | set(case['paths'])
        | opened,
        'cells': set(case['cells']),
    }


def generated(
    language: str, manifest: dict[str, Any], corpus: dict[str, dict[str, Any]]
) -> dict[str, dict[str, list[str]]]:
    r"""Each path's and cell's cases as the corpus says: the acceptance
    cases claiming it (:func:`claims`), by ID. Java runs the same case
    files as Python (Phase 3 design choice 10), so a Java path's cases are
    the cases claiming it; each Java leg is admitted where the Java host
    reports its provider, which the ratchet's no-removal rule then holds."""
    out: dict[str, dict[str, list[str]]] = {'paths': {}, 'cells': {}}
    claimed = {
        case_id: claims(case)
        for case_id, case in corpus.items()
        if case['status'] == 'acceptance'
    }
    for key in out:
        for name in _ids(manifest, key):
            out[key][name] = sorted(
                case_id
                for case_id, held in claimed.items()
                if name in held[key]
            )
    return out


#: A client test's ID, by the manifest's language: for Python a pytest node
#: ID, ``<file>::<test>``; for Java (T3.12) a JUnit test method,
#: ``java/src/test/java/<file>.java#<method>``. A manifest names only its
#: own language's tests: a Java test exercises no Python path, and none of
#: Python's a Java one.
_TESTS = {
    'python': re.compile(
        r'(?P<file>[A-Za-z0-9_./-]+\.py)::(?P<name>test_[A-Za-z0-9_]+)'
    ),
    'java': re.compile(
        r'(?P<file>java/src/test/java/[A-Za-z0-9_./-]+\.java)#(?P<name>[A-Za-z0-9_]+)'
    ),
}
#: How a file defines the test, for Python: a pytest function. A Java
#: test is the one JUnit ran, as Surefire's reports of the Java test run
#: record it (:func:`junit_ran`, :func:`junit_problem`), never read from the
#: source.
_DEFINED = {
    'python': r'^\s*def {name}\(',
}


#: Each language's form, as a problem names it.
_FORM = {
    'python': '<file>::<test>',
    'java': 'java/src/test/java/<file>.java#<method>',
}


def _test_problem(test: str, root: Any, language: str = 'python') -> str | None:
    r"""Why ``test`` names no test of ``language`` the tree defines, or None.
    ``root`` is the candidate's tree (its ``read``) or a directory."""
    found = _TESTS[language].fullmatch(test)
    if found is None:
        return f'not {_FORM[language]}'
    if root is None:
        root = HERE.parents[1]
    try:
        source = (
            (root / found['file']).read_text(encoding='utf-8')
            if isinstance(root, Path)
            else root.read(found['file'])
        )
    except Exception:  # noqa: BLE001 - an unreadable file defines no test
        source = None
    if language == 'java':
        # The file is the tree's and names the method (a cheap guard against
        # a misnamed test at landing); that JUnit ran it is the Java test
        # run's to say (junit_run).
        named = source and re.search(
            rf'\b{re.escape(found["name"])}\s*\(', source
        )
        return None if named else 'no test the tree defines'
    defined = _DEFINED[language].format(name=re.escape(found['name']))
    if not source or not re.search(defined, source, re.M):
        return 'no test the tree defines'
    return None


def junit_ran(reports: Path) -> set[tuple[str, str]]:
    r"""Each test JUnit ran and did not skip, as Surefire's reports in
    ``reports`` (``TEST-*.xml``) record it: its class and its method's name
    (a parameterized test's invocations, ``name(String)[1]``, by the name
    before its parameters). Only :func:`junit_run` reads them, from a
    directory its own Maven run filled."""
    import xml.etree.ElementTree as ElementTree

    ran: set[tuple[str, str]] = set()
    for path in sorted(reports.glob('TEST-*.xml')):
        for case in ElementTree.parse(path).getroot().iter('testcase'):
            if case.find('skipped') is not None:
                continue
            name = re.split(r'[(\[]', case.get('name', ''), maxsplit=1)[0]
            ran.add((case.get('classname', ''), name))
    return ran


def junit_problem(test: str, ran: set[tuple[str, str]]) -> str | None:
    r"""Why the Java test ``test`` (``java/src/test/java/<file>.java#<method>``)
    is no test JUnit ran, by :func:`junit_ran`, or None: its file's class,
    or a class nested in it, ran the method, and did not skip it."""
    found = _TESTS['java'].fullmatch(test)
    if found is None:
        return f'not {_FORM["java"]}'
    name = (
        found['file'].removeprefix('java/src/test/java/').removesuffix('.java')
    )
    owner = name.replace('/', '.')
    if any(
        method == found['name']
        and (cls == owner or cls.startswith(owner + '$'))
        for cls, method in ran
    ):
        return None
    return 'not a test JUnit ran (target/surefire-reports)'


def junit_run(repo: Path, command: list[str], report: Report) -> None:
    r"""The Java test run the check owns: the Surefire reports directory
    emptied (that directory alone; the rest of the build directory, the
    natives a CI job put there among it, stays), Maven's ``verify``
    (``command``) run in ``repo``, then each test the Java manifest names
    held to the reports that run left (:func:`junit_check`), and no other."""
    import shutil
    import subprocess

    reports = repo / 'java/target/surefire-reports'
    if reports.is_dir():
        shutil.rmtree(reports)
    ran = subprocess.run(command, cwd=repo, check=False)
    if ran.returncode != 0:
        report.fail(
            'rule 4', f'java: the Java test run failed ({ran.returncode})'
        )
        return
    junit_check(repo, reports, report)


def junit_check(repo: Path, reports: Path, report: Report) -> None:
    r"""Each test the Java manifest names, held to the Java test run whose
    Surefire reports :func:`junit_run` left in ``reports``: rule 4's Java
    half."""
    ran = junit_ran(reports)
    try:
        manifest = _yaml(
            (repo / MANIFESTS['java']).read_text(encoding='utf-8'),
            MANIFESTS['java'],
        )
    except (OSError, Unreadable) as error:
        report.fail('read', str(error))
        return
    for path, tests in sorted(tested(manifest).items()):
        for test in tests:
            problem = junit_problem(test, ran)
            if problem is not None:
                report.fail('rule 4', f'java: path {path}: {test}: {problem}')


def rule_4(
    manifests: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    report: Report,
    root: Any = None,
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
        for path, tests in sorted(tested(manifest).items()):
            for test in tests:
                problem = _test_problem(test, root, language)
                if problem is not None:
                    report.fail(
                        'rule 4', f'{language}: path {path}: {test}: {problem}'
                    )
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


def tested(manifest: dict[str, Any]) -> dict[str, list[str]]:
    r"""Each path a client test claims (its ``tests``, pytest node IDs),
    where the case format cannot express it (T3.2, P2-T2.16-3); rule 4
    holds each to a test the tree defines."""
    return {
        str(item['id']): [str(test) for test in item.get('tests') or []]
        for item in manifest.get('paths') or []
        if item.get('tests')
    }


def uncovered(
    manifests: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    language: str,
    active: str,
) -> list[str]:
    r"""The paths of ``language``'s manifest in ``active`` and the slices
    before it with no acceptance case claiming them on any leg
    (:func:`generated`) and no client test (:func:`tested`): the S1, S2 and
    Java rules' list (T3.2)."""
    manifest = manifests.get(language)
    if manifest is None:
        return []
    made = generated(language, manifest, corpus)
    # A claim only in the manifest's own language's form counts.
    form = _TESTS.get(language)
    claimed = {
        path: [t for t in tests if form is not None and form.fullmatch(t)]
        for path, tests in tested(manifest).items()
    }
    gated = SLICES[: SLICES.index(active) + 1]
    return [
        str(item['id'])
        for item in manifest['paths']
        if item['slice'] in gated
        and not made['paths'][str(item['id'])]
        and not claimed.get(str(item['id']))
    ]


def rule_5(
    manifests: dict[str, dict[str, Any]],
    corpus: dict[str, dict[str, Any]],
    report: Report,
    gate: tuple[str, str] = ACTIVE,
) -> list[str]:
    r"""The gated slice's paths, each with an acceptance case or a named
    client test claiming it (:func:`uncovered`), and from S1 on every cell
    of an expected outcome with an acceptance case (the S1, S2 and Java
    rules, T3.2); the paths covered are returned for the summary."""
    language, active = gate
    manifest = manifests.get(language)
    if manifest is None:
        return []
    made = generated(language, manifest, corpus)
    empty = set(uncovered(manifests, corpus, language, active))
    gated = SLICES[: SLICES.index(active) + 1]
    covered = []
    for item in manifest['paths']:
        if item['slice'] not in gated:
            continue
        name = str(item['id'])
        if name not in empty:
            covered.append(name)
        else:
            report.fail(
                'rule 5',
                f'{language}: {item["slice"]} path {name} has no acceptance '
                'case',
            )
    if active != 'S0':
        for item in manifest['cells']:
            if item.get('expect') == 'unknown':
                continue
            if not made['cells'][str(item['id'])]:
                report.fail(
                    'rule 5',
                    f'{language}: cell {item["id"]} has no acceptance case '
                    f'({active})',
                )
    if not covered and not report.problems:
        report.fail('rule 5', f'{language}: {active} has no path to gate')
    return covered


# --- rule 6: the cells' claims, held to the reference ---------------------------

#: The Arrow types the reference's driver returns (pyarrow's text of each)
#: that the core's admission table (``sdm-host``'s ``admission.rs``, decisions
#: B-D6 and H16) admits unparameterized, by the cell type admitted.
_PLAIN = {
    **{
        name: name
        for name in (
            'int8',
            'int16',
            'int32',
            'int64',
            'uint8',
            'uint16',
            'uint32',
            'uint64',
            'bool',
            'null',
        )
    },
    'halffloat': 'float32',
    'float': 'float32',
    'double': 'float64',
    'string': 'utf8',
    'large_string': 'utf8',
    'string_view': 'utf8',
    'binary': 'binary',
    'large_binary': 'binary',
    'binary_view': 'binary',
    'date32[day]': 'date32',
    'date64[ms]': 'timestamp-us',
}
#: A parameterized type's name, by the cell type admitted: the admission
#: table widens decimal32 and decimal64 and takes every list form as a list.
_HEADS = {
    'decimal32': 'decimal128',
    'decimal64': 'decimal128',
    'decimal128': 'decimal128',
    'decimal256': 'decimal256',
    'fixed_size_binary': 'binary',
    'list': 'list',
    'large_list': 'list',
    'fixed_size_list': 'list',
    'list_view': 'list',
    'large_list_view': 'list',
    'struct': 'struct',
    'map': 'map',
}
_HEAD = re.compile(r'([a-z_0-9]+)[<(\[]')
_UNIT = re.compile(
    r'(timestamp|duration|time32|time64)\[(s|ms|us|ns)(, tz=.+)?\]'
)
#: The reference's sample of a table at ``from_source``, and its read of the
#: table's declared constraints (contract/reference's statements).
_SAMPLE = re.compile(r'SELECT .+ FROM "((?:[^"]|"")+)" LIMIT [0-9]+', re.S)
_CONSTRAINTS = re.compile(
    r"FROM duckdb_constraints\(\)\s+WHERE table_name = '((?:[^']|'')+)'"
)
#: A PQL query's first column reference before its ``FOR``: its target,
#: with or without an ``EVALUATE`` or ``EXPLAIN`` before ``PREDICT``, for
#: each entity (``FOR EACH t.c``) or for the ones it names (``FOR t.c = 1``,
#: ``FOR t.c IN (...)``; T2.16a).
_TARGET = re.compile(
    r'\s*(?:(?:EVALUATE|EXPLAIN)\s+)?PREDICT\s+(.*?)\s+FOR\s+'
    r'(?:EACH\s|[A-Za-z_][A-Za-z0-9_]*\.)',
    re.S | re.IGNORECASE,
)
_COLUMN = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\.(\*|[A-Za-z_][A-Za-z0-9_]*)')
ROLES = ('primary-key', 'foreign-key', 'time-column', 'feature', 'target')


class Unevidenced(Exception):
    r"""What an observation does not show, named."""


def _inner(text: str, start: int) -> str:
    r"""``text`` from ``start`` to the first comma or unmatched closing
    bracket outside nested brackets: one type within a parameter list."""
    depth = 0
    for at in range(start, len(text)):
        char = text[at]
        if char in '<([':
            depth += 1
        elif char in '>)]':
            if depth == 0:
                return text[start:at]
            depth -= 1
        elif char == ',' and depth == 0:
            return text[start:at]
    return text[start:]


def cell_type(arrow: str) -> str | None:
    r"""The cell type the core admits a column the driver returned as
    ``arrow`` (pyarrow's text of the type) as, or None for a type outside
    the set: the admission table, read at the cell's grain (a list is a
    list whatever its items; L5 checks the host's conversions). A
    dictionary or run-end type is unwrapped to its values, at most
    :data:`MAX_DEPTH` levels; deeper is :class:`Unevidenced`, by name."""
    dictionary = False
    for _ in range(MAX_DEPTH + 1):
        head = _HEAD.match(arrow)
        if head is not None and arrow.startswith('dictionary<values='):
            # Only the innermost dictionary decides: a dictionary of text.
            dictionary = True
            arrow = _inner(arrow, len('dictionary<values='))
        elif (
            head is not None
            and head.group(1) == 'run_end_encoded'
            and ', values: ' in arrow
        ):
            dictionary = False
            arrow = _inner(arrow, arrow.index(', values: ') + len(', values: '))
        else:
            break
    else:
        raise Unevidenced('a type it does not read')
    cell = _plain_type(arrow)
    return 'dictionary-utf8' if dictionary and cell == 'utf8' else cell


def _plain_type(arrow: str) -> str | None:
    r"""The cell type of a type that wraps no other's values."""
    if arrow in _PLAIN:
        return _PLAIN[arrow]
    unit = _UNIT.fullmatch(arrow)
    if unit is not None:
        kind, scale, zone = unit.groups()
        if kind in ('time32', 'time64'):
            return None if zone else 'time64'
        if kind == 'duration':
            return (
                None if zone else f'duration-{"ns" if scale == "ns" else "us"}'
            )
        return f'timestamp-{"ns" if scale == "ns" else "us"}' + (
            '-tz' if zone else ''
        )
    head = _HEAD.match(arrow)
    return None if head is None else _HEADS.get(head.group(1))


def _strs(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(x, str) for x in value)


def _statements(line: dict[str, Any]) -> list[dict[str, Any]]:
    r"""The statements the reference ran at ``from_source``, as captured."""
    driver = line.get('driver')
    capture = driver.get('capture') if isinstance(driver, dict) else None
    statements = (
        capture.get('statements') if isinstance(capture, dict) else None
    )
    if not isinstance(statements, list):
        raise Unevidenced('no driver capture')
    return [
        each
        for each in statements
        if isinstance(each, dict)
        and each.get('stage') == 'from_source'
        and isinstance(each.get('sql'), str)
        and isinstance(each.get('results'), list)
    ]


def received(line: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    r"""Each column the reference sampled at ``from_source``, by source
    table and name, with the cell types the core admits it as: the Arrow
    schema of the driver's answer to the reference's sample of the table."""
    found: dict[tuple[str, str], set[str]] = {}
    for statement in _statements(line):
        sample = _SAMPLE.fullmatch(statement['sql'].strip())
        if sample is None:
            continue
        table = sample.group(1).replace('""', '"')
        for result in statement['results']:
            schema = result.get('schema') if isinstance(result, dict) else None
            if not isinstance(schema, list):
                continue
            for field in schema:
                if not (
                    isinstance(field, dict)
                    and isinstance(field.get('name'), str)
                    and isinstance(field.get('type'), str)
                ):
                    raise Unevidenced('a sample it does not read')
                cell = cell_type(field['type'])
                found.setdefault((table, field['name']), set()).update(
                    [cell] if cell is not None else []
                )
    if not found:
        raise Unevidenced('no sample of a table at from_source')
    return found


def _undeclared(line: dict[str, Any]) -> set[str]:
    r"""The source tables the reference found no declared constraint on:
    its read of their constraints answered no row."""
    rows: dict[str, list[Any]] = {}
    for statement in _statements(line):
        read = _CONSTRAINTS.search(statement['sql'])
        if read is None or 'FOREIGN KEY' in statement['sql']:
            continue
        rows.setdefault(read.group(1).replace("''", "'"), []).extend(
            r.get('rows') if isinstance(r, dict) else None
            for r in statement['results']
        )
    return {table for table, counts in rows.items() if counts == [0]}


def _keys(value: Any) -> set[str]:
    if isinstance(value, str):
        return {value}
    if _strs(value):
        return set(value)
    if value is None:
        return set()
    raise Unevidenced('a key it does not read')


def _described(schema: Any) -> dict[str, Any]:
    r"""The schema the reference described, as tables (each its columns,
    key and time column) and links (source table and column first)."""
    if not isinstance(schema, dict) or not isinstance(
        schema.get('tables'), list
    ):
        raise Unevidenced('a schema it does not read')
    tables: dict[str, dict[str, Any]] = {}
    for table in schema['tables']:
        columns = table.get('columns') if isinstance(table, dict) else None
        if not (
            isinstance(table, dict)
            and isinstance(table.get('name'), str)
            and isinstance(columns, list)
            and all(
                isinstance(c, dict) and isinstance(c.get('name'), str)
                for c in columns
            )
            and isinstance(table.get('time_column'), (str, type(None)))
        ):
            raise Unevidenced('a schema it does not read')
        tables[table['name']] = {
            'columns': [c['name'] for c in columns],
            'key': _keys(table.get('primary_key')),
            'time': table['time_column'],
        }
    links = schema.get('links')
    if not isinstance(links, list) or not all(
        _strs(link) and len(link) == 4 for link in links
    ):
        raise Unevidenced('a schema it does not read')
    return {'tables': tables, 'links': [tuple(link) for link in links]}


def _edited(described: dict[str, Any], edits: Any) -> dict[str, Any]:
    r"""The schema with the case's edits made, as ``run_python.py`` makes
    them once ``from_source`` returns (an edit it does not know: refused)."""
    tables, links = described['tables'], list(described['links'])
    for edit in edits if isinstance(edits, list) else [None]:
        op = edit.get('op') if isinstance(edit, dict) else None
        name = edit.get('table') if isinstance(edit, dict) else None
        table = tables.get(name) if isinstance(name, str) else None
        if op == 'set_relationships' and isinstance(
            edit.get('relationships'), list
        ):
            links = [
                tuple(
                    link.get(key)
                    for key in (
                        'source_table',
                        'source_column',
                        'target_table',
                        'target_column',
                    )
                )
                for link in edit['relationships']
                if isinstance(link, dict)
            ]
        elif table is None:
            raise Unevidenced('an edit it does not read')
        elif op == 'clear_time_column':
            table['time'] = None
        elif op == 'delete_column':
            table['columns'] = [
                c for c in table['columns'] if c != edit.get('column')
            ]
        elif op == 'set_primary_key':
            table['key'] = _keys(edit.get('column'))
        elif op == 'set_time_column':
            table['time'] = edit.get('column')
        elif op != 'set_stype':
            raise Unevidenced('an edit it does not read')
    return {'tables': tables, 'links': links}


def _target(case: dict[str, Any]) -> tuple[str, str] | None:
    r"""The column the case's query predicts, or None (a count, ``*``)."""
    given = case.get('input')
    pql = given.get('pql') if isinstance(given, dict) else None
    head = _TARGET.match(pql) if isinstance(pql, str) else None
    column = _COLUMN.search(head.group(1)) if head is not None else None
    if column is None or column.group(2) == '*':
        return None
    return column.group(1), column.group(2)


def observed(case: dict[str, Any], line: dict[str, Any]) -> set[str]:
    r"""The cells the case's reference observation shows: each column the
    reference read, at the cell type the core admits it as, in each role
    it played at a stage the case reached; :class:`Unevidenced` where the
    observation does not show what that needs."""
    outcome = line.get('outcome')
    if not isinstance(outcome, dict) or outcome.get('result') not in (
        'success',
        'refusal',
    ):
        raise Unevidenced('no outcome it reads')
    stage = outcome.get('stage')
    if stage not in ('from_source', 'materialize', 'predict'):
        raise Unevidenced('no stage it reads')
    given = case['input'] if isinstance(case.get('input'), dict) else {}
    options = given.get('from_source') or {}
    named = options.get('tables') if isinstance(options, dict) else None
    if _strs(named):
        # A list of tables names each as itself (Schema.from_source); an
        # empty one names no table, as the client reads it (G12: it chooses
        # none and refuses, never every table).
        named = {name: name for name in named}
    if named is not None and not isinstance(named, dict):
        raise Unevidenced('table options it does not read')
    types = received(line)
    declared = _declared(options if isinstance(options, dict) else {})
    to_name = {v: k for k, v in (named or {}).items()}
    free = _undeclared(line)
    public = line.get('public')
    schema = public.get('schema') if isinstance(public, dict) else None
    roles: dict[tuple[str, str], set[str]] = {}
    held: set[tuple[str, str]] = set()
    if schema is not None:
        described = _described(schema)
        held = {
            (name, column)
            for name, table in described['tables'].items()
            for column in table['columns']
        }
        if not (outcome['result'] == 'refusal' and stage == 'from_source'):
            # The edits were made: from_source returned with them.
            described = _edited(described, given.get('edits', []))
        linked = {(link[0], link[1]) for link in described['links']}
        for name, table in described['tables'].items():
            for column in table['columns']:
                mine = set()
                if column in table['key']:
                    mine.add('primary-key')
                if column == table['time']:
                    mine.add('time-column')
                if (name, column) in linked:
                    mine.add('foreign-key')
                roles[(name, column)] = mine or {'feature'}
    elif not (outcome['result'] == 'refusal' and stage == 'from_source'):
        raise Unevidenced('no schema past from_source')
    # Each column read at from_source that no described schema holds: the
    # stage refused or left it out in the role the input gave it, what the
    # options declare, or a feature where no declared constraint of its
    # table could make it anything else (a refusal covers its rejected
    # input, and claims no later stage).
    for source, column in types:
        name = to_name.get(source, None if named is not None else source)
        if name is None or (name, column) in held:
            continue
        mine = declared.get((name, column), set())
        if not mine and source in free:
            mine = {'feature'}
        if mine:
            roles.setdefault((name, column), set()).update(mine)
    target = _target(case)
    if target is not None and stage in ('materialize', 'predict'):
        roles.setdefault(target, set()).add('target')
    cells = set()
    for (name, column), played in roles.items():
        source = (named or {}).get(name, None if named is not None else name)
        for cell in types.get((source, column), set()) if source else set():
            cells.update(f'{cell}/{role}' for role in played)
    return cells


def _declared(options: dict[str, Any]) -> dict[tuple[str, str], set[str]]:
    r"""The roles the case's ``from_source`` options declare."""
    found: dict[tuple[str, str], set[str]] = {}
    keys = options.get('primary_keys') or {}
    times = options.get('time_columns') or {}
    links = options.get('relationships') or []
    if not (
        isinstance(keys, dict)
        and isinstance(times, dict)
        and isinstance(links, list)
    ):
        raise Unevidenced('options it does not read')
    for table, key in keys.items():
        for column in _keys(key):
            found.setdefault((table, column), set()).add('primary-key')
    for table, column in times.items():
        if isinstance(column, str):
            found.setdefault((table, column), set()).add('time-column')
    for link in links:
        if isinstance(link, dict):
            found.setdefault(
                (link.get('source_table'), link.get('source_column')), set()
            ).add('foreign-key')
    return found


def rule_6(
    corpus: dict[str, dict[str, Any]],
    reference: dict[str, Any],
    report: Report,
) -> None:
    r"""Each case's cell claims, acceptance and diagnostic alike, held to
    its reference observation; a claim with none to hold it to fails."""
    for case_id, case in sorted(corpus.items()):
        claims = case['cells']
        if not claims:
            continue
        line = reference.get(case_id)
        if line is None:
            report.fail(
                'rule 6',
                f'case {case_id}: claims {", ".join(claims)} with no '
                'reference observation of the case to hold them to',
            )
            continue
        try:
            shown = observed(case, line)
        except Unevidenced as error:
            report.fail(
                'rule 6',
                f'case {case_id}: claims {", ".join(claims)}, and its '
                f'reference observation shows none of them: {error}',
            )
            continue
        for claim in claims:
            if claim not in shown:
                report.fail(
                    'rule 6',
                    f'case {case_id}: claims {claim}, and no column of its '
                    'source plays that role at that type at a stage its '
                    'reference observation reached',
                )


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
        and _module('ratchet').owner_approved(
            line.strip().strip('|').split('|')[-1]
        )
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


def check(
    repo: Path,
    rev: str,
    checkpoint: str,
    gates: tuple[tuple[str, str], ...] = (ACTIVE,),
) -> tuple[Report, list[str]]:
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
    reference = read_reference(candidate, report)
    if corpus is not None and reference is not None:
        rule_6(corpus, reference, report)
    if corpus is not None and len(manifests) == len(MANIFESTS):
        rule_4(manifests, corpus, report, candidate)
        for gate in gates:
            covered = rule_5(manifests, corpus, report, gate)
            summary.append(
                f'{gate[0]} {gate[1]}: {len(covered)} paths, each with an '
                f'acceptance case or a named test; {len(corpus)} cases read'
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
        ratchet_check = ratchet.capture_check()
        try:
            decisions = ratchet_check.phase_decisions(
                before.files(DECISIONS_DIR), before.read
            )
        except ratchet_check.DecisionsError as error:
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
    # --check, or the report list's --gate SLICE alone (T3.2): the same
    # check, gating the slices named.
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--rev', default='HEAD')
    parser.add_argument('--checkpoint')
    parser.add_argument(
        '--junit-run',
        action='store_true',
        help="runs the Java tests (Maven's verify) and holds each Java test "
        'the manifest names to that run alone (run, not skipped)',
    )
    parser.add_argument('--maven', default='mvn', help='the Maven to run')
    parser.add_argument(
        '--maven-arg',
        action='append',
        default=[],
        help="an argument to Maven's run",
    )
    parser.add_argument(
        '--gate',
        action='append',
        metavar='LANGUAGE:SLICE',
        help='a slice whose paths and cells gate, such as python:S1 or '
        'java:S1 (T3.2); default python:S0, the active slice',
    )
    arguments = parser.parse_args(argv)
    if arguments.junit_run:
        junit = Report()
        root = arguments.repo or Path(__file__).resolve().parents[2]
        junit_run(
            root,
            [
                arguments.maven,
                '-B',
                *arguments.maven_arg,
                '-f',
                'java/pom.xml',
                'verify',
            ],
            junit,
        )
        for problem in junit.problems:
            print(problem)
        print(f'coverage: {"FAIL" if junit.problems else "ok"} (JUnit run)')
        return 1 if junit.problems else 0
    if arguments.checkpoint is None:
        parser.error('--checkpoint is required')
    if not arguments.check and not arguments.gate:
        parser.error('give --check, --gate LANGUAGE:SLICE, or both')
    gates = []
    for given in arguments.gate or [':'.join(ACTIVE)]:
        # A bare slice is Python's (the report list's --gate S2).
        language, _, active = (
            given.partition(':') if ':' in given else ('python', ':', given)
        )
        if language not in MANIFESTS or active not in SLICES:
            parser.error(
                f'--gate {given}: a language of {sorted(MANIFESTS)} and a slice of {SLICES}'
            )
        gates.append((language, active))
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
    report, summary = check(
        repo, arguments.rev, arguments.checkpoint, tuple(gates)
    )
    for line in summary:
        print(f'coverage: {line}')
    for problem in report.problems:
        print(problem)
    print(f'coverage: {"FAIL" if report.problems else "ok"}')
    return 1 if report.problems else 0


if __name__ == '__main__':
    sys.exit(main())
