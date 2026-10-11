#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The ratchet: a candidate judged against its checkpoint (design,
"Reference observations and the gate" and "Register v2"; decisions H2, H3,
H19 and A6). ``gate.sh`` runs the checkpoint's copy of this file.

    ratchet.py checkpoint [--repo DIR] (--base REV | --phase-push BEFORE)
    ratchet.py check [--repo DIR] --checkpoint SHA [--protected-paths TEXT]
        [--observations FILE]...
    ratchet.py accept [--repo DIR] --observations FILE...

``checkpoint`` resolves the checkpoint of the candidate, ``--repo``'s
``HEAD``: ``git merge-base CANDIDATE BASE``, never read from evidence.
``--base`` is the phase-branch tip at package start or at run time; when
the candidate is a landing commit (``Land <ID>: ...``, made by ``land.py``),
its recorded ``Base-SHA`` is the base instead, so a landing is judged
against the head it landed on even once the phase branch holds it.
``--phase-push`` is a push to the phase branch, ``BEFORE`` its previous
tip: the push must land exactly one package, the candidate being that
landing commit with ``Base-SHA`` ``BEFORE``, as a fast-forward. It refuses
a candidate that is its own checkpoint (self-comparison: the base already
holds the candidate) and a checkpoint that is not a landing commit, so
advancing the phase branch to an unreviewed commit cannot make that commit
a checkpoint. It prints the checkpoint and whether its harness judges: a
checkpoint without this file is the bootstrap, before T1.14's landing, where
the candidate's harness runs and no acceptance claim is made.

``check`` holds the candidate, ``--repo``'s committed ``HEAD``, to the
checkpoint's files, read from git at ``--checkpoint``:

- ``protected``: each changed path ``contract/protected-paths.txt`` (the
  checkpoint's) protects is named in the landing commit's ``Reviewed-by`` or
  ``Protected-paths`` trailer, or before landing in ``--protected-paths``
  (the text the orchestrator gives ``land.py``); with neither, the paths are
  listed for the landing to name;
- ``reference``: reference immutability: each recorded reference line is
  still there, its semantic projection unchanged (:func:`semantic`); a line
  may be added; a changed or removed line passes only under an approved
  register entry naming its case on ``python-ref``, or with its case's
  approved removal; a new reference commit is the owner's;
- ``cases``: a case removed, demoted from acceptance to diagnostic, or with
  its file changed passes only under an approved entry naming the case (for
  a removal or demotion, each baselined runner's exact old outcome and
  ``removed`` or ``diagnostic``); ``bases``: a base changed or removed passes
  only under an approved entry naming a case built on it, and one added
  passes;
- ``register``: ``register.py validate``, with the checkpoint's register for
  never-reused IDs; entries added or changed are listed, ``approved_by:
  owner`` ones first, for the owner's check-ins;
- ``policy`` and ``availability``: a relaxation of ``compare-policy.yaml`` or
  ``availability.yaml`` passes only when the candidate's file lists it under
  ``relaxations`` with a row of the checkpoint's decisions.md approved by the
  owner (a row the package writes itself never authorizes);
- ``known-differing``: the candidate's ``KNOWN_DIFFERING`` file a subset of
  the checkpoint's;
- with observations, ``baselines``: each checkpoint baseline
  (``contract/baselines/<cell>/<runner>.jsonl``) of a given runner equals the
  observation's projection (``compare-policy.yaml``'s ``baseline`` fields, the
  checkpoint's), or an approved entry states that case, cell, runner and
  exact old and new outcome, an owner-approved one for a new refusal;
  ``INTERNAL_ERROR`` never passes; the candidate's proposed baselines equal
  the observations and hold every acceptance case; and ``availability``: each
  observation has every field its runner has available and none it has not.

A baseline is the checkpoint's, so a branch-side edit of one changes
nothing the candidate is judged by; the identity envelope is outside every
projection, so an identity-only change passes.

``accept`` writes the proposed baselines from the observations of every
runner given: each case whose projection equals the committed reference's
on every one of them, the first baselines (T1.14) included; it is the
package's to commit and the reviewer's to review (H3).

Exit status 1 on any problem. Each problem names its rule and its case;
input values are never printed.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
POLICY = 'contract/compare-policy.yaml'
AVAILABILITY = 'contract/availability.yaml'
PROTECTED = 'contract/protected-paths.txt'
BASELINES = 'contract/baselines'
REFERENCE = 'contract/reference'
CASES = 'contract/cases'
BASES = 'contract/bases.yaml'
REGISTER = 'docs/design/rust-core/register.yaml'
DECISIONS = 'docs/design/rust-core/phase-1/decisions.md'
KNOWN_DIFFERING = 'rust/crates/sdm-core/tests/pql/known-differing.json'
#: The cell the baselines were taken in: the reference machine's (design,
#: "Runtime cells"). Their projection is platform-independent, so every cell
#: is held to them.
CELL = 'ref-pandas3'
RUNNERS = ('python-ref', 'python-core', 'python-old', 'rust-host')
PYTHON = ('python-ref', 'python-core', 'python-old')
#: The core operations that read data, whose case has a driver record on
#: rust-host.
READS = ('describe_source', 'materialize', 'predict')
#: A landing commit's subject, as land.py writes it.
LANDING = re.compile(r'Land (T[0-9]+\.[0-9]+[a-z]?|CI[0-9]+): ')
#: An owner's approval, as a decisions row's status reads it.
OWNER = re.compile(r'approved by owner \d{4}-\d{2}-\d{2}')
PANIC = 'INTERNAL_ERROR'


class Unreadable(Exception):
    r"""A file the ratchet cannot read, named."""


def _module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@functools.cache
def register_module() -> ModuleType:
    r"""This harness's ``register.py``."""
    return _module('sdm_register', HERE / 'register.py')


@functools.cache
def capture_check() -> ModuleType:
    r"""This harness's ``capture_check.py``: the E35 mask and the decisions
    rows."""
    return _module('capture_check', REPO / 'contract/runners/capture_check.py')


# --- git ---------------------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ['git', '-C', str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if done.returncode != 0:
        raise Unreadable(f'git {args[0]}: {done.stderr.strip()}')
    return done.stdout


class Tree:
    r"""A commit's files, read from git through one ``git cat-file --batch``
    process."""

    def __init__(self, repo: Path, sha: str) -> None:
        self.repo, self.sha = repo, sha
        self._cache: dict[str, str | None] = {}
        self._batch: subprocess.Popen[bytes] | None = None

    def read(self, path: str) -> str | None:
        if path in self._cache:
            return self._cache[path]
        if self._batch is None:
            self._batch = subprocess.Popen(
                ['git', '-C', str(self.repo), 'cat-file', '--batch'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
            )
        assert self._batch.stdin is not None and self._batch.stdout is not None
        self._batch.stdin.write(f'{self.sha}:{path}\n'.encode())
        self._batch.stdin.flush()
        header = self._batch.stdout.readline().split()
        text: str | None = None
        if len(header) == 3 and header[1] == b'blob':
            data = self._batch.stdout.read(int(header[2]))
            self._batch.stdout.read(1)
            try:
                text = data.decode('utf-8')
            except UnicodeDecodeError:
                raise Unreadable(f'{path}: not UTF-8') from None
        elif len(header) == 3:
            # Not a file (a tree): its contents are skipped.
            self._batch.stdout.read(int(header[2]) + 1)
        self._cache[path] = text
        return text

    def files(self, prefix: str) -> list[str]:
        out = git(self.repo, 'ls-tree', '-r', '--name-only', self.sha, prefix)
        return sorted(line for line in out.splitlines() if line)

    def __del__(self) -> None:
        if self._batch is not None:
            self._batch.kill()
            self._batch.wait()


def trailers(repo: Path, sha: str) -> dict[str, str]:
    r"""A commit's trailers, by key; a repeated key's values joined."""
    found: dict[str, str] = {}
    out = git(repo, 'log', '-1', '--format=%(trailers:unfold)', sha)
    for line in out.splitlines():
        key, sep, value = line.partition(':')
        if sep:
            key = key.strip()
            found[key] = (found.get(key, '') + ' ' + value.strip()).strip()
    return found


def subject(repo: Path, sha: str) -> str:
    return git(repo, 'log', '-1', '--format=%s', sha).strip()


def is_landing(repo: Path, sha: str) -> bool:
    r"""``sha`` is a landing commit as land.py makes them: its subject, and
    the base and checkpoint it records."""
    found = trailers(repo, sha)
    return bool(
        LANDING.match(subject(repo, sha))
        and found.get('Base-SHA')
        and found.get('Checkpoint-SHA')
    )


# --- the checkpoint ------------------------------------------------------------


@dataclass
class Checkpoint:
    sha: str
    base: str
    candidate: str
    bootstrap: bool
    notes: list[str] = field(default_factory=list)


def resolve(
    repo: Path, base: str | None = None, phase_push: str | None = None
) -> tuple[Checkpoint | None, list[str]]:
    r"""The candidate's checkpoint, or the problems that leave it none."""
    candidate = git(repo, 'rev-parse', 'HEAD^{commit}').strip()
    notes: list[str] = []
    if phase_push is not None:
        before = git(repo, 'rev-parse', f'{phase_push}^{{commit}}').strip()
        landings = [
            sha
            for sha in git(repo, 'rev-list', f'{before}..{candidate}').split()
            if LANDING.match(subject(repo, sha))
        ]
        problems = []
        if (
            git(repo, 'merge-base', before, candidate).strip() != before
            or before == candidate
        ):
            problems.append(
                'the push is not a fast-forward of the previous tip'
            )
        if landings != [candidate]:
            problems.append(
                'a push to the phase branch lands exactly one package, as '
                f'its tip: it holds {len(landings)} landing commits'
                + ('' if candidate in landings else ', and its tip is none')
            )
        elif trailers(repo, candidate).get('Base-SHA') != before:
            problems.append(
                "the landing's Base-SHA is not the phase branch's previous tip"
            )
        if problems:
            return None, problems
        base = before
    elif is_landing(repo, candidate):
        recorded = trailers(repo, candidate)['Base-SHA']
        notes.append(
            f'the candidate is a landing commit: its recorded Base-SHA '
            f'{recorded[:12]} is the base'
        )
        base = recorded
    if base is None:
        return None, ['no base: give --base or --phase-push']
    base = git(repo, 'rev-parse', f'{base}^{{commit}}').strip()
    found = git(repo, 'merge-base', candidate, base).strip()
    problems = []
    if found == candidate:
        problems.append(
            'self-comparison: the base already holds the candidate, so it '
            'would be its own checkpoint; judge it against the base recorded '
            'before the phase branch advanced'
        )
    if not is_landing(repo, found):
        problems.append(
            f'the checkpoint {found[:12]} is not a landing commit, so not a '
            'reviewed one: the phase branch holds an unlanded commit'
        )
    if problems:
        return None, problems
    if found != base:
        notes.append(
            f'the phase branch moved past the checkpoint to {base[:12]}: '
            'rebase onto it and gate again before landing'
        )
    bootstrap = Tree(repo, found).read('scripts/rust_core/ratchet.py') is None
    return Checkpoint(found, base, candidate, bootstrap, notes), []


# --- reading -------------------------------------------------------------------


def _yaml(text: str | None, name: str) -> Any:
    if text is None:
        return None
    try:
        return register_module().read_yaml(text, name)
    except register_module().Unreadable as error:
        raise Unreadable(str(error)) from None


def _json(text: str, name: str) -> Any:
    try:
        return json.loads(text)
    except (ValueError, RecursionError):
        raise Unreadable(f'{name}: not JSON it reads') from None


def jsonl(text: str | None, name: str) -> list[dict[str, Any]]:
    r"""The objects of a JSON-lines text, each line read or refused by
    name."""
    found = []
    for number, line in enumerate((text or '').splitlines(), 1):
        item = _json(line, f'{name}:{number}')
        if not isinstance(item, dict):
            raise Unreadable(f'{name}:{number}: not an object')
        found.append(item)
    return found


def cases(tree: Tree) -> dict[str, tuple[dict[str, Any], str]]:
    r"""A tree's cases by ID, each with its file's text."""
    found = {}
    for path in tree.files(CASES):
        if not path.endswith('.json'):
            continue
        text = tree.read(path) or ''
        case = _json(text, path)
        if isinstance(case, dict) and isinstance(case.get('id'), str):
            found[case['id']] = (case, text)
    return found


def baselines(tree: Tree) -> dict[str, dict[str, dict[str, Any]]]:
    r"""A tree's baselines in :data:`CELL`, by runner and case, each line's
    digest checked against its projection."""
    found: dict[str, dict[str, dict[str, Any]]] = {}
    for runner in RUNNERS:
        path = f'{BASELINES}/{CELL}/{runner}.jsonl'
        lines = jsonl(tree.read(path), path)
        by_case = {}
        for line in lines:
            if (
                set(line) != {'case', 'runner', 'cell', 'projection', 'digest'}
                or not isinstance(line['case'], str)
                or line['runner'] != runner
                or line['cell'] != CELL
                or not isinstance(line['projection'], dict)
            ):
                raise Unreadable(f'{path}: a line is not a baseline')
            if line['digest'] != digest(line['projection']):
                raise Unreadable(f'{path}: {line["case"]}: digest differs')
            if line['case'] in by_case:
                raise Unreadable(f'{path}: {line["case"]} twice')
            by_case[line['case']] = line
        if by_case:
            found[runner] = by_case
    return found


def observations(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    r"""Observations by runner and case, from their files."""
    found: dict[str, dict[str, Any]] = {}
    for path in paths:
        lines = jsonl(path.read_text(encoding='utf-8'), path.name)
        if not lines:
            raise Unreadable(f'{path.name}: no observations')
        for line in lines:
            identity = line.get('identity')
            runner = (
                identity.get('runner') if isinstance(identity, dict) else None
            )
            if runner not in RUNNERS or not isinstance(line.get('case'), str):
                raise Unreadable(f'{path.name}: an observation of no runner')
            by_case = found.setdefault(runner, {})
            if line['case'] in by_case:
                raise Unreadable(f'{path.name}: {line["case"]} twice')
            by_case[line['case']] = line
    return found


# --- projections ---------------------------------------------------------------


def at(document: Any, dotted: str) -> Any:
    r"""The value at a dotted path, or None where there is none."""
    for part in dotted.split('.'):
        if not isinstance(document, dict):
            return None
        document = document.get(part)
    return document


def projection(
    observation: dict[str, Any], fields: list[str]
) -> dict[str, Any]:
    return {name: at(observation, name) for name in fields}


def digest(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def outcome_text(projected: dict[str, Any] | None) -> str:
    r"""A projected outcome as an entry states it (the schema's form)."""
    if projected is None:
        return 'none'
    result = projected.get('outcome.result')
    text = (
        'success'
        if result == 'success'
        else f'refusal {projected.get("outcome.code") or "-"}'
    )
    if projected.get('outcome.stage'):
        text += f' at {projected["outcome.stage"]}'
    if projected.get('outcome.error'):
        text += f' ({projected["outcome.error"]})'
    return text


def semantic(
    observation: dict[str, Any], ordered: dict[str, Any] | None = None
) -> dict[str, Any]:
    r"""A reference line's semantic projection: the line without its identity
    (checked apart, ``tests/test_contract_reference.py``) and without what
    ``compare.py`` never gates: each statement's text and typed parameters
    and the relations bound (H9's diagnostics), each raw result's and
    payload's SHA-256 (integrity only), the value of ``metadata.request_id``
    (E35), and a statement's in-order digest where its rows compare as a
    multiset (``ordered`` names the others)."""
    line = json.loads(json.dumps(observation))
    line.pop('identity', None)
    driver = line.get('driver') or {}
    capture = driver.get('capture') or {}
    capture.pop('bindings', None)
    places = (ordered or {}).get(line.get('case'), {})
    counters: dict[str, int] = {}
    for statement in capture.get('statements') or []:
        stage = statement.get('stage')
        position = counters.get(stage, 0)
        counters[stage] = position + 1
        statement.pop('sql', None)
        statement.pop('parameters', None)
        for read in statement.get('results') or []:
            read.pop('digest', None)
            if position not in places.get(stage, []):
                (read.get('values') or {}).pop('ordered', None)
    public = line.get('public') or {}
    for payload in public.get('payloads') or []:
        payload.pop('sha256', None)
        payload['parsed'] = capture_check().masked(payload.get('parsed'))
    return line


# --- entries -------------------------------------------------------------------


def approved(register: Any) -> list[dict[str, Any]]:
    r"""The register's approved entries; an unreadable register has none."""
    if not isinstance(register, dict) or not isinstance(
        register.get('entries'), list
    ):
        return []
    return [
        entry
        for entry in register['entries']
        if isinstance(entry, dict) and entry.get('status') == 'approved'
    ]


def authorized(
    entries: list[dict[str, Any]],
    case: str,
    runner: str | None,
    old: str | None,
    new: str | None,
    owner: bool = False,
) -> bool:
    r"""An approved entry states this case's change, by the owner where
    ``owner``; a None part matches any."""
    for entry in entries:
        if owner and entry.get('approved_by') != 'owner':
            continue
        for item in entry.get('case_outcomes') or []:
            if (
                isinstance(item, dict)
                and item.get('case') == case
                and item.get('cell') == CELL
                and runner in (None, item.get('runner'))
                and old in (None, item.get('old'))
                and new in (None, item.get('new'))
                and PANIC not in str(item.get('new'))
            ):
                return True
    return False


def proposal(case: str, runner: str, old: str, new: str) -> str:
    r"""The ``case_outcomes`` item an entry would need, as generated text."""
    return (
        f'{{case: {case}, cell: {CELL}, runner: {runner}, '
        f"old: '{old}', new: '{new}', summary: ...}}"
    )


# --- the rules -----------------------------------------------------------------


@dataclass
class Report:
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def fail(self, rule: str, text: str) -> None:
        self.problems.append(f'{rule}: {text}')

    def note(self, rule: str, text: str) -> None:
        self.notes.append(f'{rule}: {text}')


def patterns(text: str | None) -> list[tuple[bool, re.Pattern[str]]]:
    r"""The protected-path patterns, each with whether it takes paths out."""
    found = []
    for raw in (text or '').splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        negate = line.startswith('!')
        pattern = line[1:] if negate else line
        regex = ''
        index = 0
        while index < len(pattern):
            if pattern.startswith('**', index):
                regex += '.*'
                index += 2
            elif pattern[index] == '*':
                regex += '[^/]*'
                index += 1
            else:
                regex += re.escape(pattern[index])
                index += 1
        found.append((negate, re.compile(regex)))
    return found


def protects(rules: list[tuple[bool, re.Pattern[str]]], path: str) -> bool:
    hit = False
    for negate, regex in rules:
        if regex.fullmatch(path):
            hit = not negate
    return hit


def named(path: str, text: str) -> bool:
    r"""``text`` names ``path``: the path itself, or a directory holding it
    written with a trailing ``/`` or ``/**``."""
    for token in re.findall(r'[A-Za-z0-9_.*/-]+', text):
        token = token.rstrip('.')
        if token == path:
            return True
        for suffix in ('/**', '/'):
            if token.endswith(suffix) and path.startswith(
                token[: -len(suffix)] + '/'
            ):
                return True
    return False


def check_protected(
    report: Report,
    checkpoint: Tree,
    candidate: Tree,
    changed: list[str],
    trailer: str | None,
    bootstrap: bool,
) -> None:
    text = checkpoint.read(PROTECTED)
    if text is None and bootstrap:
        text = candidate.read(PROTECTED)
        report.note('protected', "bootstrap: the candidate's list is read")
    rules = patterns(text)
    if not rules:
        report.fail('protected', f'no {PROTECTED} at the checkpoint')
        return
    touched = [path for path in changed if protects(rules, path)]
    if trailer is None:
        for path in touched:
            report.note(
                'protected', f'changed, for the landing to name: {path}'
            )
        return
    for path in touched:
        if not named(path, trailer):
            report.fail(
                'protected',
                f'{path}: changed, and the Reviewed-by and Protected-paths '
                'trailers do not name it',
            )


def check_reference(
    report: Report,
    checkpoint: Tree,
    candidate: Tree,
    entries: list[dict[str, Any]],
    removed: set[str],
) -> None:
    before = checkpoint.files(REFERENCE)
    if not before:
        return
    after = candidate.files(REFERENCE)
    if [p for p in after if p not in before] and not [
        p for p in before if p in after
    ]:
        report.fail(
            'reference',
            "a new reference commit replaces the recorded one: the owner's",
        )
        return
    policy = _yaml(checkpoint.read(POLICY), POLICY) or {}
    ordered = policy.get('ordered') if isinstance(policy, dict) else None
    for path in before:
        old = {
            line.get('case'): line
            for line in jsonl(checkpoint.read(path), path)
        }
        new = {
            line.get('case'): line for line in jsonl(candidate.read(path), path)
        }
        for case, line in old.items():
            if new.get(case) == line:
                continue
            if case not in new:
                if case not in removed:
                    report.fail('reference', f'{case}: its line removed')
            elif semantic(line, ordered) != semantic(new[case], ordered):
                if not authorized(entries, case, 'python-ref', None, None):
                    report.fail(
                        'reference',
                        f'{case}: its line changed, and no approved entry '
                        'names the case on python-ref',
                    )
            else:
                report.note('reference', f'{case}: re-recorded, identity only')


def check_cases(
    report: Report,
    checkpoint: Tree,
    candidate: Tree,
    entries: list[dict[str, Any]],
    held: dict[str, dict[str, dict[str, Any]]],
) -> tuple[set[str], set[str]]:
    r"""Removed, demoted and changed cases, and changed bases; returns the
    cases removed or demoted, and those whose removal is approved."""
    before, after = cases(checkpoint), cases(candidate)
    gone, removed = set(), set()
    for case, (old, text) in sorted(before.items()):
        if case in after:
            mine, changed = after[case][0], after[case][1] != text
            if not (
                old.get('status') == 'acceptance'
                and mine.get('status') != 'acceptance'
            ):
                if changed and not authorized(entries, case, None, None, None):
                    report.fail(
                        'cases',
                        f'{case}: its input changed, and no approved entry '
                        'names it',
                    )
                continue
            new = 'diagnostic'
        else:
            new = 'removed'
        gone.add(case)
        lines = [
            (runner, outcome_text(by_case[case]['projection']))
            for runner, by_case in held.items()
            if case in by_case
        ] or [(None, 'none')]
        missing = [
            (runner, old_text)
            for runner, old_text in lines
            if not authorized(entries, case, runner, old_text, new)
        ]
        for runner, old_text in missing:
            report.fail(
                'cases',
                f'{case}: {new}, and no approved entry states it on '
                f'{runner or "any runner"}: '
                + proposal(case, runner or '<runner>', old_text, new),
            )
        if not missing and new == 'removed':
            removed.add(case)
    old_bases = _yaml(checkpoint.read(BASES), BASES)
    new_bases = _yaml(candidate.read(BASES), BASES)
    if isinstance(old_bases, dict) and isinstance(old_bases.get('bases'), dict):
        mine = new_bases.get('bases') if isinstance(new_bases, dict) else None
        mine = mine if isinstance(mine, dict) else {}
        for name, statements in old_bases['bases'].items():
            if mine.get(name) == statements:
                continue
            users = [
                case
                for case, (each, _) in before.items()
                if (each.get('input') or {}).get('base') == name
            ]
            if not any(
                authorized(entries, case, None, None, None) for case in users
            ):
                report.fail(
                    'cases',
                    f'base {name}: changed or removed, and no approved entry '
                    'names a case built on it',
                )
    return gone, removed


def policy_relaxations(old: Any, new: Any) -> list[str]:
    r"""What the candidate's ``compare-policy.yaml`` relaxes."""
    old = old if isinstance(old, dict) else {}
    new = new if isinstance(new, dict) else {}
    found = []
    mine = new.get('rules') or {}
    for rule, cls in (old.get('rules') or {}).items():
        if rule not in mine:
            found.append(f'rule {rule} removed')
        elif cls == 'gate' and mine[rule] != 'gate':
            found.append(f'rule {rule}: gate to {mine[rule]}')
    theirs = {
        e.get('id'): e
        for e in old.get('exclusions') or []
        if isinstance(e, dict)
    }
    for item in new.get('exclusions') or []:
        key = item.get('id') if isinstance(item, dict) else None
        if key not in theirs:
            found.append(f'exclusion {key} added')
        elif theirs[key] != item:
            found.append(f'exclusion {key} changed')
    kept = new.get('ordered') or {}
    for case, stages in (old.get('ordered') or {}).items():
        for stage, positions in (stages or {}).items():
            for position in positions or []:
                if position not in ((kept.get(case) or {}).get(stage) or []):
                    found.append(f'ordered {case} {stage} {position} removed')
    reach = new.get('reach') or {}
    for case, stage in (old.get('reach') or {}).items():
        if case not in reach:
            found.append(f'reach {case} removed')
        elif reach[case] != stage:
            found.append(f'reach {case}: {stage} to {reach[case]}')
    fields = new.get('baseline') or []
    for name in old.get('baseline') or []:
        if name not in fields:
            found.append(f'baseline field {name} removed')
    return found


def availability_relaxations(old: Any, new: Any) -> list[str]:
    r"""What the candidate's ``availability.yaml`` relaxes."""
    theirs = (old or {}).get('runners') or {} if isinstance(old, dict) else {}
    mine = (new or {}).get('runners') or {} if isinstance(new, dict) else {}
    found = []
    for runner, fields in theirs.items():
        for name, value in (fields or {}).items():
            now = (mine.get(runner) or {}).get(name)
            if value == 'available' and now != 'available':
                found.append(f'{runner} {name}: available to {now or "absent"}')
    for runner, fields in mine.items():
        for name, value in (fields or {}).items():
            if value != 'available' and name not in (theirs.get(runner) or {}):
                found.append(f'{runner} {name}: added as {value}')
    return found


def relaxations(read: Any, old: Any, new: Any, path: str) -> list[str]:
    r"""``read``'s relaxations of ``path``, a file of another shape refused
    by name."""
    try:
        found: list[str] = read(old, new)
    except (AttributeError, TypeError):
        raise Unreadable(f'{path}: not a file of its format') from None
    return found


def check_relaxations(
    report: Report,
    rule: str,
    path: str,
    found: list[str],
    allowed: Any,
    decisions: str | None,
) -> None:
    rows = capture_check().decision_rows(decisions or '')
    listed = allowed if isinstance(allowed, list) else []
    name = Path(path).name
    for change in found:
        decision = next(
            (
                item.get('decision')
                for item in listed
                if isinstance(item, dict) and item.get('change') == change
            ),
            None,
        )
        row = rows.get(decision) if isinstance(decision, str) else None
        if row is None or not OWNER.search(row[1]) or f'`{name}`' not in row[0]:
            report.fail(
                rule,
                f"{change}: a relaxation, with no row of the checkpoint's "
                f'decisions.md approved by the owner and naming `{name}`',
            )
        else:
            report.note(rule, f'{change}: approved by {decision}')


def check_known_differing(
    report: Report, checkpoint: Tree, candidate: Tree
) -> None:
    def listed(tree: Tree) -> dict[str, set[str]]:
        text = tree.read(KNOWN_DIFFERING)
        if text is None:
            return {}
        document = _json(text, KNOWN_DIFFERING)
        found: dict[str, set[str]] = {}
        for item in (document or {}).get('cases') or []:
            if isinstance(item, dict) and isinstance(item.get('id'), str):
                found[item['id']] = set(map(str, item.get('parts') or []))
        return found

    before, after = listed(checkpoint), listed(candidate)
    for case, parts in sorted(after.items()):
        if case not in before:
            report.fail('known-differing', f'{case}: added')
        elif not parts <= before[case]:
            report.fail('known-differing', f'{case}: parts added')
    for case in sorted(set(before) - set(after)):
        report.note('known-differing', f'{case}: removed')


def check_register(
    report: Report, checkpoint: Tree, candidate: Tree, root: Path
) -> Any:
    module = register_module()
    try:
        mine = _yaml(candidate.read(REGISTER), REGISTER)
        theirs = _yaml(checkpoint.read(REGISTER), REGISTER)
        for problem in module.problems(mine, root, theirs):
            report.fail('register', problem)
    except (Unreadable, module.runner().RunnerFailure) as error:
        report.fail('register', str(error))
        return None
    before = {
        e.get('id'): e
        for e in (theirs or {}).get('entries') or []
        if isinstance(e, dict)
    }
    for entry in sorted(
        (mine or {}).get('entries') or [],
        key=lambda e: (e.get('approved_by') != 'owner', e.get('id')),
    ):
        if before.get(entry.get('id')) != entry:
            what = 'added' if entry.get('id') not in before else 'changed'
            report.note(
                'register',
                f'{entry.get("id")} {what}: {entry.get("status")}, approved_by '
                f'{entry.get("approved_by")}',
            )
    return mine


def check_availability(
    report: Report,
    fields: Any,
    observed: dict[str, dict[str, Any]],
    corpus: dict[str, tuple[dict[str, Any], str]],
) -> None:
    runners = (
        (fields or {}).get('runners') if isinstance(fields, dict) else None
    )
    if not isinstance(runners, dict):
        report.fail('availability', f'no {AVAILABILITY}')
        return
    for runner, by_case in observed.items():
        marked = runners.get(runner)
        if not isinstance(marked, dict):
            report.fail('availability', f'{runner}: not in {AVAILABILITY}')
            continue
        for case, observation in sorted(by_case.items()):
            spec = corpus.get(case, ({}, ''))[0]
            for name, value in marked.items():
                found = at(observation, name)
                if value == 'unavailable':
                    if found is not None:
                        report.fail(
                            'availability',
                            f'{case}: {runner}: {name} recorded, and it is '
                            'unavailable',
                        )
                    continue
                if (
                    name == 'outcome.error'
                    and at(observation, 'outcome.result') != 'refusal'
                ):
                    continue
                if (
                    name == 'driver.capture'
                    and runner == 'rust-host'
                    and not any(op in (spec.get('core') or {}) for op in READS)
                ):
                    continue
                if found is None:
                    report.fail(
                        'availability',
                        f'{case}: {runner}: {name} available, and missing',
                    )


def check_baselines(
    report: Report,
    held: dict[str, dict[str, dict[str, Any]]],
    proposed: dict[str, dict[str, dict[str, Any]]],
    old_fields: list[str],
    new_fields: list[str],
    observed: dict[str, dict[str, Any]],
    corpus: dict[str, tuple[dict[str, Any], str]],
    entries: list[dict[str, Any]],
    gone: set[str],
) -> None:
    for runner, by_case in sorted(observed.items()):
        for case, line in sorted(held.get(runner, {}).items()):
            if case in gone:
                continue
            observation = by_case.get(case)
            if observation is None:
                report.fail(
                    'baselines',
                    f'{case}: {runner}: baselined, and not observed',
                )
                continue
            now = projection(observation, old_fields)
            if now.get('outcome.code') == PANIC:
                report.fail(
                    'baselines',
                    f'{case}: {runner}: {PANIC}, a panic, which no entry accepts',
                )
                continue
            if now == line['projection']:
                continue
            old, new = outcome_text(line['projection']), outcome_text(now)
            refusal = (
                now.get('outcome.result') == 'refusal'
                and line['projection'].get('outcome.result') == 'success'
            )
            if not authorized(entries, case, runner, old, new, owner=refusal):
                report.fail(
                    'baselines',
                    f'{case}: {runner}: {old} became {new}, with no approved '
                    + ('owner ' if refusal else '')
                    + 'entry stating it: '
                    + proposal(case, runner, old, new),
                )
            else:
                report.note(
                    'baselines', f'{case}: {runner}: {old} to {new}, approved'
                )
        mine = proposed.get(runner, {})
        for case, line in sorted(mine.items()):
            observation = by_case.get(case)
            if (
                observation is None
                or projection(observation, new_fields) != line['projection']
            ):
                report.fail(
                    'baselines',
                    f'{case}: {runner}: the proposed baseline is not the '
                    'observation (ratchet.py accept writes it)',
                )
        for case, (spec, _) in sorted(corpus.items()):
            if (
                spec.get('status') == 'acceptance'
                and case in by_case
                and case not in mine
            ):
                report.fail(
                    'baselines',
                    f'{case}: {runner}: an acceptance case with no proposed baseline',
                )


def check(
    repo: Path,
    checkpoint_sha: str,
    observation_files: list[Path],
    protected_text: str | None = None,
    bootstrap: bool = False,
) -> Report:
    report = Report()
    checkpoint = Tree(repo, checkpoint_sha)
    head = git(repo, 'rev-parse', 'HEAD^{commit}').strip()
    candidate = Tree(repo, head)
    if git(repo, 'status', '--porcelain', '--untracked-files=no').strip():
        report.fail(
            'tree',
            'the worktree has uncommitted changes: the gate judges a commit',
        )
        return report
    trailer = protected_text
    if is_landing(repo, head):
        found = trailers(repo, head)
        if not found.get('Reviewed-by'):
            report.fail(
                'protected', 'the landing commit has no Reviewed-by trailer'
            )
        trailer = ' '.join(
            found.get(key, '') for key in ('Reviewed-by', 'Protected-paths')
        )
    changed = git(repo, 'diff', '--name-only', checkpoint_sha, head).split('\n')
    changed = [path for path in changed if path]
    try:
        mine = check_register(report, checkpoint, candidate, repo)
        entries = approved(mine)
        held = baselines(checkpoint)
        gone, removed = check_cases(
            report, checkpoint, candidate, entries, held
        )
        check_protected(
            report, checkpoint, candidate, changed, trailer, bootstrap
        )
        check_reference(report, checkpoint, candidate, entries, removed)
        decisions = checkpoint.read(DECISIONS)
        old_policy = _yaml(checkpoint.read(POLICY), POLICY)
        new_policy = _yaml(candidate.read(POLICY), POLICY)
        check_relaxations(
            report,
            'policy',
            POLICY,
            relaxations(policy_relaxations, old_policy, new_policy, POLICY),
            (new_policy or {}).get('relaxations')
            if isinstance(new_policy, dict)
            else None,
            decisions,
        )
        old_marks = _yaml(checkpoint.read(AVAILABILITY), AVAILABILITY)
        new_marks = _yaml(candidate.read(AVAILABILITY), AVAILABILITY)
        if old_marks is None and bootstrap:
            report.note('availability', 'bootstrap: the file is introduced')
        else:
            check_relaxations(
                report,
                'availability',
                AVAILABILITY,
                relaxations(
                    availability_relaxations, old_marks, new_marks, AVAILABILITY
                ),
                (new_marks or {}).get('relaxations')
                if isinstance(new_marks, dict)
                else None,
                decisions,
            )
        check_known_differing(report, checkpoint, candidate)
        if observation_files:
            observed = observations(observation_files)
            corpus = cases(candidate)
            check_availability(
                report,
                old_marks if old_marks is not None else new_marks,
                observed,
                corpus,
            )

            def fields(policy: Any) -> list[str]:
                found = (
                    policy.get('baseline') if isinstance(policy, dict) else None
                )
                return (
                    [str(f) for f in found] if isinstance(found, list) else []
                )

            for runner in sorted(observed):
                if runner not in held and not bootstrap:
                    report.fail(
                        'baselines',
                        f'the checkpoint holds no baselines of {runner}',
                    )
            check_baselines(
                report,
                held,
                baselines(candidate),
                fields(old_policy) or fields(new_policy),
                fields(new_policy),
                observed,
                corpus,
                entries,
                gone,
            )
    except Unreadable as error:
        report.fail('read', str(error))
    except (AttributeError, TypeError, KeyError) as error:
        # A file of another shape than its format's, whichever rule met it.
        report.fail(
            'read',
            f'a file not of its format ({type(error).__name__}: {error})',
        )
    return report


# --- accept --------------------------------------------------------------------


def accept(
    repo: Path, observation_files: list[Path]
) -> dict[str, list[dict[str, Any]]]:
    r"""The proposed baselines: per runner given, a line for each case whose
    projection equals the committed reference's on every runner given,
    fields a runner cannot observe left out of that comparison."""
    tree = Tree(repo, git(repo, 'rev-parse', 'HEAD^{commit}').strip())
    policy = _yaml(tree.read(POLICY), POLICY)
    fields = [str(f) for f in policy['baseline']]
    marks = (_yaml(tree.read(AVAILABILITY), AVAILABILITY) or {})['runners']
    reference = {}
    for path in tree.files(REFERENCE):
        reference.update(
            {line['case']: line for line in jsonl(tree.read(path), path)}
        )
    observed = observations(observation_files)
    found: dict[str, list[dict[str, Any]]] = {runner: [] for runner in observed}
    for case in sorted(cases(tree)):
        ref = reference.get(case)
        if ref is None or any(
            case not in by_case for by_case in observed.values()
        ):
            continue
        equal = all(
            at(by_case[case], name) == at(ref, name)
            for runner, by_case in observed.items()
            for name in fields
            if (marks.get(runner) or {}).get(name) != 'unavailable'
        )
        if not equal:
            continue
        for runner, by_case in observed.items():
            projected = projection(by_case[case], fields)
            found[runner].append(
                {
                    'case': case,
                    'runner': runner,
                    'cell': CELL,
                    'projection': projected,
                    'digest': digest(projected),
                }
            )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    point = commands.add_parser('checkpoint')
    point.add_argument('--repo', type=Path, default=Path('.'))
    where = point.add_mutually_exclusive_group(required=True)
    where.add_argument('--base')
    where.add_argument('--phase-push')
    judge = commands.add_parser('check')
    judge.add_argument('--repo', type=Path, default=Path('.'))
    judge.add_argument('--checkpoint', required=True)
    judge.add_argument('--protected-paths')
    judge.add_argument('--bootstrap', action='store_true')
    judge.add_argument('--observations', type=Path, action='append', default=[])
    take = commands.add_parser('accept')
    take.add_argument('--repo', type=Path, default=Path('.'))
    take.add_argument(
        '--observations', type=Path, action='append', required=True
    )
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == 'checkpoint':
            found, problems = resolve(
                arguments.repo, arguments.base, arguments.phase_push
            )
            for problem in problems:
                print(f'checkpoint: {problem}', file=sys.stderr)
            if found is None:
                return 1
            for note in found.notes:
                print(f'checkpoint: {note}', file=sys.stderr)
            print(
                found.sha,
                found.base,
                'bootstrap' if found.bootstrap else 'harness',
            )
            return 0
        if arguments.command == 'accept':
            for runner, lines in accept(
                arguments.repo, arguments.observations
            ).items():
                path = arguments.repo / BASELINES / CELL / f'{runner}.jsonl'
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    ''.join(
                        json.dumps(line, sort_keys=True) + '\n'
                        for line in lines
                    ),
                    encoding='utf-8',
                )
                print(f'accept: {len(lines)} baselines of {runner} in {path}')
            return 0
        report = check(
            arguments.repo,
            arguments.checkpoint,
            arguments.observations,
            arguments.protected_paths,
            arguments.bootstrap,
        )
    except Unreadable as error:
        print(f'ratchet: {error}\nratchet: FAIL')
        return 1
    for note in report.notes:
        print(note)
    for problem in report.problems:
        print(f'FAILS {problem}')
    print(f'ratchet: {"FAIL" if report.problems else "ok"}')
    return 1 if report.problems else 0


if __name__ == '__main__':
    sys.exit(main())
