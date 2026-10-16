#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Checks the quirk register, ``docs/design/rust-core/register.yaml``
(design, "Register v2"; decision A6):

    scripts/rust_core/register.py validate [--root DIR] [--checkpoint FILE]

It holds the file, as ``--root``'s tree has it, to:

- the schema, ``register.schema.json`` (``sdm-register/2``) beside this
  harness's register, which carries the status rules a schema can state
  (``approved_by`` null while proposed, the reviewer or the owner once
  approved or rejected);
- unique IDs, and with ``--checkpoint`` (the checkpoint's register, of
  either format) never reused: every ID the checkpoint has is still there,
  so an entry is never deleted, and a withdrawn one is ``rejected``;
- ``cases`` that resolve: a case of the corpus (``contract/cases``), or the
  legacy form of an entry written before it, a task ID (``T1.7c``) or a
  Phase 1b inventory case ID; ``case_outcomes`` name corpus cases only, but
  for a case they record as ``removed``;
- each ``decision_ref`` a row of the decisions file it names;
- an approved entry whose ``case_outcomes`` turn a success into a refusal,
  or refuse a case the committed reference accepts, approved by the owner;
  and no entry accepting ``INTERNAL_ERROR``, the code a contained panic
  answers with (``sdm-host``'s ``runs.rs``): the observations carry no
  panic flag, so every ``INTERNAL_ERROR`` is taken for one.

Whether an approval is the reviewer's or the owner's own is review's, not
this script's: there is no signing machinery (design, "Approval"). A file it
cannot read (not UTF-8, not YAML, nested past the parser) is refused by
name. Exit status 1 on any problem.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REGISTER = Path('docs/design/rust-core/register.yaml')
SCHEMA = REPO / 'docs/design/rust-core/register.schema.json'
CASES = Path('contract/cases')
REFERENCE = Path('contract/reference')
INVENTORY = Path('docs/design/rust-core/phase-0/phase-1b-inventory.yaml')
#: A task ID, the legacy form of a case: ``T0.6``, ``T1.7c``, ``T1.11f``.
TASK = re.compile(r'T[0-9]+\.[0-9]+[a-z]?')
#: A projected outcome as ratchet.py writes it (the schema's pattern).
OUTCOME = re.compile(
    r'(?P<result>success|refusal) ?(?P<code>[A-Z][A-Z0-9_]*|-)?'
    r'(?: at (?P<stage>[a-z_]+))?(?: \((?P<error>[A-Za-z_][A-Za-z0-9_]*)\))?'
)
#: The code a contained panic answers with; no entry accepts it.
PANIC = 'INTERNAL_ERROR'


class Unreadable(Exception):
    r"""A file this harness cannot read, named."""


def read_yaml(text: str, name: str) -> Any:
    r"""A YAML document, or :class:`Unreadable` naming why not: not YAML,
    or nested past what the parser reads."""
    import yaml

    try:
        return yaml.safe_load(text)
    except RecursionError:
        raise Unreadable(f'{name}: nested past what the parser reads') from None
    except yaml.YAMLError as error:
        mark = getattr(error, 'problem_mark', None)
        where = f' (line {mark.line + 1})' if mark is not None else ''
        raise Unreadable(f'{name}: not YAML{where}') from None


@functools.cache
def runner() -> ModuleType:
    r"""This harness's ``run_python.py``, whose case loader and schema
    validation this shares."""
    path = REPO / 'contract/runners/run_python.py'
    spec = importlib.util.spec_from_file_location('run_python', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def case_ids(root: Path) -> dict[str, dict[str, Any]]:
    r"""The corpus under ``root``, by ID, read by the runner's loader."""
    return {case['id']: case for case, _ in runner().load_cases([root / CASES])}


def inventory_ids(root: Path) -> set[str]:
    r"""The Phase 1b inventory's case IDs."""
    path = root / INVENTORY
    document = read_yaml(path.read_text(encoding='utf-8'), INVENTORY.name)
    if not isinstance(document, dict) or not isinstance(
        document.get('cases'), list
    ):
        raise Unreadable(f'{INVENTORY.name}: no cases')
    return {
        case['id']
        for case in document['cases']
        if isinstance(case, dict) and isinstance(case.get('id'), str)
    }


def reference_outcomes(root: Path) -> dict[str, str]:
    r"""Each case's outcome in the committed reference (``success`` or
    ``refusal``), by case."""
    found: dict[str, str] = {}
    for path in sorted((root / REFERENCE).glob('*.jsonl')):
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                observation = json.loads(line)
                found[observation['case']] = observation['outcome']['result']
            except (ValueError, RecursionError, KeyError, TypeError):
                # The reference's own test refuses such a line by name.
                continue
    return found


def decision_rows(path: Path) -> set[str]:
    r"""The row IDs of a decisions file: a number in Phase 0, a letter and
    a number from Phase 1."""
    return set(
        re.findall(
            r'^\| *([A-Z0-9][A-Z0-9-]*) *\|',
            path.read_text(encoding='utf-8'),
            re.M,
        )
    )


def parse_outcome(text: str) -> dict[str, str | None] | None:
    r"""An outcome's parts, or None for ``none``, ``removed`` and
    ``diagnostic``."""
    found = OUTCOME.fullmatch(text)
    if found is None:
        return None
    return found.groupdict()


def load(path: Path) -> Any:
    r"""A register file, read."""
    try:
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        raise Unreadable(f'{path.name}: not UTF-8') from None
    return read_yaml(text, path.name)


def _schema_problems(register: Any) -> list[str]:
    import jsonschema

    schema = json.loads(SCHEMA.read_text(encoding='utf-8'))
    cls = jsonschema.validators.validator_for(schema)
    cls.check_schema(schema)
    problems = []
    try:
        for error in cls(schema).iter_errors(register):
            where = '/'.join(str(part) for part in error.absolute_path)
            problems.append(
                f'schema: {where or "(root)"} fails {error.validator!r}'
            )
    except RecursionError:
        problems.append('schema: nested past what the validator reads')
    return sorted(problems)


def problems(
    register: Any,
    root: Path,
    checkpoint: Any = None,
) -> list[str]:
    r"""Every problem with ``register``, the register of ``root``'s tree;
    ``checkpoint`` is the checkpoint's register, if there is one."""
    found = _schema_problems(register)
    if found:
        return found
    entries: list[dict[str, Any]] = register['entries']
    ids = [entry['id'] for entry in entries]
    found += [
        f'{entry_id}: the ID repeats'
        for entry_id in sorted({i for i in ids if ids.count(i) > 1})
    ]
    if isinstance(checkpoint, dict) and isinstance(
        checkpoint.get('entries'), list
    ):
        before = {
            entry.get('id')
            for entry in checkpoint['entries']
            if isinstance(entry, dict)
        }
        found += [
            f'{entry_id}: removed, and an ID is never removed or reused'
            for entry_id in sorted(str(i) for i in before - set(ids))
        ]
    corpus = case_ids(root)
    inventory = inventory_ids(root)
    accepted = reference_outcomes(root)
    for entry in entries:
        where = entry['id']
        for case in entry['cases']:
            if not (
                case in corpus or TASK.fullmatch(case) or case in inventory
            ):
                found.append(
                    f'{where}: case {case}: not a corpus case, a task ID '
                    'or an inventory case'
                )
        ref = entry['decision_ref']
        if ref is not None:
            path = root / ref['file']
            if not path.is_file():
                found.append(f'{where}: decision_ref: no file {ref["file"]}')
            elif str(ref['row']) not in decision_rows(path):
                found.append(
                    f'{where}: decision_ref: {ref["row"]} is not a row of '
                    f'{ref["file"]}'
                )
        for outcome in entry.get('case_outcomes', []):
            found += _outcome_problems(entry, outcome, corpus, accepted)
    return found


def _outcome_problems(
    entry: dict[str, Any],
    outcome: dict[str, Any],
    corpus: dict[str, Any],
    accepted: dict[str, str],
) -> list[str]:
    where = f'{entry["id"]}: {outcome["case"]} on {outcome["runner"]}'
    found = []
    # The item names a line's key (``X``, ``X@sqlite``, ``X@postgres``):
    # the part before ``@`` is the corpus case.
    case = outcome['case'].partition('@')[0]
    if case not in corpus and outcome['new'] != 'removed':
        found.append(f'{where}: not a corpus case')
    old, new = parse_outcome(outcome['old']), parse_outcome(outcome['new'])
    if new is not None and new['code'] == PANIC:
        found.append(f'{where}: no entry accepts {PANIC}, a panic')
    refuses = new is not None and new['result'] == 'refusal'
    newly = refuses and (
        (old is not None and old['result'] == 'success')
        or accepted.get(outcome['case']) == 'success'
    )
    if (
        newly
        and entry['status'] == 'approved'
        and entry['approved_by'] != 'owner'
    ):
        found.append(
            f'{where}: refuses an input the reference or the checkpoint '
            'accepted, so only the owner approves it'
        )
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate')
    validate.add_argument('--root', type=Path, default=REPO)
    validate.add_argument('--checkpoint', type=Path)
    arguments = parser.parse_args(argv)
    try:
        register = load(arguments.root / REGISTER)
        checkpoint = (
            load(arguments.checkpoint) if arguments.checkpoint else None
        )
        found = problems(register, arguments.root, checkpoint)
    except (Unreadable, runner().RunnerFailure) as error:
        found = [str(error)]
    for problem in found:
        print(f'register: {problem}')
    print(f'register: {"FAIL" if found else "ok"}')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
