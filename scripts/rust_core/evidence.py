#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""Writes items 2 to 4 of the Phase 1b go or no-go evidence pack (design,
"Phase 1b go or no-go evidence pack"; T1.35) from a gate run's own outputs.

    scripts/rust_core/evidence.py --gate-out DIR --gate-json FILE
        --reference FILE --cases DIR --inventory FILE --register FILE
        [--heldout DIR] --out DIR

``--gate-out`` holds what ``gate.sh --locked 1b --out DIR`` kept, with the
local runs of the other runtime cells beside it: each ``*.jsonl`` file is
observations, each line's runner and cell read from its ``identity``, never
from the file's name, python-core required in each of ``run_python.py``'s
cells, python-old and rust-host too; ``compare.log``, the ``compare`` step's
output, raw or as gate.sh prints it (``    | `` before each line, the
step's block taken where gate.sh names the step), with exactly one verdict
whose count is the ``DIFFERS`` lines read; and ``ratchet.log``, the ratchet
step's. Every observation must be of the gate's run: its commit the gate
JSON's ``sha`` from a clean checkout, its case file's SHA-256 the one in
``--cases`` (or ``--heldout``); and a gating difference must name its case.
``--gate-json`` is the run's ``evidence/gate/<sha>.json``, which stays
outside the tree (design.md:432): the pack names it by its file name and
SHA-256 and never copies it, and one inside ``--out`` is refused.
``--reference`` is the committed python-ref observations, ``--cases`` the
case files, ``--inventory`` the Phase 1b inventory, ``--register`` the
register, and ``--heldout`` the held-out cases, of which only digests and
counts reach the pack. ``--out`` is ``docs/design/rust-core/phase-1/evidence``.

The pack, in ``--out``, nothing written anywhere else:

- ``cases.md`` (item 2): one row per inventory case and
  ``base.adventureworks``, in ID order: the reference's outcome, each
  runner's in each cell observed, the archive at 3887762 (not measured: it
  has no contract runner), the class and the register entries that excuse
  a difference. Outcomes are compared on the fields each runner observes
  (``compare.py``'s ``UNOBSERVED_BY_RUST_HOST``) and written as
  ``ratchet.py``'s ``outcome_text`` writes them. The class is ``correct
  success`` or ``expected refusal`` where every runner answers as the
  reference did and the comparator names no gating difference of the case;
  ``approved improvement`` where every outcome difference is stated by an
  approved entry's ``case_outcomes`` item for that case, cell (rust-host
  under ``ref-pandas3``), runner and observed outcome, an owner's for a new
  refusal; and ``regression`` otherwise. A regression includes:
  - a runner or the reference not observing the case;
  - an ``INTERNAL_ERROR`` from python-core or rust-host, a panic, even as
    the reference's; python-old's is held only where ``ratchet.py``'s
    ``old_client_crash`` holds it;
  - any gating ``DIFFERS`` line naming the case, since the gate compares
    against expectations the approved entries already moved.
  A proposed or rejected entry excuses nothing.
- ``seams.md`` (item 3): each case's deepest stage, over the core runners'
  observations and per core runner and cell (``driver Arrow``,
  ``admission``, ``inferred schema``, ``payload``, ``result``; before any,
  ``panic`` for an ``INTERNAL_ERROR`` and ``refusal`` for another refusal),
  and per seam the stages none of its cases reached, listed as unproven,
  with those rust-host alone never reached.
- ``comparator.md`` (item 4, in part): ``compare.log``'s differences counted
  by rule (``no rule`` for a line of none), the register's entries by
  approver, the owner's first, and the comparator's and the ratchet's lines
  as they printed them. ``compare.py`` prints a field's expected and
  observed values, so these lines carry values; a line naming a held-out
  case (its ID anywhere in it, a plain substring match, which may hide too
  much and never too little) is counted and never shown, and so is a
  register entry naming one anywhere, its free text included (its ID still
  stands where it excuses a listed case's difference). The cases
  for the owner's call, the type-set findings and the DT and ORD results
  are not here: they come with T1.33 part 2.
- ``heldout.md``: the held-out bundle's digests (as-built's T1.20 recipes)
  and counts, or that none was given.

A regression is not a failure here: the pack is written and says so. This
is reporting, not a gate. An input missing, unreadable or of another run
fails the run by name (a case the inventory lists with no case file, or an
observation of another commit or case file, by its ID), with nothing
written; so does a link where the pack writes. Exit status 1 then. It writes
no bytecode, so nothing lands outside ``--out`` either way.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType
from typing import Any

# Nothing outside --out, bytecode included: set before any harness module
# loads (``_module``), so neither a run that writes the pack nor one that
# fails leaves a ``__pycache__`` beside them.
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
#: The case the per-case rows hold beside the inventory's (E27).
BASELINE_CASE = 'base.adventureworks'
#: The runners the gate's 1b half runs, each required in ``--gate-out``.
RUNNERS = ('python-core', 'python-old', 'rust-host')
CORE = ('python-core', 'rust-host')
#: The stages a core runner's observation shows, shallowest first.
STAGES = ('driver Arrow', 'admission', 'inferred schema', 'payload', 'result')
CLASSES = (
    'correct success',
    'expected refusal',
    'approved improvement',
    'regression',
)
ARCHIVE = 'not measured'
#: Not a bundle file: as-built's T1.20 recipe leaves them out.
DIGEST_FILES = ('BUNDLE_SHA256', 'INPUTS_SHA256')
#: ``compare.py``'s lines, by kind of difference: a gating rule's, a
#: diagnostic rule's, and a diagnostic case's or a reported one; each
#: followed by ``<rule>: <case>: ...``, or by text of no rule.
KINDS = {
    'gating': re.compile(r'DIFFERS '),
    'diagnostic rule': re.compile(r'diagnostic rule '),
    'reported, not gated': re.compile(r'(?:diagnostic|reported \([^)]*\)): '),
}
NO_RULE = 'no rule'
VERDICT = re.compile(r'compare: (?P<count>no|[0-9]+) differences')
#: How gate.sh prints a step's output, and names the step.
PREFIX = '    | '
COMPARE_STEP = re.compile(r'compare +(?:ok|FAIL \(exit [0-9]+\))')


class Refused(Exception):
    r"""An input this cannot use, named; nothing is written."""


@functools.cache
def _module(name: str) -> ModuleType:
    r"""A module of this harness, shared, not copied: ``ratchet.py``'s
    readers, projections and ``outcome_text`` (with ``register.py`` and the
    runner's case loader through it), and ``compare.py``'s rules and what
    rust-host cannot observe."""
    spec = importlib.util.spec_from_file_location(
        f'evidence_{name}', HERE / f'{name}.py'
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before it runs: its dataclasses look their module up.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def ratchet() -> ModuleType:
    return _module('ratchet')


def compare() -> ModuleType:
    return _module('compare')


# --- inputs ------------------------------------------------------------------


def _file(option: str, path: Path) -> Path:
    if not path.is_file():
        raise Refused(f'{option} {path}: no such file')
    return path


def _dir(option: str, path: Path) -> Path:
    if not path.is_dir():
        raise Refused(f'{option} {path}: no such directory')
    return path


def _text(option: str, path: Path) -> str:
    try:
        return _file(option, path).read_text(encoding='utf-8')
    except UnicodeDecodeError:
        raise Refused(f'{option} {path}: not UTF-8') from None


def _lines(option: str, path: Path) -> list[dict[str, Any]]:
    try:
        return ratchet().jsonl(_text(option, path), path.name)
    except ratchet().Unreadable as error:
        raise Refused(f'{option}: {error}') from None


def observations(
    gate_out: Path,
) -> dict[tuple[str, str | None], dict[str, dict[str, Any]]]:
    r"""The observations in ``--gate-out``, by runner and cell, then case;
    every runner of :data:`RUNNERS` required. A python-ref file there is
    left out: the reference is ``--reference``'s."""
    found: dict[tuple[str, str | None], dict[str, dict[str, Any]]] = {}
    for path in sorted(_dir('--gate-out', gate_out).glob('*.jsonl')):
        for line in _lines('--gate-out', path):
            runner = ratchet().at(line, 'identity.runner')
            cell = ratchet().at(line, 'identity.cell')
            case = line.get('case')
            if runner not in ratchet().RUNNERS or not isinstance(case, str):
                raise Refused(f'--gate-out: {path.name}: a line of no runner')
            if runner == 'python-ref':
                continue
            by_case = found.setdefault((runner, cell), {})
            if case in by_case:
                raise Refused(
                    f'--gate-out: {case} observed twice by {_label(runner, cell)}'
                )
            by_case[case] = line
    for runner in RUNNERS:
        if not any(key[0] == runner for key in found):
            raise Refused(
                f'--gate-out {gate_out}: no {runner} observations '
                f'({runner}.jsonl)'
            )
    for cell in ratchet().register_module().runner().CELLS:
        if ('python-core', cell) not in found:
            raise Refused(
                f'--gate-out {gate_out}: no python-core observations in the '
                f'{cell} cell'
            )
    return found


def provenance(
    observed: dict[tuple[str, str | None], dict[str, dict[str, Any]]],
    commit: str,
    digests: dict[str, str],
) -> None:
    r"""Every observation is of the gate's run: its commit the gate JSON's
    ``sha``, from a clean checkout, its case file the one given, by
    SHA-256."""
    at = ratchet().at
    for runner, cell in columns(observed):
        label = _label(runner, cell)
        for case, line in sorted(observed[(runner, cell)].items()):
            if at(line, 'identity.commit.sha') != commit:
                raise Refused(
                    f'{case}: {label} observed another commit than the gate '
                    f"run's {commit}"
                )
            if at(line, 'identity.commit.clean') is not True:
                raise Refused(
                    f'{case}: {label} observed it from a checkout with '
                    'uncommitted changes (identity.commit.clean)'
                )
            if case not in digests:
                raise Refused(
                    f'{case}: {label} observed it, with no case file in '
                    '--cases or --heldout'
                )
            if at(line, 'identity.inputs.case') != digests[case]:
                raise Refused(
                    f'{case}: {label} observed another case file than the '
                    'one given (its SHA-256 differs)'
                )


def reference(path: Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    for line in _lines('--reference', path):
        case = line.get('case')
        if ratchet().at(line, 'identity.runner') != 'python-ref':
            raise Refused(f'--reference {path.name}: a line not python-ref')
        if not isinstance(case, str) or case in found:
            raise Refused(f'--reference {path.name}: {case} twice or unnamed')
        found[case] = line
    return found


def corpus(path: Path) -> dict[str, tuple[dict[str, Any], str]]:
    r"""The case files under ``--cases``, by ID, each with its file's
    SHA-256, read by the runner's loader."""
    run_python = ratchet().register_module().runner()
    try:
        loaded = run_python.load_cases([_dir('--cases', path)])
    except run_python.RunnerFailure as error:
        raise Refused(f'--cases: {error}') from None
    return {case['id']: (case, sha) for case, sha in loaded}


def _yaml(option: str, path: Path) -> Any:
    register = ratchet().register_module()
    try:
        return register.read_yaml(_text(option, path), path.name)
    except register.Unreadable as error:
        raise Refused(f'{option}: {error}') from None


def rows(inventory: Path, cases: dict[str, Any]) -> list[str]:
    r"""The per-case rows: the inventory's cases and
    :data:`BASELINE_CASE`, each with its case file."""
    document = _yaml('--inventory', inventory)
    listed = document.get('cases') if isinstance(document, dict) else None
    if not isinstance(listed, list):
        raise Refused(f'--inventory {inventory.name}: no cases')
    found = {
        case['id']
        for case in listed
        if isinstance(case, dict) and isinstance(case.get('id'), str)
    }
    found.add(BASELINE_CASE)
    for ident in sorted(found):
        if ident not in cases:
            raise Refused(
                f'{ident}: in the inventory, with no case file in --cases'
            )
    return sorted(found)


def entries(path: Path) -> list[dict[str, Any]]:
    document = _yaml('--register', path)
    listed = document.get('entries') if isinstance(document, dict) else None
    if not isinstance(listed, list):
        raise Refused(f'--register {path.name}: no entries')
    return [entry for entry in listed if isinstance(entry, dict)]


def gate_json(path: Path, out: Path) -> tuple[str, str, str]:
    r"""The gate JSON's name, SHA-256 and the commit it gated; it must be
    outside ``--out``."""
    data = _file('--gate-json', path).read_bytes()
    if path.resolve().is_relative_to(out.resolve()):
        raise Refused(
            f'--gate-json {path}: inside --out; the gate JSON stays outside '
            'the tree, named by its SHA-256 (design.md:432)'
        )
    try:
        commit = json.loads(data).get('sha')
    except (ValueError, RecursionError, AttributeError):
        raise Refused(f'--gate-json {path.name}: not a gate JSON') from None
    if not isinstance(commit, str):
        raise Refused(f'--gate-json {path.name}: no sha')
    return path.name, hashlib.sha256(data).hexdigest(), commit


# --- item 2: the per-case table ----------------------------------------------


def _label(runner: str, cell: str | None) -> str:
    if runner == 'rust-host':
        return runner
    return f'{runner} ({cell or "no cell recorded"})'


def columns(
    observed: dict[tuple[str, str | None], Any],
) -> list[tuple[str, str | None]]:
    r"""Each runner and cell observed, the reference's cell first."""
    first = ratchet().CELL
    return sorted(
        observed,
        key=lambda key: (
            RUNNERS.index(key[0]),
            key[1] != first,
            key[1] or '',
        ),
    )


def fields(runner: str) -> list[str]:
    r"""The outcome fields ``runner`` observes."""
    unobserved = (
        compare().UNOBSERVED_BY_RUST_HOST if runner == 'rust-host' else ()
    )
    return [
        name
        for name in ratchet().OUTCOME_FIELDS
        if name.removeprefix('outcome.') not in unobserved
    ]


def excused_by(
    register: list[dict[str, Any]],
    case: str,
    runner: str,
    cell: str | None,
    new: str,
    refusal: bool,
) -> str | None:
    r"""The first approved entry stating this case's outcome on this runner
    and cell (rust-host's under the reference's cell; a Python runner with
    no cell recorded matches none), by the owner for a new refusal, never
    ``INTERNAL_ERROR``; or None."""
    for entry in ratchet().approved({'entries': register}):
        if refusal and entry.get('approved_by') != 'owner':
            continue
        for item in entry.get('case_outcomes') or []:
            if (
                isinstance(item, dict)
                and item.get('case') == case
                and item.get('cell')
                == (ratchet().CELL if runner == 'rust-host' else cell)
                and item.get('runner') == runner
                and item.get('new') == new
                and ratchet().PANIC not in new
            ):
                return str(entry.get('id'))
    return None


def classify(
    case: str,
    ref: dict[str, Any] | None,
    observed: dict[tuple[str, str | None], dict[str, dict[str, Any]]],
    register: list[dict[str, Any]],
    gating: Counter[str],
) -> dict[str, Any]:
    r"""One case's row: each outcome as text, its class, the entries that
    excuse its differences and what nothing excuses. ``gating`` counts the
    comparator's gating differences by case."""
    r = ratchet()
    row: dict[str, Any] = {
        'reference': r.outcome_text(r.projection(ref, list(r.OUTCOME_FIELDS)))
        if ref is not None
        else 'not observed',
        'outcomes': {},
        'excused': set(),
        'unexcused': [],
    }
    if ref is None:
        row['unexcused'].append('reference: not observed')
    if gating[case]:
        row['unexcused'].append(
            f'compare: {gating[case]} gating difference(s), see comparator.md'
        )
    for runner, cell in columns(observed):
        label = _label(runner, cell)
        line = observed[(runner, cell)].get(case)
        if line is None:
            row['outcomes'][label] = 'not observed'
            row['unexcused'].append(f'{label}: not observed')
            continue
        now = r.projection(line, fields(runner))
        text = r.outcome_text(now)
        row['outcomes'][label] = text
        if now['outcome.code'] == r.PANIC:
            if not r.old_client_crash(runner, line, ref, ordered()):
                row['unexcused'].append(
                    f'{label}: {text}, a panic, which no entry excuses'
                )
            continue
        if ref is None:
            continue
        want = r.projection(ref, fields(runner))
        if now == want:
            continue
        refusal = (
            now['outcome.result'] == 'refusal'
            and want['outcome.result'] == 'success'
        )
        entry = excused_by(register, case, runner, cell, text, refusal)
        if entry is None:
            row['unexcused'].append(
                f'{label}: {text}, no approved '
                + ('owner ' if refusal else '')
                + 'entry stating it'
            )
        else:
            row['excused'].add(entry)
    if row['unexcused']:
        row['class'] = 'regression'
    elif row['excused']:
        row['class'] = 'approved improvement'
    elif r.at(ref, 'outcome.result') == 'success':
        row['class'] = 'correct success'
    else:
        row['class'] = 'expected refusal'
    return row


@functools.cache
def ordered() -> Any:
    r"""The comparator policy's ``ordered`` statements, which
    ``old_client_crash`` compares in order."""
    path = REPO / ratchet().POLICY
    policy = _yaml('policy', path)
    return policy.get('ordered') if isinstance(policy, dict) else None


def _cell(text: Any) -> str:
    return str(text).replace('|', '\\|')


def _table(header: list[str], body: list[list[Any]]) -> list[str]:
    return [
        '| ' + ' | '.join(header) + ' |',
        '|' + '|'.join('---' for _ in header) + '|',
        *(
            '| ' + ' | '.join(_cell(cell) for cell in line) + ' |'
            for line in body
        ),
    ]


def _named(register: list[dict[str, Any]], case: str) -> list[str]:
    r"""The entries naming ``case`` that excuse none of its differences."""
    return sorted(
        f'{entry.get("id")} ({entry.get("status")})'
        for entry in register
        if case in (entry.get('cases') or [])
        or any(
            isinstance(item, dict) and item.get('case') == case
            for item in entry.get('case_outcomes') or []
        )
    )


def cases_md(
    head: list[str],
    idents: list[str],
    found: dict[str, dict[str, Any]],
    register: list[dict[str, Any]],
) -> str:
    labels = list(next(iter(found.values()))['outcomes'])
    body = []
    for ident in idents:
        row = found[ident]
        detail = '; '.join(row['unexcused'])
        named = [
            name
            for name in _named(register, ident)
            if name.split(' ')[0] not in row['excused']
        ]
        if detail and named:
            detail += '; entries naming the case: ' + ', '.join(named)
        body.append(
            [
                ident,
                row['reference'],
                *(row['outcomes'][label] for label in labels),
                ARCHIVE,
                row['class'],
                ', '.join(sorted(row['excused'])) or '-',
                detail or '-',
            ]
        )
    counts = Counter(found[ident]['class'] for ident in idents)
    return '\n'.join(
        [
            '# Per-case table (item 2)',
            '',
            *head,
            '',
            'One row per Phase 1b inventory case and `base.adventureworks`. '
            "Each outcome as `ratchet.py`'s `outcome_text` writes it, on the "
            'fields its runner observes. The archive at 3887762 has no '
            'contract runner, so its column is not measured (C9). The class '
            'is `correct success` or `expected refusal` where every runner '
            'answers as the reference did and the comparator names no '
            'gating difference of the case, `approved improvement` where '
            'every outcome difference is stated by an approved entry (the '
            "owner's for a new refusal), and `regression` otherwise: a case "
            'a runner did not observe, a panic (`INTERNAL_ERROR` from a core '
            "runner, or python-old's unless it is the reference's own "
            'crash), or a gating difference of the comparator. An entry '
            'states a cell: one naming only `ref-pandas3` excuses nothing '
            'in another cell.',
            '',
            *(f'- {name}: {counts[name]}' for name in CLASSES),
            '',
            *_table(
                [
                    'case',
                    'reference',
                    *labels,
                    'archive (3887762)',
                    'class',
                    'register entry',
                    'unexcused',
                ],
                body,
            ),
            '',
        ]
    )


# --- item 3: the per-seam table ----------------------------------------------


def reached(line: dict[str, Any]) -> set[str]:
    r"""The stages a core runner's observation shows reached."""
    core = line.get('core') if isinstance(line.get('core'), dict) else {}
    answers = ('described', 'materialized', 'predicted')
    shown = {
        'driver Arrow': bool(core.get('fetches')),
        'admission': bool(core.get('fetches'))
        and any(core.get(name) is not None for name in answers),
        'inferred schema': core.get('described') is not None,
        'payload': core.get('materialized') is not None,
        'result': ratchet().at(core, 'predicted.result') is not None,
    }
    return {stage for stage, shows in shown.items() if shows}


def depth(lines: list[dict[str, Any]]) -> str:
    r"""The deepest stage the observations reached; before any, ``panic``
    for an ``INTERNAL_ERROR``, ``refusal`` for another refusal, else
    ``none``; ``not observed`` for none."""
    at = ratchet().at
    stages = set().union(*(reached(line) for line in lines))
    if stages:
        return max(stages, key=STAGES.index)
    if not lines:
        return 'not observed'
    codes = {at(line, 'outcome.code') for line in lines} | {
        at(line, 'core.error.code') for line in lines
    }
    if ratchet().PANIC in codes:
        return 'panic'
    if any(
        at(line, 'outcome.result') == 'refusal'
        or at(line, 'core.error') is not None
        for line in lines
    ):
        return 'refusal'
    return 'none'


def seams() -> list[str]:
    r"""The seams the case format names."""
    schema = json.loads((REPO / 'contract/case.schema.json').read_text())
    return list(schema['$defs']['seam']['enum'])


def seams_md(
    head: list[str],
    idents: list[str],
    cases: dict[str, dict[str, Any]],
    observed: dict[tuple[str, str | None], dict[str, dict[str, Any]]],
) -> str:
    core = [key for key in columns(observed) if key[0] in CORE]
    per_case = []
    by_seam: dict[str, list[tuple[set[str], set[str]]]] = {
        seam: [] for seam in seams()
    }
    for ident in idents:
        seam = cases[ident].get('seam') or '-'
        lines = {key: observed[key].get(ident) for key in core}
        seen = [line for line in lines.values() if line is not None]
        per_case.append(
            [
                ident,
                seam,
                cases[ident].get('until'),
                depth(seen),
                *(depth([line] if line else []) for line in lines.values()),
            ]
        )
        rust = (
            lines[('rust-host', None)] if ('rust-host', None) in lines else None
        )
        by_seam.setdefault(seam, []).append(
            (
                set().union(*(reached(line) for line in seen)),
                reached(rust) if rust else set(),
            )
        )
    per_seam = []
    for seam in sorted(by_seam):
        anyone = set().union(*(found for found, _ in by_seam[seam]))
        rust = set().union(*(found for _, found in by_seam[seam]))
        per_seam.append(
            [
                seam,
                len(by_seam[seam]),
                '; '.join(s for s in STAGES if s not in anyone) or 'none',
                '; '.join(s for s in STAGES if s not in rust) or 'none',
            ]
        )
    return '\n'.join(
        [
            '# Per hard seam (item 3)',
            '',
            *head,
            '',
            'The deepest stage each case reached on any core runner, and on '
            'each core runner and cell: `driver Arrow` (the core asked for '
            'data), `admission` (it answered past the data fetched), '
            '`inferred schema`, `payload`, `result`; before any of them, '
            '`panic` for an `INTERNAL_ERROR` and `refusal` for another '
            'refusal.',
            '',
            *_table(
                [
                    'case',
                    'seam',
                    'runs until',
                    'deepest',
                    *(_label(*key) for key in core),
                ],
                per_case,
            ),
            '',
            'Per seam, the stages no case of it reached on any core runner: '
            'unproven; and those rust-host alone never reached.',
            '',
            *_table(
                ['seam', 'cases', 'unproven', 'not reached on rust-host'],
                per_seam,
            ),
            '',
        ]
    )


# --- item 4: the comparator report -------------------------------------------


def compare_lines(text: str) -> list[str]:
    r"""The compare step's lines: raw, or as gate.sh prints them, the
    prefixed block after the line naming the step, or else every prefixed
    line."""
    lines = text.splitlines()
    for number, line in enumerate(lines):
        if COMPARE_STEP.fullmatch(line):
            block = []
            for after in lines[number + 1 :]:
                if not after.startswith(PREFIX):
                    break
                block.append(after.removeprefix(PREFIX))
            return block
    if any(line.startswith(PREFIX) for line in lines):
        return [
            line.removeprefix(PREFIX)
            for line in lines
            if line.startswith(PREFIX)
        ]
    return lines


def differences(
    text: str,
) -> tuple[list[tuple[str, str, str | None, str]], list[str]]:
    r"""The compare step's differences, each its kind, rule (``no rule``
    for a line of none), case and line, and its lines to show; refused by
    name unless exactly one verdict counts the ``DIFFERS`` lines read."""
    rules = compare().RULES
    found = []
    verdicts = []
    for line in compare_lines(text):
        verdict = VERDICT.fullmatch(line)
        if verdict is not None:
            verdicts.append(verdict)
            continue
        for kind, pattern in KINDS.items():
            start = pattern.match(line)
            if start is None:
                continue
            parts = line[start.end() :].split(': ', 2)
            rule = parts[0] if parts[0] in rules else NO_RULE
            case = parts[1] if rule != NO_RULE and len(parts) > 1 else None
            found.append((kind, rule, case, line))
            break
    if len(verdicts) != 1:
        raise Refused(
            f'--gate-out: compare.log: {len(verdicts)} verdict lines '
            "('compare: ... differences'), not one"
        )
    said = verdicts[0].group('count')
    count = 0 if said == 'no' else int(said)
    read = sum(kind == 'gating' for kind, *_ in found)
    unplaced = sum(kind == 'gating' and not case for kind, _, case, _ in found)
    if unplaced:
        raise Refused(
            f'--gate-out: compare.log: {unplaced} gating difference(s) name '
            'no case (a runner failure): no case could be classed by them, '
            'so the pack is not written'
        )
    if count != read:
        raise Refused(
            f'--gate-out: compare.log: the verdict counts {count} '
            f'differences, and {read} DIFFERS lines are there'
        )
    return found, [line for *_, line in found] + [verdicts[0].group(0)]


def _approver(entry: dict[str, Any]) -> tuple[int, str]:
    who = entry.get('approved_by')
    return ({'owner': 0, 'reviewer': 1}.get(str(who), 2), str(entry.get('id')))


def comparator_md(
    head: list[str],
    found: list[tuple[str, str, str | None, str]],
    lines: list[str],
    notes: str,
    register: list[dict[str, Any]],
    hidden: set[str],
    withheld: int,
) -> str:
    r"""Item 4, in part. A line naming a held-out case is counted, never
    shown; ``register`` is the entries naming none, ``withheld`` the count
    of the others."""
    counts: dict[str, Counter[str]] = {}
    for kind, rule, _, _ in found:
        counts.setdefault(rule, Counter())[kind] += 1
    shown = [line for line in lines if not _names(line, hidden)]
    noted = [line for line in notes.splitlines() if not _names(line, hidden)]
    left_out = len(lines) - len(shown) + len(notes.splitlines()) - len(noted)
    kinds = tuple(KINDS)
    order = [*compare().RULES, NO_RULE]
    by_rule = [
        [rule, *(counts[rule][kind] for kind in kinds)]
        for rule in sorted(counts, key=order.index)
    ]
    listed = [
        [
            entry.get('approved_by') or '-',
            entry.get('id'),
            entry.get('status'),
            entry.get('title'),
            len(entry.get('case_outcomes') or []),
        ]
        for entry in sorted(register, key=_approver)
    ]
    proposed = sorted(
        str(entry.get('id'))
        for entry in register
        if entry.get('status') == 'proposed'
    )
    return '\n'.join(
        [
            '# Comparator report (item 4)',
            '',
            *head,
            '',
            "The `compare` step's differences, counted by rule: a gating "
            "rule's, a diagnostic rule's, and a diagnostic case's or a "
            "reported one. Item 4 in part: the cases for the owner's call, "
            'the type-set findings and the DT-1, DT-2, DT-3 and ORD-1 '
            'results come with T1.33 part 2.',
            '',
            *(
                _table(['rule', *kinds], by_rule)
                if by_rule
                else ['No differences.']
            ),
            '',
            "The register's entries by approver, the owner's first.",
            '',
            *_table(
                ['approved by', 'entry', 'status', 'title', 'case outcomes'],
                listed,
            ),
            '',
            'Proposed, for a decision: '
            + (', '.join(proposed) or 'none')
            + '.',
            '',
            f'Register entries naming a held-out case, counted, not listed: '
            f'{withheld}.',
            '',
            "The comparator's lines as it printed them, values included, "
            'and its verdict:',
            '',
            '```text',
            *shown,
            '```',
            '',
            "The ratchet's notes:",
            '',
            '```text',
            *noted,
            '```',
            '',
            f'Lines naming a held-out case, counted above, not shown: '
            f'{left_out}.',
            '',
        ]
    )


# --- held-out ----------------------------------------------------------------


def _digest_line(name: str, data: bytes) -> bytes:
    return f'{name}\0{hashlib.sha256(data).hexdigest()}\n'.encode()


def heldout(path: Path | None) -> tuple[str, set[str], dict[str, str]]:
    r"""``heldout.md``, digests and counts only; the held-out case IDs,
    which only count elsewhere; and each case file's SHA-256 by ID. The
    digests are as-built.md's T1.20 recipes: BUNDLE over every file but the
    two digest files, INPUTS over the ``h11.*.json`` files without their
    ``expect``, as canonical JSON."""
    if path is None:
        return '# Held-out bundle\n\nNo held-out bundle was given.\n', set(), {}
    files = sorted(
        (file for file in _dir('--heldout', path).iterdir() if file.is_file()),
        key=lambda file: file.name.encode(),
    )
    bundle, inputs, idents, digests = b'', b'', set(), {}
    for file in files:
        data = file.read_bytes()
        if file.name not in DIGEST_FILES:
            bundle += _digest_line(file.name, data)
        if file.suffix != '.json':
            continue
        idents.add(file.stem)
        try:
            case = json.loads(data)
        except (ValueError, RecursionError):
            raise Refused(f'--heldout: {file.name}: not JSON') from None
        if not isinstance(case, dict):
            raise Refused(f'--heldout: {file.name}: not a case')
        if isinstance(case.get('id'), str):
            idents.add(case['id'])
            digests[case['id']] = hashlib.sha256(data).hexdigest()
        if file.name.startswith('h11.'):
            canonical = json.dumps(
                {key: value for key, value in case.items() if key != 'expect'},
                sort_keys=True,
                separators=(',', ':'),
            )
            inputs += _digest_line(file.name, canonical.encode())
    count = sum(file.suffix == '.json' for file in files)
    text = '\n'.join(
        [
            '# Held-out bundle',
            '',
            "Digests and counts only; the cases' contents stay out of the "
            "pack. The digests follow as-built.md's T1.20 recipes.",
            '',
            f'- case files: {count}',
            f'- BUNDLE_SHA256: `{hashlib.sha256(bundle).hexdigest()}`',
            f'- INPUTS_SHA256: `{hashlib.sha256(inputs).hexdigest()}`',
            '',
        ]
    )
    return text, idents, digests


def _names(line: str, idents: set[str]) -> bool:
    r"""Whether ``line`` names a held-out case: any of ``idents`` as a plain
    substring, so it may hide too much, never too little (the orchestrator's
    fail-safe rule)."""
    return any(ident in line for ident in idents)


# --- the pack ----------------------------------------------------------------


def pack(arguments: argparse.Namespace) -> dict[str, str]:
    r"""The pack's files by name, every input read first."""
    name, sha, commit = gate_json(arguments.gate_json, arguments.out)
    head = [
        f'Generated by `scripts/rust_core/evidence.py` from the gate run '
        f'`{name}` (commit `{commit}`), SHA-256 `{sha}`, which stays '
        'outside the tree '
        '(design.md:432). Regenerate it rather than edit it.'
    ]
    observed = observations(arguments.gate_out)
    differing, lines = differences(
        _text('--gate-out', arguments.gate_out / 'compare.log')
    )
    notes = _text('--gate-out', arguments.gate_out / 'ratchet.log')
    ref = reference(arguments.reference)
    loaded = corpus(arguments.cases)
    idents = rows(arguments.inventory, loaded)
    register = entries(arguments.register)
    held, hidden, digests = heldout(arguments.heldout)
    # An entry naming a held-out case anywhere, its free text included, is
    # only counted: the pack lists the others.
    listed = [
        entry
        for entry in register
        if not _names(json.dumps(entry, sort_keys=True, default=str), hidden)
    ]
    provenance(
        observed,
        commit,
        {**digests, **{ident: sha for ident, (_, sha) in loaded.items()}},
    )
    gating = Counter(
        case for kind, _, case, _ in differing if kind == 'gating' and case
    )
    found = {
        ident: classify(ident, ref.get(ident), observed, register, gating)
        for ident in idents
    }
    cases = {ident: case for ident, (case, _) in loaded.items()}
    return {
        'cases.md': cases_md(head, idents, found, listed),
        'seams.md': seams_md(head, idents, cases, observed),
        'comparator.md': comparator_md(
            head,
            differing,
            lines,
            notes,
            listed,
            hidden,
            len(register) - len(listed),
        ),
        'heldout.md': held,
    }


def write(out: Path, texts: dict[str, str]) -> None:
    r"""The pack into ``out``, never through a link: one where a file goes,
    or ``out`` itself a link, is refused before anything is written."""
    if out.is_symlink() or (out.exists() and not out.is_dir()):
        raise Refused(f'--out {out}: not a directory')
    for name in texts:
        path = out / name
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise Refused(f'--out: {name} is a link or not a file; not written')
    out.mkdir(parents=True, exist_ok=True)
    for name, text in sorted(texts.items()):
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW
        with os.fdopen(os.open(out / name, flags, 0o644), 'w') as handle:
            handle.write(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    for option in (
        '--gate-out',
        '--gate-json',
        '--reference',
        '--cases',
        '--inventory',
        '--register',
        '--out',
    ):
        parser.add_argument(option, type=Path, required=True)
    parser.add_argument('--heldout', type=Path)
    arguments = parser.parse_args(argv)
    try:
        texts = pack(arguments)
        write(arguments.out, texts)
    except Refused as error:
        print(f'evidence: {error}')
        print('evidence: FAIL, nothing written')
        return 1
    for name in sorted(texts):
        print(f'evidence: wrote {name}')
    print('evidence: ok')
    return 0


if __name__ == '__main__':
    sys.exit(main())
