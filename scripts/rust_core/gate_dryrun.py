#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

r"""The checking tools' dry run (T3.0, item 1): which planned case kinds
the tools cannot judge yet, found before the design locks, not one package
at a time mid-phase.

    scripts/rust_core/gate_dryrun.sh PLAN.yaml [--base REV] [--python PY]
        [--rust] [--gate HALF]

PLAN is the design's list of planned case kinds::

    kinds:
      - name: binary-classification
        case: {...}            # a stub case, in the case format
      - name: future-explain
        file: path/to/stub.json

It puts one stub case of each kind on a throwaway commit (a detached
worktree of HEAD under ``$TMPDIR``, removed at the end: the caller's
branch, index and files are never touched), runs the checking tools over
it, and prints each place a tool refuses a kind or cannot judge it, one
line each::

    GAP <kind> (<case id>): <tool>: <what it said>

and ``ok <kind> (<case id>)`` for a kind every tool that ran judged. A
kind refused by the case format is judged no further (an invalid case
file stops every tool that reads the corpus). A tool that fails on no line
naming a kind judged nothing: it prints ``INCOMPLETE <tool> failed ...``
and the dry run exits 1. The tools are the checkpoint's, from a worktree
of it, judging HEAD with the stubs, as the gate's harness judges a
candidate:

- ``format``: the runners' case reader and ``contract/case.schema.json``;
- ``coverage``: ``coverage.py --check`` against the checkpoint (the scope
  manifests know the kind's paths, routes and cells);
- ``ratchet``: ``ratchet.py check`` against the checkpoint;
- ``observe``, with ``--python PY`` (or ``SDM_PYTHON``; an interpreter
  holding the candidate's client, such as the lane's cell): python-core
  and python-old run each stub; a runner failure or an internal error is
  a gap;
- ``compare``, after ``observe``: ``compare.py`` on those observations; a
  stub has no reference line until python-ref records it, which is noted,
  not a gap;
- ``rust-host``, with ``--rust``: the test kit's ``sdm-contract-run``
  (cargo; heavy, run it through ``heavy``);
- ``gate``, with ``--gate HALF``: ``gate.sh --base BASE --locked HALF``
  on the throwaway commit, its failing lines attributed to the kinds they
  name (heavy too). Without ``--gate`` the run is ``INCOMPLETE`` (exit 1):
  the locked gate's refusals are part of what the design must know.

The checkpoint is ``ratchet.py checkpoint --base BASE`` (``BASE``: the
phase branch's remote head, ``origin/<phase>``, as ``phases.py`` names it).
Its gaps are the first package's tooling list. Exit 1 when there is a gap,
0 when none, 2 on a plan or setup it cannot use; it never blocks anything
else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phases  # noqa: E402

STUBS = 'contract/cases/dryrun'
IDENTITY = [
    '-c',
    'user.name=gate dry run',
    '-c',
    'user.email=gate-dryrun@localhost',
]
#: How many unattributed lines of a failing tool are shown.
SHOWN = 5


class Unusable(Exception):
    r"""A plan or setup the dry run cannot use."""


@dataclass
class Kind:
    name: str
    case: dict[str, Any]
    gaps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    judged: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return str(self.case.get('id', ''))

    def names(self) -> set[str]:
        r"""What a tool's line names this kind by: its case id, and the
        paths, routes and cells only it claims (set by :func:`claim`)."""
        return {self.id, *self.case.get('_own', [])} - {''}


def _run(
    argv: list[str], cwd: Path, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
        check=False,
    )


def git(repo: Path, *args: str) -> str:
    done = _run(['git', *args], repo)
    if done.returncode != 0:
        raise Unusable(f'git {" ".join(args)}: {done.stderr.strip()}')
    return done.stdout.strip()


def read_plan(path: Path) -> list[Kind]:
    import yaml

    document = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(document, dict) or not isinstance(
        document.get('kinds'), list
    ):
        raise Unusable(f'{path}: a plan is a mapping with a list under kinds')
    kinds = []
    for number, item in enumerate(document['kinds'], 1):
        if not isinstance(item, dict) or not isinstance(item.get('name'), str):
            raise Unusable(f'{path}: kind {number} has no name')
        case = item.get('case')
        if case is None and isinstance(item.get('file'), str):
            case = json.loads(
                (path.parent / item['file']).read_text(encoding='utf-8')
            )
        if not isinstance(case, dict) or not isinstance(case.get('id'), str):
            raise Unusable(
                f'{path}: kind {item["name"]} has no case with an id'
            )
        kinds.append(Kind(item['name'], dict(case)))
    ids = [k.id for k in kinds]
    if len(set(ids)) != len(ids):
        raise Unusable(f'{path}: two kinds share a case id')
    return kinds


def claim(kinds: list[Kind]) -> None:
    r"""Each kind's own paths, routes and cells: the ones no other kind of
    the plan claims, so a tool's line naming one is that kind's."""
    counts: dict[str, int] = {}
    for kind in kinds:
        for key in {
            *kind.case.get('paths', []),
            *kind.case.get('routes', []),
            *kind.case.get('cells', []),
        }:
            counts[key] = counts.get(key, 0) + 1
    for kind in kinds:
        own = [
            key
            for key in {
                *kind.case.get('paths', []),
                *kind.case.get('routes', []),
                *kind.case.get('cells', []),
            }
            if isinstance(key, str) and counts[key] == 1
        ]
        kind.case = {k: v for k, v in kind.case.items() if k != '_own'}
        kind.case['_own'] = own


def stub_text(kind: Kind) -> str:
    return (
        json.dumps(
            {k: v for k, v in kind.case.items() if k != '_own'}, indent=2
        )
        + '\n'
    )


#: What a stub is expected to meet, noted, never a gap (T3.2, from the full
#: dry run): no reference line until python-ref records it (the design,
#: "The checking tools' dry run": coverage's rule 6 and compare's "no
#: reference observation"), and the comparator's statement counts, which
#: are information.
EXPECTED = re.compile(r'no reference observation|statements per public call:')
#: The words that make a line naming a kind a gap, read outside the kind's
#: own names (an id such as ``dryrun.endpoint.only-call-refused`` holds
#: one).
FAILURE = re.compile(
    r'fail|refus|not |no |missing|unknown|error|invalid|differ', re.I
)


def _without(text: str, names: set[str]) -> str:
    r"""``text`` with each of ``names`` taken out, longest first."""
    for name in sorted(names, key=len, reverse=True):
        text = text.replace(name, ' ')
    return text


def attribute(
    step: str, output: str, kinds: list[Kind], failed: bool
) -> list[str]:
    r"""Each line of a tool's output naming a kind becomes that kind's gap,
    unless the stub is expected to meet it (:data:`EXPECTED`, a note); the
    lines of a failing tool that name none are returned."""
    unclaimed = []
    for line in output.splitlines():
        text = line.strip()
        if not text:
            continue
        owners = [k for k in kinds if any(n in text for n in k.names())]
        named = set().union(*(k.names() for k in owners)) if owners else set()
        if owners and EXPECTED.search(text):
            for kind in owners:
                kind.notes.append(f'{step}: {text}')
        elif owners and FAILURE.search(_without(text, named)):
            for kind in owners:
                kind.gaps.append(f'{step}: {text}')
        elif (
            failed
            and re.search(r'fail|refus|error|problem', text, re.I)
            and not re.fullmatch(r'[\w ]+: (?:FAIL|ok)', text)
        ):
            unclaimed.append(text)
    return unclaimed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('plan', type=Path)
    parser.add_argument('--base')
    parser.add_argument('--python', default=os.environ.get('SDM_PYTHON'))
    parser.add_argument('--rust', action='store_true')
    parser.add_argument('--gate', metavar='HALF')
    arguments = parser.parse_args(argv)
    try:
        return dry_run(arguments)
    except Unusable as error:
        print(f'gate dry run: {error}', file=sys.stderr)
        return 2


def dry_run(arguments: argparse.Namespace) -> int:
    repo = Path(git(Path.cwd(), 'rev-parse', '--show-toplevel'))
    kinds = read_plan(arguments.plan.resolve())
    claim(kinds)
    base = arguments.base
    if base is None:
        try:
            base = 'origin/' + phases.phase_branch(
                branch=phases.checked_out(), environ=os.environ
            )
        except phases.PhaseError as error:
            raise Unusable(f'no base: {error}; give --base') from None
    head = git(repo, 'rev-parse', 'HEAD')
    python = sys.executable
    scratch = Path(tempfile.mkdtemp(prefix='sdm-dryrun-'))
    tree = scratch / 'tree'
    git(repo, 'worktree', 'add', '--quiet', '--detach', str(tree), head)
    unattributed: list[str] = []
    notes: list[str] = []
    #: Tools that failed on no line naming a kind: what they judged is
    #: unknown, so the dry run is incomplete (Codex r1).
    broken: list[str] = []
    judge = scratch / 'judge'
    try:
        # The checkpoint of a commit above HEAD, as the stubs' commit will
        # be: HEAD itself may be the phase head, its own base.
        above = git(
            tree, *IDENTITY, 'commit-tree', git(tree, 'rev-parse', 'HEAD^{tree}'),
            '-p', head, '-m', 'gate dry run: above HEAD',
        )  # fmt: skip
        git(tree, 'checkout', '--quiet', '--detach', above)
        point = _run(
            [python, 'scripts/rust_core/ratchet.py', 'checkpoint', '--repo',
             '.', '--base', base],
            tree,
        )  # fmt: skip
        if point.returncode != 0 or not point.stdout.split():
            raise Unusable(
                f'no checkpoint for --base {base}: {(point.stdout + point.stderr).strip()}'
            )
        checkpoint = point.stdout.split()[0]
        # The checkpoint's tools judge, as the gate's harness does (Codex
        # r1): a kind HEAD's tools accept but the landed ones refuse is the
        # gap this dry run exists to find.
        git(
            repo,
            'worktree',
            'add',
            '--quiet',
            '--detach',
            str(judge),
            checkpoint,
        )
        # A stub of an existing case (a kind the corpus already has) takes
        # an id of its own, so the corpus's case is not repeated.
        existing = set()
        for path in (tree / 'contract/cases').rglob('*.json'):
            try:
                existing.add(
                    json.loads(path.read_text(encoding='utf-8')).get('id')
                )
            except (ValueError, AttributeError):
                continue
        for kind in kinds:
            if kind.id in existing:
                kind.case['id'] = f'{kind.id}.dryrun'
        # format: each stub alone through the runners' case reader.
        reader = (
            'import sys, importlib.util; from pathlib import Path\n'
            'spec = importlib.util.spec_from_file_location("dryrun_runner", sys.argv[1])\n'
            'module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)\n'
            'try:\n    module.load_cases([Path(sys.argv[2])])\n'
            'except module.RunnerFailure as failure:\n    print(failure); sys.exit(1)\n'
        )
        (scratch / 'stubs').mkdir()
        valid = []
        for kind in kinds:
            path = scratch / 'stubs' / f'{kind.id}.json'
            path.write_text(stub_text(kind), encoding='utf-8')
            done = _run(
                [
                    python,
                    '-c',
                    reader,
                    str(judge / 'contract/runners/run_python.py'),
                    str(path),
                ],
                tree,
            )
            if 'ModuleNotFoundError' in done.stderr:
                raise Unusable(
                    f"the tools' Python ({python}) cannot read cases: "
                    f'{done.stderr.strip().splitlines()[-1]}; set SDM_TOOLS_PYTHON'
                )
            if done.returncode != 0:
                said = (done.stdout + done.stderr).strip().splitlines() or [
                    'refused'
                ]
                kind.gaps.append(f'format: {said[-1]}')
            else:
                kind.judged.append('format')
                valid.append(kind)
        for kind in valid:
            target = tree / STUBS / f'{kind.id}.json'
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(stub_text(kind), encoding='utf-8')
        if valid:
            # A commit by plumbing: no hook of the caller's runs on it, and
            # it is on no branch.
            git(tree, 'add', '-A', STUBS)
            written = git(tree, 'write-tree')
            stubbed = git(
                tree,
                *IDENTITY,
                'commit-tree',
                written,
                '-p',
                above,
                '-m',
                'gate dry run: the stub cases',
            )
            git(tree, 'checkout', '--quiet', '--detach', stubbed)
        steps: list[tuple[str, list[str]]] = [
            (
                'coverage',
                [
                    python,
                    str(judge / 'scripts/rust_core/coverage.py'),
                    '--check',
                    '--repo',
                    '.',
                    '--checkpoint',
                    checkpoint,
                ],
            ),
            (
                'ratchet',
                [
                    python,
                    str(judge / 'scripts/rust_core/ratchet.py'),
                    'check',
                    '--repo',
                    '.',
                    '--checkpoint',
                    checkpoint,
                    # The stubs are the dry run's own change, named as a
                    # landing would name it (T3.2).
                    '--protected-paths',
                    f'{STUBS}/**',
                ],
            ),
        ]
        for step, command in steps:
            done = _run(command, tree)
            failed = done.returncode != 0
            loose = attribute(step, done.stdout + done.stderr, valid, failed)
            gapped = any(
                g.startswith(f'{step}:') for kind in valid for g in kind.gaps
            )
            if failed and (loose or not gapped):
                # A failure no kind explains: nothing was judged by it.
                broken.append(
                    f'{step} failed (exit {done.returncode}): '
                    + ' | '.join(
                        (
                            loose
                            or (done.stdout + done.stderr)
                            .strip()
                            .splitlines()[-2:]
                        )[:SHOWN]
                    )
                )
                continue
            for kind in valid:
                if not any(g.startswith(f'{step}:') for g in kind.gaps):
                    kind.judged.append(step)
        # observe, rust-host and compare: each kind alone, so one kind a
        # runner refuses stops no other.
        observations = scratch / 'observations'
        observations.mkdir()
        if valid and not arguments.python:
            notes.append(
                'observe and compare not run: give --python, an interpreter '
                "holding the candidate's client (the lane's cell)"
            )
        for kind in valid if (arguments.python or arguments.rust) else []:
            stub = tree / STUBS / f'{kind.id}.json'
            alone = scratch / 'alone' / kind.id
            alone.mkdir(parents=True)
            shutil.copy2(stub, alone / stub.name)
            made = []
            runs: list[tuple[str, list[str], Path]] = []
            if arguments.python:
                for mode in ('python-core', 'python-old'):
                    out = observations / f'{kind.id}.{mode}.jsonl'
                    runs.append(
                        (
                            f'observe ({mode})',
                            [arguments.python, '-I', str(judge / 'contract/runners/run_python.py'),
                             '--mode', mode, '--out', str(out), str(stub)],
                            out,
                        )
                    )  # fmt: skip
            if arguments.rust:
                out = observations / f'{kind.id}.rust-host.jsonl'
                runs.append(
                    (
                        'rust-host',
                        ['cargo', 'run', '-q', '-p', 'sdm-testkit', '--bin', 'sdm-contract-run',
                         '--locked', '--', '--out', str(out), str(stub)],
                        out,
                    )
                )  # fmt: skip
            for step, command, out in runs:
                done = _run(
                    command, tree / 'rust' if step == 'rust-host' else tree
                )
                said = (done.stdout + done.stderr).strip()
                if done.returncode != 0 or not out.is_file():
                    kind.gaps.append(
                        f'{step}: {(said.splitlines() or ["no observation"])[-1]}'
                    )
                    continue
                made.append(out)
                text = out.read_text(encoding='utf-8')
                if re.search(r'INTERNAL_ERROR|panic', text):
                    kind.gaps.append(
                        f'{step}: an internal error in its observation'
                    )
                else:
                    kind.judged.append(step)
            if not made:
                continue
            done = _run(
                [
                    python,
                    str(judge / 'scripts/rust_core/compare.py'),
                    '--cases',
                    str(alone),
                    *map(str, made),
                ],
                tree,
            )
            said = done.stdout + done.stderr
            if done.returncode == 0:
                kind.judged.append('compare')
                continue
            lines = [line.strip() for line in said.splitlines() if line.strip()]
            told = [
                line
                for line in lines
                if kind.id in line
                or re.search(
                    r'refus|cannot|unknown|not a',
                    _without(line, kind.names()),
                    re.I,
                )
            ]
            expected = [line for line in told if EXPECTED.search(line)]
            kind.notes.extend(f'compare: {line}' for line in expected)
            told = [line for line in told if not EXPECTED.search(line)]
            differing = [line for line in told if line.startswith('DIFFERS')]
            if (
                expected
                and not differing
                and not [line for line in told if 'runner failure' in line]
            ):
                # Only what a stub is expected to meet: judged.
                kind.judged.append('compare')
                continue
            kind.gaps.extend(
                f'compare: {line}' for line in (told or lines[-1:])[:SHOWN]
            )
        if valid and not arguments.gate:
            # The locked gate is part of the check (Codex r2): without the
            # half the design's cases fall in, its refusals are unknown.
            broken.append(
                'gate not run: give --gate HALF, the locked half the planned '
                "kinds fall in, so the checkpoint's locked gate judges them too"
            )
        if arguments.gate and valid:
            done = _run(
                ['bash', 'scripts/rust_core/gate.sh', '--base', base, '--locked', arguments.gate,
                 '--run', 'python-core,python-old', '--out', str(scratch / 'gate'),
                 '--protected-paths', f'{STUBS}/**'],
                tree,
                env={**os.environ, **({'PYTHON': arguments.python} if arguments.python else {})},
            )  # fmt: skip
            failed = done.returncode != 0
            loose = attribute('gate', done.stdout + done.stderr, valid, failed)
            gapped = any(
                g.startswith('gate') for kind in valid for g in kind.gaps
            )
            if failed and (loose or not gapped):
                broken.append(
                    f'gate failed (exit {done.returncode}): '
                    + ' | '.join(
                        (
                            loose
                            or (done.stdout + done.stderr)
                            .strip()
                            .splitlines()[-2:]
                        )[:SHOWN]
                    )
                )
            else:
                for kind in valid:
                    if not any(g.startswith('gate') for g in kind.gaps):
                        kind.judged.append('gate')
    finally:
        if judge.exists():
            _run(['git', 'worktree', 'remove', str(judge)], repo)
        removed = _run(['git', 'worktree', 'remove', str(tree)], repo)
        if removed.returncode == 0:
            shutil.rmtree(scratch, ignore_errors=True)
        else:
            print(
                f'gate dry run: the throwaway worktree stays at {tree}: {removed.stderr.strip()}'
            )
    for kind in kinds:
        if kind.gaps:
            for gap in kind.gaps:
                print(f'GAP {kind.name} ({kind.id}): {gap}')
        else:
            print(
                f'ok {kind.name} ({kind.id}): judged by {", ".join(kind.judged)}'
            )
        for note in kind.notes:
            print(f'note {kind.name} ({kind.id}): {note}')
    for line in unattributed:
        print(f'note: a tool failed on a line naming no kind: {line}')
    for note in notes:
        print(f'note: {note}')
    for line in broken:
        print(f'INCOMPLETE {line}')
    gapped = [k for k in kinds if k.gaps]
    print(
        f'gate dry run: {len(kinds)} kinds against checkpoint {checkpoint[:12]}; '
        + (
            f"{len(gapped)} with gaps ({', '.join(k.name for k in gapped)}): the first package's tooling list"
            if gapped
            else 'every tool that ran judges every kind'
        )
        + (
            f'; INCOMPLETE: {len(broken)} tool(s) failed naming no kind, so '
            'their judgement is unknown'
            if broken
            else ''
        )
    )
    return 1 if gapped or broken else 0


if __name__ == '__main__':
    sys.exit(main())
