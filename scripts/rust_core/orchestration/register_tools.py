#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Two chores around the quirk register that recur in every phase.

    register_tools.py proposed [--register FILE] [--decisions FILE ...]
                               [--approver owner|reviewer]
    register_tools.py resolve BASE OURS THEIRS [--path NAME]
    register_tools.py resolve --index PATH [PATH ...]

``proposed`` lists the register entries still ``status: proposed`` with who
must approve each, then the decisions rows and open decisions that wait for
the owner. The approver is the one the entry's notes name ("Approver: owner"
or "Approver: reviewer", required from Phase 1 decision E89(d)); an entry
that names none gets the old heuristic, marked ``(guess)``: the owner when
its text says it refuses an input the reference accepts or names the owner,
the independent reviewer otherwise. Review a guess by hand. The defaults are
``docs/design/rust-core/register.yaml`` and every
``docs/design/rust-core/phase-*/decisions.md``.

``resolve`` is a three-way merge of the two kinds of file that parallel
packages edit at once and that git's line merge conflicts on for no reason:

- ``register.yaml``, merged entry by entry (an entry is the block from its
  ``- id:`` line to the next entry or ``# ---`` banner; a banner is a block
  of its own). An entry one side changed takes that side's text; one only
  the other side added is inserted after the entry before it on that side
  (and after any entries only ours added there);
  one a side removed and the other left alone is removed. An entry both
  sides changed differently is a conflict, written between the usual
  ``<<<<<<<``/``=======``/``>>>>>>>`` markers. The text above the first
  entry merges the same way, as one block.
- a ``*-pending.json`` acceptance-test file, merged group by group: every
  top-level object of lists (``pending``, ``authored``) is merged per group
  as sets kept in order (ours, less what the other side removed, plus what it
  added), every other key (``about``) takes the side that changed it. A key
  both sides changed differently keeps ours and is reported as a conflict.

It exits 0 when the merge is clean, 1 on a conflict (the file is still
written), 2 on bad input. The kind of file comes from ``--path``'s suffix
(``.json`` or not), else from OURS's first character.

With ``--index`` it resolves a conflicted path of a rebase, cherry-pick or
merge in progress from the index's stages 1 to 3 and writes the result to
the working tree; it does not ``git add``. Without, it is a git merge
driver, writing the result over OURS. To use it as one, add to
``.gitattributes``::

    docs/design/rust-core/register.yaml merge=sdm-register
    contract/tests/*-pending.json merge=sdm-register
    scripts/rust_core/*-pending.json merge=sdm-register

and once per clone::

    git config merge.sdm-register.name 'register entries and pending groups'
    git config merge.sdm-register.driver \\
        'python3 scripts/rust_core/orchestration/register_tools.py resolve %O %A %B --path %P'

The tool needs only the standard library, so any ``python3`` runs it.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
REGISTER = Path('docs/design/rust-core/register.yaml')
DECISIONS_GLOB = 'docs/design/rust-core/phase-*/decisions.md'

# --- proposed ---------------------------------------------------------------

APPROVER = re.compile(r'\bApprover:\s*(owner|reviewer)\b', re.IGNORECASE)
NAMES_OWNER = re.compile(
    r"\b(for the owner|owner'?s (call|decision|choice)|needs? (the )?owner|"
    r'owner to (decide|confirm|approve)|the owner (decides|chooses))',
    re.IGNORECASE,
)
REFUSES = re.compile(
    r'(refuses?|refusal of) (an |the )?inputs? (the reference|python|it) '
    r'accept',
    re.IGNORECASE,
)
WAITS_FOR_OWNER = re.compile(
    r'owner to confirm|for the owner|waits? for the owner', re.IGNORECASE
)
SETTLED = re.compile(r'confirmed by (the )?owner|^superseded', re.IGNORECASE)


def _fields(block: str) -> dict[str, str]:
    """An entry's top-level fields as flat text: a folded or literal value
    (``>-``, ``|``) and a mapping's lines are joined to one line."""
    fields: dict[str, list[str]] = {}
    key = None
    for line in block.splitlines():
        top = re.match(r'(?:- |  )([a-z_]+):(?: (.*))?$', line)
        if top:
            key = top.group(1)
            value = (top.group(2) or '').strip()
            fields[key] = [] if value in ('>-', '>', '|', '|-') else [value]
        elif key and line.startswith('    '):
            fields[key].append(line.strip())
        elif line.strip():
            key = None
    return {k: ' '.join(v for v in vs if v) for k, vs in fields.items()}


def proposed_entries(text: str) -> list[dict[str, str]]:
    """Each proposed entry: id, kind, approver, why, decision row, title."""
    _, blocks = _split_register(text)
    rows = []
    for key, block in blocks:
        if key.startswith('banner:'):
            continue
        fields = _fields(block)
        if fields.get('status') != 'proposed':
            continue
        named = APPROVER.search(fields.get('notes', ''))
        if named:
            approver, why = named.group(1).lower(), 'named in notes'
        else:
            searched = ' '.join(
                fields.get(k, '')
                for k in ('title', 'old_behavior', 'new_behavior', 'notes')
            )
            reasons = []
            if NAMES_OWNER.search(searched):
                reasons.append('names the owner')
            if REFUSES.search(searched):
                reasons.append('refuses an accepted input')
            approver = 'owner' if reasons else 'reviewer'
            why = '; '.join(reasons or ['no owner reason']) + ' (guess)'
        row = re.search(r'\brow:\s*(\S+)', fields.get('decision_ref', ''))
        rows.append(
            {
                'id': fields.get('id', key),
                'kind': fields.get('kind', ''),
                'approver': approver,
                'why': why,
                'decision_row': row.group(1) if row else '',
                'title': fields.get('title', ''),
            }
        )
    return rows


def owner_backlog(path: Path) -> list[str]:
    """A decisions file's rows still waiting for the owner, and its open
    decisions (a ``- None...`` bullet is not one)."""
    out = []
    in_open = False
    try:
        name = str(path.resolve().relative_to(REPO))
    except ValueError:
        name = str(path)
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('## '):
            in_open = line.startswith('## Open decisions')
            continue
        if in_open and line.startswith('- '):
            if not line[2:].lower().startswith('none'):
                out.append(f'{name}: open: {line[2:]}')
            continue
        cells = [c.strip() for c in line.split('|')]
        # | ID | Decision | Reason | Status |
        if len(cells) == 6 and cells[0] == '' and cells[5] == '':
            status = cells[4]
            if WAITS_FOR_OWNER.search(status) and not SETTLED.search(status):
                out.append(f'{name}: {cells[1]}: {status}')
    return out


def run_proposed(arguments: argparse.Namespace) -> int:
    register = arguments.register or REPO / REGISTER
    rows = proposed_entries(register.read_text(encoding='utf-8'))
    if arguments.approver:
        rows = [r for r in rows if r['approver'] == arguments.approver]
    for r in rows:
        print(
            '\t'.join(
                r[k]
                for k in (
                    'id',
                    'kind',
                    'approver',
                    'why',
                    'decision_row',
                    'title',
                )
            )
        )
    counts = Counter(r['approver'] for r in rows)
    print(
        f'{len(rows)} proposed: '
        + ', '.join(f'{n} {who}' for who, n in sorted(counts.items()))
    )
    decisions = arguments.decisions
    if decisions is None:
        decisions = sorted(REPO.glob(DECISIONS_GLOB))
    for path in decisions:
        for line in owner_backlog(path):
            print(line)
    return 0


# --- resolve: register.yaml -------------------------------------------------

Block = tuple[str, str]


def _split_register(text: str) -> tuple[str, list[Block]]:
    """The text above the first block, and (key, text) for each block; a key
    is the entry's ID or ``banner:<line>``, made unique by a ``#n`` suffix if
    the same banner appears twice."""
    parts = re.split(r'(?m)^(?=- id: |# --- )', text)
    head, out = parts[0], []
    seen: Counter[str] = Counter()
    for part in parts[1:]:
        found = re.match(r'- id: (\S+)', part)
        key = found.group(1) if found else 'banner:' + part.split('\n')[0]
        seen[key] += 1
        if seen[key] > 1:
            key = f'{key}#{seen[key]}'
        out.append((key, part))
    return head, out


def _body(block: str | None) -> str | None:
    return None if block is None else block.rstrip('\n')


def _three_way(
    base: str | None, ours: str | None, theirs: str | None
) -> tuple[str | None, bool]:
    """The merged value (None: absent) and whether it conflicts. Compared
    without trailing blank lines, which only say where a block stood."""
    b, o, t = _body(base), _body(ours), _body(theirs)
    if o == t or t == b:
        return ours, False
    if o == b:
        return theirs, False
    return ours, True


def _conflict(ours: str | None, theirs: str | None, label: str) -> str:
    return (
        f'<<<<<<< ours ({label})\n{(ours or "").rstrip(chr(10))}\n'
        f'=======\n{(theirs or "").rstrip(chr(10))}\n'
        f'>>>>>>> theirs ({label})\n\n'
    )


def merge_register(base: str, ours: str, theirs: str) -> tuple[str, list[str]]:
    """The merged register and the keys that conflict."""
    base_head, base_blocks = _split_register(base)
    ours_head, ours_blocks = _split_register(ours)
    theirs_head, theirs_blocks = _split_register(theirs)
    conflicts: list[str] = []

    head, clash = _three_way(base_head, ours_head, theirs_head)
    if clash:
        conflicts.append('(head)')
        head = _conflict(ours_head, theirs_head, 'head')

    def separated(blocks: list[Block]) -> dict[str, str]:
        # A block's trailing blank lines are kept, but the last block's
        # single newline would glue it to whatever comes after it here.
        out = dict(blocks)
        if blocks:
            key = blocks[-1][0]
            out[key] = out[key].rstrip('\n') + '\n\n'
        return out

    b = dict(base_blocks)
    o = separated(ours_blocks)
    t = separated(theirs_blocks)
    order = [k for k, _ in ours_blocks]
    merged: dict[str, str] = {}

    for key in order:
        value, clash = _three_way(b.get(key), o[key], t.get(key))
        if clash:
            conflicts.append(key)
            value = _conflict(o[key], t.get(key), key)
        if value is not None:
            merged[key] = value
    theirs_order = [k for k, _ in theirs_blocks]
    for index, key in enumerate(theirs_order):
        if key in o:
            continue
        value, clash = _three_way(b.get(key), None, t[key])
        if clash:  # ours removed what theirs changed
            conflicts.append(key)
            value = _conflict(None, t[key], key)
        if value is None:
            continue
        after = next(
            (p for p in reversed(theirs_order[:index]) if p in merged), None
        )
        # After that entry, and after what only ours added there.
        where = order.index(after) + 1 if after else 0
        while where < len(order) and order[where] not in b | t:
            where += 1
        order.insert(where, key)
        merged[key] = value
    text = head + ''.join(merged[k] for k in order if k in merged)
    return text.rstrip('\n') + '\n', conflicts


# --- resolve: pending JSON --------------------------------------------------


def _merge_list(base: list, ours: list, theirs: list) -> list:
    out = [x for x in ours if not (x in base and x not in theirs)]
    out += [x for x in theirs if x not in base and x not in out]
    return out


def _is_groups(value: Any) -> bool:
    return isinstance(value, dict) and all(
        isinstance(v, list) for v in value.values()
    )


def merge_pending(base: dict, ours: dict, theirs: dict) -> tuple[dict, list]:
    """The merged pending file and the keys that conflict."""
    out = json.loads(json.dumps(ours))
    conflicts = []
    for key in list(dict.fromkeys([*ours, *theirs, *base])):
        b, o, t = base.get(key), ours.get(key), theirs.get(key)
        if all(v is None or _is_groups(v) for v in (b, o, t)):
            b, o, t = b or {}, o or {}, t or {}
            groups = {}
            for group in dict.fromkeys([*o, *t, *b]):
                merged = _merge_list(
                    b.get(group, []), o.get(group, []), t.get(group, [])
                )
                gone = group in b and (group not in t or group not in o)
                if merged or not gone:
                    groups[group] = merged
            if groups or key in ours:
                out[key] = groups
            else:
                out.pop(key, None)
            continue
        if all(v is None or isinstance(v, list) for v in (b, o, t)):
            out[key] = _merge_list(b or [], o or [], t or [])
            continue
        if o == t or t == b:
            continue
        if o == b:
            if key in theirs:
                out[key] = t
            else:
                out.pop(key, None)
            continue
        conflicts.append(key)
    return out, conflicts


def _dump_pending(value: dict) -> str:
    return json.dumps(value, indent=1, ensure_ascii=False) + '\n'


# --- resolve: driver --------------------------------------------------------


def _resolve_text(
    path_name: str, base: str, ours: str, theirs: str
) -> tuple[str, list[str]]:
    is_json = (
        path_name.endswith('.json')
        if path_name
        else ours.lstrip().startswith('{')
    )
    if not is_json:
        return merge_register(base, ours, theirs)
    sides = []
    for label, text in (('base', base), ('ours', ours), ('theirs', theirs)):
        try:
            sides.append(json.loads(text) if text.strip() else {})
        except json.JSONDecodeError as error:
            raise ValueError(f'{label} is not JSON: {error}') from None
    merged, conflicts = merge_pending(*sides)
    return _dump_pending(merged), conflicts


def _git_stage(stage: int, path: str) -> str:
    out = subprocess.run(
        ['git', 'show', f':{stage}:{path}'],
        capture_output=True,
        text=True,
    )
    # A stage is missing when a side added or deleted the whole file.
    return out.stdout if out.returncode == 0 else ''


def run_resolve(arguments: argparse.Namespace) -> int:
    jobs: list[tuple[str, tuple[str, str, str], Path]] = []
    if arguments.index:
        if arguments.files:
            print('give --index PATH or BASE OURS THEIRS', file=sys.stderr)
            return 2
        for path in arguments.index:
            sides = (
                _git_stage(1, path),
                _git_stage(2, path),
                _git_stage(3, path),
            )
            jobs.append((path, sides, Path(path)))
    else:
        if len(arguments.files) != 3:
            print('resolve takes BASE OURS THEIRS', file=sys.stderr)
            return 2
        base, ours, theirs = (Path(f) for f in arguments.files)
        texts = (
            base.read_text(encoding='utf-8'),
            ours.read_text(encoding='utf-8'),
            theirs.read_text(encoding='utf-8'),
        )
        jobs.append((arguments.path or '', texts, ours))
    status = 0
    for name, sides, target in jobs:
        try:
            text, conflicts = _resolve_text(name, *sides)
        except ValueError as error:
            print(f'{name or target}: {error}', file=sys.stderr)
            return 2
        target.write_text(text, encoding='utf-8')
        label = name or str(target)
        if conflicts:
            status = 1
            print(
                f'{label}: conflict in {", ".join(conflicts)}', file=sys.stderr
            )
        else:
            print(f'{label}: merged', file=sys.stderr)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('\n', 2)[2],
    )
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('proposed', help='list proposed entries and approvers')
    p.add_argument('--register', type=Path)
    p.add_argument('--decisions', type=Path, action='append')
    p.add_argument('--approver', choices=('owner', 'reviewer'))
    r = sub.add_parser('resolve', help='three-way merge by entry and group')
    r.add_argument('files', nargs='*', metavar='BASE OURS THEIRS')
    r.add_argument('--path', help="the file's path in the repository (%%P)")
    r.add_argument('--index', nargs='+', metavar='PATH')
    arguments = parser.parse_args(argv)
    if arguments.command == 'proposed':
        return run_proposed(arguments)
    return run_resolve(arguments)


if __name__ == '__main__':
    sys.exit(main())
