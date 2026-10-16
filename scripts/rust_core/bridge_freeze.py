#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bridge freeze: a tripwire for undeclared changes to the core route's bridge.

Decision E50 (the owner, 2026-10-10): a new product rule on a core route
lands in Rust, not in the Python bridge. This guard holds it over the files
that keep a row of the design's "Host-held semantics" ledger:

- the ``_rust/`` bridges, ``_rust/schema.py``, ``_rust/materialize.py`` and
  ``_rust/predict.py``, whole: every function and method;
- the files shared with the old route, for the functions ``COVERED`` names:
  in ``client.py``, ``query.py`` and ``materializers/relational.py`` the
  ledgered functions and those the core route reaches through them; in
  ``schema.py``, ``adapters/relational.py``, ``base.py``, ``result.py`` and
  ``decompose.py`` the ledgered rules' own bodies; and the entry points
  (``SDMClient.predict``, ``PredictiveQuery.materialize``,
  ``RelationalMaterializer.materialize``, ``Schema.from_source``) with
  ``RelationalMaterializer.__init__``;
- ``_rust/__init__.py``, the bridges' package, for its skeleton only;
- and every frozen file's skeleton, as ``<module>``: the whole module in
  order and in scope, each function's body (covered or not) replaced by
  ``pass``. So a module statement, an import, a class attribute moved in or
  out of a class, and a method, dunder or property added to, removed from
  or renamed on any class of a frozen file change it.

``COVERED`` is this file's, so coverage changes only by a reviewed edit.

The tripwire, the whole check from T1.28b (decision E56(d), enacted by the
owner 2026-10-11 as E66(f)): the committed baseline, ``bridge_freeze.json``,
holds a fingerprint of each covered function and each skeleton: the
SHA-256 of its syntax tree with no positions and no docstrings, serialized
the same way on every Python version. Against the merge base of HEAD and
the base branch, every entry whose fingerprint differs, and every entry
that is new or gone (a file leaving ``FROZEN`` included: the base's entries
for it are gone), must be declared in ``bridge_changes.json`` under
``<file>::<qualified name>``, with one reason:

- ``{"moved": "<the ledger row, naming its task T1.xx>"}``, a rule moved
  into Rust; the reason must name the task;
- ``{"refactor": "<one line>"}``, an edit that keeps behaviour; a rename
  is a change, its new name and the skeleton declared a refactor and its
  old name gone;
- ``{"gone": "<one line>"}``, an entry that left the freeze: an entry gone
  against the base is declared gone, or moved when its rule moved into
  Rust, or main, and nothing else;
- ``{"main": "<the full SHA of a commit of main> <its title>"}``, a change
  ``main`` made, brought into the phase branch by the candidate's merge of
  ``main`` (T2.0e): the SHA is 40 lower-case hex digits, then one space and
  the commit's own subject; the commit is an ancestor of ``main`` as last
  fetched (``refs/remotes/origin/main``, never a local branch) and of HEAD,
  is not an ancestor of the merge base (a commit the base already holds was
  judged when it came in), is brought in by the package's merge commit of
  ``main`` on the phase's first-parent history (reached from a parent other
  than its first; a merge inside ``main`` never counts), changes the entry
  against its first parent (entries compared by fingerprint; an entry gone,
  gone there too), the merge's first parent holds the entry as the merge
  base does (no change of the candidate's own before the merge), the
  candidate holds the entry as that merge commit holds it, and, where git
  merges the file cleanly (``git merge-tree``), the merge commit holds git's
  merge of it (a conflict's resolution is review's): ``main``'s change
  merged with the phase's, so an entry both changed passes, and a change the
  candidate made after the merge does not. ``main`` is read from the
  remote-tracking ref as it stands: CI's checkout fetches it, so CI's run,
  the evidence a landing stands on, reads the remote's ``main``; a local run
  reads the ref as last fetched. Review judges, as for the others, whether
  ``main``'s rule belongs in the bridge; the port of a ``main`` rule into
  Rust is its own package's ``moved``.

It fails on an undeclared change; on a declaration of an entry that did not
change (stale), unless the merge base's declarations hold it as it stands,
as on the phase branch after the change landed; on a change declared only
by an entry the merge base's declarations already hold (spent: a later
change states its own reason, so a declaration once landed does not wave
through every later change to its entry); on ``gone`` for an entry that is
not gone, or a gone entry declared anything but gone or moved; and on a
``moved`` that names no task. It reads each frozen file's bytes, as the
interpreter does, and refuses one declaring a source encoding other than
UTF-8 (a ``raw_unicode_escape`` or ``utf-7`` cookie runs what a UTF-8
reader takes for a comment); and it refuses a directory beside a frozen
module with the module's name, whose package would be imported in the
module's place. So a change made to frozen code by ordinary means fails
until it is declared, and review judges each reason's direction. A file
joining ``FROZEN`` has every entry new.

Retired at T1.28b (E56(d)): the secondary layer, that is the per-entry
counts of rule sites and first-party calls, the ``ALLOWANCES`` that let
them rise, the ``allowance`` reason, the netting of the entries declared
``moved`` against each other, and the rules of a spent allowance. Review
judges what a declared change adds, as it judged the allowances' rows.
The baseline's format 3 holds the fingerprints alone; format 2, which also
held the counts, is still read, its counts ignored, since the gate judges
a candidate with its checkpoint's copy of this file, which reads the
candidate's baseline in its own format: T1.28b landed in two parts, part 1
this file alone over a format 2 baseline and the ``ALLOWANCES`` literal
the copy before it read, part 2 the baseline rewritten and the literal
deleted.

Missing or malformed evidence fails: no baseline or declarations, either in
another format, a field missing or retyped, an unreadable base, no merge
base. A base that predates the baseline has nothing to compare, and the
declarations must then be empty.

Threat model (decision E56, checkpoint 4): the threat is honest drift by
cooperating agents, a rule added to the bridge by ordinary means and not
declared. Evasion that needs deliberate obfuscation (a source-encoding
cookie, ``getattr`` or other dynamic dispatch, metaclass or dunder tricks,
a shadowing package, names or code built from strings at run time) is
review's, and a finding of that kind is P3 by rule; the refusals above that
touch it are kept, not extended. The behavioural protection is the
``python-core`` against ``rust-host`` parity on the cases: the pure-Rust host
has no Python, so a rule the bridge alone holds that a case reaches shows as
a divergence. This script is frozen at bug fixes (E56, extending E49(a)).

What it does not see, review's to hold as it holds the ledger's owners and
each declared reason:

- what a declared change adds: a rule added under a declared ``moved`` or
  ``refactor`` reason passes, and review judges the reason's direction;
- a rule added inside the body of an existing function that ``COVERED``
  does not name (a first-party helper covered code already called);
- any code outside the frozen files: data covered code reads (for
  example ``models.py``'s ``RELATIONAL_LIMITS``), module code rebinding a
  covered name, a dunder set on a frozen class from elsewhere, a method on
  a superclass that is not frozen.

When a ledger row moves into Rust, the package deletes the row and the
bridge's rule, takes a function the core route no longer reaches out of
``COVERED`` (in a later change than the one that changes this file, since
the gate measures a candidate with its checkpoint's ``COVERED``), declares
each changed entry ``moved`` naming the task, and runs ``--write``.

``l0_deps.py`` runs this check, so it runs where that does: ``prepush.sh``'s
cheap tier and CI's ``rust-lint``; ``selftest.py`` seeds a fault for each
rule.

The base branch is a setting (E56(c)): ``--base``, else the environment's
``SDM_FREEZE_BASE``, else ``origin/`` and the phase branch (``phases.py``,
T2.0: ``SDM_PHASE_BRANCH``, else the phase of the branch CI runs on or the
branch checked out, else the phase in progress), so the check keeps
working once a phase branch has merged and gone; a blank one is refused.

    scripts/rust_core/bridge_freeze.py [--write] [--base REF]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tokenize
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phases  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path('scripts/rust_core/bridge_freeze.json')
CHANGES = Path('scripts/rust_core/bridge_changes.json')
#: The base branch when neither ``--base`` nor ``BASE_ENV`` names one:
#: ``origin/`` and the phase branch (:func:`phase_base`).
BASE_REF = 'origin/<the phase branch>'
#: The environment's setting of the base branch (E56(c)).
BASE_ENV = 'SDM_FREEZE_BASE'
#: The baseline's format this copy writes: the fingerprints alone.
FORMAT = 'sdm-bridge-freeze/3'
#: The format before E56(d), which also held each entry's counts and
#: calls; read, its counts ignored (see the docstring).
FORMAT_2 = 'sdm-bridge-freeze/2'
#: What a format 2 entry holds beside its fingerprint, retired.
RETIRED = frozenset({'raise', 'warn', 'branch', 'calls'})
CHANGES_FORMAT = 'sdm-bridge-changes/1'
ABOUT = (
    'The bridge freeze (decision E50): per covered function of the core '
    "route's Python bridge, and per frozen file's skeleton, its "
    'fingerprint. Regenerate with scripts/rust_core/bridge_freeze.py '
    '--write; a change against the merge base is declared in '
    'bridge_changes.json.'
)
CHANGES_ABOUT = (
    'Each change to frozen bridge code against the merge base, with one '
    'reason: moved (a rule moved into Rust, naming its task), refactor '
    '(behaviour kept), gone (an entry that left the freeze) or main (a '
    'change main made, merged by this change: the full SHA of the main '
    'commit, then its title). Entries the merge base already holds are '
    'spent and may be pruned.'
)
REASONS = ('moved', 'refactor', 'gone', 'main')
#: Where ``main`` is read for a ``main`` reason: the remote's, as last
#: fetched, never a local branch.
MAIN_REF = 'refs/remotes/origin/main'
#: A ``main`` reason: a commit's full SHA, a space, and its title.
_MAIN_REASON = re.compile(r'([0-9a-f]{40}) (\S.*)')
#: What a ``moved`` reason names: the task that moves the rule.
_TASK = re.compile(r'\bT\d+\.\d+[a-z]?\b')

_SRC = Path('packages/structured-data-models-client/src')
_CLIENT = _SRC / 'sdm_client'
SCRIPT = Path('scripts/rust_core/bridge_freeze.py')
#: The ``_rust/`` bridges, covered whole.
WHOLE = (
    _CLIENT / '_rust/schema.py',
    _CLIENT / '_rust/materialize.py',
    _CLIENT / '_rust/predict.py',
)
#: The files shared with the old route, covered for the functions named
#: here and for their module statements: the ledgered rules and what the
#: core route reaches through them, its entry points among them.
COVERED: dict[Path, tuple[str, ...]] = {
    _CLIENT / 'client.py': (
        'SDMClient._answer_in_parts',
        'SDMClient._prepare',
        'SDMClient._send',
        'SDMClient._spec',
        'SDMClient.predict',
        '_adapter_for',
        '_empty_answer',
        '_names_its_rows',
        '_predicts_on_the_core',
        '_report_absent',
        '_require_a_request',
        '_require_output_names',
    ),
    _CLIENT / 'query.py': (
        'PredictiveQuery._in_parts',
        'PredictiveQuery._refusal',
        'PredictiveQuery._resolve',
        'PredictiveQuery._validate_materialize_args',
        'PredictiveQuery.materialize',
        '_is_hashable',
    ),
    _CLIENT / 'materializers/relational.py': (
        'RelationalMaterializer.__init__',
        'RelationalMaterializer._check_limits',
        'RelationalMaterializer._check_shape',
        'RelationalMaterializer._gather',
        'RelationalMaterializer._refuse_a_name_the_wire_reserves',
        'RelationalMaterializer._run_mode_for',
        'RelationalMaterializer._unreserved',
        'RelationalMaterializer._warn_two_whole_numbers_are_being_asked_as_quantities',
        'RelationalMaterializer.materialize',
        '_columns_to_leave_out',
        '_drop_context_that_answers_the_question',
        '_entities_at_anchors',
        '_instances_matching',
        '_joins_nothing',
        '_positions',
        '_record_the_statistic',
        '_remove_instances',
        '_statistic_asked_for',
        '_tables_linked_to',
        '_tables_sent',
        '_warn_of_a_context_the_default_mode_capped',
        '_warn_of_columns_the_model_does_not_read',
        '_warn_of_relationships_that_join_nothing',
        '_warn_of_tables_left_out',
        '_warn_the_filter_chose_only_the_examples',
    ),
    # The ledgered rules' own bodies (T1.28's _preflight, model check and
    # merge, the decomposition), and Schema.from_source, the describe
    # route's entry point. T1.21d moved the schema rows into sdm-schema and
    # took out the helpers the core route no longer reaches; what building a
    # Schema still runs on it stays (its key marking, its two-ways warning,
    # which the schema bridge suppresses), and check_consistent, the old
    # route's and the tabular route's, which the self-tests seed.
    # From T1.23 (checkpoint 5, subagent P2-2): the live object's edit
    # semantics, which building a Schema runs on the core route and T1.24
    # will reach when it passes edits through (the ledger's row, now
    # T1.24's): the relationships' assignment and its checks, and the
    # table and column setters, which a new rule must not join unseen.
    _CLIENT / 'schema.py': (
        'Column.__setattr__',
        'Schema.__setattr__',
        'Schema._check_relationships',
        'Schema._mark_key_columns',
        'Schema._settle',
        'Schema.check_consistent',
        'Schema.from_source',
        '_TableShape.__setattr__',
        '_TableShape._adopt_key',
        '_TableShape._check_key',
        '_TableShape._check_time_column',
        '_TableShape._re_derive_key_columns',
        '_require_joinable',
        '_warn_two_ways_to_reach_one_table',
    ),
    _CLIENT / 'adapters/relational.py': ('RelationalAdapter._preflight',),
    _CLIENT / 'base.py': ('check_request_against_model',),
    _CLIENT / 'result.py': ('PredictionResult.merge',),
    _CLIENT / 'decompose.py': ('parts_of',),
    # The bridges' package: its module statements run on every core route.
    _CLIENT / '_rust/__init__.py': (),
}
LISTED = tuple(COVERED)
FROZEN = WHOLE + LISTED

MODULE = '<module>'
_FINGERPRINT = re.compile(r'[0-9a-f]{64}')

#: Each covered function's and skeleton's fingerprint, by name.
Measured = dict[str, str]
#: A baseline's entries by file and name: each its fingerprint.
Baseline = dict[str, dict[str, str]]


# --- measuring ---------------------------------------------------------------


def _is_docstring(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _plain(node: Any) -> Any:
    """A syntax tree as plain data: no positions, no docstrings, and no
    empty or absent field, so each Python version's spelling of a tree
    serializes alike."""
    if isinstance(node, ast.AST):
        out: list[Any] = [type(node).__name__]
        for name, value in ast.iter_fields(node):
            if name == 'body' and isinstance(value, list) and value:
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ) and _is_docstring(value[0]):
                    value = value[1:]
            if name in ('type_comment', 'kind') or value is None or value == []:
                continue
            out.append([name, _plain(value)])
        return out
    if isinstance(node, list):
        return [_plain(item) for item in node]
    if isinstance(node, (str, int, float, complex, bool, bytes)) or node is ...:
        return [type(node).__name__, repr(node)]
    return repr(node)


def fingerprint(nodes: Iterable[ast.AST]) -> str:
    """The SHA-256 of these nodes' plain form."""
    data = json.dumps([_plain(node) for node in nodes], separators=(',', ':'))
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


class _Skeleton(ast.NodeTransformer):
    """A module with every function's body replaced by ``pass``: its
    statements, classes, signatures and definitions in order and in scope,
    without what runs inside a function."""

    def _hollow(self, node: Any) -> Any:
        node.body = [ast.Pass()]
        return node

    visit_FunctionDef = _hollow
    visit_AsyncFunctionDef = _hollow


class Unreadable(ValueError):
    """A frozen file the freeze will not read as the interpreter would."""


#: The source encodings a frozen file may declare (PEP 263).
_ENCODINGS = frozenset({'utf-8', 'utf-8-sig'})


def _source_bytes(source: str | bytes, rel: Path) -> bytes:
    """A file's bytes, refused if it declares an encoding other than UTF-8:
    a cookie such as ``raw_unicode_escape`` or ``utf-7`` makes the
    interpreter run what a UTF-8 reader sees as a comment."""
    data = source.encode('utf-8') if isinstance(source, str) else source
    encoding, _ = tokenize.detect_encoding(io.BytesIO(data).readline)
    if encoding not in _ENCODINGS:
        raise Unreadable(
            f'{rel}: declares the source encoding {encoding}; a frozen file '
            'is UTF-8, so the freeze reads what the interpreter runs'
        )
    return data


def measure_text(text: str | bytes, rel: Path) -> Measured:
    """Every function and method of one file's source, and its skeleton as
    ``<module>``, parsed from its bytes as the interpreter parses them."""
    data = _source_bytes(text, rel)
    tree = ast.parse(data, filename=str(rel))
    # Whatever spells a cookie (one behind a lone CR, which
    # ``detect_encoding`` does not see, included): the file must read the
    # same as UTF-8 text, where any cookie is ignored, as it does as bytes.
    try:
        as_text = ast.parse(data.decode('utf-8-sig'), filename=str(rel))
    except (UnicodeDecodeError, SyntaxError):
        as_text = None
    if as_text is None or ast.dump(as_text) != ast.dump(tree):
        raise Unreadable(
            f'{rel}: reads otherwise as UTF-8 text than the interpreter reads '
            'it; a frozen file is UTF-8, so the freeze reads what the '
            'interpreter runs'
        )
    out: Measured = {}

    def visit(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f'{prefix}{node.name}'
                # A second definition of a name (under if and else) keeps
                # its own entry.
                while name in out:
                    name += "'"
                out[name] = fingerprint([node])
            elif isinstance(node, ast.ClassDef):
                visit(node.body, f'{prefix}{node.name}.')
            else:
                # A definition under if, try, with or a loop at this level.
                for field_name in ('body', 'orelse', 'finalbody'):
                    inner = getattr(node, field_name, None)
                    if isinstance(inner, list):
                        visit(inner, prefix)
                for handler in getattr(node, 'handlers', ()):
                    visit(handler.body, prefix)
                for case in getattr(node, 'cases', ()):
                    visit(case.body, prefix)

    visit(tree.body, '')
    skeleton = _Skeleton().visit(ast.parse(data, filename=str(rel)))
    statements = skeleton.body
    if statements and _is_docstring(statements[0]):
        statements = statements[1:]
    out[MODULE] = fingerprint(statements)
    return out


def measure_tree(root: Path) -> dict[Path, Measured | None]:
    """Each frozen file measured, or None where it does not exist."""
    return {
        rel: measure_text((root / rel).read_bytes(), rel)
        if (root / rel).is_file()
        else None
        for rel in FROZEN
    }


# --- the baseline ------------------------------------------------------------


def _covered(rel: Path, measured: Measured) -> list[str]:
    """The functions of a frozen file the freeze covers, ``<module>``
    included."""
    if rel in WHOLE:
        return sorted(measured)
    names = {*COVERED.get(rel, ()), MODULE}
    return sorted(name for name in names if name in measured)


def entries(measured: Mapping[Path, Measured | None]) -> Baseline:
    """The covered functions' fingerprints, by file and name."""
    return {
        str(rel): {name: found[name] for name in _covered(rel, found)}
        for rel, found in measured.items()
        if found is not None
    }


def render(baseline: Mapping[str, Mapping[str, str]]) -> str:
    data = {
        'format': FORMAT,
        'about': ABOUT,
        'files': {
            str(rel): {
                name: {'fingerprint': print_}
                for name, print_ in sorted(baseline[str(rel)].items())
            }
            for rel in FROZEN
            if str(rel) in baseline
        },
    }
    return json.dumps(data, indent=2, sort_keys=False) + '\n'


def _json(text: str, where: str) -> tuple[Any, list[str]]:
    try:
        return json.loads(text), []
    except ValueError as error:
        return None, [f'{where}: not JSON: {error}']
    except RecursionError:
        return None, [f'{where}: nested too deeply to read']


def load(
    text: str | None, where: str, *, base: bool = False
) -> tuple[Baseline | None, list[str]]:
    """A baseline read and checked, or the problems that stop it: its
    fingerprints, from format 3 or format 2 (whose counts are ignored, each
    entry still holding exactly the fields its format names). The base's is
    read whole: a file it froze that is no longer frozen keeps its entries,
    each of which is then gone and declared; a file frozen since holds
    nothing there."""
    if text is None:
        return None, [f'{where}: the baseline is missing']
    data, problems = _json(text, where)
    if problems:
        return None, problems
    if not isinstance(data, dict) or data.get('format') not in (
        FORMAT,
        FORMAT_2,
    ):
        return None, [
            f'{where}: not a baseline in format {FORMAT} (or {FORMAT_2})'
        ]
    if set(data) != {'format', 'about', 'files'}:
        return None, [
            f'{where}: holds {sorted(data)}, not format, about and files'
        ]
    fields = (
        {'fingerprint'}
        if data['format'] == FORMAT
        else {'fingerprint', *RETIRED}
    )
    files = data['files']
    if not isinstance(files, dict):
        return None, [f'{where}: files is not an object']
    expected = {str(rel) for rel in FROZEN}
    if not base:
        for name in sorted(expected - set(files)):
            problems.append(f'{where}: no entry for the frozen file {name}')
        for name in sorted(set(files) - expected):
            problems.append(f'{where}: {name} is not a frozen bridge file')
    out: Baseline = {}
    for rel in sorted(set(files) if base else expected & set(files)):
        functions = files[rel]
        if not isinstance(functions, dict) or not functions:
            problems.append(f'{where}: {rel} lists no function')
            continue
        out[rel] = {}
        for name, entry in functions.items():
            if not isinstance(entry, dict) or set(entry) != fields:
                problems.append(
                    f'{where}: {rel}::{name}: an entry holds exactly '
                    f'{", ".join(sorted(fields))} in {data["format"]}'
                )
            elif not (
                isinstance(entry['fingerprint'], str)
                and _FINGERPRINT.fullmatch(entry['fingerprint'])
            ):
                problems.append(
                    f'{where}: {rel}::{name}: fingerprint is not a SHA-256'
                )
            else:
                out[rel][name] = entry['fingerprint']
    return (None if problems else out), problems


def load_changes(
    text: str | None, where: str
) -> tuple[dict[str, tuple[str, str]] | None, list[str]]:
    """The declarations, ``{<file>::<function>: (reason, text)}``, checked."""
    if text is None:
        return None, [f'{where}: the declarations are missing']
    data, problems = _json(text, where)
    if problems:
        return None, problems
    if (
        not isinstance(data, dict)
        or data.get('format') != CHANGES_FORMAT
        or set(data) != {'format', 'about', 'changes'}
        or not isinstance(data['changes'], dict)
    ):
        return None, [
            f'{where}: not declarations in format {CHANGES_FORMAT} '
            '(format, about and changes)'
        ]
    out = {}
    for key, reason in data['changes'].items():
        if (
            not isinstance(reason, dict)
            or len(reason) != 1
            or next(iter(reason)) not in REASONS
            or not isinstance(next(iter(reason.values())), str)
            or not next(iter(reason.values())).strip()
        ):
            problems.append(
                f'{where}: {key}: a declaration holds one reason, '
                f'{", ".join(REASONS)}, as a line of text'
            )
            continue
        ((kind, said),) = reason.items()
        out[key] = (kind, said)
    return (None if problems else out), problems


# --- the checks --------------------------------------------------------------


def exactness(
    code: Mapping[Path, Measured | None], baseline: Baseline
) -> list[str]:
    """Where the code differs from the baseline."""
    problems = []
    covered = entries(code)
    remedy = 'rewrite the baseline (bridge_freeze.py --write)'
    for rel in FROZEN:
        found = code[rel]
        recorded = baseline.get(str(rel), {})
        if found is None:
            problems.append(f'{rel}: a frozen bridge file that does not exist')
            continue
        for name in COVERED.get(rel, ()):
            if name not in found:
                problems.append(
                    f'{rel}::{name}: COVERED names it, but the file defines '
                    'no such function: take it out of COVERED '
                    '(bridge_freeze.py) and declare the change'
                )
        now = covered[str(rel)]
        for name in sorted(set(recorded) - set(now)):
            problems.append(
                f'{rel}::{name}: in the baseline but not covered code: {remedy}'
            )
        for name in sorted(set(now) - set(recorded)):
            problems.append(
                f'{rel}::{name}: covered code the baseline does not hold: '
                f'{remedy}, and declare the change'
            )
        for name in sorted(set(now) & set(recorded)):
            if now[name] != recorded[name]:
                problems.append(
                    f'{rel}::{name}: differs from the baseline: {remedy}, '
                    'and declare the change'
                )
    return problems


def shadowing(root: Path) -> list[str]:
    """A directory beside a frozen module with the module's name: a package
    there would be imported in its place, leaving the frozen file dead."""
    return [
        f'{rel}: the directory {rel.with_suffix("")} beside it would shadow '
        'the frozen module; a frozen module keeps its name to itself'
        for rel in FROZEN
        if rel.name != '__init__.py' and (root / rel.with_suffix('')).exists()
    ]


@dataclass(frozen=True)
class Base:
    """What the tripwire reads at the merge base: its label, its baseline
    (None if it predates one) and its declarations (None where it has
    none)."""

    label: str
    baseline: str | None
    changes: str | None = None
    #: Why a ``main`` reason's text is not one for the entry (the
    #: arguments: its key, the text, and its fingerprint now, None when it
    #: is gone), or None when it is (:func:`main_judge`); without one, a
    #: ``main`` reason is refused.
    main: Callable[[str, str, str | None], str | None] | None = None


def ratchet(
    baseline: Baseline,
    base: Baseline,
    label: str,
    *,
    declared: Mapping[str, tuple[str, str]],
    spent: Mapping[str, tuple[str, str]],
    main: Callable[[str, str, str | None], str | None] | None = None,
) -> list[str]:
    """The tripwire against the merge base: every entry changed, new or
    gone is declared, once, with a reason that fits; ``main`` judges a
    ``main`` reason (:attr:`Base.main`)."""
    problems = []
    files = {str(rel): rel for rel in FROZEN}
    files.update({name: Path(name) for name in base if name not in files})
    changed: dict[str, str] = {}
    current: dict[str, str | None] = {}
    for name_of, rel in files.items():
        now = baseline.get(name_of, {})
        held = base.get(name_of, {})
        for name in sorted(set(now) | set(held)):
            key = f'{rel}::{name}'
            current[key] = now.get(name)
            if name not in held:
                changed[key] = 'new'
            elif name not in now:
                changed[key] = 'gone'
            elif now[name] != held[name]:
                changed[key] = 'changed'
    for key, how in sorted(changed.items()):
        reason = declared.get(key)
        if reason is None:
            problems.append(
                f'{key}: {how} against the base ({label}) and not declared in '
                f'{CHANGES}; a new product rule on the core route lands in '
                'Rust (E50); declare a moved rule, a refactor, or an entry '
                'gone from the freeze'
            )
        elif spent.get(key) == reason:
            problems.append(
                f'{key}: {how}, declared only by an entry the base ({label}) '
                'already holds, which is spent: declare this change anew'
            )
        elif how == 'gone' and reason[0] not in ('gone', 'moved', 'main'):
            problems.append(
                f'{key}: gone against the base ({label}), declared '
                f'{reason[0]}: an entry that left is declared gone (or moved, '
                'when its rule moved into Rust, or main, when main removed it)'
            )
        elif reason[0] == 'gone' and how != 'gone':
            problems.append(
                f'{key}: declared gone, but it is {how} against the base '
                f'({label})'
            )
        elif reason[0] == 'moved' and not _TASK.search(reason[1]):
            problems.append(
                f'{key}: a moved reason names the task that moves the rule '
                f'(T1.xx), not {reason[1]!r}'
            )
        elif reason[0] == 'main':
            why = (
                'no git history to judge it by'
                if main is None
                else main(key, reason[1], current[key])
            )
            if why is not None:
                problems.append(
                    f'{key}: {how}, declared main, which is not a change the '
                    f'candidate merged from main: {why}'
                )
    for key in sorted(set(declared) - set(changed)):
        if spent.get(key) != declared[key]:
            problems.append(
                f'{key}: declared in {CHANGES}, but unchanged against the base '
                f'({label}): a stale declaration; remove it'
            )
    return problems


def check(root: Path, base: Base | str) -> list[str]:
    """Every problem of the tree at ``root``. ``base`` is what the merge
    base holds, or why there is no base."""
    path = root / BASELINE
    text = path.read_text(encoding='utf-8') if path.is_file() else None
    baseline, problems = load(text, str(BASELINE))
    path = root / CHANGES
    declared, unreadable = load_changes(
        path.read_text(encoding='utf-8') if path.is_file() else None,
        str(CHANGES),
    )
    problems += unreadable
    if baseline is None or declared is None:
        return problems
    try:
        code = measure_tree(root)
    except SyntaxError as error:
        return [*problems, f'{error.filename}: does not parse: {error.msg}']
    except Unreadable as error:
        return [*problems, str(error)]
    problems += exactness(code, baseline)
    problems += shadowing(root)
    if isinstance(base, str):
        problems.append(f'no base to ratchet against: {base}')
        return problems
    if base.baseline is None:
        # The base predates the baseline: nothing to compare or declare.
        if declared:
            problems.append(
                f'{CHANGES}: declares {len(declared)} change(s), but the base '
                f'({base.label}) predates the baseline: nothing to declare'
            )
        return problems
    then, unreadable = load(
        base.baseline, f'{BASELINE} at the base ({base.label})', base=True
    )
    if then is None:
        return problems + unreadable
    spent: dict[str, tuple[str, str]] = {}
    if base.changes is not None:
        spent_or_none, unreadable = load_changes(
            base.changes, f'{CHANGES} at the base ({base.label})'
        )
        if spent_or_none is None:
            return problems + unreadable
        spent = spent_or_none
    problems += ratchet(
        baseline,
        then,
        base.label,
        declared=declared,
        spent=spent,
        main=base.main,
    )
    return problems


# --- git and the command line --------------------------------------------------


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ['git', '-C', str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def base_baseline(root: Path, ref: str) -> Base | str:
    """What the merge base of HEAD and ``ref`` holds, or why it cannot be
    read."""
    try:
        merge_base = _git(root, 'merge-base', 'HEAD', ref).strip()
    except (subprocess.CalledProcessError, OSError) as error:
        detail = getattr(error, 'stderr', '') or str(error)
        return f'no merge base of HEAD and {ref}: {detail.strip()}'

    def read(path: Path) -> str | None:
        try:
            _git(root, 'cat-file', '-e', f'{merge_base}:{path}')
        except subprocess.CalledProcessError:
            return None
        return _git(root, 'show', f'{merge_base}:{path}')

    return Base(
        label=f'{ref} at {merge_base[:12]}',
        baseline=read(BASELINE),
        changes=read(CHANGES),
        main=main_judge(root, merge_base),
    )


def main_judge(
    root: Path, merge_base: str
) -> Callable[[str, str, str | None], str | None]:
    """The judge of a ``main`` reason against ``merge_base``: why its text
    is not the full SHA and title of a commit of :data:`MAIN_REF` that
    HEAD holds, ``merge_base`` does not, and that changes the file named,
    or None when it is."""

    def ask(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ['git', '-C', str(root), *args],
            capture_output=True,
            text=True,
            check=False,
            stdin=subprocess.DEVNULL,
        )

    def entry_at(rev: str, rel: str, name: str) -> str | None:
        """The entry's fingerprint in ``rev``'s copy of the file, None when
        the file or the entry is not there."""
        if ask('cat-file', '-e', f'{rev}:{rel}').returncode != 0:
            return None
        data = subprocess.run(
            ['git', '-C', str(root), 'show', f'{rev}:{rel}'],
            capture_output=True,
            check=False,
            stdin=subprocess.DEVNULL,
        ).stdout
        try:
            return measure_text(data, Path(rel)).get(name)
        except (SyntaxError, Unreadable):
            return None

    def judge(key: str, text: str, now: str | None) -> str | None:
        rel, _, name = key.partition('::')
        found = _MAIN_REASON.fullmatch(text)
        if not found:
            return (
                'a main reason is the full SHA of the commit (40 lower-case '
                f'hex digits), a space and its title, not {text!r}'
            )
        sha, title = found.groups()
        if ask('cat-file', '-e', f'{sha}^{{commit}}').returncode != 0:
            return f'{sha} is no commit here'
        tip = ask('rev-parse', '--verify', '--quiet', f'{MAIN_REF}^{{commit}}')
        if tip.returncode != 0:
            return f'no {MAIN_REF} to read main from; fetch origin main'
        if ask(
            'merge-base', '--is-ancestor', sha, tip.stdout.strip()
        ).returncode:
            return (
                f'{sha[:12]} is not in main ({MAIN_REF} at '
                f'{tip.stdout.strip()[:12]}), the only branch merged so'
            )
        if ask('merge-base', '--is-ancestor', sha, 'HEAD').returncode:
            return f'{sha[:12]} is not merged into the candidate'
        if ask('merge-base', '--is-ancestor', sha, merge_base).returncode == 0:
            return (
                f'{sha[:12]} is already in the merge base {merge_base[:12]}, '
                'so this change did not bring it in'
            )
        # Only the package's own merges, on the phase's first-parent
        # history: a merge inside main (a pull request's) is never one.
        merges = ask(
            'rev-list',
            '--first-parent',
            '--merges',
            '--parents',
            f'{merge_base}..HEAD',
        ).stdout.splitlines()
        bringing = [
            (line.split()[0], line.split()[1], line.split()[2:])
            for line in merges
            if len(parents := line.split()[1:]) >= 2
            and ask('merge-base', '--is-ancestor', sha, parents[0]).returncode
            and any(
                ask('merge-base', '--is-ancestor', sha, other).returncode == 0
                for other in parents[1:]
            )
        ]
        if not bringing:
            return (
                f'{sha[:12]} is not brought in by a merge of the candidate '
                '(a commit main holds reaches the phase branch only by the '
                "package's merge of main)"
            )
        subject = ask('log', '-1', '--format=%s', sha).stdout.strip()
        if subject != title:
            return f'{sha[:12]} is titled {subject!r}, not {title!r}'
        parent = ask('rev-parse', '--verify', '--quiet', f'{sha}^1')
        if parent.returncode != 0:
            return f'{sha[:12]} is a root commit, which no merge brings in'
        changed = ask(
            'diff', '--name-only', '-z', parent.stdout.strip(), sha, '--'
        ).stdout.split('\0')
        if rel not in changed:
            return f'{sha[:12]} does not change {rel}'
        if entry_at(sha, rel, name) == entry_at(
            parent.stdout.strip(), rel, name
        ):
            return f'{sha[:12]} does not change the entry {name}'
        # The entry as the merge bringing the commit in holds it: main's
        # change merged with the phase's, which no main commit alone made.
        held = entry_at(merge_base, rel, name)
        merge, first, others = bringing[0]
        if entry_at(first, rel, name) != held:
            return (
                f'the entry {name} changed on the phase side before the merge '
                f"of main ({merge[:12]}): a change of the candidate's own "
                'is declared by its own reason, not main'
            )
        merged = entry_at(merge, rel, name)
        if merged != now:
            return (
                f'the entry {name} is not as the merge of main bringing '
                f'{sha[:12]} in ({merge[:12]}) holds it: the candidate '
                'changed it after the merge'
            )
        # Where git merges the file cleanly, the merge commit holds git's
        # merge of it; a conflict's resolution is review's, as a refactor.
        found = ask(
            'merge-tree', '--write-tree', '--name-only', first, *others
        ).stdout.split('\n')
        conflicted = set(
            found[1 : found.index('')] if '' in found else found[1:]
        )
        if found[0] and rel not in conflicted:
            if entry_at(found[0], rel, name) != merged:
                return (
                    f'the entry {name} in the merge of main ({merge[:12]}) '
                    "is not git's clean merge of its parents: the merge "
                    'commit itself changed it'
                )
        return None

    return judge


def base_ref(
    given: str | None = None, environ: Mapping[str, str] | None = None
) -> str:
    """The branch whose merge base with HEAD holds the baseline: ``given``
    (``--base``), else ``BASE_ENV`` when it is set, even blank, else
    :func:`phase_base`."""
    if given is not None:
        return given
    environ = os.environ if environ is None else environ
    if BASE_ENV in environ:
        return environ[BASE_ENV]
    return phase_base(environ)


def phase_base(environ: Mapping[str, str]) -> str:
    """``origin/`` and the phase branch ``phases.phase_branch`` names:
    ``SDM_PHASE_BRANCH``, else the phase of CI's branch
    (``GITHUB_REF_NAME``) or of the branch checked out, else the phase in
    progress."""
    branch = environ.get('GITHUB_REF_NAME') or phases.checked_out()
    return 'origin/' + phases.phase_branch(branch=branch, environ=environ)


def problems(root: Path = REPO_ROOT, ref: str | None = None) -> list[str]:
    """The check as ``l0_deps.py`` runs it, against ``base_ref(ref)``; a
    phase setting that is no phase branch is a problem, named."""
    try:
        ref = base_ref(ref)
    except phases.PhaseError as error:
        return [f'the base branch: {error}']
    if not ref.strip():
        return [f'the base branch is blank (--base or {BASE_ENV}): name one']
    return check(root, base_baseline(root, ref))


def write(root: Path) -> list[str]:
    """Rewrites the baseline from the code, in this copy's format; what
    changed."""
    path = root / BASELINE
    old: Baseline = {}
    if path.is_file():
        loaded, _ = load(path.read_text(encoding='utf-8'), str(BASELINE))
        old = loaded or {}
    code = measure_tree(root)
    missing = [str(rel) for rel, found in code.items() if found is None]
    if missing:
        raise SystemExit(f'frozen files missing: {missing}')
    new = entries(code)
    path.write_text(render(new), encoding='utf-8')
    changes = []
    for rel in FROZEN:
        before = old.get(str(rel), {})
        after = new.get(str(rel), {})
        for name in sorted(set(before) | set(after)):
            if before.get(name) != after.get(name):
                changes.append(f'{rel}::{name}')
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    parser.add_argument(
        '--base',
        default=None,
        help='the branch whose merge base with HEAD holds the baseline to '
        f'compare with (default: ${BASE_ENV}, else {BASE_REF})',
    )
    parser.add_argument(
        '--write',
        action='store_true',
        help='rewrite the baseline from the code, then check',
    )
    args = parser.parse_args(argv)
    if args.write:
        try:
            changes = write(args.root)
        except Unreadable as error:
            print(f'freeze: {error}\nfreeze: FAIL')
            return 1
        for change in changes:
            print(f'wrote {change}')
    found = problems(args.root, args.base)
    for problem in found:
        print(f'freeze: {problem}')
    print(f'freeze: {"FAIL" if found else "ok"}')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
