#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bridge freeze: no change to the core route's Python bridge is silent.

Decision E50 (the owner, 2026-10-10): a new product rule on a core route
lands in Rust, not in the Python bridge. This guard holds it over the files
that keep a row of the design's "Host-held semantics" ledger:

- the ``_rust/`` bridges, ``_rust/schema.py``, ``_rust/materialize.py`` and
  ``_rust/predict.py``, whole: every function and method, and the module's
  own statements as ``<module>``;
- the files shared with the old route, for the functions ``COVERED`` names
  and their module statements: in ``client.py``, ``query.py`` and
  ``materializers/relational.py`` the ledgered functions and those the core
  route reaches through them; in ``schema.py``, ``adapters/relational.py``,
  ``base.py``, ``result.py`` and ``decompose.py`` the ledgered rules' own
  bodies; and the entry points (``SDMClient.predict``,
  ``PredictiveQuery.materialize``, ``RelationalMaterializer.materialize``,
  ``Schema.from_source``) with ``RelationalMaterializer.__init__``, which
  holds the index once-flag. ``COVERED`` is this file's, so coverage changes
  only by a reviewed edit here.

The tripwire, the primary gate (T1.10e fix round 2, a change of method):
the committed baseline, ``bridge_freeze.json``, holds a fingerprint of each
covered function and of each file's module statements: the SHA-256 of its
syntax tree with no positions and no docstrings, serialized the same way on
every Python version. Against the merge base of HEAD and
``origin/rust-core-phase-1``, every entry whose fingerprint differs, and
every entry that is new or gone (a ``_rust/`` bridge's definitions are its
entries, so a function added, removed or renamed there is one), must be
declared in ``bridge_changes.json`` under ``<file>::<qualified name>``, with
one reason:

- ``{"moved": "<ledger row or task>"}``, a rule moved into Rust;
- ``{"allowance": "E##"}``, growth the owner approved: an ``ALLOWANCES``
  entry for that function under that row, as below;
- ``{"refactor": "<one line>"}``, an edit that keeps behaviour, a rename
  included.

It fails on an undeclared change; on a declaration of an entry that did not
change (stale); on a change declared only by an entry the merge base's
declarations already hold (spent: a later change states its own reason);
and on a ``moved`` or ``refactor`` that raises a count. So no change to
frozen code is silent, and review judges each reason's direction. A file
leaving ``FROZEN`` is not compared; one joining it has every entry new.

The counts, the secondary check: per entry the baseline also records its
rule sites,

- ``raise``: ``raise`` and ``assert`` statements;
- ``warn``: calls of ``warnings.warn`` and ``warnings.warn_explicit``;
- ``branch``: ``if`` (each ``elif`` too), conditional expressions, ``while``,
  ``for``, ``match`` cases, comprehension conditions, ``except`` handlers,
  ``contextlib.suppress``, and each operand after the first of ``and`` and
  ``or``;

and ``calls``: the first-party functions and classes it calls (dotted, a
covered one left out) and the methods it calls on objects it cannot resolve
(``.name``, listed for review, not gated). Against the merge base's entry,
or for an entry the base's baseline does not hold the function's body
measured in the base's source, or nothing, a count or a dotted first-party
call may rise only up to an ``ALLOWANCES`` entry that the base's copy of
this file does not already hold (spent), under a row of the base's
decisions.md whose status reads "approved by owner <date>" and which names
``<file>::<qualified name>`` in backticks. The baseline must match the code
exactly (``--write`` rewrites it).

Missing or malformed evidence fails: no baseline or declarations, either in
another format, a field missing or retyped, an unreadable base, no merge
base. A base that predates the baseline has nothing to compare, and the
declarations must then be empty.

What it does not see: a rule added inside a first-party helper that
``COVERED`` does not name and that a covered function already called at the
merge base. Review holds that, as it holds the ledger's owners and each
declared reason.

When a ledger row moves into Rust, the package deletes the row and the
bridge's rule, takes a function the core route no longer reaches out of
``COVERED``, declares each changed entry ``moved``, and runs ``--write``.

``l0_deps.py`` runs this check, so it runs where that does: ``prepush.sh``'s
cheap tier and CI's ``rust-lint``; ``selftest.py`` seeds a fault for each
rule.

    scripts/rust_core/bridge_freeze.py [--write] [--base REF]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE = Path('scripts/rust_core/bridge_freeze.json')
CHANGES = Path('scripts/rust_core/bridge_changes.json')
DECISIONS = Path('docs/design/rust-core/phase-1/decisions.md')
BASE_REF = 'origin/rust-core-phase-1'
FORMAT = 'sdm-bridge-freeze/2'
CHANGES_FORMAT = 'sdm-bridge-changes/1'
ABOUT = (
    'The bridge freeze (decision E50): per covered function of the core '
    "route's Python bridge, its fingerprint, its rule sites and what it "
    'calls. Regenerate with scripts/rust_core/bridge_freeze.py --write; a '
    'change against the merge base is declared in bridge_changes.json.'
)
CHANGES_ABOUT = (
    'Each change to frozen bridge code against the merge base, with one '
    'reason: moved (a rule moved into Rust), allowance (an owner-approved '
    'ALLOWANCES row) or refactor (behaviour kept). Entries the merge base '
    'already holds are spent and may be pruned.'
)
REASONS = ('moved', 'allowance', 'refactor')

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
        'RelationalMaterializer._check_entity_count',
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
    # The ledgered rules' own bodies (T1.21d's schema rows, T1.28's
    # _preflight, model check and merge, the decomposition), and
    # Schema.from_source, the describe route's entry point.
    _CLIENT / 'schema.py': (
        'Schema._mark_key_columns',
        'Schema.check_consistent',
        'Schema.from_source',
        '_infer_relationships',
        '_joinable_inferred_links',
        '_key_by_the_referenced_column',
        '_settle_edit_stamps',
        '_warn_how_time_columns_were_inferred',
        '_warn_two_ways_to_reach_one_table',
    ),
    _CLIENT / 'adapters/relational.py': ('RelationalAdapter._preflight',),
    _CLIENT / 'base.py': ('check_request_against_model',),
    _CLIENT / 'result.py': ('PredictionResult.merge',),
    _CLIENT / 'decompose.py': ('parts_of',),
}
LISTED = tuple(COVERED)
FROZEN = WHOLE + LISTED

#: The first-party top-level packages, whose functions are helpers.
FIRST_PARTY = frozenset(
    {
        'sdm_client',
        'predictive_query',
        'sdm_relational_engine',
        'sdm_api',
        'sdm_connectors',
    }
)

#: Growth the owner approved, by decisions row: each maps a covered
#: function, ``<file>::<qualified name>``, to the entry the baseline may hold
#: for it (``raise``, ``warn``, ``branch``, ``calls``), above what the
#: merge base holds. It needs a row the merge base's decisions.md already
#: holds, approved by the owner and naming ``<file>::<qualified name>`` in
#: backticks, so the row lands before the growth; an allowance the merge
#: base's copy of this file already holds grants nothing, so a consumed one
#: cannot grow the function again after a shrink. The change is declared
#: ``{"allowance": "<row>"}`` in bridge_changes.json.
ALLOWANCES: dict[str, dict[str, dict[str, Any]]] = {}

KINDS = ('raise', 'warn', 'branch')
MODULE = '<module>'
_WARN = frozenset({'warnings.warn', 'warnings.warn_explicit'})
_SUPPRESS = frozenset({'contextlib.suppress'})
_OWNER_APPROVED = re.compile(r'approved by owner \d{4}-\d{2}-\d{2}')
_FINGERPRINT = re.compile(r'[0-9a-f]{64}')


@dataclass
class Entry:
    """A covered function's fingerprint, rule sites and calls."""

    fingerprint: str = ''
    counts: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(KINDS, 0)
    )
    calls: set[str] = field(default_factory=set)

    def as_json(self) -> dict[str, Any]:
        return {
            'fingerprint': self.fingerprint,
            **self.counts,
            'calls': sorted(self.calls),
        }


Measured = dict[str, Entry]
Baseline = dict[str, dict[str, dict[str, Any]]]


# --- measuring ---------------------------------------------------------------


def module_name(rel: Path) -> str:
    """The dotted module of a source file under a package's ``src``."""
    parts = rel.with_suffix('').parts
    return '.'.join(parts[parts.index('src') + 1 :])


def _imports(nodes: Iterable[ast.AST], package: str) -> dict[str, str]:
    """The names these import statements bind, each to its dotted target."""
    bound: dict[str, str] = {}
    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    bound[alias.asname] = alias.name
                else:
                    head = alias.name.split('.')[0]
                    bound[head] = head
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level:
                parent = package.split('.')
                parent = parent[: len(parent) - node.level]
                base = '.'.join([*parent, base] if base else parent)
            for alias in node.names:
                bound[alias.asname or alias.name] = f'{base}.{alias.name}'
    return bound


def _dotted(node: ast.expr) -> list[str] | None:
    """``a.b.c`` as ``['a', 'b', 'c']``, or None if it is not a name chain."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id, *reversed(parts)]
    return None


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


class _Measure:
    """Counts one function's rule sites and resolves what it calls."""

    def __init__(
        self,
        module: str,
        top_level: frozenset[str],
        methods: Mapping[str, frozenset[str]],
        module_imports: dict[str, str],
    ) -> None:
        self.module = module
        self.top_level = top_level
        self.methods = methods
        self.module_imports = module_imports

    def entry(self, body: list[ast.AST], owner: str | None) -> Entry:
        entry = Entry(fingerprint=fingerprint(body))
        nodes = [n for stmt in body for n in ast.walk(stmt)]
        imports = {**self.module_imports, **_imports(nodes, self.module)}
        # Names this code binds itself, which shadow an import or a
        # module-level definition of the same name.
        local = {
            n.name
            for n in nodes
            if isinstance(
                n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
        }
        local |= {
            n.id
            for n in nodes
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
        }
        local |= {n.arg for n in nodes if isinstance(n, ast.arg)}
        for node in nodes:
            if isinstance(node, (ast.Raise, ast.Assert)):
                entry.counts['raise'] += 1
            elif isinstance(
                node,
                (
                    ast.If,
                    ast.IfExp,
                    ast.While,
                    ast.For,
                    ast.AsyncFor,
                    ast.match_case,
                    ast.ExceptHandler,
                ),
            ):
                entry.counts['branch'] += 1
            elif isinstance(node, ast.BoolOp):
                entry.counts['branch'] += len(node.values) - 1
            elif isinstance(node, ast.comprehension):
                entry.counts['branch'] += len(node.ifs)
            elif isinstance(node, ast.Call):
                target = self.resolve(node.func, imports, local, owner)
                if target in _WARN:
                    entry.counts['warn'] += 1
                elif target in _SUPPRESS:
                    entry.counts['branch'] += 1
                elif target is not None:
                    entry.calls.add(target)
        return entry

    def resolve(
        self,
        func: ast.expr,
        imports: dict[str, str],
        local: set[str],
        owner: str | None,
    ) -> str | None:
        """What a call reaches: a dotted name, ``.name`` for a method of an
        object it cannot resolve, or None for what is never a helper."""
        chain = _dotted(func)
        if chain is None:
            if isinstance(func, ast.Attribute):
                return f'.{func.attr}'
            return None
        head, rest = chain[0], chain[1:]
        if head in ('self', 'cls') and owner is not None and len(rest) == 1:
            if rest[0] in self.methods.get(owner, frozenset()):
                return f'{self.module}.{owner}.{rest[0]}'
        elif head not in local:
            if head in imports:
                target = '.'.join([imports[head], *rest])
                if (
                    target in _WARN | _SUPPRESS
                    or target.split('.')[0] in FIRST_PARTY
                ):
                    return target
                return None
            if head in self.top_level:
                return '.'.join([self.module, head, *rest])
        if not rest:
            return None  # a builtin, a parameter or a local
        return f'.{rest[-1]}'


def measure_text(text: str, rel: Path) -> Measured:
    """Every function and method of one file's source, and its module
    statements."""
    tree = ast.parse(text, filename=str(rel))
    module = module_name(rel)
    top_level = frozenset(
        n.name
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    )
    methods = {
        n.name: frozenset(
            m.name
            for m in n.body
            if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
        for n in tree.body
        if isinstance(n, ast.ClassDef)
    }
    module_imports = _imports(
        (n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))),
        module,
    )
    # Imports under `if TYPE_CHECKING:` and other module-level blocks.
    for node in tree.body:
        if isinstance(node, (ast.If, ast.Try)):
            module_imports.update(_imports(ast.walk(node), module))
    measuring = _Measure(module, top_level, methods, module_imports)
    out: Measured = {}
    loose: list[ast.AST] = []

    def visit(body: list[ast.stmt], prefix: str, owner: str | None) -> None:
        for index, node in enumerate(body):
            if index == 0 and _is_docstring(node):
                continue
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f'{prefix}{node.name}'
                out[name] = measuring.entry([node], owner)
            elif isinstance(node, ast.ClassDef):
                # A class's name, decorators, bases and keywords run with
                # the module.
                loose.append(ast.Constant(value=f'class {prefix}{node.name}'))
                loose.extend(
                    [*node.decorator_list, *node.bases, *node.keywords]
                )
                visit(node.body, f'{prefix}{node.name}.', node.name)
            else:
                loose.append(node)

    visit(tree.body, '', None)
    out[MODULE] = measuring.entry(loose, None)
    return out


def measure_tree(root: Path) -> dict[Path, Measured | None]:
    """Each frozen file measured, or None where it does not exist."""
    return {
        rel: measure_text((root / rel).read_text(encoding='utf-8'), rel)
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


def _covered_targets() -> set[str]:
    """Covered functions as dotted names: a call that resolves to one is
    the freeze's own, not a helper."""
    return {
        f'{module_name(rel)}.{name}'
        for rel, names in COVERED.items()
        for name in names
    }


def _strip(
    entry: Entry,
    dotted: set[str],
    measured_whole: Mapping[Path, Measured | None],
) -> Entry:
    whole = {
        f'{module_name(rel)}.{name}'
        for rel, found in measured_whole.items()
        if found is not None and rel in WHOLE
        for name in found
        if name != MODULE
    }
    calls = {call for call in entry.calls if call not in dotted | whole}
    return Entry(entry.fingerprint, dict(entry.counts), calls)


def entries(
    measured: Mapping[Path, Measured | None],
) -> dict[Path, dict[str, Entry]]:
    """The covered functions' entries, their calls of covered functions
    left out."""
    dotted = _covered_targets()
    out: dict[Path, dict[str, Entry]] = {}
    for rel, found in measured.items():
        if found is None:
            continue
        out[rel] = {
            name: _strip(found[name], dotted, measured)
            for name in _covered(rel, found)
        }
    return out


def render(baseline: Mapping[Path, Mapping[str, Entry]]) -> str:
    data = {
        'format': FORMAT,
        'about': ABOUT,
        'files': {
            str(rel): {
                name: entry.as_json()
                for name, entry in sorted(baseline[rel].items())
            }
            for rel in FROZEN
            if rel in baseline
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
    """A baseline read and checked, or the problems that stop it. The
    base's is read for the files frozen now: one no longer frozen is left
    out, and one newly frozen holds nothing."""
    if text is None:
        return None, [f'{where}: the baseline is missing']
    data, problems = _json(text, where)
    if problems:
        return None, problems
    if not isinstance(data, dict) or data.get('format') != FORMAT:
        return None, [f'{where}: not a baseline in format {FORMAT}']
    if set(data) != {'format', 'about', 'files'}:
        return None, [
            f'{where}: holds {sorted(data)}, not format, about and files'
        ]
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
    for rel in sorted(expected & set(files)):
        functions = files[rel]
        if not isinstance(functions, dict) or not functions:
            problems.append(f'{where}: {rel} lists no function')
            continue
        out[rel] = {}
        for name, entry in functions.items():
            problem = _entry_problem(entry, fingerprinted=True)
            if problem:
                problems.append(f'{where}: {rel}::{name}: {problem}')
            else:
                out[rel][name] = entry
    return (None if problems else out), problems


def _entry_problem(entry: Any, *, fingerprinted: bool = False) -> str | None:
    fields = {*KINDS, 'calls'} | ({'fingerprint'} if fingerprinted else set())
    if not isinstance(entry, dict) or set(entry) != fields:
        return f'an entry holds exactly {", ".join(sorted(fields))}'
    if fingerprinted and not (
        isinstance(entry['fingerprint'], str)
        and _FINGERPRINT.fullmatch(entry['fingerprint'])
    ):
        return 'fingerprint is not a SHA-256'
    for kind in KINDS:
        value = entry[kind]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return f'{kind} is not a count'
    calls = entry['calls']
    if (
        not isinstance(calls, list)
        or not all(isinstance(c, str) and c for c in calls)
        or calls != sorted(set(calls))
    ):
        return 'calls is not a sorted list of distinct names'
    return None


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
        now = covered[rel]
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
            if now[name].as_json() != recorded[name]:
                problems.append(
                    f'{rel}::{name}: differs from the baseline: {remedy}, '
                    'and declare the change'
                )
    return problems


def decision_rows(text: str) -> dict[str, tuple[str, str]]:
    """decisions.md's rows by ID: each row's text and its status cell."""
    rows = {}
    for line in text.splitlines():
        found = re.match(r'\| ([A-Z][A-Z0-9-]*) \|', line)
        if found:
            cells = [
                cell.strip() for cell in line.strip().strip('|').split('|')
            ]
            rows[found.group(1)] = (line, cells[-1])
    return rows


Allowances = Mapping[str, Mapping[str, Mapping[str, Any]]]


@dataclass(frozen=True)
class Base:
    """What the ratchet reads at the merge base: its label, its baseline
    (None if it predates one), its declarations, decisions.md and copy of
    this file (None where it has none), and a reader of its sources."""

    label: str
    baseline: str | None
    decisions: str
    script: str | None
    changes: str | None = None
    source: Callable[[Path], str | None] = lambda rel: None


def held_allowances(
    script: str | None,
) -> tuple[set[tuple[str, str]], list[str]]:
    """The ``(decision, <file>::<function>)`` pairs the base's copy of this
    file already grants, or why they cannot be read."""
    if script is None:
        return set(), []
    try:
        tree = ast.parse(script)
    except SyntaxError as error:
        return set(), [f"the base's {SCRIPT} does not parse: {error.msg}"]
    for node in tree.body:
        target = (
            node.target
            if isinstance(node, ast.AnnAssign)
            else node.targets[0]
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            else None
        )
        if isinstance(target, ast.Name) and target.id == 'ALLOWANCES':
            try:
                value = ast.literal_eval(node.value)  # type: ignore[arg-type]
            except ValueError:
                break
            if isinstance(value, dict) and all(
                isinstance(v, dict) for v in value.values()
            ):
                return {(d, str(k)) for d, v in value.items() for k in v}, []
            break
    return set(), [f"the base's {SCRIPT}: ALLOWANCES is not a literal dict"]


def allowance_problems(
    allowances: Allowances,
    rows: Mapping[str, tuple[str, str]],
    held: set[tuple[str, str]] = frozenset(),  # type: ignore[assignment]
) -> tuple[list[str], dict[str, dict[str, Mapping[str, Any]]]]:
    """Each allowance must sit under an owner-approved row of the base's
    decisions.md that names its ``<file>::<function>``, and hold a
    well-formed entry for a frozen file: the problems, and the entries that
    pass, by ``<file>::<function>`` and decision. A refused allowance grants
    nothing, nor does one the base already holds (``held``): it was spent."""
    problems = []
    valid: dict[str, dict[str, Mapping[str, Any]]] = {}
    files = {str(rel) for rel in FROZEN}
    for decision, granted in sorted(allowances.items()):
        where = f'ALLOWANCES[{decision}]'
        if decision not in rows:
            problems.append(
                f"{where}: not a row of the merge base's decisions.md (the "
                'row lands before the growth it allows)'
            )
            continue
        text, status = rows[decision]
        if not _OWNER_APPROVED.search(status):
            problems.append(f'{where}: the row is not approved by the owner')
            continue
        if not isinstance(granted, Mapping) or not granted:
            problems.append(f'{where}: grants nothing')
            continue
        for key, entry in sorted(granted.items()):
            rel, _, name = str(key).partition('::')
            problem = _entry_problem(entry)
            if rel not in files or not name:
                problem = 'not <frozen file>::<function>'
            elif f'`{key}`' not in text:
                problem = f'{decision} does not name `{key}`'
            if problem:
                problems.append(f'{where}[{key}]: {problem}')
            elif (decision, str(key)) not in held:
                valid.setdefault(str(key), {})[decision] = entry
    return problems, valid


_NOTHING: dict[str, Any] = {**dict.fromkeys(KINDS, 0), 'calls': []}


def _then(
    base: Baseline,
    rel: Path,
    name: str,
    measured_at_base: Mapping[Path, Measured | None],
) -> dict[str, Any] | None:
    """The entry at the merge base: the base baseline's, else the function's
    body measured in the base's source, else None."""
    held = base.get(str(rel), {}).get(name)
    if held is not None:
        return held
    found = measured_at_base.get(rel)
    if found is not None and name in found:
        return _strip(
            found[name], _covered_targets(), measured_at_base
        ).as_json()
    return None


def ratchet(
    baseline: Baseline,
    base: Baseline,
    label: str,
    *,
    declared: Mapping[str, tuple[str, str]],
    spent: Mapping[str, tuple[str, str]],
    granted: Mapping[str, Mapping[str, Mapping[str, Any]]],
    measured_at_base: Mapping[Path, Measured | None],
) -> list[str]:
    """The tripwire and the counts against the merge base."""
    problems = []
    remedy = (
        'growth needs an ALLOWANCES entry under an owner-approved decisions '
        'row of the merge base naming <file>::<function>, declared '
        '{"allowance": "<row>"} (E50)'
    )
    changed: dict[str, str] = {}
    for rel in FROZEN:
        now = baseline.get(str(rel), {})
        held = base.get(str(rel), {})
        for name in sorted(set(now) | set(held)):
            key = f'{rel}::{name}'
            if name not in held:
                changed[key] = 'new'
            elif name not in now:
                changed[key] = 'gone'
            elif now[name]['fingerprint'] != held[name]['fingerprint']:
                changed[key] = 'changed'
    for key, how in sorted(changed.items()):
        reason = declared.get(key)
        if reason is None:
            problems.append(
                f'{key}: {how} against the base ({label}) and not declared in '
                f'{CHANGES}; a new product rule on the core route lands in '
                'Rust (E50); declare a moved rule, an owner-approved '
                'allowance or a refactor'
            )
        elif spent.get(key) == reason:
            problems.append(
                f'{key}: {how}, declared only by an entry the base ({label}) '
                'already holds, which is spent: declare this change anew'
            )
    for key in sorted(set(declared) - set(changed)):
        if spent.get(key) != declared[key]:
            problems.append(
                f'{key}: declared in {CHANGES}, but unchanged against the base '
                f'({label}): a stale declaration; remove it'
            )
    # The counts, per file: what the covered code holds now against what
    # the same functions held at the base (the base's entries, and a newly
    # covered function's body as the base's source has it), so a rename or
    # an extraction keeps the totals, and growth needs an allowance.
    for rel in FROZEN:
        now = baseline.get(str(rel), {})
        if not now:
            continue
        then_entries = dict(base.get(str(rel), {}))
        for name in now:
            if name not in then_entries:
                found = _then(base, rel, name, measured_at_base)
                if found is not None:
                    then_entries[name] = found
        grants: dict[str, Mapping[str, Any]] = {}
        for name in now:
            key = f'{rel}::{name}'
            kind_said = declared.get(key)
            if kind_said is None or kind_said[0] != 'allowance':
                continue
            grant = granted.get(key, {}).get(kind_said[1])
            if grant is None:
                problems.append(
                    f'{key}: declared under allowance {kind_said[1]}, which '
                    'grants it nothing (no valid ALLOWANCES entry for it '
                    'under that row)'
                )
            else:
                grants[name] = grant
        for kind in KINDS:
            held_total = sum(e[kind] for e in then_entries.values())
            extra = sum(
                max(0, g[kind] - then_entries.get(name, _NOTHING)[kind])
                for name, g in grants.items()
            )
            now_total = sum(e[kind] for e in now.values())
            if now_total > held_total + extra:
                problems.append(
                    f'{rel}: the covered code raises {kind} from {held_total} '
                    f'(the base, {label}) to {now_total}; {remedy}'
                )
        held_calls = set().union(
            *(e['calls'] for e in then_entries.values()),
            *(g['calls'] for g in grants.values()),
        )
        for name, entry in sorted(now.items()):
            for call in sorted(set(entry['calls']) - held_calls):
                if not call.startswith('.'):
                    problems.append(
                        f'{rel}::{name}: the baseline adds a call of {call}, '
                        f'a helper the base ({label}) does not list; {remedy}'
                    )
    return problems


def check(
    root: Path,
    base: Base | str,
    *,
    allowances: Allowances = ALLOWANCES,
) -> list[str]:
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
    held: set[tuple[str, str]] = set()
    if isinstance(base, Base):
        held, unreadable = held_allowances(base.script)
        problems += unreadable
    rows = decision_rows(base.decisions if isinstance(base, Base) else '')
    refused, granted = allowance_problems(allowances, rows, held)
    problems += refused
    if baseline is None or declared is None:
        return problems
    try:
        code = measure_tree(root)
    except SyntaxError as error:
        return [*problems, f'{error.filename}: does not parse: {error.msg}']
    problems += exactness(code, baseline)
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
    measured_at_base: dict[Path, Measured | None] = {}
    for rel in FROZEN:
        source = base.source(rel)
        try:
            measured_at_base[rel] = (
                None if source is None else measure_text(source, rel)
            )
        except SyntaxError:
            measured_at_base[rel] = None
    problems += ratchet(
        baseline,
        then,
        base.label,
        declared=declared,
        spent=spent,
        granted=granted,
        measured_at_base=measured_at_base,
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
        decisions=read(DECISIONS) or '',
        script=read(SCRIPT),
        changes=read(CHANGES),
        source=read,
    )


def problems(root: Path = REPO_ROOT, ref: str = BASE_REF) -> list[str]:
    """The check as ``l0_deps.py`` runs it."""
    return check(root, base_baseline(root, ref))


def write(root: Path) -> list[str]:
    """Rewrites the baseline from the code; what changed."""
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
        after = {name: e.as_json() for name, e in new[rel].items()}
        for name in sorted(set(before) | set(after)):
            if before.get(name) != after.get(name):
                changes.append(f'{rel}::{name}')
    return changes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    parser.add_argument(
        '--base',
        default=BASE_REF,
        help='the branch whose merge base with HEAD holds the baseline to '
        'compare with',
    )
    parser.add_argument(
        '--write',
        action='store_true',
        help='rewrite the baseline from the code, then check',
    )
    args = parser.parse_args(argv)
    if args.write:
        for change in write(args.root):
            print(f'wrote {change}')
    found = problems(args.root, args.base)
    for problem in found:
        print(f'freeze: {problem}')
    print(f'freeze: {"FAIL" if found else "ok"}')
    return 1 if found else 0


if __name__ == '__main__':
    sys.exit(main())
