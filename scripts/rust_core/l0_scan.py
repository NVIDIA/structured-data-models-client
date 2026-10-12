#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The L0 scans of the Rust core (Phase 1 design, task T1.10c).

Each check reads the tree and prints every violation it finds; the script
exits non-zero when any check found one. ``l0_deps.py`` checks the crate
graph; these check the text:

- ``vocabulary``: no Python-emulation vocabulary in the shipped code of the
  domain crates and ``sdm-host`` outside the modules on
  ``emulation-allowlist.txt`` (decisions H21, E18), which must all sit in the
  parser crate. Comments and ``#[cfg(test)]`` modules are not read; an
  identifier made of the words ``py``, ``python``, ``pandas``, ``numpy`` and
  the like, a ``py``-prefixed Python type name (``pyint``, ``pystr``), a
  dunder name, or a string literal naming Python, pandas, NumPy or pyarrow,
  is emulation vocabulary. The public names the allowlisted modules define
  may be used anywhere in their crate. The parser crate's name is the
  checked tree's: its own copy of this file's ``PARSER_CRATE``, read as a
  string literal, never run, with ``CORE_CRATES``' parser entry under that
  name (``parser_crates``), so a gate running its checkpoint's copy over a
  candidate that renames the crate judges the candidate's name (a bug fix at
  T1.28w, E56(a)); a copy missing, unparsed, not a literal, or naming
  another crate is refused.
- ``sql``: no SQL fragment and no provider match in ``sdm-schema``,
  ``sdm-engine``, ``sdm-request`` or ``sdm-core`` (migration plan,
  "Ownership"; ``sdm-core`` from T1.21c). It is a
  lexical scan over what the literals stand for (escapes such as ``\\n``,
  ``\\t``, ``\\x..`` and ``\\u{..}`` decoded, line continuations
  joined): the string literals of each statement are read together, so
  a clause split across one ``format!``, ``concat!``, ``join`` or ``+`` is
  seen, as is a provider's name as a whole literal, an identifier naming a
  dialect, a backend or SQL, and ``include_str!``. Only SQL's unambiguous
  shapes count (``_SQL_STRONG``), such as a ``select … from`` with a clause
  or ``cast(… as type)``, and only outside diagnostic text, the message
  arguments of a verified error helper or a diagnostic macro
  (``diagnostic_calls``), which is exempt entirely; ``#[cfg(test)]`` modules, in the file or their
  own, are not read (decision E28, approved by owner 2026-10-09). It
  cannot see SQL assembled across statements (pieces bound to names
  first), from data that is not a literal in these crates (another crate's
  constants, values built at run time), from bytes, or shapes it does not
  hold; review covers those. The structural guard beside it is
  ``sdm-contract``'s ``tests/data_vocabulary.rs``: no field of the
  data-request vocabulary holds an expression or a SQL string.
- ``dialects``: no dialect knowledge in the shared crates other than
  ``sdm-data``, whose dialects they are, and the dev-only ``sdm-testkit``
  (decisions E50 and E54): no identifier and no word of a string literal in
  their non-test source names a term of a short, reviewed deny-list
  (``DIALECT_NAMES``, matched in any case inside the identifier or word;
  ``DIALECT_WORDS``, matched as whole words, a term of two words as the
  two in a row, since neutral names hold them: ``Filter::NotNull`` is not
  DuckDB's ``notnull``, nor ``pragmatic`` its ``pragma``; and
  ``DIALECT_PHRASES``, ``PRIMARY KEY`` in capitals in a string literal, as
  SQL writes a key's kind, prose saying primary key in lower case).
  Comments, ``#[cfg(test)]`` modules and test-module files are not read.
  It is a deny-list, not a grammar: a dialect's knowledge spelled with none
  of its terms is review's (decision E28).
- ``provider-sql``: the client's Python providers
  (``sdm_client/_rust/providers/``) write no SQL: no string literal other
  than a docstring holds an upper-case SQL keyword, a ``select … from`` in
  any case, or ``pragma`` or ``insert`` in any case; the statements they
  run are ``sdm-data``'s. It reads string literals only, so SQL assembled
  from names, or another keyword in lower case outside a ``select …
  from``, is review's (decision E28).
- ``skips``: no ignored Rust test, pytest skip or expected-failure marker,
  no test dropped by a deselect or ignore option or a conftest's ignore
  list, and no CI step allowed to fail, under ``rust/``, ``scripts/rust_core/``,
  ``contract/``, the client tests that touch ``_rust`` or ``_native`` (and
  the ``conftest.py`` files over them), or the CI jobs that run the core or
  its tests, chosen by what they run (``CI_SCOPE_MARKS``: cargo, the
  contract, these scripts, the native wheel, the client), whole jobs and
  single steps alike (design, "Workspace rules"). A job that runs the core's
  tests only through a wrapper whose name gives nothing away (``make
  test``) would not be chosen; such a job needs a mark added here. The tokens are ``SKIP_TOKENS``, matched
  exactly (decision E24), so ``skip_while`` and ``#[schemars(skip)]`` pass;
  they are written in pieces here, since the ban covers this file.
- ``bans``: the clippy bans are configured for every domain and shipped
  crate (``rust/clippy.toml``, repeated in full by any crate's own file but
  ``sdm-testkit``'s), no lint table turns them off, and no attribute in
  shipped code allows or expects them, nor ``warnings``, ``clippy::style``
  or ``clippy::all``, however spaced and inside ``cfg_attr`` too, but
  ``sdm-host``'s one allow on its one statement reading the zone lookup
  roots (decision E24); a test file may allow them, with a reason. It also
  refuses an ``allow`` or ``expect`` of them outside an attribute, as a
  macro would pass one into an attribute it builds, any allow, expect or
  attribute a macro metavariable fills (``#[allow($lint)]``,
  ``#[allow(clippy::$l)]``, ``#[$meta]``), ``#[path]`` and ``include!`` in
  shipped source, and rustflags or CI settings that allow or cap lints or
  point clippy at another configuration. It also refuses,
  lexically, a ``Zoned`` parsed from text in shipped code
  (``parse::<Zoned>``, ``: Zoned = ….parse()``, ``Zoned::from_str``), which
  looks its zone up in jiff's global database through a trait clippy
  cannot ban. A parse whose target is only inferred, such as ``s.parse()``
  returned as ``Result<Zoned, _>`` or ``Option<Zoned>``, is not seen; review
  covers it.
- ``steps``: ``Step`` has exactly the three variants ``fetch``,
  ``call_model`` and ``done``, in the Rust source and in the locked schema:
  no step asks the host for language behavior (migration plan, L0).

Threat model (decision E56, checkpoint 4): the threat is honest drift by
cooperating agents, a term, a statement, a skip or an allow written by
ordinary means. Evasion that needs deliberate obfuscation (text built at run
time or from bytes, a macro that assembles what a check reads, an identifier
spelled to slip a list, a lint cap smuggled through configuration the scan
does not read) is review's, and a finding of that kind is P3 by rule, as
E28 already holds for the SQL scan's lexical gaps. The behavioural
protection is the ``python-core`` against ``rust-host`` parity on the cases.
This script is frozen at bug fixes (E56, extending E49(a)).

``selftest.py`` holds each check to a seeded fixture it must fail on.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path('scripts/rust_core/l0_scan.py')

# The crates whose code ships and whose rules are the core's: the domain
# crates and the host layer. The binding, sdm-py, talks to Python by design,
# and sdm-testkit is dev-only (l0_deps.py classifies every crate).
CORE_CRATES = (
    'sdm-contract',
    'sdm-core',
    'sdm-pql',
    'sdm-data',
    'sdm-schema',
    'sdm-engine',
    'sdm-wire',
    'sdm-request',
    'sdm-host',
)
PARSER_CRATE = 'sdm-pql'
SQL_FREE_CRATES = ('sdm-schema', 'sdm-engine', 'sdm-request', 'sdm-core')
DEV_CRATES = ('sdm-testkit',)
ALLOWLIST = Path('scripts/rust_core/emulation-allowlist.txt')


@dataclass(frozen=True)
class Source:
    """A Rust file with its comments blanked.

    ``code`` keeps string literals; ``skeleton`` blanks their contents too.
    Both have the original's length and line breaks, so an offset in either
    is a position in the file.
    """

    code: str
    skeleton: str
    strings: tuple[tuple[int, int], ...]  # (start, end) of each literal

    def line(self, offset: int) -> int:
        return self.code.count('\n', 0, offset) + 1


def _blank(text: str) -> str:
    return ''.join('\n' if c == '\n' else ' ' for c in text)


_RAW_STRING = re.compile(r'b?r(#*)"')
_CHAR = re.compile(r"'(?:\\u\{[0-9a-fA-F]+\}|\\.|[^\\'\n])'")


def read_rust(text: str) -> Source:
    """Blank the comments of Rust source and find its string literals."""
    code: list[str] = []
    skeleton: list[str] = []
    strings: list[tuple[int, int]] = []
    i, n = 0, len(text)
    while i < n:
        if text.startswith('//', i):
            j = text.find('\n', i)
            j = n if j < 0 else j
            code.append(_blank(text[i:j]))
            skeleton.append(_blank(text[i:j]))
            i = j
            continue
        if text.startswith('/*', i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith('/*', j):
                    depth, j = depth + 1, j + 2
                elif text.startswith('*/', j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            code.append(_blank(text[i:j]))
            skeleton.append(_blank(text[i:j]))
            i = j
            continue
        raw = _RAW_STRING.match(text, i)
        if raw and (
            i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_')
        ):
            close = '"' + raw.group(1)
            j = text.find(close, raw.end())
            j = n if j < 0 else j + len(close)
            code.append(text[i:j])
            tail = len(close) if text[raw.end() : j].endswith(close) else 0
            skeleton.append(
                text[i : raw.end()]
                + _blank(text[raw.end() : j - tail])
                + text[j - tail : j]
            )
            strings.append((i, j))
            i = j
            continue
        if text[i] == '"' or (
            text.startswith('b"', i)
            and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == '_'))
        ):
            start = i
            j = i + (2 if text[i] == 'b' else 1)
            while j < n and text[j] != '"':
                j += 2 if text[j] == '\\' else 1
            j = min(j + 1, n)
            code.append(text[start:j])
            head = 2 if text[start] == 'b' else 1
            closed = j - 1 > start + head - 1 and text[j - 1] == '"'
            inner_end = j - 1 if closed else j
            skeleton.append(
                text[start : start + head]
                + _blank(text[start + head : inner_end])
                + text[inner_end:j]
            )
            strings.append((start, j))
            i = j
            continue
        if text[i] == "'":
            char = _CHAR.match(text, i)
            if char:
                code.append(_blank(char.group(0)))
                skeleton.append(_blank(char.group(0)))
                i = char.end()
                continue
        code.append(text[i])
        skeleton.append(text[i])
        i += 1
    return Source(''.join(code), ''.join(skeleton), tuple(strings))


_TEST_MODULE = re.compile(
    r'#\[cfg\(test\)\]\s*(?:pub(?:\([a-z]+\))?\s+)?mod\s+\w+\s*\{'
)


def without_test_modules(source: Source) -> Source:
    """Blank every ``#[cfg(test)] mod … { … }`` block."""
    code, skeleton = source.code, source.skeleton
    spans: list[tuple[int, int]] = []
    for match in _TEST_MODULE.finditer(skeleton):
        depth, j = 1, match.end()
        while j < len(skeleton) and depth:
            if skeleton[j] == '{':
                depth += 1
            elif skeleton[j] == '}':
                depth -= 1
            j += 1
        spans.append((match.start(), j))
    for start, end in spans:
        code = code[:start] + _blank(code[start:end]) + code[end:]
        skeleton = (
            skeleton[:start] + _blank(skeleton[start:end]) + skeleton[end:]
        )
    strings = tuple(
        s for s in source.strings if not any(a <= s[0] < b for a, b in spans)
    )
    return Source(code, skeleton, strings)


_IDENT = re.compile(r'\b[A-Za-z_][A-Za-z0-9_]*\b')


def identifiers(source: Source) -> Iterable[tuple[int, str]]:
    for match in _IDENT.finditer(source.skeleton):
        yield match.start(), match.group(0)


def string_literals(source: Source) -> Iterable[tuple[int, str]]:
    for start, end in source.strings:
        yield start, source.code[start:end]


def words(identifier: str) -> list[str]:
    """``PyIntValue`` and ``py_int_value`` both give ``py``, ``int``, ``value``."""
    spaced = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', identifier)
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    return [w.lower() for w in re.split(r'[\s_]+', spaced) if w]


def rust_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob('*.rs') if 'target' not in p.parts)


def crate_dir(root: Path, crate: str) -> Path:
    for parent in ('rust/crates', 'rust/bindings'):
        candidate = root / parent / crate
        if candidate.is_dir():
            return candidate
    return root / 'rust/crates' / crate


# --- vocabulary -------------------------------------------------------------

VOCABULARY_WORDS = frozenset(
    {
        'py',
        'python',
        'cpython',
        'pythonic',
        'pandas',
        'numpy',
        'pyarrow',
        'pyobject',
        'truthy',
        'truthiness',
        'timsort',
        'dumps',
        'datetime64',
        'timedelta64',
        'pythonhashseed',
    }
)
# A Python type or builtin behind a `py` prefix, as one word: pyint, pystr.
_PY_PREFIXED = re.compile(
    r'^py(int|float|str|repr|bool|bytes|list|dict|tuple|set|obj|object|hash|sort|round|format|len)'
)
_DUNDER = re.compile(r'^__[a-z0-9_]+__$')
_STRING_VOCABULARY = re.compile(
    r'(?i)\b(python|cpython|pandas|numpy|pyarrow|datetime64|timedelta64)\b|__[a-z]+__'
)
_PUB_ITEM = re.compile(
    r'\bpub(?:\([a-z]+\))?\s+(?:fn|struct|enum|const|type|trait|static|mod)\s+([A-Za-z_]\w*)'
)


def tree_name(
    root: Path, script: Path, name: str
) -> tuple[str | None, list[str]]:
    """The string the checked tree's own copy of ``script`` assigns to
    ``name`` at its top level, read as a literal, never run; or why it
    cannot be read. A gate runs its checkpoint's copy of a guard over a
    candidate's tree, so a crate's name in a guard's data is the tree's
    (T1.28w: the checkpoint's copy named the parser crate as it stood, so a
    package that renamed it could not pass its own gate)."""
    path = root / script
    if not path.is_file():
        return None, [f"the tree's {script} is missing"]
    try:
        tree = ast.parse(path.read_text(encoding='utf-8'))
    except SyntaxError as error:
        return None, [f"the tree's {script} does not parse: {error.msg}"]
    value: ast.expr | None = None
    for node in tree.body:
        target = (
            node.target
            if isinstance(node, ast.AnnAssign)
            else node.targets[0]
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            else None
        )
        if isinstance(target, ast.Name) and target.id == name:
            value = node.value  # type: ignore[union-attr]
    if (
        isinstance(value, ast.Constant)
        and isinstance(value.value, str)
        and value.value
    ):
        return value.value, []
    return None, [f"the tree's {script}: {name} is not a string literal"]


def parser_crates(root: Path) -> tuple[str, tuple[str, ...], list[str]]:
    """The parser crate's name in the tree at ``root`` (its copy of this
    file's ``PARSER_CRATE``), and ``CORE_CRATES`` with the parser under it;
    a name this copy gives another crate is refused."""
    name, problems = tree_name(root, SCRIPT, 'PARSER_CRATE')
    others = {*CORE_CRATES, *DIALECT_FREE_CRATES, *DEV_CRATES} - {PARSER_CRATE}
    if name in others:
        problems = [
            f"the tree's {SCRIPT}: PARSER_CRATE names {name}, another crate"
        ]
    if name is None or problems:
        return PARSER_CRATE, CORE_CRATES, problems
    crates = tuple(name if c == PARSER_CRATE else c for c in CORE_CRATES)
    return name, crates, []


def read_allowlist(root: Path, parser: str) -> tuple[list[Path], list[str]]:
    entries: list[Path] = []
    problems: list[str] = []
    path = root / ALLOWLIST
    if not path.is_file():
        return [], [f'{ALLOWLIST}: missing']
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        entry = Path(line)
        if not (root / entry).is_file():
            problems.append(f'{ALLOWLIST}:{number}: {line} does not exist')
            continue
        parser_src = Path('rust/crates') / parser / 'src'
        if parser_src not in entry.parents:
            problems.append(
                f'{ALLOWLIST}:{number}: {line} is outside {parser_src}; '
                'emulation stays in the parser crate (H21)'
            )
            continue
        entries.append(entry)
    return entries, problems


def vocabulary_term(identifier: str) -> str | None:
    if _DUNDER.match(identifier):
        return identifier
    for word in words(identifier):
        if word in VOCABULARY_WORDS or _PY_PREFIXED.match(word):
            return word
    return None


def check_vocabulary(root: Path) -> list[str]:
    parser, crates, unreadable = parser_crates(root)
    if unreadable:
        return unreadable
    allowed, problems = read_allowlist(root, parser)
    allowed_set = {root / p for p in allowed}
    # The names the allowlisted modules define, and the modules themselves,
    # may be named anywhere in their crate.
    permitted = {p.stem for p in allowed}
    for path in allowed_set:
        permitted.update(
            _PUB_ITEM.findall(read_rust(path.read_text()).skeleton)
        )
    for crate in crates:
        src = crate_dir(root, crate) / 'src'
        for path in rust_files(src):
            if path in allowed_set:
                continue
            source = without_test_modules(read_rust(path.read_text()))
            rel = path.relative_to(root)
            for offset, name in identifiers(source):
                term = vocabulary_term(name)
                if term and not (crate == parser and name in permitted):
                    problems.append(
                        f'{rel}:{source.line(offset)}: `{name}` is Python-emulation '
                        f'vocabulary ({term}); emulation lives only in the modules '
                        f'of {ALLOWLIST}'
                    )
            for offset, text in string_literals(source):
                match = _STRING_VOCABULARY.search(text)
                if match:
                    problems.append(
                        f'{rel}:{source.line(offset)}: a string names '
                        f'`{match.group(0)}`; the core takes typed data and '
                        'names no Python library or behavior'
                    )
    return problems


# --- SQL fragments ------------------------------------------------------------

# Whether a statement's literals are SQL is decided by SQL's structure, not
# by its words, since diagnostics use the same words in English ("the field
# is null", "order by time is required", "could not select a column from
# {c}"). An operand is a `{…}` placeholder, `*`, `?`, `$n`, a quoted or
# numeric literal, a call such as `max(a)`, or a word. Literals are joined
# one per line, so `^` is the start of a literal.
#
# Only shapes no sentence writes count (decision E28):
# - `select <operands> from <operand>`, then a clause, a continuation
#   (`,`, `)`, a join, `union`, `using`, `limit`, `having`, `qualify`) or
#   the end, either in capitals or at the start of a literal or after
#   `:`, `(` or a quote; a `SELECT <literal>` opening a literal
#   (`SELECT 1`); `SELECT … FROM {t} T`, a table with an alias;
# - `insert into <operand>` then `(`, `values` or `select`; `delete from
#   <operand> where`; `truncate table`; `update … set … =`; `create`,
#   `drop` or `alter` of a table or view then `(`, `as select` or the end;
#   `alter table <operand>` then `add`, `drop`, `rename`, `alter` or `set`;
#   `with x as (select`;
# - `cast(… as <type>)`, `count(*) from`, `row_number() over`,
#   `over (partition by`, `union all`, `information_schema`,
#   `tablesample`, `using sample`.
# Predicates, orderings, null tests and keywords in capitals are left to
# review: a message writes them too (`{c} > 0 and {c} is not null`).
#
# Diagnostic text is exempt entirely: the message arguments of a verified
# error helper (decision E28): one of the error types' own functions that
# build one (`CoreError::new`, `Detail::text`), a free function of the
# scanned crate every definition of which returns `CoreError` or `Detail`,
# and `.with_field` or `.caused_by` on a chain that starts at one of those;
# a name alone, a path elsewhere (`text::invalid`) or another receiver
# verifies nothing. Also `panic!`, `unreachable!`, `todo!`, `unimplemented!`,
# `.expect`, `.context`, `.with_context` and `#[error]`; for `assert!` the
# arguments after the condition, and for `assert_eq!`/`assert_ne!` (and the
# `debug_` forms) those after the two operands, which are code. The
# arguments of a call nested in a message that is not itself diagnostic
# (`internal(&run("…"))`) are code. A turbofish's type arguments nest when
# the arguments are split; where a bare `<` after a name leaves the end of
# the code unclear, no argument is diagnostic text. A message bound to a
# name first and passed on later is read as code.
_O = (
    r'(?:\{[^}]*\}|\*|\?|\$\d+|\'[^\']*\'|\d+(?:\.\d+)?'
    r'|\w+\s*\([^()]*\)|[\w."]+)'
)
_OS = rf'{_O}(?:\s+as\s+\w+)?(?:\s*,\s*{_O}(?:\s+as\s+\w+)?)*'
_CMP = r'(?:=|<>|!=|<=|>=|<|>)'
_CLAUSE_NEXT = (
    r'(?i:\s*[,)]|\s+(?:where|group|order|limit|offset|having|qualify|join|'
    r'inner|left|right|full|cross|natural|union|using|as|on)\b)'
)
_CONTINUE = rf'(?:{_CLAUSE_NEXT}|\s*$)'
# A table may carry an alias without `as`: then a clause must follow, or,
# after `select *`, the end (`select * from {t} t`); `select {c} from {t}
# first` stays English.
_SELECT = (
    rf'select\s+(?:distinct\s+)?{_OS}\s+from'
    rf'(?:\s+{_O}(?:{_CONTINUE}|\s+\w+{_CLAUSE_NEXT})|\s*\(|\s*$)'
    rf'|select\s+(?:distinct\s+)?\*\s+from\s+{_O}\s+\w+\s*$'
)
_SQL_STRONG = re.compile(
    r'(?m)'
    rf'(?i:(?:^|[:(`\'"])\s*{_SELECT})'
    r'|\bSELECT\s+(?:DISTINCT\s+)?'
    rf'{_OS}\s+FROM\s+{_O}(?:{_CONTINUE}|\s+\w+\s*$)'
    r'|(?i:^\s*select\s+(?:\*|\{[^}]*\}|\d+|\'[^\']*\'|\?|\$\d+)'
    r'\s*(?:as\s+\w+\s*)?$)'
    rf'|(?i:\binsert\s+into\s+{_O}\s*(?:\(|values\b|select\b|by\s+name\b))'
    rf'|(?i:\bdelete\s+from\s+{_O}\s+where\b|\btruncate\s+table\s+{_O})'
    rf'|(?i:\bupdate\s+{_O}\s+set\s+{_O}\s*=)'
    r'|(?i:\b(?:create|drop|alter)\s+(?:or\s+replace\s+)?'
    r'(?:temp(?:orary)?\s+)?(?:table|view|index|schema)\s+'
    rf'(?:if\s+(?:not\s+)?exists\s+)?{_O}\s*(?:\(|as\s+select\b|$))'
    rf'|(?i:\balter\s+table\s+(?:if\s+exists\s+)?{_O}\s+'
    r'(?:add|drop|rename|alter|set)\b)'
    r'|(?i:\bwith\s+\w+\s+as\s*\(\s*select\b|\bexists\s*\(\s*select\b)'
    rf'|(?i:\b(?:coalesce|nullif|ifnull|greatest|least)\s*\(\s*{_O}\s*,)'
    r'|(?i:\bvalues\s*\(\s*(?:\?|\$\d+|\{[^}]*\})\s*,)'
    r'|(?i:\bcast\s*\([^()]*?\s+as\s+\w+(?:\s*\([^()]*\))?\s*\))'
    r'|(?i:\bcount\s*\(\s*(?:\*|distinct\b[^()]*)\)\s+from\b)'
    r'|(?i:\b(?:row_number|dense_rank|rank|ntile|lag|lead)\s*\(\s*\)\s*over\b)'
    r'|(?i:\bover\s*\(\s*(?:partition|order)\s+by\b)'
    r'|(?i:\bunion\s+all\b|\b(?:information_schema|tablesample)\b'
    r'|\busing\s+sample\b)'
)
# A provider's name as a whole literal is a provider match; in a sentence
# ("came from a Spark job") it is prose.
_PROVIDER_LITERAL = re.compile(
    r'(?i)^\s*(duckdb|postgres(ql)?|snowflake|databricks|sqlite|spark|'
    r'bigquery|redshift|lakebase)\s*$'
)
_INCLUDE = re.compile(r'\binclude_(str|bytes)\s*!')
_LITERAL = re.compile(r'^(b?r?)(#*)"(?s:(.*))"\2$')
_ESCAPE = re.compile(
    r'\\(?:\n\s*|x[0-9a-fA-F]{2}|u\{[0-9a-fA-F_]{1,8}\}|[ntr0\\"\'])'
)
_SIMPLE_ESCAPES = {
    'n': '\n',
    't': '\t',
    'r': '\r',
    '0': '\0',
    '\\': '\\',
    '"': '"',
    "'": "'",
}


def _unescape(match: re.Match[str]) -> str:
    escape = match.group(0)[1:]
    if escape[0] in '\n \t\r' and escape.strip() == '':
        return ''  # a line continuation
    if escape[0] == 'x':
        return chr(int(escape[1:], 16))
    if escape[0] == 'u':
        code = int(escape[2:-1].replace('_', ''), 16)
        return chr(code) if code <= 0x10FFFF else ''
    return _SIMPLE_ESCAPES[escape]


def literal_text(literal: str) -> str:
    """The text a Rust string or byte-string literal stands for."""
    match = _LITERAL.match(literal)
    if not match:
        return literal
    if 'r' in match.group(1):
        return match.group(3)
    return _ESCAPE.sub(_unescape, match.group(3))


# The diagnostic forms whose message arguments are exempt, and how many
# leading arguments are code. Error helpers are verified, not named: see
# `Verified` and `diagnostic_calls`.
_DIAGNOSTIC_MACRO = re.compile(
    r'\b(?P<assert>assert|debug_assert)\s*!\s*\('
    r'|\b(?P<pair>assert_eq|assert_ne|debug_assert_eq|debug_assert_ne)\s*!\s*\('
    r'|\b(?:panic|unreachable|todo|unimplemented)\s*!\s*\('
    r'|\.\s*(?:expect|context|with_context)\s*\('
    r'|#\s*\[\s*error\s*\('
)
# The contract's error type and its detail type; a helper building one is a
# diagnostic.
ERROR_TYPES = ('CoreError', 'Detail')
_ERROR_RETURN = re.compile(r'^(?:CoreError|Detail)$')
_FN_NAME = re.compile(r'\bfn\s+([A-Za-z_]\w*)')
_IMPL = re.compile(r'\bimpl\s*(?:<[^{};]*?>\s*)?([A-Za-z_]\w*)\s*\{')
_ASSOC_CALL = re.compile(
    rf'\b({"|".join(ERROR_TYPES)})\s*::\s*([A-Za-z_]\w*)\s*\('
)
_PLAIN_CALL = re.compile(r'\b([A-Za-z_]\w*)\s*\(')
_CHAIN_METHOD = re.compile(r'\.\s*(with_field|caused_by)\s*\(')
_OPEN = {'(': ')', '[': ']', '{': '}'}
_CLOSE = {v: k for k, v in _OPEN.items()}


def _matching(text: str, index: int) -> int:
    """The index just past the bracket closing the one at ``index``."""
    depth, j = 0, index
    while j < len(text):
        if text[j] in _OPEN:
            depth += 1
        elif text[j] in _CLOSE:
            depth -= 1
            if not depth:
                return j + 1
        j += 1
    return len(text)


def _opening(text: str, index: int) -> int:
    """The index of the bracket opening the one closing at ``index``."""
    depth, j = 0, index
    while j >= 0:
        if text[j] in _CLOSE:
            depth += 1
        elif text[j] in _OPEN:
            depth -= 1
            if not depth:
                return j
        j -= 1
    return 0


def _return_type(text: str) -> str:
    """A return type, paths and spaces removed: `Result<u8,CoreError>`."""
    text = re.sub(r'\bwhere\b.*', '', text, flags=re.S)
    text = re.sub(r'\b(?:[A-Za-z_]\w*\s*::\s*)+(?=[A-Za-z_])', '', text)
    return re.sub(r'\s+', '', text)


def fn_signatures(skeleton: str) -> Iterable[tuple[int, str, str, str]]:
    """Each `fn`: its offset, name, parameters and return type."""
    for match in _FN_NAME.finditer(skeleton):
        j = match.end()
        while j < len(skeleton) and skeleton[j].isspace():
            j += 1
        if j < len(skeleton) and skeleton[j] == '<':  # generic parameters
            depth = 0
            while j < len(skeleton):
                depth += {'<': 1, '>': -1}.get(skeleton[j], 0)
                j += 1
                if not depth:
                    break
        open_paren = skeleton.find('(', j)
        if open_paren < 0:
            continue
        close = _matching(skeleton, open_paren)
        params = skeleton[open_paren + 1 : close - 1]
        tail = re.match(r'\s*(?:->\s*([^{;]*))?', skeleton[close:])
        ret = _return_type(tail.group(1) or '') if tail else ''
        yield match.start(), match.group(1), params, ret


@dataclass(frozen=True)
class Verified:
    """The error helpers a crate's code may call with a message.

    ``assoc`` are the error types' own functions that build one
    (`CoreError::new`, `Detail::text`), ``methods`` the error's methods that
    return it (`with_field`, `caused_by`), ``helpers`` the crate's own free
    functions every definition of which returns `CoreError` or `Detail`.
    A function returning `Result<_, CoreError>` is an operation that may
    fail, not a message's builder, so its arguments stay code.
    """

    assoc: frozenset[tuple[str, str]] = frozenset()
    methods: frozenset[str] = frozenset()
    helpers: frozenset[str] = frozenset()


NOTHING_VERIFIED = Verified()


def workspace_sources(root: Path) -> list[Path]:
    return [
        path
        for parent in ('rust/crates', 'rust/bindings')
        for path in rust_files(root / parent)
        if 'src' in path.relative_to(root / parent).parts[1:2]
    ]


def verified_helpers(root: Path, crate: str) -> Verified:
    assoc: set[tuple[str, str]] = set()
    methods: set[str] = set()
    for path in workspace_sources(root):
        skeleton = read_rust(path.read_text()).skeleton
        for impl in _IMPL.finditer(skeleton):
            if impl.group(1) not in ERROR_TYPES:
                continue
            body = skeleton[
                impl.end() - 1 : _matching(skeleton, impl.end() - 1)
            ]
            for _, name, params, ret in fn_signatures(body):
                if ret not in ('Self', impl.group(1)):
                    continue
                assoc.add((impl.group(1), name))
                takes_self = re.match(r'\s*(?:mut\s+)?self\b', params)
                if impl.group(1) == 'CoreError' and takes_self:
                    methods.add(name)
    returns: dict[str, set[bool]] = {}
    for path in rust_files(crate_dir(root, crate) / 'src'):
        skeleton = read_rust(path.read_text()).skeleton
        for _, name, _, ret in fn_signatures(skeleton):
            returns.setdefault(name, set()).add(bool(_ERROR_RETURN.match(ret)))
    helpers = {name for name, oks in returns.items() if oks == {True}}
    return Verified(frozenset(assoc), frozenset(methods), frozenset(helpers))


def _callee_before(skeleton: str, end: int) -> tuple[int, str]:
    """The path called by the `(` at ``end``: its start and text."""
    j = end - 1
    while j >= 0 and skeleton[j].isspace():
        j -= 1
    k = j
    while k >= 0 and (
        skeleton[k].isalnum() or skeleton[k] in '_:' or skeleton[k].isspace()
    ):
        k -= 1
    return k + 1, re.sub(r'\s+', '', skeleton[k + 1 : j + 1])


def _chain_verified(skeleton: str, dot: int, verified: Verified) -> bool:
    """Whether the method chain ending at ``dot`` starts at a verified call."""
    j = dot - 1
    while True:
        while j >= 0 and skeleton[j].isspace():
            j -= 1
        if j < 0 or skeleton[j] != ')':
            return False  # a name, a field or a literal: not verified
        start, callee = _callee_before(skeleton, _opening(skeleton, j))
        k = start - 1
        while k >= 0 and skeleton[k].isspace():
            k -= 1
        if k >= 0 and skeleton[k] == '.':
            if callee not in verified.methods:
                return False
            j = k - 1
            continue
        path = callee.split('::')
        if len(path) >= 2 and path[-2] in ERROR_TYPES:
            return (path[-2], path[-1]) in verified.assoc
        return len(path) == 1 and callee in verified.helpers


def diagnostic_calls(
    skeleton: str, verified: Verified
) -> list[tuple[int, int]]:
    """Each diagnostic call: the index past its `(`, and its code arguments."""
    calls = []
    for match in _DIAGNOSTIC_MACRO.finditer(skeleton):
        code_args = (
            1 if match.group('assert') else 2 if match.group('pair') else 0
        )
        calls.append((match.end(), code_args))
    for match in _ASSOC_CALL.finditer(skeleton):
        if (match.group(1), match.group(2)) in verified.assoc:
            calls.append((match.end(), 0))
    for match in _PLAIN_CALL.finditer(skeleton):
        if match.group(1) not in verified.helpers:
            continue
        before = skeleton[: match.start()].rstrip()
        if before.endswith(('.', ':')) or re.search(r'\bfn$', before):
            continue  # a method, a path or a definition, not the helper
        calls.append((match.end(), 0))
    for match in _CHAIN_METHOD.finditer(skeleton):
        if match.group(1) in verified.methods and _chain_verified(
            skeleton, match.start(), verified
        ):
            calls.append((match.end(), 0))
    return sorted(calls)


Span = tuple[int, int, tuple[tuple[int, int], ...]]


def diagnostic_spans(
    source: Source, verified: Verified = NOTHING_VERIFIED
) -> list[Span]:
    """The message arguments of the diagnostic calls.

    Each span carries the argument lists of the calls nested in it that are
    not diagnostic themselves (`internal(&run("…"))`): those are code.
    """
    spans: list[Span] = []
    skeleton = source.skeleton
    calls = diagnostic_calls(skeleton, verified)
    opens = {open_end for open_end, _ in calls}
    for open_end, code_args in calls:
        # A turbofish's type arguments (`same::<A, B>(…)`, `::<Vec<(A, B)>,
        # C>`) nest like brackets, so their commas separate nothing. A bare
        # `<` right after a name (`HashMap<K, V>` in a closure's types)
        # could open type arguments too; then where the code ends cannot be
        # told, and the arguments are all read as code.
        depth, generic, j, start = 1, 0, open_end, open_end
        unsure = False
        while j < len(skeleton) and depth:
            char = skeleton[j]
            if skeleton.startswith('::', j):
                k = j + 2
                while k < len(skeleton) and skeleton[k].isspace():
                    k += 1
                if k < len(skeleton) and skeleton[k] == '<':
                    generic += 1
                    j = k + 1
                    continue
            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
            elif generic:
                if char == '<':
                    generic += 1
                elif char == '>' and skeleton[j - 1] != '-':
                    generic -= 1
            elif (
                char == '<'
                and code_args
                and (skeleton[j - 1].isalnum() or skeleton[j - 1] == '_')
            ):
                unsure = True
            elif char == ',' and depth == 1 and code_args:
                code_args -= 1
                start = j + 1
            j += 1
        if code_args or generic or unsure:
            continue  # no message argument, or none that can be found
        nested = tuple(
            (call.end() - 1, _matching(skeleton, call.end() - 1))
            for call in _PLAIN_CALL.finditer(skeleton, start, j)
            if call.end() not in opens
        )
        spans.append((start, j, nested))
    return spans


def in_diagnostic(offset: int, spans: list[Span]) -> bool:
    return any(
        a <= offset < b and not any(c <= offset < d for c, d in nested)
        for a, b, nested in spans
    )


@dataclass(frozen=True)
class Statement:
    offset: int
    text: str  # every literal, one per line
    prose: str  # the literals outside diagnostic text, one per line
    literals: tuple[str, ...]


def statements(
    source: Source, verified: Verified = NOTHING_VERIFIED
) -> Iterable[Statement]:
    """The string literals of each statement.

    A statement runs to the next `;`, `{` or `}` outside a literal, so the
    pieces of one `format!`, `concat!`, `join` or `+` are read together.
    """
    bounds = (
        [0]
        + [m.end() for m in re.finditer(r'[;{}]', source.skeleton)]
        + [len(source.skeleton)]
    )
    diagnostic = diagnostic_spans(source, verified)
    literals = sorted(source.strings)
    index = 0
    for start, end in zip(bounds, bounds[1:], strict=False):
        parts: list[str] = []
        prose: list[str] = []
        first = None
        while index < len(literals) and literals[index][0] < end:
            s, e = literals[index]
            if s >= start:
                text = literal_text(source.code[s:e])
                parts.append(text)
                if not in_diagnostic(s, diagnostic):
                    prose.append(text)
                first = s if first is None else first
            index += 1
        if parts and first is not None:
            yield Statement(
                first, '\n'.join(parts), '\n'.join(prose), tuple(parts)
            )


def sql_evidence(statement: Statement) -> str | None:
    """The SQL a statement's literals outside diagnostic text hold, or None."""
    match = _SQL_STRONG.search(statement.prose)
    return ' '.join(match.group(0).split()) if match else None


def provider_literal(statement: Statement) -> str | None:
    """A provider's name a statement holds as a whole literal, or None."""
    for literal in statement.literals:
        provider = _PROVIDER_LITERAL.match(literal)
        if provider:
            return provider.group(1)
    return None


_TEST_MODULE_FILE = re.compile(
    r'#\[cfg\(test\)\]\s*(?:pub(?:\([a-z]+\))?\s+)?mod\s+(\w+)\s*;'
)


def test_module_files(src: Path) -> set[Path]:
    """The files of ``#[cfg(test)] mod name;`` declarations under ``src``."""
    found: set[Path] = set()
    for path in rust_files(src):
        skeleton = read_rust(path.read_text()).skeleton
        parent = (
            path.parent
            if path.name in ('lib.rs', 'main.rs', 'mod.rs')
            else path.parent / path.stem
        )
        for name in _TEST_MODULE_FILE.findall(skeleton):
            module = parent / name
            found.update({module.with_suffix('.rs'), module / 'mod.rs'})
            found.update(p for p in rust_files(module) if module.is_dir())
    return found


PROVIDER_WORDS = frozenset(
    {
        'sql',
        'dialect',
        'duckdb',
        'postgres',
        'postgresql',
        'snowflake',
        'databricks',
        'sqlite',
        'spark',
        'bigquery',
        'redshift',
        'lakebase',
    }
)


def check_sql(root: Path) -> list[str]:
    problems: list[str] = []
    for crate in SQL_FREE_CRATES:
        src = crate_dir(root, crate) / 'src'
        if not src.is_dir():
            problems.append(f'{src.relative_to(root)}: missing')
            continue
        tests = test_module_files(src)
        verified = verified_helpers(root, crate)
        for path in rust_files(src):
            source = read_rust(path.read_text())
            rel = path.relative_to(root)
            found = [
                (statement.offset, provider_literal(statement))
                for statement in statements(source)
            ]
            if path not in tests:  # SQL's shapes, outside test modules
                found += [
                    (statement.offset, sql_evidence(statement))
                    for statement in statements(
                        without_test_modules(source), verified
                    )
                ]
            for offset, evidence in sorted(found, key=lambda f: f[0]):
                if evidence:
                    problems.append(
                        f'{rel}:{source.line(offset)}: the strings '
                        f'of a statement hold `{evidence}`, SQL or a '
                        "provider's name; SQL and providers are sdm-data's "
                        'alone'
                    )
            for match in _INCLUDE.finditer(source.skeleton):
                problems.append(
                    f'{rel}:{source.line(match.start())}: `{match.group(0)}` '
                    'brings in text this scan cannot read'
                )
            for offset, name in identifiers(source):
                hit = PROVIDER_WORDS.intersection(words(name))
                if hit:
                    problems.append(
                        f'{rel}:{source.line(offset)}: `{name}` names a provider or '
                        f'SQL ({", ".join(sorted(hit))}); {crate} matches on none'
                    )
    return problems


# --- dialects ---------------------------------------------------------------

# The shared crates that hold no dialect knowledge (decision E54): every
# shipped crate but sdm-data, whose dialects they are, and the parser,
# which reads no data. sdm-testkit is dev-only.
DIALECT_FREE_CRATES = (
    'sdm-contract',
    'sdm-core',
    'sdm-request',
    'sdm-schema',
    'sdm-engine',
    'sdm-wire',
    'sdm-host',
    'sdm-py',
)
# The reviewed deny-list. A dialect's name, or its catalog's own: matched in
# any case anywhere in an identifier or a literal's word, as no neutral name
# holds one.
DIALECT_NAMES = (
    'duckdb',
    'duckdb_indexes',
    'sqlite',
    'postgres',
    'databricks',
    'snowflake',
    'information_schema',
    'dflt_value',
    # The catalog columns DuckDB's constraints answer with, and
    # PostgreSQL's catalog (T1.21d, from T1.21c's review).
    'constraint_type',
    'constraint_column_names',
    'referenced_column_names',
    'pg_catalog',
)
# Catalog terms that neutral names hold as parts or as longer words:
# matched as whole words (``words``) only, a term of two words as the two
# in a row, so ``pragmatic`` and ``printable_info`` pass (T1.21d).
DIALECT_WORDS = ('notnull', 'cid', 'pragma', 'table_info')
# A key's kind as SQL and DuckDB's ``constraint_type`` write it, in
# capitals: in a string literal only, since prose says primary key in
# lower case, as the reference's own messages do (T1.21d).
DIALECT_PHRASES = ('PRIMARY KEY',)
_WORD = re.compile(r'[A-Za-z_][A-Za-z0-9_]*')


def dialect_terms(name: str) -> list[str]:
    """The deny-list's terms ``name`` holds."""
    lowered = name.lower()
    found = [term for term in DIALECT_NAMES if term in lowered]
    parts = words(name)
    for term in DIALECT_WORDS:
        wanted = term.split('_')
        if any(
            parts[at : at + len(wanted)] == wanted
            for at in range(len(parts) - len(wanted) + 1)
        ):
            found.append(term)
    return found


def dialect_phrases(text: str) -> list[str]:
    """The deny-list's phrases a string literal's text holds, its words
    separated by any run of white space."""
    return [
        phrase
        for phrase in DIALECT_PHRASES
        if re.search(r'\b' + r'\s+'.join(phrase.split()) + r'\b', text)
    ]


def check_dialects(root: Path) -> list[str]:
    problems: list[str] = []
    for crate in DIALECT_FREE_CRATES:
        src = crate_dir(root, crate) / 'src'
        if not src.is_dir():
            problems.append(f'{src.relative_to(root)}: missing')
            continue
        tests = test_module_files(src)
        for path in rust_files(src):
            if path in tests:
                continue
            source = without_test_modules(read_rust(path.read_text()))
            rel = path.relative_to(root)
            found = [(offset, name) for offset, name in identifiers(source)]
            for offset, literal in string_literals(source):
                text = literal_text(literal)
                found += [(offset, word) for word in _WORD.findall(text)]
                found += [(offset, phrase) for phrase in dialect_phrases(text)]
            for offset, name in found:
                hit = [name] if name in DIALECT_PHRASES else dialect_terms(name)
                if hit:
                    problems.append(
                        f'{rel}:{source.line(offset)}: `{name}` names a '
                        f'dialect ({", ".join(hit)}); a dialect is '
                        f"sdm-data's alone, and {crate} holds none"
                    )
    return problems


# --- the Python providers write no SQL ---------------------------------------

PROVIDERS = Path(
    'packages/structured-data-models-client/src/sdm_client/_rust/providers'
)
# Upper-case SQL keywords, as SQL is written; a `select … from` in any case.
_PROVIDER_SQL = re.compile(
    r'\b(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|PRAGMA|ATTACH'
    r'|COPY|FROM|WHERE|JOIN|UNION|QUALIFY|VALUES)\b'
    r'|(?i:\bselect\b.*\bfrom\b)'
    # And, in lower case too, the two a statement opens with that no
    # message of a provider needs (T1.21d, from T1.21c's review).
    r'|(?i:\b(?:pragma|insert)\b)',
    re.S,
)


def _docstrings(tree: ast.AST) -> set[int]:
    """The ids of the docstring nodes of a module, class or function."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(
                node,
                (
                    ast.Module,
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and node.body
        ):
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.add(id(first.value))
    return found


def check_provider_sql(root: Path) -> list[str]:
    directory = root / PROVIDERS
    files = sorted(directory.rglob('*.py')) if directory.is_dir() else []
    if not files:
        return [f'{PROVIDERS}: no provider to read']
    problems: list[str] = []
    for path in files:
        rel = path.relative_to(root)
        try:
            tree = ast.parse(path.read_text(), filename=str(rel))
        except SyntaxError as error:
            problems.append(f'{rel}:{error.lineno}: does not parse')
            continue
        docstrings = _docstrings(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
            ):
                match = _PROVIDER_SQL.search(node.value)
                if match:
                    problems.append(
                        f'{rel}:{node.lineno}: a string holds '
                        f'`{" ".join(match.group(0).split())[:40]}`, SQL; '
                        "the statements a provider runs are sdm-data's"
                    )
    return problems


# --- the skip and ignore ban ------------------------------------------------

# Exact tokens (decision E24). Written in pieces, so this file and
# selftest.py, which the ban covers, do not hold them whole.
_I, _S, _X = 'ignore', 'skip', 'fail'
SKIP_TOKENS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ('#[' + _I + ']', re.compile(r'#\[\s*' + _I + r'\b')),
    (
        'cfg_attr(…, ' + _I + ')',
        re.compile(r'cfg_attr\s*\([^\]]*\b' + _I + r'\b'),
    ),
    # The skip marker under any alias of pytest, or imported bare.
    ('mark.' + _S, re.compile(r'\bmark\.' + _S + r'\b')),
    ('pytest.' + _S, re.compile(r'pytest\.' + _S + r'\b')),
    (_S + 'if', re.compile(r'\b' + _S + r'if\b')),
    ('x' + _X, re.compile(r'\bx' + _X + r'\b')),
    ('importor' + _S, re.compile(r'\bimportor' + _S + r'\b')),
    ('unittest.' + _S, re.compile(r'unittest\.' + _S)),
    (_S.capitalize() + 'Test', re.compile(r'\b' + _S.capitalize() + r'Test\b')),
    ('expected' + 'Failure', re.compile(r'\bexpected' + r'Failure\b')),
    (_S + 'Test', re.compile(r'\.' + _S + r'Test\b')),
    ('continue-on-' + 'error', re.compile(r'\bcontinue-on-' + r'error\b')),
    # Deselection drops a test as surely as a skip does.
    ('--de' + 'select', re.compile(r'--de' + r'select\b')),
    ('--ig' + 'nore', re.compile(r'--ig' + r'nore(?:-glob)?[=\s]')),
    ('collect_ig' + 'nore', re.compile(r'\bcollect_ig' + r'nore(?:_glob)?\b')),
)
# In Python only: a bare call of the skip function, however pytest or the
# function was imported or aliased. Rust's iterator method of that name is
# not a skip, so Rust files are not held to it.
PYTHON_SKIP_TOKENS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (_S + '(', re.compile(r'\b' + _S + r'\(')),
)
CLIENT_TESTS = Path('packages/structured-data-models-client')
CI_FILE = Path('.github/workflows/ci.yml')
_CI_JOB = re.compile(r'^  ([A-Za-z0-9_-]+):\s*$')


# What puts a CI job in scope, by what it runs: the contract's checks, the
# Rust core's scripts, cargo, the native client wheel, or the client's tests.
CI_SCOPE_MARKS = (
    'contract/',
    'scripts/rust_core/',
    'cargo ',
    'native-wheel',
    'native/*.whl',
    'structured-data-models-client',
)


def ci_job_in_scope(name: str, body: str) -> bool:
    """A Rust or contract job, or one that runs the core or its tests."""
    return (
        name.startswith(('rust-', 'contract-'))
        or name in ('native-wheel', 'wheel-smoke')
        or any(mark in body for mark in CI_SCOPE_MARKS)
    )


def tracked_files(root: Path, *paths: str) -> list[Path]:
    """The files git tracks under ``paths``, or every file when not in git."""
    try:
        out = subprocess.run(
            ['git', 'ls-files', '-z', '--', *paths],
            cwd=root,
            capture_output=True,
            check=True,
        ).stdout.decode()
        files = [root / p for p in out.split('\0') if p]
    except (subprocess.CalledProcessError, FileNotFoundError):
        files = []
    if not files:
        files = [
            p
            for path in paths
            for p in sorted((root / path).rglob('*'))
            if p.is_file() and 'target' not in p.parts
        ]
    return [p for p in files if p.is_file()]


def _text(path: Path) -> str | None:
    try:
        return path.read_text()
    except (UnicodeDecodeError, OSError):
        return None


def skip_scope(root: Path) -> list[Path]:
    files = tracked_files(root, 'rust', 'scripts/rust_core', 'contract')
    client = [
        p
        for p in tracked_files(root, str(CLIENT_TESTS))
        if p.suffix == '.py'
        and p.relative_to(root / CLIENT_TESTS).parts[0].startswith('tests')
    ]
    touching = [
        p for p in client if re.search(r'\b_(rust|native)\b', _text(p) or '')
    ]
    conftests = {
        c
        for p in touching
        for c in client
        if c.name == 'conftest.py' and c.parent in (p.parent, *p.parents)
    }
    return sorted({*files, *touching, *conftests})


def find_skips(text: str, python: bool = False) -> Iterable[tuple[int, str]]:
    tokens = SKIP_TOKENS + (PYTHON_SKIP_TOKENS if python else ())
    for token, pattern in tokens:
        for match in pattern.finditer(text):
            yield text.count('\n', 0, match.start()) + 1, token


def ci_jobs(text: str) -> dict[str, tuple[int, str]]:
    """Each job of a workflow, with its first line and its text."""
    lines = text.splitlines(keepends=True)
    jobs: dict[str, tuple[int, str]] = {}
    inside, name, start = False, '', 0
    body: list[str] = []
    for number, line in enumerate(lines, 1):
        if line.startswith('jobs:'):
            inside = True
            continue
        if not inside:
            continue
        if (
            line.strip()
            and not line.startswith(' ')
            and not line.startswith('#')
        ):
            break
        match = _CI_JOB.match(line)
        if match:
            if name:
                jobs[name] = (start, ''.join(body))
            name, start, body = match.group(1), number, []
        elif name:
            body.append(line)
    if name:
        jobs[name] = (start, ''.join(body))
    return jobs


def check_skips(root: Path) -> list[str]:
    problems: list[str] = []
    for path in skip_scope(root):
        text = _text(path)
        if text is None:
            continue
        for line, token in find_skips(text, python=path.suffix == '.py'):
            problems.append(
                f'{path.relative_to(root)}:{line}: `{token}`; nothing in the Rust '
                'core\'s scope is skipped or ignored (design, "Workspace rules")'
            )
    ci = root / CI_FILE
    if not ci.is_file():
        return [*problems, f'{CI_FILE}: missing']
    for job, (start, body) in ci_jobs(ci.read_text()).items():
        if not ci_job_in_scope(job, body):
            continue
        for line, token in find_skips(body):
            problems.append(
                f'{CI_FILE}:{start + line}: `{token}` in job {job}; the jobs '
                'that run the core or its tests fail on every failure'
            )
    return problems


# --- the clippy bans ----------------------------------------------------------

REQUIRED_TYPES = frozenset(
    {
        'std::collections::HashMap',
        'std::collections::HashSet',
        'std::hash::RandomState',
        'std::time::SystemTime',
        'std::time::Instant',
        'std::fs::File',
        'std::fs::OpenOptions',
        'rand::rngs::StdRng',
        'rand::rngs::ThreadRng',
    }
)
REQUIRED_METHODS = frozenset(
    {
        'std::time::SystemTime::now',
        'std::time::Instant::now',
        'jiff::Timestamp::now',
        'jiff::Zoned::now',
        'std::env::var',
        'std::env::var_os',
        'std::env::vars',
        'std::env::vars_os',
        'std::env::current_dir',
        'std::fs::read',
        'std::fs::read_to_string',
        'std::fs::write',
        'std::fs::read_dir',
        'std::fs::metadata',
        'std::path::Path::exists',
        'std::path::Path::is_file',
        'std::path::Path::is_dir',
        'jiff::tz::db',
        'jiff::tz::TimeZoneDatabase::from_env',
        'jiff::tz::TimeZoneDatabase::from_dir',
        'jiff::tz::TimeZone::get',
        'jiff::tz::TimeZone::system',
    }
)
# The one place shipped code may allow a ban: sdm-host reads the zone lookup
# roots the host names (decision E24). The allow must be exactly this one,
# on exactly this statement: a binding of `TimeZoneDatabase::from_dir` over
# the root's path, with at most a trailing `.ok()` or `?`.
BAN_EXEMPTION = Path('rust/crates/sdm-host/src/zones.rs')
_EXEMPT_ATTRIBUTE = re.compile(
    r'^#\[allow\(clippy::disallowed_methods,reason=""\)\]$'
)
_EXEMPT_STATEMENT = re.compile(
    r'^let (mut )?\w+ (: [^=]+ )?= TimeZoneDatabase :: from_dir '
    r'\( & root \. path \)( \. ok \( \)| \?)? ;$'
)
# What, inside an allow or expect, turns a ban off: the lints, their old
# names, the groups that hold them, and every warning at once.
BAN_SWITCHES = frozenset(
    {
        'clippy::disallowed_methods',
        'clippy::disallowed_types',
        'clippy::disallowed_method',
        'clippy::disallowed_type',
        'clippy::style',
        'clippy::all',
        'warnings',
    }
)
_ATTRIBUTE = re.compile(r'#\s*(!?)\s*\[')
_LEVEL = re.compile(r'(?<![\w:])(allow|expect)\(')
# A Zoned parsed from text looks its zone up in jiff's global database,
# which clippy cannot see through `str::parse` or `FromStr`.
_ZONED_PARSE = (
    re.compile(r'parse :: < (jiff :: )?Zoned >'),
    re.compile(r': (jiff :: )?Zoned = [^;]*\. parse \('),
    re.compile(r'\bZoned :: from_str\b'),
    re.compile(r'< (jiff :: )?Zoned as (std :: |core :: )?(str :: )?FromStr >'),
)


def tokens(text: str) -> str:
    """Rust text as its tokens, one space apart, so spacing cannot hide."""
    return ' '.join(re.findall(r'\w+|::|[^\w\s]', text))


@dataclass(frozen=True)
class Attribute:
    start: int
    end: int
    inner: bool
    text: str  # without whitespace, string contents blanked


def attributes(source: Source) -> list[Attribute]:
    """Every `#[...]` and `#![...]`, bracket-matched."""
    found = []
    skeleton = source.skeleton
    for match in _ATTRIBUTE.finditer(skeleton):
        depth, j = 1, match.end()
        while j < len(skeleton) and depth:
            if skeleton[j] == '[':
                depth += 1
            elif skeleton[j] == ']':
                depth -= 1
            j += 1
        text = re.sub(r'\s+', '', skeleton[match.start() : j])
        text = re.sub(r'"[^"]*"', '""', text)
        found.append(Attribute(match.start(), j, bool(match.group(1)), text))
    return found


def _group(text: str, start: int) -> str:
    depth, j = 1, start
    while j < len(text) and depth:
        if text[j] == '(':
            depth += 1
        elif text[j] == ')':
            depth -= 1
        j += 1
    return text[start : j - 1]


METAVARIABLE = 'a macro metavariable'


def level_switches(text: str) -> set[str]:
    """The bans the `allow(…)`s and `expect(…)`s in `text` turn off.

    `text` has no whitespace. A lint named by a macro metavariable
    (`allow($lint)`, `allow(clippy::$l)`) could be any lint, so it counts.
    """
    switches: set[str] = set()
    for match in _LEVEL.finditer(text):
        for item in _group(text, match.end()).split(','):
            if item.lower() in BAN_SWITCHES:
                switches.add(item.lower())
            elif '$' in item:
                switches.add(METAVARIABLE)
    return switches


def ban_switches(attribute: Attribute) -> set[str]:
    """The bans an attribute turns off, inside `cfg_attr` too.

    An attribute that is itself a metavariable (`#[$meta]`), or that
    `cfg_attr` builds from one, may be any allow, so it counts too.
    """
    switches = level_switches(attribute.text)
    if re.match(r'#!?\[\$', attribute.text) or re.search(
        r'cfg_attr\(.*,\$', attribute.text
    ):
        switches.add(METAVARIABLE)
    return switches


def exempt(source: Source, attribute: Attribute) -> bool:
    """Whether this is sdm-host's approved allow on its approved statement."""
    if attribute.inner or not _EXEMPT_ATTRIBUTE.match(attribute.text):
        return False
    end = source.skeleton.find(';', attribute.end)
    if end < 0:
        return False
    statement = tokens(source.skeleton[attribute.end : end + 1])
    return bool(_EXEMPT_STATEMENT.match(statement))


def ban_problems(rel: Path, source: Source, shipped: bool) -> list[str]:
    """The suppressions of the bans in one file."""
    problems = []
    exemptions = 0
    for attribute in attributes(source):
        switches = ban_switches(attribute)
        if not switches:
            continue
        line = source.line(attribute.start)
        if not shipped:
            if 'reason=' not in attribute.text:
                problems.append(
                    f'{rel}:{line}: allows a clippy ban without a reason'
                )
            continue
        if rel == BAN_EXEMPTION and exempt(source, attribute):
            exemptions += 1
            if exemptions == 1:
                continue
        for lint in sorted(switches):
            problems.append(
                f'{rel}:{line}: shipped code allows {lint}; the one '
                "exemption is sdm-host's reading of the zone lookup roots, "
                'on its one statement (E24)'
            )
    if shipped:
        # Code brought in from outside src/ would be read as test code.
        for match in re.finditer(
            r'#!?\[\s*path\s*=|\binclude\s*!', source.skeleton
        ):
            problems.append(
                f'{rel}:{source.line(match.start())}: `{match.group(0)}` brings '
                'in code from outside the scanned source'
            )
        # An allow outside an attribute reaches one through a macro
        # (`wa!(allow(clippy::disallowed_methods), fn f() {…})`); in shipped
        # code it is a switch wherever it stands.
        bare = source.skeleton
        for attribute in attributes(source):
            bare = (
                bare[: attribute.start]
                + _blank(bare[attribute.start : attribute.end])
                + bare[attribute.end :]
            )
        compact = re.sub(r'\s+', '', bare)
        for match in _LEVEL.finditer(compact):
            end = match.end() + len(_group(compact, match.end())) + 1
            for lint in sorted(level_switches(compact[match.start() : end])):
                problems.append(
                    f'{rel}: `{match.group(1)}({lint})` outside an attribute, '
                    'as a macro would pass it, turns a ban off in shipped code'
                )
        text = tokens(source.skeleton)
        for pattern in _ZONED_PARSE:
            parse = pattern.search(text)
            if parse:
                problems.append(
                    f'{rel}: `{parse.group(0)}` parses a Zoned through '
                    "jiff's global zone database; resolve the zone through "
                    "the host's lookup roots"
                )
    return problems


def clippy_files(root: Path, crate: str) -> list[Path]:
    """The configuration clippy reads for a crate: the nearest file up to rust/."""
    directory = crate_dir(root, crate)
    rust = root / 'rust'
    while True:
        found = [
            directory / n
            for n in ('clippy.toml', '.clippy.toml')
            if (directory / n).is_file()
        ]
        if found or directory == rust or rust not in directory.parents:
            return found
        directory = directory.parent


def ban_paths(config: dict[str, object], key: str) -> set[str]:
    entries = config.get(key, [])
    paths: set[str] = set()
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, str):
                paths.add(entry)
            elif isinstance(entry, dict) and isinstance(entry.get('path'), str):
                paths.add(entry['path'])
    return paths


def workspace_crates(root: Path) -> list[str]:
    names = []
    for manifest in sorted((root / 'rust').glob('*/*/Cargo.toml')):
        if 'target' in manifest.parts:
            continue
        package = tomllib.loads(manifest.read_text()).get('package', {})
        if isinstance(package.get('name'), str):
            names.append(package['name'])
    return names


# Settings outside the code that would lower the bans: a lint allowed or
# capped through rustflags, or clippy pointed at another configuration.
_SUPPRESSING_SETTING = re.compile(
    r'(?:-A|--allow)[\s=",\']*(?:clippy::|warnings\b)'
    r'|--cap-lints|CLIPPY_CONF_DIR'
)
CARGO_CONFIGS = (
    Path('.cargo/config.toml'),
    Path('.cargo/config'),
    Path('rust/.cargo/config.toml'),
    Path('rust/.cargo/config'),
)


def suppression_settings(root: Path) -> list[str]:
    problems = []
    for rel in (*CARGO_CONFIGS, CI_FILE):
        path = root / rel
        if not path.is_file():
            continue
        text = path.read_text()
        for match in _SUPPRESSING_SETTING.finditer(text):
            line = text.count('\n', 0, match.start()) + 1
            problems.append(
                f'{rel}:{line}: `{match.group(0)}` lowers or redirects the '
                'clippy bans outside the code'
            )
    return problems


def check_bans(root: Path) -> list[str]:
    problems: list[str] = []
    workspace = root / 'rust/clippy.toml'
    if not workspace.is_file():
        return ['rust/clippy.toml: missing']
    base = tomllib.loads(workspace.read_text())
    base_types = ban_paths(base, 'disallowed-types')
    base_methods = ban_paths(base, 'disallowed-methods')
    problems += suppression_settings(root)
    for missing in sorted(REQUIRED_TYPES - base_types):
        problems.append(
            f'rust/clippy.toml: the type ban on {missing} is missing'
        )
    for missing in sorted(REQUIRED_METHODS - base_methods):
        problems.append(
            f'rust/clippy.toml: the method ban on {missing} is missing'
        )
    for crate in workspace_crates(root):
        if crate in DEV_CRATES:
            continue
        configs = clippy_files(root, crate)
        if len(configs) != 1:
            problems.append(
                f'{crate}: clippy reads {len(configs)} configuration files, not one'
            )
            continue
        config = configs[0]
        if config == workspace:
            continue
        own = tomllib.loads(config.read_text())
        rel = config.relative_to(root)
        for missing in sorted(base_types - ban_paths(own, 'disallowed-types')):
            problems.append(
                f"{rel}: does not repeat the workspace's type ban on {missing}"
            )
        for missing in sorted(
            base_methods - ban_paths(own, 'disallowed-methods')
        ):
            problems.append(
                f"{rel}: does not repeat the workspace's method ban on {missing}"
            )
    # No lint table turns a ban off.
    for manifest in [
        root / 'rust/Cargo.toml',
        *sorted((root / 'rust').glob('*/*/Cargo.toml')),
    ]:
        if not manifest.is_file() or 'target' in manifest.parts:
            continue
        data = tomllib.loads(manifest.read_text())
        for rust_table in (
            data.get('workspace', {}).get('lints', {}).get('rust', {}),
            data.get('lints', {}).get('rust', {}),
        ):
            level = rust_table.get('warnings')
            setting = level.get('level') if isinstance(level, dict) else level
            if setting == 'allow':
                problems.append(
                    f'{manifest.relative_to(root)}: warnings set to allow'
                )
        tables = [
            data.get('workspace', {}).get('lints', {}).get('clippy', {}),
            data.get('lints', {}).get('clippy', {}),
        ]
        for table in tables:
            for lint, level in table.items():
                setting = (
                    level.get('level') if isinstance(level, dict) else level
                )
                if lint in (
                    'disallowed_methods',
                    'disallowed_types',
                    'style',
                ) and setting in (
                    'allow',
                    'warn',
                ):
                    problems.append(
                        f'{manifest.relative_to(root)}: clippy::{lint} set to {setting}'
                    )
                if lint == 'all' and setting == 'allow':
                    problems.append(
                        f'{manifest.relative_to(root)}: clippy::all set to allow'
                    )
    # Shipped code allows a ban only at its exemption; tests give a reason.
    for crate in workspace_crates(root):
        if crate in DEV_CRATES:
            continue
        directory = crate_dir(root, crate)
        for path in rust_files(directory):
            shipped = (directory / 'src') in path.parents
            problems += ban_problems(
                path.relative_to(root), read_rust(path.read_text()), shipped
            )
    return problems


# --- the step variants --------------------------------------------------------

STEP_VARIANTS = ('fetch', 'call_model', 'done')
_STEP_ENUM = re.compile(r'\bpub\s+enum\s+Step\s*\{')
CONTRACT = Path('rust/crates/sdm-contract')


def _snake(name: str) -> str:
    return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()


def step_variants_in_source(text: str) -> list[str] | None:
    source = read_rust(text)
    match = _STEP_ENUM.search(source.skeleton)
    if not match:
        return None
    depth, j, start = 1, match.end(), match.end()
    while j < len(source.skeleton) and depth:
        if source.skeleton[j] in '{(':
            depth += 1
        elif source.skeleton[j] in '})':
            depth -= 1
        j += 1
    body = source.skeleton[start : j - 1]
    body = re.sub(r'#\[[^\]]*\]', ' ', body)
    variants, depth, token = [], 0, ''
    for char in body:
        if char in '{(':
            if depth == 0 and token.strip():
                variants.append(token.strip())
            depth += 1
            token = ''
        elif char in '})':
            depth -= 1
        elif depth == 0:
            if char == ',':
                if token.strip():
                    variants.append(token.strip())
                token = ''
            else:
                token += char
    if token.strip():
        variants.append(token.strip())
    names = []
    for variant in variants:
        names.append(_snake(variant.split()[0]))
    return names


def step_variants_in_schema(schema: dict[str, Any]) -> list[str] | None:
    for branch in schema.get('oneOf', []):
        if not isinstance(branch, dict):
            continue
        message = branch.get('properties', {}).get('message', {})
        if message.get('const') != 'step':
            continue
        names = []
        for variant in branch.get('oneOf', []):
            const = variant.get('properties', {}).get('step', {}).get('const')
            if not isinstance(const, str):
                return None
            names.append(const)
        return names
    return None


def check_steps(root: Path) -> list[str]:
    problems: list[str] = []
    expected = sorted(STEP_VARIANTS)
    run = root / CONTRACT / 'src/run.rs'
    found = step_variants_in_source(run.read_text()) if run.is_file() else None
    if found is None:
        problems.append(f'{run.relative_to(root)}: no `pub enum Step`')
    elif sorted(found) != expected:
        problems.append(
            f'{run.relative_to(root)}: Step has {found}; it has exactly '
            f'{list(STEP_VARIANTS)}, and no step asks the host for language behavior'
        )
    schemas = sorted((root / CONTRACT / 'schema').glob('sdm-contract-*.json'))
    if not schemas:
        problems.append(f'{CONTRACT}/schema: no contract schema')
    for path in schemas:
        variants = step_variants_in_schema(json.loads(path.read_text()))
        if variants is None or sorted(variants) != expected:
            problems.append(
                f'{path.relative_to(root)}: the step message has {variants}; it '
                f'has exactly {list(STEP_VARIANTS)}'
            )
    return problems


CHECKS: dict[str, Callable[[Path], list[str]]] = {
    'vocabulary': check_vocabulary,
    'sql': check_sql,
    'dialects': check_dialects,
    'provider-sql': check_provider_sql,
    'skips': check_skips,
    'bans': check_bans,
    'steps': check_steps,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split('\n', 1)[0])
    # Checked here, not by argparse's choices, which before Python 3.14
    # refuse an empty list. No checks named means all of them.
    parser.add_argument(
        'checks', nargs='*', help=f'any of {", ".join(CHECKS)}; all by default'
    )
    parser.add_argument('--root', type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    unknown = sorted(set(args.checks) - {*CHECKS, 'all'})
    if unknown:
        parser.error(f'unknown checks: {", ".join(unknown)}')
    names = (
        list(CHECKS) if not args.checks or 'all' in args.checks else args.checks
    )
    failed = False
    for name in names:
        problems = CHECKS[name](args.root)
        for problem in problems:
            print(f'{name}: {problem}')
        print(f'{name}: {"FAIL" if problems else "ok"}')
        failed = failed or bool(problems)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
